# =============================================================================
# Draw Packing
# =============================================================================

.RE_STORE <- c("f64", "f32", "int16", "int8")
.ZSTD_LEVEL <- 9L

.byte_shuffle   <- function(b, w) if (w <= 1L) b else as.vector(t(matrix(b, nrow = w)))
.byte_unshuffle <- function(b, w) if (w <= 1L) b else as.vector(t(matrix(b, ncol = w)))

.have_qs2 <- function() requireNamespace("qs2", quietly = TRUE)

.zc <- function(b) {
  if (.have_qs2()) qs2::zstd_compress_raw(b, .ZSTD_LEVEL)
  else memCompress(b, type = "xz")
}

.zd <- function(b, zstd) {
  if (!isTRUE(zstd)) return(memDecompress(b, type = "xz"))
  if (!.have_qs2()) {
    stop("This fit was saved with zstd compression; install the 'qs2' package ",
         "to load it.", call. = FALSE)
  }
  qs2::zstd_decompress_raw(b)
}

.PACK_BLOCK <- 2^22

.col_range <- function(M) {
  P <- ncol(M)
  mn <- numeric(P); mx <- numeric(P)
  for (j in seq_len(P)) { cj <- M[, j]; mn[j] <- min(cj); mx[j] <- max(cj) }
  list(min = mn, max = mx)
}

.re_var_pattern <- "^(z_|u_)"

#' Pack per-level random-effect draws out of a draws_array
#' @noRd
.pack_re_draws <- function(da, re_store = "f32") {
  if (is.null(da)) return(list(draws = da, packed = NULL))
  store <- match.arg(re_store, .RE_STORE)
  vars <- dimnames(da)[[3]]
  re <- grep(.re_var_pattern, vars)
  if (!length(re)) return(list(draws = da, packed = NULL))
  d <- dim(da); slab <- d[1] * d[2]
  per <- max(1L, as.integer(.PACK_BLOCK %/% slab))
  has_qs2 <- .have_qs2()
  mn <- mx <- numeric(length(re))
  blocks <- list()
  for (s in seq(1L, length(re), by = per)) {
    j <- s:min(length(re), s + per - 1L)
    x <- as.vector(da[, , re[j], drop = FALSE])
    if (!all(is.finite(x))) stop(".pack_re_draws: non-finite values not supported.", call. = FALSE)
    blocks[[length(blocks) + 1L]] <- if (store %in% c("f64", "f32")) {
      w <- if (store == "f64") 8L else 4L
      .zc(.byte_shuffle(writeBin(x, raw(), w, "little"), w))
    } else {
      M <- matrix(x, slab, length(j))
      rg <- .col_range(M); mn[j] <- rg$min; mx[j] <- rg$max
      rng <- rg$max - rg$min; rng[rng <= 0] <- 1
      lv <- if (store == "int16") 65535 else 255
      codes <- as.integer(round(sweep(sweep(M, 2, rg$min, "-"), 2, rng, "/") * lv))
      if (store == "int16") .zc(.byte_shuffle(writeBin(codes - 32768L, raw(), 2L, "little"), 2L))
      else .zc(writeBin(codes - 128L, raw(), 1L, "little"))
    }
    rm(x)
    invisible(gc(FALSE))
  }
  data <- list(codec = store, dim = c(d[1], d[2], length(re)), qs2 = has_qs2, raw = blocks)
  if (!store %in% c("f64", "f32")) { data$min <- mn; data$max <- mx }
  keep <- da[, , -re, drop = FALSE]
  list(draws = keep, packed = list(data = data, vars = vars[re], order = vars, store = store))
}

#' Restore packed random-effect draws into a draws_array
#' @noRd
.unpack_re_draws <- function(da, packed) {
  if (is.null(packed)) return(da)
  p <- packed$data
  d <- p$dim; slab <- d[1] * d[2]
  order <- packed$order %||% c(dimnames(da)[[3]], packed$vars)
  out <- array(NA_real_, c(d[1], d[2], length(order)),
               dimnames = list(iteration = dimnames(da)[[1]] %||% seq_len(d[1]),
                               chain = dimnames(da)[[2]] %||% seq_len(d[2]), variable = order))
  kept <- dimnames(da)[[3]]
  if (length(kept)) out[, , match(kept, order)] <- unclass(da)
  pos <- match(packed$vars, order)
  int <- !p$codec %in% c("f64", "f32")
  if (int) {
    lv <- if (p$codec == "int16") 65535 else 255
    rng <- p$max - p$min; rng[rng <= 0] <- 1
  }
  decode <- function(b) {
    b <- .zd(b, p$qs2)
    if (p$codec == "f64") readBin(.byte_unshuffle(b, 8L), "double", length(b) %/% 8L, 8L, endian = "little")
    else if (p$codec == "f32") readBin(.byte_unshuffle(b, 4L), "double", length(b) %/% 4L, 4L, endian = "little")
    else if (p$codec == "int16") readBin(.byte_unshuffle(b, 2L), "integer", length(b) %/% 2L, 2L,
                                         endian = "little") + 32768L
    else readBin(b, "integer", length(b), 1L, endian = "little") + 128L
  }
  raw_blocks <- if (is.raw(p$raw)) list(p$raw) else p$raw
  carry <- numeric(0); v <- 0L
  for (b in raw_blocks) {
    x <- c(carry, decode(b))
    n <- length(x) %/% slab
    for (k in seq_len(n)) {
      v <- v + 1L
      s <- x[((k - 1L) * slab + 1L):(k * slab)]
      out[, , pos[v]] <- if (int) s / lv * rng[v] + p$min[v] else s
    }
    carry <- if (n * slab < length(x)) x[(n * slab + 1L):length(x)] else numeric(0)
    rm(x)
    invisible(gc(FALSE))
  }
  class(out) <- c("draws_array", "draws", "array")
  out
}
