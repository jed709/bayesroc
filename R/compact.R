# =============================================================================
# Draw Packing
# =============================================================================

.RE_STORE <- c("f32", "f64", "int16", "int8")
.ZSTD_LEVEL <- 9L

.byte_shuffle   <- function(b, w) if (w <= 1L) b else as.vector(t(matrix(b, nrow = w)))
.byte_unshuffle <- function(b, w) if (w <= 1L) b else as.vector(t(matrix(b, ncol = w)))

.have_qs2 <- function() requireNamespace("qs2", quietly = TRUE)

.zc <- function(b) {
  if (.have_qs2()) qs2::zstd_compress_raw(b, .ZSTD_LEVEL)
  else memCompress(b, type = "xz")
}

.zd <- function(b, zstd) {
  if (isTRUE(zstd)) qs2::zstd_decompress_raw(b)
  else memDecompress(b, type = "xz")
}

.col_range <- function(M) {
  P <- ncol(M)
  mn <- numeric(P); mx <- numeric(P)
  for (j in seq_len(P)) { cj <- M[, j]; mn[j] <- min(cj); mx[j] <- max(cj) }
  list(min = mn, max = mx)
}

.pack_array <- function(arr, store = "f32") {
  store <- match.arg(store, .RE_STORE)
  d <- dim(arr); if (is.null(d)) d <- length(arr)
  x <- as.double(arr)
  if (!all(is.finite(x))) {
    stop(".pack_array: non-finite values not supported.", call. = FALSE)
  }
  has_qs2 <- .have_qs2()
  if (store == "f64") {
    return(list(codec = "f64", dim = d, qs2 = has_qs2,
                raw = .zc(.byte_shuffle(writeBin(x, raw(), 8L, "little"), 8L))))
  }
  if (store == "f32") {
    return(list(codec = "f32", dim = d, qs2 = has_qs2,
                raw = .zc(.byte_shuffle(writeBin(x, raw(), 4L, "little"), 4L))))
  }
  S <- prod(d[1:2]); P <- length(x) %/% S
  M <- matrix(x, S, P)
  rg <- .col_range(M); rng <- rg$max - rg$min; rng[rng <= 0] <- 1
  lv <- if (store == "int16") 65535 else 255
  codes <- round(sweep(sweep(M, 2, rg$min, "-"), 2, rng, "/") * lv)
  b <- if (store == "int16") {
    .byte_shuffle(writeBin(as.integer(codes) - 32768L, raw(), 2L, "little"), 2L)
  } else {
    writeBin(as.integer(codes) - 128L, raw(), 1L, "little")
  }
  list(codec = store, dim = d, qs2 = has_qs2, min = rg$min, max = rg$max,
       raw = .zc(b))
}

.unpack_array <- function(p) {
  d <- p$dim; n <- prod(d)
  if (p$codec == "f64") {
    return(array(readBin(.byte_unshuffle(.zd(p$raw, p$qs2), 8L), "double", n, 8L,
                         endian = "little"), d))
  }
  if (p$codec == "f32") {
    return(array(readBin(.byte_unshuffle(.zd(p$raw, p$qs2), 4L), "double", n, 4L,
                         endian = "little"), d))
  }
  S <- prod(d[1:2]); P <- n %/% S
  lv <- if (p$codec == "int16") 65535 else 255
  ints <- if (p$codec == "int16") {
    readBin(.byte_unshuffle(.zd(p$raw, p$qs2), 2L), "integer", n, 2L,
            endian = "little") + 32768L
  } else {
    readBin(.zd(p$raw, p$qs2), "integer", n, 1L, endian = "little") + 128L
  }
  M <- matrix(ints, S, P); rng <- p$max - p$min; rng[rng <= 0] <- 1
  array(as.double(sweep(sweep(M / lv, 2, rng, "*"), 2, p$min, "+")), d)
}

.is_packed <- function(x) is.list(x) && !is.null(x$codec) && !is.null(x$raw)

.re_var_pattern <- "^(z_|u_)"

#' Pack per-level random-effect draws out of a draws_array
#' @noRd
.pack_re_draws <- function(da, re_store = "f32") {
  if (is.null(da)) return(list(draws = da, packed = NULL))
  vars <- posterior::variables(da)
  re_vars <- grep(.re_var_pattern, vars, value = TRUE)
  if (!length(re_vars)) return(list(draws = da, packed = NULL))
  sub <- posterior::subset_draws(da, variable = re_vars)
  arr <- as.array(sub)
  packed <- list(data = .pack_array(arr, re_store),
                 vars = re_vars,
                 store = re_store)
  keep <- setdiff(vars, re_vars)
  list(draws = posterior::subset_draws(da, variable = keep), packed = packed)
}

#' Restore packed random-effect draws into a draws_array
#' @noRd
.unpack_re_draws <- function(da, packed) {
  if (is.null(packed)) return(da)
  arr <- .unpack_array(packed$data)
  dimnames(arr) <- list(iteration = NULL, chain = NULL, variable = packed$vars)
  posterior::bind_draws(da, posterior::as_draws_array(arr), along = "variable")
}
