# =============================================================================
# .npy Header I/O
# =============================================================================

.npy_header_str <- function(shape, descr = "<f8") {
  shp_str <- if (length(shape) == 1) paste0(shape, ",") else paste(shape, collapse = ", ")
  hdr <- sprintf("{'descr': '%s', 'fortran_order': False, 'shape': (%s), }", descr, shp_str)
  total <- 10L + nchar(hdr) + 1L
  paste0(hdr, strrep(" ", (64L - (total %% 64L)) %% 64L), "\n")
}

.npy_write_prefix <- function(con, shape, descr = "<f8") {
  hdr <- .npy_header_str(shape, descr)
  writeBin(as.raw(c(0x93, charToRaw("NUMPY"))), con)
  writeBin(as.raw(c(1L, 0L)), con)
  writeBin(as.integer(nchar(hdr)), con, size = 2, endian = "little")
  writeBin(charToRaw(hdr), con)
}

.npy_header <- function(path) {
  con <- file(path, "rb"); on.exit(close(con))
  readBin(con, "raw", 6)
  ver <- readBin(con, "integer", 1, size = 2, endian = "little")
  hl_size <- if (ver == 1L) 2L else 4L
  hl <- readBin(con, "integer", 1, size = hl_size, endian = "little")
  hdr <- rawToChar(readBin(con, "raw", hl))
  shp <- regmatches(hdr, regexec("'shape':\\s*\\(([^)]*)\\)", hdr))[[1]][2]
  shp <- as.integer(strsplit(trimws(gsub(",\\s*$", "", shp)), ",\\s*")[[1]])
  if (!length(shp) || any(is.na(shp))) shp <- 1L
  list(shape = shp,
       descr = regmatches(hdr, regexec("'descr':\\s*'([^']+)'", hdr))[[1]][2],
       offset = 8L + hl_size + as.integer(hl))
}

#' Write a numeric array to .npy
#' @noRd
write_npy <- function(x, path, precision = c("f64", "f32")) {
  precision <- match.arg(precision)
  sz <- if (precision == "f32") 4L else 8L
  descr <- if (precision == "f32") "<f4" else "<f8"
  x <- as.array(x)
  shp <- dim(x); if (is.null(shp)) shp <- length(x)
  data <- if (length(shp) == 1) as.numeric(x) else as.numeric(aperm(x, rev(seq_along(shp))))
  con <- file(path, "wb"); on.exit(close(con))
  .npy_write_prefix(con, shp, descr)
  writeBin(data, con, size = sz, endian = "little")
  invisible(path)
}

# =============================================================================
# log_lik Sidecar Readers
# =============================================================================

.loglik_meta <- function(path) {
  h <- .npy_header(path); shp <- h$shape
  if (length(shp) != 3L)
    stop(".loglik_meta: expected 3-D [chain, draw, N] log_lik, got shape (",
         paste(shp, collapse = ", "), ").", call. = FALSE)
  isz <- as.integer(regmatches(h$descr, regexpr("[0-9]+$", h$descr)))
  if (is.na(isz) || !grepl("^[<|=]?f", h$descr))
    stop(".loglik_meta: expected float log_lik, got dtype '", h$descr, "'.", call. = FALSE)
  list(C = shp[1], D = shp[2], N = shp[3], S = as.double(shp[1]) * shp[2],
       isz = isz, offset = h$offset, descr = h$descr, shape = shp)
}

.read_loglik_SN_full <- function(path, block = 512L) {
  m <- .loglik_meta(path); S <- as.integer(m$S); N <- m$N; isz <- m$isz
  out <- matrix(0, nrow = S, ncol = N)
  con <- file(path, "rb"); on.exit(close(con))
  seek(con, m$offset, origin = "start")
  s0 <- 1L
  while (s0 <= S) {
    b <- min(block, S - s0 + 1L)
    v <- readBin(con, "double", b * N, size = isz, endian = "little")
    out[s0:(s0 + b - 1L), ] <- matrix(v, nrow = b, ncol = N, byrow = TRUE)
    s0 <- s0 + b
  }
  out
}

.read_loglik_SN_cols <- function(path, j0, j1) {
  m <- .loglik_meta(path); S <- as.integer(m$S); N <- m$N; isz <- m$isz
  nc <- as.integer(j1 - j0 + 1L)
  out <- matrix(0, nrow = S, ncol = nc)
  con <- file(path, "rb"); on.exit(close(con))
  base <- m$offset + as.double(j0 - 1L) * isz
  rowbytes <- as.double(N) * isz
  for (s in seq_len(S)) {
    seek(con, base + (s - 1) * rowbytes, origin = "start")
    out[s, ] <- readBin(con, "double", nc, size = isz, endian = "little")
  }
  out
}

.read_loglik_CDN <- function(path, block = 512L) {
  m <- .loglik_meta(path); C <- m$C; D <- m$D; N <- m$N
  S <- as.integer(m$S); isz <- m$isz
  out <- array(0, dim = c(C, D, N))
  con <- file(path, "rb"); on.exit(close(con))
  seek(con, m$offset, origin = "start")
  s <- 0L
  while (s < S) {
    b <- min(block, S - s)
    v <- readBin(con, "double", b * N, size = isz, endian = "little")
    for (i in seq_len(b)) {
      ss <- s + i
      out[(ss - 1L) %/% D + 1L, (ss - 1L) %% D + 1L, ] <- v[((i - 1L) * N + 1L):(i * N)]
    }
    s <- s + b
  }
  out
}

# =============================================================================
# log_lik Sidecar Writers
# =============================================================================

.write_loglik_npy_from_array <- function(ll, path, precision = c("f64", "f32")) {
  precision <- match.arg(precision)
  sz <- if (precision == "f32") 4L else 8L
  descr <- if (precision == "f32") "<f4" else "<f8"
  d <- dim(ll)
  if (length(d) != 3L)
    stop(".write_loglik_npy_from_array: expected [chain, draw, N] array.", call. = FALSE)
  con <- file(path, "wb"); on.exit(close(con))
  .npy_write_prefix(con, d, descr)
  for (cc in seq_len(d[1])) {
    for (dd in seq_len(d[2])) writeBin(as.double(ll[cc, dd, ]), con, size = sz,
                                       endian = "little")
  }
  invisible(path)
}

.recode_loglik_npy <- function(src, dst, precision = "f32", block = 512L) {
  m <- .loglik_meta(src)
  sz <- if (precision == "f32") 4L else 8L
  descr <- if (precision == "f32") "<f4" else "<f8"
  ci <- file(src, "rb"); co <- file(dst, "wb"); on.exit({ close(ci); close(co) })
  seek(ci, m$offset, origin = "start")
  .npy_write_prefix(co, m$shape, descr)
  S <- as.integer(m$S); N <- m$N; s0 <- 0L
  while (s0 < S) {
    b <- min(block, S - s0)
    writeBin(readBin(ci, "double", b * N, size = m$isz, endian = "little"),
             co, size = sz, endian = "little")
    s0 <- s0 + b
  }
  invisible(dst)
}

.emit_loglik_side <- function(src, side, precision = "f32") {
  tsz <- if (precision == "f32") 4L else 8L
  if (.loglik_meta(src)$isz == tsz) file.copy(src, side, overwrite = TRUE)
  else .recode_loglik_npy(src, side, precision)
  invisible(side)
}

.merge_loglik_npy <- function(paths, restart, out_path, chunk = 64e6) {
  hs <- lapply(paths, .npy_header); shp <- lapply(hs, `[[`, "shape")
  descr <- hs[[1]]$descr
  if (any(vapply(hs, function(h) h$descr, "") != descr))
    stop(".merge_loglik_npy: log_lik files have mixed dtypes.", call. = FALSE)
  isz <- as.integer(regmatches(descr, regexpr("[0-9]+$", descr)))
  if (is.na(isz) || !grepl("^[<|=]?f", descr))
    stop(".merge_loglik_npy: expected float log_lik files, got '", descr, "'.", call. = FALSE)
  if (any(vapply(shp, length, 1L) != 3L))
    stop(".merge_loglik_npy: expected 3-D [chain, draw, N] log_lik.", call. = FALSE)

  copy_block <- function(con_in, con_out, nbytes) {
    while (nbytes > 0) {
      take <- min(nbytes, chunk)
      writeBin(readBin(con_in, "raw", take), con_out)
      nbytes <- nbytes - take
    }
  }
  need <- sum(vapply(paths, file.size, 0))
  free <- .disk_free_bytes(dirname(out_path))
  if (is.finite(free) && free < need + 2^28)
    stop(sprintf(paste0(".merge_loglik_npy: merged log_lik sidecar needs ~%.1f GB but only ",
                        "%.1f GB is free on the target volume (%s)."),
                 need / 1e9, free / 1e9, dirname(out_path)), call. = FALSE)
  ok <- FALSE
  on.exit(if (!ok) unlink(out_path), add = TRUE)
  con_out <- file(out_path, "wb"); on.exit(close(con_out), add = TRUE, after = FALSE)

  if (restart == "warm") {
    C <- shp[[1]][1]; N <- shp[[1]][3]
    if (!all(vapply(shp, function(s) s[1] == C && s[3] == N, logical(1))))
      stop(".merge_loglik_npy(warm): chain/N mismatch across rounds.", call. = FALSE)
    .npy_write_prefix(con_out, c(C, sum(vapply(shp, `[`, 1L, 2L)), N), descr)
    cons <- lapply(paths, function(p) file(p, "rb"))
    on.exit(lapply(cons, close), add = TRUE)
    for (cc in seq_len(C)) {
      for (r in seq_along(paths)) {
        nb <- as.double(shp[[r]][2]) * N * isz
        seek(cons[[r]], hs[[r]]$offset + (cc - 1) * nb)
        copy_block(cons[[r]], con_out, nb)
      }
    }
  } else {
    D <- shp[[1]][2]; N <- shp[[1]][3]
    if (!all(vapply(shp, function(s) s[2] == D && s[3] == N, logical(1))))
      stop(".merge_loglik_npy(cold): draw/N mismatch across rounds.", call. = FALSE)
    .npy_write_prefix(con_out, c(sum(vapply(shp, `[`, 1L, 1L)), D, N), descr)
    for (r in seq_along(paths)) {
      con_r <- file(paths[[r]], "rb"); seek(con_r, hs[[r]]$offset)
      copy_block(con_r, con_out, as.double(shp[[r]][1]) * D * N * isz)
      close(con_r)
    }
  }
  ok <- TRUE
  invisible(out_path)
}

# =============================================================================
# Stan CSV -> Sidecar
# =============================================================================

.stan_loglik_sidecar <- function(stan_fit, path, precision = "f32") {
  files <- stan_fit$output_files()
  files <- files[file.exists(files)]
  if (!length(files)) return(NULL)
  sz <- if (precision == "f32") 4L else 8L
  descr <- if (precision == "f32") "<f4" else "<f8"

  con <- NULL
  N <- NULL
  D <- NULL
  ok <- FALSE
  on.exit({
    if (!is.null(con)) close(con)
    if (!ok) unlink(path)
  }, add = TRUE)

  for (i in seq_along(files)) {
    ch <- cmdstanr::read_cmdstan_csv(files[i], variables = "log_lik",
                                     sampler_diagnostics = "")
    dr <- ch$post_warmup_draws
    if (is.null(dr)) return(NULL)
    d <- dim(dr)
    if (is.null(N)) {
      N <- d[3]; D <- d[1]
      con <- file(path, "wb")
      .npy_write_prefix(con, c(length(files), D, N), descr)
    } else if (d[1] != D || d[3] != N) {
      stop(".stan_loglik_sidecar: ragged chains (", d[1], "x", d[3],
           " vs ", D, "x", N, ").", call. = FALSE)
    }
    for (it in seq_len(D)) {
      writeBin(as.double(dr[it, 1, ]), con, size = sz, endian = "little")
    }
    rm(ch, dr)
  }
  ok <- TRUE
  invisible(path)
}

# =============================================================================
# Resource Probes
# =============================================================================

.disk_free_bytes <- function(path) {
  out <- tryCatch({
    if (.Platform$OS.type == "windows") {
      root <- normalizePath(path, winslash = "\\", mustWork = FALSE)
      drv <- sub("^([A-Za-z]:).*$", "\\1", root)
      x <- system2("powershell", c("-NoProfile", "-Command",
                                   sprintf("(Get-PSDrive %s).Free", sub(":", "", drv))),
                   stdout = TRUE, stderr = FALSE)
      as.numeric(x[1])
    } else {
      x <- system2("df", c("-Pk", shQuote(path)), stdout = TRUE, stderr = FALSE)
      as.numeric(strsplit(trimws(x[2]), "\\s+")[[1]][4]) * 1024
    }
  }, error = function(e) NA_real_, warning = function(w) NA_real_)
  if (length(out) != 1L || !is.finite(out)) NA_real_ else out
}

.free_ram_gb <- function() {
  out <- tryCatch({
    if (.Platform$OS.type == "windows") {
      x <- system2("powershell", c("-NoProfile", "-Command",
                                   "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"),
                   stdout = TRUE, stderr = FALSE)
      as.numeric(x[1]) / 1024 / 1024
    } else {
      x <- system2("awk", c("'/MemAvailable/{print $2}'", "/proc/meminfo"), stdout = TRUE)
      as.numeric(x[1]) / 1024 / 1024
    }
  }, error = function(e) NA_real_, warning = function(w) NA_real_)
  if (length(out) != 1L || !is.finite(out)) NA_real_ else out
}
