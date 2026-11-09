# =============================================================================
# .npy Header I/O
# =============================================================================

.npy_dtype <- function(precision) {
  precision <- match.arg(precision, c("f64", "f32"))
  if (precision == "f32") list(size = 4L, descr = "<f4") else list(size = 8L, descr = "<f8")
}

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
  if (grepl("'fortran_order':\\s*True", hdr))
    stop("'", basename(path), "' is stored in Fortran order; expected C order.", call. = FALSE)
  list(shape = shp,
       descr = regmatches(hdr, regexec("'descr':\\s*'([^']+)'", hdr))[[1]][2],
       offset = 8L + hl_size + as.integer(hl))
}

.npy_float_size <- function(descr, path) {
  isz <- suppressWarnings(as.integer(regmatches(descr, regexpr("[0-9]+$", descr))))
  if (!length(isz) || is.na(isz) || !grepl("^[<|=]f(4|8)$", descr))
    stop("'", basename(path), "': expected little-endian f4/f8 log_lik, got dtype '",
         descr, "'.", call. = FALSE)
  isz
}

# =============================================================================
# log_lik Sidecar Readers
# =============================================================================

.loglik_meta <- function(path) {
  h <- .npy_header(path); shp <- h$shape
  if (length(shp) != 3L)
    stop(".loglik_meta: expected 3-D [chain, draw, N] log_lik, got shape (",
         paste(shp, collapse = ", "), ").", call. = FALSE)
  isz <- .npy_float_size(h$descr, path)
  list(C = shp[1], D = shp[2], N = shp[3], S = as.double(shp[1]) * shp[2],
       isz = isz, offset = h$offset, descr = h$descr, shape = shp)
}

.read_loglik_SN_cols <- function(path, j0, j1, block = 512L) {
  m <- .loglik_meta(path); S <- as.integer(m$S); N <- m$N; isz <- m$isz
  nc <- as.integer(j1 - j0 + 1L)
  out <- matrix(0, nrow = S, ncol = nc)
  con <- file(path, "rb"); on.exit(close(con))
  if (2 * nc >= N) {
    seek(con, m$offset, origin = "start")
    s0 <- 1L
    while (s0 <= S) {
      b <- min(block, S - s0 + 1L)
      v <- matrix(readBin(con, "double", b * N, size = isz, endian = "little"),
                  nrow = b, ncol = N, byrow = TRUE)
      out[s0:(s0 + b - 1L), ] <- v[, j0:j1, drop = FALSE]
      s0 <- s0 + b
    }
    return(out)
  }
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
  dt <- .npy_dtype(match.arg(precision)); sz <- dt$size; descr <- dt$descr
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
  dt <- .npy_dtype(precision); sz <- dt$size; descr <- dt$descr
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
  if (.loglik_meta(src)$isz == .npy_dtype(precision)$size) {
    if (!isTRUE(suppressWarnings(file.rename(src, side))) && !isTRUE(file.copy(src, side, overwrite = TRUE)))
      stop("Could not write the log-likelihood file '", side, "'.", call. = FALSE)
  } else {
    .recode_loglik_npy(src, side, precision)
  }
  invisible(side)
}

# =============================================================================
# Resource Probes
# =============================================================================

.ram_probe <- new.env(parent = emptyenv())

.free_ram_gb <- function() {
  now <- Sys.time()
  if (!is.null(.ram_probe$at) && difftime(now, .ram_probe$at, units = "secs") < 60) {
    return(.ram_probe$gb)
  }
  .ram_probe$gb <- .free_ram_gb_uncached()
  .ram_probe$at <- now
  .ram_probe$gb
}

.free_ram_gb_uncached <- function() {
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
