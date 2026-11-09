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

.merge_loglik_npy <- function(paths, restart, out_path, chunk = 64e6) {
  hs <- lapply(paths, .loglik_meta); shp <- lapply(hs, `[[`, "shape")
  descr <- hs[[1]]$descr; isz <- hs[[1]]$isz
  if (any(vapply(hs, function(h) h$descr, "") != descr))
    stop(".merge_loglik_npy: log_lik files have mixed dtypes.", call. = FALSE)

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
