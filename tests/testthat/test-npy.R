test_that("npy sidecar round-trips in every read layout", {
  set.seed(1)
  C <- 3; D <- 40; N <- 17
  ll <- array(rnorm(C * D * N), dim = c(C, D, N))
  p <- tempfile(fileext = ".npy")

  .write_loglik_npy_from_array(ll, p, "f64")
  m <- .loglik_meta(p)
  expect_equal(c(m$C, m$D, m$N, m$isz), c(C, D, N, 8L))

  expect_equal(.read_loglik_CDN(p), ll)

  ref_sn <- matrix(aperm(ll, c(2, 1, 3)), nrow = C * D, ncol = N)
  expect_equal(.read_loglik_SN_cols(p, 1L, N), ref_sn)
  expect_equal(.read_loglik_SN_cols(p, 5L, 9L), ref_sn[, 5:9])
  expect_equal(.read_loglik_SN_cols(p, 3L, 14L, block = 7L), ref_sn[, 3:14])
})

test_that("npy reader rejects layouts it cannot stream", {
  p <- tempfile(fileext = ".npy")
  con <- file(p, "wb")
  hdr <- "{'descr': '<f8', 'fortran_order': True, 'shape': (2, 3, 4), }"
  hdr <- paste0(hdr, strrep(" ", (64L - ((11L + nchar(hdr)) %% 64L)) %% 64L), "\n")
  writeBin(as.raw(c(0x93, charToRaw("NUMPY"), 1L, 0L)), con)
  writeBin(as.integer(nchar(hdr)), con, size = 2, endian = "little")
  writeBin(charToRaw(hdr), con)
  writeBin(rnorm(24), con)
  close(con)
  expect_error(.loglik_meta(p), "Fortran order")

  q <- tempfile(fileext = ".npy")
  con <- file(q, "wb")
  .npy_write_prefix(con, c(2L, 3L, 4L), ">f8")
  writeBin(rnorm(24), con, endian = "big")
  close(con)
  expect_error(.loglik_meta(q), "little-endian")
})

test_that("f32 recode preserves shape and value to single precision", {
  set.seed(2)
  ll <- array(rnorm(2 * 20 * 6), dim = c(2, 20, 6))
  p64 <- tempfile(fileext = ".npy"); p32 <- tempfile(fileext = ".npy")
  .write_loglik_npy_from_array(ll, p64, "f64")
  .emit_loglik_side(p64, p32, "f32")

  m <- .loglik_meta(p32)
  expect_equal(c(m$C, m$D, m$N, m$isz), c(2L, 20L, 6L, 4L))
  expect_equal(.read_loglik_CDN(p32), ll, tolerance = 1e-6)
})

test_that("merge binds rounds on the correct axis", {
  set.seed(3)
  C <- 3; N <- 11
  a <- array(rnorm(C * 10 * N), dim = c(C, 10, N))
  b <- array(rnorm(C * 6 * N), dim = c(C, 6, N))
  pa <- tempfile(fileext = ".npy"); pb <- tempfile(fileext = ".npy")
  .write_loglik_npy_from_array(a, pa, "f64")
  .write_loglik_npy_from_array(b, pb, "f64")

  pw <- tempfile(fileext = ".npy")
  .merge_loglik_npy(c(pa, pb), "warm", pw)
  want <- array(0, dim = c(C, 16, N))
  want[, 1:10, ] <- a; want[, 11:16, ] <- b
  expect_equal(.read_loglik_CDN(pw), want)

  d <- array(rnorm(2 * 10 * N), dim = c(2, 10, N))
  pd <- tempfile(fileext = ".npy")
  .write_loglik_npy_from_array(d, pd, "f64")
  pc <- tempfile(fileext = ".npy")
  .merge_loglik_npy(c(pa, pd), "cold", pc)
  wantc <- array(0, dim = c(5, 10, N))
  wantc[1:3, , ] <- a; wantc[4:5, , ] <- d
  expect_equal(.read_loglik_CDN(pc), wantc)
})

test_that("merge rejects mismatched rounds", {
  set.seed(4)
  a <- array(rnorm(2 * 5 * 4), dim = c(2, 5, 4))
  b <- array(rnorm(3 * 5 * 4), dim = c(3, 5, 4))
  pa <- tempfile(fileext = ".npy"); pb <- tempfile(fileext = ".npy")
  .write_loglik_npy_from_array(a, pa, "f64")
  .write_loglik_npy_from_array(b, pb, "f64")
  expect_error(.merge_loglik_npy(c(pa, pb), "warm", tempfile(fileext = ".npy")),
               "chain/N mismatch")
})
