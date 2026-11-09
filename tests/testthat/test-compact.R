.re_draws_fixture <- function(n_iter, n_chain, n_re, seed) {
  set.seed(seed)
  arr <- array(rnorm(n_iter * n_chain * (n_re + 2)) * 1.7 + 0.3, dim = c(n_iter, n_chain, n_re + 2),
               dimnames = list(NULL, NULL, c("mu", sprintf("z_subj[%d]", seq_len(n_re)), "sigma")))
  posterior::as_draws_array(arr)
}

test_that("every re_store codec round-trips within its precision", {
  da <- .re_draws_fixture(200, 4, 7, 4)
  re <- grep("^z_", posterior::variables(da), value = TRUE)
  x <- unclass(posterior::subset_draws(da, variable = re))
  s <- stats::sd(as.vector(x))
  tol <- c(f64 = 0, f32 = 1e-6, int16 = 1e-3, int8 = 0.05)
  sizes <- numeric(0)
  for (st in names(tol)) {
    sp <- .pack_re_draws(da, st)
    back <- .unpack_re_draws(sp$draws, sp$packed)
    expect_identical(posterior::variables(back), posterior::variables(da))
    expect_identical(unclass(posterior::subset_draws(back, variable = c("mu", "sigma"))),
                     unclass(posterior::subset_draws(da, variable = c("mu", "sigma"))))
    expect_lt(max(abs(unclass(posterior::subset_draws(back, variable = re)) - x)) / s, tol[[st]] + 1e-12)
    sizes[st] <- sum(lengths(sp$packed$data$raw))
  }
  expect_lt(sizes[["f32"]], sizes[["f64"]])
  expect_lt(sizes[["int16"]], sizes[["f32"]])
  expect_lt(sizes[["int8"]], sizes[["int16"]])
})

test_that("block-split packing matches single-block packing", {
  da <- .re_draws_fixture(50, 2, 9, 5)
  for (st in c("f64", "f32", "int16", "int8")) {
    sp1 <- .pack_re_draws(da, st)
    one <- .unpack_re_draws(sp1$draws, sp1$packed)
    expect_length(sp1$packed$data$raw, 1L)
    split <- local({
      local_mocked_bindings(.PACK_BLOCK = 250)
      sp <- .pack_re_draws(da, st)
      expect_gt(length(sp$packed$data$raw), 1L)
      .unpack_re_draws(sp$draws, sp$packed)
    })
    expect_identical(split, one)
  }
})

test_that(".pack_re_draws rejects non-finite draws", {
  da <- .re_draws_fixture(4, 1, 2, 6)
  arr <- unclass(da); arr[2, 1, 2] <- NA
  expect_error(.pack_re_draws(posterior::as_draws_array(arr), "f32"), "non-finite")
})

test_that("packing a draws_array removes and restores RE variables", {
  skip_if_no_jax()
  f <- .fx_fit("evsd_crossed")
  da <- posterior::as_draws_array(f$draws())
  re_vars <- grep("^(z_|u_)", posterior::variables(da), value = TRUE)
  skip_if(length(re_vars) == 0, "no per-level RE variables in this fit")

  sp <- .pack_re_draws(da, "f64")
  expect_false(any(grepl("^(z_|u_)", posterior::variables(sp$draws))))
  back <- .unpack_re_draws(sp$draws, sp$packed)
  expect_identical(posterior::variables(back), posterior::variables(da))
  expect_equal(as.array(posterior::subset_draws(back, variable = re_vars)),
               as.array(posterior::subset_draws(da, variable = re_vars)))
})
