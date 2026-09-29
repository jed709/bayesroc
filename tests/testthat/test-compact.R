test_that("every re_store codec round-trips within its precision", {
  set.seed(4)
  arr <- array(rnorm(200 * 4 * 7), dim = c(200, 4, 7))
  s <- stats::sd(as.vector(arr))
  tol <- c(f64 = 0, f32 = 1e-6, int16 = 1e-3, int8 = 0.05)
  sizes <- numeric(0)
  for (st in names(tol)) {
    p <- .pack_array(arr, st)
    back <- .unpack_array(p)
    expect_equal(dim(back), dim(arr))
    expect_lt(max(abs(back - arr)) / s, tol[[st]] + 1e-12)
    sizes[st] <- length(p$raw)
  }
  expect_lt(sizes[["f32"]], sizes[["f64"]])
  expect_lt(sizes[["int16"]], sizes[["f32"]])
  expect_lt(sizes[["int8"]], sizes[["int16"]])
})

test_that(".pack_array rejects non-finite draws", {
  arr <- array(c(1, 2, NA, 4), dim = c(2, 2, 1))
  expect_error(.pack_array(arr, "f32"), "non-finite")
})

test_that("packing a draws_array removes and restores RE variables", {
  skip_if_no_cmdstan()
  d <- .fixture_crossed()
  m <- broc(brf(item_resp | is_old ~ 1 + (1 | sub), family = evsd()), d)
  f <- fit_broc(m, backend = "stan", refresh = 0, seed = 1,
                parallel_chains = 4, iter_warmup = 200, iter_sampling = 200)
  da <- posterior::as_draws_array(f$.backend_fit$draws())
  re_vars <- grep("^(z_|u_)", posterior::variables(da), value = TRUE)
  skip_if(length(re_vars) == 0, "no per-level RE variables in this fit")

  sp <- .pack_re_draws(da, "f64")
  expect_false(any(grepl("^(z_|u_)", posterior::variables(sp$draws))))
  back <- .unpack_re_draws(sp$draws, sp$packed)
  expect_setequal(posterior::variables(back), posterior::variables(da))
  expect_equal(as.array(posterior::subset_draws(back, variable = re_vars)),
               as.array(posterior::subset_draws(da, variable = re_vars)))
})
