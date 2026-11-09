# Helpers shared with tests/testthat/
# test-loglik-sidecar.R (copied).

.agg_fixture <- function(n_per = 200) {
  d <- .fixture_univariate(sigma = 1.3, n_per = n_per)
  agg <- aggregate(list(n = rep(1L, nrow(d))),
                   by = list(item_resp = d$item_resp, is_old = d$is_old),
                   FUN = sum)
  list(trial = d, agg = agg)
}
.ls_model <- function(n_per = 200) {
  .fx_memo(paste0("ls:model:", n_per),
           broc(brf(item_resp | is_old ~ 1, family = evsd()), .agg_fixture(n_per)$agg, counts = "n"))
}
.ls_stan <- function(m, ...) {
  stan_fit_broc(m, refresh = 50, seed = 1, chains = 2,
           parallel_chains = 2, iter_warmup = 50, iter_sampling = 100, ...)
}
.ls_stan_f64 <- function() .fx_memo("ls:stan_f64", .ls_stan(.ls_model(), loglik_precision = "f64"))
.ls_jax_loo <- function() {
  .fx_memo("ls:jax_loo", fit_broc(.ls_model(120), backend = "jax", sampler = "nuts", chains = 2,
                                  parallel_chains = 2, iter_warmup = 30, iter_sampling = 50, seed = 4,
                                  loo = TRUE, save_loglik = FALSE, progress_bar = FALSE))
}

test_that("Stan fits write a log_lik sidecar and keep it out of the draws array", {
  skip_if_no_cmdstan()
  fx <- .agg_fixture()
  f <- .ls_stan(.ls_model())

  expect_true(!is.null(f$.loglik_path))
  expect_true(file.exists(f$.loglik_path))
  meta <- .loglik_meta(f$.loglik_path)
  expect_equal(meta$C, 2L)
  expect_equal(meta$D, 100L)
  expect_equal(meta$N, nrow(fx$agg))
  expect_equal(meta$isz, 4L)

  ll <- as.matrix(f$draws(variables = "log_lik", format = "matrix"))
  expect_equal(dim(ll), c(200L, nrow(fx$agg)))
})

test_that("Stan and JAX sidecars agree on the same model", {
  skip_if_no_cmdstan()
  skip_if_no_jax()
  m <- .ls_model(120)
  fs <- stan_fit_broc(m, refresh = 50, seed = 2,
                 parallel_chains = 2, chains = 2,
                 iter_warmup = 100, iter_sampling = 150)
  fj <- fit_broc(m, backend = "jax", sampler = "nuts", seed = 2, chains = 2, parallel_chains = 2,
                 iter_warmup = 100, iter_sampling = 150, progress_bar = FALSE)

  ms <- .loglik_meta(fs$.loglik_path)
  mj <- .loglik_meta(fj$.loglik_path)
  expect_equal(ms$N, mj$N)
  expect_equal(ms$C, mj$C)
  expect_equal(ms$D, mj$D)

  ls <- loo(fs, cores = 1)
  lj <- loo(fj, cores = 1)
  expect_equal(dim(ls)[2], dim(lj)[2])
  expect_equal(ls$estimates["elpd_loo", "Estimate"],
               lj$estimates["elpd_loo", "Estimate"],
               tolerance = 3 * ls$estimates["elpd_loo", "SE"])
})
