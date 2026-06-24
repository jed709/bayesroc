# Guards for fit_broc() sampler arguments that were previously broken or
# backend-asymmetric: seed reproducibility (Stan seed was never passed to
# cmdstanr), thinning, and the cmdstanr-only `...` warning on JAX.

.bd1 <- function(f) {
  mean(posterior::as_draws_matrix(f$draws(variables = "beta_dprime[1]"))[, 1])
}

test_that("Stan: seed makes fits reproducible", {
  skip_if_no_cmdstan()
  d <- .fixture_univariate(sigma = 1.2)
  m <- broc(brf(item_resp | is_old ~ 1, sigma ~ 1, family = uvsd()), d)
  a <- fit_broc(m, backend = "stan", seed = 42, chains = 2, parallel_chains = 2,
                iter_warmup = 200, iter_sampling = 200, refresh = 0)
  b <- fit_broc(m, backend = "stan", seed = 42, chains = 2, parallel_chains = 2,
                iter_warmup = 200, iter_sampling = 200, refresh = 0)
  expect_identical(.bd1(a), .bd1(b))
})

test_that("Stan: thin keeps every k-th draw", {
  skip_if_no_cmdstan()
  d <- .fixture_univariate(sigma = 1.2)
  m <- broc(brf(item_resp | is_old ~ 1, sigma ~ 1, family = uvsd()), d)
  fs <- fit_broc(m, backend = "stan", seed = 1, chains = 2, parallel_chains = 2,
                 iter_warmup = 200, iter_sampling = 400, thin = 2, refresh = 0)
  expect_equal(nrow(posterior::as_draws_matrix(fs$draws(variables = "beta_dprime[1]"))), 400)
})

test_that("JAX: thin keeps every k-th draw", {
  skip_if_no_jax()
  d <- .fixture_univariate(sigma = 1.2)
  m <- broc(brf(item_resp | is_old ~ 1, sigma ~ 1, family = uvsd()), d)
  fj <- fit_broc(m, backend = "jax", chains = 2L, num_warmup = 200L,
                 num_samples = 400L, thin = 2L, seed = 1, progress_bar = FALSE)
  expect_equal(fj$.iter_sampling, 200L)
})

test_that("JAX: parallel_chains = 1 runs sequentially, same draws shape", {
  skip_if_no_jax()
  d <- .fixture_univariate(sigma = 1.2)
  m <- broc(brf(item_resp | is_old ~ 1, sigma ~ 1, family = uvsd()), d)
  fs <- fit_broc(m, backend = "jax", chains = 2L, parallel_chains = 1L,
                 num_warmup = 100L, num_samples = 100L, seed = 1, progress_bar = FALSE)
  fp <- fit_broc(m, backend = "jax", chains = 2L, parallel_chains = 2L,
                 num_warmup = 100L, num_samples = 100L, seed = 1, progress_bar = FALSE)
  expect_identical(dim(fs$draws()), dim(fp$draws()))
})

test_that("JAX warns that `...` is cmdstanr-only", {
  skip_if_no_jax()
  d <- .fixture_univariate(sigma = 1.2)
  m <- broc(brf(item_resp | is_old ~ 1, sigma ~ 1, family = uvsd()), d)
  expect_warning(
    fit_broc(m, backend = "jax", chains = 2L, num_warmup = 100L,
             num_samples = 100L, seed = 1, progress_bar = FALSE, sig_figs = 5),
    "forwarded to cmdstanr"
  )
})
