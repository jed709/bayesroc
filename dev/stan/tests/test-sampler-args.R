# Helpers shared with tests/testthat/
# test-sampler-args.R (copied).

.bd1 <- function(f) {
  mean(posterior::as_draws_matrix(f$draws(variables = "beta_dprime[1]"))[, 1])
}

test_that("Stan: seed makes fits reproducible and thin keeps every k-th draw", {
  skip_if_no_cmdstan()
  m <- .fx_model("uvsd")
  a <- stan_fit_broc(m, seed = 42, chains = 2, parallel_chains = 2,
                iter_warmup = 50, iter_sampling = 40, thin = 2, refresh = 50)
  b <- stan_fit_broc(m, seed = 42, chains = 2, parallel_chains = 2,
                iter_warmup = 50, iter_sampling = 40, thin = 2, refresh = 50)
  expect_identical(.bd1(a), .bd1(b))
  expect_equal(nrow(posterior::as_draws_matrix(a$draws(variables = "beta_dprime[1]"))), 40)
})
