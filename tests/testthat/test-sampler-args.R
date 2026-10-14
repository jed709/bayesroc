
.bd1 <- function(f) {
  mean(posterior::as_draws_matrix(f$draws(variables = "beta_dprime[1]"))[, 1])
}

test_that("Stan: seed makes fits reproducible", {
  skip_if_no_cmdstan()
  d <- .fixture_univariate(sigma = 1.2)
  m <- broc(brf(item_resp | is_old ~ 1, sigma ~ 1, family = uvsd()), d)
  a <- fit_broc(m, backend = "stan", seed = 42, chains = 2, parallel_chains = 2,
                iter_warmup = 200, iter_sampling = 200, refresh = 50)
  b <- fit_broc(m, backend = "stan", seed = 42, chains = 2, parallel_chains = 2,
                iter_warmup = 200, iter_sampling = 200, refresh = 50)
  expect_identical(.bd1(a), .bd1(b))
})

test_that("Stan: thin keeps every k-th draw", {
  skip_if_no_cmdstan()
  d <- .fixture_univariate(sigma = 1.2)
  m <- broc(brf(item_resp | is_old ~ 1, sigma ~ 1, family = uvsd()), d)
  fs <- fit_broc(m, backend = "stan", seed = 1, chains = 2, parallel_chains = 2,
                 iter_warmup = 200, iter_sampling = 400, thin = 2, refresh = 50)
  expect_equal(nrow(posterior::as_draws_matrix(fs$draws(variables = "beta_dprime[1]"))), 400)
})

test_that("JAX: thin keeps every k-th draw", {
  skip_if_no_jax()
  d <- .fixture_univariate(sigma = 1.2)
  m <- broc(brf(item_resp | is_old ~ 1, sigma ~ 1, family = uvsd()), d)
  fj <- fit_broc(m, backend = "jax", chains = 2L, iter_warmup = 200L,
                 iter_sampling = 400L, thin = 2L, seed = 1, progress_bar = FALSE)
  expect_equal(fj$.iter_sampling, 200L)
})

test_that("JAX: parallel_chains = 1 runs sequentially, same draws shape", {
  skip_if_no_jax()
  d <- .fixture_univariate(sigma = 1.2)
  m <- broc(brf(item_resp | is_old ~ 1, sigma ~ 1, family = uvsd()), d)
  fs <- fit_broc(m, backend = "jax", chains = 2L, parallel_chains = 1L,
                 iter_warmup = 100L, iter_sampling = 100L, seed = 1, progress_bar = FALSE)
  fp <- fit_broc(m, backend = "jax", chains = 2L, parallel_chains = 2L,
                 iter_warmup = 100L, iter_sampling = 100L, seed = 1, progress_bar = FALSE)
  expect_identical(dim(fs$draws()), dim(fp$draws()))
})

test_that("JAX warns that `...` is cmdstanr-only", {
  skip_if_no_jax()
  d <- .fixture_univariate(sigma = 1.2)
  m <- broc(brf(item_resp | is_old ~ 1, sigma ~ 1, family = uvsd()), d)
  expect_warning(
    fit_broc(m, backend = "jax", chains = 2L, iter_warmup = 100L,
             iter_sampling = 100L, seed = 1, progress_bar = FALSE, sig_figs = 5),
    "forwarded to cmdstanr"
  )
})

test_that("one chain, or chains run one at a time, compile and sample on both JAX samplers", {
  skip_if_no_jax()
  m <- suppressMessages(broc(brf(item_resp | is_old ~ 1 + (1 | sub), family = uvsd()),
                             .fixture_crossed(n_sub = 8, n_item = 16)))
  for (s in c("drghmc", "nuts")) for (it in list(300, "auto")) for (ch in 1:2) {
    f <- suppressMessages(fit_broc(m, backend = "jax", sampler = s, chains = ch, parallel_chains = 1,
                                   iter_warmup = 200, iter_sampling = it, seed = 1, progress_bar = FALSE,
                                   control = if (identical(it, "auto")) list(check_every = 200, max_draws = 400)))
    expect_true(all(is.finite(posterior::as_draws_matrix(f$draws(variables = "beta_dprime")))),
                label = paste(s, it, ch, "chains"))
  }
})
