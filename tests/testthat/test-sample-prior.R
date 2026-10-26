.sp_model <- function() {
  data(prm09, package = "bayesroc", envir = environment())
  d <- prm09[prm09$sub %in% unique(prm09$sub)[1:5], ]
  d$sub <- factor(d$sub)
  suppressMessages(broc(brf(resp | cond ~ 1 + (1 | sub), sigma ~ 1, family = uvsd()), data = d))
}

.sp_check <- function(fit) {
  dr <- posterior::as_draws_df(fit$draws())
  b <- dr[["beta_dprime[1]"]]; s <- dr[["beta_sigma[1]"]]
  t <- dr[[grep("^sigma_dprime_sub", names(dr), value = TRUE)[1]]]
  expect_lt(abs(mean(b) - 1.5), 0.1)
  expect_lt(abs(stats::sd(b) - 0.75), 0.08)
  expect_lt(abs(mean(s)), 0.06)
  expect_lt(abs(stats::sd(s) - 0.5), 0.06)
  expect_lt(abs(mean(t) - sqrt(2 / pi)), 0.1)
}

test_that("sample_prior = 'only' adds a prior_only flag to the Stan data and guards the likelihood", {
  m <- .sp_model()
  expect_identical(m$stan_data$prior_only, 0L)
  expect_match(m$stan_code, "int<lower=0, upper=1> prior_only;", fixed = TRUE)
  expect_match(m$stan_code, "if (!prior_only) {", fixed = TRUE)
  expect_false(build_numpyro_config(m)$prior_only)
  expect_error(fit_broc(m, sample_prior = "only", loo = TRUE), "needs the likelihood")
})

test_that("sample_prior = 'only' removes exactly the likelihood from the Stan log density", {
  skip_if_no_cmdstan()
  m <- .sp_model()
  f <- suppressMessages(fit_broc(m, backend = "stan", sample_prior = "only", save_loglik = TRUE,
                                 loglik_precision = "f64", chains = 1, iter_warmup = 20, iter_sampling = 5,
                                 seed = 1, refresh = 50, show_messages = FALSE))
  expect_identical(attr(f, "broc_model")$stan_data$prior_only, 1L)
  csv <- f$.backend_fit$output_files()[1]
  mod <- .compile_broc_stan(m)
  sd_prior <- m$stan_data; sd_prior$prior_only <- 1L
  lp_prior <- .cmdstan_log_prob(mod, sd_prior, csv)
  lp_post <- .cmdstan_log_prob(mod, m$stan_data, csv)
  ll <- rowSums(unclass(posterior::as_draws_matrix(f$draws(variables = "log_lik"))))
  expect_lt(max(abs((lp_post[, 1] - lp_prior[, 1]) - ll) / pmax(1, abs(ll))), 1e-6)
  expect_true(all(abs(ll) > 1))
})

test_that("sample_prior = 'only' runs on JAX without a likelihood and is reported", {
  skip_if_no_jax()
  m <- .sp_model()
  f <- fit_broc(m, backend = "jax", sampler = "drghmc", sample_prior = "only", chains = 1,
                iter_warmup = 50, iter_sampling = 10, seed = 1, progress_bar = FALSE)
  expect_null(f$.log_lik)
  expect_true(all(is.finite(posterior::as_draws_matrix(f$draws(variables = "beta_dprime")))))
  expect_output(print(summary(f)), "sampled from the prior only")
  p <- predict(f, type = "prediction", summary = FALSE, ndraws = 10)
  expect_equal(dim(p), c(10L, nrow(m$data)))
})

test_that("sample_prior = 'only' samples the prior on JAX DR-G-HMC, JAX NUTS and Stan", {
  skip_if_not_slow()
  skip_on_cran()
  skip_if_no_jax()
  m <- .sp_model()
  f1 <- fit_broc(m, backend = "jax", sampler = "drghmc", sample_prior = "only", chains = 4,
                 iter_warmup = 500, iter_sampling = 1000, seed = 1, progress_bar = FALSE)
  .sp_check(f1)
  expect_null(f1$.log_lik)
  expect_output(print(summary(f1)), "sampled from the prior only")
  f2 <- fit_broc(m, backend = "jax", sampler = "nuts", sample_prior = "only", chains = 4,
                 iter_warmup = 500, iter_sampling = 1000, seed = 1, progress_bar = FALSE)
  .sp_check(f2)
  skip_if_not(requireNamespace("cmdstanr", quietly = TRUE))
  f3 <- fit_broc(m, backend = "stan", sample_prior = "only", chains = 4,
                 iter_warmup = 500, iter_sampling = 1000, seed = 1, progress_bar = FALSE)
  .sp_check(f3)
  p <- predict(f1, type = "prediction", summary = FALSE, ndraws = 20)
  expect_equal(dim(p), c(20L, nrow(m$data)))
})

test_that("R cell probabilities stay exact in the far tails, matching Stan's log-likelihood at prior draws", {
  skip_if_no_cmdstan()
  m <- .sp_model()
  f <- suppressMessages(fit_broc(m, backend = "stan", sample_prior = "only", save_loglik = TRUE,
                                 loglik_precision = "f64", chains = 1, iter_warmup = 20, iter_sampling = 20,
                                 seed = 2, refresh = 50, show_messages = FALSE))
  ll_s <- unclass(posterior::as_draws_matrix(f$draws(variables = "log_lik")))
  ll_r <- .obs_loglik_r(m, build_data_source_from_model(m), posterior::as_draws_matrix(f$draws()), re_formula = NULL)
  expect_lt(min(ll_s), log(1e-10))
  expect_lt(max(abs(ll_r - ll_s) / pmax(1, abs(ll_s))), 1e-6)
})
