.sp_model <- function() {
  data(prm09, package = "bayesroc", envir = environment())
  d <- prm09[prm09$sub %in% unique(prm09$sub)[1:5], ]
  d$sub <- factor(d$sub)
  suppressMessages(broc(brf(resp | cond ~ 1 + (1 | sub), sigma ~ 1, family = uvsd()), data = d))
}

test_that("sample_prior = 'only' adds a prior_only flag to the model data and the JAX config", {
  m <- .sp_model()
  expect_identical(m$stan_data$prior_only, 0L)
  expect_false(build_numpyro_config(m)$prior_only)
  expect_error(fit_broc(m, sample_prior = "only", loo = TRUE), "needs the likelihood")
})

test_that("sample_prior = 'only' runs on JAX without a likelihood and is reported", {
  skip_if_no_jax()
  m <- .sp_model()
  f <- fit_broc(m, sample_prior = "only", chains = 1,
                iter_warmup = 50, iter_sampling = 10, seed = 1, progress_bar = FALSE)
  expect_null(f$.loglik_path)
  expect_true(all(is.finite(posterior::as_draws_matrix(f$draws(variables = "beta_dprime")))))
  expect_output(print(summary(f)), "sampled from the prior only")
  p <- predict(f, type = "prediction", summary = FALSE, ndraws = 10)
  expect_equal(dim(p), c(10L, nrow(m$data)))
})
