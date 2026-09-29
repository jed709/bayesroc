
test_that("drghmc: posterior means agree with nuts (jax)", {
  skip_if_no_jax()
  d <- .fixture_univariate(sigma = 1.3)
  m <- broc(brf(item_resp | is_old ~ 1, sigma ~ 1, family = uvsd()), d,
            batch_likelihood = TRUE)
  fn <- fit_broc(m, backend = "jax", sampler = "nuts", seed = 1,
                 iter_warmup = 500, iter_sampling = 500, chains = 4,
                 progress_bar = FALSE)
  fj <- fit_broc(m, backend = "jax", sampler = "drghmc", seed = 1,
                 iter_warmup = 1000, iter_sampling = 3000, chains = 4,
                 progress_bar = FALSE)
  vars <- c("beta_dprime[1]", "beta_sigma[1]", "beta_thresh_mid[1]")
  sn <- posterior::as_draws_matrix(fn$draws(variables = vars))
  sj <- posterior::as_draws_matrix(fj$draws(variables = vars))
  for (v in vars) {
    expect_lt(abs(mean(sn[, v]) - mean(sj[, v])), 0.05,
              label = sprintf("|mean_nuts - mean_drghmc|(%s)", v))
  }
})

test_that("drghmc: diagnostics carry DR stages, no divergences/treedepth", {
  skip_if_no_jax()
  d <- .fixture_univariate(sigma = 1.3)
  m <- broc(brf(item_resp | is_old ~ 1, sigma ~ 1, family = uvsd()), d,
            batch_likelihood = TRUE)
  fj <- fit_broc(m, backend = "jax", sampler = "drghmc", seed = 1,
                 iter_warmup = 800, iter_sampling = 1500, chains = 4,
                 progress_bar = FALSE)
  dd <- fj$diagnostic_summary()
  expect_true(all(c("accept", "dr_stage") %in% names(dd)))
  expect_named(dd$dr_stage, c("s1", "s2", "s3", "rej"))
  expect_equal(sum(dd$dr_stage), 1, tolerance = 1e-6)
  banner <- capture.output(print(summary(fj)))
  expect_true(any(grepl("Sampler \\(drghmc\\):", banner)))
  expect_false(any(grepl("divergent|treedepth", banner)))
})

test_that("drghmc: control knobs plumb through; unknown keys warn", {
  skip_if_no_jax()
  d <- .fixture_univariate(sigma = 1.3)
  m <- broc(brf(item_resp | is_old ~ 1, sigma ~ 1, family = uvsd()), d,
            batch_likelihood = TRUE)
  expect_warning(
    fit_broc(m, backend = "jax", sampler = "drghmc", control = list(bogus = 1),
             iter_warmup = 200, iter_sampling = 200, chains = 4,
             progress_bar = FALSE),
    "unknown control keys")
  fj <- fit_broc(m, backend = "jax", sampler = "drghmc",
                 control = list(precision = "f32"), seed = 1,
                 iter_warmup = 800, iter_sampling = 1500, chains = 4,
                 progress_bar = FALSE)
  sm <- fj$summary(variables = "beta_dprime[1]")
  expect_lt(sm$rhat[1], 1.05)
})

test_that("drghmc requires the jax backend", {
  d <- .fixture_univariate()
  m <- broc(brf(item_resp | is_old ~ 1, sigma ~ 1, family = uvsd()), d,
            batch_likelihood = TRUE)
  expect_error(fit_broc(m, backend = "stan", sampler = "drghmc"),
               "requires backend = 'jax'")
})
