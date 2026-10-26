
test_that("external_to_internal_param maps user-facing aliases", {
  expect_equal(external_to_internal_param("rec", "cdp"), "dprime")
  expect_equal(external_to_internal_param("fam", "cdp"), "dprime2")
  expect_equal(external_to_internal_param("sigma_R", "cdp"), "sigma")
  expect_equal(external_to_internal_param("mu", "cumulative"), "dprime")
  expect_equal(external_to_internal_param("dprime", "cdp"), "dprime")
  expect_equal(external_to_internal_param("sigma", "uvsdt"), "sigma")
})

test_that("get_broc_prior accepts a brf/formula without building the model first", {
  d <- .fixture_univariate(sigma = 1.3)
  spec <- brf(item_resp | is_old ~ 1, sigma ~ 1, family = uvsd())

  p_model   <- get_broc_prior(broc(spec, data = d))
  p_brf     <- get_broc_prior(spec, data = d)
  p_formula <- get_broc_prior(item_resp | is_old ~ 1, data = d, family = uvsd())

  expect_s3_class(p_brf, "data.frame")
  expect_true(all(c("class", "dpar", "coef", "group", "prior", "source") %in% names(p_brf)))
  expect_identical(p_model, p_brf)
  expect_equal(nrow(p_formula), nrow(p_brf))

  up <- broc_prior("normal(2, 0.5)", class = "b", dpar = "dprime")
  expect_true(any(get_broc_prior(spec, data = d, priors = up)$source == "user"))

  expect_error(get_broc_prior(item_resp | is_old ~ 1, family = uvsd()), "data")
  expect_error(get_broc_prior(42), "broc_model")
})

test_that("get_numpyro_config exposes the JAX config as list and JSON", {
  d <- .fixture_univariate(sigma = 1.3)
  m <- broc(brf(item_resp | is_old ~ 1, sigma ~ 1, family = uvsd()), d)

  cfg <- get_numpyro_config(m)
  expect_type(cfg, "list")
  expect_identical(cfg$family, "uvsdt")
  expect_equal(cfg$N, nrow(d))
  expect_true(all(c("dprime", "sigma") %in% names(cfg$params)))

  skip_if_not_installed("jsonlite")
  js <- get_numpyro_config(m, json = TRUE)
  expect_s3_class(js, "json")
  expect_identical(jsonlite::fromJSON(js)$family, "uvsdt")

  expect_error(get_numpyro_config(42), "broc_model")
})

test_that("plot() default dpar and summary() diagnostics banner work", {
  skip_if_no_cmdstan()
  d <- .fixture_univariate(sigma = 1.3)
  fit <- .fx_fit("uvsd", "stan")

  p <- plot(fit)
  expect_s3_class(p, "ggplot")

  out <- paste(capture.output(print(summary(fit))), collapse = "\n")
  expect_match(out, "Sampler:")
  expect_match(out, "E-BFMI")

  lp <- posterior_linpred(fit, dpar = "sigma", summary = TRUE)
  expect_true(nrow(lp) == nrow(d))
})
