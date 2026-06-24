# API ergonomics guards (Tier 4 quick wins):
#   - external/internal dpar aliasing (posterior_linpred accepts user-facing names)
#   - plot.broc_fit has a working default dpar
#   - summary() print shows a sampler-diagnostics banner

test_that("external_to_internal_param maps user-facing aliases", {
  # cdp: rec/fam/sigma_R/sigma_F <-> dprime/dprime2/sigma/sigma2
  expect_equal(external_to_internal_param("rec", "cdp"), "dprime")
  expect_equal(external_to_internal_param("fam", "cdp"), "dprime2")
  expect_equal(external_to_internal_param("sigma_R", "cdp"), "sigma")
  # cumulative: mu/cutpoints <-> dprime/criterion
  expect_equal(external_to_internal_param("mu", "cumulative"), "dprime")
  # non-alias passes through unchanged
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
  # brf path reproduces the build-first path exactly
  expect_identical(p_model, p_brf)
  # bare-formula path yields the same spec (sigma/criterion default to ~ 1)
  expect_equal(nrow(p_formula), nrow(p_brf))

  # user_priors override still applies on the formula path
  up <- broc_prior("normal(2, 0.5)", class = "b", dpar = "dprime")
  expect_true(any(get_broc_prior(spec, data = d, user_priors = up)$source == "user"))

  # formula path without data is an error
  expect_error(get_broc_prior(item_resp | is_old ~ 1, family = uvsd()), "data")
  # unsupported object type is an error
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
  m <- broc(brf(item_resp | is_old ~ 1, sigma ~ 1, family = uvsd()), d)
  fit <- fit_broc(m, backend = "stan", refresh = 0, seed = 1,
                  parallel_chains = 4, iter_warmup = 300, iter_sampling = 300)

  # plot(fit) must not error with no dpar supplied
  p <- plot(fit)
  expect_s3_class(p, "ggplot")

  # summary print shows the sampler-diagnostics banner
  out <- paste(capture.output(print(summary(fit))), collapse = "\n")
  expect_match(out, "Sampler:")
  expect_match(out, "E-BFMI")

  # posterior_linpred accepts an internal dpar (alias path covered by unit test above)
  lp <- posterior_linpred(fit, dpar = "sigma", summary = TRUE)
  expect_true(nrow(lp) == nrow(d))
})
