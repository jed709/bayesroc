# Per-family smoke tests. Each family fits a tiny synthetic dataset under
# Stan, then walks summary -> predict -> pp_check -> plot_roc, asserting
# that nothing errors and shapes are right. JAX is exercised separately in
# test-stan_jax_agreement.R.
#
# Each fit uses very few iterations (300/300) — we're testing wiring, not
# convergence. Set BAYESROC_TEST_FAST=false to run a longer suite.

.fast_iter   <- if (identical(Sys.getenv("BAYESROC_TEST_FAST", "true"), "true")) 300 else 1000
.fast_chains <- 4

# Helper: run the standard set of post-fit calls and assert they don't error.
.assert_post_fit_works <- function(fit, has_roc = TRUE,
                                    pp_check_types = c("bars", "bars_grouped")) {
  s <- summary(fit)
  expect_s3_class(s, "broc_summary")

  p <- predict(fit, summary = FALSE, ndraws = 50, seed = 1)
  expect_true(is.array(p))

  for (t in pp_check_types) {
    g <- pp_check(fit, type = t, ndraws = 50)
    expect_s3_class(g, "ggplot")
  }

  if (has_roc) {
    g <- plot_roc_curve(fit, ndraws = 50)
    expect_s3_class(g, "ggplot")
  }
}


test_that("evsd: smoke", {
  skip_if_no_cmdstan()
  d <- .fixture_univariate()
  m <- broc(brf(item_resp | is_old ~ 1, family = evsd()), d,
            batch_likelihood = TRUE)
  fit <- fit_broc(m, backend = "stan", refresh = 0,
                  parallel_chains = .fast_chains,
                  iter_warmup = .fast_iter, iter_sampling = .fast_iter)
  .assert_post_fit_works(fit)
})

test_that("uvsd: smoke", {
  skip_if_no_cmdstan()
  d <- .fixture_univariate(sigma = 1.3)
  m <- broc(brf(item_resp | is_old ~ 1, sigma ~ 1, family = uvsd()), d,
            batch_likelihood = TRUE)
  fit <- fit_broc(m, backend = "stan", refresh = 0,
                  parallel_chains = .fast_chains,
                  iter_warmup = .fast_iter, iter_sampling = .fast_iter)
  .assert_post_fit_works(fit)
})

test_that("dpsd: smoke", {
  skip_if_no_cmdstan()
  d <- .fixture_univariate()
  m <- broc(brf(item_resp | is_old ~ 1, lambda ~ 1, family = dpsd()), d,
            batch_likelihood = TRUE)
  fit <- fit_broc(m, backend = "stan", refresh = 0,
                  parallel_chains = .fast_chains,
                  iter_warmup = .fast_iter, iter_sampling = .fast_iter)
  .assert_post_fit_works(fit)
})

test_that("mixture: smoke", {
  skip_if_no_cmdstan()
  d <- .fixture_univariate()
  m <- broc(brf(item_resp | is_old ~ 1,
                dprime2 ~ 1, lambda ~ 1, family = mixture()), d,
            batch_likelihood = TRUE)
  fit <- fit_broc(m, backend = "stan", refresh = 0,
                  parallel_chains = .fast_chains,
                  iter_warmup = .fast_iter, iter_sampling = .fast_iter)
  .assert_post_fit_works(fit)
})

test_that("source_mixture: smoke", {
  skip_if_no_cmdstan()
  d <- .fixture_source_mixture()
  m <- broc(brf(item_resp | source ~ 1, lambda ~ 1, family = source_mixture()),
            d, batch_likelihood = TRUE)
  fit <- fit_broc(m, backend = "stan", refresh = 0,
                  parallel_chains = .fast_chains,
                  iter_warmup = .fast_iter, iter_sampling = .fast_iter)
  s <- summary(fit); expect_s3_class(s, "broc_summary")
  p <- predict(fit, summary = FALSE, ndraws = 50, seed = 1)
  expect_true(is.array(p))
})

test_that("bivariate_gaussian: smoke", {
  skip_if_no_cmdstan()
  d <- .fixture_bivariate_sdt()
  m <- broc(brf(resp(y1, y2) | item_type ~ 1,
                discrim ~ 1, criterion ~ 1, criterion2 ~ 1, rho ~ 1,
                family = bivariate_gaussian()), d, batch_likelihood = TRUE)
  fit <- fit_broc(m, backend = "stan", refresh = 0,
                  parallel_chains = .fast_chains,
                  iter_warmup = .fast_iter, iter_sampling = .fast_iter)
  .assert_post_fit_works(fit, pp_check_types = "bars")
})

test_that("bivariate_dp: smoke", {
  skip_if_no_cmdstan()
  d <- .fixture_bivariate_sdt()
  m <- broc(brf(resp(y1, y2) | item_type ~ 1,
                discrim ~ 1, lambda ~ 1, lambda2 ~ 1,
                criterion ~ 1, criterion2 ~ 1, rho ~ 1,
                family = bivariate_dp()), d, batch_likelihood = TRUE)
  fit <- fit_broc(m, backend = "stan", refresh = 0,
                  parallel_chains = .fast_chains,
                  iter_warmup = .fast_iter, iter_sampling = .fast_iter)
  .assert_post_fit_works(fit, pp_check_types = "bars")
})

test_that("vrdp2d: smoke", {
  skip_if_no_cmdstan()
  d <- .fixture_bivariate_sdt()
  m <- broc(brf(resp(y1, y2) | item_type ~ 1,
                dprime2 ~ 1, discrim ~ 1, lambda ~ 1,
                criterion ~ 1, criterion2 ~ 1,
                family = vrdp2d()), d, batch_likelihood = TRUE)
  fit <- fit_broc(m, backend = "stan", refresh = 0,
                  parallel_chains = .fast_chains,
                  iter_warmup = .fast_iter, iter_sampling = .fast_iter)
  .assert_post_fit_works(fit, pp_check_types = "bars")
})

test_that("cumulative: smoke", {
  skip_if_no_cmdstan()
  d <- .fixture_univariate()
  m <- broc(brf(item_resp ~ cond, cutpoints ~ 1, family = cumulative()),
            d, batch_likelihood = TRUE)
  fit <- fit_broc(m, backend = "stan", refresh = 0,
                  parallel_chains = .fast_chains,
                  iter_warmup = .fast_iter, iter_sampling = .fast_iter)
  s <- summary(fit); expect_s3_class(s, "broc_summary")
  p <- predict(fit, summary = FALSE, ndraws = 50, seed = 1)
  expect_true(is.array(p))
  g <- pp_check(fit, type = "bars", ndraws = 50)
  expect_s3_class(g, "ggplot")
  # plot_roc_curve intentionally errors for cumulative — verify
  expect_error(plot_roc_curve(fit, ndraws = 50), "not supported for 'cumulative'")
})

test_that("bivariate_cumulative: smoke", {
  skip_if_no_cmdstan()
  d <- .fixture_bivariate_cumulative()
  m <- broc(brf(resp(y1, y2) ~ 1,
                discrim ~ 1, criterion ~ 1, criterion2 ~ 1, rho ~ 1,
                family = bivariate_cumulative()), d, batch_likelihood = TRUE)
  fit <- fit_broc(m, backend = "stan", refresh = 0,
                  parallel_chains = .fast_chains,
                  iter_warmup = .fast_iter, iter_sampling = .fast_iter)
  s <- summary(fit); expect_s3_class(s, "broc_summary")
  p <- predict(fit, summary = FALSE, ndraws = 50, seed = 1)
  expect_true(is.array(p))
  for (resp in 1:2) {
    g <- pp_check(fit, type = "bars", ndraws = 50, response = resp)
    expect_s3_class(g, "ggplot")
  }
})

test_that("cdp: smoke (rotello_2005)", {
  skip_if_no_cmdstan()
  d <- .fixture_cdp()
  m <- broc(brf(resp(rating, rk) | is_old ~ condition - 1,
                fam ~ condition - 1, criterion ~ condition - 1,
                rec_crit ~ condition - 1, sigma_R ~ condition - 1,
                family = cdp(old_levels = 2:6), counts = "count"),
            d, batch_likelihood = TRUE)
  fit <- fit_broc(m, backend = "stan", refresh = 0,
                  parallel_chains = .fast_chains,
                  iter_warmup = .fast_iter, iter_sampling = .fast_iter)
  s <- summary(fit); expect_s3_class(s, "broc_summary")
  p <- predict(fit, summary = FALSE, ndraws = 50, seed = 1)
  expect_true(is.array(p))
  for (t in c("bars", "bars_grouped")) {
    g <- pp_check(fit, type = t, group = "condition", ndraws = 50)
    expect_s3_class(g, "ggplot")
  }
  # response = 2 (R/K dimension)
  g <- pp_check(fit, type = "bars_grouped", group = "condition",
                response = 2, ndraws = 50)
  expect_s3_class(g, "ggplot")
})
