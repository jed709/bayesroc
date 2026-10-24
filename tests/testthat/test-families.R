
.fast_iter   <- if (identical(Sys.getenv("BAYESROC_TEST_FAST", "true"), "true")) 300 else 1000
.fast_chains <- 4

.renders <- function(g) {
  expect_s3_class(g, "ggplot")
  expect_no_error(ggplot2::ggplot_build(g))
}

.assert_post_fit_works <- function(fit, has_roc = TRUE,
                                    pp_check_types = c("bars", "rootogram", "stat", "lines")) {
  s <- summary(fit)
  expect_s3_class(s, "broc_summary")

  p <- predict(fit, summary = FALSE, ndraws = 50, seed = 1)
  expect_true(is.array(p))

  for (t in pp_check_types) .renders(pp_check(fit, type = t, ndraws = 50))
  .renders(plot(fit))
  .renders(plot_estimates(fit, scale = "coef"))
  .renders(plot_distributions(fit, ndraws = 30))
  if (has_roc) {
    .renders(plot_roc(fit, ndraws = 50))
    .renders(plot_roc(fit, ndraws = 50, scale = "z"))
  }
}

test_that("evsd: smoke", {
  skip_if_no_cmdstan()
  d <- .fixture_univariate()
  m <- broc(brf(item_resp | is_old ~ 1, family = evsd()), d)
  fit <- fit_broc(m, backend = "stan", refresh = 50,
                  parallel_chains = .fast_chains,
                  iter_warmup = .fast_iter, iter_sampling = .fast_iter)
  .assert_post_fit_works(fit)
})

test_that("metad: smoke", {
  skip_if_no_cmdstan()
  d <- .sim_metad(m_old = 0.7)
  m <- suppressMessages(broc(brf(conf | old ~ 1 + (1 | sub), criterion ~ 1 + (1 | sub), mratio_old ~ cond,
                                 mratio_new ~ 1, family = metad()), d))
  fit <- fit_broc(m, backend = "stan", refresh = 50,
                  parallel_chains = .fast_chains,
                  iter_warmup = .fast_iter, iter_sampling = .fast_iter)
  .assert_post_fit_works(fit)
  est <- estimates(fit)
  expect_setequal(intersect(est$parameter, c("mratio_old", "mratio_new")), c("mratio_old", "mratio_new"))
  expect_true(all(est$estimate[est$parameter %in% c("mratio_old", "mratio_new")] > 0))
})

test_that("evsd with Gaussian meta-noise: smoke", {
  skip_if_no_cmdstan()
  d <- .sim_gnoise()
  m <- suppressMessages(broc(brf(conf | old ~ 1 + (1 | sub), criterion ~ 1 + (1 | sub), sigma_meta ~ 1,
                                 family = evsd()), d))
  fit <- fit_broc(m, backend = "stan", refresh = 50,
                  parallel_chains = .fast_chains,
                  iter_warmup = .fast_iter, iter_sampling = .fast_iter)
  .assert_post_fit_works(fit)
  est <- estimates(fit)
  expect_true("sigma_meta" %in% est$parameter)
})

test_that("evsd with lognormal meta-noise: smoke", {
  skip_if_no_cmdstan()
  d <- .sim_lnoise()
  m <- suppressMessages(broc(brf(conf | old ~ 1 + (1 | sub), criterion ~ 1 + (1 | sub), sigma_meta ~ 1,
                                 family = evsd(meta_noise = "lognormal")), d))
  fit <- fit_broc(m, backend = "stan", refresh = 50,
                  parallel_chains = .fast_chains,
                  iter_warmup = .fast_iter, iter_sampling = .fast_iter)
  .assert_post_fit_works(fit)
  est <- estimates(fit)
  expect_true("sigma_meta" %in% est$parameter)
})

test_that("uvsd: smoke", {
  skip_if_no_cmdstan()
  d <- .fixture_univariate(sigma = 1.3)
  m <- broc(brf(item_resp | is_old ~ 1, sigma ~ 1, family = uvsd()), d)
  fit <- fit_broc(m, backend = "stan", refresh = 50,
                  parallel_chains = .fast_chains,
                  iter_warmup = .fast_iter, iter_sampling = .fast_iter)
  .assert_post_fit_works(fit)
})

test_that("dpsd: smoke", {
  skip_if_no_cmdstan()
  d <- .fixture_univariate()
  m <- broc(brf(item_resp | is_old ~ 1, lambda ~ 1, family = dpsd()), d)
  fit <- fit_broc(m, backend = "stan", refresh = 50,
                  parallel_chains = .fast_chains,
                  iter_warmup = .fast_iter, iter_sampling = .fast_iter)
  .assert_post_fit_works(fit)
})

test_that("mixture: smoke", {
  skip_if_no_cmdstan()
  d <- .fixture_univariate()
  m <- broc(brf(item_resp | is_old ~ 1,
                dprime2 ~ 1, lambda ~ 1, family = mixture()), d)
  fit <- fit_broc(m, backend = "stan", refresh = 50,
                  parallel_chains = .fast_chains,
                  iter_warmup = .fast_iter, iter_sampling = .fast_iter)
  .assert_post_fit_works(fit)
})

test_that("source_mixture: smoke", {
  skip_if_no_cmdstan()
  d <- .fixture_source_mixture()
  m <- broc(brf(item_resp | source ~ 1, lambda ~ 1, family = source_mixture()),
            d)
  fit <- fit_broc(m, backend = "stan", refresh = 50,
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
                family = bivariate_gaussian()), d)
  fit <- fit_broc(m, backend = "stan", refresh = 50,
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
                family = bivariate_dp()), d)
  fit <- fit_broc(m, backend = "stan", refresh = 50,
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
                family = vrdp2d()), d)
  fit <- fit_broc(m, backend = "stan", refresh = 50,
                  parallel_chains = .fast_chains,
                  iter_warmup = .fast_iter, iter_sampling = .fast_iter)
  .assert_post_fit_works(fit, pp_check_types = "bars")
})

test_that("cumulative: smoke", {
  skip_if_no_cmdstan()
  d <- .fixture_univariate()
  m <- broc(brf(item_resp ~ 1, cutpoints ~ cond, cutpoints_scale ~ 1, family = cumulative()),
            d)
  fit <- fit_broc(m, backend = "stan", refresh = 50,
                  parallel_chains = .fast_chains,
                  iter_warmup = .fast_iter, iter_sampling = .fast_iter)
  s <- summary(fit); expect_s3_class(s, "broc_summary")
  p <- predict(fit, summary = FALSE, ndraws = 50, seed = 1)
  expect_true(is.array(p))
  .renders(pp_check(fit, type = "bars", ndraws = 50))
  .renders(plot(fit))
  .renders(plot_distributions(fit, ndraws = 30))
  expect_error(plot_roc(fit, ndraws = 50), "not defined for the cumulative")
})

test_that("bivariate_cumulative: smoke", {
  skip_if_no_cmdstan()
  d <- .fixture_bivariate_cumulative()
  m <- broc(brf(resp(y1, y2) ~ 1,
                discrim ~ 1, criterion ~ 1, criterion2 ~ 1, rho ~ 1,
                family = bivariate_cumulative()), d)
  fit <- fit_broc(m, backend = "stan", refresh = 50,
                  parallel_chains = .fast_chains,
                  iter_warmup = .fast_iter, iter_sampling = .fast_iter)
  s <- summary(fit); expect_s3_class(s, "broc_summary")
  p <- predict(fit, summary = FALSE, ndraws = 50, seed = 1)
  expect_true(is.array(p))
  for (resp in 1:2) {
    .renders(pp_check(fit, type = "bars", ndraws = 50, response = resp))
    .renders(plot_distributions(fit, response = resp, ndraws = 30))
  }
  .renders(plot(fit))
})

test_that("cdp: smoke (rotello_2005)", {
  skip_if_no_cmdstan()
  d <- .fixture_cdp()
  m <- broc(brf(resp(rating, rk) | is_old ~ condition - 1,
                fam ~ condition - 1, criterion ~ condition - 1,
                rec_crit ~ condition - 1, sigma_R ~ condition - 1,
                family = cdp(old_levels = 2:6), counts = "count"),
            d)
  fit <- fit_broc(m, backend = "stan", refresh = 50,
                  parallel_chains = .fast_chains,
                  iter_warmup = .fast_iter, iter_sampling = .fast_iter)
  s <- summary(fit); expect_s3_class(s, "broc_summary")
  p <- predict(fit, summary = FALSE, ndraws = 50, seed = 1)
  expect_true(is.array(p))
  for (t in c("bars", "rootogram")) .renders(pp_check(fit, type = t, by = "condition", ndraws = 50))
  .renders(pp_check(fit, by = "condition", response = 2, ndraws = 50))
  .renders(plot_roc(fit, by = "condition", ndraws = 50))
  .renders(plot_distributions(fit, by = "condition", ndraws = 30))
  .renders(plot(fit))
})
