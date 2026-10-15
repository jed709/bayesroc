test_that("a cumulative response formula takes no predictors", {
  expect_identical(.cumulative_rhs("1", "cutpoints"), "0")
  expect_identical(.cumulative_rhs("0", "cutpoints"), "0")
  expect_error(.cumulative_rhs("x", "cutpoints"), "cutpoints ~ x")
  expect_error(.cumulative_rhs("1 + (1 | sub)", "cutpoints"), "cutpoints_scale")
  expect_error(.cumulative_rhs("f", "cutpoints2", "The mu2 formula"), "The mu2 formula")
})

test_that("model options set in brf() reach broc(), and broc() takes precedence", {
  d <- .fixture_univariate()
  pr <- broc_prior("normal(1, 0.3)", class = "b", dpar = "dprime")
  m <- suppressMessages(broc(brf(item_resp | is_old ~ 1, family = evsd(), priors = pr), d))
  expect_match(m$stan_code, "normal(1, 0.3)", fixed = TRUE)
  pr2 <- broc_prior("normal(2, 0.3)", class = "b", dpar = "dprime")
  m2 <- suppressMessages(broc(brf(item_resp | is_old ~ 1, family = evsd(), priors = pr), d, priors = pr2))
  expect_match(m2$stan_code, "normal(2, 0.3)", fixed = TRUE)
  expect_warning(suppressMessages(broc(brf(item_resp | is_old ~ 1, family = evsd(), smooth_info = TRUE), d)),
                 "no population-level smooth terms")
})

test_that("threads_per_chain builds the reduce_sum variant at fit time", {
  d <- .fixture_crossed()
  m <- suppressMessages(broc(brf(item_resp | is_old ~ 1 + (1 | sub), family = evsd()), d))
  expect_false(grepl("reduce_sum", m$stan_code, fixed = TRUE))
  mt <- .with_threads(m)
  expect_true(grepl("reduce_sum", mt$stan_code, fixed = TRUE))
  expect_gte(mt$stan_data$grainsize, 1L)
  dc <- d
  mc <- .with_threads(suppressMessages(broc(brf(item_resp ~ 1, cutpoints ~ 1 + (1 | sub), family = cumulative()), dc)))
  expect_true(grepl("reduce_sum", mc$stan_code, fixed = TRUE))
  expect_false(grepl("criterion", mc$stan_code, fixed = TRUE))
  skip_if_no_cmdstan()
  f <- suppressMessages(fit_broc(m, backend = "stan", chains = 2, parallel_chains = 2, iter_warmup = 200,
                                 iter_sampling = 200, seed = 1, refresh = 50, threads_per_chain = 2))
  expect_s3_class(f, "broc_fit")
  expect_true(grepl("reduce_sum", attr(f, "broc_model")$stan_code, fixed = TRUE))
})

test_that("keep_re drops raw random-effect draws on the Stan backend, including when saved", {
  skip_if_no_cmdstan()
  d <- .fixture_crossed()
  m <- suppressMessages(broc(brf(item_resp | is_old ~ 1 + (1 | sub), family = evsd()), d))
  for (kr in c(FALSE, TRUE)) {
    f <- suppressMessages(fit_broc(m, backend = "stan", chains = 2, parallel_chains = 2, iter_warmup = 200,
                                   iter_sampling = 200, seed = 1, refresh = 50, keep_re = kr))
    v <- posterior::variables(f$draws())
    expect_identical(any(grepl("^u_dprime_sub_raw", v)), kr)
    expect_true(any(grepl("^u_dprime_sub\\[", v)))
    tf <- tempfile(fileext = ".rds")
    save_broc_fit(f, tf, verbose = FALSE)
    g <- load_broc_fit(tf)
    expect_identical(any(grepl("^u_dprime_sub_raw", posterior::variables(g$draws()))), kr)
    expect_identical(g$.keep_re, kr)
  }
})

test_that("seed makes pp_check and plot_roc draw subsets reproducible", {
  skip_if_no_cmdstan()
  d <- .fixture_univariate()
  m <- suppressMessages(broc(brf(item_resp | is_old ~ 1, family = evsd()), d))
  f <- suppressMessages(fit_broc(m, backend = "stan", chains = 2, parallel_chains = 2, iter_warmup = 200,
                                 iter_sampling = 200, seed = 1, refresh = 50))
  a <- ggplot2::ggplot_build(pp_check(f, ndraws = 20, seed = 4))$data
  b <- ggplot2::ggplot_build(pp_check(f, ndraws = 20, seed = 4))$data
  expect_equal(a, b)
  r1 <- ggplot2::ggplot_build(plot_roc(f, ndraws = 20, seed = 4))$data
  r2 <- ggplot2::ggplot_build(plot_roc(f, ndraws = 20, seed = 4))$data
  expect_equal(r1, r2)
  set.seed(99); x1 <- runif(1)
  set.seed(99); invisible(pp_check(f, ndraws = 20, seed = 4)); x2 <- runif(1)
  expect_identical(x1, x2)
})
