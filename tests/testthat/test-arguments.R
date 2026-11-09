test_that("a cumulative response formula takes no predictors", {
  expect_identical(.cumulative_rhs("1", "cutpoints"), "0")
  expect_identical(.cumulative_rhs("0", "cutpoints"), "0")
  expect_error(.cumulative_rhs("x", "cutpoints"), "cutpoints ~ x")
  expect_error(.cumulative_rhs("1 + (1 | sub)", "cutpoints"), "cutpoints_shift")
  expect_error(.cumulative_rhs("f", "cutpoints2", "The mu2 formula"), "The mu2 formula")
})

test_that("model options set in brf() reach broc(), and broc() takes precedence", {
  d <- .fixture_univariate()
  pr <- broc_prior("normal(1, 0.3)", class = "b", dpar = "dprime")
  m <- suppressMessages(broc(brf(item_resp | is_old ~ 1, family = evsd(), priors = pr), d))
  expect_identical(unname(unlist(build_numpyro_config(m)$params$dprime$priors)), "normal(1, 0.3)")
  pr2 <- broc_prior("normal(2, 0.3)", class = "b", dpar = "dprime")
  m2 <- suppressMessages(broc(brf(item_resp | is_old ~ 1, family = evsd(), priors = pr), d, priors = pr2))
  expect_identical(unname(unlist(build_numpyro_config(m2)$params$dprime$priors)), "normal(2, 0.3)")
  expect_warning(suppressMessages(broc(brf(item_resp | is_old ~ 1, family = evsd(), smooth_info = TRUE), d)),
                 "no population-level smooth terms")
})

test_that("seed makes pp_check and plot_roc draw subsets reproducible", {
  skip_if_no_jax()
  f <- .fx_fit("evsd")
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
