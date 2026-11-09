# Helpers shared with tests/testthat/test-prior-distributions.R (copied).


test_that("Stan code emits translated densities for half_*/laplace priors", {
  d <- .fixture_univariate()
  d$sub <- factor(rep(1:6, length.out = nrow(d)))
  p <- c(broc_prior("laplace(0, 1)", class = "b", dpar = "dprime"),
         broc_prior("half_normal(0.5)", class = "sd", dpar = "dprime", group = "sub"))
  m <- broc(brf(item_resp | is_old ~ 1 + (1 | sub), sigma ~ 1, family = uvsd()),
            data = d, prior = p)
  code <- get_stan_code(.with_stan_code(m))
  expect_true(grepl("beta_dprime ~ double_exponential(0, 1);", code, fixed = TRUE))
  expect_true(grepl("sigma_dprime_sub ~ normal(0, 0.5);", code, fixed = TRUE))
  expect_false(grepl("half_normal", code, fixed = TRUE))
  expect_false(grepl("laplace(", code, fixed = TRUE))
})

test_that("lkj correlation priors written as lkj() reach Stan as lkj_corr_cholesky()", {
  d <- .fixture_crossed()
  m <- suppressMessages(broc(brf(item_resp | is_old ~ 1 + (1 + is_old | sub), family = evsd()), d,
                             priors = broc_prior("lkj(2)", class = "cor")))
  code <- get_stan_code(.with_stan_code(m))
  expect_true(grepl("lkj_corr_cholesky(2)", code, fixed = TRUE))
  expect_false(grepl("~ lkj(", code, fixed = TRUE))
})

test_that("to_stan_prior translates JAX-style names; identity otherwise", {
  expect_equal(to_stan_prior("laplace(0, 1)"), "double_exponential(0, 1)")
  expect_equal(to_stan_prior("half_normal(0.5)"), "normal(0, 0.5)")
  expect_equal(to_stan_prior("half_cauchy(2)"), "cauchy(0, 2)")
  expect_equal(to_stan_prior("half_student_t(3, 2.5)"), "student_t(3, 0, 2.5)")
  expect_equal(to_stan_prior("normal(0, 1)"), "normal(0, 1)")
  expect_equal(to_stan_prior("student_t(3, 0, 2.5)"), "student_t(3, 0, 2.5)")
  expect_equal(to_stan_prior("lkj(2)"), "lkj(2)")
})
