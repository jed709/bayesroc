test_that("posterior_linpred returns S x N matrix", {
  skip_if_no_jax()
  d <- .fixture_univariate(sigma = 1.3)
  fit <- .fx_fit("uvsd")
  lp <- posterior_linpred(fit, dpar = "dprime", ndraws = 100, seed = 1)
  expect_true(is.matrix(lp))
  expect_equal(nrow(lp), 100)
  expect_equal(ncol(lp), nrow(d))
})

test_that("posterior_linpred applies link transform (sigma post-exp)", {
  skip_if_no_jax()
  fit <- .fx_fit("uvsd")
  s <- posterior_linpred(fit, dpar = "sigma", ndraws = 50, seed = 1)
  expect_true(all(s > 0))
})

test_that("posterior_linpred summary = TRUE returns tidy data frame", {
  skip_if_no_jax()
  d <- .fixture_univariate()
  fit <- .fx_fit("evsd")
  df <- posterior_linpred(fit, dpar = "dprime", summary = TRUE,
                               ndraws = 100, seed = 1)
  expect_s3_class(df, "data.frame")
  expect_named(df, c("trial", "mean", "lower", "upper"))
  expect_equal(nrow(df), nrow(d))
})

test_that("posterior_epred returns expected category in [1, K]", {
  skip_if_no_jax()
  fit <- .fx_fit("evsd")
  ep <- posterior_epred(fit, ndraws = 50, seed = 1)
  expect_true(all(ep >= 1 & ep <= 6))
})

test_that("predict(seed = ...) is reproducible", {
  skip_if_no_jax()
  fit <- .fx_fit("evsd")
  p1 <- predict(fit, summary = FALSE, ndraws = 50, seed = 7)
  p2 <- predict(fit, summary = FALSE, ndraws = 50, seed = 7)
  expect_equal(p1, p2, tolerance = 0)
  p3 <- predict(fit, summary = FALSE, ndraws = 50, seed = 8)
  expect_false(isTRUE(all.equal(p1, p3, tolerance = 0)))
})
