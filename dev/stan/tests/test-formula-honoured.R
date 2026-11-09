# Helpers shared with tests/testthat/
# test-formula-honoured.R (copied).

.honour_data <- function(seed = 1) {
  set.seed(seed)
  d <- expand.grid(subj = factor(1:10), item = factor(1:24))
  n <- nrow(d)
  d$a <- factor(sample(c("a1", "a2"), n, TRUE))
  d$b <- factor(sample(c("b1", "b2"), n, TRUE))
  d$cc <- factor(sample(c("c1", "c2"), n, TRUE))
  d$x <- rnorm(n)
  d$old <- sample(0:1, n, TRUE)
  d$conf <- sample(1:6, n, TRUE)
  d$item_type <- sample(1:3, n, TRUE)
  d$conf2 <- sample(1:6, n, TRUE)
  d
}
.dprime_cols <- function(f, d = .honour_data(), ...) colnames(suppressMessages(broc(f, d, ...))$stan_data$X_dprime)
.l2_smooth_model <- function(batch = TRUE) {
  bv <- .fixture_bivariate_sdt()
  set.seed(10); bv$x <- rnorm(nrow(bv))
  .broc_with(batch, brf(resp(y1, y2) | item_type ~ 1, discrim ~ 1, rho ~ 1, criterion ~ 1, criterion2 ~ 1,
                        lambda ~ 1, lambda2 ~ 1 + s(x, k = 5), lambda2_B ~ 1 + s(x, k = 5),
                        family = bivariate_dp()), bv)
}

test_that("lambda2 and lambda2_B smooths: batch and plain Stan agree", {
  skip_if_no_cmdstan()
  r <- .stan_parity(suppressMessages(.l2_smooth_model(FALSE)), suppressMessages(.l2_smooth_model(TRUE)))
  expect_equal(r$dim[1], r$dim[2])
  expect_lt(r$lp, 1e-8)
  expect_lt(r$grad, 1e-8)
})
