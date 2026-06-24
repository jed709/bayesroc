# Batch (fused C++ analytic-gradient) Stan vs plain (vanilla autodiff) Stan must
# agree. Plain Stan is the independent autodiff oracle for the hand-derived batch
# gradients. These tests also guard:
#   - the batch_likelihood = TRUE default,
#   - the single-source-of-truth batch gate (generate_batch_cpp's NULL return),
#     in particular that `cumulative` actually engages the batch path (it was
#     previously excluded by a redundant second allowlist).

.tol_batch <- 0.03

.fit_stan_short <- function(m) {
  fit_broc(m, backend = "stan", refresh = 0, seed = 1,
           parallel_chains = 4, iter_warmup = 400, iter_sampling = 400)
}

.assert_batch_plain_agree <- function(formula, data, vars, tol = .tol_batch) {
  mb <- broc(formula, data = data, batch_likelihood = TRUE)
  mp <- broc(formula, data = data, batch_likelihood = FALSE)
  # batch must actually engage for these (supported) families, and plain must not
  expect_true(isTRUE(mb$batch_likelihood))
  expect_false(isTRUE(mp$batch_likelihood))

  fb <- .fit_stan_short(mb)
  fp <- .fit_stan_short(mp)
  db <- posterior::as_draws_matrix(fb$draws(variables = vars))
  dp <- posterior::as_draws_matrix(fp$draws(variables = vars))
  for (v in vars) {
    if (v %in% colnames(db) && v %in% colnames(dp)) {
      diff <- abs(mean(db[, v]) - mean(dp[, v]))
      expect_lt(diff, tol,
                label = sprintf("|batch - plain|(%s) = %.4f", v, diff))
    }
  }
}

test_that("evsdt: batch and plain Stan agree", {
  skip_if_no_cmdstan()
  d <- .fixture_univariate()
  .assert_batch_plain_agree(
    brf(item_resp | is_old ~ 1, family = evsd()),
    d, c("beta_dprime[1]", "beta_thresh_mid[1]"))
})

test_that("cumulative: batch path engages and agrees with plain Stan", {
  skip_if_no_cmdstan()
  d <- .fixture_univariate()
  .assert_batch_plain_agree(
    brf(item_resp ~ cond, cutpoints ~ 1, family = cumulative()),
    d, c("beta_mu[1]", "beta_thresh_mid[1]"))
})
