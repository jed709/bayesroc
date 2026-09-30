
.tol_batch <- 0.03

.fit_stan_short <- function(m) {
  fit_broc(m, backend = "stan", refresh = 50, seed = 1,
           parallel_chains = 4, iter_warmup = 400, iter_sampling = 400)
}

.assert_batch_plain_agree <- function(formula, data, vars, tol = .tol_batch) {
  mb <- broc(formula, data = data)
  mp <- .broc_plain(formula, data = data)
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
