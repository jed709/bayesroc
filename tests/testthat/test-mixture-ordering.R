.sim_mixture <- function(seed = 11, n_sub = 10, n_per = 80) {
  set.seed(seed)
  n <- n_sub * n_per
  d <- data.frame(sub = factor(rep(seq_len(n_sub), each = n_per)))
  d$is_old <- rbinom(n, 1, 0.5)
  d$a <- factor(sample(c("a1", "a2"), n, TRUE))
  d$b <- factor(sample(c("b1", "b2"), n, TRUE))
  d1 <- ifelse(d$a == "a2", 2.2, 1.8)
  d2 <- ifelse(d$b == "b2", 0.8, 0)
  z <- ifelse(d$is_old == 1, ifelse(runif(n) < 0.6, rnorm(n, d1), rnorm(n, d2)), rnorm(n))
  d$y <- as.integer(cut(z, c(-Inf, -0.5, 0.3, 1, 1.7, Inf)))
  d
}

.mixture_model <- function(d) {
  broc(brf(y | is_old ~ a, dprime2 ~ b + (1 | sub), lambda ~ 1, family = mixture()), d)
}

test_that("mixture: dprime and dprime2 keep their own formulas, ordered once on the old items' average", {
  d <- .sim_mixture()
  m <- suppressMessages(.mixture_model(d))
  sd <- m$stan_data
  expect_identical(colnames(sd$X_dprime), c("(Intercept)", "aa2"))
  expect_identical(colnames(sd$X_dprime2), c("(Intercept)", "bb2"))
  old <- d$is_old == 1
  expect_equal(as.numeric(sd$dprime_xbar), unname(colMeans(sd$X_dprime[old, ])))
  expect_equal(as.numeric(sd$dprime2_xbar), unname(colMeans(sd$X_dprime2[old, ])))
  expect_equal(sd$dprime_pivot, 1L)
})

test_that("mixture ordering: JAX keeps the average order in every draw and R reproduces its log-likelihood", {
  skip_if_no_jax()
  d <- .sim_mixture()
  m <- suppressMessages(.mixture_model(d))
  f <- suppressMessages(fit_broc(m, chains = 1, iter_warmup = 50,
                                 iter_sampling = 20, seed = 1, progress_bar = FALSE))
  dr <- f$draws(format = "matrix")
  b1 <- dr[, c("beta_dprime[1]", "beta_dprime[2]"), drop = FALSE]
  b2 <- dr[, c("beta_dprime2[1]", "beta_dprime2[2]"), drop = FALSE]
  gap <- b1 %*% as.numeric(m$stan_data$dprime_xbar) - b2 %*% as.numeric(m$stan_data$dprime2_xbar)
  expect_true(all(gap > 0))
  ll_r <- .obs_loglik_r(m, build_data_source_from_model(m), dr, re_formula = NULL)
  expect_lt(max(abs(ll_r - .read_loglik_SN_cols(.loglik_path(f), 1L, m$model_data$N))), 1e-5)
})
