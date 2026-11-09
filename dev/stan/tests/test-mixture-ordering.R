# Helpers shared with tests/testthat/
# test-mixture-ordering.R (copied).

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
.mixture_model <- function(d, batch = TRUE) {
  .broc_with(batch, brf(y | is_old ~ a, dprime2 ~ b + (1 | sub), lambda ~ 1, family = mixture()), d)
}

test_that("mixture ordering: batch and plain Stan agree", {
  skip_if_no_cmdstan()
  d <- .sim_mixture()
  mp <- suppressMessages(.mixture_model(d, batch = FALSE))
  mb <- suppressMessages(.mixture_model(d, batch = TRUE))
  modp <- .compile_broc_stan(mp)
  modb <- .compile_broc_stan(mb)
  fp <- modb$sample(data = mb$stan_data, chains = 1, iter_warmup = 20, iter_sampling = 5,
                    init = 0.5, sig_figs = 18, refresh = 50, seed = 1, max_treedepth = 6, show_messages = FALSE)
  a <- .cmdstan_log_prob(modp, mp$stan_data, fp$output_files()[1])
  b <- .cmdstan_log_prob(modb, mb$stan_data, fp$output_files()[1])
  expect_lt(max(abs(a[, 1] - b[, 1])), 1e-8)
  expect_lt(max(abs(a[, -1] - b[, -1]) / pmax(1, abs(a[, -1]))), 1e-8)
})

test_that("lure-only mixture: batch and plain Stan agree, and R reproduces the JAX log-likelihood", {
  d <- .sim_mixture()
  f <- brf(y | is_old ~ 1 + (1 | sub), lambda_L ~ 1, dprime_L ~ 1, family = mixture())
  if (!isTRUE(tryCatch(nzchar(cmdstanr::cmdstan_path()), error = function(e) FALSE))) skip("CmdStan not installed.")
  r <- .stan_parity(suppressMessages(.broc_plain(f, d)), suppressMessages(broc(f, d)))
  expect_lt(r$lp, 1e-8)
  expect_lt(r$grad, 1e-8)
  skip_if_no_jax()
  m <- suppressMessages(broc(f, d))
  fj <- suppressMessages(fit_broc(m, backend = "jax", sampler = "drghmc", chains = 1, iter_warmup = 50,
                                  iter_sampling = 5, seed = 1, progress_bar = FALSE))
  ll_r <- .obs_loglik_r(m, build_data_source_from_model(m), fj$draws(format = "matrix"), re_formula = NULL)
  expect_lt(max(abs(ll_r - .read_loglik_SN_cols(.loglik_path(fj), 1L, m$model_data$N))), 1e-5)
})
