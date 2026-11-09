# Helpers shared with tests/testthat/
# test-cumulative-roles.R (copied).

.sim_ordinal <- function(seed = 1, R = 30, I = 40, K = 5) {
  set.seed(seed)
  g <- expand.grid(rater = seq_len(R), item = seq_len(I))
  u_r <- rnorm(R, 0, 0.6); u_i <- rnorm(I, 0, 0.4)
  grp <- rep(c("a", "b"), length.out = R)
  eta <- u_r[g$rater] + u_i[g$item] + 0.7 * (grp[g$rater] == "b")
  tau <- c(-1.2, -0.3, 0.4, 1.3)
  y <- 1L + rowSums(outer(eta + rnorm(nrow(g)), tau, ">"))
  data.frame(rating = as.integer(y), rater = factor(g$rater), item = factor(g$item), grp = factor(grp[g$rater]))
}

test_that("the random-intercept ordinal model: JAX and batch Stan log-likelihoods match R", {
  skip_if_no_jax()
  skip_if_no_cmdstan()
  d <- .sim_ordinal()
  m <- broc(brf(rating ~ 1, cutpoints_shift ~ grp + (1 | rater) + (1 | item), family = cumulative()), d)
  fj <- suppressMessages(fit_broc(m, backend = "jax", sampler = "drghmc", chains = 1, iter_warmup = 50,
                                  iter_sampling = 5, seed = 2, progress_bar = FALSE))
  ll_r <- .obs_loglik_r(m, build_data_source_from_model(m), fj$draws(format = "matrix"), re_formula = NULL)
  expect_lt(max(abs(ll_r - .read_loglik_SN_cols(.loglik_path(fj), 1L, m$model_data$N))), 1e-5, label = "JAX vs R log_lik")
  fs <- suppressMessages(stan_fit_broc(m, chains = 1, iter_warmup = 100, iter_sampling = 5,
                                  seed = 2, refresh = 50, show_messages = FALSE))
  dm <- posterior::as_draws_matrix(fs$draws())
  ll_s <- as.matrix(dm[, grep("^log_lik\\[", colnames(dm)), drop = FALSE])
  if (!ncol(ll_s)) ll_s <- .read_loglik_SN_cols(.loglik_path(fs), 1L, m$model_data$N)
  ll_r <- .obs_loglik_r(m, build_data_source_from_model(m), dm, re_formula = NULL)
  expect_lt(max(abs(ll_s - ll_r)), 1e-5, label = "Stan vs R log_lik")
})

test_that("the random-intercept ordinal model fits on both backends and recovers its structure", {
  skip_if_not_slow()
  skip_if_no_jax()
  skip_if_no_cmdstan()
  d <- .sim_ordinal()
  m <- broc(brf(rating ~ 1, cutpoints_shift ~ grp + (1 | rater) + (1 | item), family = cumulative()), d)
  fj <- fit_broc(m, backend = "jax", sampler = "drghmc", chains = 2, parallel_chains = 2, iter_warmup = 500,
                 iter_sampling = 400, seed = 2, progress_bar = FALSE)
  fs <- stan_fit_broc(m, chains = 2, parallel_chains = 2, iter_warmup = 400, iter_sampling = 400,
                 refresh = 0, seed = 2)
  vars <- c("beta_thresh_mid[1]", "beta_thresh_mid[2]")
  sj <- posterior::summarise_draws(fj$draws(variables = vars, format = "draws_array"), "mean", "sd", "ess_bulk")
  ss <- posterior::summarise_draws(fs$draws(variables = vars, format = "draws_array"), "mean", "sd", "ess_bulk")
  z <- abs(sj$mean - ss$mean) / sqrt(sj$sd^2 / sj$ess_bulk + ss$sd^2 / ss$ess_bulk)
  expect_true(all(z < 4), info = paste(sprintf("%s z=%.1f", vars, z), collapse = "; "))
  expect_gt(min(sj$ess_bulk), 150)
  expect_lt(sj$mean[2], -0.3)
  expect_s3_class(summary(fj), "broc_summary")
})
