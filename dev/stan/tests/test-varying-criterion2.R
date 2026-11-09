# Helpers shared with tests/testthat/
# test-varying-criterion2.R (copied).

.sim_varying_cov <- function(seed = 31, n_sub = 12, n_per = 60, amp = 0.6) {
  set.seed(seed)
  n <- n_sub * n_per
  d <- data.frame(sub = factor(rep(seq_len(n_sub), each = n_per)))
  d$item_type <- factor(sample(c("new", "A", "B"), n, TRUE), levels = c("new", "A", "B"))
  d$x <- round(rnorm(n), 3)
  d$w <- round(runif(n, -2, 2), 3)
  u <- rnorm(n_sub, 0, 0.5)[d$sub]
  mu1 <- ifelse(d$item_type == "new", 0, 1.2)
  mu2 <- ifelse(d$item_type == "A", -1.0, ifelse(d$item_type == "B", 1.0, 0))
  z1 <- rnorm(n, mu1); z2 <- rnorm(n, mu2)
  d$det <- as.integer(1L + rowSums(outer(z1, c(-0.8, 0, 0.8), ">")))
  base <- rbind(c(-1.2, -0.3, 0.6), c(-1.0, -0.1, 0.8), c(-0.8, 0.1, 1.0), c(-0.6, 0.3, 1.2))
  c2 <- base[d$det, ] + 0.5 * d$x + amp * sin(d$w) + u
  d$src <- as.integer(1L + rowSums(z2 > c2))
  d
}
.varying_model <- function(d, f2, fam, batch = TRUE) {
  .broc_with(batch, brf(resp(det, src) | item_type ~ 1, discrim ~ 1, rho ~ 1, criterion ~ 1, f2,
           family = fam), d)
}
.sim_varying_cells <- function(...) {
  d <- .sim_varying_cov(...)
  d$g <- factor(ifelse(as.integer(d$sub) %% 2 == 0, "E1", "E2"))
  d
}

test_that("varying source criteria: batch and plain Stan agree with covariates, smooths and REs", {
  skip_if_no_cmdstan()
  d <- .sim_varying_cov()
  cases <- list(
    list(criterion2 ~ 1 + x + s(w, k = 5) + (1 | sub),
         bivariate_gaussian(source_criteria = "varying")),
    list(criterion2 ~ 1 + x + (1 | sub),
         bivariate_gaussian(source_criteria = "varying_shared_new", old_levels = 3:4))
  )
  for (cs in cases) {
    mp <- suppressMessages(.varying_model(d, cs[[1]], cs[[2]], batch = FALSE))
    mb <- suppressMessages(.varying_model(d, cs[[1]], cs[[2]], batch = TRUE))
    expect_false(is.null(.with_stan_code(mb)$batch_cpp))
    r <- .stan_parity(mp, mb)
    expect_equal(r$dim[1], r$dim[2])
    expect_lt(r$lp, 1e-8)
    expect_lt(r$grad, 1e-8)
  }
})

test_that("varying source criteria: JAX recovers a criterion2 covariate and matches Stan", {
  skip_if_not_slow()
  skip_if_no_jax()
  skip_if_no_cmdstan()
  d <- .sim_varying_cov(n_sub = 20, n_per = 120, amp = 0)
  m <- suppressMessages(.varying_model(d, criterion2 ~ 1 + x + (1 | sub),
                                       bivariate_gaussian(source_criteria = "varying")))
  fj <- fit_broc(m, backend = "jax", sampler = "nuts", chains = 2, parallel_chains = 2,
                 iter_warmup = 500, iter_sampling = 500, seed = 4, progress_bar = FALSE)
  fs <- stan_fit_broc(m, chains = 2, parallel_chains = 2, refresh = 50,
                 iter_warmup = 500, iter_sampling = 500, seed = 4)
  mj <- mean(posterior::as_draws_df(fj$draws())[["beta_thresh_mid_2_cov[1]"]])
  ms <- mean(posterior::as_draws_df(fs$draws())[["beta_thresh_mid_2_cov[1]"]])
  expect_equal(mj, 0.5, tolerance = 0.1)
  expect_equal(mj, ms, tolerance = 0.05)
})

test_that("varying source criteria with cell means: batch and plain Stan agree", {
  skip_if_no_cmdstan()
  d <- .sim_varying_cells()
  cases <- list(
    list(criterion2 ~ 0 + g, bivariate_gaussian(source_criteria = "varying")),
    list(criterion2 ~ 0 + g + x + (1 | sub), bivariate_gaussian(source_criteria = "varying")),
    list(criterion2 ~ 0 + g + (1 | sub), bivariate_gaussian(source_criteria = "varying_shared_new", old_levels = 3:4))
  )
  for (cs in cases) {
    mp <- suppressMessages(.varying_model(d, cs[[1]], cs[[2]], batch = FALSE))
    mb <- suppressMessages(.varying_model(d, cs[[1]], cs[[2]], batch = TRUE))
    r <- .stan_parity(mp, mb)
    expect_equal(r$dim[1], r$dim[2])
    expect_lt(r$lp, 1e-8)
    expect_lt(r$grad, 1e-8)
  }
})

test_that("varying source criteria with cell means: predict() matches the Stan log_lik", {
  skip_if_no_cmdstan()
  d <- .sim_varying_cells()
  m <- suppressMessages(.varying_model(d, criterion2 ~ 0 + g + x + (1 | sub), bivariate_gaussian(source_criteria = "varying")))
  fit <- stan_fit_broc(m, refresh = 50, chains = 1, iter_warmup = 15, iter_sampling = 5, seed = 3)
  sd <- attr(fit, "broc_model")$stan_data
  ll <- as.matrix(fit$draws(variables = "log_lik", format = "matrix"))
  pr <- predict(fit, type = "response", summary = FALSE)
  obs <- vapply(seq_len(ncol(ll)), function(n) pr[, n, sd$y[n], sd$y2[n]], numeric(nrow(ll)))
  expect_lt(max(abs(obs - exp(ll))), 1e-4)
})
