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
    expect_false(is.null(mb$batch_cpp))
    modp <- .compile_broc_stan(mp)
    modb <- .compile_broc_stan(mb)
    fp <- modp$sample(data = mp$stan_data, chains = 1, iter_warmup = 100, iter_sampling = 5,
                      init = 0, sig_figs = 18, refresh = 50, seed = 1, show_messages = FALSE)
    a <- .cmdstan_log_prob(modp, mp$stan_data, fp$output_files()[1])
    b <- .cmdstan_log_prob(modb, mb$stan_data, fp$output_files()[1])
    expect_equal(dim(a), dim(b))
    expect_lt(max(abs(a[, 1] - b[, 1])), 1e-8)
    expect_lt(max(abs(a[, -1] - b[, -1]) / pmax(1, abs(a[, -1]))), 1e-8)
  }
})

test_that("varying source criteria: predict() matches log_lik with covariates and smooths", {
  skip_if_no_cmdstan()
  d <- .sim_varying_cov()
  m <- suppressMessages(.varying_model(d, criterion2 ~ 1 + x + s(w, k = 5) + (1 | sub),
                                       bivariate_gaussian(source_criteria = "varying")))
  fit <- fit_broc(m, backend = "stan", refresh = 50, chains = 2, parallel_chains = 2,
                  iter_warmup = 250, iter_sampling = 250, seed = 3)
  model <- attr(fit, "broc_model")
  sd <- model$stan_data
  ll <- as.matrix(fit$draws(variables = "log_lik", format = "matrix"))
  pr <- predict(fit, type = "response", summary = FALSE)
  obs <- vapply(seq_len(ncol(ll)), function(n) pr[, n, sd$y[n], sd$y2[n]], numeric(nrow(ll)))
  expect_lt(max(abs(obs - exp(ll))), 1e-4)
  nd <- d[1:30, ]
  pn <- predict(fit, newdata = nd, type = "response", summary = FALSE)
  expect_lt(max(abs(pn - pr[, 1:30, , , drop = FALSE])), 1e-8)
})

test_that("varying source criteria: JAX recovers a criterion2 covariate and matches Stan", {
  skip_if_no_jax()
  skip_if_no_cmdstan()
  d <- .sim_varying_cov(n_sub = 20, n_per = 120, amp = 0)
  m <- suppressMessages(.varying_model(d, criterion2 ~ 1 + x + (1 | sub),
                                       bivariate_gaussian(source_criteria = "varying")))
  fj <- fit_broc(m, backend = "jax", chains = 2, parallel_chains = 2,
                 iter_warmup = 500, iter_sampling = 500, seed = 4, progress_bar = FALSE)
  fs <- fit_broc(m, backend = "stan", chains = 2, parallel_chains = 2, refresh = 50,
                 iter_warmup = 500, iter_sampling = 500, seed = 4)
  mj <- mean(posterior::as_draws_df(fj$draws())[["beta_thresh_mid_2_cov[1]"]])
  ms <- mean(posterior::as_draws_df(fs$draws())[["beta_thresh_mid_2_cov[1]"]])
  expect_equal(mj, 0.5, tolerance = 0.1)
  expect_equal(mj, ms, tolerance = 0.05)
})

test_that("shared new-response criteria: one set with its own random effects for the new bins, one per old bin", {
  d <- .sim_varying_cov()
  fam <- bivariate_gaussian(source_criteria = "varying_shared_new", old_levels = 3:4)
  m <- suppressMessages(.varying_model(d, criterion2 ~ 1 + (1 | sub), fam))
  sd <- m$stan_data
  expect_equal(sd$n_vbin, 3L)
  expect_equal(as.integer(sd$vbin), c(1L, 1L, 2L, 3L))
  re <- m$model_data$criterion2$random$sub
  expect_equal(re$dim, 3L * (sd$K2 - 1L))
  expect_equal(re$bin_levels, c("new", "3", "4"))
  labs <- .re_dim_labels(m, "criterion2", re)
  expect_true(all(grepl("(bin [34]|new),", labs)))
  expect_equal(sum(grepl("new,", labs)), sd$K2 - 1L)
  full <- suppressMessages(.varying_model(d, criterion2 ~ 1 + (1 | sub),
                                          bivariate_gaussian(source_criteria = "varying")))
  expect_equal(full$stan_data$n_vbin, 4L)
  expect_equal(full$model_data$criterion2$random$sub$dim, 4L * (sd$K2 - 1L))
})

test_that("shared new-response criteria: JAX log_lik matches predict() cell probabilities", {
  skip_if_no_jax()
  d <- .sim_varying_cov(n_sub = 10, n_per = 80)
  fam <- bivariate_gaussian(source_criteria = "varying_shared_new", old_levels = 3:4)
  m <- suppressMessages(.varying_model(d, criterion2 ~ 1 + x + (1 | sub), fam))
  fit <- fit_broc(m, backend = "jax", sampler = "nuts", chains = 2, parallel_chains = 2,
                  iter_warmup = 150, iter_sampling = 50, seed = 5, progress_bar = FALSE, save_loglik = TRUE)
  vars <- posterior::variables(fit$draws())
  expect_equal(sum(startsWith(vars, "beta_thresh_mid_2_varying[")), 3L)
  expect_equal(sum(startsWith(vars, "sigma_criterion2_sub[")), 3L * (m$stan_data$K2 - 1L))
  sd <- attr(fit, "broc_model")$stan_data
  ll <- as.matrix(fit$draws(variables = "log_lik", format = "matrix"))
  pr <- predict(fit, type = "response", summary = FALSE)
  obs <- vapply(seq_len(ncol(ll)), function(n) pr[, n, sd$y[n], sd$y2[n]], numeric(nrow(ll)))
  expect_lt(max(abs(obs - exp(ll))), 1e-5)
  est <- suppressWarnings(estimates(fit, pars = "criterion2"))
  expect_setequal(unique(sub(", c[0-9]+$", "", est$criterion)), c("new", "bin 3", "bin 4"))
})
