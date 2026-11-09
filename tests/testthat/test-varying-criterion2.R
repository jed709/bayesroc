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

.varying_model <- function(d, f2, fam) {
  broc(brf(resp(det, src) | item_type ~ 1, discrim ~ 1, rho ~ 1, criterion ~ 1, f2,
           family = fam), d)
}

test_that("varying source criteria: predict() matches log_lik with covariates and smooths", {
  skip_if_no_jax()
  d <- .sim_varying_cov()
  m <- suppressMessages(.varying_model(d, criterion2 ~ 1 + x + s(w, k = 5) + (1 | sub),
                                       bivariate_gaussian(source_criteria = "varying")))
  fit <- fit_broc(m, chains = 1,
                  iter_warmup = 15, iter_sampling = 5, seed = 3)
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
  fit <- fit_broc(m, backend = "jax", sampler = "nuts", chains = 1,
                  iter_warmup = 15, iter_sampling = 5, seed = 5, progress_bar = FALSE, save_loglik = TRUE)
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

.sim_varying_cells <- function(...) {
  d <- .sim_varying_cov(...)
  d$g <- factor(ifelse(as.integer(d$sub) %% 2 == 0, "E1", "E2"))
  d
}

test_that("varying source criteria: a cell-means criterion2 formula gives each cell its own sets, not one set plus a shift", {
  d <- .sim_varying_cells()
  cm <- suppressMessages(.varying_model(d, criterion2 ~ 0 + g + x + (1 | sub), bivariate_gaussian(source_criteria = "varying")))
  sd <- cm$stan_data
  expect_equal(c(sd$n_vbin, sd$n_vcell, sd$n_vset), c(4L, 2L, 8L))
  expect_identical(cm$model_data$criterion2$coef_names, c("gE1", "gE2", "x"))
  expect_identical(cm$model_data$criterion2$cov_names, "x")
  expect_equal(as.integer(sd$vcell), as.integer(d$g))
  expect_identical(.vset_labels(cm$model_data)[c(1, 5, 8)], c("1:gE1", "1:gE2", "4:gE2"))
  expect_length(build_numpyro_config(cm)$prior_vary2$mid, 8L)
  tr <- suppressMessages(.varying_model(d, criterion2 ~ 1 + g + x, bivariate_gaussian(source_criteria = "varying")))
  expect_equal(c(tr$stan_data$n_vcell, tr$stan_data$n_vset), c(1L, 4L))
  expect_identical(tr$model_data$criterion2$cov_names, c("gE2", "x"))
  sn <- suppressMessages(.varying_model(d, criterion2 ~ 0 + g,
                                        bivariate_gaussian(source_criteria = "varying_shared_new", old_levels = 3:4)))
  expect_equal(c(sn$stan_data$n_vbin, sn$stan_data$n_vset), c(3L, 6L))
})

test_that("varying source criteria with cell means: R reproduces the JAX log-likelihood", {
  skip_if_no_jax()
  d <- .sim_varying_cells()
  cases <- list(
    list(criterion2 ~ 0 + g + x + (1 | sub), bivariate_gaussian(source_criteria = "varying")),
    list(criterion2 ~ 0 + g, bivariate_gaussian(source_criteria = "varying_shared_new", old_levels = 3:4)),
    list(criterion2 ~ 0 + g + s(x, k = 5), bivariate_gaussian(source_criteria = "varying"))
  )
  for (cs in cases) {
    m <- suppressMessages(.varying_model(d, cs[[1]], cs[[2]]))
    f <- suppressMessages(fit_broc(m, backend = "jax", sampler = "drghmc", chains = 1, iter_warmup = 50,
                                   iter_sampling = 5, seed = 1, progress_bar = FALSE))
    expect_equal(sum(startsWith(posterior::variables(f$draws()), "beta_thresh_mid_2_varying[")), m$stan_data$n_vset)
    ll_r <- .obs_loglik_r(m, build_data_source_from_model(m), f$draws(format = "matrix"), re_formula = NULL)
    expect_lt(max(abs(ll_r - .read_loglik_SN_cols(.loglik_path(f), 1L, m$model_data$N))), 1e-5)
  }
})

test_that("varying source criteria: a criterion2 smooth gets its own curve in each set", {
  d <- .sim_varying_cells()
  m <- suppressMessages(.varying_model(d, criterion2 ~ 1 + s(w, k = 5), bivariate_gaussian(source_criteria = "varying")))
  sm <- m$model_data$smooth_data$criterion2
  expect_length(sm, 1L)
  expect_identical(sm[[1]]$by, ".c2bin")
  expect_length(sm[[1]]$components, m$stan_data$n_vbin)
  expect_error(suppressMessages(.varying_model(d, criterion2 ~ 1 + s(w, by = g, k = 5),
                                               bivariate_gaussian(source_criteria = "varying"))), "by=")
})
