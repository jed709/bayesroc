# Helpers shared with tests/testthat/
# test-criterion2-re.R (copied).

.sim_crit2_crossed <- function(seed = 21, n_sub = 30, n_item = 40, sd_sub = 1.2,
                               sd_item = 1.2) {
  set.seed(seed)
  d <- expand.grid(sub = seq_len(n_sub), item = seq_len(n_item))
  d$item_type <- factor(c("new", "A", "B")[(d$item %% 3) + 1], levels = c("new", "A", "B"))
  u_sub <- rnorm(n_sub, 0, sd_sub)
  u_item <- rnorm(n_item, 0, sd_item)
  mu1 <- ifelse(d$item_type == "new", 0, 1.0)
  mu2 <- ifelse(d$item_type == "A", 0.8, ifelse(d$item_type == "B", -0.8, 0))
  z1 <- rnorm(nrow(d), mu1, 1)
  z2 <- rnorm(nrow(d), mu2, 1)
  shift2 <- u_sub[d$sub] + u_item[d$item]
  d$y1 <- as.integer(1L + rowSums(outer(z1, c(-1, 0, 1), ">")))
  d$y2 <- as.integer(1L + rowSums((z2 - shift2) > outer(rep(1, nrow(d)), c(-0.6, 0.2, 1.0))))
  d$sub <- factor(d$sub); d$item <- factor(d$item)
  d
}
.mid_sd <- function(f, g) {
  dr <- posterior::as_draws_df(f$draws())
  mean(dr[[sprintf("sigma_criterion2_%s[2]", g)]])
}
.ll_vs_r <- function(m, f) {
  ll_r <- .obs_loglik_r(m, build_data_source_from_model(m), f$draws(format = "matrix"), re_formula = NULL)
  max(abs(ll_r - .read_loglik_SN_cols(.loglik_path(f), 1L, m$model_data$N)))
}

test_that("batch Stan applies every criterion2 random-effect grouping factor in the likelihood", {
  skip_if_no_cmdstan()
  d <- .sim_crit2_crossed(seed = 22)
  m <- broc(brf(resp(y1, y2) | item_type ~ 1, discrim ~ 1, rho ~ 1, criterion ~ 1,
                criterion2 ~ 1 + (1 | sub) + (1 | item),
                family = bivariate_gaussian()), d)
  f <- suppressMessages(stan_fit_broc(m, chains = 1, iter_warmup = 100, iter_sampling = 5,
                                 seed = 4, refresh = 50, show_messages = FALSE))
  vars <- posterior::variables(f$draws())
  expect_true(any(grepl("^u_criterion2_sub\\[", vars)))
  expect_true(any(grepl("^u_criterion2_item\\[", vars)))
  expect_lt(.ll_vs_r(m, f), 1e-5, label = "Stan vs R log_lik")
})

test_that("JAX and Stan agree on crossed criterion2 random effects", {
  skip_if_not_slow()
  skip_if_no_jax()
  skip_if_no_cmdstan()
  d <- .sim_crit2_crossed(seed = 22)
  m <- broc(brf(resp(y1, y2) | item_type ~ 1, discrim ~ 1, rho ~ 1, criterion ~ 1,
                criterion2 ~ 1 + (1 | sub) + (1 | item),
                family = bivariate_gaussian()), d)
  fj <- fit_broc(m, backend = "jax", sampler = "nuts", chains = 2, parallel_chains = 2,
                 iter_warmup = 500, iter_sampling = 500, seed = 4, progress_bar = FALSE)
  fs <- stan_fit_broc(m, chains = 2, parallel_chains = 2, refresh = 50,
                 iter_warmup = 500, iter_sampling = 500, seed = 4)
  for (g in c("sub", "item")) {
    expect_equal(.mid_sd(fj, g), .mid_sd(fs, g), tolerance = 0.2)
  }
})
