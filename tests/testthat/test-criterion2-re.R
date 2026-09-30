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

test_that("JAX applies every criterion2 random-effect grouping factor", {
  skip_if_no_jax()
  d <- .sim_crit2_crossed()
  m <- broc(brf(resp(y1, y2) | item_type ~ 1, discrim ~ 1, rho ~ 1, criterion ~ 1,
                criterion2 ~ 1 + (1 | sub) + (1 | item),
                family = bivariate_gaussian()), d)
  f <- fit_broc(m, backend = "jax", chains = 2, parallel_chains = 2,
                iter_warmup = 500, iter_sampling = 500, seed = 3, progress_bar = FALSE)
  expect_gt(.mid_sd(f, "sub"), 0.8)
  expect_gt(.mid_sd(f, "item"), 0.8)
  vars <- posterior::variables(f$.draws_array)
  expect_true(any(grepl("^u_criterion2_sub\\[", vars)))
  expect_true(any(grepl("^u_criterion2_item\\[", vars)))
})

test_that("JAX and Stan agree on crossed criterion2 random effects", {
  skip_if_no_jax()
  skip_if_no_cmdstan()
  d <- .sim_crit2_crossed(seed = 22)
  m <- broc(brf(resp(y1, y2) | item_type ~ 1, discrim ~ 1, rho ~ 1, criterion ~ 1,
                criterion2 ~ 1 + (1 | sub) + (1 | item),
                family = bivariate_gaussian()), d)
  fj <- fit_broc(m, backend = "jax", chains = 2, parallel_chains = 2,
                 iter_warmup = 500, iter_sampling = 500, seed = 4, progress_bar = FALSE)
  fs <- fit_broc(m, backend = "stan", chains = 2, parallel_chains = 2, refresh = 50,
                 iter_warmup = 500, iter_sampling = 500, seed = 4)
  for (g in c("sub", "item")) {
    expect_equal(.mid_sd(fj, g), .mid_sd(fs, g), tolerance = 0.2)
  }
})

test_that("cross-correlated criterion2 effects move source, not detection, thresholds", {
  skip_if_no_jax()
  d <- .sim_crit2_crossed(seed = 23, n_item = 30, sd_item = 0)
  m <- broc(brf(resp(y1, y2) | item_type ~ 1 + (1 | p | sub), discrim ~ 1, rho ~ 1,
                criterion ~ 1, criterion2 ~ 1 + (1 | p | sub),
                family = bivariate_gaussian()), d)
  f <- fit_broc(m, backend = "jax", chains = 2, parallel_chains = 2,
                iter_warmup = 500, iter_sampling = 500, seed = 5, progress_bar = FALSE)
  dr <- posterior::as_draws_df(f$.draws_array)
  u2 <- grep("^u_criterion2_sub_from_p\\[", names(dr), value = TRUE)
  expect_gt(length(u2), 0)
  expect_equal(mean(dr[["beta_thresh_mid[1]"]]), 0, tolerance = 0.25)
})
