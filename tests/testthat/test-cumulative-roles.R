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

test_that("predictors go on the cutpoints, as per-cutpoint effects or one shift with cutpoints_scale", {
  d <- .sim_ordinal()
  expect_error(broc(brf(rating ~ grp + (1 | rater), family = cumulative()), d), "cutpoints ~ grp")
  m <- broc(brf(rating ~ 1, cutpoints ~ grp + (1 | rater) + (1 | item), cutpoints_scale ~ 1, family = cumulative()), d)
  md <- m$model_data
  expect_equal(md$stan_data$P_dprime, 0)
  expect_length(md$dprime_random, 0)
  expect_true(md$criterion$location_scale)
  expect_equal(unname(vapply(md$criterion$random, function(r) as.integer(r$dim), 1L)), c(1L, 1L))
  expect_equal(md$criterion_scale_fixed$n_coef, 0)
  m2 <- broc(brf(rating ~ 1, cutpoints ~ grp + (1 | rater), family = cumulative()), d)
  expect_null(m2$model_data$criterion$location_scale)
  expect_equal(m2$model_data$criterion$random$rater$dim, 4)
  d1 <- d; d1$one <- factor("x")
  expect_error(broc(brf(rating ~ 1, cutpoints ~ one + (1 | rater), family = cumulative()), d1),
               "Variable 'one' has a single level")
})

test_that("the random-intercept ordinal model fits on both backends and recovers its structure", {
  skip_if_no_jax()
  skip_if_no_cmdstan()
  d <- .sim_ordinal()
  m <- broc(brf(rating ~ 1, cutpoints ~ grp + (1 | rater) + (1 | item), cutpoints_scale ~ 1, family = cumulative()), d)
  fj <- fit_broc(m, backend = "jax", sampler = "drghmc", chains = 2, parallel_chains = 2, iter_warmup = 500,
                 iter_sampling = 400, seed = 2, progress_bar = FALSE)
  fs <- fit_broc(m, backend = "stan", chains = 2, parallel_chains = 2, iter_warmup = 400, iter_sampling = 400,
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

test_that("bivariate_cumulative uses cutpoints names, accepts the old spelling and refuses mu2 predictors", {
  d <- .fixture_bivariate_cumulative()
  m_new <- broc(brf(resp(y1, y2) ~ 1, cutpoints ~ 1, cutpoints2 ~ 1, rho ~ 1, family = bivariate_cumulative()), d)
  m_old <- broc(brf(resp(y1, y2) ~ 1, cutpoints1 ~ 1, cutpoints2 ~ 1, rho ~ 1, family = bivariate_cumulative()), d)
  expect_identical(m_new$stan_code, m_old$stan_code)
  expect_true(any(grepl("cutpoints", names(m_new$stan_data))) && !any(grepl("cutpoints1", names(m_new$stan_data))))
  expect_equal(.param_name_map("bivariate_cumulative")$criterion, "cutpoints")
  expect_error(broc(brf(resp(y1, y2) ~ 1, mu2 ~ cond, family = bivariate_cumulative()), d), "cutpoints2 ~ cond")
  expect_error(broc(brf(resp(y1, y2) ~ cond, family = bivariate_cumulative()), d), "cutpoints ~ cond")
})
