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

test_that("mu keeps its random effects, cutpoints defaults, 0 + f is rewritten, overlap is refused", {
  d <- .sim_ordinal()
  m <- broc(brf(rating ~ grp + (1 | rater) + (1 | item), family = cumulative()), d)
  re <- m$model_data$dprime_random
  expect_setequal(names(re), c("rater", "item"))
  expect_equal(unname(m$model_data$dprime_fixed$coef_names), "grpb")
  cfg <- build_numpyro_config(m)
  expect_setequal(vapply(cfg$random_effects, function(r) paste(r$param, r$group), character(1)),
                  c("dprime rater", "dprime item"))
  expect_message(m2 <- broc(brf(rating ~ 0 + grp + (1 | rater), cutpoints ~ 1, family = cumulative()), d), "cell-means")
  expect_equal(unname(m2$model_data$dprime_fixed$coef_names), "grpb")
  expect_error(broc(brf(rating ~ grp, cutpoints ~ grp, family = cumulative()), d), "predictor\\(s\\) grp")
  expect_error(broc(brf(rating ~ 1 + (1 | rater), cutpoints ~ 1 + (1 | rater), family = cumulative()), d),
               "grouping factor\\(s\\) rater")
  m3 <- broc(brf(rating ~ 1 + (1 | item), cutpoints ~ grp + (1 | rater), family = cumulative()), d)
  expect_equal(names(m3$model_data$dprime_random), "item")
  expect_true(m3$model_data$criterion$n_coef > 1L)
  d1 <- d; d1$one <- factor("x")
  expect_error(broc(brf(rating ~ one + (1 | rater), family = cumulative()), d1), "Variable 'one' has a single level")
})

test_that("the random-intercept ordinal model fits on both backends and recovers its structure", {
  skip_if_no_jax()
  skip_if_no_cmdstan()
  d <- .sim_ordinal()
  m <- broc(brf(rating ~ grp + (1 | rater) + (1 | item), family = cumulative()), d)
  fj <- fit_broc(m, backend = "jax", sampler = "drghmc", chains = 2, parallel_chains = 2, iter_warmup = 500,
                 iter_sampling = 400, seed = 2, progress_bar = FALSE)
  fs <- fit_broc(m, backend = "stan", chains = 2, parallel_chains = 2, iter_warmup = 400, iter_sampling = 400,
                 refresh = 0, seed = 2)
  vj <- posterior::variables(fj$.draws_array)
  expect_true(any(grepl("rater", vj)))
  vars <- c("beta_mu[1]", "beta_thresh_mid[1]")
  sj <- posterior::summarise_draws(fj$draws(variables = vars, format = "draws_array"), "mean", "sd", "ess_bulk")
  ss <- posterior::summarise_draws(fs$draws(variables = vars, format = "draws_array"), "mean", "sd", "ess_bulk")
  z <- abs(sj$mean - ss$mean) / sqrt(sj$sd^2 / sj$ess_bulk + ss$sd^2 / ss$ess_bulk)
  expect_true(all(z < 4), info = paste(sprintf("%s z=%.1f", vars, z), collapse = "; "))
  # the cutpoints' location and the mu random intercepts share a direction: the map couples them
  expect_gt(min(sj$ess_bulk), 150)
  expect_gt(sj$mean[1], 0.3)
  expect_s3_class(summary(fj), "broc_summary")
})

test_that("bivariate_cumulative uses mu / cutpoints names and accepts the old spelling", {
  d <- .fixture_bivariate_cumulative()
  m_new <- broc(brf(resp(y1, y2) ~ 1, mu2 ~ 1, cutpoints ~ 1, cutpoints2 ~ 1, rho ~ 1, family = bivariate_cumulative()), d)
  m_old <- broc(brf(resp(y1, y2) ~ 1, mu2 ~ 1, cutpoints1 ~ 1, cutpoints2 ~ 1, rho ~ 1, family = bivariate_cumulative()), d)
  expect_identical(m_new$stan_code, m_old$stan_code)
  expect_true(any(grepl("cutpoints", names(m_new$stan_data))) && !any(grepl("cutpoints1", names(m_new$stan_data))))
  expect_equal(.param_name_map("bivariate_cumulative")$criterion, "cutpoints")
})
