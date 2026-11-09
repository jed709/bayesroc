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

test_that("predictors go on the cutpoints, as per-cutpoint effects or one shift with cutpoints_shift", {
  d <- .sim_ordinal()
  expect_error(broc(brf(rating ~ grp + (1 | rater), family = cumulative()), d), "cutpoints ~ grp")
  m <- broc(brf(rating ~ 1, cutpoints_shift ~ grp + (1 | rater) + (1 | item), family = cumulative()), d)
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

test_that("bivariate_cumulative uses cutpoints names and refuses mu2 predictors", {
  d <- .fixture_bivariate_cumulative()
  m_new <- broc(brf(resp(y1, y2) ~ 1, cutpoints ~ 1, cutpoints2 ~ 1, rho ~ 1, family = bivariate_cumulative()), d)
  expect_true(any(grepl("cutpoints", names(m_new$stan_data))))
  expect_equal(.param_name_map("bivariate_cumulative")$criterion, "cutpoints")
  expect_error(broc(brf(resp(y1, y2) ~ 1, mu2 ~ cond, family = bivariate_cumulative()), d), "cutpoints2 ~ cond")
  expect_error(broc(brf(resp(y1, y2) ~ cond, family = bivariate_cumulative()), d), "cutpoints ~ cond")
})
