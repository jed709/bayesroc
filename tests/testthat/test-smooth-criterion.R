fixture_criterion_smooth <- function(seed = 7, n_sub = 12, n_item = 40) {
  set.seed(seed)
  d <- expand.grid(item = seq_len(n_item), sub = seq_len(n_sub))
  d$old <- as.integer((d$item + d$sub) %% 2 == 0)
  d$x <- round(runif(nrow(d), -2, 2), 3)
  tau <- c(-1.0, -0.35, 0.3, 0.95, 1.7)
  n <- nrow(d)
  lat <- ifelse(d$old == 1, rnorm(n, 1.0, 1), rnorm(n))
  thr <- outer(rep(1, n), tau) + 0.45 * sin(1.6 * d$x)
  d$conf <- as.integer(1L + rowSums(lat > thr))
  d$sub <- factor(d$sub)
  d
}

pop_means <- function(f) {
  s <- posterior::summarise_draws(f$draws(format = "array"), "mean")
  s <- s[grep("^beta_|^sds_", s$variable), ]
  nm <- sub("[[]([0-9]+)[]]$", "[\\1,1]", s$variable)
  stats::setNames(s$mean, nm)
}

test_that("criterion smooth keeps its unpenalized null-space columns", {
  d <- fixture_criterion_smooth()
  m <- broc(brf(conf | old ~ 1, criterion ~ s(x), family = evsd()), data = d)
  expect_equal(m$stan_data$P_criterion, 2L)
  expect_true(any(grepl("^sXs", colnames(m$stan_data$X_criterion))))
  expect_false(m$model_data$criterion$is_intercept_only)

  m_cc <- broc(brf(conf | old ~ 1, criterion ~ s(x, bs = "cc"), family = evsd()), data = d)
  expect_equal(m_cc$stan_data$P_criterion, 1L)

  m_none <- broc(brf(conf | old ~ 1, criterion ~ 1, family = evsd()), data = d)
  expect_equal(m_none$stan_data$P_criterion, 1L)
  expect_true(m_none$model_data$criterion$is_intercept_only)
})

test_that("criterion smooth: JAX log_lik matches the R reference", {
  skip_if_no_jax()
  d <- fixture_criterion_smooth()
  m <- broc(brf(conf | old ~ 1, criterion ~ s(x), family = evsd()), data = d)
  f <- suppressMessages(fit_broc(m, backend = "jax", sampler = "nuts", chains = 1, iter_warmup = 15, iter_sampling = 5,
                                 seed = 1, progress_bar = FALSE))
  dm <- posterior::as_draws_matrix(f$draws())
  ll_j <- as.matrix(posterior::as_draws_matrix(f$draws(variables = "log_lik")))
  expect_lt(max(abs(ll_j - .obs_loglik_r(m, build_data_source_from_model(m), dm, NULL))), 1e-5)
})
