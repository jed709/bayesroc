.smooth_data <- function(seed = 41, n = 900) {
  set.seed(seed)
  d <- data.frame(item_type = factor(sample(c("new", "A", "B"), n, TRUE), levels = c("new", "A", "B")),
                  w = round(runif(n, -2, 2), 3), is_old = rbinom(n, 1, 0.5))
  mu1 <- ifelse(d$item_type == "new", 0, 1.1)
  mu2 <- ifelse(d$item_type == "A", -1.0, ifelse(d$item_type == "B", 1.0, 0))
  shift <- 0.5 * sin(d$w)
  d$det <- as.integer(1L + rowSums((rnorm(n, mu1) - shift) > outer(rep(1, n), c(-1.0, -0.3, 0.4, 1.1, 1.8))))
  d$src <- as.integer(1L + rowSums((rnorm(n, mu2) - shift) > outer(rep(1, n), c(-0.9, -0.25, 0.45, 1.2))))
  d$resp <- as.integer(1L + rowSums((rnorm(n, 1.2 * d$is_old) - shift) >
                                    outer(rep(1, n), c(-1.0, -0.3, 0.4, 1.1, 1.8))))
  d
}

test_that("criterion2 smooth: JAX log_lik matches the R reference", {
  skip_if_no_jax()
  d <- .smooth_data()
  m <- suppressMessages(broc(brf(resp(det, src) | item_type ~ 1, discrim ~ 1, rho ~ 1,
                                 criterion ~ 1, criterion2 ~ 1 + s(w, k = 5),
                                 family = bivariate_gaussian()), d))
  f <- suppressMessages(fit_broc(m, backend = "jax", sampler = "nuts", chains = 1, iter_warmup = 15, iter_sampling = 5,
                                 seed = 5, progress_bar = FALSE))
  dm <- posterior::as_draws_matrix(f$draws())
  ll_j <- as.matrix(posterior::as_draws_matrix(f$draws(variables = "log_lik")))
  expect_lt(max(abs(ll_j - .obs_loglik_r(m, build_data_source_from_model(m), dm, NULL))), 1e-5)
})
