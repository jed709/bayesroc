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

test_that("univariate criterion smooth: batch and plain Stan agree in value and gradient", {
  skip_if_no_cmdstan()
  d <- .smooth_data()
  mk <- function(b) suppressMessages(.broc_with(b, brf(resp | is_old ~ 1, criterion ~ 1 + s(w, k = 5),
                                              family = evsd()), d))
  mb <- mk(TRUE)
  expect_false(is.null(mb$batch_cpp))
  r <- .stan_parity(mk(FALSE), mb)
  expect_equal(r$dim[1], r$dim[2])
  expect_lt(r$lp, 1e-8)
  expect_lt(r$grad, 1e-8)
})

test_that("criterion2 smooth: batch and plain Stan agree, and predict() matches log_lik", {
  skip_if_no_cmdstan()
  d <- .smooth_data()
  mk <- function(b) suppressMessages(.broc_with(b, brf(resp(det, src) | item_type ~ 1, discrim ~ 1, rho ~ 1,
                                              criterion ~ 1, criterion2 ~ 1 + s(w, k = 5),
                                              family = bivariate_gaussian()), d))
  mb <- mk(TRUE)
  r <- .stan_parity(mk(FALSE), mb)
  expect_lt(r$lp, 1e-8)
  expect_lt(r$grad, 1e-8)

  fit <- fit_broc(mb, backend = "stan", refresh = 50, chains = 2, parallel_chains = 2,
                  iter_warmup = 250, iter_sampling = 250, seed = 3)
  sd <- mb$stan_data
  ll <- as.matrix(fit$draws(variables = "log_lik", format = "matrix"))
  pr <- predict(fit, type = "response", summary = FALSE)
  obs <- vapply(seq_len(ncol(ll)), function(n) pr[, n, sd$y[n], sd$y2[n]], numeric(nrow(ll)))
  expect_lt(max(abs(obs - exp(ll))), 1e-4)
})

test_that("criterion2 smooth: JAX and Stan agree", {
  skip_if_no_cmdstan()
  skip_if_no_jax()
  d <- .smooth_data(seed = 42, n = 1500)
  m <- suppressMessages(broc(brf(resp(det, src) | item_type ~ 1, discrim ~ 1, rho ~ 1,
                                 criterion ~ 1, criterion2 ~ 1 + s(w, k = 5),
                                 family = bivariate_gaussian()), d))
  fj <- fit_broc(m, backend = "jax", chains = 2, parallel_chains = 2, iter_warmup = 500,
                 iter_sampling = 500, seed = 5, progress_bar = FALSE)
  fs <- fit_broc(m, backend = "stan", chains = 2, parallel_chains = 2, refresh = 50,
                 iter_warmup = 500, iter_sampling = 500, seed = 5)
  v <- c("beta_discrim[1]", "beta_thresh_mid_2[1]", "beta_thresh_mid_2[2]")
  sj <- posterior::summarise_draws(posterior::subset_draws(fj$draws(), variable = v), "mean", "sd")
  ss <- posterior::summarise_draws(posterior::subset_draws(fs$draws(), variable = v), "mean", "sd")
  z <- abs(sj$mean - ss$mean) / sqrt((sj$sd^2 + ss$sd^2) / 2)
  expect_true(all(z < 0.3))
})

test_that("cyclic criterion smooth compiles and batch matches plain Stan", {
  skip_if_no_cmdstan()
  d <- .smooth_data()
  mk <- function(b) suppressMessages(.broc_with(b, brf(resp | is_old ~ 1,
                                              criterion ~ 1 + s(w, bs = "cc", k = 6),
                                              family = evsd()), d))
  mb <- mk(TRUE)
  expect_true(grepl("s_criterion", mb$stan_code))
  r <- .stan_parity(mk(FALSE), mb)
  expect_lt(r$lp, 1e-8)
  expect_lt(r$grad, 1e-8)
})
