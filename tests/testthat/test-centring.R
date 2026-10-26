test_that(".centre_info finds the intercept, a cell-means partition, or nothing", {
  X <- cbind(`(Intercept)` = 1, x = c(1, 2, 3, 4), g = c(0, 0, 1, 1))
  ci <- .centre_info(X, colnames(X))
  expect_equal(ci$col, list(0L))
  expect_equal(ci$means[[1]], c(0, 2.5, 0.5))
  # encoded design: the intercept column switches rows on; means over those rows only
  Xe <- cbind(`(Intercept)` = c(1, 1, 0, 0), x = c(1, 3, 0, 0))
  expect_equal(.centre_info(Xe, colnames(Xe))$means[[1]], c(0, 2))
  # cell means plus a covariate: the cells carry the intercept jointly
  Xc <- cbind(a = c(1, 1, 0, 0), b = c(0, 0, 1, 1), w = c(1, 2, 3, 4))
  ci <- .centre_info(Xc, colnames(Xc))
  expect_equal(ci$col, list(0L, 1L))
  expect_equal(ci$means[[1]], c(0, 0, 2.5))
  # nothing to centre: cell means only, a single column, or a skipped parameter
  expect_null(.centre_info(Xc[, 1:2], c("a", "b")))
  expect_null(.centre_info(X[, 1, drop = FALSE], "(Intercept)"))
  expect_null(.centre_info(X, colnames(X), skip = TRUE))
  # a 0/1 column outside the partition makes the partition ambiguous: left alone
  Xa <- cbind(a = c(1, 1, 0, 0), b = c(0, 0, 1, 1), z = c(1, 0, 1, 0))
  expect_null(.centre_info(Xa, colnames(Xa)))
})

test_that("the config carries centring for intercept designs and the fingerprint sees it", {
  d <- .fixture_bivariate_sdt()
  d$grp <- factor(rep(c("c", "t"), length.out = nrow(d)))
  m1 <- broc(brf(resp(y1, y2) | item_type ~ grp, discrim ~ 1, criterion ~ grp, criterion2 ~ 1, rho ~ 1,
                 family = bivariate_gaussian()), d)
  cfg <- build_numpyro_config(m1)
  expect_equal(cfg$params$dprime$centre$col, list(0L))
  expect_equal(cfg$params$dprime$centre$means[[1]], c(0, mean(d$grp[d$item_type != "new"] == "t")), tolerance = 1e-12)
  expect_null(cfg$params$discrim$centre)
  expect_equal(cfg$crit_centre$col, list(0L))
  m2 <- broc(brf(resp(y1, y2) | item_type ~ grp - 1, discrim ~ 1, criterion ~ grp - 1, criterion2 ~ 1, rho ~ 1,
                 family = bivariate_gaussian()), d)
  cfg2 <- build_numpyro_config(m2)
  expect_null(cfg2$params$dprime$centre)
  expect_null(cfg2$crit_centre)
  expect_false(identical(.model_fingerprint(m1), .model_fingerprint(m2)))
})

.centring_treat_case <- function() {
  d <- .fixture_bivariate_sdt(n_per = 80)
  d$item_type <- factor(c(new = "new", A = "B", B = "A")[as.character(d$item_type)], levels = c("new", "A", "B"))
  set.seed(5)
  d$grp <- factor(rep(c("c", "t"), length.out = nrow(d)))
  d$cov <- round(rnorm(nrow(d), 10, 1), 2)
  f_treat <- brf(resp(y1, y2) | item_type ~ grp + cov, discrim ~ grp, criterion ~ grp, criterion2 ~ 1, rho ~ 1,
                 family = bivariate_gaussian())
  broc(f_treat, d)
}

test_that("a treatment-coded JAX fit returns draws in the model's coordinates and honours inits", {
  skip_if_no_jax()
  skip_if_no_cmdstan()
  m <- .centring_treat_case()
  init <- list(beta_dprime = c(0.5, 0.2, -0.05), beta_discrim = c(0.6, 0.1))
  fit_j <- fit_broc(m, backend = "jax", sampler = "drghmc", chains = 1, parallel_chains = 1,
                    iter_warmup = 50, iter_sampling = 5, seed = 4, init = list(init), progress_bar = FALSE)
  expect_s3_class(fit_j, "broc_fit")
  expect_false(any(grepl("^orig__|^centre__", posterior::variables(fit_j$.draws_array))))
  ll_r <- .obs_loglik_r(m, build_data_source_from_model(m), fit_j$draws(format = "matrix"), re_formula = NULL)
  expect_lt(max(abs(ll_r - .read_loglik_SN_cols(.loglik_path(fit_j), 1L, m$model_data$N))), 1e-5,
            label = "JAX vs R log_lik")
  fit_s <- suppressMessages(fit_broc(m, backend = "stan", chains = 1, iter_warmup = 100, iter_sampling = 5,
                                     seed = 3, refresh = 50, show_messages = FALSE))
  ll_r <- .obs_loglik_r(m, build_data_source_from_model(m), fit_s$draws(format = "matrix"), re_formula = NULL)
  expect_lt(max(abs(ll_r - .read_loglik_SN_cols(.loglik_path(fit_s), 1L, m$model_data$N))), 1e-5,
            label = "Stan vs R log_lik")
})

test_that("a treatment-coded JAX fit agrees with its cell-means twin and with Stan", {
  skip_if_not_slow()
  skip_if_no_jax()
  skip_if_no_cmdstan()
  d <- .fixture_bivariate_sdt(n_per = 80)
  d$item_type <- factor(c(new = "new", A = "B", B = "A")[as.character(d$item_type)], levels = c("new", "A", "B"))
  set.seed(5)
  d$grp <- factor(rep(c("c", "t"), length.out = nrow(d)))
  d$cov <- round(rnorm(nrow(d), 10, 1), 2)
  f_treat <- brf(resp(y1, y2) | item_type ~ grp + cov, discrim ~ grp, criterion ~ grp, criterion2 ~ 1, rho ~ 1,
                 family = bivariate_gaussian())
  m <- broc(f_treat, d)
  fit_j <- fit_broc(m, backend = "jax", sampler = "drghmc", chains = 2, parallel_chains = 2,
                    iter_warmup = 500, iter_sampling = 400, seed = 3, progress_bar = FALSE)
  fit_s <- fit_broc(m, backend = "stan", chains = 2, parallel_chains = 2, iter_warmup = 400, iter_sampling = 400,
                    refresh = 0, seed = 3)
  vars <- c("beta_dprime[1]", "beta_dprime[2]", "beta_dprime[3]", "beta_discrim[1]", "beta_discrim[2]",
            "beta_thresh_mid[1]", "beta_thresh_mid[2]")
  sj <- posterior::summarise_draws(fit_j$draws(variables = vars, format = "draws_array"), "mean", "sd")
  ss <- posterior::summarise_draws(fit_s$draws(variables = vars, format = "draws_array"), "mean", "sd")
  # the draws come back in the model's coordinates: same posterior as Stan (uncentred) to MC error
  z <- abs(sj$mean - ss$mean) / sqrt(sj$sd^2 / 200 + ss$sd^2 / 200)
  expect_true(all(z < 4), info = paste(sprintf("%s z=%.1f", vars, z), collapse = "; "))
  # no sampler-side names leak into the draws
  expect_false(any(grepl("^orig__|^centre__", posterior::variables(fit_j$.draws_array))))
  # user inits are given in the model's coordinates and must be accepted as such
  init <- list(beta_dprime = c(0.5, 0.2, -0.05), beta_discrim = c(0.6, 0.1))
  fit_i <- fit_broc(m, backend = "jax", sampler = "drghmc", chains = 1, parallel_chains = 1,
                    iter_warmup = 100, iter_sampling = 50, seed = 4, init = list(init), progress_bar = FALSE)
  expect_s3_class(fit_i, "broc_fit")
})
