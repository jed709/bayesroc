# Helpers shared with tests/testthat/
# test-centring.R (copied).

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
  fit_s <- stan_fit_broc(m, chains = 2, parallel_chains = 2, iter_warmup = 400, iter_sampling = 400,
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
