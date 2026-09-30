fixture_crit2 <- function(seed = 41, n = 1500, slope = 0.5) {
  set.seed(seed)
  d <- data.frame(
    item_type = factor(sample(c("new", "A", "B"), n, TRUE), levels = c("new", "A", "B")),
    x = round(runif(n, -2, 2), 3)
  )
  tau1 <- c(-1.0, -0.3, 0.4, 1.1, 1.8)
  tau2 <- c(-0.9, -0.25, 0.45, 1.2)
  mu1 <- ifelse(d$item_type == "new", 0, 1.1)
  mu2 <- ifelse(d$item_type == "A", -1.0, ifelse(d$item_type == "B", 1.0, 0))
  lat1 <- rnorm(n, mu1, 1)
  lat2 <- rnorm(n, mu2, 1)
  d$det <- as.integer(1L + rowSums(outer(lat1, tau1, ">")))
  d$src <- as.integer(1L + rowSums((lat2 - slope * d$x) > outer(rep(1, n), tau2)))
  d
}

by_name <- function(fit, pattern) {
  s <- posterior::summarise_draws(fit$draws(format = "array"), "mean")
  s <- s[grep(pattern, s$variable), ]
  stats::setNames(s$mean, sub("[[]([0-9]+)[]]$", "[\\1,1]", s$variable))
}

test_that("criterion2 design matrix reaches the JAX config", {
  d <- fixture_crit2()
  m <- broc(brf(resp(det, src) | item_type ~ 1, discrim ~ 1, rho ~ 1,
                criterion ~ 1, criterion2 ~ 1 + x, family = bivariate_gaussian()),
            data = d)
  cfg <- get_numpyro_config(m)
  expect_equal(m$stan_data$P_criterion2, 2L)
  expect_false(is.null(cfg$params$criterion2))
  expect_equal(cfg$params$criterion2$n_coef, 2L)
  expect_equal(max(abs(as.matrix(cfg$params$criterion2$X) -
                         as.matrix(m$stan_data$X_criterion2))), 0)
})

test_that("batch C++ uses per-observation gaps in the criterion2 gradient", {
  d <- fixture_crit2()
  m <- broc(brf(resp(det, src) | item_type ~ 1, discrim ~ 1, rho ~ 1,
                criterion ~ 1, criterion2 ~ 1 + x, family = bivariate_gaussian()),
            data = d)
  l <- strsplit(m$batch_cpp$cpp_code, "\n", fixed = TRUE)[[1]]
  expect_gt(length(grep("obs_exp_gaps2\\(gi\\) = egval2;", l)), 0)
  grad_lines <- grep("double g = d \\* \\(?-?pop_exp_gaps2", l, value = TRUE)
  expect_length(grad_lines, 0)
})

test_that("criterion2 linear covariate agrees across Stan and JAX", {
  skip_on_cran()
  skip_if_no_cmdstan()
  skip_if_no_jax()
  d <- fixture_crit2()
  m <- broc(brf(resp(det, src) | item_type ~ 1, discrim ~ 1, rho ~ 1,
                criterion ~ 1, criterion2 ~ 1 + x, family = bivariate_gaussian()),
            data = d)

  fs <- fit_broc(m, backend = "stan", chains = 2, parallel_chains = 2,
                 iter_warmup = 600, iter_sampling = 600, seed = 13, refresh = 50)
  fj <- fit_broc(m, backend = "jax", chains = 2, parallel_chains = 2,
                 iter_warmup = 600, iter_sampling = 600, seed = 13, progress_bar = FALSE)

  a <- by_name(fs, "^beta_thresh_mid_2|^beta_log_gaps_2|^beta_discrim")
  b <- by_name(fj, "^beta_thresh_mid_2|^beta_log_gaps_2|^beta_discrim")
  common <- intersect(names(a), names(b))
  expect_gte(length(common), 8)
  expect_equal(unname(a[common]), unname(b[common]), tolerance = 0.08)

  # the gap intercepts must not collapse to zero (JAX .set() indentation bug)
  gap_int <- grep("^beta_log_gaps_2\\[[0-9]+,1\\]$", names(b), value = TRUE)
  expect_gt(max(abs(b[gap_int])), 0.1)
})
