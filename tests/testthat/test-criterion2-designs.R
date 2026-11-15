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
