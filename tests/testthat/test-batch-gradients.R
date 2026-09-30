.batch_grad_err <- function(m, seed = 1, init = NULL) {
  expect_false(is.null(m$batch_cpp), label = "model has a batch C++ likelihood")
  mod <- .compile_broc_stan(m)
  g <- mod$diagnose(data = m$stan_data, seed = seed, init = init, epsilon = 1e-6, error = 1e6)$gradients()
  max(abs(g$error) / pmax(1, abs(g$finite_diff)))
}

.fixture_crit_smooth <- function(seed = 7, n_sub = 10, n_item = 30) {
  set.seed(seed)
  d <- expand.grid(item = seq_len(n_item), sub = seq_len(n_sub))
  d$old <- as.integer((d$item + d$sub) %% 2 == 0)
  d$x <- round(runif(nrow(d), -2, 2), 3)
  n <- nrow(d)
  lat <- ifelse(d$old == 1, rnorm(n, 1.0, 1), rnorm(n))
  thr <- outer(rep(1, n), c(-1.0, -0.35, 0.3, 0.95, 1.7)) + 0.45 * sin(1.6 * d$x)
  d$conf <- as.integer(1L + rowSums(lat > thr))
  d$sub <- factor(d$sub)
  d
}

.fixture_crit2_design <- function(seed = 41, n = 900) {
  set.seed(seed)
  d <- data.frame(item_type = factor(sample(c("new", "A", "B"), n, TRUE), levels = c("new", "A", "B")),
                  x = round(runif(n, -2, 2), 3))
  mu1 <- ifelse(d$item_type == "new", 0, 1.1)
  mu2 <- ifelse(d$item_type == "A", -1.0, ifelse(d$item_type == "B", 1.0, 0))
  d$det <- as.integer(1L + rowSums(outer(rnorm(n, mu1), c(-1.0, -0.3, 0.4, 1.1, 1.8), ">")))
  d$src <- as.integer(1L + rowSums((rnorm(n, mu2) - 0.5 * sin(d$x)) > outer(rep(1, n), c(-0.9, -0.25, 0.45, 1.2))))
  d
}

test_that("batch C++ gradients match finite differences: univariate, both gap links", {
  skip_if_no_cmdstan()
  d <- .fixture_crit_smooth()
  for (gl in c("log", "softplus")) {
    specs <- list(
      brf(conf | old ~ 1, family = evsd(), gap_link = gl),
      brf(conf | old ~ 1, sigma ~ 1, family = uvsd(), gap_link = gl),
      brf(conf | old ~ 1, criterion ~ 1 + x, family = evsd(), gap_link = gl),
      brf(conf | old ~ 1, criterion ~ s(x), family = evsd(), gap_link = gl),
      brf(conf | old ~ 1, criterion ~ 1 + (1 | sub), family = evsd(), gap_link = gl)
    )
    for (i in seq_along(specs)) {
      m <- broc(specs[[i]], data = d)
      expect_lt(.batch_grad_err(m), 1e-4, label = paste("univariate spec", i, "gap_link", gl))
    }
  }
})

test_that("batch C++ gradients match finite differences: bivariate criterion2 designs", {
  skip_if_no_cmdstan()
  d <- .fixture_crit2_design()
  for (gl in c("log", "softplus")) {
    specs <- list(
      brf(resp(det, src) | item_type ~ 1, discrim ~ 1, rho ~ 1, criterion ~ 1, criterion2 ~ 1,
          family = bivariate_gaussian(), gap_link = gl),
      brf(resp(det, src) | item_type ~ 1, discrim ~ 1, rho ~ 1, criterion ~ 1, criterion2 ~ 1 + x,
          family = bivariate_gaussian(), gap_link = gl),
      brf(resp(det, src) | item_type ~ 1, discrim ~ 1, rho ~ 1, criterion ~ 1, criterion2 ~ s(x),
          family = bivariate_gaussian(), gap_link = gl)
    )
    for (i in seq_along(specs)) {
      m <- broc(specs[[i]], data = d)
      expect_lt(.batch_grad_err(m, init = 0.5), 1e-4, label = paste("bivariate spec", i, "gap_link", gl))
    }
  }
})
