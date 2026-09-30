fixture_cdp_rkg <- function(seed = 21, n_sub = 8, n_item = 40) {
  set.seed(seed)
  mu_R <- 0.9; sigma_R <- 2.0; mu_F <- 0.8; sigma_F <- 0.8
  c_R <- 0.8; c_K <- 0.3
  tau <- c(0.2, 0.9, 1.6)
  rows <- expand.grid(sub = seq_len(n_sub), item = seq_len(n_item))
  rows$old <- rep(c(0L, 1L), length.out = nrow(rows))
  n <- nrow(rows)
  R <- ifelse(rows$old == 1L, rnorm(n, mu_R, sigma_R), rnorm(n))
  F_ <- ifelse(rows$old == 1L, rnorm(n, mu_F, sigma_F), rnorm(n))
  M <- R + F_
  rows$conf <- as.integer(1L + rowSums(outer(M, tau, ">")))
  rk <- ifelse(R > c_R, 1L, ifelse(F_ > c_K, 2L, 3L))
  rows$rk <- ifelse(rows$conf >= 2L, rk, NA_integer_)
  rows$sub <- factor(rows$sub)
  rows
}

cdp_rkg_model <- function(d) {
  broc(brf(resp(conf, rk) | old ~ 1,
           fam ~ 1,
           criterion ~ 1,
           rec_crit ~ 1,
           know_crit ~ 1,
           sigma_R ~ 1,
           family = cdp(old_levels = 2:4)),
       data = d)
}

test_that("cdp R/K/G agrees across Stan and JAX backends at high rho", {
  skip_on_cran()
  skip_if_no_cmdstan()
  skip_if_no_jax()
  d <- fixture_cdp_rkg()
  m <- cdp_rkg_model(d)

  fs <- fit_broc(m, backend = "stan", chains = 2, parallel_chains = 2,
                 iter_warmup = 400, iter_sampling = 400, seed = 3, refresh = 50)
  fj <- fit_broc(m, backend = "jax", chains = 2, parallel_chains = 2,
                 iter_warmup = 400, iter_sampling = 400, seed = 3,
                 progress_bar = FALSE)

  strip_idx <- function(v) sub("\\[[0-9]+(,[0-9]+)?\\]$", "", v)
  tab <- function(f) {
    s <- posterior::summarise_draws(f$draws(format = "array"), mean)
    s <- s[grep("^beta_", s$variable), ]
    stats::setNames(s$mean, make.unique(strip_idx(s$variable)))
  }
  a <- tab(fs)
  b <- tab(fj)
  common <- intersect(names(a), names(b))
  expect_gte(length(common), 7)
  expect_equal(unname(a[common]), unname(b[common]), tolerance = 0.05)
})

test_that("cdp R/K/G predicted probabilities are finite and sum to one", {
  skip_on_cran()
  skip_if_no_jax()
  d <- fixture_cdp_rkg()
  m <- cdp_rkg_model(d)
  fj <- fit_broc(m, backend = "jax", chains = 2, parallel_chains = 2,
                 iter_warmup = 300, iter_sampling = 300, seed = 5,
                 progress_bar = FALSE)
  p <- predict(fj, type = "response", ndraws = 20)
  expect_true(all(is.finite(p$mean)))
  totals <- as.vector(tapply(p$mean, p$trial, sum))
  expect_equal(totals, rep(1, length(totals)), tolerance = 1e-6)
})
