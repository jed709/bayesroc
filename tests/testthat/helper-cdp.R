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
