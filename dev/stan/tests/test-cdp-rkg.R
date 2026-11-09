# Helpers shared with tests/testthat/
# test-cdp-rkg.R (copied).

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
.cdp_rkg_jax_fit <- function() {
  .fx_memo("cdp_rkg:jax", suppressMessages(fit_broc(cdp_rkg_model(fixture_cdp_rkg()), backend = "jax", sampler = "drghmc",
                                                     chains = 1, iter_warmup = 50, iter_sampling = 5, seed = 5,
                                                     progress_bar = FALSE)))
}

test_that("cdp R/K/G: JAX and batch Stan log-likelihoods match R", {
  skip_if_no_cmdstan()
  skip_if_no_jax()
  d <- fixture_cdp_rkg()
  m <- cdp_rkg_model(d)
  fj <- .cdp_rkg_jax_fit()
  ll_r <- .obs_loglik_r(m, build_data_source_from_model(m), fj$draws(format = "matrix"), re_formula = NULL)
  expect_lt(max(abs(ll_r - .read_loglik_SN_cols(.loglik_path(fj), 1L, m$model_data$N))), 1e-5, label = "JAX vs R log_lik")
  fs <- suppressMessages(stan_fit_broc(m, chains = 1, iter_warmup = 100, iter_sampling = 5,
                                  seed = 3, refresh = 50, show_messages = FALSE))
  ll_r <- .obs_loglik_r(m, build_data_source_from_model(m), fs$draws(format = "matrix"), re_formula = NULL)
  expect_lt(max(abs(ll_r - .read_loglik_SN_cols(.loglik_path(fs), 1L, m$model_data$N))), 1e-5, label = "Stan vs R log_lik")
})

test_that("cdp R/K/G agrees across Stan and JAX backends at high rho", {
  skip_if_not_slow()
  skip_on_cran()
  skip_if_no_cmdstan()
  skip_if_no_jax()
  d <- fixture_cdp_rkg()
  m <- cdp_rkg_model(d)

  fs <- stan_fit_broc(m, chains = 2, parallel_chains = 2,
                 iter_warmup = 400, iter_sampling = 400, seed = 3, refresh = 50)
  fj <- fit_broc(m, backend = "jax", sampler = "nuts", chains = 2, parallel_chains = 2,
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
  expect_equal(sum(fs$diagnostic_summary()$num_divergent), 0)
  expect_equal(sum(fj$diagnostic_summary()$num_divergent), 0)
})

test_that("cdp R/K/G cells far in the tails: batch C++ matches plain Stan", {
  skip_if_no_cmdstan()
  d <- fixture_cdp_rkg()
  mk <- function(b) suppressMessages(.broc_with(b, brf(resp(conf, rk) | old ~ 1, fam ~ 1, criterion ~ 1,
                                              rec_crit ~ 1, know_crit ~ 1, sigma_R ~ 1,
                                              family = cdp(old_levels = 2:4)),
                                          data = d))
  for (ed in list(list("beta_rec_crit[1]" = 4.5, "beta_know_crit[1]" = 0.3),
                  list("beta_rec_crit[1]" = 0.8, "beta_know_crit[1]" = 4.0),
                  list("beta_rec_crit[1]" = -3.0, "beta_know_crit[1]" = -3.5))) {
    r <- .stan_parity_at(mk(FALSE), mk(TRUE), ed)
    lab <- paste(names(ed), unlist(ed), collapse = " ")
    expect_true(all(is.finite(r$lp_values)), label = lab)
    expect_lt(r$lp, 1e-7, label = paste(lab, "log density"))
    expect_lt(r$grad, 1e-6, label = paste(lab, "gradient"))
  }
})
