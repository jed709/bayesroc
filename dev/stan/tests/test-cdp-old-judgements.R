# Helpers shared with tests/testthat/
# test-cdp-old-judgements.R (copied).

fixture_cdp_judged <- function(n_rkg = 3L, seed = 4, n_sub = 8, n_item = 40, drop = 0, new_judged = FALSE) {
  set.seed(seed)
  mu_R <- 1.1; sigma_R <- 1.4; mu_F <- 0.7; sigma_F <- 0.9
  c_R <- 1.0; c_K <- 0.2
  tau <- c(0.1, 0.9, 1.8)
  rows <- expand.grid(sub = seq_len(n_sub), item = seq_len(n_item))
  rows$old <- rep(c(0L, 1L), length.out = nrow(rows))
  n <- nrow(rows)
  R <- ifelse(rows$old == 1L, rnorm(n, mu_R, sigma_R), rnorm(n))
  F_ <- ifelse(rows$old == 1L, rnorm(n, mu_F, sigma_F), rnorm(n))
  rows$conf <- as.integer(1L + rowSums(outer(R + F_, tau, ">")))
  rows$rk <- if (n_rkg == 3L) ifelse(R > c_R, 1L, ifelse(F_ > c_K, 2L, 3L)) else ifelse(R > c_R, 1L, 2L)
  if (!new_judged) rows$rk[rows$conf < 3L] <- NA_integer_
  if (drop > 0) {
    old_rows <- which(rows$conf >= 3L)
    rows$rk[sample(old_rows, round(drop * length(old_rows)))] <- NA_integer_
  }
  rows$sub <- factor(rows$sub)
  rows
}
cdp_judged_formula <- function(n_rkg, family = cdp(old_levels = 3:4)) {
  if (n_rkg == 3L) brf(resp(conf, rk) | old ~ 1, fam ~ 1, criterion ~ 1, rec_crit ~ 1, know_crit ~ 1,
                       sigma_R ~ 1, family = family)
  else brf(resp(conf, rk) | old ~ 1, fam ~ 1, criterion ~ 1, rec_crit ~ 1, sigma_R ~ 1, family = family)
}

test_that("cdp with missing old-response judgements: batch C++ matches plain Stan (R/K and R/K/G)", {
  skip_if_no_cmdstan()
  for (n_rkg in 2:3) for (drop in c(0, 0.25)) {
    d <- fixture_cdp_judged(n_rkg, drop = drop)
    mk <- function(b) suppressMessages(.broc_with(b, cdp_judged_formula(n_rkg), data = d))
    edits <- list(list("beta_rec_crit[1]" = 1.0), list("beta_rec_crit[1]" = 3.5), list("beta_rec_crit[1]" = -2.5))
    if (n_rkg == 3L) edits <- lapply(edits, function(e) c(e, list("beta_know_crit[1]" = 0.4)))
    for (ed in edits) {
      r <- .stan_parity_at(mk(FALSE), mk(TRUE), ed)
      lab <- sprintf("n_rkg %d drop %.2f %s", n_rkg, drop, paste(names(ed), unlist(ed), collapse = " "))
      expect_true(all(is.finite(r$lp_values)), label = lab)
      expect_lt(r$lp, 1e-7, label = paste(lab, "log density"))
      expect_lt(r$grad, 1e-6, label = paste(lab, "gradient"))
    }
  }
})

test_that("cdp with missing old-response judgements: JAX and batch Stan log-likelihoods match R, predictions sum to one", {
  skip_if_no_cmdstan()
  skip_if_no_jax()
  d <- fixture_cdp_judged(3L, n_sub = 20, n_item = 60, drop = 0.1)
  m <- suppressMessages(broc(cdp_judged_formula(3L), data = d))
  fj <- suppressMessages(fit_broc(m, backend = "jax", sampler = "drghmc", chains = 1, iter_warmup = 50,
                                  iter_sampling = 5, seed = 3, progress_bar = FALSE))
  pooled <- ifelse(m$stan_data$y %in% m$family$old_levels, 0, log(m$model_data$K - m$family$J))
  ll_r <- sweep(.obs_loglik_r(m, build_data_source_from_model(m), fj$draws(format = "matrix"), re_formula = NULL), 2, pooled, "+")
  expect_lt(max(abs(ll_r - .read_loglik_SN_cols(.loglik_path(fj), 1L, m$model_data$N))), 1e-5, label = "JAX vs R log_lik")
  fs <- suppressMessages(stan_fit_broc(m, chains = 1, iter_warmup = 100, iter_sampling = 5,
                                  seed = 3, refresh = 50, show_messages = FALSE))
  ll_r <- sweep(.obs_loglik_r(m, build_data_source_from_model(m), fs$draws(format = "matrix"), re_formula = NULL), 2, pooled, "+")
  expect_lt(max(abs(ll_r - .read_loglik_SN_cols(.loglik_path(fs), 1L, m$model_data$N))), 1e-5, label = "Stan vs R log_lik")
  p <- predict(fj, type = "response", ndraws = 5)
  expect_equal(as.vector(tapply(p$mean, p$trial, sum)), rep(1, nrow(d)), tolerance = 1e-6)
})
