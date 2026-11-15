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
  .fx_memo("cdp_rkg:jax", suppressMessages(fit_broc(cdp_rkg_model(fixture_cdp_rkg()), chains = 1,
                                                     iter_warmup = 50, iter_sampling = 5, seed = 5,
                                                     progress_bar = FALSE)))
}

test_that("cdp R/K/G predicted probabilities are finite and sum to one", {
  skip_on_cran()
  skip_if_no_jax()
  fj <- .cdp_rkg_jax_fit()
  p <- predict(fj, type = "response", ndraws = 5)
  expect_true(all(is.finite(p$mean)))
  totals <- as.vector(tapply(p$mean, p$trial, sum))
  expect_equal(totals, rep(1, length(totals)), tolerance = 1e-6)
})

test_that("R-side CDP cells match the high-precision reference and partition the band", {
  fx <- utils::read.csv(testthat::test_path("fixtures", "cdp_cell_ref.csv"))
  lp <- mapply(cdp_log_cell_r, fx$kind, fx$mu_R, fx$sigma_R, fx$mu_F, fx$sigma_F, fx$c_R,
               fx$c_K, fx$tau_lo, fx$tau_hi)
  fin <- is.finite(fx$log_p)
  expect_lt(max(abs(lp[fin] - fx$log_p[fin]) / pmax(1, abs(fx$log_p[fin]))), 1e-9)
  expect_true(all(!is.finite(lp[!fin])))
  dev <- mapply(function(a, b, c, d, e, f, g, h) {
    s <- sqrt(b^2 + d^2)
    sum(compute_rkg_probs_r(a, b, c, d, e, f, g, h)) - (pnorm((h - a - c) / s) - pnorm((g - a - c) / s))
  }, fx$mu_R, fx$sigma_R, fx$mu_F, fx$sigma_F, fx$c_R, fx$c_K, fx$tau_lo, fx$tau_hi)
  expect_lt(max(abs(dev)), 5e-12)
})

test_that("R/K/G know-criterion knob: X v = 1, orthonormal complement, Guess trials and bands", {
  d <- fixture_cdp_rkg()
  d$cond <- factor(ifelse(as.integer(d$sub) %% 2 == 0, "a", "b"))
  d$x <- seq(-1, 1, length.out = nrow(d))
  # a random effect on the know criterion (or thresholds, or rec_crit) puts it in the Guess
  # bound and switches the slack transform off
  f <- brf(resp(conf, rk) | old ~ 1, fam ~ 1, criterion ~ 1, rec_crit ~ 1, know_crit ~ 1 + x + (1 | sub),
           sigma_R ~ 1, family = cdp(old_levels = 2:4))
  expect_equal(suppressMessages(broc(f, data = d))$stan_data$kc_knob, 0L)
  f <- brf(resp(conf, rk) | old ~ 1, fam ~ 1, criterion ~ (1 | sub), rec_crit ~ 1, know_crit ~ 1,
           sigma_R ~ 1, family = cdp(old_levels = 2:4))
  expect_equal(suppressMessages(broc(f, data = d))$stan_data$kc_knob, 0L)
  for (kc in list(know_crit ~ 1, know_crit ~ 0 + cond + x)) {
    f <- brf(resp(conf, rk) | old ~ 1, fam ~ 1, criterion ~ 1, rec_crit ~ 1, kc, sigma_R ~ 1,
             family = cdp(old_levels = 2:4))
    sd <- suppressMessages(broc(f, data = d))$stan_data
    expect_equal(sd$kc_knob, 1L)
    X <- as.matrix(sd$X_know_crit)
    expect_equal(as.vector(X %*% sd$kc_v), rep(1, nrow(X)), tolerance = 1e-10)
    B <- sd$kc_Bperp
    if (ncol(B) > 0) {
      expect_equal(crossprod(B), diag(ncol(B)), tolerance = 1e-10)
      expect_equal(drop(crossprod(B, sd$kc_v)), rep(0, ncol(B)), tolerance = 1e-10)
    }
    band <- match(sd$y, sd$old_level_map)
    expect_equal(as.vector(sd$guess_idx), which(!is.na(band) & sd$rk == 3))
    expect_equal(as.vector(sd$guess_band), band[sd$guess_idx])
  }
})

test_that("criterion formulas that force the criterion through zero are rejected at build", {
  d <- fixture_cdp_rkg()
  d$x <- seq(-1, 1, length.out = nrow(d))
  f <- brf(resp(conf, rk) | old ~ 1, fam ~ 1, criterion ~ 1, rec_crit ~ 1, know_crit ~ 0 + x, sigma_R ~ 1,
           family = cdp(old_levels = 2:4))
  expect_error(broc(f, data = d), "`know_crit` has no intercept")
  f2 <- brf(resp(conf, rk) | old ~ 1, fam ~ 1, criterion ~ 1, rec_crit ~ 0 + x, know_crit ~ 1, sigma_R ~ 1,
            family = cdp(old_levels = 2:4))
  expect_error(broc(f2, data = d), "`rec_crit` has no intercept")
})

test_that("JAX honours a user init for the reparameterised know criterion", {
  skip_if_no_jax()
  m <- cdp_rkg_model(fixture_cdp_rkg())
  ini <- list(list(beta_rec_crit = 0.87, beta_know_crit = 0.4, beta_dprime = 1.2),
              list(beta_rec_crit = 0.9, beta_know_crit = 0.3, beta_dprime = 1.1))
  msgs <- character(0)
  f <- withCallingHandlers(
    fit_broc(m, sampler = "nuts", chains = 2, parallel_chains = 2, iter_warmup = 100, iter_sampling = 20,
             seed = 4, init = ini, progress_bar = FALSE),
    message = function(e) { msgs <<- c(msgs, conditionMessage(e)); invokeRestart("muffleMessage") })
  expect_false(any(grepl("ignored", msgs)))
  expect_equal(sum(f$diagnostic_summary()$num_divergent), 0)
})
