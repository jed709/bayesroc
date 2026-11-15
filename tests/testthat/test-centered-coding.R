.centered_cases <- function() {
  d <- .fixture_univariate(seed = 5, n_per = 150)
  d$is_old <- ifelse(d$is_old == 1, 0.5, -0.5)
  list(
    dpsd = list(brf(item_resp | is_old ~ 1, lambda ~ 1, sigma ~ 1, family = dpsd()), d),
    mixture = list(brf(item_resp | is_old ~ 1, dprime2 ~ 1, lambda ~ 1, sigma ~ 1, family = mixture()), d)
  )
}

test_that("centered is_old coding: JAX matches R", {
  skip_if_no_jax()
  for (nm in names(cs <- .centered_cases())) {
    c1 <- cs[[nm]]
    m <- suppressMessages(broc(c1[[1]], c1[[2]]))
    f <- suppressMessages(fit_broc(m, chains = 1, iter_warmup = 50,
                                   iter_sampling = 5, seed = 1, progress_bar = FALSE))
    ll_r <- .obs_loglik_r(m, build_data_source_from_model(m), f$draws(format = "matrix"), re_formula = NULL)
    ll_j <- .read_loglik_SN_cols(.loglik_path(f), 1L, m$model_data$N)
    expect_lt(max(abs(ll_r - ll_j)), 1e-5, label = paste(nm, "JAX vs R log_lik"))
  }
})
