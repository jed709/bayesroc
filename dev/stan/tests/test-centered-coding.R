# Helpers shared with tests/testthat/
# test-centered-coding.R (copied).

.centered_cases <- function() {
  d <- .fixture_univariate(seed = 5, n_per = 150)
  d$is_old <- ifelse(d$is_old == 1, 0.5, -0.5)
  list(
    dpsd = list(brf(item_resp | is_old ~ 1, lambda ~ 1, sigma ~ 1, family = dpsd()), d),
    mixture = list(brf(item_resp | is_old ~ 1, dprime2 ~ 1, lambda ~ 1, sigma ~ 1, family = mixture()), d)
  )
}

test_that("centered is_old coding: plain and batch Stan agree and match R", {
  skip_if_no_cmdstan()
  for (nm in names(cs <- .centered_cases())) {
    c1 <- cs[[nm]]
    mb <- suppressMessages(broc(c1[[1]], c1[[2]]))
    mp <- suppressMessages(.broc_plain(c1[[1]], c1[[2]]))
    r <- .stan_parity(mp, mb)
    expect_lt(r$lp, 1e-8, label = paste(nm, "plain vs batch log density"))
    expect_lt(r$grad, 1e-8, label = paste(nm, "plain vs batch gradient"))
    f <- suppressMessages(stan_fit_broc(mb, chains = 1, iter_warmup = 100, iter_sampling = 5,
                                   seed = 1, refresh = 50, show_messages = FALSE))
    dm <- f$draws(format = "matrix")
    ll_r <- .obs_loglik_r(mb, build_data_source_from_model(mb), dm, re_formula = NULL)
    ll_s <- .read_loglik_SN_cols(.loglik_path(f), 1L, mb$model_data$N)
    expect_lt(max(abs(ll_r - ll_s)), 1e-5, label = paste(nm, "Stan vs R log_lik"))
  }
})
