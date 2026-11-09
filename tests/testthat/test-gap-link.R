
test_that("gap priors reach the JAX config under either gap link, including per-gap criterion2 priors", {
  d <- .fixture_bivariate_sdt()
  pr <- c(broc_prior("normal(0, 0.3)", class = "b", dpar = "log_gaps"),
          broc_prior("normal(0, 0.4)", class = "b", dpar = "log_gaps2"),
          broc_prior("normal(0.1, 0.2)", class = "b", dpar = "log_gaps2", coef = "[2]"))
  for (gl in c("log", "softplus")) {
    m <- broc(brf(resp(y1, y2) | item_type ~ 1, discrim ~ 1, rho ~ 1, criterion ~ 1, criterion2 ~ 1,
                  family = bivariate_gaussian()), d, priors = pr, gap_link = gl)
    cfg <- get_numpyro_config(m)
    expect_identical(cfg$prior_log_gaps, "normal(0, 0.3)")
    expect_identical(cfg$prior_log_gaps2, "normal(0, 0.4)")
    expect_identical(cfg$prior_log_gaps2_per_gap[2], "normal(0.1, 0.2)")
    expect_identical(cfg$prior_log_gaps2_per_gap[1], "normal(0, 0.4)")
  }
})
