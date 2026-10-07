test_that("softplus gap link reaches every threshold construction in the generated Stan", {
  d <- .fixture_bivariate_sdt()
  fams <- list(
    bivariate_gaussian(source_criteria = "varying"),
    bivariate_gaussian(source_criteria = "varying_shared_new", old_levels = 3:4)
  )
  for (fam in fams) {
    m <- .broc_plain(brf(resp(y1, y2) | item_type ~ 1, discrim ~ 1, rho ~ 1, criterion ~ 1, criterion2 ~ 1,
                  family = fam), d, gap_link = "softplus")
    code <- paste(m$stan_code, collapse = "\n")
    expect_false(grepl("(?<![A-Za-z0-9_])exp\\((beta_raw_gaps|beta_log_gaps|raw_gap|log_gap)", code, perl = TRUE))
    expect_false(grepl("beta_log_gaps", code, fixed = TRUE))
    expect_true(grepl("log1p_exp(beta_raw_gaps", code, fixed = TRUE))
    skip_if_no_cmdstan()
    f <- tempfile(fileext = ".stan")
    writeLines(m$stan_code, f)
    expect_true(cmdstanr::cmdstan_model(f, compile = FALSE)$check_syntax(quiet = TRUE))
  }
})

test_that("gap priors reach both backends under either gap link, including per-gap criterion2 priors", {
  d <- .fixture_bivariate_sdt()
  pr <- c(broc_prior("normal(0, 0.3)", class = "b", dpar = "log_gaps"),
          broc_prior("normal(0, 0.4)", class = "b", dpar = "log_gaps2"),
          broc_prior("normal(0.1, 0.2)", class = "b", dpar = "log_gaps2", coef = "[2]"))
  for (gl in c("log", "softplus")) {
    m <- broc(brf(resp(y1, y2) | item_type ~ 1, discrim ~ 1, rho ~ 1, criterion ~ 1, criterion2 ~ 1,
                  family = bivariate_gaussian()), d, priors = pr, gap_link = gl)
    code <- paste(m$stan_code, collapse = "\n")
    g1 <- if (gl == "softplus") "beta_raw_gaps" else "beta_log_gaps"
    expect_true(grepl(paste0(g1, ") ~ normal(0, 0.3)"), code, fixed = TRUE) ||
                  grepl(paste0(g1, " ~ normal(0, 0.3)"), code, fixed = TRUE))
    expect_true(grepl(paste0(g1, "_2[2, 1] ~ normal(0.1, 0.2)"), code, fixed = TRUE))
    expect_true(grepl(paste0(g1, "_2[1, 1] ~ normal(0, 0.4)"), code, fixed = TRUE))
    cfg <- get_numpyro_config(m)
    expect_identical(cfg$prior_log_gaps, "normal(0, 0.3)")
    expect_identical(cfg$prior_log_gaps2, "normal(0, 0.4)")
    expect_identical(cfg$prior_log_gaps2_per_gap[2], "normal(0.1, 0.2)")
    expect_identical(cfg$prior_log_gaps2_per_gap[1], "normal(0, 0.4)")
  }
})
