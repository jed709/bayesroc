test_that("softplus gap link reaches every threshold construction in the generated Stan", {
  d <- .fixture_bivariate_sdt()
  fams <- list(
    bivariate_gaussian(source_criteria = "varying"),
    bivariate_gaussian(source_criteria = "varying_shared_new", old_levels = 3:4)
  )
  for (fam in fams) {
    m <- .broc_plain(brf(resp(y1, y2) | item_type ~ 1, discrim ~ 1, rho ~ 1, criterion ~ 1, criterion2 ~ 1,
                  family = fam), d, gap_link = "softplus")
    code <- paste(.with_stan_code(m)$stan_code, collapse = "\n")
    expect_false(grepl("(?<![A-Za-z0-9_])exp\\((beta_raw_gaps|beta_log_gaps|raw_gap|log_gap)", code, perl = TRUE))
    expect_false(grepl("beta_log_gaps", code, fixed = TRUE))
    expect_true(grepl("log1p_exp(beta_raw_gaps", code, fixed = TRUE))
    skip_if_no_cmdstan()
    f <- tempfile(fileext = ".stan")
    writeLines(.with_stan_code(m)$stan_code, f)
    expect_true(cmdstanr::cmdstan_model(f, compile = FALSE)$check_syntax(quiet = TRUE))
  }
})
