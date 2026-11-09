# Helpers shared with tests/testthat/
# test-save_fit.R (copied).

.nuke_stan_csvs <- function(fit) {
  csvs <- tryCatch(fit$.backend_fit$output_files(), error = function(e) character(0))
  csvs <- csvs[file.exists(csvs)]
  if (length(csvs)) file.remove(csvs)
  invisible(csvs)
}
.draws_mean <- function(f, var = "beta_dprime[1]") {
  m <- posterior::as_draws_matrix(f$draws(variables = var, format = "matrix"))
  mean(m[, 1])
}

test_that("Stan: save_broc_fit()/load_broc_fit() survive loss of temp CSVs", {
  skip_if_no_cmdstan()
  m <- .fx_model("uvsd")
  fit <- stan_fit_broc(m, refresh = 50, seed = 1,
                  parallel_chains = 1, chains = 1,
                  iter_warmup = 30, iter_sampling = 20)
  ref <- .draws_mean(fit)

  f <- tempfile(fileext = ".rds")
  save_broc_fit(fit, f, verbose = FALSE)
  .nuke_stan_csvs(fit)

  fit2 <- load_broc_fit(f)
  expect_s3_class(fit2, "broc_fit")
  expect_equal(.draws_mean(fit2), ref, tolerance = 1e-8)
  expect_no_error(summary(fit2))
})
