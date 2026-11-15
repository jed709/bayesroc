
.draws_mean <- function(f, var = "beta_dprime[1]") {
  m <- posterior::as_draws_matrix(f$draws(variables = var, format = "matrix"))
  mean(m[, 1])
}

test_that("fit_broc(file=) saves, reloads instead of refitting, and refits on demand", {
  skip_if_no_jax()
  m <- .fx_model("uvsd")

  path <- tempfile()
  rds  <- paste0(path, ".rds")

  fit <- fit_broc(m, seed = 1,
                  parallel_chains = 1, chains = 1,
                  iter_warmup = 30, iter_sampling = 20, file = path)
  expect_true(file.exists(rds))
  ref <- .draws_mean(fit)

  fit_loaded <- fit_broc(m, seed = 999,
                         parallel_chains = 1, chains = 1,
                         iter_warmup = 30, iter_sampling = 20, file = path)
  expect_equal(.draws_mean(fit_loaded), ref, tolerance = 1e-8)

  fit_refit <- fit_broc(m, seed = 7,
                        parallel_chains = 1, chains = 1,
                        iter_warmup = 30, iter_sampling = 20,
                        file = path, file_refit = TRUE)
  expect_s3_class(fit_refit, "broc_fit")
  expect_no_error(summary(fit_refit))
})
