
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

test_that("fit_broc(file=) works on the JAX backend", {
  skip_if_no_jax()
  m <- .fx_model("uvsd")
  path <- tempfile()

  fit <- fit_broc(m, backend = "jax", sampler = "nuts", chains = 2L,
                  iter_warmup = 30L, iter_sampling = 20L,
                  seed = 1, progress_bar = FALSE, file = path)
  expect_true(file.exists(paste0(path, ".rds")))
  ref <- .draws_mean(fit)

  fit_loaded <- fit_broc(m, backend = "jax", sampler = "nuts", chains = 2L,
                         iter_warmup = 30L, iter_sampling = 20L,
                         seed = 2, progress_bar = FALSE, file = path)
  expect_equal(.draws_mean(fit_loaded), ref, tolerance = 1e-8)
})

test_that("draws in centred coordinates are replaced by the model's coordinates", {
  a <- array(c(1, 2, 3, 4, 10, 20), c(2, 1, 3),
             dimnames = list(NULL, NULL, c("beta_dprime[1]", "beta_dprime[2]", "orig__beta_dprime[1]")))
  a <- posterior::as_draws_array(a)
  r <- .restore_centred_draws(a)
  expect_identical(posterior::variables(r), c("beta_dprime[1]", "beta_dprime[2]"))
  expect_equal(as.vector(r[, , "beta_dprime[1]"]), c(10, 20))
  expect_equal(as.vector(r[, , "beta_dprime[2]"]), c(3, 4))
  expect_s3_class(r, "draws_array")
  expect_identical(.restore_centred_draws(r), r)
})
