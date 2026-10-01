
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
  d <- .fixture_univariate(sigma = 1.2)
  m <- broc(brf(item_resp | is_old ~ 1, sigma ~ 1, family = uvsd()), d)
  fit <- fit_broc(m, backend = "stan", refresh = 50, seed = 1,
                  parallel_chains = 2, chains = 2,
                  iter_warmup = 200, iter_sampling = 200)
  ref <- .draws_mean(fit)

  f <- tempfile(fileext = ".rds")
  save_broc_fit(fit, f, verbose = FALSE)
  .nuke_stan_csvs(fit)

  fit2 <- load_broc_fit(f)
  expect_s3_class(fit2, "broc_fit")
  expect_equal(.draws_mean(fit2), ref, tolerance = 1e-8)
  expect_no_error(summary(fit2))
})

test_that("fit_broc(file=) saves, reloads instead of refitting, and refits on demand", {
  skip_if_no_cmdstan()
  d <- .fixture_univariate(sigma = 1.2)
  m <- broc(brf(item_resp | is_old ~ 1, sigma ~ 1, family = uvsd()), d)

  path <- tempfile()
  rds  <- paste0(path, ".rds")

  fit <- fit_broc(m, backend = "stan", refresh = 50, seed = 1,
                  parallel_chains = 2, chains = 2,
                  iter_warmup = 200, iter_sampling = 200, file = path)
  expect_true(file.exists(rds))
  ref <- .draws_mean(fit)

  fit_loaded <- fit_broc(m, backend = "stan", refresh = 50, seed = 999,
                         parallel_chains = 2, chains = 2,
                         iter_warmup = 200, iter_sampling = 200, file = path)
  expect_equal(.draws_mean(fit_loaded), ref, tolerance = 1e-8)

  fit_refit <- fit_broc(m, backend = "stan", refresh = 50, seed = 7,
                        parallel_chains = 2, chains = 2,
                        iter_warmup = 200, iter_sampling = 200,
                        file = path, file_refit = TRUE)
  expect_s3_class(fit_refit, "broc_fit")
  expect_no_error(summary(fit_refit))
})

test_that("fit_broc(file=) works on the JAX backend", {
  skip_if_no_jax()
  d <- .fixture_univariate(sigma = 1.2)
  m <- broc(brf(item_resp | is_old ~ 1, sigma ~ 1, family = uvsd()), d)
  path <- tempfile()

  fit <- fit_broc(m, backend = "jax", chains = 2L,
                  iter_warmup = 200L, iter_sampling = 200L,
                  seed = 1, progress_bar = FALSE, file = path)
  expect_true(file.exists(paste0(path, ".rds")))
  ref <- .draws_mean(fit)

  fit_loaded <- fit_broc(m, backend = "jax", chains = 2L,
                         iter_warmup = 200L, iter_sampling = 200L,
                         seed = 2, progress_bar = FALSE, file = path)
  expect_equal(.draws_mean(fit_loaded), ref, tolerance = 1e-8)
})
