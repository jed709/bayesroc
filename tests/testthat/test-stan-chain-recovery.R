.mock_dropped <- function(fit, kept, files) {
  list(
    num_chains = function() kept,
    output_files = function(include_failed = FALSE) files,
    draws = function(variables = NULL, ...) {
      d <- fit$draws(variables = variables, ...)
      posterior::subset_draws(d, chain = seq_len(kept))
    }
  )
}

test_that("a chain cmdstanr dropped is read back when its output is complete, else fit_broc stops", {
  skip_if_no_cmdstan()
  m <- .fx_model("evsd")
  f <- fit_broc(m, backend = "stan", chains = 2, parallel_chains = 2,
                iter_warmup = 20, iter_sampling = 100, refresh = 50, seed = 3)
  sf <- f$.backend_fit
  files <- sf$output_files()
  expect_identical(.recover_dropped_chains(sf, 2), sf)

  expect_warning(rec <- .recover_dropped_chains(.mock_dropped(sf, 1, files), 2),
                 "read back from their CSV files")
  expect_equal(rec$num_chains(), 2)
  expect_equal(posterior::ndraws(rec$draws("lp__")), 200)

  bad <- tempfile(fileext = ".csv")
  lines <- readLines(files[2])
  writeLines(lines[seq_len(length(lines) - 60)], bad)
  expect_error(.recover_dropped_chains(.mock_dropped(sf, 1, c(files[1], bad)), 2),
               "refusing to return a fit built from a subset of chains")
})
