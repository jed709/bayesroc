test_that("total_cores is validated", {
  expect_null(.check_total_cores(NULL))
  expect_error(.check_total_cores(0), "whole number")
  expect_error(.check_total_cores(2.5), "whole number")
  expect_error(.check_total_cores(c(1, 2)), "whole number")
  n <- parallel::detectCores(logical = TRUE)
  if (!is.na(n)) expect_error(.check_total_cores(n + 1), "exceeds")
  if (!identical(Sys.info()[["sysname"]], "Darwin")) expect_identical(.check_total_cores(1), 1L)
})

test_that("CPUs are taken P-cores first, one per physical core first, and the full set is restored", {
  skip_if_no_jax()
  skip_on_os("mac")
  out <- system2(jax_python(), c(test_path("python", "cpu_limit_check.py"), dirname(find_backend_script())),
                 stdout = TRUE, stderr = FALSE)
  res <- jsonlite::fromJSON(sub("^RESULT ", "", grep("^RESULT ", out, value = TRUE)))
  expect_equal(res$hybrid, c(0, 2, 4, 1, 3, 5, 6, 7))
  expect_equal(res$plain, c(0, 1, 2, 3, 4, 5, 6, 7))
  if (res$start_n >= 2) {
    expect_length(res$limited, 2L)
    expect_equal(res$now, res$limited)
    expect_length(res$restored, res$start_n)
  }
})

test_that("a JAX fit honours total_cores", {
  skip_if_no_jax()
  skip_on_os("mac")
  d <- .fixture_univariate()
  m <- suppressMessages(broc(brf(item_resp | is_old ~ 1, family = evsd()), data = d))
  msgs <- capture.output(fit <- suppressMessages(fit_broc(m, backend = "jax", sampler = "drghmc", chains = 2,
                                                          parallel_chains = 2, iter_warmup = 50, iter_sampling = 10,
                                                          total_cores = 2, seed = 1, loo = FALSE, save_loglik = FALSE)),
                         type = "output")
  expect_true(any(grepl("limited to 2 logical CPUs", msgs)))
  expect_equal(fit$num_chains(), 2)
})
