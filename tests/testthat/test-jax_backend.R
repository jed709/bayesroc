# Tests for the JAX backend management API. These don't fit any models —
# just exercise install/check/jax_python/set_jax_python.

test_that("jax_backend_dir() returns a path", {
  d <- jax_backend_dir()
  expect_type(d, "character")
  expect_length(d, 1)
})

test_that("check_jax_backend() returns a result with the expected fields", {
  capture.output(res <- check_jax_backend())
  expect_type(res, "list")
  expect_named(res, c("status", "python", "jax_version", "numpyro_version", "message"))
  expect_true(res$status %in% c("ok", "compat_warning", "missing_numpyro",
                                 "no_jax", "no_python"))
})

test_that("set_jax_python errors on nonexistent path", {
  expect_error(set_jax_python("/no/such/python.exe"), "not found")
})

test_that("install_jax_backend() errors when overwrite = FALSE on existing venv", {
  # Only run if we already have a managed venv (to avoid creating one in CI
  # just for this check; CI installs one in setup so it'll be present).
  # managed_venv_python is internal — use ::: to access it.
  skip_if_not(file.exists(bayesroc:::managed_venv_python()))
  expect_error(install_jax_backend(overwrite = FALSE), "already exists")
})
