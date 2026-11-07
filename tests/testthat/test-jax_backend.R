
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
  skip_if_not(file.exists(bayesroc:::managed_venv_python()))
  expect_error(install_jax_backend(overwrite = FALSE), "already exists")
})

test_that("XLA compile cache is opt-in, lives under R_user_dir and can be moved", {
  old <- options(bayesroc.xla_cache = NULL)
  on.exit(options(old))
  expect_identical(.xla_cache_dir(), "0")
  options(bayesroc.xla_cache = TRUE)
  expect_identical(.xla_cache_dir(),
                   normalizePath(file.path(tools::R_user_dir("bayesroc", "cache"), "xla"),
                                 winslash = "/", mustWork = FALSE))
  options(bayesroc.xla_cache = FALSE)
  expect_identical(.xla_cache_dir(), "0")
  d <- tempfile("xla")
  options(bayesroc.xla_cache = d)
  expect_identical(.xla_cache_dir(), normalizePath(d, winslash = "/", mustWork = FALSE))
})
