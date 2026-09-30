.run_py_check <- function(script, args = character(0)) {
  py <- jax_python()
  out <- system2(py, c(testthat::test_path("python", script), dirname(find_backend_script()), args),
                 stdout = TRUE, stderr = FALSE)
  line <- grep("^RESULT ", out, value = TRUE)
  expect_length(line, 1)
  jsonlite::fromJSON(sub("^RESULT ", "", line))
}

.kernels <- c(
  paste0(rep(c("simple", "dpsdt", "mixture", "source_mixture"), each = 3), "_",
         c("probit", "logit", "cloglog")),
  "bivariate_sdt", "bounded_bivariate_sdt", "bivariate_dp", "bounded_bivariate_dp",
  "vrdp2d", "cdp_rk", "cdp_rkg"
)

test_that("every fused kernel's custom JVP is the exact derivative of its value", {
  skip_if_no_jax()
  res <- .run_py_check("jvp_check.py", paste(.kernels, collapse = ","))
  for (k in .kernels) {
    r <- res[[k]]
    expect_true(isTRUE(r$custom_jvp), label = paste(k, "has a custom JVP"))
    expect_true(r$finite, label = k)
    expect_lt(r$primal_abs, 1e-10, label = paste(k, "JVP primal vs plain value"))
    expect_lt(r$tangent_abs, 1e-8, label = paste(k, "JVP tangent vs autodiff of the value"))
  }
})

test_that("bivariate normal rectangle and CDF are accurate, native and JAX paths agree", {
  skip_if_no_jax()
  lib <- .bvn_ffi_lib()
  res <- .run_py_check("bvn_check.py", c(if (is.null(lib)) '""' else lib,
                                         testthat::test_path("fixtures", "bvn_tail_ref.csv")))
  expect_lt(res$rect_abs_err, 1e-9)
  expect_lt(res$rect_rel_err, 1e-6)
  expect_lt(res$tail_rel_err, 1e-8)
  expect_lt(res$tail_fallback_rel_err, 1e-8)
  expect_lt(res$cdf_abs_err, 1e-8)
  expect_lt(res$acc_phi_abs_err, 1e-13)
  expect_lt(res$acc_phi_tail_rel_err, 1e-12)
  if (!is.null(lib)) {
    expect_true(res$ffi_loaded)
    expect_lt(res$ffi_vs_jax_rect, 2e-6)
    expect_lt(res$ffi_vs_jax_cdf, 1e-8)
  }
})
