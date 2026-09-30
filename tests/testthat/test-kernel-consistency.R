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

test_that("with the native kernels loaded, every probit and bivariate JVP matches finite differences", {
  skip_if_no_jax()
  lib <- .bvn_ffi_lib()
  ks <- c(paste0(c("simple", "dpsdt", "mixture", "source_mixture"), "_probit"),
          "bivariate_sdt", "bounded_bivariate_sdt", "bivariate_dp", "bounded_bivariate_dp", "vrdp2d")
  res <- .run_py_check("native_jvp_check.py", c(lib, paste(ks, collapse = ",")))
  expect_true(res$ffi_loaded)
  for (k in ks) expect_lt(res[[k]], 1e-6, label = paste(k, "native JVP vs finite differences"))
})

test_that("f32 mode agrees with f64 to single-precision rounding", {
  skip_if_no_jax()
  lib <- .bvn_ffi_lib()
  res <- .run_py_check("f32_check.py", lib)
  expect_true(res$ffi_loaded)
  expect_true(res$f32)
  native <- setdiff(names(res$rel_err), "probit_diff")
  for (k in native) expect_lt(res$rel_err[[k]], 2e-7, label = paste(k, "f32 vs f64"))
  expect_lt(res$rel_err$probit_diff, 1e-4, label = "f32 probit cells vs f64 (single-precision cancellation)")
})

test_that("bivariate normal rectangle and CDF are accurate, native and JAX paths agree", {
  skip_if_no_jax()
  lib <- .bvn_ffi_lib()
  res <- .run_py_check("bvn_check.py", c(lib,
                                         testthat::test_path("fixtures", "bvn_tail_ref.csv")))
  expect_lt(res$rect_abs_err, 1e-9)
  expect_lt(res$rect_rel_err, 1e-6)
  expect_lt(res$tail_rel_err, 1e-8)
  expect_lt(res$tail_fallback_rel_err, 1e-8)
  expect_lt(res$cdf_abs_err, 1e-8)
  expect_lt(res$acc_phi_abs_err, 1e-13)
  expect_lt(res$acc_phi_tail_rel_err, 1e-12)
  expect_true(res$ffi_loaded)
  expect_lt(res$ffi_vs_jax_rect, 2e-6)
  expect_lt(res$ffi_vs_jax_cdf, 1e-8)
  expect_lt(res$probit_cell_rel_err, 1e-8)
  expect_lt(res$probit_cell_fd_err, 1e-5)
})
