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
  "vrdp2d", "cdp_rk", "cdp_rkg", "cdp_rkg_tail", "cdp_rk_corr", "cdp_rkg_corr", "cdp_rkg_tail_corr"
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
          "bivariate_sdt", "bounded_bivariate_sdt", "bivariate_dp", "bounded_bivariate_dp", "vrdp2d",
          "cdp_rk", "cdp_rkg", "cdp_rkg_tail", "cdp_rk_corr", "cdp_rkg_corr", "cdp_rkg_tail_corr")
  res <- .run_py_check("native_jvp_check.py", c(lib, paste(ks, collapse = ",")))
  expect_true(res$ffi_loaded)
  for (k in ks) {
    tol <- if (k %in% c("cdp_rkg_tail", "cdp_rkg_tail_corr")) 1e-5 else 1e-6
    expect_lt(res[[k]], tol, label = paste(k, "native JVP vs finite differences"))
  }
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

test_that("bivariate normal rectangle and CDF are accurate at every |rho| including 0.9, 0.99 and 0.999 and in the far tails, native and JAX paths agree", {
  skip_if_no_jax()
  lib <- .bvn_ffi_lib()
  res <- .run_py_check("bvn_check.py", c(lib,
                                         testthat::test_path("fixtures", "bvn_tail_ref.csv"),
                                         testthat::test_path("fixtures", "cdp_cell_ref.csv"),
                                         testthat::test_path("fixtures", "bvn_tail_hirho_ref.csv")))
  expect_lt(res$rect_abs_err, 1e-9)
  expect_lt(res$rect_rel_err, 1e-6)
  expect_lt(res$hi_rho_rect_rel_err, 1e-6)
  expect_lt(res$hi_rho_cdf_abs_err, 1e-9)
  expect_lt(res$hi_rho_tail_rel_err, 1e-8)
  expect_lt(res$hi_rho_tail_rel_err_999, 1e-5)
  expect_lt(res$hi_rho_ffi_vs_jax, 1e-8)
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
  expect_lt(res$cdp_cell_rel_err, 1e-10)
  expect_lt(res$cdp_cell_jax_rel_err, 1e-10)
  expect_true(res$cdp_cell_empty_ok)
})

test_that("the transport-map kernel matches a reference solve and its gradients at every width", {
  skip_if_no_jax()
  widths <- c(1, 5, 31, 64, 65, 84, 130)
  res <- .run_py_check("tmap_kernel_check.py", c(.bvn_ffi_lib(), paste(widths, collapse = ",")))
  expect_true(res$ffi_loaded)
  for (D in widths) expect_lt(res[[paste0("D", D)]], 1e-10, label = paste("width", D, "kernel vs reference"))
})

test_that("NUTS checks U-turns across subtree boundaries, so resonant step sizes do not run to the depth limit", {
  skip_if_no_jax()
  res <- .run_py_check("nuts_turn_check.py")
  expect_gt(res$stock_resonant, 200)
  expect_lt(res$checked_resonant, 80)
  expect_equal(res$max_draw_diff_off_resonance, 0)
  expect_equal(res$steps_off_resonance[1], res$steps_off_resonance[2])
})
