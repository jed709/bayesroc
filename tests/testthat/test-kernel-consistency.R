.run_kernel_check <- function(script, cases = NULL) {
  py <- jax_python()
  args <- c(testthat::test_path("python", script), jax_backend_dir())
  if (!is.null(cases)) args <- c(args, paste(cases, collapse = ","))
  out <- system2(py, args, stdout = TRUE, stderr = FALSE)
  line <- grep("^RESULT ", out, value = TRUE)
  expect_length(line, 1)
  jsonlite::fromJSON(sub("^RESULT ", "", line))
}

.consistent_kernels <- c(
  "simple_probit", "simple_logit",
  "dpsdt_probit", "dpsdt_logit",
  "mixture_probit", "mixture_logit",
  "source_mixture_probit", "source_mixture_logit"
)

test_that("fused kernel JVPs are the derivative of their own value", {
  skip_if_no_jax()
  res <- .run_kernel_check("jvp_check.py", .consistent_kernels)
  for (k in .consistent_kernels) {
    r <- res[[k]]
    expect_true(r$finite, label = k)
    expect_lt(r$primal_abs, 1e-10, label = paste(k, "JVP primal vs plain value"))
    expect_lt(r$tangent_abs, 1e-6, label = paste(k, "JVP tangent vs finite difference"))
  }
})
