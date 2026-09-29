
skip_if_no_cmdstan <- function() {
  testthat::skip_if_not_installed("cmdstanr")
  ok <- tryCatch({
    p <- cmdstanr::cmdstan_path()
    nzchar(p) && dir.exists(p)
  }, error = function(e) FALSE, warning = function(w) FALSE)
  if (!ok) testthat::skip("CmdStan not installed.")
}

skip_if_no_jax <- function() {
  res <- tryCatch({
    utils::capture.output(.chk <- bayesroc::check_jax_backend())
    .chk
  }, error = function(e) list(status = "no_python"))
  if (!isTRUE(res$status == "ok")) {
    testthat::skip(paste("JAX backend not available:", res$status))
  }
}
