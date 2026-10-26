skip_if_no_cmdstan <- function() {
  testthat::skip_if_not_installed("cmdstanr")
  ok <- tryCatch({
    p <- cmdstanr::cmdstan_path()
    nzchar(p) && dir.exists(p)
  }, error = function(e) FALSE, warning = function(w) FALSE)
  if (!ok) testthat::skip("CmdStan not installed.")
}

.skip_cache <- new.env(parent = emptyenv())

skip_if_no_jax <- function() {
  if (is.null(.skip_cache$jax)) {
    .skip_cache$jax <- tryCatch({
      utils::capture.output(.chk <- bayesroc::check_jax_backend())
      .chk$status
    }, error = function(e) "no_python")
  }
  if (!identical(.skip_cache$jax, "ok")) testthat::skip(paste("JAX backend not available:", .skip_cache$jax))
}

.slow_tier <- function() identical(Sys.getenv("BAYESROC_TEST_FAST", "true"), "false")

skip_if_not_slow <- function() {
  if (!.slow_tier()) testthat::skip("slow tier: set BAYESROC_TEST_FAST=false")
}
