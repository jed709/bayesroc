skip_if_no_cmdstan <- function() {
  testthat::skip_if_not_installed("cmdstanr")
  ok <- tryCatch({
    p <- cmdstanr::cmdstan_path()
    nzchar(p) && dir.exists(p)
  }, error = function(e) FALSE, warning = function(w) FALSE)
  if (!ok) testthat::skip("CmdStan not installed.")
}
