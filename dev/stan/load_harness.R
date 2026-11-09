# =============================================================================
# Stan benchmark harness (development only, not shipped with the package)
# =============================================================================
# The Stan programs bayesroc used to ship, kept for the paper's performance
# comparison: for any broc_model, the program with the batch C++ likelihood
# (analytic gradients in a user header) or the Stan-language likelihood.
#
#   h <- source("dev/stan/load_harness.R")$value
#   fit <- h$stan_fit_broc(model, chains = 4)                       # batch C++ likelihood
#   model$.plain_stan <- TRUE; fit <- h$stan_fit_broc(model)       # Stan-language likelihood
#   h$get_stan_code(h$.with_stan_code(model))                       # the generated program
#
# The harness runs on top of the installed package's namespace (its model
# building, data and helpers), so build and install the package first. Needs
# cmdstanr and CmdStan.

load_stan_harness <- function(dir = file.path("dev", "stan")) {
  if (!requireNamespace("cmdstanr", quietly = TRUE)) stop("the Stan harness needs cmdstanr", call. = FALSE)
  env <- new.env(parent = asNamespace("bayesroc"))
  env$.stan_harness_dir <- normalizePath(dir, winslash = "/")
  for (f in c("generate_stan.R", "generate_batch_cpp.R", "prior_init.R", "stan_fit.R")) {
    sys.source(file.path(dir, f), envir = env)
  }
  plain <- new.env(parent = env)
  sys.source(file.path(dir, "plain_stan.R"), envir = plain)
  env$.plain_env <- plain
  env
}

load_stan_harness()
