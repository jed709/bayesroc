# Re-export generics from other packages so users can call e.g. `pp_check(fit)`,
# `loo(fit)`, `posterior_epred(fit)`, `posterior_linpred(fit)` without having to
# load bayesplot / loo / rstantools alongside bayesroc.

#' @importFrom bayesplot pp_check
#' @export
bayesplot::pp_check

#' @importFrom loo loo
#' @export
loo::loo

#' @importFrom loo loo_compare
#' @export
loo::loo_compare

#' @importFrom rstantools posterior_epred
#' @export
rstantools::posterior_epred

#' @importFrom rstantools posterior_linpred
#' @export
rstantools::posterior_linpred
