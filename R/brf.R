#' Multi-Response Marker for Bivariate and CDP Models
#'
#' Used inside [brf()] formulas to specify multiple response variables. The
#' arguments are positional: the specified family determines whether the second
#' variable is a source response (bivariate families) or R/K variable (CDP family).
#'
#' @param response1 The primary response variable (e.g., detection confidence)
#' @param response2 The second response variable (source confidence for
#'   bivariate families, R/K for CDP families)
#' @return Not called directly: this is a marker parsed from the (unevaluated)
#'   `brf()` formula. Invoking it directly raises an error.
#' @examples
#' # Bivariate: resp(det_conf, src_conf) | type ~ ...
#' # CDP:       resp(conf, rk_var) | old ~ ...
#' @export
resp <- function(response1, response2) {
  stop("resp() should only be used inside brf() formulas, not called directly.")
}

#' Compose a Multi-Parameter SDT Model Formula
#'
#' Formula specification for bayesroc models. The first argument is the formula
#' for the primary parameter (d'; the response alone in cumulative models); each additional formula's left-hand side
#' names the parameter it applies to (e.g. `criterion`, `sigma`, `lambda`).
#'
#' @param formula Two-sided formula for the primary parameter (d'),
#'   `response | is_old ~ predictors`. `is_old` may be coded 0/1 (new/old) or
#'   centered (-0.5/0.5); centered coding is not supported with lure mixtures or
#'   multi-response families. For multi-response families, use [resp()] on the
#'   left-hand side.
#' @param ... Additional one-sided formulas, one per parameter, with the
#'   parameter name on the left. E.g. `criterion ~ 1 + (1 | subj)`,
#'   `sigma ~ cond`, `lambda ~ 1`.
#' @param encoding_vars Character vector of column names to treat as
#'   encoding-phase (study) manipulations for which lure items do not have a
#'   meaningful value. Listing a column here auto-wraps it in `encoding()`. A
#'   value passed to [broc()] takes precedence over one set here.
#' @inheritParams broc
#' @return A `brf` object: the model specification to pass to [broc()].
#'
#' @section Random effects:
#' Group-level terms follow lme4/brms syntax. `(1 + x | g)` estimates every
#' correlation among the term's effects; `(1 + x || g)` keeps the effects of
#' different terms uncorrelated (it is shorthand for `(1 | g) + (0 + x | g)`).
#' Several terms for one grouping factor may be given; their effects are
#' correlated within a term and uncorrelated across terms. `(1 | id | g)` in
#' several parameter formulas correlates those parameters' effects.
#'
#' On the thresholds (`criterion`, `criterion2`) every term has one effect per
#' threshold component (the middle threshold and each gap), and by default those
#' are correlated with one another within the term. Wrapping the grouping factor
#' as `gr(g, cor_thresholds = FALSE)` makes the threshold components uncorrelated
#' with one another while keeping each component's correlations across terms:
#' `(1 + x | gr(g, cor_thresholds = FALSE))` correlates each threshold's
#' intercept with its own slope only, and
#' `(1 + x || gr(g, cor_thresholds = FALSE))` makes every effect independent.
#'
#' @section Location-scale criteria:
#' A `criterion_scale` formula (`cutpoints_scale` in the cumulative families) gives the
#' criteria a location and a scale, \eqn{c_j = L + e^{s} (t_j - \bar t)}, where the shape
#' \eqn{t} comes from the gap intercepts. The `criterion` formula then sets the location
#' \eqn{L}, the mean criterion: each fixed effect, random effect or smooth has one
#' coefficient and moves every criterion together. The `criterion_scale` formula sets the
#' log scale \eqn{s}: each term stretches or shrinks every gap by the same factor. Its
#' intercept is fixed at 0, because the gap intercepts already set the spread, and it takes
#' fixed effects, random effects (including `|id|` correlations with other parameters) and
#' smooths. `criterion_scale ~ 1` gives criteria that only shift. `cor_thresholds` has no
#' effect here, since each criterion term has a single component.
#'
#' `criterion2_scale` (`cutpoints2_scale` in [bivariate_cumulative()]) does the same for the
#' source criteria of the bivariate families. With varying source criteria, each set of source
#' criteria (one per detection response bin) has its own location and shape intercepts, its
#' own random effects on the location, and its own scale coefficients: every term of the
#' `criterion2_scale` formula is estimated separately for each bin.
#'
#' @examples
#' # Standard SDT
#' brf(conf | old ~ cond + (1|subj),
#'      criterion ~ 1 + (1|subj),
#'      sigma ~ 1)
#'
#' # Bivariate with resp()
#' brf(resp(det_conf, src_conf) | type ~ 1 + (1|subj),
#'      discrim ~ 1, criterion ~ 1, criterion2 ~ 1, rho ~ 1)
#'
#' # CDP with resp()
#' brf(resp(conf, rk_var) | old ~ 1 + (1|subj),
#'      fam ~ 1, criterion ~ 1, rec_crit ~ 1)
#'
#' # Location-scale criteria
#' brf(conf | old ~ cond + (1|subj),
#'      criterion ~ cond + (1|subj), criterion_scale ~ cond + (1|subj))
#'
#' # Cumulative
#' brf(rating ~ 1, cutpoints ~ cond + (1|subj), cutpoints_scale ~ 1)
#' @export
brf <- function(formula, ..., family = NULL, priors = NULL, encoding_vars = NULL,
                counts = NULL, gap_link = NULL, smooth_info = NULL) {

  if (!inherits(formula, "formula")) {
    stop("First argument to brf() must be a formula.")
  }
  formula_str <- as.character(formula)
  if (length(formula_str) != 3) {
    stop("First argument to brf() must be a two-sided formula (response ~ predictors).")
  }

  dots <- list(...)
  params <- list()

  for (i in seq_along(dots)) {
    f <- dots[[i]]
    if (!inherits(f, "formula")) {
      stop("All unnamed arguments to brf() must be formulas. Argument ", i + 1,
           " is of class '", class(f)[1], "'.")
    }

    f_str <- as.character(f)
    if (length(f_str) != 3) {
      stop("Secondary formulas in brf() must be two-sided (param_name ~ predictors). ",
           "Got one-sided formula: ", deparse(f))
    }

    param_name <- trimws(deparse(f[[2]]))

    if (!grepl("^[a-zA-Z_][a-zA-Z0-9_]*$", param_name)) {
      stop("LHS of secondary formula must be a simple parameter name, got: '", param_name, "'")
    }

    if (param_name %in% names(params)) {
      stop("Duplicate parameter name in brf(): '", param_name, "'")
    }

    rhs_formula <- as.formula(paste("~", paste(deparse(f[[3]]), collapse = " ")), env = environment(f))
    params[[param_name]] <- rhs_formula
  }

  if (!is.null(family) && !inherits(family, "broc_family")) {
    stop("family must be created with evsd(), uvsd(), dpsd(), mixture(), cumulative(), ",
         "bivariate_cumulative(), source_mixture(), bivariate_gaussian(), bivariate_dp(), ",
         "cdp(), or vrdp2d()")
  }

  result <- list(
    primary = formula,
    params = params,
    family = family,
    priors = priors,
    encoding_vars = encoding_vars,
    counts = counts,
    gap_link = gap_link,
    smooth_info = smooth_info
  )

  class(result) <- "brf"
  result
}

#' Print method for brf objects
#' @param x A `brf` object.
#' @param ... Ignored
#' @export
print.brf <- function(x, ...) {
  cat("brf() formula composition\n")
  cat("==========================\n")
  cat("Primary: ", deparse(x$primary), "\n")

  if (length(x$params) > 0) {
    for (nm in names(x$params)) {
      cat("  ", nm, ": ", deparse(x$params[[nm]]), "\n")
    }
  }

  if (!is.null(x$family)) {
    cat("Family:  ", family_display_name(x$family$family), "\n")
  }
  if (!is.null(x$encoding_vars)) {
    cat("Encoding variables: ", paste(x$encoding_vars, collapse = ", "), "\n")
  }
  if (!is.null(x$counts)) {
    cat("Counts: ", x$counts, "\n")
  }

  invisible(x)
}
