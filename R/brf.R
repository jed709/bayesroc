#' resp() -- Multi-response marker for bivariate/CDP models
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
#' for the primary parameter (d' / mu); each additional formula's left-hand side
#' names the parameter it applies to (e.g. `criterion`, `sigma`, `lambda`).
#'
#' @param formula Two-sided formula for the primary parameter (d' / mu),
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
#' # Cumulative
#' brf(rating ~ cond, cutpoints ~ 1 + (1|subj))
#' @export
brf <- function(formula, ..., family = NULL, encoding_vars = NULL,
                 counts = NULL, cor_threshold = NULL, threads = NULL,
                 gap_link = NULL) {

  # Validate primary formula
  if (!inherits(formula, "formula")) {
    stop("First argument to brf() must be a formula.")
  }
  formula_str <- as.character(formula)
  if (length(formula_str) != 3) {
    stop("First argument to brf() must be a two-sided formula (response ~ predictors).")
  }

  # Process additional formulas from ...
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

    # Extract LHS as parameter name
    param_name <- trimws(deparse(f[[2]]))

    # Validate it's a simple name (no operators, no function calls)
    if (!grepl("^[a-zA-Z_][a-zA-Z0-9_]*$", param_name)) {
      stop("LHS of secondary formula must be a simple parameter name, got: '", param_name, "'")
    }

    # Check for duplicates
    if (param_name %in% names(params)) {
      stop("Duplicate parameter name in brf(): '", param_name, "'")
    }

    # Store as one-sided formula (just the RHS)
    rhs_formula <- as.formula(paste("~", paste(deparse(f[[3]]), collapse = " ")), env = environment(f))
    params[[param_name]] <- rhs_formula
  }

  # Validate family if provided
  if (!is.null(family) && !inherits(family, "broc_family")) {
    stop("family must be created with evsd(), uvsd(), dpsd(), mixture(), cumulative(), ",
         "bivariate_cumulative(), source_mixture(), bivariate_gaussian(), bivariate_dp(), ",
         "cdp(), or vrdp2d()")
  }

  result <- list(
    primary = formula,
    params = params,
    family = family,
    encoding_vars = encoding_vars,
    counts = counts,
    cor_threshold = cor_threshold,
    threads = threads,
    gap_link = gap_link
  )

  class(result) <- "brf"
  result
}


#' Print method for brf objects
#' @param x An brf object
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
