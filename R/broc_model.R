#' SDT Model Specification and Fitting
#'
#' Main interface for specifying and fitting SDT models.

#' Check if a formula is equivalent to ~ NULL
#' @param x Object to check
#' @return TRUE if x is ~ NULL or equivalent
#' @noRd
identical_to_null_formula <- function(x) {
  if (!inherits(x, "formula")) return(FALSE)
  f_str <- paste(deparse(x), collapse = "")
  grepl("^~\\s*NULL\\s*$", f_str, ignore.case = TRUE)
}

#' Validate that the data contains the columns referenced by every formula
#'
#' Catches the common build-time mistakes (typo'd column name in a formula,
#' missing counts column, encoding factor not in data, data not a data.frame)
#' at broc() with an actionable message, instead of letting them surface as
#' a confusing Stan compile error or an internal indexing crash.
#'
#' @param data User's data argument
#' @param brf_obj Already-validated brf object
#' @param family broc_family
#' @param encoding_vars,counts user args from broc()
#' @noRd
validate_broc_data <- function(data, brf_obj, family, encoding_vars, counts) {

  if (is.null(data)) {
    stop("`data` argument is required.", call. = FALSE)
  }
  if (!is.data.frame(data)) {
    stop("`data` must be a data.frame, got ", paste(class(data), collapse = "/"),
         ".", call. = FALSE)
  }
  if (nrow(data) == 0L) {
    stop("`data` has 0 rows.", call. = FALSE)
  }

  formula_vars <- character(0)
  formula_sources <- character(0)
  add_formula_vars <- function(f, label) {
    if (is.null(f) || identical_to_null_formula(f)) return(invisible())
    vs <- tryCatch(all.vars(f), error = function(e) character(0))
    f_str <- paste(deparse(f), collapse = " ")
    cor_ids <- regmatches(f_str,
                          gregexpr("\\|\\s*([A-Za-z_.][A-Za-z0-9_.]*)\\s*\\|",
                                   f_str))[[1]]
    if (length(cor_ids) > 0) {
      cor_ids <- gsub("\\|", "", cor_ids)
      cor_ids <- trimws(cor_ids)
      vs <- setdiff(vs, cor_ids)
    }
    if (length(vs) > 0) {
      formula_vars   <<- c(formula_vars, vs)
      formula_sources <<- c(formula_sources, rep(label, length(vs)))
    }
  }
  add_formula_vars(brf_obj$primary, "primary")
  for (pname in names(brf_obj$params)) {
    add_formula_vars(brf_obj$params[[pname]], pname)
  }

  cols <- names(data)
  missing_idx <- which(!formula_vars %in% cols)
  missing_idx <- missing_idx[formula_vars[missing_idx] != ".cumulative_is_old"]

  if (length(missing_idx) > 0) {
    miss <- unique(data.frame(var = formula_vars[missing_idx],
                              src = formula_sources[missing_idx],
                              stringsAsFactors = FALSE))
    miss_lines <- vapply(seq_len(nrow(miss)), function(i) {
      sprintf("  - '%s' (referenced in %s formula)", miss$var[i], miss$src[i])
    }, character(1))
    stop("The following variable(s) referenced in your formulas are not in `data`:\n",
         paste(miss_lines, collapse = "\n"), call. = FALSE)
  }

  if (!is.null(counts)) {
    if (!is.character(counts) || length(counts) != 1) {
      stop("`counts` must be a single column name (character).", call. = FALSE)
    }
    if (!counts %in% cols) {
      stop("`counts = \"", counts, "\"` but '", counts,
           "' is not a column in `data`.", call. = FALSE)
    }
    cv <- data[[counts]]
    if (!is.numeric(cv) || any(is.na(cv)) || any(cv < 0) || any(abs(cv - round(cv)) > 1e-8)) {
      stop("`counts` column '", counts, "' must be non-negative whole numbers (no NAs).",
           call. = FALSE)
    }
  }

  if (!is.null(encoding_vars)) {
    miss_enc <- setdiff(encoding_vars, cols)
    if (length(miss_enc) > 0) {
      stop("`encoding_vars` references column(s) not in `data`: ",
           paste(miss_enc, collapse = ", "), call. = FALSE)
    }
  }

  prim <- brf_obj$primary
  if (!is.null(prim) && length(as.character(prim)) >= 2) {
    lhs_str <- deparse(prim[[2]])
    if (grepl("\\|", lhs_str) && family$family %in%
        c("evsdt", "uvsdt", "dpsdt", "mixture", "cdp")) {
      bar_var <- trimws(strsplit(lhs_str, "\\|")[[1]][2])
      if (!grepl("^[a-zA-Z_.][a-zA-Z0-9_.]*$", bar_var)) {
      } else if (bar_var %in% cols) {
        v <- data[[bar_var]]
        if (is.numeric(v) || is.logical(v)) {
          uvals <- unique(v[!is.na(v)])
          if (!all(uvals %in% c(0, 1)) && !all(uvals %in% c(-0.5, 0.5))) {
            stop("Column '", bar_var, "' (used as is_old indicator) must be ",
                 "coded 0/1 (new/old) or -0.5/0.5 (centered). Got: ",
                 paste(head(uvals, 6), collapse = ", "), ".", call. = FALSE)
          }
        }
      }
    }
  }

  invisible(TRUE)
}

.as_is_old <- function(v) {
  suppressWarnings(if (is.factor(v)) as.numeric(as.character(v)) else as.numeric(v))
}

#' Specify a Bayesian SDT Model
#'
#' Builds a model specification from a formula, family, and (optional) priors,
#' and generates the backend code needed to fit it. `fit_broc()` is used to fit
#' the object created by `broc()`.
#'
#' @param formula A [brf()] object, or a bare two-sided formula
#'   `response | is_old ~ predictors`.
#' @param data A data frame containing the response, the `is_old` indicator, and
#'   every predictor and grouping factor named in `formula`.
#' @param family The SDT model family: one of [evsd()], [uvsd()], [dpsd()],
#'   [mixture()], [source_mixture()], [bivariate_gaussian()], [bivariate_dp()],
#'   [vrdp2d()], [cdp()], [cumulative()], or [bivariate_cumulative()]. May also be
#'   set in [brf()]; a value in broc() takes precedence.
#' @param priors A prior specification from [broc_prior()]. If `NULL`, the
#'   family's weakly-informative default priors are used (inspect them with
#'   [get_broc_prior()]).
#' @param encoding_vars Character vector of column names to treat as
#'   encoding-phase (study) manipulations for which lure items do not have a
#'   meaningful value. Listing a column here auto-wraps it in `encoding()`. May
#'   also be set in [brf()]; a value in broc() takes precedence.
#' @param counts Name of a column giving per-row trial counts, for aggregated
#'   (count) data rather than one row per trial. May also be set in [brf()].
#' @param gap_link Link for threshold gaps: `"log"` (default) or `"softplus"`.
#'   `NULL` uses the family default.
#' @param smooth_info If `TRUE`, sample the coefficients of population-level
#'   smooth terms on a scale matched to how strongly the data inform them. This
#'   is an exact change of variables, so the posterior is unchanged; it can remove
#'   divergences when the smooths' SDs are weakly identified. Model construction
#'   runs a short maximum-likelihood pilot to measure that information. Random smooths (`bs = "fs"`, `"re"`) keep the
#'   default parameterization. Default `FALSE`.
#' @section Data coding:
#' \describe{
#'   \item{Responses}{Whole numbers or an ordered factor. Every value from the
#'     lowest to the highest observed is a response category, so an unused middle
#'     category is kept; responses coded from 0 or from a value above 1 are
#'     re-based to start at 1 (with a message). Text labels need an ordered factor
#'     whose level order is the response scale.}
#'   \item{Old/new indicator}{0/1 (new/old), logical, or -0.5/0.5 (centered; not
#'     for [cdp()]). Text labels such as "old"/"new" are rejected.}
#'   \item{Source (source_mixture)}{A two-level factor (first level = source A),
#'     0/1, 1/2 or logical.}
#'   \item{Item type (bivariate families)}{Codes 1/2/3 or 0/1/2 (new, source A,
#'     source B), or a factor whose levels are in that order.}
#'   \item{Remember/know (cdp)}{"remember"/"know"/"guess", "R"/"K"/"G" or 1/2/3;
#'     may be missing on any response (that response then enters through its
#'     confidence rating alone).}
#'   \item{Counts}{Non-negative whole numbers; rows with a count of zero are
#'     dropped.}
#'   \item{Missing values}{Rows with a missing value in any variable the model
#'     uses are dropped with a message; their row numbers are stored in
#'     `model$dropped_rows`. Missing source ratings in bivariate models are an
#'     error.}
#' }
#'
#' @return A `broc_model` object, ready to pass to [fit_broc()].
#'
#' @examples
#' \dontrun{
#' # Unequal-variance model (construction only; pass to fit_broc() to fit)
#' m <- broc(brf(conf | old ~ cond + (1 | subj),
#'               criterion ~ 1 + (1 | subj),
#'               sigma ~ 1,
#'               family = uvsd()),
#'           data = dat)
#'
#' # Bare-formula shorthand (criterion defaults to ~ 1)
#' broc(conf | old ~ cond, data = dat, family = evsd())
#' }
#' @export
broc <- function(formula = NULL, data = NULL, family = NULL, priors = NULL,
                encoding_vars = NULL, counts = NULL,
                gap_link = NULL, smooth_info = NULL) {

  # --- 1. Normalize input: bare formula -> brf, brf -> use directly ---
  nc <- function(a, b) if (!is.null(a)) a else b

  if (inherits(formula, "brf")) {
    brf_obj <- formula
  } else if (inherits(formula, "formula")) {
    brf_obj <- brf(formula)
  } else if (is.null(formula)) {
    stop("formula argument is required. Provide a formula or brf() object.")
  } else {
    stop("First argument must be a formula or brf() object.")
  }

  if (inherits(priors, "formula")) {
    stop("broc() accepts a single formula. For a multi-parameter model, wrap the ",
         "formulas in brf(), e.g. broc(brf(resp | is_old ~ ..., criterion ~ ...), ",
         "data, family = ...).", call. = FALSE)
  }

  # --- 2. Merge options: broc() non-NULL wins, else brf value, else defaults ---
  family <- nc(family, nc(brf_obj$family, evsd()))
  encoding_vars <- nc(encoding_vars, brf_obj$encoding_vars)
  counts <- nc(counts, brf_obj$counts)
  smooth_info <- nc(smooth_info, nc(brf_obj$smooth_info, FALSE))
  priors <- nc(priors, brf_obj$priors)
  gap_link <- match.arg(nc(gap_link, nc(brf_obj$gap_link, "log")),
                         c("log", "softplus"))
  if (!inherits(family, "broc_family")) {
    stop("family must be created with evsd(), uvsd(), dpsd(), mixture(), cumulative(), bivariate_cumulative(), source_mixture(), bivariate_gaussian(), bivariate_dp(), cdp(), or vrdp2d()")
  }


  # --- 2b. Up-front data validation ----------------------------------------
  validate_broc_data(data, brf_obj, family, encoding_vars, counts)
  nd <- .drop_incomplete_rows(data, brf_obj, family, encoding_vars, counts)
  data <- .drop_unused_levels(nd$data, brf_obj, encoding_vars)
  dropped_rows <- nd$dropped

  # --- 2c. Drop zero-count rows ---------------------------------------------
  if (!is.null(counts)) {
    cv <- data[[counts]]
    zero_rows <- which(cv == 0)
    if (length(zero_rows) > 0) {
      message("Dropping ", length(zero_rows), " row(s) with count = 0")
      data <- data[-zero_rows, , drop = FALSE]
      rownames(data) <- NULL
    }
  }

  # --- 3. Unpack formulas from brf ---
  dprime <- brf_obj$primary
  params <- brf_obj$params
  gx <- .expand_grouping(c(list(.primary = dprime), params), data)
  dprime <- gx$formulas$.primary
  params <- gx$formulas[names(params)]
  data <- gx$data

  # --- 3a. Reject formulas naming params not in the family ---
  valid_names <- names(family$params)
  if (!is.null(family$external_aliases)) {
    valid_names <- c(valid_names,
                     unlist(family$external_aliases, use.names = FALSE))
  }
  bad <- setdiff(names(params), valid_names)
  if (length(bad) > 0) {
    stop(sprintf("Parameter%s '%s' not valid for family '%s'. Valid: %s.",
                 if (length(bad) == 1) "" else "s",
                 paste(bad, collapse = "', '"),
                 family_display_name(family$family),
                 paste(sort(valid_names), collapse = ", ")),
         call. = FALSE)
  }

  params <- .canonical_param_formulas(params, family)

  criterion  <- params[["criterion"]]
  sigma      <- params[["sigma"]]
  lambda     <- params[["lambda"]]
  dprime2    <- params[["dprime2"]]
  sigma2     <- params[["sigma2"]]
  dprime_B   <- params[["dprime_B"]]
  lambda_B   <- params[["lambda_B"]]
  discrim    <- params[["discrim"]]
  discrim_B  <- params[["discrim_B"]]
  sigma_B    <- params[["sigma_B"]]
  sigma2_B   <- params[["sigma2_B"]]
  rho        <- params[["rho"]]
  rho_B      <- params[["rho_B"]]
  rho_N      <- params[["rho_N"]]
  criterion2 <- params[["criterion2"]]
  rec_crit   <- params[["rec_crit"]]
  know_crit  <- params[["know_crit"]]
  dprime_L   <- params[["dprime_L"]]
  sigma_L    <- params[["sigma_L"]]
  lambda_L   <- params[["lambda_L"]]
  lambda2    <- params[["lambda2"]]
  lambda2_B  <- params[["lambda2_B"]]
  criterion_scale <- params[["criterion_scale"]] %||% params[["cutpoints_scale"]]
  criterion2_scale <- params[["criterion2_scale"]] %||% params[["cutpoints2_scale"]]

  if (!is.null(params[["cutpoints"]]) && is.null(criterion)) {
    criterion <- params[["cutpoints"]]
  }

  if (family$family == "bivariate_cumulative") {
    if (!is.null(params[["mu2"]])) discrim <- params[["mu2"]]
    if (!is.null(params[["cutpoints2"]])) criterion2 <- params[["cutpoints2"]]
  }

  if (family$family %in% c("cdp")) {
    if (!is.null(params[["fam"]])) dprime2 <- params[["fam"]]
    if (!is.null(params[["sigma_R"]])) sigma <- params[["sigma_R"]]
    if (!is.null(params[["sigma_F"]])) sigma2 <- params[["sigma_F"]]
  }

  encoding_supported_families <- c("evsdt", "uvsdt", "dpsdt", "mixture", "cdp", "bivariate_sdt", "bivariate_dp", "vrdp2d")

  if (!is.null(encoding_vars) && !family$family %in% encoding_supported_families) {
    if (family$family == "cumulative") {
      stop("encoding_vars cannot be used with cumulative() family. ",
           "Cumulative models do not have an old/new distinction.")
    } else if (family$family == "source_mixture") {
      stop("encoding_vars cannot be used with source_mixture() family. ",
           "Source mixture models use source (A/B) rather than old/new.")
    } else {
      stop("encoding_vars is not supported for ", family_display_name(family$family), " family.")
    }
  }
  enc_in_crit <- intersect(encoding_vars, unlist(lapply(list(criterion, criterion2), function(f)
    if (inherits(f, "formula")) all.vars(f))))
  if (length(enc_in_crit)) {
    warning("Encoding variable in criterion formula: ", paste(enc_in_crit, collapse = ", "), call. = FALSE)
  }
  for (lp in c("rho_N", "dprime_L", "sigma_L", "lambda_L")) {
    f <- params[[lp]]
    if (!inherits(f, "formula")) next
    enc <- unique(c(intersect(encoding_vars, all.vars(f)),
                    if (grepl("encoding\\s*\\(", paste(deparse(f), collapse = ""))) "encoding()"))
    if (length(enc)) {
      stop(lp, " describes new items only, which have no encoding condition, so it cannot depend on an ",
           "encoding variable (", paste(enc, collapse = ", "), ").", call. = FALSE)
    }
  }

  # --- 4. Handle cumulative family early-return ---
  if (family$family == "cumulative") {
    res <- fit_cumulative(mu = dprime, cutpoints = criterion, data = data,
                          family = family, priors = priors, counts = counts,
                          gap_link = gap_link, smooth_info = smooth_info,
                          cutpoints_scale = criterion_scale)
    res$dropped_rows <- dropped_rows
    .check_priors_matched(res, priors)
    return(res)
  }

  # --- Handle bivariate_cumulative family early-return ---
  if (family$family == "bivariate_cumulative") {
    res <- fit_bivariate_cumulative(
      mu1 = dprime, mu2 = discrim,
      cutpoints = criterion, cutpoints2 = criterion2,
      rho = rho, data = data, family = family,
      priors = priors, counts = counts,
      gap_link = gap_link, smooth_info = smooth_info,
      cutpoints_scale = criterion_scale, cutpoints2_scale = criterion2_scale)
    res$dropped_rows <- dropped_rows
    .check_priors_matched(res, priors)
    return(res)
  }

  if (is.null(criterion)) {
    message("Note: criterion formula not specified. Using criterion ~ 1.")
    criterion <- ~ 1
  }

  # --- 5. CDP validation ---
  if (family$family == "cdp") {
    if (is.null(dprime2)) {
      stop("fam formula is required for CDP family")
    }
    if (is.null(rec_crit)) {
      stop("rec_crit formula is required for CDP family")
    }
  }
  if (!is.null(encoding_vars)) {
    dprime <- apply_encoding_vars(dprime, encoding_vars)
    if (!is.null(sigma)) sigma <- apply_encoding_vars(sigma, encoding_vars)
    if (!is.null(lambda)) lambda <- apply_encoding_vars(lambda, encoding_vars)
    if (!is.null(dprime2)) dprime2 <- apply_encoding_vars(dprime2, encoding_vars)
    if (!is.null(sigma2)) sigma2 <- apply_encoding_vars(sigma2, encoding_vars)
    if (!is.null(dprime_B)) dprime_B <- apply_encoding_vars(dprime_B, encoding_vars)
    if (!is.null(lambda_B)) lambda_B <- apply_encoding_vars(lambda_B, encoding_vars)
    if (!is.null(discrim)) discrim <- apply_encoding_vars(discrim, encoding_vars)
    if (!is.null(discrim_B)) discrim_B <- apply_encoding_vars(discrim_B, encoding_vars)
    if (!is.null(sigma_B)) sigma_B <- apply_encoding_vars(sigma_B, encoding_vars)
    if (!is.null(sigma2_B)) sigma2_B <- apply_encoding_vars(sigma2_B, encoding_vars)
    if (!is.null(rho)) rho <- apply_encoding_vars(rho, encoding_vars)
    if (!is.null(rho_B)) rho_B <- apply_encoding_vars(rho_B, encoding_vars)
    if (!is.null(rho_N)) rho_N <- apply_encoding_vars(rho_N, encoding_vars)
  }

  sigma_is_fixed <- is.null(sigma) || identical_to_null_formula(sigma)
  lambda_is_fixed <- is.null(lambda) || identical_to_null_formula(lambda)
  dprime2_is_fixed <- is.null(dprime2) || identical_to_null_formula(dprime2)
  sigma2_is_fixed <- is.null(sigma2) || identical_to_null_formula(sigma2)
  dprime_B_is_fixed <- is.null(dprime_B) || identical_to_null_formula(dprime_B)
  lambda_B_is_fixed <- is.null(lambda_B) || identical_to_null_formula(lambda_B)
  discrim_is_fixed <- is.null(discrim) || identical_to_null_formula(discrim)
  discrim_B_is_fixed <- is.null(discrim_B) || identical_to_null_formula(discrim_B)
  sigma_B_is_fixed <- is.null(sigma_B) || identical_to_null_formula(sigma_B)
  sigma2_B_is_fixed <- is.null(sigma2_B) || identical_to_null_formula(sigma2_B)
  rho_is_fixed <- is.null(rho) || identical_to_null_formula(rho)
  rho_B_is_fixed <- is.null(rho_B) || identical_to_null_formula(rho_B)
  rho_N_is_fixed <- is.null(rho_N) || identical_to_null_formula(rho_N)
  criterion2_is_fixed <- is.null(criterion2) || identical_to_null_formula(criterion2)

  if (sigma_is_fixed) sigma <- NULL
  if (lambda_is_fixed) lambda <- NULL
  if (dprime2_is_fixed) dprime2 <- NULL
  if (sigma2_is_fixed) sigma2 <- NULL
  if (dprime_B_is_fixed) dprime_B <- NULL
  if (lambda_B_is_fixed) lambda_B <- NULL
  if (discrim_is_fixed) discrim <- NULL
  if (discrim_B_is_fixed) discrim_B <- NULL
  if (sigma_B_is_fixed) sigma_B <- NULL
  if (sigma2_B_is_fixed) sigma2_B <- NULL
  if (rho_is_fixed) rho <- NULL
  if (rho_B_is_fixed) rho_B <- NULL
  if (rho_N_is_fixed) rho_N <- NULL
  if (criterion2_is_fixed) criterion2 <- NULL

  internal_arg_map <- list(
    dprime     = dprime,     dprime2    = dprime2,
    criterion  = criterion,  criterion2 = criterion2,
    sigma      = sigma,      sigma2     = sigma2,
    lambda     = lambda,     lambda2    = lambda2,
    lambda2_B  = lambda2_B,
    rec_crit   = rec_crit,   know_crit  = know_crit,
    dprime_B   = dprime_B,   lambda_B   = lambda_B,
    discrim    = discrim,    discrim_B  = discrim_B,
    sigma_B    = sigma_B,    sigma2_B   = sigma2_B,
    rho        = rho,        rho_B      = rho_B,
    rho_N      = rho_N,
    criterion_scale = criterion_scale,
    criterion2_scale = criterion2_scale
  )
  provided_params <- list()
  for (pname in names(family$params)) {
    if (pname %in% names(internal_arg_map)) {
      provided_params[pname] <- list(internal_arg_map[[pname]])
    }
  }

  validate_family_params(family, provided_params)
  
  if (family$family == "uvsdt" && is.null(sigma)) {
    message("Note: uvsd family specified but no sigma formula provided. Using sigma ~ 1.")
    sigma <- ~ 1
  }
  
  if (family$family == "dpsdt" && is.null(lambda)) {
    stop("lambda formula is required for family 'dpsd'")
  }
  if (family$family == "mixture" && is.null(lambda) && is.null(lambda_L)) {
    stop("mixture family requires either lambda (signal mixture) or lambda_L + dprime_L (lure mixture)")
  }
  
  if (family$family == "source_mixture") {
    if (is.null(lambda)) {
      stop("lambda formula is required for source_mixture family")
    }
    if (is.null(dprime_B)) {
      message("Note: dprime_B not specified - constraining dprime_B = -dprime")
    }
    if (is.null(lambda_B)) {
      message("Note: lambda_B not specified - constraining lambda_B = lambda")
    }
  }
  
  if (family$family == "bivariate_sdt") {
    if (is.null(discrim)) {
      stop("discrim formula is required for bivariate_gaussian family")
    }
    if (is.null(rho)) {
      stop("rho formula is required for bivariate_gaussian family")
    }
    if (is.null(criterion2)) {
      stop("criterion2 formula is required for bivariate_gaussian family")
    }
    
    if (is.null(dprime_B)) {
      message("Note: dprime_B not specified - constraining dprime_B = dprime")
    }
    if (is.null(discrim_B)) {
      message("Note: discrim_B not specified - constraining discrim_B = -discrim")
    }
    if (is.null(sigma)) {
      message("Note: sigma not specified - fixing sigma = 1")
    }
    if (is.null(sigma_B)) {
      if (!is.null(sigma)) {
        message("Note: sigma_B not specified - constraining sigma_B = sigma")
      } else {
        message("Note: sigma_B not specified - fixing sigma_B = 1")
      }
    }
    if (is.null(sigma2)) {
      message("Note: sigma2 not specified - fixing sigma2 = 1")
    }
    if (is.null(sigma2_B)) {
      if (!is.null(sigma2)) {
        message("Note: sigma2_B not specified - constraining sigma2_B = sigma2")
      } else {
        message("Note: sigma2_B not specified - fixing sigma2_B = 1")
      }
    }
    if (is.null(rho_B)) {
      message("Note: rho_B not specified - constraining rho_B = -rho")
    }
    if (is.null(rho_N)) {
      message("Note: rho_N not specified - fixing rho_N = 0")
    }
  }

  if (family$family == "bivariate_dp") {
    if (is.null(discrim)) {
      stop("discrim formula is required for bivariate_dp family")
    }
    if (is.null(rho)) {
      stop("rho formula is required for bivariate_dp family")
    }
    if (is.null(criterion2)) {
      stop("criterion2 formula is required for bivariate_dp family")
    }
    if (is.null(lambda)) {
      stop("lambda formula is required for bivariate_dp family (item recollection probability)")
    }
    if (is.null(lambda2)) {
      stop("lambda2 formula is required for bivariate_dp family (source recollection probability)")
    }

    if (is.null(dprime_B)) {
      message("Note: dprime_B not specified - constraining dprime_B = dprime")
    }
    if (is.null(discrim_B)) {
      message("Note: discrim_B not specified - constraining discrim_B = -discrim")
    }
    if (is.null(sigma)) {
      message("Note: sigma not specified - fixing sigma = 1")
    }
    if (is.null(sigma_B)) {
      if (!is.null(sigma)) {
        message("Note: sigma_B not specified - constraining sigma_B = sigma")
      } else {
        message("Note: sigma_B not specified - fixing sigma_B = 1")
      }
    }
    if (is.null(sigma2)) {
      message("Note: sigma2 not specified - fixing sigma2 = 1")
    }
    if (is.null(sigma2_B)) {
      if (!is.null(sigma2)) {
        message("Note: sigma2_B not specified - constraining sigma2_B = sigma2")
      } else {
        message("Note: sigma2_B not specified - fixing sigma2_B = 1")
      }
    }
    if (is.null(rho_B)) {
      message("Note: rho_B not specified - constraining rho_B = -rho")
    }
    if (is.null(rho_N)) {
      message("Note: rho_N not specified - fixing rho_N = 0")
    }
    if (is.null(lambda_B)) {
      message("Note: lambda_B not specified - constraining lambda_B = lambda")
    }
    if (is.null(lambda2_B)) {
      message("Note: lambda2_B not specified - constraining lambda2_B = lambda2")
    }
  }

  if (family$family == "vrdp2d") {
    if (is.null(dprime2)) {
      stop("dprime2 formula is required for vrdp2d family (interpreted as d'_R recollection boost)")
    }
    if (is.null(discrim)) {
      stop("discrim formula is required for vrdp2d family (interpreted as d'_S source discriminability)")
    }
    if (is.null(lambda)) {
      stop("lambda formula is required for vrdp2d family (interpreted as R recollection probability)")
    }
    if (is.null(criterion2)) {
      stop("criterion2 formula is required for vrdp2d family (source dimension thresholds)")
    }
    if (is.null(discrim_B)) {
      message("Note: discrim_B not specified - constraining discrim_B = -discrim")
    }

    if (is.null(sigma)) {
      message("Note: sigma not specified - fixing sigma = 1")
    }
    if (is.null(sigma2)) {
      message("Note: sigma2 not specified - fixing sigma2 = 1")
    }
  }

  parsed <- parse_broc_formula(
    dprime = dprime,
    criterion = criterion,
    sigma = sigma,
    lambda = lambda,
    dprime2 = dprime2,
    sigma2 = sigma2,
    dprime_B = dprime_B,
    lambda_B = lambda_B,
    discrim = discrim,
    discrim_B = discrim_B,
    sigma_B = sigma_B,
    sigma2_B = sigma2_B,
    rho = rho,
    rho_B = rho_B,
    rho_N = rho_N,
    criterion2 = criterion2,
    rec_crit = rec_crit,
    know_crit = know_crit,
    dprime_L = dprime_L,
    sigma_L = sigma_L,
    lambda_L = lambda_L,
    lambda2 = lambda2,
    lambda2_B = lambda2_B,
    criterion_scale = criterion_scale,
    criterion2_scale = criterion2_scale,
    data = data,
    family_name = family$family
  )
  
  prior_lookup <- build_prior_lookup(priors, family = family)
  
  model_data <- build_model_data(parsed, data, family, counts = counts,
                                 encoding_vars = encoding_vars)
  model_data$gap_link <- gap_link
  prior_lookup <- .finish_prior_lookup(prior_lookup, model_data)
  model_data$prior_lookup <- prior_lookup
  model_data$gap_link <- gap_link

  is_old_var <- parsed$response$is_old
  if (!is.null(is_old_var) && is_old_var %in% names(data) &&
      !is.null(model_data$stan_data$is_old) &&
      length(model_data$stan_data$is_old) == nrow(data)) {
    data[[is_old_var]] <- as.numeric(model_data$stan_data$is_old)
  }

  if (!is.null(model_data$old_levels) && (is.null(family$old_levels) || identical(family$family, "cdp"))) {
    family$old_levels <- model_data$old_levels
    family$J <- length(model_data$old_levels)
  }
  if (!is.null(model_data$rk_levels)) family$rk_levels <- model_data$rk_levels
  if (!is.null(model_data$stan_data$n_rkg) && is.null(family$n_rkg)) {
    family$n_rkg <- as.integer(model_data$stan_data$n_rkg)
  }

  si <- .apply_smooth_info(smooth_info, model_data, family, gap_link, data)
  model_data <- si$model_data

  code <- .generate_model_code(model_data, family, "standard")

  result <- list(
    parsed = parsed,
    model_data = model_data,
    stan_code = code$stan_code,
    stan_data = code$stan_data,
    family = family,
    prior_lookup = prior_lookup,
    data = data,
    gap_link = gap_link,
    formula = brf_obj,
    batch_likelihood = code$batch_likelihood,
    batch_cpp = code$batch_cpp,
    smooth_info = si$pilot,
    dropped_rows = dropped_rows
  )

  class(result) <- c("broc_model", "list")
  .check_priors_matched(result, priors)
  result
}

#' Build All Model Data Structures
#'
#' @param parsed Parsed formula
#' @param data Data frame
#' @param family Model family
#' @param counts Optional column name for count weights
#' @return List with all model data
#' @noRd
.rkg_knob_data <- function(stan_data, re_in_bound) {
  # the know-criterion slack transform is used only when the Guess bound is fixed-effects
  # only: with random effects on the thresholds or criteria the bound couples every
  # effect through a max over trials and sampling collapses (measured, Oct 2026)
  if (!isTRUE(stan_data$n_rkg == 3) || is.null(stan_data$X_know_crit)) return(stan_data)
  stan_data$kc_knob <- 0L
  if (isTRUE(re_in_bound)) return(stan_data)
  band <- match(stan_data$y, stan_data$old_level_map)
  live <- if (is.null(stan_data$counts)) rep(TRUE, length(band)) else stan_data$counts > 0
  g <- which(!is.na(band) & stan_data$rk == 3 & live)
  stan_data$kc_knob <- as.integer(length(g) > 0)
  if (!length(g)) return(stan_data)
  X <- as.matrix(stan_data$X_know_crit)
  sv <- svd(X)
  keep <- sv$d > max(sv$d) * 1e-10
  v <- drop(sv$v[, keep, drop = FALSE] %*% ((t(sv$u[, keep, drop = FALSE]) %*% rep(1, nrow(X))) / sv$d[keep]))
  P <- ncol(X)
  stan_data$N_guess <- length(g)
  stan_data$guess_idx <- as.array(as.integer(g))
  stan_data$guess_band <- as.array(as.integer(band[g]))
  stan_data$kc_v <- as.array(v)
  stan_data$kc_Bperp <- if (P > 1) qr.Q(qr(matrix(v, ncol = 1)), complete = TRUE)[, -1, drop = FALSE]
                        else matrix(0, nrow = 1, ncol = 0)
  stan_data
}

.check_criterion_designs <- function(stan_data, family_name = "evsdt", parsed = NULL) {
  cum <- family_name %in% c("cumulative", "bivariate_cumulative")
  labels <- if (cum) c(X_criterion = "cutpoints", X_criterion2 = "cutpoints2")
            else c(X_criterion = "criterion", X_criterion2 = "criterion2", X_rec_crit = "rec_crit", X_know_crit = "know_crit")
  for (nm in names(labels)) {
    X <- stan_data[[nm]]
    if (is.null(X) || !length(X)) next
    X <- as.matrix(X)
    v <- qr.coef(qr(X), rep(1, nrow(X)))
    v[is.na(v)] <- 0
    if (max(abs(X %*% v - 1)) > 1e-8) {
      cols <- colnames(X)
      stop("`", labels[[nm]], "` has no intercept: with ", paste(cols, collapse = ", "),
           " the ", if (cum) "cutpoints are" else "criteria are", " fixed at 0 wherever the covariate is 0. ",
           "Add the factor's own term for per-level intercepts (e.g. `f + f:x - 1`) or use `f * x`.", call. = FALSE)
    }
  }
  invisible(NULL)
}

build_model_data <- function(parsed, data, family, counts = NULL,
                             encoding_vars = NULL) {
  
  N <- nrow(data)
  resp_col <- parsed$response$response
  rc <- .code_response(data[[resp_col]], resp_col)
  data[[resp_col]] <- rc$y
  K <- rc$K
  response_labels <- rc$labels
  resp_offset <- as.integer(rc$offset)
  if (resp_offset != 0L && !is.null(family$old_levels)) family$old_levels <- family$old_levels - resp_offset
  is_old_var <- parsed$response$is_old
  source_var <- parsed$response$source
  item_type_var <- parsed$response$item_type
  rk_var <- parsed$response$rk
  encoding_vars <- unique(c(parsed$encoding_vars, encoding_vars))
  is_cumulative <- family$family == "cumulative"
  is_bivariate_cumulative <- family$family == "bivariate_cumulative"
  is_source_mixture <- family$family == "source_mixture"
  is_bivariate <- family$family %in% c("bivariate_sdt", "bivariate_dp", "bivariate_cumulative")
  is_vrdp2d <- family$family == "vrdp2d"
  is_cdp <- family$family == "cdp"
  is_bivariate_dp <- family$family == "bivariate_dp"
  if ((is_bivariate || is_vrdp2d) && !is.null(item_type_var) && !startsWith(item_type_var, ".")) {
    data[[item_type_var]] <- .code_item_type(data[[item_type_var]], item_type_var)
  }
  if (is_source_mixture && !is.null(source_var)) {
    data[[source_var]] <- .code_source(data[[source_var]], source_var)
  }
  if ((is_bivariate || is_vrdp2d) && !is.null(item_type_var)) {
    item_type_vals <- data[[item_type_var]]
    if (is.factor(item_type_vals)) {
      is_old_for_encoding <- as.integer(item_type_vals) > 1
    } else {
      is_old_for_encoding <- as.integer(item_type_vals) > 1
    }
    data$.is_old_bivariate <- as.integer(is_old_for_encoding)
    is_old_var <- ".is_old_bivariate"
  }
  
  K2 <- NULL
  response2_labels <- NULL
  if ((is_bivariate || is_vrdp2d) && !is.null(parsed$response2)) {
    resp2_col <- parsed$response2
    if (!resp2_col %in% names(data)) {
      stop("Second response variable '", resp2_col, "' not found in data. ",
           "Check your resp() specification: resp(response1, response2)")
    }
    rc2 <- .code_response(data[[resp2_col]], resp2_col)
    data[[resp2_col]] <- rc2$y
    K2 <- rc2$K
    response2_labels <- rc2$labels
    if (K2 < 2) {
      stop("Second response variable '", resp2_col, "' must have at least 2 unique values, got ", K2)
    }
  }
  
  is_vrdp2d <- family$family == "vrdp2d"
  has_sigma <- !is.null(parsed[["sigma"]]) && family$family %in% c("uvsdt", "dpsdt", "mixture", "bivariate_sdt", "bivariate_dp", "vrdp2d", "cdp")
  has_lambda <- !is.null(parsed[["lambda"]]) && family$family %in% c("dpsdt", "mixture", "source_mixture", "vrdp2d", "bivariate_dp")
  has_dprime2 <- !is.null(parsed[["dprime2"]]) && family$family %in% c("mixture", "cdp", "vrdp2d")
  has_sigma2 <- !is.null(parsed[["sigma2"]]) && family$family %in% c("mixture", "bivariate_sdt", "bivariate_dp", "cdp", "vrdp2d")
  has_dprime_B <- !is.null(parsed[["dprime_B"]]) && family$family %in% c("source_mixture", "bivariate_sdt", "bivariate_dp")
  has_lambda_B <- !is.null(parsed[["lambda_B"]]) && family$family %in% c("source_mixture", "bivariate_dp")

  has_discrim <- !is.null(parsed[["discrim"]]) && (is_bivariate || is_vrdp2d)
  has_discrim_B <- !is.null(parsed[["discrim_B"]]) && (is_bivariate || is_vrdp2d)
  has_sigma_B <- !is.null(parsed[["sigma_B"]]) && is_bivariate
  has_sigma2_B <- !is.null(parsed[["sigma2_B"]]) && is_bivariate
  has_rho <- !is.null(parsed[["rho"]]) && is_bivariate
  has_rho_B <- !is.null(parsed[["rho_B"]]) && is_bivariate
  has_rho_N <- !is.null(parsed[["rho_N"]]) && is_bivariate
  has_criterion2 <- !is.null(parsed[["criterion2"]]) && (is_bivariate || is_vrdp2d)
  has_lambda2 <- !is.null(parsed[["lambda2"]]) && is_bivariate_dp
  has_lambda2_B <- !is.null(parsed[["lambda2_B"]]) && is_bivariate_dp
  has_criterion_scale <- !is.null(parsed[["criterion_scale"]])
  has_criterion2_scale <- !is.null(parsed[["criterion2_scale"]]) && has_criterion2
  
  has_rec_crit <- !is.null(parsed$rec_crit) && is_cdp
  has_know_crit <- !is.null(parsed$know_crit) && is_cdp

  has_dprime_L <- !is.null(parsed[["dprime_L"]]) && family$family == "mixture"
  has_sigma_L <- !is.null(parsed[["sigma_L"]]) && family$family == "mixture"
  has_lambda_L <- !is.null(parsed[["lambda_L"]]) && family$family == "mixture"
  
  if (has_dprime_L && !has_lambda_L) {
    stop("lambda_L formula is required when dprime_L is specified (lure mixture model)")
  }
  if (has_lambda_L && !has_dprime_L) {
    stop("dprime_L formula is required when lambda_L is specified (lure mixture model)")
  }
  has_lure_mixture <- has_dprime_L && has_lambda_L
  
  needs_ordered_dprime <- has_dprime2 && family$family == "mixture"
  
  stan_data <- list(N = N, N_ll = N, K = K)
  stan_data$y <- as.integer(data[[parsed$response$response]])

  if (K == 2 && family$family != "evsdt") {
    stop("Binary responses (K=2) are only supported for the evsd() family. ",
         "Models with sigma, lambda, or other parameters require K >= 3 for identifiability.")
  }

  if (is_cdp) {
    cdp_is_old <- .code_is_old(data[[is_old_var]], is_old_var, allow_centered = FALSE)
    stan_data$is_old <- cdp_is_old
    
    rk_values <- data[[rk_var]]
    if (is.character(rk_values) || is.factor(rk_values)) {
      rk_values <- as.character(rk_values)
      rk_numeric <- ifelse(toupper(rk_values) %in% c("R", "REMEMBER", "1"), 1L,
                           ifelse(toupper(rk_values) %in% c("K", "KNOW", "2"), 2L,
                                  ifelse(toupper(rk_values) %in% c("G", "GUESS", "3"), 3L, NA_integer_)))
    } else {
      rk_numeric <- as.integer(rk_values)
    }

    has_rk <- !is.na(rk_numeric)
    if (!any(has_rk)) {
      stop("No rows have a usable remember/know value in `", rk_var, "`.", call. = FALSE)
    }
    judged <- sort(unique(stan_data$y[has_rk]))
    if (is.null(family$old_levels)) {
      thr_levels <- judged
    } else {
      if (any(family$old_levels < 1L | family$old_levels > K)) {
        stop("cdp(old_levels = ...) must be response values within the data's range (",
             paste(range(rc$labels), collapse = " to "), ", in the data's own coding).", call. = FALSE)
      }
      thr_levels <- sort(union(family$old_levels, judged))
    }
    thr_levels <- seq.int(max(min(thr_levels), 2L), K)
    if (is.null(family$old_levels)) {
      message("cdp(): remember/know judgements found on responses ",
              paste(rc$labels[range(judged)], collapse = " to "), "; ",
              if (min(judged) == 1L) "modelling every confidence level."
              else paste0("responses below ", rc$labels[min(thr_levels)], " are treated as one \"new\" category."))
    } else if (!identical(as.integer(thr_levels), as.integer(family$old_levels))) {
      message("cdp(): remember/know judgements were given on responses ",
              paste(rc$labels[range(judged)], collapse = " to "), ", beyond old_levels (",
              paste(rc$labels[range(family$old_levels)], collapse = " to "), "); fitting judgements on all of them, ",
              if (min(thr_levels) == 2L) "with every confidence level modelled."
              else paste0("with responses below ", rc$labels[min(thr_levels)], " treated as one \"new\" category."))
    }
    family$old_levels <- as.integer(thr_levels)
    family$J <- length(thr_levels)
    family$rk_levels <- if (any(has_rk & !(stan_data$y %in% thr_levels))) seq_len(K) else as.integer(thr_levels)

    n_rkg <- max(rk_numeric, na.rm = TRUE)
    if (!n_rkg %in% c(2L, 3L) || any(rk_numeric[has_rk] < 1L)) {
      stop("rk values must have 2 (R/K) or 3 (R/K/G) levels coded 1, 2(, 3); found: ",
           paste(sort(unique(rk_numeric[has_rk])), collapse = ", "))
    }
    stan_data$n_rkg <- as.integer(n_rkg)
    family$n_rkg <- n_rkg

    rk_numeric[is.na(rk_numeric)] <- 0L
    stan_data$rk <- rk_numeric

    if (n_rkg == 3L && !has_know_crit) {
      stop("know_crit formula is required when rk data has 3 levels (R/K/G mode)")
    }
    if (n_rkg == 2L && has_know_crit) {
      warning("know_crit formula ignored: rk data has only 2 levels (R/K mode)")
      has_know_crit <- FALSE
    }

    old_levels <- family$old_levels
    J <- family$J
    new_levels <- setdiff(1:K, old_levels)
    n_new <- length(new_levels)

    stan_data$J <- J
    stan_data$n_new <- n_new
    stan_data$old_level_map <- as.array(old_levels)
    stan_data$new_level_map <- as.array(new_levels)
  } else if (is_bivariate || is_vrdp2d) {
    item_type_vals <- data[[item_type_var]]
    if (is.factor(item_type_vals)) {
      levels_vec <- levels(item_type_vals)
      stan_data$item_type <- as.integer(item_type_vals)
    } else {
      stan_data$item_type <- as.integer(item_type_vals)
    }
    y2_raw <- as.integer(data[[parsed$response2]])
    if (any(is.na(y2_raw))) {
      n_na <- sum(is.na(y2_raw))
      stop(sprintf(
        "Found %d row(s) with NA in the source response (`%s`). The ",
        n_na, parsed$response2),
        "source-conditional-on-old paradigm (where source is only rated when ",
        "a trial is judged old) is not supported. Either ",
        "(a) drop rows with missing source ratings, or (b) impute them.",
        call. = FALSE)
    }
    stan_data$y2 <- y2_raw
    stan_data$K2 <- K2
    stan_data$K1 <- K
    if (identical(family$source_criteria, "varying_shared_new")) {
      old_resp <- family$old_levels
      if (any(old_resp < 1L) || any(old_resp > K)) {
        stop("old_levels must be detection responses within the data's range")
      }
      if (length(unique(old_resp)) >= K) {
        stop("old_levels covers every detection response, so there are no \"new\" responses to share ",
             "source criteria; use source_criteria = \"varying\".")
      }
    }
    if (isTRUE(family$varying_source_criteria)) {
      vb <- .varying_bins(family, K)
      stan_data$n_vbin <- vb$n
      stan_data$vbin <- as.array(vb$of)
    }
    stan_data$is_old <- NULL
  } else if (is_source_mixture) {
    stan_data$source <- as.integer(data[[source_var]])
    stan_data$is_old <- NULL
  } else {
    is_old_vals <- .code_is_old(data[[is_old_var]], is_old_var, allow_centered = TRUE)
    unique_vals <- sort(unique(is_old_vals))
    centered_coding<- all(unique_vals %in% c(-0.5, 0.5))

    if (centered_coding) {
      if (is_cdp) {
        stop("Centered coding for is_old (e.g., 0.5/-0.5) is not supported with the CDP family. ",
             "Please recode is_old as 0/1.")
      }

      if (length(encoding_vars) > 0 || has_lure_mixture) {
        warning("Centered coding for is_old detected (values: ",
                paste(unique_vals, collapse = ", "),
                ") but encoding factors or lure mixture parameters are present. ",
                "Recoding is_old to 0/1.")
        is_old_vals <- ifelse(is_old_vals > 0, 1, 0)
        centered_coding <- FALSE
      }
    }

    data[[is_old_var]] <- is_old_vals
    stan_data$is_old <- is_old_vals
  }
  
  has_counts <- !is.null(counts)
  if (has_counts) {
    if (!counts %in% names(data)) {
      stop("counts variable '", counts, "' not found in data")
    }
    stan_data$counts <- as.integer(data[[counts]])
  }
  
  allow_empty_dprime <- (is_cumulative || is_bivariate_cumulative) &&
    length(parsed$dprime$fixed$terms) == 0 &&
    !parsed$dprime$fixed$intercept
  
  if (allow_empty_dprime) {
    dprime_fixed <- list(
      X = matrix(0, nrow = N, ncol = 1),
      n_coef = 0,
      coef_names = character(0),
      has_encoding = FALSE
    )
    dprime_random <- build_random_effects_structure(parsed$dprime$random, data, encoding_vars, is_old_var)
    
    if (is_cumulative || is_bivariate_cumulative) {
      dprime_random <- drop_reference_level_from_random_slopes(dprime_random, data)
    }
    
    stan_data$P_dprime <- 0
    stan_data$X_dprime <- matrix(0, nrow = N, ncol = 1)
  } else {
    dprime_fixed_formula <- build_formula_from_parsed(parsed$dprime$fixed, allow_empty = is_cumulative || is_bivariate_cumulative)
    dprime_cond <- extract_conditional_from_parsed(parsed$dprime)
    dprime_fixed <- build_fixed_effects_matrix(dprime_fixed_formula, data, encoding_vars, is_old_var,
                                                conditional_terms = dprime_cond,
                                                parsed_fixed = parsed$dprime$fixed)
    dprime_random <- build_random_effects_structure(parsed$dprime$random, data, encoding_vars, is_old_var)
    
    if (is_cumulative && dprime_fixed$n_coef > 0) {
      intercept_col <- which(colnames(dprime_fixed$X) == "(Intercept)")
      if (length(intercept_col) > 0) {
        dprime_fixed$X <- dprime_fixed$X[, -intercept_col, drop = FALSE]
        dprime_fixed$coef_names <- dprime_fixed$coef_names[-intercept_col]
        dprime_fixed$n_coef <- dprime_fixed$n_coef - 1
        
        if (dprime_fixed$n_coef == 0) {
          dprime_fixed$X <- matrix(0, nrow = N, ncol = 1)
          dprime_fixed$coef_names <- character(0)
        }
      }
    }
    
    if (is_cumulative || is_bivariate_cumulative) {
      dprime_random <- drop_reference_level_from_random_slopes(dprime_random, data)
    }
    
    stan_data$P_dprime <- dprime_fixed$n_coef
    stan_data$X_dprime <- dprime_fixed$X
  }
  
  smooth_data <- list()

  n_thresh <- if (is_cdp) family$J else K - 1
  if (has_criterion_scale && n_thresh < 2L) {
    stop("A criterion_scale formula needs at least two criteria (three response categories): ",
         "with one criterion there are no gaps to scale.", call. = FALSE)
  }
  n_thresh_comp <- if (has_criterion_scale) 1L else n_thresh
  criterion_fixed_formula <- build_formula_from_parsed(parsed$criterion$fixed)
  criterion_cond <- extract_conditional_from_parsed(parsed$criterion)
  criterion <- build_criterion_structure(criterion_fixed_formula, parsed$criterion$random, data, n_thresh_comp,
                                         conditional_terms = criterion_cond,
                                         parsed_fixed = parsed$criterion$fixed)
  if (has_criterion_scale) criterion$location_scale <- TRUE
  
  crit_smooth_info <- build_param_smooth_info(parsed$criterion, data, encoding_vars, is_old_var)
  if (!is.null(crit_smooth_info)) {
    for (si in seq_along(crit_smooth_info)) {
      crit_smooth_info[[si]]$n_thresh <- n_thresh_comp
    }
    smooth_data[["criterion"]] <- crit_smooth_info
    criterion <- append_smooth_null_space(criterion, crit_smooth_info)
    stan_data <- add_smooth_to_stan_data(stan_data, "criterion", crit_smooth_info)
  }

  stan_data$P_criterion <- criterion$n_coef
  stan_data$X_criterion <- criterion$X
  
  sigma_fixed <- NULL
  sigma_random <- NULL
  if (has_sigma) {
    sigma_fixed_formula <- build_formula_from_parsed(parsed$sigma$fixed)
    sigma_cond <- extract_conditional_from_parsed(parsed$sigma)
    sigma_fixed <- build_fixed_effects_matrix(sigma_fixed_formula, data, encoding_vars, is_old_var,
                                               conditional_terms = sigma_cond,
                                               parsed_fixed = parsed$sigma$fixed)
    sigma_random <- build_random_effects_structure(parsed$sigma$random, data, encoding_vars, is_old_var)
    
    stan_data$P_sigma <- sigma_fixed$n_coef
    stan_data$X_sigma <- sigma_fixed$X
  }
  
  criterion2_scale_fixed <- NULL
  criterion2_scale_random <- NULL
  if (has_criterion2_scale) {
    if (isTRUE(family$varying_source_criteria)) {
      vb2 <- .varying_bins(family, K)
      data$.c2bin <- factor(vb2$labels[vb2$of[stan_data$y]], levels = vb2$labels)
      own <- .criterion_scale_design(parsed$criterion2_scale, data, encoding_vars, is_old_var, N)
      parsed$criterion2_scale <- parse_broc_parameter_formula(.per_bin_formula(parsed$criterion2_scale, ".c2bin"))
      parsed$criterion2_scale$random <- .merge_group_terms(parsed$criterion2_scale$random, "criterion2_scale")
    }
    criterion2_scale_fixed <- suppressMessages(.criterion_scale_design(parsed$criterion2_scale, data, encoding_vars,
                                                                       is_old_var, N))
    if (isTRUE(family$varying_source_criteria)) {
      keep <- colnames(criterion2_scale_fixed$X) %in%
        as.vector(outer(paste0(".c2bin", levels(data$.c2bin), ":"), own$coef_names, paste0))
      criterion2_scale_fixed$X <- criterion2_scale_fixed$X[, keep, drop = FALSE]
      criterion2_scale_fixed$coef_names <- criterion2_scale_fixed$coef_names[keep]
      criterion2_scale_fixed$n_coef <- sum(keep)
    }
    criterion2_scale_random <- build_random_effects_structure(parsed$criterion2_scale$random, data, encoding_vars, is_old_var)
    stan_data$P_criterion2_scale <- criterion2_scale_fixed$n_coef
    stan_data$X_criterion2_scale <- criterion2_scale_fixed$X
  }

  criterion_scale_fixed <- NULL
  criterion_scale_random <- NULL
  if (has_criterion_scale) {
    criterion_scale_fixed <- .criterion_scale_design(parsed$criterion_scale, data, encoding_vars, is_old_var, N)
    criterion_scale_random <- build_random_effects_structure(parsed$criterion_scale$random, data, encoding_vars, is_old_var)
    stan_data$P_criterion_scale <- criterion_scale_fixed$n_coef
    stan_data$X_criterion_scale <- criterion_scale_fixed$X
  }

  lambda_fixed <- NULL
  lambda_random <- NULL
  if (has_lambda) {
    lambda_fixed_formula <- build_formula_from_parsed(parsed$lambda$fixed)
    lambda_cond <- extract_conditional_from_parsed(parsed$lambda)
    lambda_fixed <- build_fixed_effects_matrix(lambda_fixed_formula, data, encoding_vars, is_old_var,
                                                conditional_terms = lambda_cond,
                                                parsed_fixed = parsed$lambda$fixed)
    lambda_random <- build_random_effects_structure(parsed$lambda$random, data, encoding_vars, is_old_var)
    
    stan_data$P_lambda <- lambda_fixed$n_coef
    stan_data$X_lambda <- lambda_fixed$X
  }
  
  dprime2_fixed <- NULL
  dprime2_random <- NULL
  if (has_dprime2) {
    dprime2_fixed_formula <- build_formula_from_parsed(parsed$dprime2$fixed)
    dprime2_cond <- extract_conditional_from_parsed(parsed$dprime2)
    dprime2_fixed <- build_fixed_effects_matrix(dprime2_fixed_formula, data, encoding_vars, is_old_var,
                                                 conditional_terms = dprime2_cond,
                                                 parsed_fixed = parsed$dprime2$fixed)
    dprime2_random <- build_random_effects_structure(parsed$dprime2$random, data, encoding_vars, is_old_var)
    
    stan_data$P_dprime2 <- dprime2_fixed$n_coef
    stan_data$X_dprime2 <- dprime2_fixed$X
  }
  
  if (needs_ordered_dprime) {
    stan_data[c("dprime_pivot", "dprime_xbar", "dprime2_xbar")] <-
      .dprime_order_data(dprime_fixed$X, dprime2_fixed$X, stan_data$is_old)
  }
  
  sigma2_fixed <- NULL
  sigma2_random <- NULL
  if (has_sigma2) {
    sigma2_fixed_formula <- build_formula_from_parsed(parsed$sigma2$fixed)
    sigma2_cond <- extract_conditional_from_parsed(parsed$sigma2)
    sigma2_fixed <- build_fixed_effects_matrix(sigma2_fixed_formula, data, encoding_vars, is_old_var,
                                                conditional_terms = sigma2_cond,
                                                parsed_fixed = parsed$sigma2$fixed)
    sigma2_random <- build_random_effects_structure(parsed$sigma2$random, data, encoding_vars, is_old_var)
    
    stan_data$P_sigma2 <- sigma2_fixed$n_coef
    stan_data$X_sigma2 <- sigma2_fixed$X
  }
  
  dprime_B_fixed <- NULL
  dprime_B_random <- NULL
  if (has_dprime_B) {
    dprime_B_fixed_formula <- build_formula_from_parsed(parsed$dprime_B$fixed)
    dprime_B_cond <- extract_conditional_from_parsed(parsed$dprime_B)
    dprime_B_fixed <- build_fixed_effects_matrix(dprime_B_fixed_formula, data, encoding_vars, is_old_var,
                                                  conditional_terms = dprime_B_cond,
                                                  parsed_fixed = parsed$dprime_B$fixed)
    dprime_B_random <- build_random_effects_structure(parsed$dprime_B$random, data, encoding_vars, is_old_var)
    
    stan_data$P_dprime_B <- dprime_B_fixed$n_coef
    stan_data$X_dprime_B <- dprime_B_fixed$X
  }
  
  lambda_B_fixed <- NULL
  lambda_B_random <- NULL
  if (has_lambda_B) {
    lambda_B_fixed_formula <- build_formula_from_parsed(parsed$lambda_B$fixed)
    lambda_B_cond <- extract_conditional_from_parsed(parsed$lambda_B)
    lambda_B_fixed <- build_fixed_effects_matrix(lambda_B_fixed_formula, data, encoding_vars, is_old_var,
                                                  conditional_terms = lambda_B_cond,
                                                  parsed_fixed = parsed$lambda_B$fixed)
    lambda_B_random <- build_random_effects_structure(parsed$lambda_B$random, data, encoding_vars, is_old_var)
    
    stan_data$P_lambda_B <- lambda_B_fixed$n_coef
    stan_data$X_lambda_B <- lambda_B_fixed$X
  }
  
  discrim_fixed <- NULL
  discrim_random <- NULL
  if (has_discrim) {
    allow_empty_discrim <- is_bivariate_cumulative &&
      length(parsed$discrim$fixed$terms) == 0 &&
      !parsed$discrim$fixed$intercept

    if (allow_empty_discrim) {
      discrim_fixed <- list(
        X = matrix(0, nrow = N, ncol = 1),
        n_coef = 0,
        coef_names = character(0),
        has_encoding = FALSE
      )
      discrim_random <- build_random_effects_structure(parsed$discrim$random, data, encoding_vars, is_old_var)
      stan_data$P_discrim <- 0
      stan_data$X_discrim <- matrix(0, nrow = N, ncol = 1)
    } else {
      discrim_fixed_formula <- build_formula_from_parsed(parsed$discrim$fixed)
      discrim_cond <- extract_conditional_from_parsed(parsed$discrim)
      discrim_fixed <- build_fixed_effects_matrix(discrim_fixed_formula, data, encoding_vars, is_old_var,
                                                   conditional_terms = discrim_cond,
                                                   parsed_fixed = parsed$discrim$fixed)
      discrim_random <- build_random_effects_structure(parsed$discrim$random, data, encoding_vars, is_old_var)

      stan_data$P_discrim <- discrim_fixed$n_coef
      stan_data$X_discrim <- discrim_fixed$X
    }
  }
  
  discrim_B_fixed <- NULL
  discrim_B_random <- NULL
  if (has_discrim_B) {
    discrim_B_fixed_formula <- build_formula_from_parsed(parsed$discrim_B$fixed)
    discrim_B_cond <- extract_conditional_from_parsed(parsed$discrim_B)
    discrim_B_fixed <- build_fixed_effects_matrix(discrim_B_fixed_formula, data, encoding_vars, is_old_var,
                                                   conditional_terms = discrim_B_cond,
                                                   parsed_fixed = parsed$discrim_B$fixed)
    discrim_B_random <- build_random_effects_structure(parsed$discrim_B$random, data, encoding_vars, is_old_var)
    
    stan_data$P_discrim_B <- discrim_B_fixed$n_coef
    stan_data$X_discrim_B <- discrim_B_fixed$X
  }
  
  sigma_B_fixed <- NULL
  sigma_B_random <- NULL
  if (has_sigma_B) {
    sigma_B_fixed_formula <- build_formula_from_parsed(parsed$sigma_B$fixed)
    sigma_B_cond <- extract_conditional_from_parsed(parsed$sigma_B)
    sigma_B_fixed <- build_fixed_effects_matrix(sigma_B_fixed_formula, data, encoding_vars, is_old_var,
                                                 conditional_terms = sigma_B_cond,
                                                 parsed_fixed = parsed$sigma_B$fixed)
    sigma_B_random <- build_random_effects_structure(parsed$sigma_B$random, data, encoding_vars, is_old_var)
    
    stan_data$P_sigma_B <- sigma_B_fixed$n_coef
    stan_data$X_sigma_B <- sigma_B_fixed$X
  }
  
  sigma2_B_fixed <- NULL
  sigma2_B_random <- NULL
  if (has_sigma2_B) {
    sigma2_B_fixed_formula <- build_formula_from_parsed(parsed$sigma2_B$fixed)
    sigma2_B_cond <- extract_conditional_from_parsed(parsed$sigma2_B)
    sigma2_B_fixed <- build_fixed_effects_matrix(sigma2_B_fixed_formula, data, encoding_vars, is_old_var,
                                                  conditional_terms = sigma2_B_cond,
                                                  parsed_fixed = parsed$sigma2_B$fixed)
    sigma2_B_random <- build_random_effects_structure(parsed$sigma2_B$random, data, encoding_vars, is_old_var)
    
    stan_data$P_sigma2_B <- sigma2_B_fixed$n_coef
    stan_data$X_sigma2_B <- sigma2_B_fixed$X
  }
  
  rho_fixed <- NULL
  rho_random <- NULL
  if (has_rho) {
    rho_fixed_formula <- build_formula_from_parsed(parsed$rho$fixed)
    rho_cond <- extract_conditional_from_parsed(parsed$rho)
    rho_fixed <- build_fixed_effects_matrix(rho_fixed_formula, data, encoding_vars, is_old_var,
                                             conditional_terms = rho_cond,
                                             parsed_fixed = parsed$rho$fixed)
    rho_random <- build_random_effects_structure(parsed$rho$random, data, encoding_vars, is_old_var)
    
    stan_data$P_rho <- rho_fixed$n_coef
    stan_data$X_rho <- rho_fixed$X
  }
  
  rho_B_fixed <- NULL
  rho_B_random <- NULL
  if (has_rho_B) {
    rho_B_fixed_formula <- build_formula_from_parsed(parsed$rho_B$fixed)
    rho_B_cond <- extract_conditional_from_parsed(parsed$rho_B)
    rho_B_fixed <- build_fixed_effects_matrix(rho_B_fixed_formula, data, encoding_vars, is_old_var,
                                               conditional_terms = rho_B_cond,
                                               parsed_fixed = parsed$rho_B$fixed)
    rho_B_random <- build_random_effects_structure(parsed$rho_B$random, data, encoding_vars, is_old_var)
    
    stan_data$P_rho_B <- rho_B_fixed$n_coef
    stan_data$X_rho_B <- rho_B_fixed$X
  }
  
  rho_N_fixed <- NULL
  rho_N_random <- NULL
  if (has_rho_N) {
    rho_N_fixed_formula <- build_formula_from_parsed(parsed$rho_N$fixed)
    rho_N_cond <- extract_conditional_from_parsed(parsed$rho_N)
    rho_N_fixed <- build_fixed_effects_matrix(rho_N_fixed_formula, data, encoding_vars, is_old_var,
                                               conditional_terms = rho_N_cond,
                                               parsed_fixed = parsed$rho_N$fixed)
    rho_N_random <- build_random_effects_structure(parsed$rho_N$random, data, encoding_vars, is_old_var)
    
    stan_data$P_rho_N <- rho_N_fixed$n_coef
    stan_data$X_rho_N <- rho_N_fixed$X
  }
  
  lambda2_fixed <- NULL
  lambda2_random <- NULL
  if (has_lambda2) {
    lambda2_fixed_formula <- build_formula_from_parsed(parsed$lambda2$fixed)
    lambda2_cond <- extract_conditional_from_parsed(parsed$lambda2)
    lambda2_fixed <- build_fixed_effects_matrix(lambda2_fixed_formula, data, encoding_vars, is_old_var,
                                                 conditional_terms = lambda2_cond,
                                                 parsed_fixed = parsed$lambda2$fixed)
    lambda2_random <- build_random_effects_structure(parsed$lambda2$random, data, encoding_vars, is_old_var)

    stan_data$P_lambda2 <- lambda2_fixed$n_coef
    stan_data$X_lambda2 <- lambda2_fixed$X
  }

  lambda2_B_fixed <- NULL
  lambda2_B_random <- NULL
  if (has_lambda2_B) {
    lambda2_B_fixed_formula <- build_formula_from_parsed(parsed$lambda2_B$fixed)
    lambda2_B_cond <- extract_conditional_from_parsed(parsed$lambda2_B)
    lambda2_B_fixed <- build_fixed_effects_matrix(lambda2_B_fixed_formula, data, encoding_vars, is_old_var,
                                                   conditional_terms = lambda2_B_cond,
                                                   parsed_fixed = parsed$lambda2_B$fixed)
    lambda2_B_random <- build_random_effects_structure(parsed$lambda2_B$random, data, encoding_vars, is_old_var)

    stan_data$P_lambda2_B <- lambda2_B_fixed$n_coef
    stan_data$X_lambda2_B <- lambda2_B_fixed$X
  }

  rec_crit_fixed <- NULL
  rec_crit_random <- NULL
  if (has_rec_crit) {
    rec_crit_fixed_formula <- build_formula_from_parsed(parsed$rec_crit$fixed)
    rec_crit_cond <- extract_conditional_from_parsed(parsed$rec_crit)
    rec_crit_fixed <- build_fixed_effects_matrix(rec_crit_fixed_formula, data, encoding_vars, is_old_var,
                                                  conditional_terms = rec_crit_cond,
                                                  parsed_fixed = parsed$rec_crit$fixed)
    rec_crit_random <- build_random_effects_structure(parsed$rec_crit$random, data, encoding_vars, is_old_var)
    
    stan_data$P_rec_crit <- rec_crit_fixed$n_coef
    stan_data$X_rec_crit <- rec_crit_fixed$X
  }
  
  know_crit_fixed <- NULL
  know_crit_random <- NULL
  if (has_know_crit) {
    know_crit_fixed_formula <- build_formula_from_parsed(parsed$know_crit$fixed)
    know_crit_cond <- extract_conditional_from_parsed(parsed$know_crit)
    know_crit_fixed <- build_fixed_effects_matrix(know_crit_fixed_formula, data, encoding_vars, is_old_var,
                                                   conditional_terms = know_crit_cond,
                                                   parsed_fixed = parsed$know_crit$fixed)
    know_crit_random <- build_random_effects_structure(parsed$know_crit$random, data, encoding_vars, is_old_var)

    stan_data$P_know_crit <- know_crit_fixed$n_coef
    stan_data$X_know_crit <- know_crit_fixed$X
  }

  dprime_L_fixed <- NULL
  dprime_L_random <- NULL
  if (has_dprime_L) {
    dprime_L_fixed_formula <- build_formula_from_parsed(parsed$dprime_L$fixed)
    dprime_L_cond <- extract_conditional_from_parsed(parsed$dprime_L)
    dprime_L_fixed <- build_fixed_effects_matrix(dprime_L_fixed_formula, data, encoding_vars, is_old_var,
                                                  conditional_terms = dprime_L_cond,
                                                  parsed_fixed = parsed$dprime_L$fixed)
    dprime_L_random <- build_random_effects_structure(parsed$dprime_L$random, data, encoding_vars, is_old_var)
    
    stan_data$P_dprime_L <- dprime_L_fixed$n_coef
    stan_data$X_dprime_L <- dprime_L_fixed$X
  }
  
  sigma_L_fixed <- NULL
  sigma_L_random <- NULL
  if (has_sigma_L) {
    sigma_L_fixed_formula <- build_formula_from_parsed(parsed$sigma_L$fixed)
    sigma_L_cond <- extract_conditional_from_parsed(parsed$sigma_L)
    sigma_L_fixed <- build_fixed_effects_matrix(sigma_L_fixed_formula, data, encoding_vars, is_old_var,
                                                 conditional_terms = sigma_L_cond,
                                                 parsed_fixed = parsed$sigma_L$fixed)
    sigma_L_random <- build_random_effects_structure(parsed$sigma_L$random, data, encoding_vars, is_old_var)
    
    stan_data$P_sigma_L <- sigma_L_fixed$n_coef
    stan_data$X_sigma_L <- sigma_L_fixed$X
  }
  
  lambda_L_fixed <- NULL
  lambda_L_random <- NULL
  if (has_lambda_L) {
    lambda_L_fixed_formula <- build_formula_from_parsed(parsed$lambda_L$fixed)
    lambda_L_cond <- extract_conditional_from_parsed(parsed$lambda_L)
    lambda_L_fixed <- build_fixed_effects_matrix(lambda_L_fixed_formula, data, encoding_vars, is_old_var,
                                                  conditional_terms = lambda_L_cond,
                                                  parsed_fixed = parsed$lambda_L$fixed)
    lambda_L_random <- build_random_effects_structure(parsed$lambda_L$random, data, encoding_vars, is_old_var)
    
    stan_data$P_lambda_L <- lambda_L_fixed$n_coef
    stan_data$X_lambda_L <- lambda_L_fixed$X
  }
  
  criterion2 <- NULL
  if (has_criterion2) {
    criterion2_fixed_formula <- build_formula_from_parsed(parsed$criterion2$fixed)
    criterion2_cond <- extract_conditional_from_parsed(parsed$criterion2)

    if (has_criterion2_scale && K2 - 1L < 2L) {
      stop("A criterion2_scale formula needs at least two source criteria (three source response ",
           "categories): with one criterion there are no gaps to scale.", call. = FALSE)
    }
    varying2 <- isTRUE(family$varying_source_criteria)
    n2_comp <- if (has_criterion2_scale && !varying2) 1L else K2 - 1L
    criterion2 <- build_criterion_structure(criterion2_fixed_formula, parsed$criterion2$random, data, n2_comp,
                                            conditional_terms = criterion2_cond,
                                            parsed_fixed = parsed$criterion2$fixed)
    if (has_criterion2_scale) criterion2$location_scale <- TRUE

    if (isTRUE(family$varying_source_criteria) && length(parsed$criterion2$smooth)) {
      sm2 <- parsed$criterion2$smooth
      if (any(grepl("\\bby\\s*=", sm2))) {
        stop("A smooth in criterion2 already has a by= variable; with varying source criteria each detection-",
             "response set gets its own curve (by the set), so one smooth cannot take a second by= variable.",
             call. = FALSE)
      }
      vb2 <- .varying_bins(family, K)
      data$.c2bin <- factor(vb2$labels[vb2$of[stan_data$y]], levels = vb2$labels)
      parsed$criterion2$smooth <- sub("\\)\\s*$", ", by = .c2bin)", sm2)
      parsed$criterion2$fixed$smooth_terms <- parsed$criterion2$smooth
    }
    crit2_smooth_info <- build_param_smooth_info(parsed$criterion2, data, encoding_vars, is_old_var)
    if (!is.null(crit2_smooth_info)) {
      n_thresh2 <- if (has_criterion2_scale) 1L else K2 - 1
      for (si in seq_along(crit2_smooth_info)) {
        crit2_smooth_info[[si]]$n_thresh <- n_thresh2
      }
      smooth_data[["criterion2"]] <- crit2_smooth_info
      criterion2 <- append_smooth_null_space(criterion2, crit2_smooth_info)
      stan_data <- add_smooth_to_stan_data(stan_data, "criterion2", crit2_smooth_info)
    }

    if (isTRUE(family$varying_source_criteria)) {
      c2_names <- colnames(criterion2$X) %||% criterion2$coef_names
      vc <- .varying_cells(criterion2$X, c2_names, criterion2$location_coefs)
      cov_cols <- which(!c2_names %in% c("(Intercept)", vc$names))
      criterion2$cov_names <- c2_names[cov_cols]
      criterion2$vcell_names <- vc$labels
      stan_data$P_criterion2_cov <- length(cov_cols)
      stan_data$X_criterion2_cov <- criterion2$X[, cov_cols, drop = FALSE]
      stan_data$n_vcell <- length(vc$labels)
      stan_data$vcell <- as.array(vc$of)
      stan_data$n_vset <- .varying_bins(family, K)$n * length(vc$labels)
    }

    stan_data$P_criterion2 <- criterion2$n_coef
    stan_data$X_criterion2 <- criterion2$X

    if (isTRUE(family$varying_source_criteria) && !is.null(criterion2$random)) {
      vb <- .varying_bins(family, K)
      for (group in names(criterion2$random)) {
        criterion2$random[[group]] <- .varying_crit2_re(criterion2$random[[group]],
                                                        if (has_criterion2_scale) "per_bin" else "full", vb$n, K2)
        criterion2$random[[group]]$bin_levels <- vb$labels
      }
    }
  }

  for (crit in list(criterion$random, criterion2$random)) {
    for (re in crit) {
      if (!is.null(re$cor_id) && !all(unlist(re$set_cor_thresholds %||% TRUE))) {
        stop("gr(", re$group, ", cor_thresholds = FALSE) cannot be combined with a cross-parameter ",
             "correlation group (| ", re$cor_id, " |): a correlation group correlates all of its effects.",
             call. = FALSE)
      }
    }
  }
  
  all_groups <- unique(c(
    names(dprime_random),
    names(criterion$random),
    if (has_sigma) names(sigma_random) else NULL,
    if (has_lambda) names(lambda_random) else NULL,
    if (has_dprime2) names(dprime2_random) else NULL,
    if (has_sigma2) names(sigma2_random) else NULL,
    if (has_dprime_B) names(dprime_B_random) else NULL,
    if (has_lambda_B) names(lambda_B_random) else NULL,
    if (has_discrim) names(discrim_random) else NULL,
    if (has_discrim_B) names(discrim_B_random) else NULL,
    if (has_sigma_B) names(sigma_B_random) else NULL,
    if (has_sigma2_B) names(sigma2_B_random) else NULL,
    if (has_rho) names(rho_random) else NULL,
    if (has_rho_B) names(rho_B_random) else NULL,
    if (has_rho_N) names(rho_N_random) else NULL,
    if (has_rec_crit) names(rec_crit_random) else NULL,
    if (has_know_crit) names(know_crit_random) else NULL,
    if (has_criterion2) names(criterion2$random) else NULL,
    if (has_dprime_L) names(dprime_L_random) else NULL,
    if (has_sigma_L) names(sigma_L_random) else NULL,
    if (has_lambda_L) names(lambda_L_random) else NULL,
    if (has_lambda2) names(lambda2_random) else NULL,
    if (has_lambda2_B) names(lambda2_B_random) else NULL,
    if (has_criterion_scale) names(criterion_scale_random) else NULL,
    if (has_criterion2_scale) names(criterion2_scale_random) else NULL
  ))

  for (group in all_groups) {
    re_info <- get_re_info_from_all(group, dprime_random, criterion$random,
                                    sigma_random, lambda_random,
                                    dprime2_random, sigma2_random,
                                    dprime_B_random, lambda_B_random,
                                    discrim_random, discrim_B_random,
                                    sigma_B_random, sigma2_B_random,
                                    rho_random, rho_B_random, rho_N_random,
                                    rec_crit_random, know_crit_random,
                                    if (has_criterion2) criterion2$random else NULL,
                                    dprime_L_random, sigma_L_random, lambda_L_random,
                                    lambda2_random, lambda2_B_random, criterion_scale_random,
                                    criterion2_scale_random)
    if (!is.null(re_info)) {
      stan_data[[paste0("N_", group)]] <- re_info$n_groups
      stan_data[[group]] <- re_info$group_idx
    }
  }
  
  stan_data <- add_term_indices(stan_data, "dprime", dprime_random)
  if (has_sigma) stan_data <- add_term_indices(stan_data, "sigma", sigma_random)
  if (has_lambda) stan_data <- add_term_indices(stan_data, "lambda", lambda_random)
  if (has_dprime2) stan_data <- add_term_indices(stan_data, "dprime2", dprime2_random)
  if (has_sigma2) stan_data <- add_term_indices(stan_data, "sigma2", sigma2_random)
  if (has_dprime_B) stan_data <- add_term_indices(stan_data, "dprime_B", dprime_B_random)
  if (has_lambda_B) stan_data <- add_term_indices(stan_data, "lambda_B", lambda_B_random)
  if (has_discrim) stan_data <- add_term_indices(stan_data, "discrim", discrim_random)
  if (has_discrim_B) stan_data <- add_term_indices(stan_data, "discrim_B", discrim_B_random)
  if (has_sigma_B) stan_data <- add_term_indices(stan_data, "sigma_B", sigma_B_random)
  if (has_sigma2_B) stan_data <- add_term_indices(stan_data, "sigma2_B", sigma2_B_random)
  if (has_rho) stan_data <- add_term_indices(stan_data, "rho", rho_random)
  if (has_rho_B) stan_data <- add_term_indices(stan_data, "rho_B", rho_B_random)
  if (has_rho_N) stan_data <- add_term_indices(stan_data, "rho_N", rho_N_random)
  if (has_rec_crit) stan_data <- add_term_indices(stan_data, "rec_crit", rec_crit_random)
  if (has_know_crit) stan_data <- add_term_indices(stan_data, "know_crit", know_crit_random)
  if (has_dprime_L) stan_data <- add_term_indices(stan_data, "dprime_L", dprime_L_random)
  if (has_sigma_L) stan_data <- add_term_indices(stan_data, "sigma_L", sigma_L_random)
  if (has_lambda_L) stan_data <- add_term_indices(stan_data, "lambda_L", lambda_L_random)
  if (has_lambda2) stan_data <- add_term_indices(stan_data, "lambda2", lambda2_random)
  if (has_lambda2_B) stan_data <- add_term_indices(stan_data, "lambda2_B", lambda2_B_random)
  if (has_criterion_scale) stan_data <- add_term_indices(stan_data, "criterion_scale", criterion_scale_random)
  if (has_criterion2_scale) stan_data <- add_term_indices(stan_data, "criterion2_scale", criterion2_scale_random)

  for (group in names(criterion$random)) {
    re <- criterion$random[[group]]
    if (isTRUE(re$use_z_matrix) && !is.null(re$Z)) {
      z_name <- paste0("Z_criterion_", group)
      stan_data[[z_name]] <- re$Z
    } else if (!is.null(re$term_idx)) {
      idx_name <- paste0("idx_criterion_", group)
      dim_name <- paste0("D_idx_criterion_", group)
      stan_data[[dim_name]] <- re$n_cond_levels
      stan_data[[idx_name]] <- re$term_idx
    }
  }
  
  if (has_criterion2 && !is.null(criterion2$random)) {
    for (group in names(criterion2$random)) {
      re <- criterion2$random[[group]]
      if (isTRUE(re$use_z_matrix) && !is.null(re$Z)) {
        z_name <- paste0("Z_criterion2_", group)
        stan_data[[z_name]] <- re$Z
      } else if (!is.null(re$term_idx)) {
        idx_name <- paste0("idx_criterion2_", group)
        dim_name <- paste0("D_idx_criterion2_", group)
        stan_data[[dim_name]] <- re$n_cond_levels
        stan_data[[idx_name]] <- re$term_idx
      }
    }
  }
  
  # ---- Smooth terms processing ----

  all_param_fixed <- list(
    dprime = dprime_fixed,
    sigma = if (has_sigma) sigma_fixed else NULL,
    lambda = if (has_lambda) lambda_fixed else NULL,
    dprime2 = if (has_dprime2) dprime2_fixed else NULL,
    sigma2 = if (has_sigma2) sigma2_fixed else NULL,
    dprime_B = if (has_dprime_B) dprime_B_fixed else NULL,
    lambda_B = if (has_lambda_B) lambda_B_fixed else NULL,
    discrim = if (has_discrim) discrim_fixed else NULL,
    discrim_B = if (has_discrim_B) discrim_B_fixed else NULL,
    sigma_B = if (has_sigma_B) sigma_B_fixed else NULL,
    sigma2_B = if (has_sigma2_B) sigma2_B_fixed else NULL,
    rho = if (has_rho) rho_fixed else NULL,
    rho_B = if (has_rho_B) rho_B_fixed else NULL,
    rho_N = if (has_rho_N) rho_N_fixed else NULL,
    lambda2 = if (has_lambda2) lambda2_fixed else NULL,
    lambda2_B = if (has_lambda2_B) lambda2_B_fixed else NULL,
    rec_crit = if (has_rec_crit) rec_crit_fixed else NULL,
    know_crit = if (has_know_crit) know_crit_fixed else NULL,
    dprime_L = if (has_dprime_L) dprime_L_fixed else NULL,
    sigma_L = if (has_sigma_L) sigma_L_fixed else NULL,
    lambda_L = if (has_lambda_L) lambda_L_fixed else NULL,
    criterion_scale = if (has_criterion_scale) criterion_scale_fixed else NULL,
    criterion2_scale = if (has_criterion2_scale) criterion2_scale_fixed else NULL
  )

  for (pname in names(all_param_fixed)) {
    if (is.null(all_param_fixed[[pname]])) next
    sm_info <- build_param_smooth_info(parsed[[pname]], data, encoding_vars, is_old_var)
    if (is.null(sm_info)) next

    smooth_data[[pname]] <- sm_info

    fixed_ref <- get(paste0(pname, "_fixed"))
    if (fixed_ref$n_coef == 0) {
      fixed_ref$X <- matrix(0, nrow = N, ncol = 0)
      fixed_ref$coef_names <- character(0)
    }
    for (sm in sm_info) {
      for (comp in sm$components) {
        if (NCOL(comp$Xs) > 0) {
          fixed_ref$X <- cbind(fixed_ref$X, comp$Xs)
          fixed_ref$coef_names <- colnames(fixed_ref$X)
          fixed_ref$n_coef <- ncol(fixed_ref$X)
        }
      }
    }
    assign(paste0(pname, "_fixed"), fixed_ref)
    stan_data[[paste0("X_", pname)]] <- fixed_ref$X
    stan_data[[paste0("P_", pname)]] <- fixed_ref$n_coef

    stan_data <- add_smooth_to_stan_data(stan_data, pname, sm_info)
  }

  cross_cor <- build_cross_cor_info_v2(parsed, dprime_random, sigma_random,
                                       criterion$random, lambda_random,
                                       dprime2_random, sigma2_random,
                                       dprime_B_random, lambda_B_random,
                                       discrim_random, discrim_B_random,
                                       sigma_B_random, sigma2_B_random,
                                       rho_random, rho_B_random, rho_N_random,
                                       if (has_criterion2) criterion2$random else NULL,
                                       dprime_L_random, sigma_L_random, lambda_L_random,
                                       rec_crit_random, know_crit_random,
                                       lambda2_random, lambda2_B_random,
                                       criterion_scale_random, criterion2_scale_random)

  has_dprime_fixed <- dprime_fixed$n_coef > 0
  has_discrim_fixed <- if (!is.null(discrim_fixed)) discrim_fixed$n_coef > 0 else FALSE
  
  .check_criterion_designs(stan_data, family$family)
  if (isTRUE(has_know_crit)) {
    re_in_bound <- any(lengths(list(criterion$random, rec_crit_random, know_crit_random)) > 0)
    stan_data <- .rkg_knob_data(stan_data, re_in_bound)
  }

  list(
    N = N,
    K = K,
    K2 = K2,
    response_labels = response_labels,
    response2_labels = response2_labels,
    family = family$family,
    family_obj = family,
    has_sigma = has_sigma,
    has_lambda = has_lambda,
    has_dprime2 = has_dprime2,
    has_sigma2 = has_sigma2,
    has_dprime_B = has_dprime_B,
    has_lambda_B = has_lambda_B,
    has_discrim = has_discrim,
    has_discrim_B = has_discrim_B,
    has_sigma_B = has_sigma_B,
    has_sigma2_B = has_sigma2_B,
    has_rho = has_rho,
    has_rho_B = has_rho_B,
    has_rho_N = has_rho_N,
    has_criterion2 = has_criterion2,
    has_dprime_fixed = has_dprime_fixed,
    has_discrim_fixed = has_discrim_fixed,
    needs_ordered_dprime = needs_ordered_dprime,
    encoding_vars = encoding_vars,
    is_old_var = is_old_var,
    source_var = source_var,
    item_type_var = item_type_var,
    dprime_fixed = dprime_fixed,
    dprime_random = dprime_random,
    criterion = criterion,
    criterion2 = criterion2,
    sigma_fixed = sigma_fixed,
    sigma_random = sigma_random,
    lambda_fixed = lambda_fixed,
    lambda_random = lambda_random,
    dprime2_fixed = dprime2_fixed,
    dprime2_random = dprime2_random,
    sigma2_fixed = sigma2_fixed,
    sigma2_random = sigma2_random,
    dprime_B_fixed = dprime_B_fixed,
    dprime_B_random = dprime_B_random,
    lambda_B_fixed = lambda_B_fixed,
    lambda_B_random = lambda_B_random,
    discrim_fixed = discrim_fixed,
    discrim_random = discrim_random,
    discrim_B_fixed = discrim_B_fixed,
    discrim_B_random = discrim_B_random,
    sigma_B_fixed = sigma_B_fixed,
    sigma_B_random = sigma_B_random,
    sigma2_B_fixed = sigma2_B_fixed,
    sigma2_B_random = sigma2_B_random,
    rho_fixed = rho_fixed,
    rho_random = rho_random,
    rho_B_fixed = rho_B_fixed,
    rho_B_random = rho_B_random,
    rho_N_fixed = rho_N_fixed,
    rho_N_random = rho_N_random,
    has_lambda2 = has_lambda2,
    lambda2_fixed = lambda2_fixed,
    lambda2_random = lambda2_random,
    has_lambda2_B = has_lambda2_B,
    lambda2_B_fixed = lambda2_B_fixed,
    lambda2_B_random = lambda2_B_random,
    rec_crit_fixed = rec_crit_fixed,
    rec_crit_random = rec_crit_random,
    has_rec_crit = has_rec_crit,
    know_crit_fixed = know_crit_fixed,
    know_crit_random = know_crit_random,
    has_know_crit = has_know_crit,
    has_dprime_L = has_dprime_L,
    has_sigma_L = has_sigma_L,
    has_lambda_L = has_lambda_L,
    has_lure_mixture = has_lure_mixture,
    dprime_L_fixed = dprime_L_fixed,
    dprime_L_random = dprime_L_random,
    sigma_L_fixed = sigma_L_fixed,
    sigma_L_random = sigma_L_random,
    lambda_L_fixed = lambda_L_fixed,
    lambda_L_random = lambda_L_random,
    has_criterion_scale = has_criterion_scale,
    criterion_scale_fixed = criterion_scale_fixed,
    criterion_scale_random = criterion_scale_random,
    has_criterion2_scale = has_criterion2_scale,
    criterion2_scale_fixed = criterion2_scale_fixed,
    criterion2_scale_random = criterion2_scale_random,
    centered_coding = if (exists("centered_coding")) centered_coding else FALSE,
    has_counts = has_counts,
    varying_source_criteria = isTRUE(family$varying_source_criteria),
    bounded = isTRUE(family$bounded),
    source_criteria = family$source_criteria,
    old_levels = family$old_levels,
    rk_levels = family$rk_levels,
    varying_re = if (isTRUE(family$varying_source_criteria)) "full" else "shared",
    cross_cor = cross_cor,
    smooth_data = if (length(smooth_data) > 0) smooth_data else NULL,
    stan_data = stan_data
  )
}

#' Helper to get RE info from any parameter
#' @noRd
get_re_info_from_all <- function(group, dprime_re, criterion_re, sigma_re, 
                                 lambda_re, dprime2_re, sigma2_re,
                                 dprime_B_re = NULL, lambda_B_re = NULL,
                                 discrim_re = NULL, discrim_B_re = NULL,
                                 sigma_B_re = NULL, sigma2_B_re = NULL,
                                 rho_re = NULL, rho_B_re = NULL, rho_N_re = NULL,
                                 rec_crit_re = NULL, know_crit_re = NULL,
                                 criterion2_re = NULL,
                                 dprime_L_re = NULL, sigma_L_re = NULL, lambda_L_re = NULL,
                                 lambda2_re = NULL, lambda2_B_re = NULL,
                                 criterion_scale_re = NULL, criterion2_scale_re = NULL) {
  if (!is.null(dprime_re) && group %in% names(dprime_re)) return(dprime_re[[group]])
  if (!is.null(criterion_re) && group %in% names(criterion_re)) return(criterion_re[[group]])
  if (!is.null(sigma_re) && group %in% names(sigma_re)) return(sigma_re[[group]])
  if (!is.null(lambda_re) && group %in% names(lambda_re)) return(lambda_re[[group]])
  if (!is.null(dprime2_re) && group %in% names(dprime2_re)) return(dprime2_re[[group]])
  if (!is.null(sigma2_re) && group %in% names(sigma2_re)) return(sigma2_re[[group]])
  if (!is.null(dprime_B_re) && group %in% names(dprime_B_re)) return(dprime_B_re[[group]])
  if (!is.null(lambda_B_re) && group %in% names(lambda_B_re)) return(lambda_B_re[[group]])
  if (!is.null(discrim_re) && group %in% names(discrim_re)) return(discrim_re[[group]])
  if (!is.null(discrim_B_re) && group %in% names(discrim_B_re)) return(discrim_B_re[[group]])
  if (!is.null(sigma_B_re) && group %in% names(sigma_B_re)) return(sigma_B_re[[group]])
  if (!is.null(sigma2_B_re) && group %in% names(sigma2_B_re)) return(sigma2_B_re[[group]])
  if (!is.null(rho_re) && group %in% names(rho_re)) return(rho_re[[group]])
  if (!is.null(rho_B_re) && group %in% names(rho_B_re)) return(rho_B_re[[group]])
  if (!is.null(rho_N_re) && group %in% names(rho_N_re)) return(rho_N_re[[group]])
  if (!is.null(rec_crit_re) && group %in% names(rec_crit_re)) return(rec_crit_re[[group]])
  if (!is.null(know_crit_re) && group %in% names(know_crit_re)) return(know_crit_re[[group]])
  if (!is.null(criterion2_re) && group %in% names(criterion2_re)) return(criterion2_re[[group]])
  if (!is.null(dprime_L_re) && group %in% names(dprime_L_re)) return(dprime_L_re[[group]])
  if (!is.null(sigma_L_re) && group %in% names(sigma_L_re)) return(sigma_L_re[[group]])
  if (!is.null(lambda_L_re) && group %in% names(lambda_L_re)) return(lambda_L_re[[group]])
  if (!is.null(lambda2_re) && group %in% names(lambda2_re)) return(lambda2_re[[group]])
  if (!is.null(lambda2_B_re) && group %in% names(lambda2_B_re)) return(lambda2_B_re[[group]])
  if (!is.null(criterion_scale_re) && group %in% names(criterion_scale_re)) return(criterion_scale_re[[group]])
  if (!is.null(criterion2_scale_re) && group %in% names(criterion2_scale_re)) return(criterion2_scale_re[[group]])
  NULL
}

#' Add term indices for a parameter's random effects
#' @noRd
.criterion_scale_design <- function(parsed_param, data, encoding_vars, is_old_var, N) {
  fx <- parsed_param$fixed
  if (!isTRUE(fx$intercept) && length(fx$terms)) {
    message("criterion_scale: the gaps carry the scale's intercept, so its formula is coded ",
            "with contrasts and no intercept.")
  }
  fx$intercept <- TRUE
  if (!length(fx$terms)) {
    return(list(X = matrix(0, nrow = N, ncol = 0), n_coef = 0, coef_names = character(0),
                has_encoding = FALSE, formula = ~ 1))
  }
  fe <- build_fixed_effects_matrix(build_formula_from_parsed(fx), data, encoding_vars, is_old_var,
                                   conditional_terms = extract_conditional_from_parsed(parsed_param),
                                   parsed_fixed = fx)
  keep <- colnames(fe$X) != "(Intercept)"
  fe$X <- fe$X[, keep, drop = FALSE]
  fe$coef_names <- fe$coef_names[keep]
  fe$n_coef <- sum(keep)
  fe
}

.per_bin_formula <- function(parsed_param, bin) {
  fx <- parsed_param$fixed
  fixed <- if (length(fx$terms)) paste0(bin, ":", fx$terms)
  sm <- vapply(fx$smooth_terms %||% character(0), function(x) sub("\\)$", paste0(", by = ", bin, ")"), x), "")
  one <- function(r) {
    lhs <- c("0", if (isTRUE(r$intercept)) bin, if (length(r$terms)) paste0(bin, ":", r$terms))
    grp <- if (!is.null(r$cor_id)) paste0(r$cor_id, " | ", r$group) else r$group
    sprintf("(%s %s %s)", paste(lhs, collapse = " + "), if (isFALSE(r$correlated) && is.null(r$cor_id)) "||" else "|", grp)
  }
  re <- unlist(lapply(parsed_param$random, function(r) if (length(r$parts)) vapply(r$parts, one, "") else one(r)),
               use.names = FALSE)
  rhs <- c(fixed, sm, re)
  stats::as.formula(paste("~", if (length(rhs)) paste(rhs, collapse = " + ") else "1"))
}

add_term_indices <- function(stan_data, param_name, random_effects) {
  if (is.null(random_effects)) return(stan_data)
  
  for (group in names(random_effects)) {
    re <- random_effects[[group]]
    
    if (isTRUE(re$use_z_matrix) && !is.null(re$Z)) {
      z_name <- paste0("Z_", param_name, "_", group)
      stan_data[[z_name]] <- re$Z
    } else if (!is.null(re$term_idx)) {
      idx_name <- paste0("idx_", param_name, "_", group)
      dim_name <- paste0("D_idx_", param_name, "_", group)
      stan_data[[dim_name]] <- re$dim
      stan_data[[idx_name]] <- re$term_idx
    }
  }
  stan_data
}

#' Build Smooth Info for a Parameter
#'
#' If the parsed parameter has smooth terms, calls build_smooth_terms().
#' Returns NULL if no smooth terms are present.
#'
#' @param parsed_param Parsed parameter structure (with $smooth field if smooths present)
#' @param data Data frame
#' @return List of smooth structures or NULL
#' @noRd
build_param_smooth_info <- function(parsed_param, data, encoding_vars = NULL, is_old_var = NULL) {
  if (is.null(parsed_param) || is.null(parsed_param$smooth) || length(parsed_param$smooth) == 0) {
    return(NULL)
  }
  build_smooth_terms(parsed_param$smooth, data, encoding_vars = encoding_vars, is_old_var = is_old_var)
}

#' Add Smooth Penalized Matrices to Stan Data
#'
#' For each smooth component's penalized Zs matrices, adds them to stan_data.
#'
#' @param stan_data Current stan_data list
#' @param param_name Parameter name (e.g., "dprime")
#' @param smooth_info Output of build_smooth_terms() or NULL
#' @return Updated stan_data
#' @noRd
add_smooth_to_stan_data <- function(stan_data, param_name, smooth_info) {
  if (is.null(smooth_info) || length(smooth_info) == 0) return(stan_data)

  for (sm in smooth_info) {
    for (j in seq_along(sm$components)) {
      comp <- sm$components[[j]]
      comp_suffix <- comp$san_label

      for (k in seq_along(comp$Zs_list)) {
        zs_name <- paste0("Zs_", param_name, "_", comp_suffix, "_", k)
        nbasis_name <- paste0("nbasis_", param_name, "_", comp_suffix, "_", k)
        stan_data[[zs_name]] <- as.matrix(comp$Zs_list[[k]])
        stan_data[[nbasis_name]] <- ncol(comp$Zs_list[[k]])
      }
    }
  }

  stan_data
}

#' Build Formula from Parsed Fixed Effects
#' @noRd
build_formula_from_parsed <- function(fixed, allow_empty = FALSE) {
  if (length(fixed$terms) == 0) {
    if (fixed$intercept) {
      return(~ 1)
    } else if (allow_empty) {
      return(~ 0)
    } else {
      stop("Must have either intercept or terms")
    }
  }
  
  terms_str <- paste(fixed$terms, collapse = " + ")
  
  if (fixed$intercept) {
    formula_str <- paste("~", terms_str)
  } else {
    formula_str <- paste("~ 0 +", terms_str)
  }
  
  as.formula(formula_str)
}

#' Build Cross-Correlation Info
#' @noRd
build_cross_cor_info_v2 <- function(parsed, dprime_random, sigma_random, 
                                    criterion_random, lambda_random = NULL,
                                    dprime2_random = NULL, sigma2_random = NULL,
                                    dprime_B_random = NULL, lambda_B_random = NULL,
                                    discrim_random = NULL, discrim_B_random = NULL,
                                    sigma_B_random = NULL, sigma2_B_random = NULL,
                                    rho_random = NULL, rho_B_random = NULL, rho_N_random = NULL,
                                    criterion2_random = NULL,
                                    dprime_L_random = NULL, sigma_L_random = NULL, lambda_L_random = NULL,
                                    rec_crit_random = NULL, know_crit_random = NULL,
                                    lambda2_random = NULL,
                                    lambda2_B_random = NULL,
                                    criterion_scale_random = NULL,
                                    criterion2_scale_random = NULL) {
  if (length(parsed$cross_cor_ids) == 0) return(list())
  
  cross_cor <- list()
  
  for (cor_id in parsed$cross_cor_ids) {
    members <- list()
    group <- NULL
    
    param_re_pairs <- list(
      dprime = dprime_random,
      sigma = sigma_random,
      criterion = criterion_random,
      lambda = lambda_random,
      dprime2 = dprime2_random,
      sigma2 = sigma2_random,
      dprime_B = dprime_B_random,
      lambda_B = lambda_B_random,
      discrim = discrim_random,
      discrim_B = discrim_B_random,
      sigma_B = sigma_B_random,
      sigma2_B = sigma2_B_random,
      rho = rho_random,
      rho_B = rho_B_random,
      rho_N = rho_N_random,
      criterion2 = criterion2_random,
      dprime_L = dprime_L_random,
      sigma_L = sigma_L_random,
      lambda_L = lambda_L_random,
      rec_crit = rec_crit_random,
      know_crit = know_crit_random,
      lambda2 = lambda2_random,
      lambda2_B = lambda2_B_random,
      criterion_scale = criterion_scale_random,
      criterion2_scale = criterion2_scale_random
    )
    
    for (param_name in names(param_re_pairs)) {
      re_list <- param_re_pairs[[param_name]]
      if (is.null(re_list)) next
      
      for (g in names(re_list)) {
        re <- re_list[[g]]
        if (!is.null(re$cor_id) && re$cor_id == cor_id) {
          members[[length(members) + 1]] <- list(param = param_name, group = g, dim = re$dim)
          if (is.null(group)) group <- g
        }
      }
    }
    
    groups <- unique(vapply(members, `[[`, "", "group"))
    if (length(groups) > 1L) {
      stop("Correlation label '|", cor_id, "|' is used with different grouping factors (",
           paste(groups, collapse = ", "), "). Effects can only be correlated within one grouping factor: ",
           "use a separate label for each, e.g. (1|", cor_id, "1|", groups[1], ") and (1|", cor_id, "2|", groups[2], ").",
           call. = FALSE)
    }
    if (length(members) > 0) {
      cross_cor[[cor_id]] <- list(
        group = group,
        members = members,
        total_dim = sum(sapply(members, `[[`, "dim"))
      )
    }
  }
  
  cross_cor
}

#' Print Method for SDT Model
#' @param x The object to print.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @export
print.broc_model <- function(x, ...) {
  cat("bayesroc model\n")
  cat("==============\n")
  cat("Family:", family_display_name(x$family$family), "\n\n")
  cat("Formula:\n")
  attr(x$parsed, ".broc_family") <- x$family
  print(x$parsed)

  cat("\nData dimensions:\n")
  cat("  N =", x$model_data$N, "\n")
  cat("  K =", x$model_data$K, "\n")

  ext <- if (!is.null(x$family$external_aliases)) x$family$external_aliases
         else list()
  display_name <- function(internal) {
    if (!is.null(ext[[internal]])) ext[[internal]] else internal
  }

  param_order <- c("dprime", "sigma", "lambda", "dprime2", "sigma2",
                   "dprime_B", "lambda_B",
                   "discrim", "discrim_B", "sigma_B", "sigma2_B",
                   "rho", "rho_B", "rho_N", "lambda2",
                   "criterion", "criterion2",
                   "rec_crit", "know_crit",
                   "dprime_L", "sigma_L", "lambda_L", "criterion_scale", "criterion2_scale")
  for (pname in param_order) {
    fe <- x$model_data[[paste0(pname, "_fixed")]]
    has_flag <- if (pname == "dprime") isTRUE(x$model_data$has_dprime_fixed)
                else if (pname == "criterion") !is.null(x$model_data$criterion)
                else if (pname == "criterion2") !is.null(x$model_data$criterion2)
                else isTRUE(x$model_data[[paste0("has_", pname)]])
    if (!isTRUE(has_flag)) next
    n_coef <- if (!is.null(fe$n_coef)) fe$n_coef
              else if (pname == "criterion") x$model_data$criterion$n_coef
              else if (pname == "criterion2") x$model_data$criterion2$n_coef
              else NA
    if (is.na(n_coef)) next
    cat(sprintf("  P_%s = %d\n", display_name(pname), n_coef))
  }
  invisible(x)
}

#' Extract Generated Stan Code
#'
#' Returns the Stan program generated by [broc()] for inspection or manual use.
#'
#' @param x A `broc_model` object created by [broc()].
#' @return A character string of class `stan_code` containing the generated Stan program.
#' @seealso [get_stan_data()] for the matching data list; [get_numpyro_config()]
#'   for the equivalent artifact used by the JAX backend.
#' @examples
#' \dontrun{
#' model <- broc(brf(conf | old ~ cond), data = dat, family = evsd())
#' get_stan_code(model)
#' }
#' @export
get_stan_code <- function(x) {
  if (!inherits(x, "broc_model")) stop("x must be an broc_model object")
  x$stan_code
}

#' Extract Stan Data List
#'
#' Returns the data list that would be passed to Stan's `$sample()` method.
#'
#' @param x A `broc_model` object created by [broc()].
#' @return A named list of data values formatted for Stan.
#' @seealso [get_stan_code()] for the generated program; [get_numpyro_config()]
#'   for the equivalent artifact used by the JAX backend.
#' @examples
#' \dontrun{
#' model <- broc(brf(conf | old ~ cond), data = dat, family = evsd())
#' str(get_stan_data(model))
#' }
#' @export
get_stan_data <- function(x) {
  if (!inherits(x, "broc_model")) stop("x must be an broc_model object")
  x$stan_data
}

#' Extract the Generated NumPyro (JAX) Config
#'
#' Returns the config list that [broc()] generates for the JAX backend, the
#' counterpart to [get_stan_code()]/[get_stan_data()]. It describes the model
#' (family, link, response data, design matrices, priors, threshold and
#' random-effect structure) and is serialized to JSON for [fit_broc()] with
#' `backend = "jax"`.
#'
#' @param x A `broc_model` object created by [broc()].
#' @param json If `TRUE`, return the pretty-printed JSON string actually written
#'   for the Python backend (matrices expanded to row lists), of class `json`.
#'   If `FALSE` (default), return the R list.
#' @return The config as a named list, or, when `json = TRUE`, a length-one
#'   character vector of class `json`.
#' @seealso [get_stan_code()], [get_stan_data()].
#' @examples
#' \dontrun{
#' model <- broc(brf(conf | old ~ cond), data = dat, family = evsd())
#' cfg <- get_numpyro_config(model)
#' str(cfg)
#' # Exactly what fit_broc(backend = "jax") sends to Python:
#' cat(get_numpyro_config(model, json = TRUE))
#' }
#' @export
get_numpyro_config <- function(x, json = FALSE) {
  if (!inherits(x, "broc_model")) stop("x must be an broc_model object")
  config <- build_numpyro_config(x)
  if (!isTRUE(json)) return(config)

  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("Package 'jsonlite' is required for json = TRUE.")
  }
  config_json <- rapply(config, function(v) {
    if (is.matrix(v)) lapply(seq_len(nrow(v)), function(i) as.numeric(v[i, ])) else v
  }, how = "replace")
  js <- jsonlite::toJSON(config_json, auto_unbox = TRUE, digits = 17,
                         null = "null", pretty = TRUE)
  structure(as.character(js), class = "json")
}

#' Compile and Fit a Bayesian ROC Model
#'
#' Takes a `broc_model` object (from [broc()]) and runs MCMC sampling via
#' Stan (cmdstanr) or JAX (NumPyro). Returns a `broc_fit` object that wraps
#' the backend-specific result and provides a uniform interface for downstream
#' analysis (`summary(fit)`, `predict(fit)`, `pp_check(fit)`, etc.).
#'
#' @param model A `broc_model` object created by [broc()].
#' @param chains Number of MCMC chains (default 4).
#' @param parallel_chains Number of chains to run in parallel (default 4). For
#'   the JAX backend this is all-or-nothing: chains run in parallel across
#'   devices when `parallel_chains >= chains`, and sequentially on one device
#'   when `parallel_chains < chains`.
#' @param iter_warmup Number of warmup/adaptation iterations per chain. Default
#'   2000 for `sampler = "nuts"`, 2500 for `"drghmc"`.
#' @param iter_sampling Number of post-warmup draws per chain (default 2000 for
#'   `sampler = "nuts"`, 10000 for `"drghmc"`), or `"auto"` to sample until every
#'   monitored parameter reaches the ESS and R-hat targets in `control` (JAX backend;
#'   see *Adaptive-length sampling* below).
#' @param thin Period for saving samples (default 1). `thin = k` keeps every
#'   `k`-th post-warmup draw. With `iter_sampling = "auto"` the draws are thinned
#'   as they are sampled and `control$max_kept` is ignored.
#' @param sampler Sampling algorithm: `"nuts"` (default) or `"drghmc"`, a
#'   delayed-rejection generalized HMC sampler (JAX backend only).
#' @param control Named list of sampler tuning parameters (defaults in
#'   parentheses; an unknown key is an error):
#'   * `"nuts"`: `adapt_delta` (0.8), `max_treedepth` (10), `step_size` (1.0),
#'     `precision` ("f64" or "f32"; JAX backend only), `transport` (`TRUE`; JAX backend
#'     only; see *Group transport map*).
#'   * `"drghmc"`: `acc_target` (0.92), `gamma` (0.32), `integrator`
#'     ("bcss3" or "bcss2"), `dr_k` (2), `dr_r` (4), `adapt_windows` (10),
#'     `eps_c` (1.0), `mass_burn_frac` (0.4), `mass_end_frac` (0.7), `precision`
#'     ("f64" or "f32"), `transport` (`TRUE`: random effects are sampled through a
#'     per-group transport map; see *Group transport map*), `transport_wide` (`TRUE`:
#'     every random-effect block is mapped; `FALSE` leaves a block of more than ten
#'     effects per group non-centred).
#'   * `iter_sampling = "auto"` (either sampler): `ess_target` (400; one number, or
#'     named values for `pop`, `sd` and `cor`, e.g. `c(pop = 1000, sd = 400)`),
#'     `rhat_target` (1.01), `max_draws` (draws per chain; 50000 for NUTS, 500000 for
#'     DR-G-HMC), `max_time` (seconds of sampling after warmup; none), `check_every`
#'     (draws per chain between checks; 100 for NUTS, 1000 for DR-G-HMC) and
#'     `max_kept` (draws per chain kept in the fit; default `max_draws`, i.e. no
#'     thinning).
#' @param init Initial-value strategy. Default `0`: all parameters start at 0 on
#'   the unconstrained scale. Other options: `"prior"` samples each parameter
#'   from its prior; `"random"` uses Stan's U(-2, 2) on the unconstrained scale
#'   (can be unstable for log-linked parameters); a single number `x` sets a
#'   U(-x, x) initialization radius; or a function or list of per-chain init lists.
#' @param refresh How often the Stan backend prints progress, in iterations
#'   (default 100). Stan only; the JAX progress bar has no refresh-rate control.
#' @param threads_per_chain Number of threads per chain for within-chain parallelism
#'   via `reduce_sum` (Stan backend). When greater than 1 the model is compiled with
#'   a `reduce_sum` likelihood; `NULL` (default) or 1 runs each chain on one thread.
#' @param backend Inference backend: `"stan"` (default, via cmdstanr) or `"jax"`
#'   (via NumPyro). The JAX backend requires a configured Python environment
#'   (see [install_jax_backend()]). It caches compiled XLA programs under
#'   `tools::R_user_dir("bayesroc", "cache")`; set
#'   `options(bayesroc.xla_cache = FALSE)` to disable the cache or to a path to
#'   move it.
#' @param python Optional path to the Python executable for the JAX backend.
#'   `NULL` (default) uses the managed environment (see [install_jax_backend()]).
#' @param progress_bar If `TRUE` (default), show the sampler's progress. On the JAX backend
#'   this is one bar per phase (warmup, sampling and, with `iter_sampling = "auto"`, each
#'   round between convergence checks), redrawn in place in the console and printed at each
#'   quarter instead when output goes to a file or a knitted document.
#' @param seed Optional integer seed for reproducible sampling. `NULL` (default)
#'   uses a random seed.
#' @param file Optional path to save the fitted model to. When set,
#'   the fit is written there with [save_broc_fit()] once sampling finishes (a
#'   `.rds` extension is appended if absent). If the file already exists, the
#'   saved fit is loaded and returned instead of refitting, unless
#'   `file_refit = TRUE`. `NULL` (default) does not save.
#' @param file_refit If `FALSE` (default), an existing `file` is loaded and
#'   sampling is skipped; if `TRUE`, the model is refit and the file overwritten.
#'   Ignored when `file` is `NULL`.
#' @param save_loglik Keep the per-observation log-likelihood (default `TRUE`) so
#'   [loo()] works on the fit and on a fit saved with [save_broc_fit()] or `file =`.
#'   With `FALSE` it is dropped; `loo = TRUE` still computes PSIS-LOO first and
#'   stores the result on the fit.
#' @param loo If `TRUE`, PSIS-LOO is computed at fit time and stored on the fit
#'   (`fit$.loo`, also saved with the fit). Default `FALSE`.
#' @param loo_cores Number of worker processes used when `loo = TRUE` (default 4).
#' @param loglik_precision Storage precision for the log-likelihood sidecar,
#'   `"f32"` (default) or `"f64"`. With the Stan backend the values are read from
#'   CmdStan's CSV output, so their precision is also limited by `sig_figs`.
#' @param keep_re If `TRUE`, the raw standardized random-effect draws
#'   (`u_*_raw`, the sampler's own coordinates) are kept in the fit and in saved
#'   fits. Default `FALSE`. The per-level random effects themselves (`u_*`) are
#'   always kept. Applies to both backends (Stan writes the raw draws to its output
#'   files regardless; they are left out of `$draws()`, summaries and saved fits).
#' @param re_store Codec for random-effect draws when the fit is saved: `"f64"`
#'   (default, lossless), `"f32"`, `"int16"` or `"int8"`. [save_broc_fit()] uses it unless
#'   given its own `re_store`.
#' @param resume A JAX fit of the same model to continue: its chains are extended
#'   exactly (no new warmup) and its draws kept. With `iter_sampling = "auto"`
#'   sampling continues until the targets in `control` are met (for example after a
#'   `max_time` or `max_draws` cap); with a number, that many more draws per chain are
#'   taken. The sampler and number of chains are those of `resume`. A fit saved with
#'   [save_broc_fit()] keeps its sampler state as `<file>_state.pkl` (a Python pickle,
#'   which can run code when loaded: only resume fits that bayesroc wrote on your
#'   own machine).
#' @param ... Additional arguments forwarded to cmdstanr's `$sample()` (Stan
#'   backend only; a warning is issued if any are passed with `backend = "jax"`).
#' @return A `broc_fit` object. It carries low-level accessors `$draws()`,
#'   `$summary()`, `$diagnostic_summary()`, `$loo()`, `$num_chains()`,
#'   `$metadata()`, and `$save_object()`, but S3 methods
#'   [summary()][summary.broc_fit], [plot()][plot.broc_fit],
#'   [predict()][predict.broc_fit], [pp_check()][pp_check.broc_fit], and
#'   [loo()][loo.broc_fit] are also provided.
#' @examples
#' \dontrun{
#' model <- broc(brf(conf | old ~ cond + (1|subj),
#'                   sigma ~ 1), data = dat, family = uvsd())
#' fit <- fit_broc(model)
#' fit <- fit_broc(model, backend = "jax", chains = 2)
#'
#' # Save on first run; reload instead of refitting on later runs.
#' fit <- fit_broc(model, file = "my_fit")          # writes my_fit.rds
#' fit <- fit_broc(model, file = "my_fit")          # loads my_fit.rds
#' fit <- fit_broc(model, file = "my_fit", file_refit = TRUE)      # refits
#' }
#' @section Adaptive-length sampling:
#' With `iter_sampling = "auto"` one set of chains is warmed up once and then
#' extended in rounds of `check_every` draws per chain. After each round the
#' rank-normalised bulk and tail ESS and R-hat (Vehtari et al., 2021, the estimators
#' [summary()] reports) are computed for every monitored parameter: population-level
#' (`pop`: fixed effects, thresholds and other family parameters), random-effect and
#' smooth SDs (`sd`) and random-effect correlations (`cor`). Sampling stops once
#' every class meets its ESS target and every R-hat is at most `rhat_target`, or at
#' `max_draws` or `max_time`. One line per round reports progress and what is still
#' unmet. The next round samples while the check runs; when a check takes longer
#' than a round, checks are spaced out so sampling does not wait for them.
#'
#' By default every draw is kept. `thin = k` keeps every `k`-th draw as it is
#' sampled. Alternatively `max_kept` caps the draws kept per chain: once a run would
#' exceed it, the kept draws are thinned evenly (every 2nd, then 4th, ... draw). In
#' every case the stopping rule is evaluated on exactly the draws the fit keeps, so
#' `summary()` shows the ESS and R-hat the rule saw. Divergence and other sampler
#' diagnostics count every transition. A message reports the memory used by the
#' kept draws once it passes 4 GB, and again each time it doubles. The result carries attributes
#' `"stop_reason"`, `"ess_trace"` (one row per round), `"targets"`,
#' `"draws_per_chain"` and `"thin_stride"`; a warning is issued when a cap stops the
#' run first.
#' @section Group transport map:
#' With many groups, or many observations per group, the standard deviations and
#' correlations of random effects are tied to the groups' standardised effects through the
#' likelihood, and a sampler has to move all of them together to change either. DR-G-HMC
#' removes most of that tie by sampling each group's effects in coordinates built from a
#' Gaussian approximation of that group's likelihood: part of the warmup runs in the
#' model's own coordinates, the approximation is built from where the chains are, and it is
#' rebuilt once before the step size is settled. Effects of one grouping factor are mapped
#' together, and they follow the population coefficients and any enclosing or crossed
#' factor they share predictor terms with. The map is a one-to-one change of variables
#' with its Jacobian, so the posterior of every model parameter is unchanged; draws are
#' returned in the model's own parameters. Each gradient costs more, by an amount that
#' grows with the number of groups relative to the number of observations. NUTS on the
#' JAX backend samples in the same coordinates, its map built by a DR-G-HMC warmup that
#' runs before its own (in float32 a float64 process builds it). Both samplers draw each
#' random-effect standard deviation on the quantile scale of its prior. Set
#' `control = list(transport = FALSE)` to sample the standardised effects directly.
#'
#' A block of more than ten effects per group (source criteria that vary by response
#' set, or condition-specific effects on several parameters) carries most of the map's
#' cost, which grows with the cube of the block's width. Every block is mapped by default;
#' `control = list(transport_wide = FALSE)` leaves every such block non-centred.
#' @references Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C.
#'   (2021). Rank-normalization, folding, and localization: An improved R-hat for
#'   assessing convergence of MCMC. *Bayesian Analysis*, 16(2), 667-718.
#' @seealso [broc()] for model specification, `summary(fit)` for results.
#' @export
fit_broc <- function(model,
                    chains = 4,
                    parallel_chains = 4,
                    iter_warmup = NULL,
                    iter_sampling = NULL,
                    thin = 1,
                    sampler = c("nuts", "drghmc"),
                    control = list(),
                    init = 0,
                    refresh = 100,
                    threads_per_chain = NULL,
                    backend = c("stan", "jax"),
                    python = NULL,
                    progress_bar = TRUE,
                    seed = NULL,
                    file = NULL,
                    file_refit = FALSE,
                    save_loglik = TRUE,
                    loo = FALSE,
                    loo_cores = 4,
                    loglik_precision = c("f32", "f64"),
                    keep_re = FALSE,
                    re_store = c("f64", "f32", "int16", "int8"),
                    resume = NULL,
                    ...) {
  loglik_precision <- match.arg(loglik_precision)
  re_store <- match.arg(re_store)
  if (!is.logical(save_loglik) || length(save_loglik) != 1L || is.na(save_loglik)) {
    stop("`save_loglik` must be TRUE or FALSE.", call. = FALSE)
  }

  if (!is.null(resume)) {
    if (!inherits(resume, "broc_fit")) stop("`resume` must be a broc_fit.", call. = FALSE)
    if (missing(backend)) backend <- "jax"
    if (missing(sampler)) sampler <- resume$.sampler %||% "nuts"
    if (missing(chains)) chains <- resume$.num_chains
    if (missing(parallel_chains)) parallel_chains <- chains
    prev <- attr(resume, "sampler_control")
    if (!is.null(prev)) {
      control <- c(control, prev$control[setdiff(names(prev$control), names(control))])
      for (k in c("adapt_delta", "max_treedepth", "step_size")) {
        if (is.null(control[[k]]) && !is.null(prev[[k]]) && sampler == "nuts") control[[k]] <- prev[[k]]
      }
    }
  }
  backend <- match.arg(backend)
  sampler <- match.arg(sampler)
  auto <- identical(iter_sampling, "auto")
  if (!auto && !is.null(iter_sampling) &&
      (!is.numeric(iter_sampling) || length(iter_sampling) != 1L || !is.finite(iter_sampling) ||
       iter_sampling < 1 || iter_sampling != round(iter_sampling))) {
    stop("`iter_sampling` must be a positive whole number of draws per chain, or \"auto\".",
         call. = FALSE)
  }
  if ((auto || !is.null(resume)) && backend != "jax") {
    stop(if (auto) "iter_sampling = \"auto\"" else "`resume`",
         " needs backend = \"jax\": continuing chains exactly needs sampler state that ",
         "cmdstanr cannot return.", call. = FALSE)
  }
  if (sampler == "drghmc" && backend != "jax") {
    stop("sampler = 'drghmc' requires backend = 'jax'.", call. = FALSE)
  }
  if (!is.logical(file_refit) || length(file_refit) != 1L || is.na(file_refit)) {
    stop("`file_refit` must be TRUE or FALSE.", call. = FALSE)
  }

  if (!is.null(file)) {
    if (!is.character(file) || length(file) != 1L) {
      stop("`file` must be a single file path (character).", call. = FALSE)
    }
    if (!grepl("\\.rds$", file, ignore.case = TRUE)) file <- paste0(file, ".rds")
    if (file.exists(file) && !file_refit) {
      message("Loading existing fit from ", file,
              " (set file_refit = TRUE to refit).")
      return(load_broc_fit(file))
    }
  }
  .save_if_requested <- function(fit) {
    fit$.re_store <- re_store
    if (!is.null(file)) save_broc_fit(fit, file, verbose = FALSE)
    fit
  }


  if (is.null(iter_warmup))   iter_warmup   <- if (sampler == "drghmc") 2500L else 2000L
  if (is.null(iter_sampling)) iter_sampling <- if (sampler == "drghmc") 10000L else 2000L

  .ctrl <- if (is.list(control)) control else list()
  .auto_given <- intersect(names(.ctrl), .auto_keys)
  if (length(.auto_given) && !auto) {
    stop("control$", paste(.auto_given, collapse = ", control$"), " only applies with ",
         "iter_sampling = \"auto\".", call. = FALSE)
  }
  .auto <- if (auto) .auto_settings(.ctrl, sampler, chains, as.integer(thin)) else NULL
  .ctrl <- .ctrl[setdiff(names(.ctrl), .auto_keys)]
  .nuts_keys <- c("adapt_delta", "max_treedepth", "step_size", "precision", "transport")
  .drghmc_keys <- c("acc_target", "gamma", "integrator", "dr_k", "dr_r",
                    "adapt_windows", "eps_c", "mass_burn_frac", "mass_end_frac",
                    "precision", "transport", "transport_links", "transport_wide")
  if (!is.null(.ctrl[["transport"]]) && !(is.logical(.ctrl[["transport"]]) && length(.ctrl[["transport"]]) == 1L &&
                                          !is.na(.ctrl[["transport"]]))) {
    stop("`control$transport` must be TRUE or FALSE.", call. = FALSE)
  }
  if (!is.null(.ctrl$transport_wide) && !(is.logical(.ctrl$transport_wide) && length(.ctrl$transport_wide) == 1L &&
                                          !is.na(.ctrl$transport_wide))) {
    stop("`control$transport_wide` must be TRUE or FALSE.", call. = FALSE)
  }
  .valid <- if (sampler == "drghmc") .drghmc_keys else .nuts_keys
  .unknown <- setdiff(names(.ctrl), .valid)
  if (length(.unknown)) {
    stop("Unknown control key(s) for sampler '", sampler, "': ",
         paste(.unknown, collapse = ", "), ". Valid keys: ",
         paste(.valid, collapse = ", "), call. = FALSE)
  }
  if (sampler == "drghmc" && is.null(resume)) {
    .windows <- .ctrl$adapt_windows %||% 10L
    if (iter_warmup < .windows) {
      stop("sampler = 'drghmc' needs iter_warmup >= control$adapt_windows (", .windows,
           "); got ", iter_warmup, ".", call. = FALSE)
    }
  }
  if (sampler == "nuts" && !is.null(.ctrl$precision) && backend != "jax") {
    warning("control$precision is JAX-only; ignored for the Stan backend.",
            call. = FALSE)
  }
  if (sampler == "nuts" && !is.null(.ctrl[["transport"]]) && backend != "jax") {
    warning("control$transport is JAX-only; ignored for the Stan backend.",
            call. = FALSE)
  }
  adapt_delta   <- .ctrl$adapt_delta   %||% 0.8
  max_treedepth <- .ctrl$max_treedepth %||% 10
  step_size     <- .ctrl$step_size

  # --- Init strategy normalization ---
  init_strategy_jax <- NULL
  if (inherits(model, "broc_model")) {
    if (is.character(init) && length(init) == 1) {
      if (init == "prior") {
        if (backend == "stan") {
          init <- prior_init(model, chains = chains, seed = seed)
        } else {
          init_strategy_jax <- "prior"
          init <- 0
        }
      } else if (init == "random") {
        if (backend == "stan") {
          init <- 2
        } else {
          init_strategy_jax <- "random"
          init <- 2
        }
      } else {
        stop("`init` string must be 'prior' or 'random'; got '", init, "'.",
             call. = FALSE)
      }
    } else if (is.numeric(init) && length(init) == 1) {
      init_strategy_jax <- "numeric"
    }
  }
  init_values_jax <- NULL
  if (backend == "jax" && (is.function(init) || is.list(init))) {
    init_values_jax <- .jax_init_values(init, chains)
    init_strategy_jax <- "numeric"
    init <- 2
  }

  if (!inherits(model, "broc_model")) {
    if (inherits(model, "brf")) {
      stop("`model` must be a broc_model object built with broc(), but you ",
           "passed a brf() formula composition. Build the model first with ",
           "`broc(<your brf>, data, ...)` and pass the result to fit_broc().",
           call. = FALSE)
    }
    stop("`model` must be a broc_model object built with broc(); got an ",
         "object of class \"", paste(class(model), collapse = "/"), "\".",
         call. = FALSE)
  }

  if (backend == "jax") {
    # ---- JAX/NumPyro backend ----
    if (length(list(...)) > 0) {
      warning("Arguments in `...` are forwarded to cmdstanr (Stan backend) only ",
              "and are ignored by the JAX backend.", call. = FALSE)
    }
    chain_method <- if (parallel_chains >= chains) "parallel" else "sequential"
    if (auto || !is.null(resume)) {
      fit <- .fit_auto(model, chains = chains, chain_method = chain_method,
                       iter_warmup = iter_warmup, sampler = sampler, ctrl = .ctrl,
                       auto = .auto %||% list(), fixed_draws = if (auto) NULL else iter_sampling,
                       thin = thin, init = init, init_strategy = init_strategy_jax,
                       init_values = init_values_jax, python = python,
                       progress_bar = progress_bar, seed = seed,
                       save_loglik = save_loglik || isTRUE(loo),
                       loglik_precision = loglik_precision, keep_re = keep_re,
                       adapt_delta = adapt_delta, max_treedepth = max_treedepth,
                       step_size = step_size, resume = resume)
      return(.save_if_requested(.finalise_loglik(fit, loo = loo, loo_cores = loo_cores,
                                                 save_loglik = save_loglik,
                                                 state_path = fit$.state_path)))
    }
    t0 <- proc.time()["elapsed"]
    jax_result <- fit_jax(model,
      chains = chains,
      warmup = iter_warmup,
      samples = iter_sampling,
      thin = thin,
      init = init,
      init_strategy = init_strategy_jax,
      seed = seed,
      adapt_delta = adapt_delta,
      max_treedepth = max_treedepth,
      step_size = step_size,
      sampler = sampler,
      control = .ctrl,
      save_loglik = save_loglik || isTRUE(loo),
      chain_method = chain_method,
      python = python,
      progress_bar = progress_bar,
      loglik_precision = loglik_precision,
      keep_re = keep_re,
      init_values = init_values_jax,
      save_state = TRUE
    )
    elapsed <- as.numeric(proc.time()["elapsed"] - t0)

    src <- new.env()
    src$samples <- .jax_samples_as_stan(jax_result$samples, model)
    jax_result$samples <- NULL
    kept <- dim(src$samples[[1]])[2]
    draws_arr <- numpyro_samples_to_draws_array(src, n_chains = chains, n_draws = kept)
    rm(src)

    fit <- new_broc_fit(
      backend_fit = jax_result,
      model = model,
      backend = "jax",
      num_chains = chains,
      iter_sampling = kept,
      elapsed = elapsed,
      draws_array = draws_arr,
      max_treedepth = max_treedepth,
      sampler = sampler,
      save_loglik = save_loglik,
      loglik_path = jax_result$loglik_path,
      keep_re = keep_re
    )
    fit$.thin <- as.integer(thin)
    attr(fit, "draws_per_chain") <- as.integer(iter_sampling)
    attr(fit, "thin_stride") <- as.integer(thin)
    attr(fit, "sampler_control") <- list(adapt_delta = adapt_delta, max_treedepth = max_treedepth,
                                         step_size = step_size, control = .ctrl)
    attr(fit, "model_fingerprint") <- .model_fingerprint(model)
    return(.save_if_requested(.finalise_loglik(fit, loo = loo, loo_cores = loo_cores,
                                               save_loglik = save_loglik,
                                               state_path = jax_result$state_path)))
  }

  # ---- Stan backend (default) ----
  if (!requireNamespace("cmdstanr", quietly = TRUE)) {
    stop("Package 'cmdstanr' is required.")
  }

  t0 <- proc.time()["elapsed"]

  stan_refresh <- if (isTRUE(progress_bar)) refresh else 0
  model$stan_data$N_ll <- if (isTRUE(save_loglik) || isTRUE(loo)) as.integer(model$stan_data$N) else 0L

  if (!is.null(threads_per_chain) && threads_per_chain > 1) {
    model <- .with_threads(model)
    mod <- .compile_broc_stan(model, threads = TRUE)
    tpc <- as.integer(threads_per_chain)
    N <- model$stan_data$N
    model$stan_data$grainsize <- max(1L, as.integer(N / (2L * tpc)))
    stan_fit <- .stan_sample(mod,
      data = model$stan_data,
      chains = chains,
      parallel_chains = parallel_chains,
      iter_warmup = iter_warmup,
      iter_sampling = iter_sampling,
      thin = thin,
      adapt_delta = adapt_delta,
      max_treedepth = max_treedepth,
      step_size = step_size,
      init = init,
      refresh = stan_refresh,
      seed = seed,
      threads_per_chain = tpc,
      ...
    )
  } else {
    mod <- .compile_broc_stan(model)
    stan_fit <- .stan_sample(mod,
      data = model$stan_data,
      chains = chains,
      parallel_chains = parallel_chains,
      iter_warmup = iter_warmup,
      iter_sampling = iter_sampling,
      thin = thin,
      adapt_delta = adapt_delta,
      max_treedepth = max_treedepth,
      step_size = step_size,
      init = init,
      refresh = stan_refresh,
      seed = seed,
      ...
    )
  }

  elapsed <- as.numeric(proc.time()["elapsed"] - t0)
  stan_fit <- .recover_dropped_chains(stan_fit, chains)

  stan_side <- NULL
  if (isTRUE(save_loglik) || isTRUE(loo)) {
    stan_side <- tryCatch(
      .stan_loglik_sidecar(stan_fit, tempfile("bayesroc_ll_", fileext = ".npy"),
                           loglik_precision),
      error = function(e) {
        warning("Could not build the log_lik sidecar (", conditionMessage(e),
                "); loo() will read from the CmdStan draws instead.",
                call. = FALSE)
        NULL
      })
  }

  .save_if_requested(.finalise_loglik(new_broc_fit(
    backend_fit = stan_fit,
    model = model,
    backend = "stan",
    num_chains = chains,
    iter_sampling = iter_sampling,
    elapsed = elapsed,
    sampler = sampler,
    save_loglik = save_loglik,
    loglik_path = stan_side,
    keep_re = keep_re
  ), loo = loo, loo_cores = loo_cores, save_loglik = save_loglik))
}

.finalise_loglik <- function(fit, loo, loo_cores, save_loglik,
                             state_path = NULL) {
  fit$.state_path <- state_path
  if (isTRUE(loo)) {
    res <- loo.broc_fit(fit, cores = loo_cores)
    attr(res, "broc_loo_type") <- "trial"
    fit$.loo <- res
    if (!isTRUE(save_loglik)) fit <- suppressMessages(drop_loglik(fit))
  }
  fit
}

#' Apply Encoding Factors to Formula
#' 
#' Wraps specified factor names in encoding() in the formula.
#' @param formula A formula
#' @param encoding_vars Character vector of variable names to wrap
#' @return Modified formula with encoding() wrappers
#' @noRd
apply_encoding_vars <- function(formula, encoding_vars) {
  if (is.null(formula) || is.null(encoding_vars) || length(encoding_vars) == 0) {
    return(formula)
  }
  
  f_str <- paste(deparse(formula, width.cutoff = 500), collapse = " ")
  
  for (fac in encoding_vars) {
    if (grepl(paste0("encoding\\s*\\(\\s*", fac), f_str)) next

    chars <- strsplit(f_str, "")[[1]]
    n <- length(chars)
    in_smooth <- logical(n)
    i <- 1
    while (i <= n) {
      rest <- substr(f_str, i, n)
      if (grepl("^(s|t2)\\s*\\(", rest)) {
        start <- regexpr("\\(", rest) + i - 1
        depth <- 1
        j <- start + 1
        while (j <= n && depth > 0) {
          if (chars[j] == "(") depth <- depth + 1
          if (chars[j] == ")") depth <- depth - 1
          j <- j + 1
        }
        in_smooth[i:(j-1)] <- TRUE
        i <- j
      } else {
        i <- i + 1
      }
    }

    pattern <- paste0("(?<!encoding\\()\\b", fac, "\\b")
    matches <- gregexpr(pattern, f_str, perl = TRUE)[[1]]
    if (matches[1] > 0) {
      for (mi in rev(seq_along(matches))) {
        pos <- matches[mi]
        match_len <- attr(matches, "match.length")[mi]
        if (!any(in_smooth[pos:(pos + match_len - 1)])) {
          f_str <- paste0(
            substr(f_str, 1, pos - 1),
            paste0("encoding(", fac, ")"),
            substr(f_str, pos + match_len, nchar(f_str))
          )
        }
      }
    }
  }
  
  as.formula(f_str, env = environment(formula))
}

#' Fit Cumulative Ordinal Model
#'
#' Routed through the EVSDT path with all observations treated as "old" and the
#' latent mean fixed at 0; every effect is on the cutpoints.
#' @noRd
fit_cumulative <- function(mu, cutpoints, data, family, priors, counts = NULL,
                           gap_link = "log", smooth_info = FALSE, cutpoints_scale = NULL) {
  
  if (is.null(mu)) {
    stop("mu formula is required for cumulative() family")
  }
  if (is.null(cutpoints)) cutpoints <- ~ 1
  
  mu_str <- paste(deparse(mu, width.cutoff = 500), collapse = " ")
  if (!grepl("~", mu_str)) {
    stop("mu formula must be two-sided: response ~ predictors")
  }
  
  parts <- strsplit(mu_str, "~")[[1]]
  response_var <- trimws(parts[1])
  mu_rhs <- trimws(parts[2])
  
  if (!response_var %in% names(data)) {
    stop("Response variable '", response_var, "' not found in data")
  }
  
  
  mu_rhs_final <- .cumulative_rhs(mu_rhs, "cutpoints")

  data$.cumulative_is_old <- 1L
  
  dprime_formula <- as.formula(
    paste0(response_var, " | .cumulative_is_old ~ ", mu_rhs_final),
    env = environment(mu)
  )
  
  criterion_formula <- cutpoints
  
  parsed <- parse_broc_formula(
    dprime = dprime_formula,
    criterion = criterion_formula,
    sigma = NULL,
    lambda = NULL,
    dprime2 = NULL,
    sigma2 = NULL,
    criterion_scale = cutpoints_scale,
    data = data,
    family_name = "cumulative"
  )
  
  parsed$is_cumulative <- TRUE
  parsed$encoding_vars <- character(0)
  
  prior_lookup <- build_prior_lookup(priors, family = family)
  
  
  model_data <- build_model_data(parsed, data, family, counts = counts,
                                 encoding_vars = NULL)
  model_data$gap_link <- gap_link
  prior_lookup <- .finish_prior_lookup(prior_lookup, model_data)
  model_data$prior_lookup <- prior_lookup
  model_data$is_cumulative <- TRUE
  model_data$gap_link <- gap_link

  si <- .apply_smooth_info(smooth_info, model_data, family, gap_link, data)
  model_data <- si$model_data

  code <- .generate_model_code(model_data, family, "cumulative")

  result <- list(
    parsed = parsed,
    model_data = model_data,
    stan_code = code$stan_code,
    stan_data = code$stan_data,
    family = family,
    prior_lookup = prior_lookup,
    data = data,
    batch_likelihood = code$batch_likelihood,
    batch_cpp = code$batch_cpp,
    smooth_info = si$pilot
  )

  class(result) <- c("broc_model", "cumulative_model", "list")
  result
}

#' Rename parameters in Stan code for cumulative models
#' @noRd
rename_cumulative_params <- function(stan_code) {
  stan_code <- gsub("dprime", "mu", stan_code)
  stan_code <- gsub("d'", "mu", stan_code)
  
  stan_code <- gsub("criterion", "cutpoints", stan_code)
  stan_code <- gsub("Criterion", "Cutpoints", stan_code)
  
  stan_code <- gsub("// mu fixed effects", "// Mu fixed effects", stan_code)
  stan_code <- gsub("// Cutpoints fixed effects", "// Cutpoints fixed effects", stan_code)
  
  stan_code
}

#' Rename stan_data keys for cumulative models
#' @noRd
rename_stan_data_cumulative <- function(stan_data) {
  old_names <- names(stan_data)
  
  new_names <- gsub("dprime", "mu", old_names)
  
  new_names <- gsub("criterion", "cutpoints", new_names)
  
  names(stan_data) <- new_names
  stan_data
}

#' Drop Reference Level from Random Slopes for Cumulative Models
#' 
#' In cumulative models, random slopes for the reference level of a factor
#' are not identified separately from cutpoints random intercepts (both shift
#' location for those observations). This function drops the reference level
#' column from random slopes, giving treatment-style random effects.
#' 
#' @param re_list List of random effects structures from build_random_effects_structure
#' @param data The data frame
#' @return Modified re_list with reference level columns removed from varying slopes
#' @noRd
drop_reference_level_from_random_slopes <- function(re_list, data) {
  
  if (is.null(re_list) || length(re_list) == 0) {
    return(re_list)
  }
  
  any_dropped <- FALSE
  
  for (group in names(re_list)) {
    re <- re_list[[group]]
    
    if (re$type != "varying_slope" || is.null(re$term)) {
      next
    }
    
    term_var <- re$term
    if (!term_var %in% names(data)) {
      next
    }
    
    if (!is.factor(data[[term_var]])) {
      next
    }
    
    ref_level <- levels(data[[term_var]])[1]
    ref_col_name <- paste0(term_var, ref_level)
    
    ref_idx <- which(re$level_names == ref_col_name)
    
    if (length(ref_idx) == 0) {
      ref_idx <- which(re$level_names == ref_level)
    }
    
    if (length(ref_idx) == 0 || re$dim <= 1) {
      next
    }
    
    any_dropped <- TRUE
    
    keep_idx <- setdiff(1:re$dim, ref_idx)
    
    re$dim <- length(keep_idx)
    re$level_names <- re$level_names[keep_idx]
    
    if (!is.null(re$term_idx)) {
      new_idx_map <- integer(re$dim + 1)
      new_idx_map[ref_idx] <- 0
      new_counter <- 1
      for (i in keep_idx) {
        new_idx_map[i] <- new_counter
        new_counter <- new_counter + 1
      }
      
      re$term_idx <- new_idx_map[re$term_idx]
    }
    
    if (!is.null(re$Z) && re$use_z_matrix) {
      re$Z <- re$Z[, keep_idx, drop = FALSE]
    }
    
    re_list[[group]] <- re
  }
  
  if (any_dropped) {
    message("Note: Reference level dropped from random slopes on mu for identifiability. ",
            "In cumulative models, random slopes for the reference level are confounded ",
            "with cutpoints random intercepts.")
  }

  re_list
}

#' Fit Bivariate Cumulative Ordinal Model
#' @noRd
fit_bivariate_cumulative <- function(mu1, mu2, cutpoints, cutpoints2, rho,
                                     data, family, priors, counts = NULL,
                                     gap_link = "log", smooth_info = FALSE, cutpoints_scale = NULL,
                                     cutpoints2_scale = NULL) {

  if (is.null(mu1)) {
    stop("The primary formula resp(y1, y2) ~ ... is required for bivariate_cumulative() family")
  }
  if (is.null(cutpoints)) cutpoints <- ~ 1
  if (is.null(cutpoints2)) cutpoints2 <- ~ 1

  if (is.null(rho)) {
    message("Note: rho formula not specified. Using rho ~ 1.")
    rho <- ~ 1
  }


  mu1_str <- paste(deparse(mu1, width.cutoff = 500), collapse = " ")
  if (!grepl("~", mu1_str)) {
    stop("The primary formula must be two-sided: resp(y1, y2) ~ predictors")
  }

  parts <- strsplit(mu1_str, "~")[[1]]
  lhs <- trimws(parts[1])
  mu1_rhs <- trimws(parts[2])

  resp_match <- regmatches(lhs, regexec("resp\\(([^,]+),\\s*([^)]+)\\)", lhs))[[1]]
  if (length(resp_match) < 3) {
    stop("bivariate_cumulative() requires resp(y1, y2) on the LHS. ",
         "Got: ", lhs)
  }
  response1_var <- trimws(resp_match[2])
  response2_var <- trimws(resp_match[3])

  if (!response1_var %in% names(data)) {
    stop("Response variable '", response1_var, "' not found in data")
  }
  if (!response2_var %in% names(data)) {
    stop("Response variable '", response2_var, "' not found in data")
  }

  mu1_rhs <- .cumulative_rhs(mu1_rhs, "cutpoints")

  mu2_str <- if (is.null(mu2)) "~ 0" else paste(deparse(mu2, width.cutoff = 500), collapse = " ")
  if (grepl("~", mu2_str)) {
    mu2_rhs <- trimws(strsplit(mu2_str, "~")[[1]][2])
  } else {
    mu2_rhs <- mu2_str
  }
  mu2_rhs <- .cumulative_rhs(mu2_rhs, "cutpoints2", "The mu2 formula")

  data$.bivar_cum_item_type <- factor(rep("A", nrow(data)), levels = c("new", "A"))

  dprime_formula <- as.formula(
    paste0("resp(", response1_var, ", ", response2_var,
           ") | .bivar_cum_item_type ~ ", mu1_rhs),
    env = environment(mu1)
  )

  discrim_formula <- as.formula(paste0("~ ", mu2_rhs), env = environment(mu1))

  criterion_formula <- cutpoints
  criterion2_formula <- cutpoints2

  parsed <- parse_broc_formula(
    dprime = dprime_formula,
    criterion = criterion_formula,
    sigma = NULL,
    lambda = NULL,
    dprime2 = NULL,
    sigma2 = NULL,
    discrim = discrim_formula,
    discrim_B = NULL,
    sigma_B = NULL,
    sigma2_B = NULL,
    rho = rho,
    rho_B = NULL,
    rho_N = NULL,
    criterion2 = criterion2_formula,
    lambda2 = NULL,
    criterion_scale = cutpoints_scale,
    criterion2_scale = cutpoints2_scale,
    data = data,
    family_name = "bivariate_cumulative"
  )

  parsed$is_bivariate_cumulative <- TRUE
  parsed$encoding_vars <- character(0)

  prior_lookup <- build_prior_lookup(priors, family = family)


  model_data <- build_model_data(parsed, data, family, counts = counts)
  model_data$gap_link <- gap_link
  prior_lookup <- .finish_prior_lookup(prior_lookup, model_data)
  model_data$prior_lookup <- prior_lookup
  model_data$is_bivariate_cumulative <- TRUE
  model_data$gap_link <- gap_link

  si <- .apply_smooth_info(smooth_info, model_data, family, gap_link, data)
  model_data <- si$model_data

  code <- .generate_model_code(model_data, family, "bivariate_cumulative")

  result <- list(
    parsed = parsed,
    model_data = model_data,
    stan_code = code$stan_code,
    stan_data = code$stan_data,
    family = family,
    prior_lookup = prior_lookup,
    data = data,
    batch_likelihood = code$batch_likelihood,
    batch_cpp = code$batch_cpp,
    smooth_info = si$pilot
  )

  class(result) <- c("broc_model", "bivariate_cumulative_model", "list")
  result
}

.cumulative_rhs <- function(rhs, cut_name, what = "The response formula") {
  if (!length(attr(stats::terms(stats::as.formula(paste("~", rhs))), "term.labels"))) return("0")
  stop(what, " of a cumulative model takes no predictors; they go on the cutpoints. `", cut_name, " ~ ", rhs,
       "` gives each cutpoint its own effects, and adding a `", cut_name, "_scale` formula makes `", cut_name,
       " ~ ", rhs, "` move all cutpoints together (a positive coefficient moves responses down).", call. = FALSE)
}

#' Rename parameters in Stan code for bivariate cumulative models
#' @noRd
rename_bivariate_cumulative_params <- function(stan_code) {
  cut <- regexpr("\ndata {", stan_code, fixed = TRUE)
  if (cut > 0 && startsWith(stan_code, "functions")) {
    return(paste0(substr(stan_code, 1, cut), rename_bivariate_cumulative_params(substring(stan_code, cut + 1))))
  }
  stan_code <- gsub("dprime", "mu", stan_code)
  stan_code <- gsub("d'", "mu", stan_code)
  stan_code <- gsub("discrim", "mu2", stan_code)
  stan_code <- gsub("criterion2", "cutpoints2", stan_code)
  stan_code <- gsub("Criterion2", "Cutpoints2", stan_code)
  stan_code <- gsub("criterion", "cutpoints", stan_code)
  stan_code <- gsub("Criterion", "Cutpoints", stan_code)
  stan_code
}

#' Rename stan_data keys for bivariate cumulative models
#' @noRd
rename_stan_data_bivariate_cumulative <- function(stan_data) {
  old_names <- names(stan_data)
  new_names <- gsub("dprime", "mu", old_names)
  new_names <- gsub("discrim", "mu2", new_names)
  new_names <- gsub("criterion2", "cutpoints2", new_names)
  new_names <- gsub("criterion", "cutpoints", new_names)
  names(stan_data) <- new_names
  stan_data
}

.ls_kernel_view <- function(model_data, family) {
  fam <- if (is.list(family)) family$family else family
  N <- model_data$N
  md <- model_data
  row_re <- function(group, J) {
    list(X = matrix(1, N, 1), n_coef = 1L, coef_names = "(Intercept)", is_intercept_only = TRUE, n_thresh = J,
         random = stats::setNames(list(list(group = group, group_idx = seq_len(N), n_groups = N, dim = J,
                                            n_re_terms = 1L, type = "intercept", correlated = FALSE,
                                            level_names = "(Intercept)")), group))
  }
  drop <- character(0)
  if (isTRUE(md$has_criterion_scale)) {
    md$criterion <- row_re("lsrow", if (fam == "cdp") md$stan_data$J else md$K - 1L)
    md$has_criterion_scale <- FALSE
    md$criterion_scale_fixed <- NULL
    md$criterion_scale_random <- NULL
    drop <- c(drop, "criterion", "criterion_scale")
  }
  if (isTRUE(md$has_criterion2_scale)) {
    md$criterion2 <- row_re("lsrow2", md$K2 - 1L)
    md$varying_source_criteria <- FALSE
    md$has_criterion2_scale <- FALSE
    md$criterion2_scale_fixed <- NULL
    md$criterion2_scale_random <- NULL
    drop <- c(drop, "criterion2", "criterion2_scale")
  }
  for (p in drop) md$smooth_data[[p]] <- NULL
  if (!length(md$smooth_data)) md$smooth_data <- NULL
  md$cross_cor <- lapply(md$cross_cor, function(cc) {
    cc$members <- Filter(function(m) !m$param %in% drop, cc$members)
    cc
  })
  md
}

.ls_kernel_args <- function(batch_cpp, ls1, ls2) {
  ren <- c(if (ls1) c(X_criterion = "X_ls_one", beta_thresh_mid = "ls_zero_mid", beta_log_gaps = "ls_zero_gaps",
                      u_criterion_lsrow = "u_ls_rows", lsrow = "ls_row_idx"),
           if (ls2) c(X_criterion2 = "X_ls_one", beta_thresh_mid_2 = "ls_zero_mid", beta_log_gaps_2 = "ls_zero_gaps2",
                      u_criterion2_lsrow2 = "u_ls_rows2", lsrow2 = "ls_row_idx"))
  for (k in c("stan_call", "stan_call_vec", "stan_call_threaded")) {
    if (is.null(batch_cpp[[k]])) next
    for (nm in names(ren)) {
      batch_cpp[[k]] <- gsub(paste0("(?<![A-Za-z0-9_])", nm, "(?![A-Za-z0-9_])"), ren[[nm]], batch_cpp[[k]], perl = TRUE)
    }
  }
  batch_cpp
}

.model_kind <- function(model) {
  if (inherits(model, "bivariate_cumulative_model")) "bivariate_cumulative"
  else if (inherits(model, "cumulative_model")) "cumulative"
  else "standard"
}

.generate_model_code <- function(model_data, family, kind) {
  use_batch <- is.null(.plain_stan_env())
  batch_cpp <- NULL
  if (use_batch) {
    ls <- isTRUE(model_data$has_criterion_scale) || isTRUE(model_data$has_criterion2_scale)
    batch_cpp <- generate_batch_cpp(if (ls) .ls_kernel_view(model_data, family) else model_data, family)
    if (is.null(batch_cpp)) stop("internal: no batch likelihood for this model", call. = FALSE)
    if (ls) batch_cpp <- .ls_kernel_args(batch_cpp, isTRUE(model_data$has_criterion_scale),
                                         isTRUE(model_data$has_criterion2_scale))
  }
  stan_code <- generate_stan_code_v2(model_data, family, batch_info = batch_cpp)
  stan_data <- model_data$stan_data
  if (kind == "cumulative") {
    stan_code <- rename_cumulative_params(stan_code)
    stan_data <- rename_stan_data_cumulative(stan_data)
    if (!is.null(batch_cpp)) {
      for (k in c("stan_decl", "stan_decl_vec", "stan_call", "stan_call_vec", "stan_partial",
                  "stan_call_threaded")) {
        if (!is.null(batch_cpp[[k]])) batch_cpp[[k]] <- rename_cumulative_params(batch_cpp[[k]])
      }
      for (pr in list(c("X_dprime", "X_mu"), c("beta_dprime", "beta_mu"), c("Z_dprime_", "Z_mu_"),
                      c("u_dprime_", "u_mu_"), c("idx_dprime_", "idx_mu_"),
                      c("X_criterion", "X_cutpoints"), c("idx_criterion_", "idx_cutpoints_"),
                      c("Z_criterion_", "Z_cutpoints_"), c("u_criterion_", "u_cutpoints_"))) {
        batch_cpp$cpp_code <- gsub(pr[1], pr[2], batch_cpp$cpp_code, fixed = TRUE)
      }
    }
  } else if (kind == "bivariate_cumulative") {
    stan_code <- rename_bivariate_cumulative_params(stan_code)
    stan_data <- rename_stan_data_bivariate_cumulative(stan_data)
  }
  list(stan_code = stan_code, stan_data = stan_data, batch_cpp = batch_cpp,
       batch_likelihood = use_batch)
}

.with_threads <- function(model) {
  md <- model$model_data
  md$threads <- TRUE
  md$stan_data$grainsize <- max(1L, as.integer(md$stan_data$N / 4))
  code <- .generate_model_code(md, model$family, .model_kind(model))
  model$model_data <- md
  model$stan_code <- code$stan_code
  model$stan_data <- code$stan_data
  model$batch_cpp <- code$batch_cpp
  model
}

.compile_broc_stan <- function(model, threads = FALSE) {
  batch <- isTRUE(model$batch_likelihood) && !is.null(model$batch_cpp)
  hdr <- NULL
  deps <- character(0)
  if (batch) {
    core_hpp <- system.file("stan/batch_sdt_core.hpp", package = "bayesroc")
    if (core_hpp == "") {
      core_hpp <- file.path(find.package("bayesroc", quiet = TRUE), "stan", "batch_sdt_core.hpp")
      if (!file.exists(core_hpp)) {
        core_hpp <- file.path(system.file(package = "bayesroc"), "stan", "batch_sdt_core.hpp")
      }
    }
    hdr <- c(sprintf('#include "%s"', core_hpp), "", model$batch_cpp$cpp_code)
    deps <- c(core_hpp, file.path(dirname(core_hpp), "..", "include", "bayesroc", "bvn_tail.h"))
    deps <- deps[file.exists(deps)]
  }
  key <- digest_string(c(model$stan_code, hdr, unname(tools::md5sum(deps)), isTRUE(threads),
                         tryCatch(as.character(cmdstanr::cmdstan_version()), error = function(e) ""),
                         tryCatch(cmdstanr::cmdstan_path(), error = function(e) ""), R.version$platform))
  dir <- file.path(tools::R_user_dir("bayesroc", "cache"), "stan")
  if (!dir.exists(dir) && !dir.create(dir, recursive = TRUE, showWarnings = FALSE)) dir <- tempdir()
  base <- file.path(dir, paste0("m_", substr(key, 1, 16)))
  stan_file <- paste0(base, ".stan")
  if (!file.exists(stan_file)) writeLines(model$stan_code, stan_file)
  args <- list(stan_file)
  if (isTRUE(threads)) args$cpp_options <- list(stan_threads = TRUE)
  if (batch) {
    batch_hpp <- paste0(base, "_batch.hpp")
    if (!file.exists(batch_hpp)) writeLines(hdr, batch_hpp)
    args$user_header <- batch_hpp
    args$stanc_options <- list("allow-undefined" = TRUE)
  }
  .muffle_stan_header_warnings(do.call(cmdstanr::cmdstan_model, args))
}

.muffle_stan_header_warnings <- function(expr) {
  withCallingHandlers(expr, warning = function(w) {
    if (grepl("overloaded-virtual|model_base_crtp|was hidden", conditionMessage(w))) {
      invokeRestart("muffleWarning")
    }
  })
}

.stan_sample <- function(mod, ..., output_basename = NULL) {
  mod$sample(..., output_basename = output_basename %||% basename(tempfile(paste0(mod$model_name(), "-"))))
}

.recover_dropped_chains <- function(stan_fit, chains) {
  kept <- stan_fit$num_chains()
  if (kept >= chains) return(stan_fit)
  files <- stan_fit$output_files(include_failed = TRUE)
  rebuilt <- tryCatch(
    suppressWarnings(cmdstanr::as_cmdstan_fit(files, check_diagnostics = FALSE)),
    error = function(e) NULL)
  if (!is.null(rebuilt) && rebuilt$num_chains() == chains &&
      all(posterior::ndraws(rebuilt$draws("lp__")) ==
          posterior::ndraws(stan_fit$draws("lp__")) / kept * chains)) {
    warning(sprintf(paste0("cmdstanr reported %d of %d chains as failed, but every chain's ",
                           "output is complete; all %d chains were read back from their CSV files."),
                    chains - kept, chains, chains), call. = FALSE)
    return(rebuilt)
  }
  stop(sprintf(paste0("%d of %d Stan chains failed and their output is incomplete; ",
                      "refusing to return a fit built from a subset of chains. ",
                      "Inspect the chains' output or refit."), chains - kept, chains), call. = FALSE)
}

# The sets of source criteria and which one each detection response reads. "varying": one set
# per response. "varying_shared_new": the "new" responses all read set 1 and each "old"
# response its own set after it. Every set carries the same structure (coefficients, smooths
# and random effects).
.varying_bins <- function(family, K) {
  if (identical(family$source_criteria, "varying_shared_new")) {
    lv <- sort(unique(as.integer(family$old_levels)))
    of <- rep(1L, K)
    of[lv] <- 1L + seq_along(lv)
    labels <- c("new", as.character(lv))
  } else {
    of <- seq_len(K)
    labels <- as.character(of)
  }
  list(n = length(labels), of = of, labels = labels)
}

.varying_cells <- function(X, names, location) {
  loc <- if ("(Intercept)" %in% names) character(0) else intersect(location, names)
  if (!length(loc)) return(list(names = character(0), labels = "(Intercept)", of = rep(1L, nrow(X))))
  L <- X[, loc, drop = FALSE]
  of <- max.col(L, ties.method = "first")
  none <- rowSums(L) == 0
  labels <- loc
  if (any(none)) {
    labels <- c(labels, "(none)")
    of[none] <- length(labels)
  }
  list(names = loc, labels = labels, of = as.integer(of))
}

.vcell_of <- function(md, N = length(md$stan_data$y)) as.integer(md$stan_data$vcell %||% rep(1L, N))
.n_vcell <- function(md) as.integer(md$stan_data$n_vcell %||% 1L)
.vset_labels <- function(md) {
  b <- .vbin_labels(md)
  cl <- md$criterion2$vcell_names %||% "(Intercept)"
  if (length(cl) == 1L) return(b)
  paste0(rep(b, times = length(cl)), ":", rep(cl, each = length(b)))
}

# vbin of a model, including fits saved before the map existed (one set per response)
.vbin_of <- function(md) as.integer(md$stan_data$vbin %||% seq_len(md$K))
.vbin_labels <- function(md) {
  if (identical(md$source_criteria, "varying_shared_new")) c("new", as.character(sort(unique(as.integer(md$old_levels)))))
  else as.character(seq_len(max(.vbin_of(md))))
}

.varying_crit2_re <- function(re, mode, n_bins, K2) {
  n_terms <- as.integer(re$n_re_terms %||% 1L)
  if (!is.null(re$term_idx) && !isTRUE(re$use_z_matrix)) {
    Z <- matrix(0, length(re$term_idx), n_terms, dimnames = list(NULL, re$level_names))
    on <- which(re$term_idx > 0L)
    Z[cbind(on, re$term_idx[on])] <- 1
    re$Z <- Z
    re$use_z_matrix <- TRUE
    re$term_idx <- NULL
  }
  n_base <- switch(mode, shared = 1L, per_bin = as.integer(n_bins), full = as.integer(n_bins * (K2 - 1L)))
  re$n_re_terms <- n_terms
  re$dim <- n_base * n_terms
  re$varying_re_mode <- mode
  re$n_bins <- n_bins
  col_set <- re$col_set %||% rep(1L, n_terms)
  set_ct <- re$set_cor_thresholds %||% stats::setNames(rep(TRUE, length(unique(col_set))), unique(col_set))
  re$cor_blocks <- .re_cor_blocks(n_terms, n_base, col_set, set_ct)
  re$correlated <- length(re$cor_blocks) > 0
  re
}

.dprime_order_data <- function(X1, X2, is_old) {
  old <- if (is.null(is_old)) rep(TRUE, nrow(X1)) else is_old == max(is_old)
  x1 <- colMeans(X1[old, , drop = FALSE])
  x2 <- colMeans(X2[old, , drop = FALSE])
  list(unname(which.max(abs(x1))), as.array(unname(x1)), as.array(unname(x2)))
}

.canonical_param_formulas <- function(params, family) {
  params <- params[!vapply(params, identical_to_null_formula, logical(1))]
  m <- .param_name_map(family$family)
  for (internal in unique(c("dprime", names(m)))) {
    spell <- c(internal, m[[internal]])
    given <- intersect(spell, names(params))
    if (!length(given)) next
    if (internal == "dprime") {
      stop("A separate '", given[1], " ~' formula is not used: ", m$dprime %||% "dprime",
           " takes the right-hand side of the main formula (response ~ ...). Put its terms there.", call. = FALSE)
    }
    if (length(given) > 1L) {
      stop("Formulas for '", paste(given, collapse = "' and '"), "' name the same parameter; give only one.",
           call. = FALSE)
    }
    if (given != internal) {
      params[[internal]] <- params[[given]]
      params[[given]] <- NULL
    }
  }
  params
}

.expand_grouping <- function(formulas, data) {
  defs <- attr(data, "bayesroc_group_defs") %||% list()
  col_for <- function(vars) {
    hit <- Filter(function(d) identical(d, vars), defs)
    if (length(hit)) return(names(hit)[1])
    nm <- paste(vars, collapse = "_")
    while (nm %in% c(names(data), names(defs))) nm <- paste0(nm, "_")
    defs[[nm]] <<- vars
    data[[nm]] <<- interaction(data[vars], drop = TRUE, sep = ":")
    nm
  }
  chain <- function(e) {
    if (is.call(e) && identical(e[[1]], as.name("/"))) return(c(chain(e[[2]]), list(all.vars(e[[3]]))))
    list(all.vars(e))
  }
  group_sym <- function(vars) if (length(vars) == 1L) as.name(vars) else as.name(col_for(vars))
  expand_term <- function(e) {
    if (!(is.call(e) && identical(e[[1]], as.name("(")) && is.call(e[[2]]) &&
          as.character(e[[2]][[1]]) %in% c("|", "||"))) return(list(e))
    bar <- e[[2]]
    g <- bar[[3]]
    wrap <- function(sym) sym
    if (is.call(g) && identical(g[[1]], as.name("gr")) && length(g) >= 2L) {
      gr_call <- g
      g <- g[[2]]
      wrap <- function(sym) { out <- gr_call; out[[2]] <- sym; out }
    }
    if (is.symbol(g)) return(list(e))
    levels <- chain(g)
    nested <- Reduce(function(acc, v) c(acc, list(c(acc[[length(acc)]], v))), levels[-1], list(levels[[1]]))
    lhs <- bar[[2]]
    id <- if (is.call(lhs) && identical(lhs[[1]], as.name("|"))) as.character(lhs[[3]])
    lapply(seq_along(nested), function(i) {
      gs <- group_sym(nested[[i]])
      new_lhs <- if (!is.null(id)) {
        lab <- if (i == 1L) id else paste0(id, "_", as.character(gs))
        call("|", lhs[[2]], as.name(lab))
      } else lhs
      call("(", call(as.character(bar[[1]]), new_lhs, wrap(gs)))
    })
  }
  flat <- function(e) {
    if (is.call(e) && length(e) == 3L && identical(e[[1]], as.name("+"))) return(c(flat(e[[2]]), flat(e[[3]])))
    if (is.call(e) && length(e) == 3L && identical(e[[1]], as.name("-"))) return(list(call("-", walk(e[[2]]), e[[3]])))
    expand_term(e)
  }
  walk <- function(e) Reduce(function(a, b) call("+", a, b), flat(e))
  out <- lapply(formulas, function(f) {
    if (!inherits(f, "formula")) return(f)
    rhs <- f[[length(f)]]
    new_rhs <- walk(rhs)
    if (identical(new_rhs, rhs)) return(f)
    f[[length(f)]] <- new_rhs
    f
  })
  if (length(defs)) attr(data, "bayesroc_group_defs") <- defs
  list(formulas = out, data = data)
}

.add_group_columns <- function(newdata, defs) {
  for (nm in names(defs)) {
    if (is.null(newdata[[nm]])) newdata[[nm]] <- interaction(newdata[defs[[nm]]], drop = TRUE, sep = ":")
  }
  newdata
}
