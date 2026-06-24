#' SDT Model Specification and Fitting
#'
#' Main interface for specifying and fitting SDT models.




#' Check if a formula is equivalent to ~ NULL
#' @param x Object to check
#' @return TRUE if x is ~ NULL or equivalent
#' @noRd
identical_to_null_formula <- function(x) {
  if (!inherits(x, "formula")) return(FALSE)
  # Convert to string and check
  f_str <- paste(deparse(x), collapse = "")
  # Check for patterns like "~ NULL", "~NULL", "~ null"
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

  # Collect every variable referenced by every formula in brf_obj. all.vars()
  # walks the parse tree and returns user-supplied names while skipping
  # function names, so it handles resp(), encoding(), conditional(),
  # is_old(), s(), t2(), |group, etc. -- but it ALSO picks up the brms
  # cross-parameter correlation IDs in (terms |id| group), which aren't
  # data columns. Strip those before checking.
  formula_vars <- character(0)
  formula_sources <- character(0)  # parallel: which formula referenced each var
  add_formula_vars <- function(f, label) {
    if (is.null(f) || identical_to_null_formula(f)) return(invisible())
    vs <- tryCatch(all.vars(f), error = function(e) character(0))
    # Strip cross-cor IDs: anything appearing between two pipes inside a
    # parenthesized RE term, e.g. (... |p| participant).
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
  # Filter out the cumulative-family fake column that gets injected later
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

  # counts column must exist if specified
  if (!is.null(counts)) {
    if (!is.character(counts) || length(counts) != 1) {
      stop("`counts` must be a single column name (character).", call. = FALSE)
    }
    if (!counts %in% cols) {
      stop("`counts = \"", counts, "\"` but '", counts,
           "' is not a column in `data`.", call. = FALSE)
    }
    cv <- data[[counts]]
    if (!is.numeric(cv) || any(is.na(cv)) || any(cv < 0)) {
      stop("`counts` column '", counts, "' must be non-negative integers (no NAs).",
           call. = FALSE)
    }
  }

  # encoding_vars must exist as columns. A numeric column is allowed and
  # enters as a continuous encoding-only slope (no factor coercion).
  if (!is.null(encoding_vars)) {
    miss_enc <- setdiff(encoding_vars, cols)
    if (length(miss_enc) > 0) {
      stop("`encoding_vars` references column(s) not in `data`: ",
           paste(miss_enc, collapse = ", "), call. = FALSE)
    }
  }

  # Validate is_old/source/item_type/rk coding where known; the parser catches
  # the rest with its own messages.
  prim <- brf_obj$primary
  if (!is.null(prim) && length(as.character(prim)) >= 2) {
    lhs_str <- deparse(prim[[2]])
    # Look for "y | is_old_col" pattern
    if (grepl("\\|", lhs_str) && family$family %in%
        c("evsdt", "uvsdt", "dpsdt", "mixture", "cdp")) {
      bar_var <- trimws(strsplit(lhs_str, "\\|")[[1]][2])
      if (!grepl("^[a-zA-Z_.][a-zA-Z0-9_.]*$", bar_var)) {
        # complex expression -- skip
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

# Coerce an is_old column to numeric by its labels: factor("0", "1") -> 0/1
# (not the factor codes 1/2).
.as_is_old <- function(v) {
  if (is.factor(v)) as.numeric(as.character(v)) else as.numeric(v)
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
#' @param cor_threshold If `TRUE` (default), random effects on the K-1 thresholds
#'   are modeled jointly (correlated) via the threshold parameterization; `FALSE`
#'   treats them as independent.
#' @param threads If `TRUE`, enable within-chain `reduce_sum` threading for the
#'   Stan backend (set the thread count per chain via `fit_broc(threads_per_chain=)`).
#'   Default `FALSE`.
#' @param gap_link Link for threshold gaps: `"log"` (default) or `"softplus"`.
#'   `NULL` uses the family default.
#' @param batch_likelihood If `TRUE` (default), build the fused batch likelihood
#'   with analytic gradients for the Stan backend.
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
                cor_threshold = NULL, threads = NULL, gap_link = NULL,
                batch_likelihood = TRUE) {

  # --- 1. Normalize input: bare formula -> brf, brf -> use directly ---
  nc <- function(a, b) if (!is.null(a)) a else b  # null-coalesce helper

  if (inherits(formula, "brf")) {
    brf_obj <- formula
  } else if (inherits(formula, "formula")) {
    brf_obj <- brf(formula)
  } else if (is.null(formula)) {
    stop("formula argument is required. Provide a formula or brf() object.")
  } else {
    stop("First argument must be a formula or brf() object.")
  }

  # broc() takes ONE formula (or a brf object); it has no `...`. A multi-formula
  # call like broc(resp ~ ..., criterion ~ ...) binds the extra formula(s)
  # positionally to `priors`, which otherwise fails later with a cryptic error.
  # Catch it here with an actionable message.
  if (inherits(priors, "formula")) {
    stop("broc() accepts a single formula. For a multi-parameter model, wrap the ",
         "formulas in brf(), e.g. broc(brf(resp | is_old ~ ..., criterion ~ ...), ",
         "data, family = ...).", call. = FALSE)
  }

  # --- 2. Merge options: broc() non-NULL wins, else brf value, else defaults ---
  family <- nc(family, nc(brf_obj$family, evsd()))
  encoding_vars <- nc(encoding_vars, brf_obj$encoding_vars)
  counts <- nc(counts, brf_obj$counts)
  cor_threshold <- nc(cor_threshold, nc(brf_obj$cor_threshold, TRUE))
  threads <- nc(threads, nc(brf_obj$threads, FALSE))
  gap_link <- match.arg(nc(gap_link, nc(brf_obj$gap_link, "log")),
                         c("log", "softplus"))
  if (!inherits(family, "broc_family")) {
    stop("family must be created with evsd(), uvsd(), dpsd(), mixture(), cumulative(), bivariate_cumulative(), source_mixture(), bivariate_gaussian(), bivariate_dp(), cdp(), or vrdp2d()")
  }

  # Validate cor_threshold
  if (!is.logical(cor_threshold) || length(cor_threshold) != 1) {
    stop("cor_threshold must be TRUE or FALSE")
  }

  # --- 2b. Up-front data validation ----------------------------------------
  # Validate data before code generation so errors surface clearly here.
  validate_broc_data(data, brf_obj, family, encoding_vars, counts)

  # --- 2c. Drop zero-count rows ---------------------------------------------
  # When `counts` is specified, rows with count = 0 contribute nothing to the
  # likelihood (since L_n = count_n * log p_n = 0 regardless of p_n) but cause
  # numerical problems downstream: JAX silently produces NaN in log_lik when
  # log(p) = -Inf, breaking loo and other diagnostics; Stan throws a runtime
  # error from inside Stan. Auto-filter them out and message the user so the
  # behavior is observable.
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
  dprime <- brf_obj$primary       # The two-sided primary formula
  params <- brf_obj$params        # Named list of one-sided formulas

  # --- 3a. Reject formulas naming params not in the family ---
  # E.g. `lambda ~ 1` with `family = uvsd()` should error here, not get
  # silently dropped. Valid names = internal param names + any external
  # aliases this family declares (e.g. cumulative's `mu`/`cutpoints`,
  # cdp's `rec`/`fam`/`sigma_R`/`sigma_F`).
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

  # Handle aliases: brf param names -> internal names
  # CDP: rec -> dprime mapping, fam -> dprime2, sigma_R -> sigma, sigma_F -> sigma2
  # Cumulative: cutpoints -> criterion, mu is the primary (dprime)
  # General: mu is an alias for dprime (primary formula)

  # Extract parameters from brf$params by name
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

  # Cumulative alias: cutpoints -> criterion
  if (!is.null(params[["cutpoints"]]) && is.null(criterion)) {
    criterion <- params[["cutpoints"]]
  }

  # Bivariate cumulative aliases: mu2 -> discrim, cutpoints1 -> criterion, cutpoints2 -> criterion2
  if (family$family == "bivariate_cumulative") {
    if (!is.null(params[["mu2"]])) discrim <- params[["mu2"]]
    if (!is.null(params[["cutpoints1"]])) criterion <- params[["cutpoints1"]]
    if (!is.null(params[["cutpoints2"]])) criterion2 <- params[["cutpoints2"]]
  }

  # CDP aliases: fam -> dprime2, sigma_R -> sigma, sigma_F -> sigma2
  # Also handle rec -> dprime as the primary (rec is just the primary in brf)
  if (family$family %in% c("cdp")) {
    if (!is.null(params[["fam"]])) dprime2 <- params[["fam"]]
    if (!is.null(params[["sigma_R"]])) sigma <- params[["sigma_R"]]
    if (!is.null(params[["sigma_F"]])) sigma2 <- params[["sigma_F"]]
  }

  # Families that support encoding_vars (have old/new or target/lure distinction)
  encoding_supported_families <- c("evsdt", "uvsdt", "dpsdt", "mixture", "cdp", "bivariate_sdt", "bivariate_dp", "vrdp2d")

  # Check encoding_vars compatibility
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

  # --- 4. Handle cumulative family early-return ---
  if (family$family == "cumulative") {
    return(fit_cumulative(mu = dprime, cutpoints = criterion, data = data,
                          family = family, priors = priors, counts = counts,
                          cor_threshold = cor_threshold, threads = threads,
                          gap_link = gap_link,
                          batch_likelihood = batch_likelihood))
  }

  # --- Handle bivariate_cumulative family early-return ---
  if (family$family == "bivariate_cumulative") {
    return(fit_bivariate_cumulative(
      mu1 = dprime, mu2 = discrim,
      cutpoints1 = criterion, cutpoints2 = criterion2,
      rho = rho, data = data, family = family,
      priors = priors, counts = counts,
      cor_threshold = cor_threshold, threads = threads,
      gap_link = gap_link, batch_likelihood = batch_likelihood))
  }

  # Default criterion to ~ 1 if not specified
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
    # rec_crit stays as rec_crit, know_crit stays as know_crit
    # criterion stays as criterion
  }
  # Apply encoding_vars to formulas if specified
  if (!is.null(encoding_vars)) {
    dprime <- apply_encoding_vars(dprime, encoding_vars)
    if (!is.null(sigma)) sigma <- apply_encoding_vars(sigma, encoding_vars)
    if (!is.null(lambda)) lambda <- apply_encoding_vars(lambda, encoding_vars)
    if (!is.null(dprime2)) dprime2 <- apply_encoding_vars(dprime2, encoding_vars)
    if (!is.null(sigma2)) sigma2 <- apply_encoding_vars(sigma2, encoding_vars)
    if (!is.null(dprime_B)) dprime_B <- apply_encoding_vars(dprime_B, encoding_vars)
    if (!is.null(lambda_B)) lambda_B <- apply_encoding_vars(lambda_B, encoding_vars)
    # Bivariate SDT parameters
    if (!is.null(discrim)) discrim <- apply_encoding_vars(discrim, encoding_vars)
    if (!is.null(discrim_B)) discrim_B <- apply_encoding_vars(discrim_B, encoding_vars)
    if (!is.null(sigma_B)) sigma_B <- apply_encoding_vars(sigma_B, encoding_vars)
    if (!is.null(sigma2_B)) sigma2_B <- apply_encoding_vars(sigma2_B, encoding_vars)
    if (!is.null(rho)) rho <- apply_encoding_vars(rho, encoding_vars)
    if (!is.null(rho_B)) rho_B <- apply_encoding_vars(rho_B, encoding_vars)
    if (!is.null(rho_N)) rho_N <- apply_encoding_vars(rho_N, encoding_vars)
    # Note: criterion and criterion2 do NOT get encoding factors applied
  }

  # Detect which parameters are fixed (NULL or ~ NULL)
  # Must do this BEFORE formula parsing
  sigma_is_fixed <- is.null(sigma) || identical_to_null_formula(sigma)
  lambda_is_fixed <- is.null(lambda) || identical_to_null_formula(lambda)
  dprime2_is_fixed <- is.null(dprime2) || identical_to_null_formula(dprime2)
  sigma2_is_fixed <- is.null(sigma2) || identical_to_null_formula(sigma2)
  dprime_B_is_fixed <- is.null(dprime_B) || identical_to_null_formula(dprime_B)
  lambda_B_is_fixed <- is.null(lambda_B) || identical_to_null_formula(lambda_B)
  # Bivariate SDT parameters
  discrim_is_fixed <- is.null(discrim) || identical_to_null_formula(discrim)
  discrim_B_is_fixed <- is.null(discrim_B) || identical_to_null_formula(discrim_B)
  sigma_B_is_fixed <- is.null(sigma_B) || identical_to_null_formula(sigma_B)
  sigma2_B_is_fixed <- is.null(sigma2_B) || identical_to_null_formula(sigma2_B)
  rho_is_fixed <- is.null(rho) || identical_to_null_formula(rho)
  rho_B_is_fixed <- is.null(rho_B) || identical_to_null_formula(rho_B)
  rho_N_is_fixed <- is.null(rho_N) || identical_to_null_formula(rho_N)
  criterion2_is_fixed <- is.null(criterion2) || identical_to_null_formula(criterion2)

  # Convert fixed parameters to actual NULL
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

  # Collect provided parameters for family validation, keyed by internal Stan
  # names (external aliases were translated earlier in broc()). Uses
  # [key] <- list(val) so NULL entries are preserved.
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
    rho_N      = rho_N
  )
  provided_params <- list()
  for (pname in names(family$params)) {
    if (pname %in% names(internal_arg_map)) {
      provided_params[pname] <- list(internal_arg_map[[pname]])
    }
  }

  # Validate parameters for this family
  validate_family_params(family, provided_params)
  
  # Handle default messages for optional parameters
  if (family$family == "uvsdt" && is.null(sigma)) {
    message("Note: uvsd family specified but no sigma formula provided. Using sigma ~ 1.")
    sigma <- ~ 1
  }
  
  if (family$family == "dpsdt" && is.null(lambda)) {
    stop("lambda formula is required for family 'dpsd'")
  }
  # For mixture: lambda is required only if no lure mixture is specified
  # (users can use mixture() for lure-only models without a signal mixture)
  if (family$family == "mixture" && is.null(lambda) && is.null(lambda_L)) {
    stop("mixture family requires either lambda (signal mixture) or lambda_L + dprime_L (lure mixture)")
  }
  
  # Handle source_mixture family requirements
  if (family$family == "source_mixture") {
    if (is.null(lambda)) {
      stop("lambda formula is required for source_mixture family")
    }
    # Note: dprime_B and lambda_B being NULL means symmetric/equal constraints
    if (is.null(dprime_B)) {
      message("Note: dprime_B not specified - constraining dprime_B = -dprime")
    }
    if (is.null(lambda_B)) {
      message("Note: lambda_B not specified - constraining lambda_B = lambda")
    }
  }
  
  # Handle bivariate_sdt family requirements
  if (family$family == "bivariate_sdt") {
    # Required parameters
    if (is.null(discrim)) {
      stop("discrim formula is required for bivariate_gaussian family")
    }
    if (is.null(rho)) {
      stop("rho formula is required for bivariate_gaussian family")
    }
    if (is.null(criterion2)) {
      stop("criterion2 formula is required for bivariate_gaussian family")
    }
    
    # Optional parameter messages
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

  # Handle bivariate_dp family requirements
  if (family$family == "bivariate_dp") {
    # Required parameters
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

    # Optional parameter messages
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

  # Handle vrdp2d family requirements
  if (family$family == "vrdp2d") {
    # Required parameters
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
    
    # Optional parameter messages
    if (is.null(sigma2)) {
      message("Note: sigma2 not specified - fixing sigma_S = 1")
    }
  }

  # Parse formulas - only pass non-NULL formulas
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
    data = data,
    family_name = family$family
  )
  
  # Build prior lookup (handles legacy priors and new broc_prior() format)
  prior_lookup <- build_prior_lookup(priors, family = family)
  
  # Build all data structures
  model_data <- build_model_data(parsed, data, family, counts = counts,
                                 cor_threshold = cor_threshold,
                                 encoding_vars = encoding_vars)
  model_data$prior_lookup <- prior_lookup
  model_data$threads <- isTRUE(threads)
  model_data$gap_link <- gap_link

  # build_model_data() may recode is_old (centered 0.5/-0.5 -> 0/1) when encoding
  # factors or lure mixtures are present. Mirror that onto the stored data frame
  # so predict()/epred()/pp_check() -- which compute is_old * dprime against
  # model$data -- match the fitted likelihood. Rows are aligned (zero-count drop
  # already happened above, before build_model_data()).
  is_old_var <- parsed$response$is_old
  if (!is.null(is_old_var) && is_old_var %in% names(data) &&
      !is.null(model_data$stan_data$is_old) &&
      length(model_data$stan_data$is_old) == nrow(data)) {
    data[[is_old_var]] <- as.numeric(model_data$stan_data$is_old)
  }

  # Propagate fields that build_model_data() may have inferred / set on its
  # local copy of `family` (R passes by value, so the local `family` here
  # would otherwise still see the unmutated version).
  if (!is.null(model_data$old_levels) && is.null(family$old_levels)) {
    family$old_levels <- model_data$old_levels
    family$J <- length(model_data$old_levels)
  }
  if (!is.null(model_data$stan_data$n_rkg) && is.null(family$n_rkg)) {
    family$n_rkg <- as.integer(model_data$stan_data$n_rkg)
  }

  # Add grainsize to stan_data when threading is enabled
  # Default uses N/(2*threads) following brms convention; fit_broc() can override
  if (isTRUE(threads)) {
    N <- model_data$stan_data$N
    model_data$stan_data$grainsize <- max(1L, as.integer(N / 4))  # assumes 2 threads; fit_broc adjusts
  }

  # Determine batch likelihood mode. generate_batch_cpp() returns NULL for any
  # unsupported family/config, so it is the single source of truth for batch
  # support -- no separate allowlist is maintained here (they used to drift).
  use_batch <- isTRUE(batch_likelihood)

  # Generate batch C++ if applicable
  batch_cpp <- NULL
  if (use_batch) {
    batch_cpp <- generate_batch_cpp(model_data, family)
    if (is.null(batch_cpp)) use_batch <- FALSE
  }

  # Generate Stan code (with batch function declaration if applicable)
  stan_code <- generate_stan_code_v2(model_data, family, batch_info = batch_cpp)

  result <- list(
    parsed = parsed,
    model_data = model_data,
    stan_code = stan_code,
    stan_data = model_data$stan_data,
    family = family,
    prior_lookup = prior_lookup,
    data = data,
    threads = isTRUE(threads),
    gap_link = gap_link,
    formula = brf_obj,
    batch_likelihood = use_batch,
    batch_cpp = batch_cpp
  )

  class(result) <- c("broc_model", "list")
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
build_model_data <- function(parsed, data, family, counts = NULL,
                             cor_threshold = TRUE, encoding_vars = NULL) {
  
  N <- nrow(data)
  K <- length(unique(data[[parsed$response$response]]))
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
  # For bivariate_sdt/bivariate_dp and vrdp2d, create an is_old equivalent from item_type
  # "new" items are lures (is_old = 0), "A" and "B" items are targets (is_old = 1)
  if ((is_bivariate || is_vrdp2d) && !is.null(item_type_var)) {
    item_type_vals <- data[[item_type_var]]
    if (is.factor(item_type_vals)) {
      # First level should be "new" (reference)
      is_old_for_encoding <- as.integer(item_type_vals) > 1
    } else {
      # Assume 1 = new, 2+ = old
      is_old_for_encoding <- as.integer(item_type_vals) > 1
    }
    # Create a temporary column name for encoding purposes
    data$.is_old_bivariate <- as.integer(is_old_for_encoding)
    is_old_var <- ".is_old_bivariate"
  }
  
  # For bivariate_sdt and vrdp2d, get K2 from second response
  K2 <- NULL
  if ((is_bivariate || is_vrdp2d) && !is.null(parsed$response2)) {
    resp2_col <- parsed$response2
    if (!resp2_col %in% names(data)) {
      stop("Second response variable '", resp2_col, "' not found in data. ",
           "Check your resp() specification: resp(response1, response2)")
    }
    K2 <- length(unique(data[[resp2_col]]))
    if (K2 < 2) {
      stop("Second response variable '", resp2_col, "' must have at least 2 unique values, got ", K2)
    }
  }
  
  # Determine which parameters are present based on family and formulas
  # A parameter is "present" if it has a non-NULL formula in parsed
  # IMPORTANT: Use [["..."]] not $... to avoid R's partial matching
  # (e.g., parsed$sigma would match parsed$sigma2)
  is_vrdp2d <- family$family == "vrdp2d"
  has_sigma <- !is.null(parsed[["sigma"]]) && family$family %in% c("uvsdt", "dpsdt", "mixture", "bivariate_sdt", "bivariate_dp", "vrdp2d", "cdp")
  has_lambda <- !is.null(parsed[["lambda"]]) && family$family %in% c("dpsdt", "mixture", "source_mixture", "vrdp2d", "bivariate_dp")
  has_dprime2 <- !is.null(parsed[["dprime2"]]) && family$family %in% c("mixture", "cdp", "vrdp2d")
  has_sigma2 <- !is.null(parsed[["sigma2"]]) && family$family %in% c("mixture", "bivariate_sdt", "bivariate_dp", "cdp", "vrdp2d")
  has_dprime_B <- !is.null(parsed[["dprime_B"]]) && family$family %in% c("source_mixture", "bivariate_sdt", "bivariate_dp")
  has_lambda_B <- !is.null(parsed[["lambda_B"]]) && family$family %in% c("source_mixture", "bivariate_dp")

  # Bivariate SDT/DP and vrdp2d specific parameters
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
  
  # CDP specific parameters
  has_rec_crit <- !is.null(parsed$rec_crit) && is_cdp
  has_know_crit <- !is.null(parsed$know_crit) && is_cdp

  # Lure mixture parameters (mixture family only)
  has_dprime_L <- !is.null(parsed[["dprime_L"]]) && family$family == "mixture"
  has_sigma_L <- !is.null(parsed[["sigma_L"]]) && family$family == "mixture"
  has_lambda_L <- !is.null(parsed[["lambda_L"]]) && family$family == "mixture"
  
  # Validate lure mixture: lambda_L is required if dprime_L is specified
  if (has_dprime_L && !has_lambda_L) {
    stop("lambda_L formula is required when dprime_L is specified (lure mixture model)")
  }
  if (has_lambda_L && !has_dprime_L) {
    stop("dprime_L formula is required when lambda_L is specified (lure mixture model)")
  }
  has_lure_mixture <- has_dprime_L && has_lambda_L
  
  # For mixture with free dprime and dprime2, need ordered constraint for identifiability
  needs_ordered_dprime <- has_dprime2 && family$family == "mixture"
  
  # Initialize stan_data
  stan_data <- list(N = N, K = K)
  stan_data$y <- as.integer(data[[parsed$response$response]])

  # Auto-convert 0-indexed responses (0..K-1) to 1-indexed (1..K) -- Stan requires 1-indexed
  y_vals <- sort(unique(stan_data$y))
  if (length(y_vals) >= 2L && y_vals[1] == 0L &&
      identical(y_vals, seq.int(0L, length(y_vals) - 1L))) {
    K_shifted <- length(y_vals)
    message("0-indexed responses detected (range 0..", K_shifted - 1L,
            ") -- converting to 1..", K_shifted, " for Stan.")
    stan_data$y <- stan_data$y + 1L
    K <- K_shifted
    stan_data$K <- K
  }

  # K=2 (binary) is only supported for EVSDT
  if (K == 2 && family$family != "evsdt") {
    stop("Binary responses (K=2) are only supported for the evsd() family. ",
         "Models with sigma, lambda, or other parameters require K >= 3 for identifiability.")
  }

  # Handle item type / source / is_old / rk based on family
  if (is_cdp) {
    # CDP model: need is_old, rk, and old_levels
    cdp_is_old <- .as_is_old(data[[is_old_var]])
    if (!all(sort(unique(cdp_is_old)) %in% c(0, 1))) {
      stop("Centered coding for is_old (e.g., 0.5/-0.5) is not supported with the CDP family. ",
           "Please recode is_old as 0/1.")
    }
    stan_data$is_old <- cdp_is_old
    
    # Process R/K(/G) variable
    rk_values <- data[[rk_var]]
    if (is.character(rk_values) || is.factor(rk_values)) {
      rk_values <- as.character(rk_values)
      rk_numeric <- ifelse(toupper(rk_values) %in% c("R", "REMEMBER", "1"), 1L,
                           ifelse(toupper(rk_values) %in% c("K", "KNOW", "2"), 2L,
                                  ifelse(toupper(rk_values) %in% c("G", "GUESS", "3"), 3L, NA_integer_)))
    } else {
      rk_numeric <- as.integer(rk_values)
    }

    # Infer old_levels from data when not provided to cdp().
    # By the CDP data model, rk is recorded only for "old" responses, so
    # the bins where rk is non-NA define the old confidence levels.
    if (is.null(family$old_levels)) {
      has_rk <- !is.na(rk_numeric)
      if (!any(has_rk)) {
        stop("Cannot infer old_levels: no rows have non-NA rk values. ",
             "Either provide rk for 'old' responses or pass old_levels explicitly to cdp().",
             call. = FALSE)
      }
      inferred <- sort(unique(stan_data$y[has_rk]))
      family$old_levels <- inferred
      family$J <- length(inferred)
      message("cdp(): inferred old_levels = c(",
              paste(inferred, collapse = ", "), ") from rk data.")
    }

    # Determine which rows have "old" responses (need valid rk)
    old_resp_mask <- stan_data$y %in% family$old_levels

    # Validate: old-level responses must have valid rk
    invalid_old <- old_resp_mask & is.na(rk_numeric)
    if (any(invalid_old)) {
      n_bad <- sum(invalid_old)
      stop("Found ", n_bad, " row(s) with response in old_levels but missing/invalid rk value. ",
           "R/K judgments are required for all 'old' responses (confidence levels ",
           paste(family$old_levels, collapse = ", "), ").")
    }

    # Auto-detect R/K vs R/K/G mode from old-level responses only
    rk_old <- rk_numeric[old_resp_mask]
    n_rkg <- max(rk_old, na.rm = TRUE)
    if (!n_rkg %in% c(2L, 3L)) {
      stop("rk values must have 2 (R/K) or 3 (R/K/G) unique levels, found: ", n_rkg)
    }
    stan_data$n_rkg <- as.integer(n_rkg)
    family$n_rkg <- n_rkg

    # Replace NAs with dummy value (1) for non-old responses -- Stan never reads these
    rk_numeric[is.na(rk_numeric)] <- 1L
    stan_data$rk <- rk_numeric

    # Validate know_crit vs n_rkg
    if (n_rkg == 3L && !has_know_crit) {
      stop("know_crit formula is required when rk data has 3 levels (R/K/G mode)")
    }
    if (n_rkg == 2L && has_know_crit) {
      warning("know_crit formula ignored: rk data has only 2 levels (R/K mode)")
      has_know_crit <- FALSE
    }

    # Get old_levels from family
    old_levels <- family$old_levels
    J <- family$J
    new_levels <- setdiff(1:K, old_levels)
    n_new <- length(new_levels)

    stan_data$J <- J
    stan_data$n_new <- n_new
    stan_data$old_level_map <- as.array(old_levels)
    stan_data$new_level_map <- as.array(new_levels)
  } else if (is_bivariate || is_vrdp2d) {
    # For bivariate_sdt and vrdp2d, use item_type: 1=new, 2=A, 3=B
    item_type_vals <- data[[item_type_var]]
    if (is.factor(item_type_vals)) {
      # Expect levels like c("new", "A", "B") with "new" as reference
      levels_vec <- levels(item_type_vals)
      # Check that first level is the "new" reference
      stan_data$item_type <- as.integer(item_type_vals)  # 1, 2, 3
    } else {
      stan_data$item_type <- as.integer(item_type_vals)
    }
    # Add second response and K2.
    # NA in y2 (source-rating-only-on-old paradigm) is reserved for a future
    # release -- see roadmap. For now, reject NA early with a clear error so
    # the user doesn't get a confusing downstream Stan/JAX failure.
    y2_raw <- as.integer(data[[parsed$response2]])
    if (any(is.na(y2_raw))) {
      n_na <- sum(is.na(y2_raw))
      stop(sprintf(
        "Found %d row(s) with NA in the source response (`%s`). The ",
        n_na, parsed$response2),
        "source-conditional-on-old paradigm (where source is only rated when ",
        "a trial is judged old) is not yet supported. For now, either: ",
        "(a) drop rows with missing source ratings, or (b) impute them. ",
        "This feature is on the v0.2 roadmap.",
        call. = FALSE)
    }
    stan_data$y2 <- y2_raw
    stan_data$K2 <- K2
    # Rename K to K1 for clarity in bivariate
    stan_data$K1 <- K
    # Build is_new_response lookup for new_source_criteria = "shared"
    if (identical(family$new_source_criteria, "shared")) {
      old_resp <- family$old_levels
      if (any(old_resp < 1L) || any(old_resp > K)) {
        stop("old_levels values must be between 1 and ", K,
             " (the number of detection response levels)")
      }
      is_new_response <- as.integer(!(seq_len(K) %in% old_resp))
      stan_data$is_new_response <- as.array(is_new_response)
    }
    # No is_old for bivariate/vrdp2d
    stan_data$is_old <- NULL
  } else if (is_source_mixture) {
    # source must be two groups: 0 (Source A) / 1 (Source B). Factors are
    # mapped by level order; numeric must already be 0/1.
    source_vals <- data[[source_var]]
    if (is.factor(source_vals)) {
      if (nlevels(droplevels(source_vals)) != 2L) {
        stop("source must have exactly two levels for source_mixture(). Got ",
             nlevels(droplevels(source_vals)), " level(s).", call. = FALSE)
      }
      stan_data$source <- as.integer(droplevels(source_vals)) - 1L
    } else {
      sv <- as.integer(source_vals)
      if (!all(sv %in% c(0L, 1L))) {
        stop("source must be coded 0 (Source A) / 1 (Source B) for source_mixture(). ",
             "Got: ", paste(sort(unique(sv)), collapse = ", "), ".", call. = FALSE)
      }
      stan_data$source <- sv
    }
    # No is_old for source_mixture
    stan_data$is_old <- NULL
  } else {
    is_old_raw <- data[[is_old_var]]
    is_old_vals <- .as_is_old(is_old_raw)
    # A factor/character indicator with non-numeric labels (e.g. "old"/"new")
    # coerces to all-NA via .as_is_old(); catch that explicitly rather than let
    # it silently become an all-NA is_old column.
    if (anyNA(is_old_vals) && !anyNA(is_old_raw)) {
      stop("is_old (column '", is_old_var, "') could not be coerced to 0/1: a ",
           "factor/character indicator must use numeric labels ('0'/'1' or ",
           "'-0.5'/'0.5'), not e.g. 'old'/'new'. Got: ",
           paste(utils::head(unique(as.character(is_old_raw)), 6), collapse = ", "),
           ".", call. = FALSE)
    }
    unique_vals <- sort(unique(is_old_vals))

    # is_old must be 0/1 (treatment) or -0.5/0.5 (centered); nothing else.
    if (length(unique_vals) == 0 ||
        (!all(unique_vals %in% c(0, 1)) && !all(unique_vals %in% c(-0.5, 0.5)))) {
      stop("is_old must be coded 0/1 (new/old) or -0.5/0.5 (centered). Got: ",
           paste(unique_vals, collapse = ", "), ".", call. = FALSE)
    }
    centered_coding <- all(unique_vals %in% c(-0.5, 0.5))

    if (centered_coding) {
      # Error for CDP
      if (is_cdp) {
        stop("Centered coding for is_old (e.g., 0.5/-0.5) is not supported with the CDP family. ",
             "Please recode is_old as 0/1.")
      }

      # Recode to 0/1 with warning if encoding_vars or lure mixture
      if (length(encoding_vars) > 0 || has_lure_mixture) {
        warning("Centered coding for is_old detected (values: ",
                paste(unique_vals, collapse = ", "),
                ") but encoding factors or lure mixture parameters are present. ",
                "Recoding is_old to 0/1.")
        is_old_vals <- ifelse(is_old_vals > 0, 1, 0)
        centered_coding <- FALSE
      }
    }

    # Sync the working data frame with the (possibly recoded) is_old before the
    # design-matrix builders run below. They identify target rows for encoding()
    # via `data[[is_old_var]] == 1`; if the recode (centered 0.5/-0.5 -> 0/1)
    # were not mirrored here, those builders would still see centered values,
    # match zero target rows, and emit an all-zero d' design matrix (with the
    # lure-only level still present) -- i.e. encoding() would silently no-op.
    data[[is_old_var]] <- is_old_vals
    stan_data$is_old <- is_old_vals
  }
  
  # Handle counts for aggregated data
  has_counts <- !is.null(counts)
  if (has_counts) {
    if (!counts %in% names(data)) {
      stop("counts variable '", counts, "' not found in data")
    }
    stan_data$counts <- as.integer(data[[counts]])
  }
  
  # Build dprime structure
  # For cumulative, allow empty fixed effects (mu = 0 with only random effects)
  allow_empty_dprime <- (is_cumulative || is_bivariate_cumulative) &&
    length(parsed$dprime$fixed$terms) == 0 &&
    !parsed$dprime$fixed$intercept
  
  if (allow_empty_dprime) {
    # Create a dummy design matrix with a single column of zeros
    # This is a hack to keep the structure consistent - the Stan code will handle it
    dprime_fixed <- list(
      X = matrix(0, nrow = N, ncol = 1),
      n_coef = 0,  # Signal that there are no actual fixed effects
      coef_names = character(0),
      has_encoding = FALSE
    )
    dprime_random <- build_random_effects_structure(parsed$dprime$random, data, encoding_vars, is_old_var)
    
    # For cumulative/bivariate_cumulative: drop reference level from random slopes (non-identified)
    if (is_cumulative || is_bivariate_cumulative) {
      dprime_random <- drop_reference_level_from_random_slopes(dprime_random, data)
    }
    
    stan_data$P_dprime <- 0
    stan_data$X_dprime <- matrix(0, nrow = N, ncol = 1)  # Placeholder
  } else {
    dprime_fixed_formula <- build_formula_from_parsed(parsed$dprime$fixed, allow_empty = is_cumulative || is_bivariate_cumulative)
    dprime_cond <- extract_conditional_from_parsed(parsed$dprime)
    dprime_fixed <- build_fixed_effects_matrix(dprime_fixed_formula, data, encoding_vars, is_old_var,
                                                conditional_terms = dprime_cond,
                                                parsed_fixed = parsed$dprime$fixed)
    dprime_random <- build_random_effects_structure(parsed$dprime$random, data, encoding_vars, is_old_var)
    
    # For cumulative models with predictors: drop the intercept column to use treatment coding
    # This ensures the reference level is fixed at 0 (absorbed into thresholds)
    if (is_cumulative && dprime_fixed$n_coef > 0) {
      intercept_col <- which(colnames(dprime_fixed$X) == "(Intercept)")
      if (length(intercept_col) > 0) {
        dprime_fixed$X <- dprime_fixed$X[, -intercept_col, drop = FALSE]
        dprime_fixed$coef_names <- dprime_fixed$coef_names[-intercept_col]
        dprime_fixed$n_coef <- dprime_fixed$n_coef - 1
        
        # If the only column was removed, treat as no fixed effects
        if (dprime_fixed$n_coef == 0) {
          dprime_fixed$X <- matrix(0, nrow = N, ncol = 1)
          dprime_fixed$coef_names <- character(0)
        }
      }
    }
    
    # For cumulative/bivariate_cumulative: drop reference level from random slopes (non-identified)
    if (is_cumulative || is_bivariate_cumulative) {
      dprime_random <- drop_reference_level_from_random_slopes(dprime_random, data)
    }
    
    stan_data$P_dprime <- dprime_fixed$n_coef
    stan_data$X_dprime <- dprime_fixed$X
  }
  
  # Initialize smooth data collection (used by criterion, criterion2, and all other params)
  smooth_data <- list()

  # Build criterion structure
  # Number of thresholds: J for CDP, K-1 for standard models
  n_thresh <- if (is_cdp) family$J else K - 1
  criterion_fixed_formula <- build_formula_from_parsed(parsed$criterion$fixed)
  criterion_cond <- extract_conditional_from_parsed(parsed$criterion)
  criterion <- build_criterion_structure(criterion_fixed_formula, parsed$criterion$random, data, n_thresh,
                                         conditional_terms = criterion_cond,
                                         parsed_fixed = parsed$criterion$fixed)
  
  # Criterion smooth terms
  # NOTE: unlike other parameters, Xs is NOT appended to X_criterion.
  # X_criterion multiplies both beta_thresh_mid and beta_log_gaps, so appending
  # Xs would give the unpenalized smooth per-gap coefficients while the penalized
  # Zs only shifts the mid-anchor -- an inconsistency. Instead, the entire smooth
  # is penalized (Zs only) and shifts the mid-anchor uniformly.
  crit_smooth_info <- build_param_smooth_info(parsed$criterion, data, encoding_vars, is_old_var)
  if (!is.null(crit_smooth_info)) {
    # Tag criterion smooth with n_thresh for per-threshold parameterization
    for (si in seq_along(crit_smooth_info)) {
      crit_smooth_info[[si]]$n_thresh <- n_thresh
    }
    smooth_data[["criterion"]] <- crit_smooth_info
    stan_data <- add_smooth_to_stan_data(stan_data, "criterion", crit_smooth_info)
  }

  stan_data$P_criterion <- criterion$n_coef
  stan_data$X_criterion <- criterion$X
  
  # Build sigma structure if needed (renamed from theta)
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
  
  # Build lambda structure if needed
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
  
  # Build dprime2 structure if needed
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
  
  # For ordered constraint: dprime2 uses the same design matrix as dprime
  # This ensures the ordered constraint applies to each predictor combination
  if (needs_ordered_dprime) {
    # dprime2 uses dprime's design matrix
    dprime2_fixed <- dprime_fixed
    stan_data$P_dprime2 <- dprime_fixed$n_coef
    stan_data$X_dprime2 <- dprime_fixed$X
  }
  
  # Build sigma2 structure if needed
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
  
  # Build dprime_B structure if needed (for source_mixture)
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
  
  # Build lambda_B structure if needed (for source_mixture)
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
  
  # Build discrim structure if needed (for bivariate_sdt/bivariate_cumulative)
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
  
  # Build discrim_B structure if needed (for bivariate_sdt)
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
  
  # Build sigma_B structure if needed (for bivariate_sdt)
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
  
  # Build sigma2_B structure if needed (for bivariate_sdt)
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
  
  # Build rho structure if needed (for bivariate_sdt)
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
  
  # Build rho_B structure if needed (for bivariate_sdt)
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
  
  # Build rho_N structure if needed (for bivariate_sdt)
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
  
  # Build lambda2 structure if needed (for bivariate_dp: R_S Source A)
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

  # Build lambda2_B structure if needed (for bivariate_dp: R_S Source B)
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

  # Build rec_crit structure if needed (for cdp)
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
  
  # Build know_crit structure if needed (for cdp R/K/G)
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


  # Build lure mixture structures if needed (for mixture family)
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
  
  # Build criterion2 structure if needed (for bivariate_sdt)
  criterion2 <- NULL
  if (has_criterion2) {
    criterion2_fixed_formula <- build_formula_from_parsed(parsed$criterion2$fixed)
    criterion2_cond <- extract_conditional_from_parsed(parsed$criterion2)

    criterion2 <- build_criterion_structure(criterion2_fixed_formula, parsed$criterion2$random, data, K2,
                                            conditional_terms = criterion2_cond,
                                            parsed_fixed = parsed$criterion2$fixed)

    # Criterion2 smooth terms (same approach as criterion: Zs only, no Xs)
    crit2_smooth_info <- build_param_smooth_info(parsed$criterion2, data, encoding_vars, is_old_var)
    if (!is.null(crit2_smooth_info)) {
      n_thresh2 <- K2 - 1
      for (si in seq_along(crit2_smooth_info)) {
        crit2_smooth_info[[si]]$n_thresh <- n_thresh2
      }
      smooth_data[["criterion2"]] <- crit2_smooth_info
      stan_data <- add_smooth_to_stan_data(stan_data, "criterion2", crit2_smooth_info)
    }

    stan_data$P_criterion2 <- criterion2$n_coef
    stan_data$X_criterion2 <- criterion2$X

    # Override RE dimensions for varying source criteria based on varying_re
    varying_re_mode <- if (!is.null(family$varying_re)) family$varying_re else "shared"
    cor_threshold_flag <- cor_threshold  # from broc() argument

    if (isTRUE(family$varying_source_criteria) && !is.null(criterion2$random)) {
      # Compute n_bins: how many detection bins get source thresholds
      n_bins <- K  # All K1 levels
      
      for (group in names(criterion2$random)) {
        re <- criterion2$random[[group]]
        
        if (varying_re_mode == "shared") {
          # One shift for all bins
          criterion2$random[[group]]$dim <- 1
          criterion2$random[[group]]$varying_re_mode <- "shared"
        } else if (varying_re_mode == "per_bin") {
          # One shift per detection bin
          criterion2$random[[group]]$dim <- n_bins
          criterion2$random[[group]]$varying_re_mode <- "per_bin"
        } else if (varying_re_mode == "full") {
          # Full: first + gaps per bin = n_bins * (K2-1)
          criterion2$random[[group]]$dim <- n_bins * (K2 - 1)
          criterion2$random[[group]]$varying_re_mode <- "full"
        }
        criterion2$random[[group]]$n_bins <- n_bins
        
        # Warn if cor_threshold = FALSE and there's a cross-parameter cor_id
        if (!cor_threshold_flag && !is.null(re$cor_id)) {
          warning("cor_threshold = FALSE: criterion2 random effects cannot participate in ",
                  "cross-parameter correlation group '", re$cor_id, "'. ",
                  "The cor_id is being ignored for criterion2 threshold REs.",
                  call. = FALSE)
          criterion2$random[[group]]$cor_id <- NULL
        }
        if (!cor_threshold_flag) {
          criterion2$random[[group]]$correlated <- FALSE
        }
      }
    }
    
    # Also handle cor_threshold for non-varying criterion2
    if (!isTRUE(family$varying_source_criteria) && !cor_threshold_flag && !is.null(criterion2$random)) {
      for (group in names(criterion2$random)) {
        if (!is.null(criterion2$random[[group]]$cor_id)) {
          warning("cor_threshold = FALSE: criterion2 random effects cannot participate in ",
                  "cross-parameter correlation group '", criterion2$random[[group]]$cor_id, "'.",
                  call. = FALSE)
          criterion2$random[[group]]$cor_id <- NULL
        }
        criterion2$random[[group]]$correlated <- FALSE
      }
    }
  }

  # Handle cor_threshold for criterion (detection thresholds) for ALL families.
  # Setting correlated <- FALSE here propagates to both Stan and JAX backends
  # (Stan also gates on cor_threshold directly, but JAX only sees `correlated`).
  if (!cor_threshold && !is.null(criterion$random)) {
    for (group in names(criterion$random)) {
      if (!is.null(criterion$random[[group]]$cor_id)) {
        warning("cor_threshold = FALSE: criterion random effects cannot participate in ",
                "cross-parameter correlation group '", criterion$random[[group]]$cor_id, "'.",
                call. = FALSE)
        criterion$random[[group]]$cor_id <- NULL
      }
      criterion$random[[group]]$correlated <- FALSE
    }
  }
  
  # Add grouping factor data
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
    if (has_lambda2_B) names(lambda2_B_random) else NULL
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
                                    lambda2_random, lambda2_B_random)
    if (!is.null(re_info)) {
      stan_data[[paste0("N_", group)]] <- re_info$n_groups
      stan_data[[group]] <- re_info$group_idx
    }
  }
  
  # Add term indices for random effects
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

  # Add criterion term indices and Z matrices
  for (group in names(criterion$random)) {
    re <- criterion$random[[group]]
    if (isTRUE(re$use_z_matrix) && !is.null(re$Z)) {
      # Use Z matrix approach for criterion
      z_name <- paste0("Z_criterion_", group)
      stan_data[[z_name]] <- re$Z
    } else if (!is.null(re$term_idx)) {
      # Use index approach
      idx_name <- paste0("idx_criterion_", group)
      dim_name <- paste0("D_idx_criterion_", group)
      stan_data[[dim_name]] <- re$n_cond_levels
      stan_data[[idx_name]] <- re$term_idx
    }
    # For intercept-only (n_cond_levels=1), neither is needed
  }
  
  # Add criterion2 term indices and Z matrices (for bivariate_sdt)
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
  # Build smooth structures for any parameter that has s()/t2() terms.
  # Appends Xs (unpenalized) columns to the parameter's X matrix,
  # adds Zs (penalized) matrices to stan_data, and stores metadata.
  # (smooth_data was initialized earlier, before criterion processing)

  # List of all parameters and their fixed effects structures
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
    rec_crit = if (has_rec_crit) rec_crit_fixed else NULL,
    know_crit = if (has_know_crit) know_crit_fixed else NULL,
    dprime_L = if (has_dprime_L) dprime_L_fixed else NULL,
    sigma_L = if (has_sigma_L) sigma_L_fixed else NULL,
    lambda_L = if (has_lambda_L) lambda_L_fixed else NULL
  )

  for (pname in names(all_param_fixed)) {
    if (is.null(all_param_fixed[[pname]])) next
    sm_info <- build_param_smooth_info(parsed[[pname]], data, encoding_vars, is_old_var)
    if (is.null(sm_info)) next

    smooth_data[[pname]] <- sm_info

    # Append Xs columns to the parameter's X matrix
    fixed_ref <- get(paste0(pname, "_fixed"))
    for (sm in sm_info) {
      for (comp in sm$components) {
        if (NCOL(comp$Xs) > 0) {
          fixed_ref$X <- cbind(fixed_ref$X, comp$Xs)
          fixed_ref$coef_names <- colnames(fixed_ref$X)
          fixed_ref$n_coef <- ncol(fixed_ref$X)
        }
      }
    }
    # Write back the updated fixed structure
    assign(paste0(pname, "_fixed"), fixed_ref)
    # Update stan_data X and P
    stan_data[[paste0("X_", pname)]] <- fixed_ref$X
    stan_data[[paste0("P_", pname)]] <- fixed_ref$n_coef

    # Add Zs matrices to stan_data
    stan_data <- add_smooth_to_stan_data(stan_data, pname, sm_info)
  }

  # Build cross-correlation info
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
                                       lambda2_random, lambda2_B_random)

  # Track if dprime/discrim have any fixed effects (for cumulative/bivariate_cumulative with only random effects)
  has_dprime_fixed <- dprime_fixed$n_coef > 0
  has_discrim_fixed <- if (!is.null(discrim_fixed)) discrim_fixed$n_coef > 0 else FALSE
  
  list(
    N = N,
    K = K,
    K2 = K2,  # For bivariate_sdt
    family = family$family,
    family_obj = family,  # Full family object including link
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
    has_dprime_fixed = has_dprime_fixed,  # For cumulative/bivariate_cumulative with only random effects
    has_discrim_fixed = has_discrim_fixed,  # For bivariate_cumulative with only random effects
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
    centered_coding = if (exists("centered_coding")) centered_coding else FALSE,
    has_counts = has_counts,
    varying_source_criteria = isTRUE(family$varying_source_criteria),
    bounded = isTRUE(family$bounded),
    new_source_criteria = family$new_source_criteria,
    old_levels = family$old_levels,
    varying_re = if (!is.null(family$varying_re)) family$varying_re else "shared",
    cor_threshold = cor_threshold,  # from broc() argument
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
                                 lambda2_re = NULL, lambda2_B_re = NULL) {
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
  NULL
}


#' Add term indices for a parameter's random effects
#' @noRd
add_term_indices <- function(stan_data, param_name, random_effects) {
  if (is.null(random_effects)) return(stan_data)
  
  for (group in names(random_effects)) {
    re <- random_effects[[group]]
    
    if (isTRUE(re$use_z_matrix) && !is.null(re$Z)) {
      # Use full Z matrix approach
      z_name <- paste0("Z_", param_name, "_", group)
      stan_data[[z_name]] <- re$Z
    } else if (!is.null(re$term_idx)) {
      # Use fast index approach
      idx_name <- paste0("idx_", param_name, "_", group)
      dim_name <- paste0("D_idx_", param_name, "_", group)
      stan_data[[dim_name]] <- re$dim
      stan_data[[idx_name]] <- re$term_idx
    }
    # For intercept-only (dim=1, no term_idx), neither is needed - just use group index
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
      # Build component suffix: san_label, plus bylevel if present
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
      # For cumulative mu with only random effects - return a formula that gives
      # a single column of zeros (handled specially below)
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


#' Build Cross-Correlation Info (updated for new parameters)
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
                                    lambda2_B_random = NULL) {
  if (length(parsed$cross_cor_ids) == 0) return(list())
  
  cross_cor <- list()
  
  for (cor_id in parsed$cross_cor_ids) {
    members <- list()
    group <- NULL
    
    # Check all parameters for this cor_id
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
      lambda2_B = lambda2_B_random
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
  # Pass family along so print.broc_formula can show user-facing parameter
  # names (rec/fam/mu/cutpoints/...) instead of the internal Stan names.
  attr(x$parsed, ".broc_family") <- x$family
  print(x$parsed)

  cat("\nData dimensions:\n")
  cat("  N =", x$model_data$N, "\n")
  cat("  K =", x$model_data$K, "\n")

  # Walk every modeled parameter once, in a stable order, using the family's
  # external_aliases so cdp/cumulative/bivariate_cumulative show user-facing names.
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
                   "dprime_L", "sigma_L", "lambda_L")
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
  # Mirror fit_jax's serialization: matrices -> list of row vectors.
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
#' @param iter_warmup Number of warmup/adaptation iterations per chain (default 2000).
#' @param iter_sampling Number of post-warmup sampling iterations per chain (default 2000).
#' @param thin Period for saving samples (default 1). `thin = k` keeps every
#'   `k`-th post-warmup draw.
#' @param num_warmup,num_samples,num_chains NumPyro-style aliases for
#'   `iter_warmup`, `iter_sampling`, and `chains`. When supplied, each overrides
#'   its canonical counterpart.
#' @param adapt_delta Target acceptance probability for NUTS (default 0.8).
#' @param max_treedepth Maximum tree depth for NUTS (default 10).
#' @param init Initial-value strategy. Default `0`: all parameters start at 0 on
#'   the unconstrained scale. Other options: `"prior"` samples each parameter
#'   from its prior; `"random"` uses Stan's U(-2, 2) on the unconstrained scale
#'   (can be unstable for log-linked parameters); a single number `x` sets a
#'   U(-x, x) initialization radius; or a function or list of per-chain init lists.
#' @param refresh How often the Stan backend prints progress, in iterations
#'   (default 100). Stan only; the JAX progress bar has no refresh-rate control.
#' @param threads_per_chain Number of threads per chain for within-chain parallelism
#'   via `reduce_sum`. Only used when the model was built with `threads = TRUE`.
#' @param backend Inference backend: `"stan"` (default, via cmdstanr) or `"jax"`
#'   (via NumPyro). The JAX backend requires a configured Python environment
#'   (see [install_jax_backend()]).
#' @param python Optional path to the Python executable for the JAX backend.
#'   `NULL` (default) uses the managed environment (see [install_jax_backend()]).
#' @param progress_bar If `TRUE` (default), show the sampler progress bar.
#' @param seed Optional integer seed for reproducible sampling. `NULL` (default)
#'   uses a random seed.
#' @param file Optional path to save the fitted model to. When set,
#'   the fit is written there with [save_broc_fit()] once sampling finishes (a
#'   `.rds` extension is appended if absent). If the file already exists, the
#'   saved fit is loaded and returned instead of refitting -- unless
#'   `file_refit = TRUE`. `NULL` (default) does not save.
#' @param file_refit If `FALSE` (default), an existing `file` is loaded and
#'   sampling is skipped; if `TRUE`, the model is refit and the file overwritten.
#'   Ignored when `file` is `NULL`.
#' @param save_loglik If `TRUE` (default), the per-observation log-likelihood is
#'   kept so the saved fit supports [loo()]. If `FALSE`, `log_lik` is dropped when
#'   the fit is saved.
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
#' @seealso [broc()] for model specification, `summary(fit)` for results.
#' @export
fit_broc <- function(model,
                    chains = 4,
                    parallel_chains = 4,
                    iter_warmup = 2000,
                    iter_sampling = 2000,
                    thin = 1,
                    adapt_delta = 0.8,
                    max_treedepth = 10,
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
                    num_warmup = NULL,
                    num_samples = NULL,
                    num_chains = NULL,
                    ...) {

  backend <- match.arg(backend)
  if (!is.logical(file_refit) || length(file_refit) != 1L || is.na(file_refit)) {
    stop("`file_refit` must be TRUE or FALSE.", call. = FALSE)
  }

  # On-disk caching. With `file` set, save the fit there when sampling
  # finishes; if the file already exists and `file_refit = FALSE` (default), load
  # and return it instead of refitting. `.rds` is appended if absent.
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
  # Save the finished fit to `file` (if set) before returning it.
  .save_if_requested <- function(fit) {
    if (!is.null(file)) save_broc_fit(fit, file, verbose = FALSE)
    fit
  }

  # NumPyro-style aliases mapped onto the canonical iteration/chain args (both
  # backends use iter_warmup / iter_sampling / chains). num_* take precedence.
  if (!is.null(num_warmup))  iter_warmup   <- num_warmup
  if (!is.null(num_samples)) iter_sampling <- num_samples
  if (!is.null(num_chains))  chains        <- num_chains

  # --- Init strategy normalization ---
  # `init` accepts:
  #   0         - all params at 0 on the unconstrained scale (DEFAULT)
  #   "prior"   - sample each parameter from its prior
  #   "random"  - cmdstanr's U(-2, 2) (can be unstable for log-linked params)
  #   <numeric> - cmdstanr's U(-x, x)
  #   <list>    - user-supplied per-chain init lists
  init_strategy_jax <- NULL  # set below for JAX backend handling
  if (inherits(model, "broc_model")) {
    if (is.character(init) && length(init) == 1) {
      if (init == "prior") {
        if (backend == "stan") {
          init <- prior_init(model, chains = chains, seed = seed)
        } else {
          init_strategy_jax <- "prior"
          init <- 0  # passed to cmdstanr path, not used for JAX
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
      # numeric init: cmdstanr accepts directly; JAX side mirrors via radius
      init_strategy_jax <- "numeric"
    }
  }

  # Validate input: must be a broc_model object (output of broc()), not a
  # brf() formula composition. Common mistake: passing the brf() result
  # directly to fit_broc() instead of building the model with broc() first.
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
    if (...length() > 0) {
      warning("Arguments in `...` are forwarded to cmdstanr (Stan backend) only ",
              "and are ignored by the JAX backend.", call. = FALSE)
    }
    # JAX parallelism is all-or-nothing (pmap needs one device per chain), so
    # parallel_chains maps onto chain_method: full parallel, or sequential.
    chain_method <- if (parallel_chains >= chains) "parallel" else "sequential"
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
      chain_method = chain_method,
      python = python,
      progress_bar = progress_bar
    )
    elapsed <- as.numeric(proc.time()["elapsed"] - t0)

    # Use the actual kept-draw count (after thinning) for the draws array and
    # the stored iter_sampling, which loo() and chain bookkeeping rely on.
    kept <- dim(jax_result$samples[[1]])[2]
    draws_arr <- numpyro_samples_to_draws_array(
      jax_result$samples,
      n_chains = chains,
      n_draws = kept
    )

    return(.save_if_requested(new_broc_fit(
      backend_fit = jax_result,
      model = model,
      backend = "jax",
      num_chains = chains,
      iter_sampling = kept,
      elapsed = elapsed,
      draws_array = draws_arr,
      max_treedepth = max_treedepth,
      save_loglik = save_loglik
    )))
  }

  # ---- Stan backend (default) ----
  if (!requireNamespace("cmdstanr", quietly = TRUE)) {
    stop("Package 'cmdstanr' is required.")
  }

  stan_file <- tempfile(fileext = ".stan")
  writeLines(model$stan_code, stan_file)

  t0 <- proc.time()["elapsed"]

  # Batch likelihood: write C++ header and set compilation options
  batch_compile_opts <- list()
  if (isTRUE(model$batch_likelihood) && !is.null(model$batch_cpp)) {
    # Write combined header: core + model-specific function
    core_hpp <- system.file("stan/batch_sdt_core.hpp", package = "bayesroc")
    if (core_hpp == "") {
      # Development mode: look relative to package root
      core_hpp <- file.path(find.package("bayesroc", quiet = TRUE), "stan", "batch_sdt_core.hpp")
      if (!file.exists(core_hpp)) {
        # Try inst/ path for devtools::load_all()
        pkg_root <- system.file(package = "bayesroc")
        core_hpp <- file.path(pkg_root, "stan", "batch_sdt_core.hpp")
      }
    }

    batch_hpp <- tempfile(fileext = ".hpp")
    hpp_lines <- c(
      sprintf('#include "%s"', core_hpp),
      "",
      model$batch_cpp$cpp_code
    )
    writeLines(hpp_lines, batch_hpp)

    batch_compile_opts <- list(
      user_header = batch_hpp,
      stanc_options = list("allow-undefined" = TRUE)
    )
  }

  # progress_bar toggles Stan's iteration progress (refresh = 0 suppresses it);
  # refresh sets the frequency when on.
  stan_refresh <- if (isTRUE(progress_bar)) refresh else 0

  if (isTRUE(model$threads)) {
    # Compile with threading support (+ batch user header if applicable)
    compile_args <- list(stan_file, cpp_options = list(stan_threads = TRUE))
    if (length(batch_compile_opts) > 0) {
      compile_args$user_header <- batch_compile_opts$user_header
      compile_args$stanc_options <- batch_compile_opts$stanc_options
    }
    mod <- do.call(cmdstanr::cmdstan_model, compile_args)
    tpc <- if (!is.null(threads_per_chain)) threads_per_chain else 2L
    # Recompute grainsize based on actual threads_per_chain (brms convention: N/(2*tpc))
    N <- model$stan_data$N
    model$stan_data$grainsize <- max(1L, as.integer(N / (2L * tpc)))
    stan_fit <- mod$sample(
      data = model$stan_data,
      chains = chains,
      parallel_chains = parallel_chains,
      iter_warmup = iter_warmup,
      iter_sampling = iter_sampling,
      thin = thin,
      adapt_delta = adapt_delta,
      max_treedepth = max_treedepth,
      init = init,
      refresh = stan_refresh,
      seed = seed,
      threads_per_chain = tpc,
      ...
    )
  } else {
    if (!is.null(threads_per_chain)) {
      warning("threads_per_chain ignored: model was not built with threads = TRUE")
    }
    if (length(batch_compile_opts) > 0) {
      mod <- cmdstanr::cmdstan_model(stan_file,
        user_header = batch_compile_opts$user_header,
        stanc_options = batch_compile_opts$stanc_options)
    } else {
      mod <- cmdstanr::cmdstan_model(stan_file)
    }
    stan_fit <- mod$sample(
      data = model$stan_data,
      chains = chains,
      parallel_chains = parallel_chains,
      iter_warmup = iter_warmup,
      iter_sampling = iter_sampling,
      thin = thin,
      adapt_delta = adapt_delta,
      max_treedepth = max_treedepth,
      init = init,
      refresh = stan_refresh,
      seed = seed,
      ...
    )
  }

  elapsed <- as.numeric(proc.time()["elapsed"] - t0)

  .save_if_requested(new_broc_fit(
    backend_fit = stan_fit,
    model = model,
    backend = "stan",
    num_chains = chains,
    iter_sampling = iter_sampling,
    elapsed = elapsed,
    save_loglik = save_loglik
  ))
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
  
  # Convert formula to string
  f_str <- paste(deparse(formula, width.cutoff = 500), collapse = " ")
  
  # For each encoding factor, wrap it in encoding()
  # Need to be careful about:
  # - Word boundaries and not double-wrapping
  # - NOT replacing inside smooth terms s(...) and t2(...) where the factor
  #   may appear as a by= argument that mgcv needs to see as a bare name
  for (fac in encoding_vars) {
    # Skip if already wrapped in encoding()
    if (grepl(paste0("encoding\\s*\\(\\s*", fac), f_str)) next

    # Extract smooth terms, replace factor in non-smooth parts only
    # Split formula into smooth and non-smooth segments
    chars <- strsplit(f_str, "")[[1]]
    n <- length(chars)
    in_smooth <- logical(n)
    i <- 1
    while (i <= n) {
      # Detect s( or t2( at position i
      rest <- substr(f_str, i, n)
      if (grepl("^(s|t2)\\s*\\(", rest)) {
        # Find the matching close paren
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

    # Replace only in non-smooth positions
    pattern <- paste0("(?<!encoding\\()\\b", fac, "\\b")
    # Find all matches
    matches <- gregexpr(pattern, f_str, perl = TRUE)[[1]]
    if (matches[1] > 0) {
      # Process matches in reverse order to preserve positions
      for (mi in rev(seq_along(matches))) {
        pos <- matches[mi]
        match_len <- attr(matches, "match.length")[mi]
        # Check if any position in the match range is inside a smooth
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
  
  # Parse back to formula
  as.formula(f_str, env = environment(formula))
}


#' Fit Cumulative Ordinal Model
#' 
#' Internal function for fitting cumulative ordinal regression models.
#' Routes through the existing SDT infrastructure by treating it as a simplified
#' EVSDT model where all observations are "old" (no signal/noise distinction).
#' 
#' Note: For identifiability, mu uses treatment coding where the reference level
#' of each factor is fixed at 0 (absorbed into the thresholds). This is different
#' from cell-means coding (~ 0 + factor) which would leave location unidentified.
#' 
#' IMPORTANT: Random intercepts on mu are automatically suppressed because they
#' are not identified separately from random effects on cutpoints. Only the 
#' difference (cutpoint - mu) matters for the likelihood. Random slopes on mu
#' (e.g., varying condition effects) ARE identified and are retained.
#' @noRd
fit_cumulative <- function(mu, cutpoints, data, family, priors, counts = NULL,
                           cor_threshold = TRUE, threads = FALSE,
                           gap_link = "log", batch_likelihood = TRUE) {
  
  if (is.null(mu)) {
    stop("mu formula is required for cumulative() family")
  }
  if (is.null(cutpoints)) {
    stop("cutpoints formula is required for cumulative() family")
  }
  
  # Parse mu formula to extract response variable
  mu_str <- paste(deparse(mu, width.cutoff = 500), collapse = " ")
  if (!grepl("~", mu_str)) {
    stop("mu formula must be two-sided: response ~ predictors")
  }
  
  parts <- strsplit(mu_str, "~")[[1]]
  response_var <- trimws(parts[1])
  mu_rhs <- trimws(parts[2])
  
  # Validate response exists
  if (!response_var %in% names(data)) {
    stop("Response variable '", response_var, "' not found in data")
  }
  
  # For cumulative models, the intercept needs special handling:
  # - Standard treatment coding: ~ condition gives intercept + (K-1) dummies
  # - Goal: just the (K-1) dummies, with reference level = 0
  # - This is NOT the same as ~ 0 + condition (cell means coding)
  #
  # Strategy: Keep formula as-is (with implicit intercept), but mark it
  # so build_fixed_effects_matrix_cumulative drops the intercept column.
  # This gives treatment contrasts with reference = 0.
  
  # Check if user explicitly used 0 + or - 1 (warn them)
  has_explicit_no_intercept <- grepl("\\b0\\s*\\+|\\-\\s*1\\b", mu_rhs)
  if (has_explicit_no_intercept) {
    message("Note: For cumulative models, ~ 0 + factor uses cell-means coding. ",
            "Consider using ~ factor for treatment coding (reference level = 0).")
  }
  
  # Extract random effects from mu formula
  re_pattern <- "\\([^)]+\\|[^)]+\\)"
  re_matches <- gregexpr(re_pattern, mu_rhs)
  re_terms <- regmatches(mu_rhs, re_matches)[[1]]
  
  # IDENTIFIABILITY: Check for random intercepts on mu and suppress them
  # Random intercepts on mu are not identified separately from random intercepts
  
  # on cutpoints - only their difference matters. Suppress mu random intercepts
  # and let all location variation be captured by cutpoints random effects.
  # Random SLOPES on mu (e.g., (0 + condition|subject)) ARE identified.
  
  suppressed_re <- character(0)
  retained_re <- character(0)
  
  for (re in re_terms) {
    # Parse the random effect: (terms | group) or (terms || group)
    re_inner <- gsub("^\\(|\\)$", "", re)
    # Handle || (uncorrelated): splits into 3 parts, collapse to 2
    re_parts <- strsplit(re_inner, "\\|")[[1]]
    is_uncorrelated <- length(re_parts) == 3 && trimws(re_parts[2]) == ""
    if (is_uncorrelated) {
      re_parts <- c(re_parts[1], re_parts[3])
    }
    re_lhs <- trimws(re_parts[1])
    
    # In R, random effects include an intercept UNLESS explicitly suppressed with 0+ or -1
    # So (day_after|id) implicitly means (1 + day_after|id)
    # Only (0 + day_after|id) or (day_after - 1|id) suppress the intercept
    has_explicit_no_intercept <- grepl("^0\\s*\\+", re_lhs) || grepl("-\\s*1", re_lhs)
    has_random_intercept <- !has_explicit_no_intercept
    
    if (has_random_intercept) {
      # Check if there are also slopes (anything beyond just "1" or empty)
      # Strip explicit "1" and "+" to see if slopes remain
      stripped <- gsub("^1\\s*\\+\\s*", "", re_lhs)
      stripped <- gsub("\\s*\\+\\s*1\\s*$", "", stripped)
      stripped <- gsub("\\s*\\+\\s*1\\s*\\+\\s*", " + ", stripped)
      has_slopes <- stripped != "1" && stripped != "" && stripped != re_lhs ||
        (stripped == re_lhs && re_lhs != "1" && re_lhs != "")
      
      if (has_slopes) {
        # Has slopes too - remove just the intercept, keep slopes
        # (1 + day_after | id) -> (0 + day_after | id)
        # (day_after | id) -> (0 + day_after | id)  [implicit intercept]
        new_lhs <- stripped
        if (new_lhs == "" || new_lhs == "1") {
          # Was intercept-only, suppress entirely
          suppressed_re <- c(suppressed_re, re)
        } else {
          # Keep the slopes, preserve || if uncorrelated
          bar <- if (is_uncorrelated) "||" else "|"
          new_re <- paste0("(0 + ", new_lhs, " ", bar, re_parts[2], ")")
          retained_re <- c(retained_re, new_re)
          suppressed_re <- c(suppressed_re, re)  # Mark original as suppressed for replacement
        }
      } else {
        # Intercept-only random effect - suppress entirely
        suppressed_re <- c(suppressed_re, re)
      }
    } else {
      # No random intercept (e.g., (0 + condition | subject)) - keep as is
      retained_re <- c(retained_re, re)
    }
  }
  
  # Issue message if any random intercepts were suppressed
  if (length(suppressed_re) > 0) {
    message("Note: Random intercepts on mu suppressed for identifiability. ",
            "In cumulative models, only (cutpoint - mu) is identified, so random ",
            "intercepts on mu are confounded with random intercepts on cutpoints. ",
            "Location variation is captured by cutpoints random effects.")
  }
  
  # Remove all original RE from RHS, then add back retained ones
  fixed_rhs <- mu_rhs
  for (re in re_terms) {
    fixed_rhs <- gsub(re, "", fixed_rhs, fixed = TRUE)
  }
  fixed_rhs <- gsub("\\+\\s*\\+", "+", fixed_rhs)  # Clean up double +
  fixed_rhs <- gsub("^\\s*\\+|\\+\\s*$", "", fixed_rhs)  # Remove leading/trailing +
  fixed_rhs <- trimws(fixed_rhs)
  
  # Check what's in the fixed effects
  if (fixed_rhs == "" || fixed_rhs == "1") {
    # Intercept-only or empty - mu has no fixed predictors
    has_mu_fixed <- FALSE
    if (length(retained_re) > 0) {
      mu_rhs_final <- paste(c("0", retained_re), collapse = " + ")
    } else {
      mu_rhs_final <- "0"
    }
  } else {
    has_mu_fixed <- TRUE
    # Rebuild with fixed effects and any retained random effects
    if (length(retained_re) > 0) {
      mu_rhs_final <- paste(c(fixed_rhs, retained_re), collapse = " + ")
    } else {
      mu_rhs_final <- fixed_rhs
    }
  }
  
  # Create a fake is_old variable (all 1s - no signal/noise distinction)
  data$.cumulative_is_old <- 1L
  
  # Reconstruct dprime formula with fake is_old
  dprime_formula <- as.formula(
    paste0(response_var, " | .cumulative_is_old ~ ", mu_rhs_final),
    env = environment(mu)
  )
  
  # cutpoints becomes criterion (already one-sided, no changes needed)
  criterion_formula <- cutpoints
  
  # Now use the standard SDT parsing pipeline
  parsed <- parse_broc_formula(
    dprime = dprime_formula,
    criterion = criterion_formula,
    sigma = NULL,
    lambda = NULL,
    dprime2 = NULL,
    sigma2 = NULL,
    data = data,
    family_name = "cumulative"
  )
  
  # Mark this as a cumulative model (no encoding vars possible)
  parsed$is_cumulative <- TRUE
  parsed$encoding_vars <- character(0)
  
  # Build prior lookup
  prior_lookup <- build_prior_lookup(priors, family = family)
  
  # Override dprime default for cumulative models: slopes centered at 0, not 1
  # In SDT models, normal(1, 1) makes sense for d' (typically positive).
  # In cumulative models, mu has no intercept (absorbed into cutpoints),
  # so beta_mu are regression slopes that should be centered at 0.
  if (is.null(prior_lookup$b_dpar[["dprime"]]) && identical(prior_lookup$b[["dprime"]], "normal(1, 1)")) {
    prior_lookup$b[["dprime"]] <- "normal(0, 1)"
  }
  
  # Build all data structures using the standard pipeline.
  # cumulative() has no old/new distinction, so encoding_vars is always NULL
  # here (and was previously a stray reference to an undefined variable).
  model_data <- build_model_data(parsed, data, family, counts = counts,
                                 cor_threshold = cor_threshold,
                                 encoding_vars = NULL)
  model_data$prior_lookup <- prior_lookup
  model_data$is_cumulative <- TRUE
  model_data$threads <- isTRUE(threads)
  model_data$gap_link <- gap_link

  # Add grainsize to stan_data when threading is enabled
  if (isTRUE(threads)) {
    N <- model_data$stan_data$N
    model_data$stan_data$grainsize <- max(1L, as.integer(N / 4))
  }

  # Determine batch likelihood mode
  use_batch <- isTRUE(batch_likelihood)

  # Generate batch C++ if applicable
  batch_cpp <- NULL
  if (use_batch) {
    batch_cpp <- generate_batch_cpp(model_data, family)
    if (is.null(batch_cpp)) use_batch <- FALSE
  }

  # Generate Stan code using the standard generator (with batch info if applicable)
  # The cumulative family will be handled in generate_stan_code_v2
  stan_code <- generate_stan_code_v2(model_data, family, batch_info = batch_cpp)

  # Rename parameters in Stan code for cumulative models:
  # dprime -> mu, criterion -> cutpoints
  stan_code <- rename_cumulative_params(stan_code)

  # Also rename batch C++ code for cumulative models:
  # dprime -> mu, criterion -> cutpoints in Stan declarations/calls
  if (use_batch && !is.null(batch_cpp)) {
    batch_cpp$stan_decl <- rename_cumulative_params(batch_cpp$stan_decl)
    batch_cpp$stan_call <- rename_cumulative_params(batch_cpp$stan_call)
    if (!is.null(batch_cpp$stan_partial))
      batch_cpp$stan_partial <- rename_cumulative_params(batch_cpp$stan_partial)
    if (!is.null(batch_cpp$stan_call_threaded))
      batch_cpp$stan_call_threaded <- rename_cumulative_params(batch_cpp$stan_call_threaded)
    # Rename C++ argument names that reference dprime/criterion in Stan data
    batch_cpp$cpp_code <- gsub("X_dprime", "X_mu", batch_cpp$cpp_code)
    batch_cpp$cpp_code <- gsub("beta_dprime", "beta_mu", batch_cpp$cpp_code)
    batch_cpp$cpp_code <- gsub("Z_dprime_", "Z_mu_", batch_cpp$cpp_code)
    batch_cpp$cpp_code <- gsub("u_dprime_", "u_mu_", batch_cpp$cpp_code)
    batch_cpp$cpp_code <- gsub("idx_dprime_", "idx_mu_", batch_cpp$cpp_code)
    batch_cpp$cpp_code <- gsub("X_criterion", "X_cutpoints", batch_cpp$cpp_code)
    batch_cpp$cpp_code <- gsub("idx_criterion_", "idx_cutpoints_", batch_cpp$cpp_code)
    batch_cpp$cpp_code <- gsub("Z_criterion_", "Z_cutpoints_", batch_cpp$cpp_code)
    batch_cpp$cpp_code <- gsub("u_criterion_", "u_cutpoints_", batch_cpp$cpp_code)
  }

  # Also rename in stan_data
  stan_data <- model_data$stan_data
  stan_data <- rename_stan_data_cumulative(stan_data)

  result <- list(
    parsed = parsed,
    model_data = model_data,
    stan_code = stan_code,
    stan_data = stan_data,
    family = family,
    prior_lookup = prior_lookup,
    data = data,
    threads = isTRUE(threads),
    batch_likelihood = use_batch,
    batch_cpp = batch_cpp
  )

  class(result) <- c("broc_model", "cumulative_model", "list")
  result
}


#' Rename parameters in Stan code for cumulative models
#' @noRd
rename_cumulative_params <- function(stan_code) {
  # Replace dprime -> mu (careful with word boundaries)
  stan_code <- gsub("dprime", "mu", stan_code)
  stan_code <- gsub("d'", "mu", stan_code)  # In comments
  
  # Replace criterion -> cutpoints
  stan_code <- gsub("criterion", "cutpoints", stan_code)
  stan_code <- gsub("Criterion", "Cutpoints", stan_code)  # In comments
  
  # Fix the comment at top
  stan_code <- gsub("// mu fixed effects", "// Mu fixed effects", stan_code)
  stan_code <- gsub("// Cutpoints fixed effects", "// Cutpoints fixed effects", stan_code)
  
  stan_code
}


#' Rename stan_data keys for cumulative models
#' @noRd
rename_stan_data_cumulative <- function(stan_data) {
  # Get all names
  old_names <- names(stan_data)
  
  # Replace dprime -> mu
  new_names <- gsub("dprime", "mu", old_names)
  
  # Replace criterion -> cutpoints  
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
    
    # Only process varying slopes (not intercepts)
    if (re$type != "varying_slope" || is.null(re$term)) {
      next
    }
    
    # Check if the term is a factor
    term_var <- re$term
    if (!term_var %in% names(data)) {
      next
    }
    
    if (!is.factor(data[[term_var]])) {
      next
    }
    
    # Get the reference level (first level)
    ref_level <- levels(data[[term_var]])[1]
    ref_col_name <- paste0(term_var, ref_level)
    
    # Find the reference level column in level_names
    ref_idx <- which(re$level_names == ref_col_name)
    
    if (length(ref_idx) == 0) {
      # Try without the variable name prefix (some model.matrix styles)
      ref_idx <- which(re$level_names == ref_level)
    }
    
    if (length(ref_idx) == 0 || re$dim <= 1) {
      # Reference level not found or only one level - skip
      next
    }
    
    any_dropped <- TRUE
    
    # Drop the reference level column
    keep_idx <- setdiff(1:re$dim, ref_idx)
    
    re$dim <- length(keep_idx)
    re$level_names <- re$level_names[keep_idx]
    
    # Update term_idx - need to renumber after dropping
    if (!is.null(re$term_idx)) {
      # Create mapping from old indices to new indices
      # Old indices that were ref_idx become 0 (will be skipped)
      # Other indices get renumbered
      new_idx_map <- integer(re$dim + 1)  # +1 for the dropped level
      new_idx_map[ref_idx] <- 0
      new_counter <- 1
      for (i in keep_idx) {
        new_idx_map[i] <- new_counter
        new_counter <- new_counter + 1
      }
      
      re$term_idx <- new_idx_map[re$term_idx]
    }
    
    # Update Z matrix if present
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
fit_bivariate_cumulative <- function(mu1, mu2, cutpoints1, cutpoints2, rho,
                                     data, family, priors, counts = NULL,
                                     cor_threshold = TRUE, threads = FALSE,
                                     gap_link = "log", batch_likelihood = TRUE) {

  if (is.null(mu1)) {
    stop("mu1 (primary) formula is required for bivariate_cumulative() family")
  }
  if (is.null(cutpoints1)) {
    stop("cutpoints1 formula is required for bivariate_cumulative() family")
  }
  if (is.null(cutpoints2)) {
    stop("cutpoints2 formula is required for bivariate_cumulative() family")
  }

  # Default rho to ~ 1 if not specified
  if (is.null(rho)) {
    message("Note: rho formula not specified. Using rho ~ 1.")
    rho <- ~ 1
  }

  # Default mu2 to ~ 1 if not specified
  if (is.null(mu2)) {
    message("Note: mu2 formula not specified. Using mu2 ~ 1.")
    mu2 <- ~ 1
  }

  # Parse mu1 formula to extract response variables (resp(y1, y2) ~ predictors)
  mu1_str <- paste(deparse(mu1, width.cutoff = 500), collapse = " ")
  if (!grepl("~", mu1_str)) {
    stop("mu1 formula must be two-sided: resp(y1, y2) ~ predictors")
  }

  parts <- strsplit(mu1_str, "~")[[1]]
  lhs <- trimws(parts[1])
  mu1_rhs <- trimws(parts[2])

  # Parse resp(y1, y2) from LHS -- supports both positional: resp(y1, y2)
  resp_match <- regmatches(lhs, regexec("resp\\(([^,]+),\\s*([^)]+)\\)", lhs))[[1]]
  if (length(resp_match) < 3) {
    stop("bivariate_cumulative() requires resp(y1, y2) on the LHS. ",
         "Got: ", lhs)
  }
  response1_var <- trimws(resp_match[2])
  response2_var <- trimws(resp_match[3])

  # Validate responses exist
  if (!response1_var %in% names(data)) {
    stop("Response variable '", response1_var, "' not found in data")
  }
  if (!response2_var %in% names(data)) {
    stop("Response variable '", response2_var, "' not found in data")
  }

  # --- Apply identifiability constraints to mu1 ---
  mu1_rhs <- .suppress_random_intercepts(mu1_rhs, "mu1")

  # --- Apply identifiability constraints to mu2 ---
  mu2_str <- paste(deparse(mu2, width.cutoff = 500), collapse = " ")
  if (grepl("~", mu2_str)) {
    mu2_rhs <- trimws(strsplit(mu2_str, "~")[[1]][2])
  } else {
    mu2_rhs <- mu2_str
  }
  mu2_rhs <- .suppress_random_intercepts(mu2_rhs, "mu2")

  # Create fake item_type column (all observations are "Source A" = level 2)
  data$.bivar_cum_item_type <- factor(rep("A", nrow(data)), levels = c("new", "A"))

  # Reconstruct primary formula with item_type for the parser
  dprime_formula <- as.formula(
    paste0("resp(", response1_var, ", ", response2_var,
           ") | .bivar_cum_item_type ~ ", mu1_rhs),
    env = environment(mu1)
  )

  # Construct discrim formula (one-sided)
  discrim_formula <- as.formula(paste0("~ ", mu2_rhs), env = environment(mu1))

  # cutpoints1 becomes criterion, cutpoints2 becomes criterion2
  criterion_formula <- cutpoints1
  criterion2_formula <- cutpoints2

  # Now use the standard SDT parsing pipeline
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
    data = data,
    family_name = "bivariate_cumulative"
  )

  # Mark this as a bivariate cumulative model
  parsed$is_bivariate_cumulative <- TRUE
  parsed$encoding_vars <- character(0)

  # Build prior lookup
  prior_lookup <- build_prior_lookup(priors, family = family)

  # Override dprime/discrim defaults for bivariate cumulative
  if (is.null(prior_lookup$b_dpar[["dprime"]]) && identical(prior_lookup$b[["dprime"]], "normal(1, 1)")) {
    prior_lookup$b[["dprime"]] <- "normal(0, 1)"
  }
  if (is.null(prior_lookup$b_dpar[["discrim"]]) && identical(prior_lookup$b[["discrim"]], "normal(1, 1)")) {
    prior_lookup$b[["discrim"]] <- "normal(0, 1)"
  }

  # Build all data structures using the standard pipeline
  model_data <- build_model_data(parsed, data, family, counts = counts,
                                 cor_threshold = cor_threshold)
  model_data$prior_lookup <- prior_lookup
  model_data$is_bivariate_cumulative <- TRUE
  model_data$threads <- isTRUE(threads)
  model_data$gap_link <- gap_link

  # Add grainsize to stan_data when threading is enabled
  if (isTRUE(threads)) {
    N <- model_data$stan_data$N
    model_data$stan_data$grainsize <- max(1L, as.integer(N / 4))
  }

  # Batch likelihood C++ generation
  use_batch <- isTRUE(batch_likelihood)
  batch_info <- NULL
  if (use_batch) {
    batch_info <- generate_batch_cpp(model_data, family)
  }

  # Generate Stan code
  stan_code <- generate_stan_code_v2(model_data, family, batch_info = batch_info)

  # Rename parameters in Stan code for bivariate cumulative
  stan_code <- rename_bivariate_cumulative_params(stan_code)

  # Also rename in stan_data
  stan_data <- model_data$stan_data
  stan_data <- rename_stan_data_bivariate_cumulative(stan_data)

  result <- list(
    parsed = parsed,
    model_data = model_data,
    stan_code = stan_code,
    stan_data = stan_data,
    family = family,
    prior_lookup = prior_lookup,
    data = data,
    threads = isTRUE(threads),
    batch_likelihood = use_batch,
    batch_cpp = batch_info
  )

  class(result) <- c("broc_model", "bivariate_cumulative_model", "list")
  result
}


#' Suppress random intercepts for identifiability in cumulative-type models
#' @noRd
.suppress_random_intercepts <- function(rhs, param_name) {
  re_pattern <- "\\([^)]+\\|[^)]+\\)"
  re_matches <- gregexpr(re_pattern, rhs)
  re_terms <- regmatches(rhs, re_matches)[[1]]

  suppressed_re <- character(0)
  retained_re <- character(0)

  for (re in re_terms) {
    re_inner <- gsub("^\\(|\\)$", "", re)
    re_parts <- strsplit(re_inner, "\\|")[[1]]
    re_lhs <- trimws(re_parts[1])

    has_explicit_no_intercept <- grepl("^0\\s*\\+", re_lhs) || grepl("-\\s*1", re_lhs)
    has_random_intercept <- !has_explicit_no_intercept

    if (has_random_intercept) {
      stripped <- gsub("^1\\s*\\+\\s*", "", re_lhs)
      stripped <- gsub("\\s*\\+\\s*1\\s*$", "", stripped)
      stripped <- gsub("\\s*\\+\\s*1\\s*\\+\\s*", " + ", stripped)
      has_slopes <- stripped != "1" && stripped != "" && stripped != re_lhs ||
        (stripped == re_lhs && re_lhs != "1" && re_lhs != "")

      if (has_slopes) {
        new_lhs <- stripped
        if (new_lhs == "" || new_lhs == "1") {
          suppressed_re <- c(suppressed_re, re)
        } else {
          new_re <- paste0("(0 + ", new_lhs, " |", re_parts[2], ")")
          retained_re <- c(retained_re, new_re)
          suppressed_re <- c(suppressed_re, re)
        }
      } else {
        suppressed_re <- c(suppressed_re, re)
      }
    } else {
      retained_re <- c(retained_re, re)
    }
  }

  if (length(suppressed_re) > 0) {
    message("Note: Random intercepts on ", param_name, " suppressed for identifiability. ",
            "In bivariate cumulative models, only (cutpoint - mu) is identified, so random ",
            "intercepts on ", param_name, " are confounded with random intercepts on cutpoints.")
  }

  fixed_rhs <- rhs
  for (re in re_terms) {
    fixed_rhs <- gsub(re, "", fixed_rhs, fixed = TRUE)
  }
  fixed_rhs <- gsub("\\+\\s*\\+", "+", fixed_rhs)
  fixed_rhs <- gsub("^\\s*\\+|\\+\\s*$", "", fixed_rhs)
  fixed_rhs <- trimws(fixed_rhs)

  if (fixed_rhs == "" || fixed_rhs == "1") {
    if (length(retained_re) > 0) {
      rhs_final <- paste(c("0", retained_re), collapse = " + ")
    } else {
      rhs_final <- "0"
    }
  } else {
    if (length(retained_re) > 0) {
      rhs_final <- paste(c(fixed_rhs, retained_re), collapse = " + ")
    } else {
      rhs_final <- fixed_rhs
    }
  }

  rhs_final
}


#' Rename parameters in Stan code for bivariate cumulative models
#' @noRd
rename_bivariate_cumulative_params <- function(stan_code) {
  stan_code <- gsub("dprime", "mu1", stan_code)
  stan_code <- gsub("d'", "mu1", stan_code)
  stan_code <- gsub("discrim", "mu2", stan_code)
  stan_code <- gsub("criterion2", "cutpoints2", stan_code)
  stan_code <- gsub("Criterion2", "Cutpoints2", stan_code)
  stan_code <- gsub("criterion", "cutpoints1", stan_code)
  stan_code <- gsub("Criterion", "Cutpoints1", stan_code)
  stan_code
}


#' Rename stan_data keys for bivariate cumulative models
#' @noRd
rename_stan_data_bivariate_cumulative <- function(stan_data) {
  old_names <- names(stan_data)
  new_names <- gsub("dprime", "mu1", old_names)
  new_names <- gsub("discrim", "mu2", new_names)
  new_names <- gsub("criterion2", "cutpoints2", new_names)
  new_names <- gsub("criterion", "cutpoints1", new_names)
  names(stan_data) <- new_names
  stan_data
}
