# =============================================================================
# Posterior linpred / epred + allow_new_levels support
# =============================================================================

#' Posterior draws of a linear predictor (dpar)
#'
#' Returns posterior draws of a distributional parameter at each row of `newdata`
#' on the response scale.
#'
#' @param object A `broc_fit` object from [fit_broc()].
#' @param transform Ignored.
#' @param newdata Optional data frame. If `NULL`, uses the training data.
#' @param dpar Distributional parameter to extract (default `"dprime"`). Valid
#'   names depend on the family -- the same names used in [brf()] and
#'   [broc_prior()].
#' @param re_formula `NULL` (default) includes random effects; `NA` excludes
#'   them (population-level prediction).
#' @param ndraws Optional number of posterior draws to subsample. Default
#'   uses all draws.
#' @param summary If `TRUE`, returns a data frame with `mean`, `lower`,
#'   `upper`, `trial`. If `FALSE` (default), returns the full S x N matrix.
#' @param prob CI width when `summary = TRUE` (default 0.95).
#' @param allow_new_levels If `TRUE`, sample random effects from their prior for
#'   grouping levels in `newdata` not seen during training. Default `FALSE` errors
#'   if `newdata` introduces new levels.
#' @param seed Optional integer for reproducibility (subsampling and
#'   new-level RE draws). Restores caller's RNG state on exit.
#' @param ... Ignored.
#' @return If `summary = FALSE`, an S x N matrix. If `TRUE`, a data frame with
#'   `trial`, `mean`, `lower`, `upper`.
#' @seealso [posterior_epred()] for expected response value;
#'   [predict()] for category probabilities.
#' @importFrom rstantools posterior_linpred
#' @method posterior_linpred broc_fit
#' @export
posterior_linpred.broc_fit <- function(object, transform = FALSE, newdata = NULL,
                                       dpar = "dprime",
                                       re_formula = NULL, ndraws = NULL,
                                       summary = FALSE, prob = 0.95,
                                       allow_new_levels = FALSE, seed = NULL, ...) {
  fit <- object

  if (!is.null(seed)) {
    old_seed <- if (exists(".Random.seed", envir = .GlobalEnv))
      get(".Random.seed", envir = .GlobalEnv) else NULL
    on.exit({
      if (is.null(old_seed)) {
        if (exists(".Random.seed", envir = .GlobalEnv))
          rm(".Random.seed", envir = .GlobalEnv)
      } else {
        assign(".Random.seed", old_seed, envir = .GlobalEnv)
      }
    }, add = TRUE)
    set.seed(seed)
  }

  model <- attr(fit, "broc_model")
  if (is.null(model)) {
    stop("Fit object must have 'broc_model' attribute.", call. = FALSE)
  }
  family_name <- model$family$family
  is_cumulative <- family_name == "cumulative"
  is_bivariate_cumulative <- family_name == "bivariate_cumulative"

  # Accept user-facing aliases (e.g. cdp "rec"/"fam", cumulative "mu") by
  # mapping them to internal names before validation.
  dpar <- external_to_internal_param(dpar, family_name)

  # Validate dpar
  valid_dpars <- c("dprime", "sigma", "lambda", "dprime2", "sigma2",
                   "dprime_B", "lambda_B",
                   "discrim", "discrim_B", "sigma_B", "sigma2_B",
                   "rho", "rho_B", "rho_N", "lambda2", "lambda2_B",
                   "rec_crit", "know_crit",
                   "dprime_L", "sigma_L", "lambda_L")
  if (!dpar %in% valid_dpars) {
    stop("Unknown dpar '", dpar, "'. Valid: ",
         paste(valid_dpars, collapse = ", "), call. = FALSE)
  }

  # ---- Data source (with optional new-level allowance) ----
  if (!is.null(newdata)) {
    data_src <- rebuild_design_matrices_for_linpred(model, newdata,
                                                    allow_new_levels)
  } else {
    data_src <- build_data_source_from_model(model)
  }
  N <- data_src$N

  # ---- Extract draws ----
  vars_needed <- get_predict_variables(model, include_re = !identical(re_formula, NA))
  # Also need sigma_RE / L_corr_RE for new-level sampling
  if (allow_new_levels) {
    extra <- get_re_prior_variables(model)
    meta_base <- unique(sub("\\[.*$", "", fit$metadata()$variables))
    vars_needed <- unique(c(vars_needed, intersect(extra, meta_base)))
  }
  draws <- fit$draws(variables = vars_needed, format = "matrix")
  S <- nrow(draws)
  if (!is.null(ndraws) && ndraws < S) {
    idx <- sample.int(S, ndraws)
    draws <- draws[idx, , drop = FALSE]
    S <- ndraws
  }

  # Stash allow_new_levels flag in data_src so compute_re_contribution sees it
  data_src$.allow_new_levels <- isTRUE(allow_new_levels)

  # ---- Compute ----
  trial_params <- compute_trial_params(draws, model, data_src, re_formula, S, N)

  if (!dpar %in% names(trial_params)) {
    stop("dpar '", dpar, "' is not part of the fitted model.", call. = FALSE)
  }

  # compute_trial_params transposes internally, so trial_params[[dpar]] is
  # already S x N -- no further transpose needed.
  out <- trial_params[[dpar]]

  if (!summary) return(out)

  probs_lower <- (1 - prob) / 2
  probs_upper <- 1 - probs_lower
  data.frame(
    trial = seq_len(N),
    mean  = colMeans(out),
    lower = apply(out, 2, quantile, probs = probs_lower),
    upper = apply(out, 2, quantile, probs = probs_upper)
  )
}


#' Posterior expected response value
#'
#' For ordinal models, returns `E[Y | X] = sum_k k * P(Y = k)` for each row of
#' `newdata`, per posterior draw.
#'
#' @inheritParams posterior_linpred.broc_fit
#' @param ... Further arguments passed to [predict.broc_fit()] (e.g. `cores`).
#' @return If `summary = FALSE`, an S x N matrix of expected category indices.
#'   If `TRUE`, a data frame with `trial`, `mean`, `lower`, `upper`.
#' @seealso [posterior_linpred()], [predict()].
#' @importFrom rstantools posterior_epred
#' @method posterior_epred broc_fit
#' @export
posterior_epred.broc_fit <- function(object, newdata = NULL,
                                     re_formula = NULL, ndraws = NULL,
                                     summary = FALSE, prob = 0.95,
                                     allow_new_levels = FALSE, seed = NULL,
                                     ...) {
  fit <- object
  probs <- predict(fit, newdata = newdata, type = "response",
                   re_formula = re_formula, ndraws = ndraws,
                   summary = FALSE, prob = prob,
                   seed = seed,
                   allow_new_levels = allow_new_levels,
                   ...)
  d <- dim(probs)
  if (length(d) == 4) {
    # bivariate: probs is S x N x K1 x K2 -- collapse to two epreds
    K1 <- d[3]; K2 <- d[4]
    e1 <- apply(probs, c(1, 2), function(p) sum(rowSums(matrix(p, K1, K2)) * seq_len(K1)))
    e2 <- apply(probs, c(1, 2), function(p) sum(colSums(matrix(p, K1, K2)) * seq_len(K2)))
    out <- list(y1 = e1, y2 = e2)
    if (!summary) return(out)
    summarize_epred <- function(mat, label) {
      probs_lower <- (1 - prob) / 2
      probs_upper <- 1 - probs_lower
      data.frame(trial = seq_len(ncol(mat)),
                 response = label,
                 mean = colMeans(mat),
                 lower = apply(mat, 2, quantile, probs = probs_lower),
                 upper = apply(mat, 2, quantile, probs = probs_upper))
    }
    return(rbind(summarize_epred(out$y1, "y1"),
                 summarize_epred(out$y2, "y2")))
  }
  # Univariate: S x N x K -> S x N
  K <- d[3]
  k_vec <- seq_len(K)
  out <- apply(probs, c(1, 2), function(p) sum(p * k_vec))
  if (!summary) return(out)
  probs_lower <- (1 - prob) / 2
  probs_upper <- 1 - probs_lower
  data.frame(
    trial = seq_len(ncol(out)),
    mean  = colMeans(out),
    lower = apply(out, 2, quantile, probs = probs_lower),
    upper = apply(out, 2, quantile, probs = probs_upper)
  )
}


# =============================================================================
# allow_new_levels helpers
# =============================================================================

#' @noRd
get_re_prior_variables <- function(model) {
  # Base names of the prior SD / correlation variables needed to sample
  # u ~ N(0, sigma) for new RE levels. Returns base names only (de-indexed);
  # the caller intersects against (de-indexed) fit variables and lets the draws
  # accessor expand each base to its elements.
  md <- model$model_data
  is_cumulative <- model$family$family == "cumulative"
  is_bivariate_cumulative <- model$family$family == "bivariate_cumulative"
  vars <- character(0)

  add_re <- function(vars, re_list, stan_pname) {
    for (group in names(re_list)) {
      re <- re_list[[group]]
      vars <- c(vars, paste0("sigma_", stan_pname, "_", group))
      if (isTRUE(re$correlated) && (re$dim %||% 1L) > 1L) {
        vars <- c(vars, paste0("L_corr_", stan_pname, "_", group),
                  paste0("corr_", stan_pname, "_", group))
      }
    }
    vars
  }

  # Fixed-parameter REs (flat md$<pname>_random)
  re_param_names <- c("dprime", "sigma", "lambda", "dprime2", "sigma2",
                       "dprime_B", "lambda_B", "discrim", "discrim_B",
                       "sigma_B", "sigma2_B", "rho", "rho_B", "rho_N",
                       "lambda2", "rec_crit", "know_crit",
                       "dprime_L", "sigma_L", "lambda_L")
  for (pname in re_param_names) {
    re_list <- md[[paste0(pname, "_random")]]
    if (is.null(re_list)) next
    stan_pname <- if (is_cumulative && pname == "dprime") "mu" else
      if (is_bivariate_cumulative && pname == "dprime") "mu1" else
      if (is_bivariate_cumulative && pname == "discrim") "mu2" else pname
    vars <- add_re(vars, re_list, stan_pname)
  }

  # Criterion / criterion2 REs are nested (md$criterion$random)
  crit_name <- if (is_cumulative) "cutpoints" else
    if (is_bivariate_cumulative) "cutpoints1" else "criterion"
  if (!is.null(md$criterion$random)) vars <- add_re(vars, md$criterion$random, crit_name)
  crit2_name <- if (is_bivariate_cumulative) "cutpoints2" else "criterion2"
  if (!is.null(md$criterion2$random)) vars <- add_re(vars, md$criterion2$random, crit2_name)

  # Smooth penalty SDs, for sampling a new by-level smooth from its prior.
  if (!is.null(md$smooth_data)) {
    for (pname in names(md$smooth_data)) {
      for (sm in md$smooth_data[[pname]]) {
        for (comp in sm$components) {
          for (k in seq_along(comp$Zs_list)) {
            vars <- c(vars, paste0("sds_", pname, "_", comp$san_label, "_", k))
          }
        }
      }
    }
  }

  unique(vars)
}


#' @noRd
rebuild_design_matrices_for_linpred <- function(model, newdata, allow_new_levels) {
  # Wraps rebuild_design_matrices and turns "new levels found" warnings into
  # errors when allow_new_levels = FALSE, or suppresses them when TRUE.
  was_warn <- list()
  data_src <- withCallingHandlers(
    rebuild_design_matrices(model, newdata),
    warning = function(w) {
      msg <- conditionMessage(w)
      if (grepl("New levels in grouping variable", msg, fixed = FALSE)) {
        was_warn[[length(was_warn) + 1]] <<- msg
        invokeRestart("muffleWarning")
      }
    }
  )
  if (length(was_warn) > 0 && !allow_new_levels) {
    stop(paste(c("New levels found in newdata. Set allow_new_levels = TRUE",
                 "to sample random effects from their prior, or remove the",
                 "new levels.", "", was_warn), collapse = "\n"),
         call. = FALSE)
  }
  data_src
}


# =============================================================================
# Patches to compute_re_contribution to honor data_src$.allow_new_levels
# =============================================================================
# Implementation note: the actual modification of compute_re_contribution
# lives in predict.R since it's intertwined with the existing RE-lookup loop.
# This file only provides the user-facing wrappers and the discovery of which
# extra prior-side variables to pull from the fit.
