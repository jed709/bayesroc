
# =============================================================================
# Gap Transform Helpers
# =============================================================================

.gap_transform <- function(x, gap_link = "log") {
  if (identical(gap_link, "softplus")) log1p(.exp_v(x)) else .exp_v(x)
}

.gap_param <- function(gap_link = "log", suffix = "") {
  base <- if (identical(gap_link, "softplus")) "beta_raw_gaps" else "beta_log_gaps"
  paste0(base, suffix)
}

.gap_names_1d <- function(draws, gap_prefix) {
  avail <- colnames(draws)
  test_1d <- paste0(gap_prefix, "[1]")
  test_2d <- paste0(gap_prefix, "[1,1]")
  (test_1d %in% avail) && !(test_2d %in% avail)
}

.gap_cols <- function(gap_prefix, k, P, use_1d) {
  if (use_1d) {
    paste0(gap_prefix, "[", k, "]")
  } else {
    paste0(gap_prefix, "[", k, ",", 1:P, "]")
  }
}

# =============================================================================
# Main Prediction Function
# =============================================================================

#' Posterior Predictions from a Fitted Bayesian ROC Model
#'
#' Generate posterior predictive response probabilities or simulated responses
#' from a fitted model.
#'
#' @param object A `broc_fit` object from [fit_broc()].
#' @param newdata Optional data frame for out-of-sample prediction. If `NULL`
#'   (default), uses the training data.
#' @param type `"response"` (default) returns posterior category probabilities
#'   P(Y = k); `"prediction"` returns simulated categorical responses.
#' @param re_formula Controls random effects inclusion. `NULL` (default) includes
#'   all random effects; `NA` excludes all (population-level predictions only).
#' @param ndraws Number of posterior draws to use. `NULL` (default) uses all
#'   available draws.
#' @param summary If `TRUE` (default), returns posterior mean and credible
#'   intervals. If `FALSE`, returns the full S x N x K array of draws.
#' @param prob Width of credible intervals when `summary = TRUE` (default 0.95).
#' @param cores Number of cores for the probability computation (default 1).
#' @param response Which response to predict. For bivariate families: `NULL`
#'   (default) gives the joint detection x source cells, `1` the detection
#'   response alone and `2` the source response alone, each summed over the
#'   other response. For CDP: `NULL` or `1` the confidence response, `2` the
#'   R/K/G judgement. Ignored for other families.
#' @param seed Optional integer for reproducibility. Controls the random
#'   subsampling of `ndraws` and (for `type = "prediction"`) the categorical
#'   draws used to simulate responses. The previous `.Random.seed` is restored
#'   on exit. Default `NULL` leaves the RNG untouched.
#' @param allow_new_levels If `TRUE`, allow `newdata` to contain random-effect
#'   levels not seen during fitting; their effects are drawn from the estimated
#'   population distribution, once per new level and draw, so all rows of a new
#'   level share it, and effects of a cross-parameter group are drawn jointly.
#'   Default `FALSE` errors if `newdata` introduces new levels.
#' @param ... Ignored.
#' @return Depends on `type` and `summary`:
#'   \describe{
#'     \item{`type = "response"`, `summary = TRUE`}{A data frame with one row per
#'       observation and response category: `trial` (row of the data), the data
#'       columns of that row (from `newdata`, or the data used for fitting),
#'       `category`, and the posterior `mean` probability with `lower` and `upper`
#'       interval bounds.}
#'     \item{`type = "response"`, `summary = FALSE`}{An S x N x K array where
#'       S = draws, N = observations, K = response categories.}
#'     \item{`type = "prediction"`, `summary = TRUE`}{A data frame with `trial`,
#'       the data columns, and the modal and mean predicted category.}
#'     \item{`type = "prediction"`, `summary = FALSE`}{An S x N matrix of
#'       simulated categorical responses.}
#'     \item{Missing source ratings}{In the bivariate families, when some rows
#'       have no source rating, a simulated source rating (`type = "prediction"`,
#'       data used for fitting) is `NA` wherever the design would not have asked
#'       for one: for a simulated detection response that never carries one, or,
#'       when the pattern is not a function of the detection response, for the
#'       rows that have none. Probabilities (`type = "response"`) are the model's
#'       cells.}
#'   }
#' @examples
#' \dontrun{
#' fit <- fit_broc(model)
#' # Category probabilities (summarized)
#' preds <- predict(fit)
#' # Full posterior array
#' preds_full <- predict(fit, summary = FALSE, ndraws = 100)
#' # Population-level only (no random effects)
#' preds_pop <- predict(fit, re_formula = NA)
#' }
#' @seealso [pp_check()] for visual posterior predictive checks.
#' @method predict broc_fit
#' @export
predict.broc_fit <- function(object, newdata = NULL, type = c("response", "prediction"),
                             re_formula = NULL, ndraws = NULL, summary = TRUE,
                             prob = 0.95, cores = 1, response = NULL, seed = NULL,
                             allow_new_levels = FALSE, ...) {
  out <- .predict_broc(object, newdata = newdata, type = type, re_formula = re_formula, ndraws = ndraws,
                       summary = summary, prob = prob, cores = cores, response = response, seed = seed,
                       allow_new_levels = allow_new_levels, ...)
  if (isTRUE(summary) && is.data.frame(out) && "trial" %in% names(out)) {
    src <- if (!is.null(newdata)) newdata else attr(object, "broc_model")$data
    if (!is.null(src)) {
      keep <- setdiff(names(src)[!startsWith(names(src), ".")], names(out))
      if (length(keep)) {
        cols <- src[out$trial, keep, drop = FALSE]
        rownames(cols) <- NULL
        out <- cbind(out["trial"], cols, out[setdiff(names(out), "trial")])
      }
    }
  }
  out
}

.predict_broc <- function(object, newdata = NULL, type = c("response", "prediction"),
                          re_formula = NULL, ndraws = NULL, summary = TRUE,
                          prob = 0.95, cores = 1, response = NULL, seed = NULL,
                          allow_new_levels = FALSE, margin = NULL, ...) {
  fit <- object
  type <- match.arg(type)

  restore_seed <- .seed_scope(seed)
  on.exit(restore_seed(), add = TRUE)

  model <- attr(fit, "broc_model")
  if (is.null(model)) {
    stop("Fit object must have 'broc_model' attribute. Use fit_broc() to fit the model.")
  }

  family <- model$family
  family_name <- family$family
  gap_link <- if (!is.null(model$gap_link)) model$gap_link else "log"

  supported <- c("evsdt", "uvsdt", "dpsdt", "mixture", "cumulative", "source_mixture",
                  "bivariate_sdt", "bivariate_dp", "bivariate_cumulative", "vrdp2d", "cdp", "metadt", "mafc")
  if (!family_name %in% supported) {
    stop("predict() does not yet support family '", family_display_name(family_name),
         "'. Supported: ",
         paste(vapply(supported, family_display_name, character(1)), collapse = ", "))
  }

  is_cumulative <- family_name == "cumulative"
  is_bivariate <- family_name %in% c("bivariate_sdt", "bivariate_dp", "vrdp2d", "bivariate_cumulative")
  is_cdp <- family_name == "cdp"
  link_name <- family$link$name
  K <- model$model_data$K
  K2 <- model$model_data$K2

  if (!is.null(response) && (is_bivariate || is_cdp) &&
      !(length(response) == 1L && isTRUE(as.integer(response) %in% 1:2))) {
    stop("`response` must be NULL, 1 or 2.", call. = FALSE)
  }
  one_response <- is_bivariate && is.null(margin) && !is.null(response)
  if (one_response) margin <- as.integer(response)
  if (cores > 1) {
    old_threads <- .bvn_native$threads
    .bvn_native$threads <- as.integer(cores)
    on.exit(.bvn_native$threads <- old_threads, add = TRUE)
  }

  # ---- Data source ----
  if (!is.null(newdata)) {
    data_src <- rebuild_design_matrices_for_linpred(model, newdata,
                                                    allow_new_levels)
  } else {
    data_src <- build_data_source_from_model(model)
  }
  data_src$.allow_new_levels <- isTRUE(allow_new_levels)
  data_src$.new_u_cache <- new.env(parent = emptyenv())

  N <- data_src$N

  # ---- Extract posterior draws ----
  vars_needed <- get_predict_variables(model, include_re = !identical(re_formula, NA))
  if (allow_new_levels) {
    extra <- get_re_prior_variables(model)
    meta_base <- unique(sub("\\[.*$", "", fit$metadata()$variables))
    extra <- intersect(extra, meta_base)
    vars_needed <- unique(c(vars_needed, extra))
  }
  draws <- fit$draws(variables = vars_needed, format = "matrix")
  S <- nrow(draws)

  if (!is.null(ndraws) && ndraws < S) {
    idx <- sample.int(S, ndraws)
    draws <- draws[idx, , drop = FALSE]
    S <- ndraws
  }

  # ---- Compute trial-level parameters ----
  trial_params <- compute_trial_params(draws, model, data_src, re_formula, S, N)

  # ---- Compute thresholds ----
  n_thresh_K <- if (is_cdp) family$J + 1 else K
  thresh <- if (family_name == "mafc") compute_bias(draws, model, data_src, re_formula, S, N, K)
            else compute_thresholds(draws, model, data_src, re_formula, S, N, n_thresh_K, crit_scale = trial_params[["criterion_scale"]])

  # one response's probabilities of a bivariate family, [S, N, K]: margin = 1 detection,
  # 2 source
  if (is_bivariate && identical(as.integer(margin), 1L)) {
    probs <- compute_category_probs(trial_params, thresh, data_src, model, link_name, S, N, K,
                                    cores = cores, margin = 1L)
    if (!one_response) return(probs)
  } else {
    thresh2 <- NULL
    if (is_bivariate) {
      thresh2 <- compute_thresholds_2(draws, model, data_src, re_formula, S, N)
    }

    # ---- Compute category probabilities ----
    probs <- compute_category_probs(trial_params, thresh, data_src, model, link_name, S, N, K,
                                      thresh2 = thresh2, cores = cores, margin = margin)
    if (is_bivariate && identical(as.integer(margin), 2L) && !one_response) return(probs)
  }
  if (one_response) {
    K <- dim(probs)[3]
    is_bivariate <- FALSE
  }

  # ---- Bivariate output path (4D probs: S x N x K1 x K2) ----
  if (is_bivariate) {
    K1 <- K

    if (type == "prediction") {
      pred_y1 <- matrix(0L, nrow = S, ncol = N)
      pred_y2 <- matrix(0L, nrow = S, ncol = N)
      for (s in seq_len(S)) {
        for (n in seq_len(N)) {
          p_joint <- pmax(as.vector(probs[s, n, , ]), 0)
          p_joint <- p_joint / sum(p_joint)
          cell <- sample.int(K1 * K2, 1L, prob = p_joint)
          pred_y1[s, n] <- ((cell - 1) %% K1) + 1L
          pred_y2[s, n] <- ((cell - 1) %/% K1) + 1L
        }
      }
      sdz <- if (is.null(newdata)) .source_design(model)
      if (!is.null(sdz)) {
        pred_y2[if (sdz$rule) !(pred_y1 %in% sdz$levels) else matrix(sdz$obs[col(pred_y2)] == 0L, S, N)] <- NA_integer_
      }
      if (summary) {
        pred_summary <- data.frame(
          trial = seq_len(N),
          mode_y1 = apply(pred_y1, 2, function(x) which.max(tabulate(x, nbins = K1))),
          mode_y2 = apply(pred_y2, 2, function(x) which.max(tabulate(x, nbins = K2))),
          mean_y1 = colMeans(pred_y1),
          mean_y2 = colMeans(pred_y2)
        )
        return(pred_summary)
      }
      return(list(y1 = pred_y1, y2 = pred_y2))
    }

    if (summary) {
      probs_lower <- (1 - prob) / 2
      probs_upper <- 1 - probs_lower

      result_list <- vector("list", K1 * K2)
      idx <- 0
      for (j in seq_len(K1)) {
        for (k in seq_len(K2)) {
          idx <- idx + 1
          p_jk <- probs[, , j, k]
          result_list[[idx]] <- data.frame(
            trial = seq_len(N),
            category1 = j,
            category2 = k,
            mean = colMeans(p_jk),
            lower = (q_jk <- .col_quantiles(p_jk, c(probs_lower, probs_upper)))[1, ],
            upper = q_jk[2, ]
          )
        }
      }
      result <- do.call(rbind, result_list)
      rownames(result) <- NULL
      return(result)
    }

    return(probs)
  }

  # ---- CDP response=2 path: R/K(/G) probabilities ----
  if (is_cdp && identical(as.integer(response), 2L)) {
    p_remember <- attr(probs, "p_remember")
    p_know <- attr(probs, "p_know")
    old_levels <- if (!is.null(family$rk_levels)) family$rk_levels else family$old_levels
    n_rkg <- if (!is.null(family$n_rkg)) family$n_rkg else 2L

    rk_probs <- array(0, dim = c(S, N, n_rkg))
    pl <- probs[, , old_levels, drop = FALSE]
    p_old <- rowSums(pl, dims = 2L)
    small <- p_old < 1e-10
    p_r <- rowSums(p_remember[, , old_levels, drop = FALSE] * pl, dims = 2L) / p_old
    if (n_rkg == 3 && !is.null(p_know)) {
      p_k <- rowSums(p_know[, , old_levels, drop = FALSE] * pl, dims = 2L) / p_old
      rk_probs[, , 1] <- p_r; rk_probs[, , 2] <- p_k; rk_probs[, , 3] <- pmax(1 - p_r - p_k, 0)
    } else {
      rk_probs[, , 1] <- p_r; rk_probs[, , 2] <- pmax(1 - p_r, 0)
    }
    if (any(small)) for (r in seq_len(n_rkg)) rk_probs[, , r][small] <- 1 / n_rkg

    rk_labels <- if (n_rkg == 3) c("Remember", "Know", "Guess") else c("Remember", "Know")

    if (type == "prediction") {
      pred_rk <- matrix(.pp_draw_categories(matrix(pmax(rk_probs, 0), S * N, n_rkg), n_rkg), S, N)
      if (summary) {
        return(data.frame(
          trial = seq_len(N),
          mode = apply(pred_rk, 2, function(x) rk_labels[which.max(tabulate(x, nbins = n_rkg))]),
          p_remember = colMeans(rk_probs[, , 1])
        ))
      }
      return(pred_rk)
    }

    if (summary) {
      probs_lower <- (1 - prob) / 2
      probs_upper <- 1 - probs_lower
      result_list <- vector("list", n_rkg)
      for (r in seq_len(n_rkg)) {
        result_list[[r]] <- data.frame(
          trial = seq_len(N),
          category = rk_labels[r],
          mean = colMeans(rk_probs[, , r]),
          lower = (q_r <- .col_quantiles(rk_probs[, , r], c(probs_lower, probs_upper)))[1, ],
          upper = q_r[2, ]
        )
      }
      return(do.call(rbind, result_list))
    }
    return(rk_probs)
  }

  # ---- CDP output path ----
  if (is_cdp && type == "prediction") {
    p_remember <- attr(probs, "p_remember")
    p_know <- attr(probs, "p_know")
    old_levels <- if (!is.null(family$rk_levels)) family$rk_levels else family$old_levels
    n_rkg <- if (!is.null(family$n_rkg)) family$n_rkg else 2L

    pred_y <- matrix(.pp_draw_categories(matrix(pmax(probs, 0), S * N, K), K), S, N)
    pred_rk <- matrix(NA_integer_, nrow = S, ncol = N)
    at <- cbind(as.vector(row(pred_y)), as.vector(col(pred_y)), as.vector(pred_y))
    judged <- as.vector(pred_y) %in% old_levels
    if (any(judged)) {
      p_r <- p_remember[at[judged, , drop = FALSE]]
      p_r[is.na(p_r)] <- 0
      if (n_rkg == 3 && !is.null(p_know)) {
        p_k <- p_know[at[judged, , drop = FALSE]]
        p_k[is.na(p_k)] <- 0
        rk <- .pp_draw_categories(cbind(p_r, p_k, pmax(1 - p_r - p_k, 0)), 3L)
      } else {
        rk <- ifelse(stats::runif(sum(judged)) < p_r, 1L, 2L)
      }
      pred_rk[judged] <- rk
    }

    if (summary) {
      pred_summary <- data.frame(
        trial = seq_len(N),
        mode = apply(pred_y, 2, function(x) which.max(tabulate(x, nbins = K))),
        mean = colMeans(pred_y)
      )
      return(pred_summary)
    }
    return(list(y = pred_y, rk = pred_rk))
  }

  # ---- Standard output path (univariate families + CDP response) ----
  if (type == "prediction") {
    pred_mat <- matrix(0L, nrow = S, ncol = N)
    for (s in seq_len(S)) {
      for (n in seq_len(N)) {
        p <- pmax(probs[s, n, ], 0)
        p <- p / sum(p)
        pred_mat[s, n] <- sample.int(K, 1L, prob = p)
      }
    }
    if (summary) {
      pred_summary <- data.frame(
        trial = seq_len(N),
        mode = apply(pred_mat, 2, function(x) {
          tab <- tabulate(x, nbins = K)
          which.max(tab)
        }),
        mean = colMeans(pred_mat)
      )
      return(pred_summary)
    }
    return(pred_mat)
  }

  # ---- Type = "response": return probabilities ----
  if (summary) {
    probs_lower <- (1 - prob) / 2
    probs_upper <- 1 - probs_lower

    result_list <- vector("list", K)
    for (k in seq_len(K)) {
      p_k <- probs[, , k]
      result_list[[k]] <- data.frame(
        trial = seq_len(N),
        category = k,
        mean = colMeans(p_k),
        lower = (q_k <- .col_quantiles(p_k, c(probs_lower, probs_upper)))[1, ],
        upper = q_k[2, ]
      )
    }
    result <- do.call(rbind, result_list)
    rownames(result) <- NULL
    return(result)
  }

  probs
}

# =============================================================================
# Data Source Construction
# =============================================================================

#' Build data source from the stored model (training data)
#' @noRd
build_data_source_from_model <- function(model) {
  md <- model$model_data
  is_cumulative <- model$family$family == "cumulative"

  src <- list(
    N = md$N,
    is_old = md$stan_data$is_old,
    source = md$stan_data$source
  )

  src$item_type <- md$stan_data$item_type
  src$y2 <- md$stan_data$y2
  src$rk <- md$stan_data$rk

  dprime_name <- if (is_cumulative) "mu" else "dprime"
  if (isTRUE(md$has_dprime_fixed)) {
    src$X_dprime <- md$dprime_fixed$X
  } else {
    src$X_dprime <- NULL
  }

  src$X_criterion <- md$criterion$X

  param_names <- c("sigma", "lambda", "dprime2", "sigma2",
                    "dprime_B", "lambda_B",
                    "dprime_L", "sigma_L", "lambda_L",
                    "discrim", "discrim_B", "sigma_B", "sigma2_B",
                    "rho", "rho_B", "rho_N", "lambda2", "lambda2_B", "mratio", "mratio_N", "sigma_meta", "rec_crit", "know_crit",
                    "criterion_scale", "criterion2_scale")
  for (pname in param_names) {
    if (isTRUE(md[[paste0("has_", pname)]])) {
      src[[paste0("X_", pname)]] <- md[[paste0(pname, "_fixed")]]$X
    }
  }

  all_re_params <- c("dprime", "sigma", "lambda", "dprime2", "sigma2",
                      "dprime_B", "lambda_B",
                      "dprime_L", "sigma_L", "lambda_L", "rec_crit", "know_crit",
                      "discrim", "discrim_B", "sigma_B", "sigma2_B",
                      "rho", "rho_B", "rho_N", "lambda2", "lambda2_B", "mratio", "mratio_N", "sigma_meta", "criterion_scale", "criterion2_scale")
  for (pname in all_re_params) {
    re_list <- md[[paste0(pname, "_random")]]
    if (!is.null(re_list)) {
      src[[paste0(pname, "_random")]] <- re_list
    }
  }

  src$criterion_random <- md$criterion$random
  if (identical(model$family$family, "mafc")) {
    src$target_pos <- md$stan_data$target_pos
    src$m_alt <- md$stan_data$m_alt
    src$X_bias <- md$bias$X
    src$bias_random <- md$bias$random
  }

  if (isTRUE(md$has_criterion2) && !is.null(md$criterion2)) {
    src$X_criterion2 <- md$criterion2$X
    src$criterion2_random <- md$criterion2$random
  }

  src$cross_cor <- md$cross_cor

  src$smooth_data <- md$smooth_data
  src$.frame <- model$data

  src
}

# =============================================================================
# Variable Extraction
# =============================================================================

#' Get Stan variable names needed for prediction
#' @noRd
get_predict_variables <- function(model, include_re = TRUE) {
  vars <- character(0)
  md <- model$model_data
  is_cumulative <- model$family$family == "cumulative"
  is_bivariate_cumulative <- model$family$family == "bivariate_cumulative"
  dprime_name <- if (is_cumulative || is_bivariate_cumulative) "mu" else "dprime"
  criterion_name <- if (is_cumulative || is_bivariate_cumulative) "cutpoints" else "criterion"
  K <- md$K
  is_cdp <- model$family$family == "cdp"
  gap_link <- if (!is.null(model$gap_link)) model$gap_link else "log"

  if (isTRUE(md$has_dprime_fixed)) {
    P <- md$dprime_fixed$n_coef
    if (P > 0) vars <- c(vars, paste0("beta_", dprime_name, "[", seq_len(P), "]"))
  }

  if (identical(model$family$family, "mafc")) {
    if (!is.null(md$bias)) vars <- c(vars, "beta_bias")
  } else {
  P_crit <- md$criterion$n_coef
  vars <- c(vars, paste0("beta_thresh_mid[", 1:P_crit, "]"))

  n_gaps <- if (is_cdp) model$family$J - 1 else K - 2
  if (n_gaps > 0) {
    vars <- c(vars, .gap_param(gap_link))
  }
  }

  param_names <- c("sigma", "lambda", "dprime2", "sigma2",
                    "dprime_B", "lambda_B",
                    "dprime_L", "sigma_L", "lambda_L",
                    "discrim", "discrim_B", "sigma_B", "sigma2_B",
                    "rho", "rho_B", "rho_N", "lambda2", "lambda2_B", "mratio", "mratio_N", "sigma_meta", "rec_crit", "know_crit",
                    "criterion_scale", "criterion2_scale")
  for (pname in param_names) {
    has_flag <- if (pname %in% c("dprime2")) {
      md$has_dprime2 || isTRUE(md$needs_ordered_dprime)
    } else {
      isTRUE(md[[paste0("has_", pname)]])
    }
    if (has_flag) {
      P <- md[[paste0(pname, "_fixed")]]$n_coef
      if (P > 0) {
        stan_pname <- .smooth_stan_pname(model, pname)
        vars <- c(vars, paste0("beta_", stan_pname, "[", seq_len(P), "]"))
      }
    }
  }

  if (isTRUE(md$has_criterion2)) {
    if (isTRUE(md$varying_source_criteria)) {
      K2 <- md$K2
      n_sets <- (md$stan_data$n_vbin %||% K) * .n_vcell(md)
      vars <- c(vars, paste0("beta_thresh_mid_2_varying[", seq_len(n_sets), "]"))
      if (K2 > 2) {
        vars <- c(vars, .gap_param(gap_link, "_2_varying"))
      }
      P_cov <- md$stan_data$P_criterion2_cov %||% 0L
      if (P_cov > 0) {
        vars <- c(vars, paste0("beta_thresh_mid_2_cov[", seq_len(P_cov), "]"))
        if (K2 > 2 && !isTRUE(md$has_criterion2_scale)) vars <- c(vars, .gap_param(gap_link, "_2_cov"))
      }
    } else {
      P_crit2 <- md$criterion2$n_coef
      K2 <- md$K2
      vars <- c(vars, paste0("beta_thresh_mid_2[", 1:P_crit2, "]"))
      if (K2 > 2) {
        vars <- c(vars, .gap_param(gap_link, "_2"))
      }
    }
  }

  if (include_re) {
    vars <- c(vars, get_re_variables(model))
  }

  if (!is.null(md$smooth_data)) {
    for (pname in names(md$smooth_data)) {
      for (sm in md$smooth_data[[pname]]) {
        n_thresh_sm <- if (!is.null(sm$n_thresh)) sm$n_thresh else 0L
        for (comp in sm$components) {
          for (k in seq_along(comp$Zs_list)) {
            nbasis <- ncol(comp$Zs_list[[k]])
            s_name <- paste0("s_", pname, "_", comp$san_label, "_", k)
            if (n_thresh_sm > 0) {
              for (t in seq_len(n_thresh_sm)) {
                vars <- c(vars, paste0(s_name, "[", t, ",", 1:nbasis, "]"))
              }
            } else {
              vars <- c(vars, paste0(s_name, "[", 1:nbasis, "]"))
            }
          }
        }
      }
    }
  }

  unique(vars)
}

#' Get RE variable names for prediction
#' @noRd
get_re_variables <- function(model) {
  vars <- character(0)
  md <- model$model_data
  is_cumulative <- model$family$family == "cumulative"
  is_bivariate_cumulative <- model$family$family == "bivariate_cumulative"
  criterion_name <- if (is_cumulative || is_bivariate_cumulative) "cutpoints" else "criterion"

  all_re_params <- c("dprime", "sigma", "lambda", "dprime2", "sigma2",
                      "dprime_B", "lambda_B",
                      "dprime_L", "sigma_L", "lambda_L", "rec_crit", "know_crit",
                      "discrim", "discrim_B", "sigma_B", "sigma2_B",
                      "rho", "rho_B", "rho_N", "lambda2", "lambda2_B", "mratio", "mratio_N", "sigma_meta", "criterion_scale", "criterion2_scale")
  for (pname in all_re_params) {
    re_list <- md[[paste0(pname, "_random")]]
    if (is.null(re_list)) next
    stan_pname <- .smooth_stan_pname(model, pname)

    for (group in names(re_list)) {
      re <- re_list[[group]]
      var_base <- if (!is.null(re$cor_id)) {
        paste0("u_", stan_pname, "_", group, "_from_", re$cor_id)
      } else {
        paste0("u_", stan_pname, "_", group)
      }

      vars <- c(vars, var_base)
    }
  }

  for (group in names(md$criterion$random)) {
    re <- md$criterion$random[[group]]
    var_base <- if (!is.null(re$cor_id)) {
      paste0("u_", criterion_name, "_", group, "_from_", re$cor_id)
    } else {
      paste0("u_", criterion_name, "_", group)
    }
    vars <- c(vars, var_base)
  }

  for (group in names(md$bias$random)) {
    re <- md$bias$random[[group]]
    vars <- c(vars, if (!is.null(re$cor_id)) paste0("u_bias_", group, "_from_", re$cor_id) else paste0("u_bias_", group))
  }

  crit2_stan <- if (is_bivariate_cumulative) "cutpoints2" else "criterion2"
  if (isTRUE(md$has_criterion2) && !is.null(md$criterion2$random)) {
    for (group in names(md$criterion2$random)) {
      re <- md$criterion2$random[[group]]
      var_base <- if (!is.null(re$cor_id)) {
        paste0("u_", crit2_stan, "_", group, "_from_", re$cor_id)
      } else {
        paste0("u_", crit2_stan, "_", group)
      }
      vars <- c(vars, var_base)
    }
  }

  vars
}

# =============================================================================
# Trial Parameter Computation
# =============================================================================

#' Compute trial-level parameters for all draws
#' @return Named list of S x N matrices
#' @noRd
compute_trial_params <- function(draws, model, data_src, re_formula, S, N) {
  md <- model$model_data
  is_cumulative <- model$family$family == "cumulative"
  is_bivariate_cumulative <- model$family$family == "bivariate_cumulative"
  dprime_stan <- if (is_cumulative || is_bivariate_cumulative) "mu" else "dprime"
  include_re <- !identical(re_formula, NA)
  params <- list()

  if (isTRUE(md$has_dprime_fixed)) {
    P <- md$dprime_fixed$n_coef
    if (P > 0) {
      beta <- draws[, paste0("beta_", dprime_stan, "[", seq_len(P), "]"), drop = FALSE]
      params[["dprime"]] <- data_src$X_dprime %*% t(beta)
    } else {
      params[["dprime"]] <- matrix(0, nrow = nrow(data_src$X_dprime),
                              ncol = nrow(draws))
    }
  } else {
    params[["dprime"]] <- matrix(0, nrow = N, ncol = S)
  }

  if (include_re) {
    re_list <- data_src$dprime_random
    if (!is.null(re_list)) {
      params[["dprime"]] <- params[["dprime"]] + compute_re_contribution(
        draws, re_list, data_src, "dprime", dprime_stan, S, N, md
      )
    }
  }

  params[["dprime"]] <- params[["dprime"]] + compute_smooth_contribution(
    draws, data_src$smooth_data, "dprime", S, N,
    allow_new = isTRUE(data_src$.allow_new_levels)
  )

  dprime_link <- model$family$params[["dprime"]]$link
  if (identical(dprime_link, "log")) {
    params[["dprime"]] <- exp(params[["dprime"]])
  }

  .plink <- function(pname) {
    p <- model$family$params[[pname]]
    if (!is.null(p)) p$link else "identity"
  }

  .inv_link <- function(pname) {
    lnk <- .plink(pname)
    switch(lnk,
      log = "exp", logit = "logis", logis = "logis",
      fisherz = "tanh", identity = "identity",
      "identity"
    )
  }

  param_specs <- list(
    sigma   = list(has = md$has_sigma,   link = "exp"),
    lambda  = list(has = md$has_lambda,  link = "logis"),
    dprime2 = list(has = md$has_dprime2 || isTRUE(md$needs_ordered_dprime), link = "identity"),
    sigma2  = list(has = isTRUE(md$has_sigma2), link = "exp"),
    dprime_B = list(has = isTRUE(md$has_dprime_B), link = .inv_link("dprime_B")),
    lambda_B = list(has = isTRUE(md$has_lambda_B), link = "logis"),
    dprime_L = list(has = isTRUE(md$has_dprime_L), link = "identity"),
    sigma_L  = list(has = isTRUE(md$has_sigma_L), link = "exp"),
    lambda_L = list(has = isTRUE(md$has_lambda_L), link = "logis"),
    discrim    = list(has = isTRUE(md$has_discrim),    link = .inv_link("discrim"),
                     stan_name = if (is_bivariate_cumulative) "mu2" else NULL),
    discrim_B  = list(has = isTRUE(md$has_discrim_B),  link = .inv_link("discrim_B")),
    sigma_B    = list(has = isTRUE(md$has_sigma_B),    link = "exp"),
    sigma2_B   = list(has = isTRUE(md$has_sigma2_B),   link = "exp"),
    rho        = list(has = isTRUE(md$has_rho),        link = .inv_link("rho")),
    rho_B      = list(has = isTRUE(md$has_rho_B),      link = .inv_link("rho_B")),
    rho_N      = list(has = isTRUE(md$has_rho_N),      link = "tanh"),
    lambda2    = list(has = isTRUE(md$has_lambda2),    link = "logis"),
    lambda2_B  = list(has = isTRUE(md$has_lambda2_B),  link = "logis"),
    rec_crit   = list(has = isTRUE(md$has_rec_crit),   link = "identity"),
    know_crit  = list(has = isTRUE(md$has_know_crit),  link = "identity"),
    mratio     = list(has = isTRUE(md$has_mratio),     link = "exp"),
    mratio_N   = list(has = isTRUE(md$has_mratio_N),   link = "exp"),
    sigma_meta = list(has = isTRUE(md$has_sigma_meta), link = "exp"),
    criterion_scale = list(has = isTRUE(md$has_criterion_scale), link = "exp",
                           stan_name = .smooth_stan_pname(model, "criterion_scale")),
    criterion2_scale = list(has = isTRUE(md$has_criterion2_scale), link = "exp",
                            stan_name = .smooth_stan_pname(model, "criterion2_scale"))
  )

  for (pname in names(param_specs)) {
    spec <- param_specs[[pname]]
    if (!isTRUE(spec$has)) next

    stan_pname <- if (!is.null(spec$stan_name)) spec$stan_name else pname
    P <- md[[paste0(pname, "_fixed")]]$n_coef
    X <- data_src[[paste0("X_", pname)]]
    if (P > 0) {
      beta <- draws[, paste0("beta_", stan_pname, "[", seq_len(P), "]"), drop = FALSE]
      param_raw <- X %*% t(beta)
    } else {
      param_raw <- matrix(0, nrow = N, ncol = nrow(draws))
    }

    if (include_re) {
      re_list <- data_src[[paste0(pname, "_random")]]
      if (!is.null(re_list)) {
        param_raw <- param_raw + compute_re_contribution(
          draws, re_list, data_src, pname, stan_pname, S, N, md
        )
      }
    }

    param_raw <- param_raw + compute_smooth_contribution(
      draws, data_src$smooth_data, pname, S, N,
      allow_new = isTRUE(data_src$.allow_new_levels), stan_pname = stan_pname
    )

    if (spec$link == "exp") {
      params[[pname]] <- exp(param_raw)
    } else if (spec$link == "logis") {
      params[[pname]] <- 1 / (1 + exp(-param_raw))
    } else if (spec$link == "tanh") {
      params[[pname]] <- tanh(param_raw)
    } else {
      params[[pname]] <- param_raw
    }
  }

  for (nm in names(params)) {
    params[[nm]] <- t(params[[nm]])
  }

  params
}

# =============================================================================
# Random Effects Contribution
# =============================================================================

#' Compute Smooth Contribution for One Parameter
#'
#' For each smooth component, extracts s_ draws (= sds * zs) and computes Zs %*% t(s).
#'
#' @param draws S x P matrix of posterior draws
#' @param smooth_data smooth_data list from model_data (or NULL)
#' @param pname Parameter name (e.g., "dprime")
#' @param S Number of draws
#' @param N Number of observations
#' @return N x S matrix of smooth contributions (zeros if no smooths)
#' @noRd
compute_smooth_contribution <- function(draws, smooth_data, pname, S, N,
                                         allow_new = FALSE, stan_pname = pname) {
  if (is.null(smooth_data) || is.null(smooth_data[[pname]])) {
    return(matrix(0, nrow = N, ncol = S))
  }

  contrib <- matrix(0, nrow = N, ncol = S)

  for (sm in smooth_data[[pname]]) {
    for (comp in sm$components) {
      for (k in seq_along(comp$Zs_list)) {
        Zs <- comp$Zs_list[[k]]
        nbasis <- ncol(Zs)
        if (isTRUE(comp$new_level)) {
          if (!allow_new || nbasis == 0) next
          sds_name <- paste0("sds_", stan_pname, "_", comp$san_label, "_", k)
          sds_col <- intersect(c(sds_name, paste0(sds_name, "[1]")), colnames(draws))
          if (length(sds_col) == 0) next
          sds_draws <- as.numeric(draws[, sds_col[1]])
          s_draws <- matrix(rnorm(S * nbasis), nrow = S, ncol = nbasis) * sds_draws
          contrib <- contrib + Zs %*% t(s_draws)
        } else {
          s_name <- paste0("s_", stan_pname, "_", comp$san_label, "_", k)
          s_vars <- paste0(s_name, "[", 1:nbasis, "]")
          s_draws <- draws[, s_vars, drop = FALSE]
          contrib <- contrib + Zs %*% t(s_draws)
        }
      }
    }
  }

  contrib
}

#' Compute Criterion Smooth Contribution to Thresholds
#'
#' For per-threshold criterion smooths, computes the smooth contribution
#' to the mid-anchor (modifies thresh_mid) and each gap (modifies gaps).
#'
#' @param draws S x P matrix of posterior draws
#' @param smooth_data smooth_data list from model_data
#' @param S Number of draws
#' @param N Number of observations
#' @param n_thresh Number of thresholds (K-1)
#' @return List with mid (N x S) and gaps (list of N x S matrices, one per gap)
#' @noRd
compute_criterion_smooth_contribution <- function(draws, smooth_data, S, N, n_thresh,
                                                 pname = "criterion") {
  mid_contrib <- matrix(0, nrow = N, ncol = S)
  n_gaps <- n_thresh - 1
  gap_contribs <- lapply(seq_len(n_gaps), function(g) matrix(0, nrow = N, ncol = S))

  if (is.null(smooth_data) || is.null(smooth_data[[pname]])) {
    return(list(mid = mid_contrib, gaps = gap_contribs))
  }

  for (sm in smooth_data[[pname]]) {
    for (comp in sm$components) {
      if (isTRUE(comp$new_level)) next
      for (k in seq_along(comp$Zs_list)) {
        Zs <- comp$Zs_list[[k]]
        nbasis <- ncol(Zs)
        s_name <- paste0("s_", pname, "_", comp$san_label, "_", k)

        s_mid_vars <- paste0(s_name, "[1,", 1:nbasis, "]")
        s_mid_draws <- draws[, s_mid_vars, drop = FALSE]
        mid_contrib <- mid_contrib + Zs %*% t(s_mid_draws)

        for (g in seq_len(n_gaps)) {
          s_gap_vars <- paste0(s_name, "[", 1 + g, ",", 1:nbasis, "]")
          s_gap_draws <- draws[, s_gap_vars, drop = FALSE]
          gap_contribs[[g]] <- gap_contribs[[g]] + Zs %*% t(s_gap_draws)
        }
      }
    }
  }

  list(mid = mid_contrib, gaps = gap_contribs)
}

#' Compute RE contribution for one parameter across all groups
#' @return N x S matrix of RE contributions
#' @noRd
compute_re_contribution <- function(draws, re_list, data_src, pname, stan_pname, S, N, md) {
  contrib <- matrix(0, nrow = N, ncol = S)
  allow_new <- isTRUE(data_src$.allow_new_levels)

  for (group in names(re_list)) {
    re <- re_list[[group]]
    group_idx <- re$group_idx

    var_base <- if (!is.null(re$cor_id)) {
      paste0("u_", stan_pname, "_", group, "_from_", re$cor_id)
    } else {
      paste0("u_", stan_pname, "_", group)
    }

    n_groups <- re$n_groups
    new_mask <- group_idx == 0L
    new_pos <- .new_level_pos(re, N)
    n_new <- max(c(0L, new_pos))
    u_cols <- function(d) {
      nm <- paste0(var_base, "[", 1:n_groups, ",", d, "]")
      if (re$dim == 1L && !all(nm %in% colnames(draws))) nm <- paste0(var_base, "[", 1:n_groups, "]")
      nm
    }

    if (re$dim == 1 && is.null(re$term_idx)) {
      u_names <- paste0(var_base, "[", 1:n_groups, "]")
      if (!all(u_names %in% colnames(draws))) {
        u_names <- paste0(var_base, "[", 1:n_groups, ",1]")
      }
      u_draws <- draws[, u_names, drop = FALSE]
      valid <- group_idx > 0 & group_idx <= n_groups
      if (any(valid)) {
        contrib[valid, ] <- contrib[valid, ] + t(u_draws[, group_idx[valid], drop = FALSE])
      }
      if (allow_new && n_new > 0) {
        u_new <- .draw_new_u(draws, re, pname, stan_pname, group, n_new, S, md, data_src)
        contrib[new_mask, ] <- contrib[new_mask, ] + t(matrix(u_new[, new_pos[new_mask], 1], nrow = S))
      }
    } else if (!isTRUE(re$use_z_matrix) && !is.null(re$term_idx)) {
      u_draws_list <- vector("list", re$dim)
      for (d in seq_len(re$dim)) {
        u_draws_list[[d]] <- draws[, u_cols(d), drop = FALSE]
      }
      term_idx <- re$term_idx
      u_new_all <- if (allow_new && n_new > 0) .draw_new_u(draws, re, pname, stan_pname, group, n_new, S, md, data_src)
      for (d in seq_len(re$dim)) {
        mask <- group_idx > 0 & group_idx <= n_groups & term_idx == d
        if (any(mask)) {
          contrib[mask, ] <- contrib[mask, ] + t(u_draws_list[[d]][, group_idx[mask], drop = FALSE])
        }
        new_term_mask <- new_mask & term_idx == d
        if (!is.null(u_new_all) && any(new_term_mask)) {
          contrib[new_term_mask, ] <- contrib[new_term_mask, ] +
            t(matrix(u_new_all[, new_pos[new_term_mask], d], nrow = S))
        }
      }
    } else if (isTRUE(re$use_z_matrix)) {
      Z <- re$Z
      u_draws_all <- array(NA_real_, dim = c(S, n_groups, re$dim))
      for (d in seq_len(re$dim)) {
        u_draws_all[, , d] <- draws[, u_cols(d), drop = FALSE]
      }
      valid <- group_idx > 0 & group_idx <= n_groups
      for (n in which(valid)) {
        g <- group_idx[n]
        z_n <- Z[n, ]
        u_g <- u_draws_all[, g, , drop = FALSE]
        dim(u_g) <- c(S, re$dim)
        contrib[n, ] <- contrib[n, ] + u_g %*% z_n
      }
      if (allow_new && n_new > 0) {
        u_new_all <- .draw_new_u(draws, re, pname, stan_pname, group, n_new, S, md, data_src)
        for (n_idx in which(new_mask)) {
          u_i <- u_new_all[, new_pos[n_idx], , drop = FALSE]
          dim(u_i) <- c(S, re$dim)
          contrib[n_idx, ] <- contrib[n_idx, ] + u_i %*% Z[n_idx, ]
        }
      }
    }
  }

  contrib
}

#' @noRd
sample_new_u_uncorrelated <- function(draws, stan_pname, group, dim,
                                       term = 1L, n_new, S) {
  base <- paste0("sigma_", stan_pname, "_", group)
  candidate_names <- c(
    paste0(base, "[", term, "]"),
    if (term == 1L && dim == 1L) base else NULL
  )
  hit <- candidate_names[candidate_names %in% colnames(draws)][1]
  if (is.na(hit)) {
    stop("allow_new_levels: cannot find prior SD '", base,
         "' (or '[", term, "]' variant) in fit draws. ",
         "This RE structure may not be supported with allow_new_levels = TRUE.",
         call. = FALSE)
  }
  sigma_draws <- as.numeric(draws[, hit])
  matrix(rnorm(S * n_new) * sigma_draws, nrow = S, ncol = n_new)
}

#' @noRd
sample_new_u_correlated <- function(draws, stan_pname, group, dim, n_new, S,
                                     correlated) {
  out <- array(NA_real_, dim = c(S, n_new, dim))

  if (!correlated) {
    for (d in seq_len(dim)) {
      u_d <- sample_new_u_uncorrelated(draws, stan_pname, group, dim = dim,
                                        term = d, n_new = n_new, S = S)
      out[, , d] <- u_d
    }
    return(out)
  }

  sigma_names <- paste0("sigma_", stan_pname, "_", group, "[", seq_len(dim), "]")
  if (!all(sigma_names %in% colnames(draws))) {
    stop("allow_new_levels: cannot find sigma vector for correlated RE '",
         stan_pname, "_", group, "'. ",
         "Per-bin / smooth REs are not yet supported with allow_new_levels = TRUE.",
         call. = FALSE)
  }
  sigma_mat <- as.matrix(draws[, sigma_names, drop = FALSE])

  L_prefix <- paste0("L_corr_", stan_pname, "_", group)
  L_names <- paste0(L_prefix, "[", rep(seq_len(dim), each = dim), ",",
                    rep(seq_len(dim), times = dim), "]")
  if (!all(L_names %in% colnames(draws))) {
    stop("allow_new_levels: cannot find Cholesky factor '", L_prefix,
         "[i,j]' in fit draws. RE may not be sampled with a Cholesky parameterization.",
         call. = FALSE)
  }
  L_arr <- array(as.numeric(draws[, L_names, drop = FALSE]),
                 dim = c(S, dim, dim))

  for (s in seq_len(S)) {
    sigma_s <- as.numeric(sigma_mat[s, ])
    L_s <- t(matrix(as.numeric(L_arr[s, , ]), nrow = dim, ncol = dim))
    Sigma_chol <- diag(sigma_s, nrow = dim) %*% L_s
    z <- matrix(rnorm(n_new * dim), nrow = n_new, ncol = dim)
    out[s, , ] <- z %*% t(Sigma_chol)
  }
  out
}

# =============================================================================
# Threshold Computation
# =============================================================================

#' Compute thresholds: S x N x (K-1) array
#' @noRd
compute_thresholds <- function(draws, model, data_src, re_formula, S, N, K, crit_scale = NULL) {
  md <- model$model_data
  if (isTRUE(md$has_criterion_scale)) {
    if (is.null(crit_scale)) crit_scale <- compute_trial_params(draws, model, data_src, re_formula, S, N)$criterion_scale
    return(.location_scale_thresholds(draws, model, data_src, re_formula, S, N, K, crit_scale))
  }
  is_cumulative <- model$family$family == "cumulative"
  is_bivariate_cumulative <- model$family$family == "bivariate_cumulative"
  criterion_name <- if (is_cumulative || is_bivariate_cumulative) "cutpoints" else "criterion"
  include_re <- !identical(re_formula, NA)
  gap_link <- if (!is.null(model$gap_link)) model$gap_link else "log"
  n_thresh <- K - 1
  P_crit <- md$criterion$n_coef
  X_crit <- data_src$X_criterion

  mid <- ceiling(n_thresh / 2)
  n_upper <- n_thresh - mid

  crit_random <- data_src$criterion_random

  crit_new_u <- if (include_re && !is.null(crit_random) &&
                    isTRUE(data_src$.allow_new_levels)) {
    sample_criterion_new_u(draws, crit_random, criterion_name, S, md, data_src, "criterion")
  } else NULL

  thresh <- array(NA_real_, dim = c(S, N, n_thresh))

    gap_prefix <- .gap_param(gap_link)
    use_1d <- .gap_names_1d(draws, gap_prefix)

    beta_mid <- draws[, paste0("beta_thresh_mid[", 1:P_crit, "]"), drop = FALSE]
    thresh_mid_val <- X_crit %*% t(beta_mid)

    if (include_re && !is.null(crit_random)) {
      thresh_mid_val <- thresh_mid_val + compute_criterion_re(
        draws, crit_random, data_src, criterion_name, S, N,
        thresh_k = mid, md = md, new_u_cache = crit_new_u
      )
    }

    crit_sm <- compute_criterion_smooth_contribution(
      draws, data_src$smooth_data, S, N, n_thresh
    )
    thresh_mid_val <- thresh_mid_val + crit_sm$mid

    thresh[, , mid] <- t(thresh_mid_val)

    if (n_upper > 0) {
      for (k in (mid + 1):n_thresh) {
        gap_row <- k - mid
        beta_gaps_k <- draws[, .gap_cols(gap_prefix, gap_row, P_crit, use_1d), drop = FALSE]
        log_gap <- X_crit %*% t(beta_gaps_k)

        if (include_re && !is.null(crit_random)) {
          log_gap <- log_gap + compute_criterion_re(
            draws, crit_random, data_src, criterion_name, S, N,
            thresh_k = k, md = md, new_u_cache = crit_new_u
          )
        }

        if (gap_row <= length(crit_sm$gaps)) {
          log_gap <- log_gap + crit_sm$gaps[[gap_row]]
        }

        thresh[, , k] <- thresh[, , k - 1] + .gap_transform(t(log_gap), gap_link)
      }
    }

    if (mid > 1) {
      for (k_down in seq_len(mid - 1)) {
        k <- mid - k_down
        gap_row <- n_upper + k_down
        beta_gaps_k <- draws[, .gap_cols(gap_prefix, gap_row, P_crit, use_1d), drop = FALSE]
        log_gap <- X_crit %*% t(beta_gaps_k)

        if (include_re && !is.null(crit_random)) {
          log_gap <- log_gap + compute_criterion_re(
            draws, crit_random, data_src, criterion_name, S, N,
            thresh_k = k, md = md, new_u_cache = crit_new_u
          )
        }

        if (gap_row <= length(crit_sm$gaps)) {
          log_gap <- log_gap + crit_sm$gaps[[gap_row]]
        }

        thresh[, , k] <- thresh[, , k + 1] - .gap_transform(t(log_gap), gap_link)
      }
    }

  thresh
}

.ls_shape <- function(gap_draws, n_thresh, gap_link) {
  S <- nrow(gap_draws)
  mid <- ceiling(n_thresh / 2)
  n_up <- n_thresh - mid
  shape <- matrix(0, S, n_thresh)
  for (k in seq_len(n_thresh - 1L)) {
    j <- if (k <= n_up) mid + k else mid - (k - n_up)
    shape[, j] <- .gap_transform(gap_draws[, k], gap_link)
  }
  if (mid < n_thresh) shape[, (mid + 1):n_thresh] <- t(apply(shape[, (mid + 1):n_thresh, drop = FALSE], 1, cumsum))
  if (mid > 1) shape[, (mid - 1):1] <- -t(apply(shape[, (mid - 1):1, drop = FALSE], 1, cumsum))
  shape - rowMeans(shape)
}

.location_scale_thresholds_2 <- function(draws, model, data_src, re_formula, S, N) {
  md <- model$model_data
  K1 <- md$K
  n_thresh2 <- md$K2 - 1L
  gap_link <- model$gap_link %||% "log"
  include_re <- !identical(re_formula, NA)
  c2_name <- if (model$family$family == "bivariate_cumulative") "cutpoints2" else "criterion2"
  crit2_random <- data_src$criterion2_random
  new_u <- if (include_re && !is.null(crit2_random) && isTRUE(data_src$.allow_new_levels)) {
    sample_criterion_new_u(draws, crit2_random, c2_name, S, md, data_src, "criterion2")
  }
  sm_mid <- t(compute_criterion_smooth_contribution(draws, data_src$smooth_data, S, N, 1L, pname = "criterion2")$mid)
  if (!isTRUE(md$varying_source_criteria)) {
    P2 <- md$criterion2$n_coef
    loc <- t(data_src$X_criterion2 %*% t(draws[, paste0("beta_thresh_mid_2[", seq_len(P2), "]"), drop = FALSE]))
    if (include_re && !is.null(crit2_random)) {
      loc <- loc + t(compute_criterion_re(draws, crit2_random, data_src, c2_name, S, N, thresh_k = 1L, md = md,
                                          new_u_cache = new_u))
    }
    loc <- loc + sm_mid
    gp <- .gap_param(gap_link, "_2")
    use_1d <- .gap_names_1d(draws, gp)
    shape <- .ls_shape(draws[, vapply(seq_len(n_thresh2 - 1L), function(k) .gap_cols(gp, k, 1L, use_1d), ""), drop = FALSE],
                       n_thresh2, gap_link)
    sc <- compute_trial_params(draws, model, data_src, re_formula, S, N)$criterion2_scale
    out <- array(NA_real_, c(S, N, n_thresh2))
    for (j in seq_len(n_thresh2)) out[, , j] <- loc + sc * shape[, j]
    return(out)
  }
  vbin <- .vbin_of(md)
  labels <- .vbin_labels(md)
  n_vbin <- max(vbin)
  cell <- .vcell_rows(md, data_src$X_criterion2)
  P_cov <- md$stan_data$P_criterion2_cov %||% 0L
  shift_of <- function(ds, sm) {
    if (P_cov == 0) return(sm)
    Xc <- ds$X_criterion2[, .crit2_cov_cols(md), drop = FALSE]
    sm + draws[, paste0("beta_thresh_mid_2_cov[", seq_len(P_cov), "]"), drop = FALSE] %*% t(Xc)
  }
  bin_smooth <- !is.null(md$smooth_data[["criterion2"]])
  shift <- shift_of(data_src, sm_mid)
  frame <- data_src$.frame %||% model$data
  gv <- .gap_param(gap_link, "_2_varying")
  out <- array(0, c(S, N, K1, n_thresh2))
  by_bin <- list()
  for (y1 in seq_len(K1)) {
    vb <- vbin[y1]
    key <- as.character(vb)
    if (is.null(by_bin[[key]])) {
      fb <- frame
      fb$.c2bin <- factor(rep(labels[vb], nrow(fb)), levels = labels)
      ds_b <- rebuild_design_matrices(model, fb)
      ds_b$.allow_new_levels <- data_src$.allow_new_levels
      if (bin_smooth) {
        shift <- shift_of(ds_b, t(compute_criterion_smooth_contribution(draws, ds_b$smooth_data, S, N, 1L,
                                                                        pname = "criterion2")$mid))
      }
      sets <- vb + n_vbin * (cell - 1L)
      loc <- matrix(draws[, paste0("beta_thresh_mid_2_varying[", sets, "]")], S, N) + shift
      if (include_re && !is.null(crit2_random)) {
        for (group in names(crit2_random)) {
          re <- crit2_random[[group]]
          var_base <- if (!is.null(re$cor_id)) paste0("u_", c2_name, "_", group, "_from_", re$cor_id)
                      else paste0("u_", c2_name, "_", group)
          base_col <- if (re$dim > (re$n_re_terms %||% 1L)) vb else 1L
          loc <- loc + .crit2_varying_u(draws, re, var_base, base_col, S, N,
                                        if (!is.null(new_u)) new_u[[group]], .new_level_pos(re, N))
        }
      }
      sc <- compute_trial_params(draws, model, ds_b, re_formula, S, N)$criterion2_scale
      blk <- array(0, c(S, N, n_thresh2))
      for (cc in unique(cell)) {
        on <- which(cell == cc)
        shape <- .ls_shape(draws[, paste0(gv, "[", vb + n_vbin * (cc - 1L), ",", seq_len(n_thresh2 - 1L), "]"), drop = FALSE],
                           n_thresh2, gap_link)
        for (j in seq_len(n_thresh2)) blk[, on, j] <- loc[, on, drop = FALSE] + sc[, on, drop = FALSE] * shape[, j]
      }
      by_bin[[key]] <- blk
    }
    out[, , y1, ] <- by_bin[[key]]
  }
  out
}

.location_scale_thresholds <- function(draws, model, data_src, re_formula, S, N, K, crit_scale) {
  md <- model$model_data
  criterion_name <- if (model$family$family %in% c("cumulative", "bivariate_cumulative")) "cutpoints" else "criterion"
  gap_link <- model$gap_link %||% "log"
  n_thresh <- K - 1
  P_crit <- md$criterion$n_coef
  crit_random <- data_src$criterion_random
  include_re <- !identical(re_formula, NA)
  beta_mid <- draws[, paste0("beta_thresh_mid[", seq_len(P_crit), "]"), drop = FALSE]
  loc <- data_src$X_criterion %*% t(beta_mid)
  if (include_re && !is.null(crit_random)) {
    crit_new_u <- if (isTRUE(data_src$.allow_new_levels)) {
      sample_criterion_new_u(draws, crit_random, criterion_name, S, md, data_src, "criterion")
    }
    loc <- loc + compute_criterion_re(draws, crit_random, data_src, criterion_name, S, N,
                                      thresh_k = 1L, md = md, new_u_cache = crit_new_u)
  }
  loc <- loc + compute_criterion_smooth_contribution(draws, data_src$smooth_data, S, N, 1L)$mid
  gap_prefix <- .gap_param(gap_link)
  use_1d <- .gap_names_1d(draws, gap_prefix)
  mid <- ceiling(n_thresh / 2)
  shape <- matrix(0, S, n_thresh)
  for (k in seq_len(n_thresh - 1L)) {
    g <- .gap_transform(draws[, .gap_cols(gap_prefix, k, 1L, use_1d)], gap_link)
    j <- if (k <= n_thresh - mid) mid + k else mid - (k - (n_thresh - mid))
    shape[, j] <- g
  }
  if (mid < n_thresh) shape[, (mid + 1):n_thresh] <- t(apply(shape[, (mid + 1):n_thresh, drop = FALSE], 1, cumsum))
  if (mid > 1) shape[, (mid - 1):1] <- -t(apply(shape[, (mid - 1):1, drop = FALSE], 1, cumsum))
  shape <- shape - rowMeans(shape)
  loc <- t(loc)
  thresh <- array(NA_real_, dim = c(S, N, n_thresh))
  for (j in seq_len(n_thresh)) thresh[, , j] <- loc + crit_scale * shape[, j]
  thresh
}

#' Compute criterion RE contribution for a specific threshold
#' @return N x S matrix
#' @noRd
compute_criterion_re <- function(draws, crit_random, data_src, criterion_name, S, N,
                                  thresh_k, md, new_u_cache = NULL) {
  contrib <- matrix(0, nrow = N, ncol = S)

  for (group in names(crit_random)) {
    re <- crit_random[[group]]
    group_idx <- re$group_idx
    n_groups <- re$n_groups
    new_pos <- .new_level_pos(re, N)
    u_new <- if (!is.null(new_u_cache)) new_u_cache[[group]] else NULL

    var_base <- if (!is.null(re$cor_id)) {
      paste0("u_", criterion_name, "_", group, "_from_", re$cor_id)
    } else {
      paste0("u_", criterion_name, "_", group)
    }

    n_re_terms <- if (!is.null(re$n_re_terms)) re$n_re_terms else {
      if (!is.null(re$n_cond_levels)) re$n_cond_levels else 1
    }
    one_d <- re$dim == 1L && !paste0(var_base, "[1,1]") %in% colnames(draws)
    ucols <- function(gs, d) if (one_d) paste0(var_base, "[", gs, "]") else paste0(var_base, "[", gs, ",", d, "]")

    if (isTRUE(re$use_z_matrix)) {
      Z <- re$Z
      col_start <- (thresh_k - 1) * n_re_terms + 1
      col_end <- thresh_k * n_re_terms

      u_draws_all <- array(NA_real_, dim = c(S, n_groups, re$dim))
      for (d in seq_len(re$dim)) {
        u_draws_all[, , d] <- draws[, ucols(1:n_groups, d), drop = FALSE]
      }

      for (n in seq_len(N)) {
        g <- group_idx[n]
        if (g > 0 && g <= n_groups) {
          z_n <- Z[n, ]
          u_g <- u_draws_all[, g, col_start:col_end, drop = FALSE]
          dim(u_g) <- c(S, n_re_terms)
          contrib[n, ] <- contrib[n, ] + u_g %*% z_n
        } else if (!is.null(u_new) && new_pos[n] > 0) {
          z_n <- Z[n, ]
          u_g <- u_new[, new_pos[n], col_start:col_end, drop = FALSE]
          dim(u_g) <- c(S, n_re_terms)
          contrib[n, ] <- contrib[n, ] + u_g %*% z_n
        }
      }
    } else if (n_re_terms == 1 && is.null(re$term_idx)) {
      u_draws <- matrix(NA_real_, nrow = S, ncol = n_groups)
      for (g in seq_len(n_groups)) {
        u_draws[, g] <- draws[, ucols(g, thresh_k)]
      }
      for (n in seq_len(N)) {
        g <- group_idx[n]
        if (g > 0 && g <= n_groups) {
          contrib[n, ] <- contrib[n, ] + u_draws[, g]
        } else if (!is.null(u_new) && new_pos[n] > 0) {
          contrib[n, ] <- contrib[n, ] + u_new[, new_pos[n], thresh_k]
        }
      }
    } else if (!is.null(re$term_idx)) {
      term_idx <- re$term_idx
      u_draws <- array(NA_real_, dim = c(S, n_groups, re$dim))
      for (d in seq_len(re$dim)) {
        u_draws[, , d] <- draws[, ucols(1:n_groups, d), drop = FALSE]
      }
      for (n in seq_len(N)) {
        g <- group_idx[n]
        tidx <- term_idx[n]
        col <- (thresh_k - 1) * n_re_terms + tidx
        if (tidx > 0 && col <= re$dim) {
          if (g > 0 && g <= n_groups) {
            contrib[n, ] <- contrib[n, ] + u_draws[, g, col]
          } else if (!is.null(u_new) && new_pos[n] > 0) {
            contrib[n, ] <- contrib[n, ] + u_new[, new_pos[n], col]
          }
        }
      }
    }
  }

  contrib
}

sample_criterion_new_u <- function(draws, crit_random, criterion_name, S, md, data_src = NULL,
                                   pname = "criterion") {
  cache <- list()
  for (group in names(crit_random)) {
    re <- crit_random[[group]]
    n_new <- max(c(0L, .new_level_pos(re, length(re$group_idx))))
    if (n_new == 0) next
    if (!is.null(re$cor_id)) {
      cache[[group]] <- .sample_new_u_cross(draws, md, re$cor_id, group, pname, n_new, S,
                                            data_src$.new_u_cache %||% new.env(parent = emptyenv()))
      next
    }
    sig0 <- paste0("sigma_", criterion_name, "_", group)
    sig_names <- c(sig0, paste0(sig0, "[", seq_len(max(re$dim, 1L)), "]"))
    if (!any(sig_names %in% colnames(draws))) {
      warning("allow_new_levels: prior SD for criterion RE '", group,
              "' not found in draws; new levels zeroed for this group.",
              call. = FALSE)
      next
    }
    use_corr <- isTRUE(re$correlated) && re$dim > 1
    cache[[group]] <- sample_new_u_correlated(
      draws, criterion_name, group, dim = re$dim, n_new = n_new, S = S,
      correlated = use_corr)
  }
  if (length(cache)) cache else NULL
}

# =============================================================================
# Criterion2 Threshold Computation (bivariate_sdt, vrdp2d)
# =============================================================================

#' Compute second-dimension thresholds: S x N x (K2-1) array
#'
#' For non-varying models: uses beta_thresh_mid_2, beta_log_gaps_2, X_criterion2.
#' For varying models: returns S x N x K1 x (K2-1) array indexed by item response.
#'
#' @return For non-varying: S x N x (K2-1) array.
#'   For varying: S x N x K1 x (K2-1) array.
#' @noRd
compute_thresholds_2 <- function(draws, model, data_src, re_formula, S, N) {
  md <- model$model_data
  if (isTRUE(md$has_criterion2_scale)) return(.location_scale_thresholds_2(draws, model, data_src, re_formula, S, N))
  K2 <- md$K2
  K1 <- md$K
  include_re <- !identical(re_formula, NA)
  varying <- isTRUE(md$varying_source_criteria)
  gap_link <- if (!is.null(model$gap_link)) model$gap_link else "log"
  is_bivariate_cumulative <- model$family$family == "bivariate_cumulative"
  criterion2_stan_name <- if (is_bivariate_cumulative) "cutpoints2" else "criterion2"

  n_thresh2 <- K2 - 1
  mid2 <- ceiling(n_thresh2 / 2)
  n_upper2 <- n_thresh2 - mid2

  crit2_random <- data_src$criterion2_random
  crit2_new_u <- if (include_re && !is.null(crit2_random) &&
                     isTRUE(data_src$.allow_new_levels)) {
    sample_criterion_new_u(draws, crit2_random, criterion2_stan_name, S, md, data_src, "criterion2")
  } else NULL

  if (varying) {
    n_bins <- K1
    vbin <- .vbin_of(md)

    n_vbin <- max(vbin)
    cell <- .vcell_rows(md, data_src$X_criterion2)
    beta_mid_v <- draws[, paste0("beta_thresh_mid_2_varying[", seq_len(n_vbin * .n_vcell(md)), "]"), drop = FALSE]

    thresh2 <- array(0, dim = c(S, N, K1, n_thresh2))

    P_cov <- md$stan_data$P_criterion2_cov %||% 0L
    shifts_of <- function(ds) {
      mid <- matrix(0, nrow = S, ncol = N)
      gaps <- lapply(seq_len(max(0L, n_thresh2 - 1L)), function(g) matrix(0, nrow = S, ncol = N))
      if (P_cov > 0) {
        Xc <- ds$X_criterion2[, .crit2_cov_cols(md), drop = FALSE]
        mid <- mid + draws[, paste0("beta_thresh_mid_2_cov[", seq_len(P_cov), "]"), drop = FALSE] %*% t(Xc)
        gcov <- .gap_param(gap_link, "_2_cov")
        for (g in seq_along(gaps)) {
          gaps[[g]] <- gaps[[g]] + draws[, paste0(gcov, "[", g, ",", seq_len(P_cov), "]"), drop = FALSE] %*% t(Xc)
        }
      }
      sm <- compute_criterion_smooth_contribution(draws, ds$smooth_data, S, N, n_thresh2, pname = "criterion2")
      mid <- mid + t(sm$mid)
      for (g in seq_along(gaps)) gaps[[g]] <- gaps[[g]] + t(sm$gaps[[g]])
      list(mid = mid, gaps = gaps)
    }
    bin_smooth <- !is.null(md$smooth_data[["criterion2"]])
    base_shift <- if (!bin_smooth) shifts_of(data_src)
    frame <- data_src$.frame %||% model$data
    bin_labels <- .vbin_labels(md)

    for (b in seq_len(n_bins)) {
      bin_idx <- b
      vb <- vbin[b]
      sh <- base_shift
      if (bin_smooth) {
        fb <- frame
        fb$.c2bin <- factor(rep(bin_labels[vb], nrow(fb)), levels = bin_labels)
        sh <- shifts_of(rebuild_design_matrices(model, fb))
      }
      shift_mid <- sh$mid
      shift_gap <- sh$gaps
      sets <- vb + n_vbin * (cell - 1L)

      re_shift <- matrix(0, nrow = S, ncol = N)
      if (include_re && !is.null(data_src$criterion2_random)) {
        for (group in names(data_src$criterion2_random)) {
          re <- data_src$criterion2_random[[group]]
          group_idx <- re$group_idx
          n_groups <- re$n_groups
          new_pos <- .new_level_pos(re, N)
          u_new <- if (!is.null(crit2_new_u)) crit2_new_u[[group]] else NULL
          varying_re_mode <- if (!is.null(re$varying_re_mode)) re$varying_re_mode else "shared"
          var_base <- if (!is.null(re$cor_id)) {
            paste0("u_criterion2_", group, "_from_", re$cor_id)
          } else {
            paste0("u_criterion2_", group)
          }

          n_terms <- re$n_re_terms %||% 1L
          base_col <- switch(varying_re_mode,
                             shared = 1L,
                             per_bin = if (re$dim > n_terms) vb else 1L,
                             full = (vb - 1) * n_thresh2 + mid2)
          re_shift <- re_shift + .crit2_varying_u(draws, re, var_base, base_col, S, N, u_new, new_pos)
        }
      }

      thresh2[, , bin_idx, mid2] <- matrix(beta_mid_v[, sets], nrow = S, ncol = N) + re_shift + shift_mid

      if (n_upper2 > 0) {
        for (k2 in (mid2 + 1):n_thresh2) {
          gap_row <- k2 - mid2
          beta_gap <- draws[, paste0(.gap_param(gap_link, "_2_varying"), "[", sets, ",", gap_row, "]")]

          re_gap_shift <- matrix(0, nrow = S, ncol = N)
          if (include_re && !is.null(data_src$criterion2_random)) {
            for (group in names(data_src$criterion2_random)) {
              re <- data_src$criterion2_random[[group]]
              varying_re_mode <- if (!is.null(re$varying_re_mode)) re$varying_re_mode else "shared"
              if (varying_re_mode == "full") {
                group_idx <- re$group_idx
                n_groups <- re$n_groups
                new_pos <- .new_level_pos(re, N)
                u_new <- if (!is.null(crit2_new_u)) crit2_new_u[[group]] else NULL
                var_base <- if (!is.null(re$cor_id)) {
                  paste0("u_criterion2_", group, "_from_", re$cor_id)
                } else {
                  paste0("u_criterion2_", group)
                }
                re_gap_shift <- re_gap_shift + .crit2_varying_u(draws, re, var_base, (vb - 1) * n_thresh2 + k2,
                                                                S, N, u_new, new_pos)
              }
            }
          }

          thresh2[, , bin_idx, k2] <- thresh2[, , bin_idx, k2 - 1] +
            .gap_transform(matrix(beta_gap, nrow = S, ncol = N) + re_gap_shift + shift_gap[[gap_row]],
                           gap_link)
        }
      }

      if (mid2 > 1) {
        for (k2_down in seq_len(mid2 - 1)) {
          k2 <- mid2 - k2_down
          gap_row <- n_upper2 + k2_down
          beta_gap <- draws[, paste0(.gap_param(gap_link, "_2_varying"), "[", sets, ",", gap_row, "]")]

          re_gap_shift <- matrix(0, nrow = S, ncol = N)
          if (include_re && !is.null(data_src$criterion2_random)) {
            for (group in names(data_src$criterion2_random)) {
              re <- data_src$criterion2_random[[group]]
              varying_re_mode <- if (!is.null(re$varying_re_mode)) re$varying_re_mode else "shared"
              if (varying_re_mode == "full") {
                group_idx <- re$group_idx
                n_groups <- re$n_groups
                new_pos <- .new_level_pos(re, N)
                u_new <- if (!is.null(crit2_new_u)) crit2_new_u[[group]] else NULL
                var_base <- if (!is.null(re$cor_id)) {
                  paste0("u_criterion2_", group, "_from_", re$cor_id)
                } else {
                  paste0("u_criterion2_", group)
                }
                re_gap_shift <- re_gap_shift + .crit2_varying_u(draws, re, var_base, (vb - 1) * n_thresh2 + k2,
                                                                S, N, u_new, new_pos)
              }
            }
          }

          thresh2[, , bin_idx, k2] <- thresh2[, , bin_idx, k2 + 1] -
            .gap_transform(matrix(beta_gap, nrow = S, ncol = N) + re_gap_shift + shift_gap[[gap_row]],
                           gap_link)
        }
      }
    }


    return(thresh2)

  } else {
    P_crit2 <- md$criterion2$n_coef
    X_crit2 <- data_src$X_criterion2

    crit2_random <- data_src$criterion2_random

    thresh2 <- array(NA_real_, dim = c(S, N, n_thresh2))

    gap_prefix_2 <- .gap_param(gap_link, "_2")
    use_1d_2 <- .gap_names_1d(draws, gap_prefix_2)

    beta_mid_2 <- draws[, paste0("beta_thresh_mid_2[", 1:P_crit2, "]"), drop = FALSE]
    thresh_mid_val <- X_crit2 %*% t(beta_mid_2)

    if (include_re && !is.null(crit2_random)) {
      thresh_mid_val <- thresh_mid_val + compute_criterion_re(
        draws, crit2_random, data_src, criterion2_stan_name, S, N,
        thresh_k = mid2, md = md, new_u_cache = crit2_new_u
      )
    }

    crit2_sm <- compute_criterion_smooth_contribution(
      draws, data_src$smooth_data, S, N, n_thresh2, pname = "criterion2"
    )
    thresh_mid_val <- thresh_mid_val + crit2_sm$mid
    thresh2[, , mid2] <- t(thresh_mid_val)

    if (n_upper2 > 0) {
      for (k in (mid2 + 1):n_thresh2) {
        gap_row <- k - mid2
        beta_gaps_k <- draws[, .gap_cols(gap_prefix_2, gap_row, P_crit2, use_1d_2), drop = FALSE]
        log_gap <- X_crit2 %*% t(beta_gaps_k)
        if (gap_row <= length(crit2_sm$gaps)) log_gap <- log_gap + crit2_sm$gaps[[gap_row]]

        if (include_re && !is.null(crit2_random)) {
          log_gap <- log_gap + compute_criterion_re(
            draws, crit2_random, data_src, criterion2_stan_name, S, N,
            thresh_k = k, md = md, new_u_cache = crit2_new_u
          )
        }

        thresh2[, , k] <- thresh2[, , k - 1] + .gap_transform(t(log_gap), gap_link)
      }
    }

    if (mid2 > 1) {
      for (k_down in seq_len(mid2 - 1)) {
        k <- mid2 - k_down
        gap_row <- n_upper2 + k_down
        beta_gaps_k <- draws[, .gap_cols(gap_prefix_2, gap_row, P_crit2, use_1d_2), drop = FALSE]
        log_gap <- X_crit2 %*% t(beta_gaps_k)
        if (gap_row <= length(crit2_sm$gaps)) log_gap <- log_gap + crit2_sm$gaps[[gap_row]]

        if (include_re && !is.null(crit2_random)) {
          log_gap <- log_gap + compute_criterion_re(
            draws, crit2_random, data_src, criterion2_stan_name, S, N,
            thresh_k = k, md = md, new_u_cache = crit2_new_u
          )
        }

        thresh2[, , k] <- thresh2[, , k + 1] - .gap_transform(t(log_gap), gap_link)
      }
    }

    return(thresh2)
  }
}

# =============================================================================
# Family-Specific Probability Functions
# =============================================================================

# Detection-response probabilities of the bivariate families, [S, N, K1]: the joint cell
# probabilities summed over the source response. That margin is a univariate normal (a
# two-part mixture for the recollection models), so no bivariate integral is needed.
.detection_margin_parts <- function(family_name, params, item_type, S, N, model) {
  md <- model$model_data
  L <- S * N
  vec <- function(nm, alt) if (!is.null(params[[nm]])) as.vector(params[[nm]]) else alt
  one <- rep(1, L)
  if (family_name == "bivariate_cumulative") {
    old <- rep(TRUE, L)
    mu <- vec("dprime", rep(0, L)); sig <- one
  } else {
    it <- rep(item_type, each = S)
    old <- it != 1L
    isB <- it == 3L
    mu <- as.vector(params[["dprime"]])
    sig <- vec("sigma", one)
    if (family_name != "vrdp2d") {
      if (isTRUE(md$has_dprime_B)) mu[isB] <- as.vector(params[["dprime_B"]])[isB]
      if (!is.null(params[["sigma_B"]])) sig[isB] <- as.vector(params[["sigma_B"]])[isB]
    }
    mu[!old] <- 0
    sig[!old] <- 1
  }
  lam <- if (family_name %in% c("bivariate_dp", "vrdp2d")) ifelse(old, as.vector(params[["lambda"]]), 0) else NULL
  if (family_name == "bivariate_dp" && !is.null(params[["lambda_B"]])) lam[isB] <- as.vector(params[["lambda_B"]])[isB]
  mu_rec <- if (family_name == "vrdp2d") mu + ifelse(old, as.vector(params[["dprime2"]]), 0) else NULL
  list(mu = mu, sig = sig, lam = lam, mu_rec = mu_rec,
       kind = switch(family_name, bivariate_dp = 1L, vrdp2d = 2L, 0L))
}

probs_detection_margin <- function(family_name, params, thresh, item_type, S, N, K1, model) {
  pt <- .detection_margin_parts(family_name, params, item_type, S, N, model)
  mu <- pt[["mu"]]; sig <- pt[["sig"]]; lam <- pt[["lam"]]; mu_rec <- pt[["mu_rec"]]
  L <- S * N
  one <- rep(1, L)
  out <- array(0, dim = c(S, N, K1))
  below <- numeric(L)
  for (j in seq_len(K1)) {
    upto <- if (j == K1) one else {
      cj <- as.vector(thresh[, , j])
      f <- .pnorm_v((cj - mu) / sig)
      if (pt[["kind"]] == 1L) (1 - lam) * f
      else if (pt[["kind"]] == 2L) (1 - lam) * f + lam * .pnorm_v((cj - mu_rec) / sig)
      else f
    }
    out[, , j] <- upto - below
    below <- upto
  }
  out
}

# =============================================================================
# Source-response margin of a bivariate family, [S, N, K2], without forming the K1 x K2 table.
# Every cell formula is the one the joint functions use (the same helpers on the same quantities);
# the detection response is summed out analytically where the source criteria do not depend on
# it (the margin of a bivariate normal is univariate; the bounded model has its own marginal
# helper) and by accumulating the existing cells otherwise. options(bayesroc.source_margin_fast =
# FALSE) selects the joint-then-sum path.
# =============================================================================

# Phi2 of a standard bivariate normal at (z1, z2) elementwise; edges may be infinite.
.bvn_cdf_v <- function(z1, z2, rho) {
  n <- length(rho)
  out <- numeric(n)
  lo <- (z1 == -Inf) | (z2 == -Inf)
  i1 <- which(z1 == Inf & !lo); if (length(i1)) out[i1] <- .pnorm_v(z2[i1])
  i2 <- which(z2 == Inf & z1 != Inf & !lo); if (length(i2)) out[i2] <- .pnorm_v(z1[i2])
  fin <- which(!lo & z1 != Inf & z2 != Inf)
  if (length(fin)) {
    out[fin] <- if (.bvn_native_ready() && isTRUE(all(abs(rho[fin]) < 1))) .bvn_native_cdf(z1[fin], z2[fin], rho[fin])
                else binormal_cdf_r(z1[fin], z2[fin], rho[fin])
  }
  out
}

# The trial-level quantities of the Gaussian bivariate families for the trials `cols`, as the joint
# functions assemble them: means, SDs and correlation per (draw, trial) with the A/B sign
# conventions and the bounded-model flips applied, plus the item-type index sets.
.biv_trial_setup <- function(params, item_type, cols, S, md) {
  nc <- length(cols); L <- S * nc
  it <- rep(item_type[cols], each = S)
  iA <- which(it == 2L); iB <- which(it == 3L); iold <- which(it != 1L); inew <- which(it == 1L)
  sub <- function(m) as.vector(m[, cols, drop = FALSE])
  opt <- function(nm, alt) if (!is.null(params[[nm]])) sub(params[[nm]]) else alt
  by_type <- function(new, A, B) { v <- rep_len(new, L); v[iA] <- A[iA]; v[iB] <- B[iB]; v }
  one <- rep(1, L)
  dA <- sub(params[["dprime"]])
  mu1 <- by_type(0, dA, if (isTRUE(md$has_dprime_B)) sub(params[["dprime_B"]]) else dA)
  qA <- sub(params[["discrim"]])
  # B items: discrim_B and rho_B are the signed source-axis position and correlation of B items whatever the
  # model (as in the likelihood); omitted, they mirror the A values. The bounded model is written with A
  # mirrored, so the old items are flipped below and the given B values are pre-flipped here.
  bnd <- isTRUE(md$bounded)
  sgn <- if (bnd) -1 else 1
  mu2 <- by_type(0, qA, if (isTRUE(md$has_discrim_B)) sgn * sub(params[["discrim_B"]]) else -qA)
  s1A <- opt("sigma", one); sig1 <- by_type(1, s1A, opt("sigma_B", s1A))
  s2A <- opt("sigma2", one); sig2 <- by_type(1, s2A, opt("sigma2_B", s2A))
  rA <- sub(params[["rho"]])
  rho <- by_type(opt("rho_N", 0), rA, if (!is.null(params[["rho_B"]])) sgn * sub(params[["rho_B"]]) else -rA)
  if (bnd) { mu2[iold] <- -mu2[iold]; rho[iold] <- -rho[iold] }
  lA <- opt("lambda", double(L))
  l2A <- opt("lambda2", double(L))
  list(nc = nc, L = L, it = it, iA = iA, iB = iB, iold = iold, inew = inew, sub = sub,
       mu1 = mu1, mu2 = mu2, sig1 = sig1, sig2 = sig2, rho = rho,
       lam = by_type(0, lA, opt("lambda_B", lA)), lam2 = by_type(0, l2A, opt("lambda2_B", l2A)))
}

# Source criterion k (0 = -Inf, K2 = +Inf) for detection response j, as a length-L vector, with
# the varying-criteria case exactly as the joint functions read it.
.source_edge <- function(thresh2, cols, j, k, K2, L, nc, varying) {
  if (k < 1L) return(rep(-Inf, L))
  if (k > K2 - 1L) return(rep(Inf, L))
  if (varying) as.vector(thresh2[, cols, j, k])
  else as.vector(thresh2[, cols, k])
}

probs_source_margin <- function(family_name, params, thresh, thresh2, item_type, S, N, K1, K2, model) {
  md <- model$model_data
  varying <- isTRUE(md$varying_source_criteria)
  bounded <- isTRUE(md$bounded)
  j_dependent <- varying
  nat <- .source_margin_native(family_name, params, thresh, thresh2, item_type, S, N, K1, K2, md)
  if (!is.null(nat)) return(nat)
  out <- array(0, dim = c(S, N, K2))
  per <- max(1L, as.integer(2e6 %/% S))

  if (family_name == "vrdp2d") {
    one_cell <- function(mu, sig, lo, hi) .cell_v((lo - mu) / sig, (hi - mu) / sig)
    for (b0 in seq(1L, N, by = per)) {
      cols <- b0:min(N, b0 + per - 1L); nc <- length(cols); L <- S * nc
      it <- rep(item_type[cols], each = S); isB <- it == 3L; isnew <- it == 1L
      sub <- function(m) as.vector(m[, cols, drop = FALSE])
      zero <- rep(0, L); one <- rep(1, L)
      dF <- sub(params[["dprime"]]); dR <- dF + sub(params[["dprime2"]]); lam <- sub(params[["lambda"]])
      s1 <- if (!is.null(params[["sigma"]])) sub(params[["sigma"]]) else one
      s2 <- if (!is.null(params[["sigma2"]])) sub(params[["sigma2"]]) else one
      qA <- sub(params[["discrim"]])
      src <- if (isTRUE(md$has_discrim_B)) ifelse(isB, sub(params[["discrim_B"]]), qA) else ifelse(isB, -qA, qA)
      # new items: familiarity only, unit normal on both dimensions
      muF1 <- ifelse(isnew, 0, dF); sF1 <- ifelse(isnew, 1, s1)
      muR1 <- ifelse(isnew, 0, dR)
      muR2 <- ifelse(isnew, 0, src); sR2 <- ifelse(isnew, 1, s2)
      lam[isnew] <- 0
      edge <- function(j, k) .source_edge(thresh2, cols, j, k, K2, L, nc, varying)
      if (!varying) {
        for (k in seq_len(K2)) {
          c_lo <- edge(1L, k - 1L); c_hi <- edge(1L, k)
          out[, cols, k] <- (1 - lam) * one_cell(zero, one, c_lo, c_hi) + lam * one_cell(muR2, sR2, c_lo, c_hi)
        }
      } else {
        acc <- vector("list", K2); for (k in seq_len(K2)) acc[[k]] <- numeric(L)
        for (j in seq_len(K1)) {
          y_lo <- if (j > 1L) as.vector(thresh[, cols, j - 1L]) else rep(-Inf, L)
          y_hi <- if (j < K1) as.vector(thresh[, cols, j]) else rep(Inf, L)
          p1F <- one_cell(muF1, sF1, y_lo, y_hi); p1R <- one_cell(muR1, sF1, y_lo, y_hi)
          for (k in seq_len(K2)) {
            c_lo <- edge(j, k - 1L); c_hi <- edge(j, k)
            acc[[k]] <- acc[[k]] + (1 - lam) * p1F * one_cell(zero, one, c_lo, c_hi) + lam * p1R * one_cell(muR2, sR2, c_lo, c_hi)
          }
        }
        for (k in seq_len(K2)) out[, cols, k] <- acc[[k]]
      }
    }
    return(pmax(out, 0))
  }

  if (family_name == "bivariate_cumulative") {
    for (b0 in seq(1L, N, by = per)) {
      cols <- b0:min(N, b0 + per - 1L); nc <- length(cols); L <- S * nc
      mu2 <- if (!is.null(params[["discrim"]])) as.vector(params[["discrim"]][, cols, drop = FALSE]) else rep(0, L)
      for (k in seq_len(K2)) {
        c_lo <- if (k > 1L) as.vector(thresh2[, cols, k - 1L]) else rep(-Inf, L)
        c_hi <- if (k < K2) as.vector(thresh2[, cols, k]) else rep(Inf, L)
        out[, cols, k] <- .cell_v(c_lo - mu2, c_hi - mu2)
      }
    }
    return(pmax(out, 0))
  }

  dp <- family_name == "bivariate_dp"
  for (b0 in seq(1L, N, by = per)) {
    cols <- b0:min(N, b0 + per - 1L)
    st <- .biv_trial_setup(params, item_type, cols, S, md)
    L <- st$L; nc <- st$nc; iold <- st$iold; inew <- st$inew
    mu1 <- st$mu1; mu2 <- st$mu2; sig1 <- st$sig1; sig2 <- st$sig2; rho <- st$rho
    edge <- function(j, k) .source_edge(thresh2, cols, j, k, K2, L, nc, varying)
    lam <- if (dp) st$lam else NULL
    lam2 <- if (dp) st$lam2 else NULL
    fam <- vector("list", K2); for (k in seq_len(K2)) fam[[k]] <- numeric(L)
    if (!j_dependent) {
      # familiarity summed over the detection response = the source margin
      for (k in seq_len(K2)) {
        c_lo <- edge(1L, k - 1L); c_hi <- edge(1L, k)
        v <- .pnorm_v((c_hi - mu2) / sig2) - .pnorm_v((c_lo - mu2) / sig2)
        if (bounded && length(iold)) v[iold] <- .bounded_marginal_source_v(c_lo[iold], c_hi[iold], mu1[iold], mu2[iold], sig1[iold], sig2[iold], rho[iold])
        fam[[k]] <- v
      }
    } else {
      # criteria depend on the detection response: accumulate the cells over j, sharing the Phi2
      # corners between neighbouring source cells for the unbounded trials
      for (j in seq_len(K1)) {
        y_lo <- if (j > 1L) as.vector(thresh[, cols, j - 1L]) else rep(-Inf, L)
        y_hi <- if (j < K1) as.vector(thresh[, cols, j]) else rep(Inf, L)
        iu <- if (bounded) inew else seq_len(L)
        if (length(iu)) {
          z1lo <- (y_lo[iu] - mu1[iu]) / sig1[iu]; z1hi <- (y_hi[iu] - mu1[iu]) / sig1[iu]; r <- rho[iu]
          F_prev_hi <- numeric(length(iu)); F_prev_lo <- numeric(length(iu))
          for (k in seq_len(K2)) {
            c_hi <- edge(j, k)
            z2 <- (c_hi[iu] - mu2[iu]) / sig2[iu]
            F_hi <- .bvn_cdf_v(z1hi, z2, r); F_lo <- .bvn_cdf_v(z1lo, z2, r)
            fam[[k]][iu] <- fam[[k]][iu] + (F_hi - F_prev_hi) - (F_lo - F_prev_lo)
            F_prev_hi <- F_hi; F_prev_lo <- F_lo
          }
        }
        if (bounded && length(iold)) {
          for (k in seq_len(K2)) {
            c_lo <- edge(j, k - 1L); c_hi <- edge(j, k)
            fam[[k]][iold] <- fam[[k]][iold] + .bounded_cell_v(y_lo[iold], y_hi[iold], c_lo[iold], c_hi[iold], mu1[iold], mu2[iold], sig1[iold], sig2[iold], rho[iold])
          }
        }
      }
    }
    for (k in seq_len(K2)) {
      v <- fam[[k]]
      if (dp && length(iold)) {
        i <- iold
        c_lo <- edge(K1, k - 1L); c_hi <- edge(K1, k)
        src <- if (bounded) .bounded_marginal_source_v(c_lo[i], c_hi[i], mu1[i], mu2[i], sig1[i], sig2[i], rho[i])
               else .cell_v((c_lo[i] - mu2[i]) / sig2[i], (c_hi[i] - mu2[i]) / sig2[i])
        corner <- (st$it[i] == 2L & k == 1L) | (st$it[i] == 3L & k == K2)
        v[i] <- (1 - lam[i]) * v[i] + lam[i] * (1 - lam2[i]) * src + corner * lam[i] * lam2[i]
      }
      out[, cols, k] <- v
    }
  }
  pmax(out, 0)
}

# P(source response above the criterion path) for a bivariate family, [S, N, U], directly: for each grid
# point u the detection response is summed out with the path's source criterion c_j(u) as the lower edge
# and +Inf as the upper edge, so nothing is formed per bin and no tail sums are needed. `path` is
# [S, N, K1, U] when the source criteria vary with the detection response, else [S, N, U]. The cell
# formulas are those of probs_source_margin / the joint functions.
probs_source_survival <- function(family_name, params, thresh, path, item_type, S, N, K1, U, model) {
  md <- model$model_data
  bounded <- isTRUE(md$bounded)
  by_j <- length(dim(path)) == 4L
  out <- array(0, dim = c(S, N, U))
  per <- max(1L, as.integer(2e6 %/% S))
  cpath <- function(cols, j, u, L, nc) {
    if (by_j) as.vector(path[, cols, j, u]) else as.vector(path[, cols, u])
  }

  if (family_name == "vrdp2d") {
    cell <- function(mu, sig, lo, hi) .cell_v((lo - mu) / sig, (hi - mu) / sig)
    for (b0 in seq(1L, N, by = per)) {
      cols <- b0:min(N, b0 + per - 1L); nc <- length(cols); L <- S * nc
      it <- rep(item_type[cols], each = S); isB <- it == 3L; isnew <- it == 1L
      sub <- function(m) as.vector(m[, cols, drop = FALSE])
      one <- rep(1, L)
      dF <- sub(params[["dprime"]]); dR <- dF + sub(params[["dprime2"]]); lam <- sub(params[["lambda"]])
      s1 <- if (!is.null(params[["sigma"]])) sub(params[["sigma"]]) else one
      s2 <- if (!is.null(params[["sigma2"]])) sub(params[["sigma2"]]) else one
      qA <- sub(params[["discrim"]])
      src <- if (isTRUE(md$has_discrim_B)) ifelse(isB, sub(params[["discrim_B"]]), qA) else ifelse(isB, -qA, qA)
      muF1 <- ifelse(isnew, 0, dF); sF1 <- ifelse(isnew, 1, s1); muR1 <- ifelse(isnew, 0, dR)
      muR2 <- ifelse(isnew, 0, src); sR2 <- ifelse(isnew, 1, s2); lam[isnew] <- 0
      wF <- vector("list", K1); wR <- vector("list", K1)
      for (j in seq_len(K1)) {
        y_lo <- if (j > 1L) as.vector(thresh[, cols, j - 1L]) else rep(-Inf, L)
        y_hi <- if (j < K1) as.vector(thresh[, cols, j]) else rep(Inf, L)
        wF[[j]] <- (1 - lam) * cell(muF1, sF1, y_lo, y_hi); wR[[j]] <- lam * cell(muR1, sF1, y_lo, y_hi)
      }
      if (!by_j) {
        pF <- Reduce(`+`, wF); pR <- Reduce(`+`, wR)
        for (u in seq_len(U)) { cu <- cpath(cols, 1L, u, L, nc); out[, cols, u] <- pF * (1 - .pnorm_v(cu)) + pR * (1 - .pnorm_v((cu - muR2) / sR2)) }
      } else {
        for (u in seq_len(U)) {
          acc <- numeric(L)
          for (j in seq_len(K1)) { cu <- cpath(cols, j, u, L, nc); acc <- acc + wF[[j]] * (1 - .pnorm_v(cu)) + wR[[j]] * (1 - .pnorm_v((cu - muR2) / sR2)) }
          out[, cols, u] <- acc
        }
      }
    }
    return(pmin(pmax(out, 0), 1))
  }

  if (family_name == "bivariate_cumulative") {
    for (b0 in seq(1L, N, by = per)) {
      cols <- b0:min(N, b0 + per - 1L); nc <- length(cols); L <- S * nc
      mu2 <- if (!is.null(params[["discrim"]])) as.vector(params[["discrim"]][, cols, drop = FALSE]) else rep(0, L)
      for (u in seq_len(U)) out[, cols, u] <- 1 - .pnorm_v(cpath(cols, 1L, u, L, nc) - mu2)
    }
    return(pmin(pmax(out, 0), 1))
  }

  dp <- family_name == "bivariate_dp"
  for (b0 in seq(1L, N, by = per)) {
    cols <- b0:min(N, b0 + per - 1L)
    st <- .biv_trial_setup(params, item_type, cols, S, md)
    L <- st$L; nc <- st$nc; iold <- st$iold; inew <- st$inew
    mu1 <- st$mu1; mu2 <- st$mu2; sig1 <- st$sig1; sig2 <- st$sig2; rho <- st$rho
    lam <- if (dp) st$lam else NULL
    lam2 <- if (dp) st$lam2 else NULL
    iu <- if (bounded) inew else seq_len(L)
    z1 <- lapply(seq_len(K1), function(j) list(
      lo = if (j > 1L) as.vector(thresh[, cols, j - 1L]) else rep(-Inf, L),
      hi = if (j < K1) as.vector(thresh[, cols, j]) else rep(Inf, L)))
    for (u in seq_len(U)) {
      fam <- numeric(L)
      if (!by_j) {
        cu <- cpath(cols, 1L, u, L, nc)
        fam <- 1 - .pnorm_v((cu - mu2) / sig2)
        if (bounded && length(iold)) fam[iold] <- .bounded_marginal_source_v(cu[iold], rep(Inf, length(iold)), mu1[iold], mu2[iold], sig1[iold], sig2[iold], rho[iold])
      } else {
        for (j in seq_len(K1)) {
          cu <- cpath(cols, j, u, L, nc)
          if (length(iu)) {
            z2 <- (cu[iu] - mu2[iu]) / sig2[iu]
            a <- (z1[[j]]$lo[iu] - mu1[iu]) / sig1[iu]; b <- (z1[[j]]$hi[iu] - mu1[iu]) / sig1[iu]
            # P(y in bin j, src > c) = P(y in bin j) - P(y in bin j, src <= c)
            fam[iu] <- fam[iu] + (.pnorm_v(b) - .pnorm_v(a)) - (.bvn_cdf_v(b, z2, rho[iu]) - .bvn_cdf_v(a, z2, rho[iu]))
          }
          if (bounded && length(iold))
            fam[iold] <- fam[iold] + .bounded_cell_v(z1[[j]]$lo[iold], z1[[j]]$hi[iold], cu[iold], rep(Inf, length(iold)), mu1[iold], mu2[iold], sig1[iold], sig2[iold], rho[iold])
        }
      }
      v <- fam
      if (dp && length(iold)) {
        i <- iold
        cu <- cpath(cols, K1, u, L, nc)
        srcs <- if (bounded) .bounded_marginal_source_v(cu[i], rep(Inf, length(i)), mu1[i], mu2[i], sig1[i], sig2[i], rho[i])
                else 1 - .pnorm_v((cu[i] - mu2[i]) / sig2[i])
        cornerB <- st$it[i] == 3L
        v[i] <- (1 - lam[i]) * v[i] + lam[i] * (1 - lam2[i]) * srcs + cornerB * lam[i] * lam2[i]
      }
      out[, cols, u] <- v
    }
  }
  pmin(pmax(out, 0), 1)
}


# The source margin through the native source-survival kernels (vrdp2d; bivariate_sdt and bivariate_dp,
# bounded or not): the marginal survival of every trial at each of its source criteria, differenced into the
# K2 categories. NULL (R path) without the library or the kernel. options(bayesroc.source_margin_native =
# FALSE) forces the R path.
.source_margin_native <- function(family_name, params, thresh, thresh2, item_type, S, N, K1, K2, md) {
  if (!isTRUE(getOption("bayesroc.source_margin_native", TRUE)) || !.bvn_native_ready()) return(NULL)
  J1 <- K1 - 1L; J2 <- K2 - 1L
  varying <- length(dim(thresh2)) == 4L
  u <- as.double(seq_len(J2)); w <- rep(1, N)
  dims <- c(S, N, J1, J2, J2, varying, 1L, .native_threads())
  if (family_name == "vrdp2d") {
    if (is.null(.bvn_native$sym_srcsurv)) return(NULL)
    pt <- .vrdp2d_source_parts(params, item_type, S, N, md)
    sv <- .C(.bvn_native$sym_srcsurv, as.double(thresh), as.double(thresh2), as.double(pt[["muF"]]), as.double(pt[["sig1"]]),
             as.double(pt[["muR"]]), as.double(pt[["lam"]]), as.double(pt[["mu2"]]), as.double(pt[["sig2"]]), u, w,
             as.integer(c(dims, 1L)), 8.0, acc = double(S * N * J2), NAOK = TRUE)$acc
  } else if (family_name %in% c("bivariate_sdt", "bivariate_dp")) {
    if (is.null(.bvn_native$sym_srcsurv_g)) return(NULL)
    dp <- family_name == "bivariate_dp"
    st <- .biv_trial_setup(params, item_type, seq_len(N), S, md)
    oldv <- as.double(st$it != 1L) + as.double(st$it == 3L)
    lam <- if (dp) st$lam else double(S * N)
    lam2 <- if (dp) st$lam2 else double(S * N)
    sv <- .C(.bvn_native$sym_srcsurv_g, as.double(thresh), as.double(thresh2), as.double(st$mu1), as.double(st$mu2),
             as.double(st$sig1), as.double(st$sig2), as.double(st$rho), oldv, as.double(lam), as.double(lam2), u, w,
             as.integer(c(dims, isTRUE(md$bounded), dp, 1L)), 8.0, acc = double(S * N * J2), NAOK = TRUE)$acc
  } else {
    return(NULL)
  }
  sv <- array(sv, c(S, N, J2))
  out <- array(0, c(S, N, K2))
  out[, , 1L] <- 1 - sv[, , 1L]
  if (J2 > 1L) for (k in 2:J2) out[, , k] <- sv[, , k - 1L] - sv[, , k]
  out[, , K2] <- sv[, , J2]
  pmax(out, 0)
}

# Type-7 quantiles of every column of m at `probs`, as a length(probs) x ncol(m) matrix: one sort per
# column instead of one quantile() call (with its label formatting) per column.
.col_quantiles <- function(m, probs) {
  m <- as.matrix(m); S <- nrow(m)
  srt <- matrix(m[order(rep(seq_len(ncol(m)), each = S), m, method = "radix")], nrow = S)
  h <- (S - 1) * probs + 1; j <- pmin(pmax(floor(h), 1), S); g <- h - j; jp <- pmin(j + 1, S)
  out <- matrix(0, length(probs), ncol(m))
  for (i in seq_along(probs)) out[i, ] <- (1 - g[i]) * srt[j[i], ] + g[i] * srt[jp[i], ]
  out
}

#' Compute category probabilities (dispatcher)
#' @return S x N x K array (or S x N x K1 x K2 for bivariate families)
#' @noRd
# The package objects a cluster worker needs to run `roots`: the transitive closure of the
# functions and package-level constants they reference, exported from the package
# environment (not the installed namespace, so a load_all() session works too).
.export_pkg_fns <- function(cl, roots) {
  pkg_env <- environment(compute_category_probs)
  have <- ls(pkg_env, all.names = TRUE)
  need <- character(); todo <- roots
  while (length(todo)) {
    nm <- todo[1]; todo <- todo[-1]
    if (nm %in% need || !(nm %in% have)) next
    need <- c(need, nm)
    obj <- get(nm, envir = pkg_env)
    if (is.function(obj)) {
      g <- codetools::findGlobals(obj, merge = FALSE)
      todo <- c(todo, setdiff(intersect(c(g$functions, g$variables), have), need))
    }
  }
  parallel::clusterExport(cl, need, envir = pkg_env)
}

compute_category_probs <- function(trial_params, thresh, data_src, model, link_name, S, N, K,
                                    thresh2 = NULL, cores = 1, margin = NULL) {
  family_name <- model$family$family
  K2 <- model$model_data$K2
  is_bivariate <- family_name %in% c("bivariate_sdt", "bivariate_dp", "vrdp2d", "bivariate_cumulative")
  native <- .bvn_native_ready() && (is_bivariate || (family_name == "cdp" && !is.null(.bvn_native$sym_cdp) &&
                                                     isTRUE(getOption("bayesroc.cdp_native", TRUE))))
  if (native && cores > 1) {
    # the native kernels thread the integrals themselves
    old_threads <- .bvn_native$threads
    .bvn_native$threads <- as.integer(cores)
    on.exit(.bvn_native$threads <- old_threads, add = TRUE)
  }
  if (is_bivariate && identical(as.integer(margin), 1L)) {
    return(probs_detection_margin(family_name, trial_params, thresh, data_src$item_type, S, N, K, model))
  }
  if (is_bivariate && identical(as.integer(margin), 2L) && isTRUE(getOption("bayesroc.source_margin_fast", TRUE))) {
    return(probs_source_margin(family_name, trial_params, thresh, thresh2, data_src$item_type, S, N, K, K2, model))
  }

  compute_chunk <- function(idx) {
    n_chunk <- length(idx)
    tp <- lapply(trial_params, function(m) m[, idx, drop = FALSE])
    th <- thresh[, idx, , drop = FALSE]
    th2 <- if (!is.null(thresh2)) {
      if (length(dim(thresh2)) == 4) {
        thresh2[, idx, , , drop = FALSE]
      } else {
        thresh2[, idx, , drop = FALSE]
      }
    }
    ds <- list(
      is_old = if (!is.null(data_src$is_old)) data_src$is_old[idx] else NULL,
      source = if (!is.null(data_src$source)) data_src$source[idx] else NULL,
      item_type = if (!is.null(data_src$item_type)) data_src$item_type[idx] else NULL,
      target_pos = if (!is.null(data_src$target_pos)) data_src$target_pos[idx] else NULL,
      m_alt = if (!is.null(data_src$m_alt)) data_src$m_alt[idx] else NULL
    )

    switch(family_name,
      evsdt          = probs_evsdt(tp, th, ds$is_old, link_name, S, n_chunk, K, model$model_data$meta_noise),
      uvsdt          = probs_uvsdt(tp, th, ds$is_old, link_name, S, n_chunk, K),
      dpsdt          = probs_dpsdt(tp, th, ds$is_old, link_name, S, n_chunk, K,
                                    has_sigma = isTRUE(model$model_data$has_sigma)),
      mixture        = probs_mixture(tp, th, ds$is_old, link_name, S, n_chunk, K, model = model),
      cumulative     = probs_cumulative(tp, th, link_name, S, n_chunk, K),
      source_mixture = probs_source_mixture(tp, th, ds$source, link_name, S, n_chunk, K, model = model),
      bivariate_sdt  = probs_bivariate_sdt(tp, th, th2, ds$item_type, S, n_chunk, K, K2, model = model),
      bivariate_dp   = probs_bivariate_dp(tp, th, th2, ds$item_type, S, n_chunk, K, K2, model = model),
      bivariate_cumulative = probs_bivariate_cumulative(tp, th, th2, S, n_chunk, K, K2, model = model),
      vrdp2d         = probs_vrdp2d(tp, th, th2, ds$item_type, S, n_chunk, K, K2, model = model),
      cdp            = probs_cdp(tp, th, ds$is_old, S, n_chunk, K, model = model),
      metadt         = probs_metadt(tp, th, ds$is_old, S, n_chunk, K, model = model),
      mafc           = probs_mafc(tp, th, ds, S, n_chunk, K)
    )
  }

  source_margin <- function(p) {
    if (is_bivariate && identical(as.integer(margin), 2L)) rowSums(aperm(p, c(1, 2, 4, 3)), dims = 3L) else p
  }
  if (native && family_name != "vrdp2d") return(source_margin(compute_chunk(seq_len(N))))
  if (cores <= 1 || N < 2) {
    return(source_margin(compute_chunk(seq_len(N))))
  }

  n_workers <- min(cores, N)
  chunks <- split(seq_len(N), rep(seq_len(n_workers), length.out = N))

  chunk_data <- lapply(chunks, function(idx) {
    list(
      idx = idx,
      trial_params = lapply(trial_params, function(m) m[, idx, drop = FALSE]),
      thresh = thresh[, idx, , drop = FALSE],
      thresh2 = if (!is.null(thresh2)) {
        if (length(dim(thresh2)) == 4) thresh2[, idx, , , drop = FALSE] else thresh2[, idx, , drop = FALSE]
      },
      is_old = if (!is.null(data_src$is_old)) data_src$is_old[idx] else NULL,
      source = if (!is.null(data_src$source)) data_src$source[idx] else NULL,
      item_type = if (!is.null(data_src$item_type)) data_src$item_type[idx] else NULL,
      target_pos = if (!is.null(data_src$target_pos)) data_src$target_pos[idx] else NULL,
      m_alt = if (!is.null(data_src$m_alt)) data_src$m_alt[idx] else NULL
    )
  })

  for (i in seq_along(chunk_data)) {
    chunk_data[[i]]$family_name <- family_name
    chunk_data[[i]]$link_name <- link_name
    chunk_data[[i]]$S <- S
    chunk_data[[i]]$K <- K
    chunk_data[[i]]$K2 <- K2
    chunk_data[[i]]$model <- model
  }

  rm(trial_params, thresh, thresh2, data_src)

  cl <- parallel::makeCluster(n_workers)
  on.exit(parallel::stopCluster(cl))
  parallel::clusterCall(cl, .libPaths, .libPaths())

  .export_pkg_fns(cl, "compute_category_probs")

  .predict_worker <- function(cd) {
    n_chunk <- length(cd$idx)
    ds <- list(is_old = cd$is_old, source = cd$source, item_type = cd$item_type, target_pos = cd$target_pos, m_alt = cd$m_alt)
    switch(cd$family_name,
      evsdt          = probs_evsdt(cd$trial_params, cd$thresh, ds$is_old, cd$link_name, cd$S, n_chunk, cd$K,
                                   cd$model$model_data$meta_noise),
      uvsdt          = probs_uvsdt(cd$trial_params, cd$thresh, ds$is_old, cd$link_name, cd$S, n_chunk, cd$K),
      dpsdt          = probs_dpsdt(cd$trial_params, cd$thresh, ds$is_old, cd$link_name, cd$S, n_chunk, cd$K,
                                    has_sigma = isTRUE(cd$model$model_data$has_sigma)),
      mixture        = probs_mixture(cd$trial_params, cd$thresh, ds$is_old, cd$link_name, cd$S, n_chunk, cd$K,
                                      model = cd$model),
      cumulative     = probs_cumulative(cd$trial_params, cd$thresh, cd$link_name, cd$S, n_chunk, cd$K),
      source_mixture = probs_source_mixture(cd$trial_params, cd$thresh, ds$source, cd$link_name, cd$S, n_chunk, cd$K,
                                             model = cd$model),
      bivariate_sdt  = probs_bivariate_sdt(cd$trial_params, cd$thresh, cd$thresh2, ds$item_type,
                                            cd$S, n_chunk, cd$K, cd$K2, model = cd$model),
      bivariate_dp   = probs_bivariate_dp(cd$trial_params, cd$thresh, cd$thresh2, ds$item_type,
                                            cd$S, n_chunk, cd$K, cd$K2, model = cd$model),
      bivariate_cumulative = probs_bivariate_cumulative(cd$trial_params, cd$thresh, cd$thresh2,
                                            cd$S, n_chunk, cd$K, cd$K2, model = cd$model),
      vrdp2d         = probs_vrdp2d(cd$trial_params, cd$thresh, cd$thresh2, ds$item_type,
                                     cd$S, n_chunk, cd$K, cd$K2, model = cd$model),
      cdp            = probs_cdp(cd$trial_params, cd$thresh, ds$is_old, cd$S, n_chunk, cd$K, model = cd$model),
      metadt         = probs_metadt(cd$trial_params, cd$thresh, ds$is_old, cd$S, n_chunk, cd$K, model = cd$model),
      mafc           = probs_mafc(cd$trial_params, cd$thresh, ds, cd$S, n_chunk, cd$K)
    )
  }
  environment(.predict_worker) <- globalenv()

  chunk_results <- parallel::clusterApply(cl, chunk_data, .predict_worker)

  if (is_bivariate) {
    probs <- array(0, dim = c(S, N, K, K2))
    for (i in seq_along(chunks)) {
      probs[, chunks[[i]], , ] <- chunk_results[[i]]
    }
  } else {
    probs <- array(0, dim = c(S, N, K))
    for (i in seq_along(chunks)) {
      probs[, chunks[[i]], ] <- chunk_results[[i]]
    }
  }

  if (family_name == "cdp" && !is.null(attr(chunk_results[[1]], "p_remember"))) {
    p_rem <- array(0, dim = c(S, N, K))
    for (i in seq_along(chunks)) {
      p_rem[, chunks[[i]], ] <- attr(chunk_results[[i]], "p_remember")
    }
    attr(probs, "p_remember") <- p_rem

    if (!is.null(attr(chunk_results[[1]], "p_know"))) {
      p_kn <- array(0, dim = c(S, N, K))
      for (i in seq_along(chunks)) {
        p_kn[, chunks[[i]], ] <- attr(chunk_results[[i]], "p_know")
      }
      attr(probs, "p_know") <- p_kn
    }
  }

  source_margin(probs)
}

#' EVSDT probabilities
#' @noRd
probs_evsdt <- function(params, thresh, is_old, link_name, S, N, K, meta_noise = NULL) {
  if (!is.null(params[["sigma_meta"]])) {
    cells <- if (identical(meta_noise, "lognormal")) .lnoise_cells else .gnoise_cells
    return(pmax(cells(params[["dprime"]], params[["sigma_meta"]], thresh, matrix(is_old, S, N, byrow = TRUE), K),
                0))
  }
  .ordinal_cells(thresh, sweep(params[["dprime"]], 2, is_old, "*"), 1, link_name, K)
}

#' Meta-d' probabilities
#' @noRd
probs_metadt <- function(params, thresh, is_old, S, N, K, model) {
  md <- model$model_data
  one <- matrix(1, S, N)
  m_old <- params[["mratio"]] %||% one
  m_new <- if (isTRUE(md$metad_mratio_shared)) m_old else params[["mratio_N"]] %||% one
  pmax(.metad_cells(params[["dprime"]], m_old, m_new, thresh, matrix(is_old, S, N, byrow = TRUE),
                    md$stan_data$is_old_mid, identical(md$metac, "relative"), K, model$family$link$name), 0)
}

.link_p <- function(x, link = "probit", lower.tail = TRUE) {
  switch(link,
    probit = stats::pnorm(x, lower.tail = lower.tail),
    logit = stats::plogis(x, lower.tail = lower.tail),
    cloglog = if (lower.tail) -expm1(-exp(pmin(x, 700))) else exp(-exp(pmin(x, 700))),
    stop("Unknown link: ", link))
}

.link_med <- function(link) if (identical(link, "cloglog")) log(log(2)) else 0

.metad_cells <- function(d, m_old, m_new, thresh, a, q, relative, K, link = "probit") {
  h <- K %/% 2L
  med <- .link_med(link)
  pdiff <- function(lo, hi) ifelse(lo >= med, .link_p(lo, link, FALSE) - .link_p(hi, link, FALSE),
                                   .link_p(hi, link) - .link_p(lo, link))
  cc <- thresh[, , h]
  mid <- q * d
  mu <- a * d
  out <- array(0, c(dim(d), K))
  for (old in c(FALSE, TRUE)) {
    M <- if (old) m_old else m_new
    mu_m <- mid + M * (mu - mid)
    s <- if (relative) (M - 1) * (cc - mid) else 0
    t1 <- .link_p(cc - mu, link, lower.tail = !old)
    t2 <- .link_p(cc + s - mu_m, link, lower.tail = !old)
    for (k in if (old) (h + 1L):K else seq_len(h)) {
      lo <- if (k == 1L) array(-Inf, dim(d)) else thresh[, , k - 1L] + s - mu_m
      hi <- if (k == K) array(Inf, dim(d)) else thresh[, , k] + s - mu_m
      out[, , k] <- t1 * pdiff(lo, hi) / t2
    }
  }
  out
}

.gnoise_cells <- function(d, sm, thresh, a, K) {
  h <- K %/% 2L
  mu <- a * d
  tau <- sqrt(1 + sm^2)
  rho <- as.vector(1 / tau)
  zc <- as.vector(thresh[, , h] - mu)
  out <- array(0, c(dim(d), K))
  inf <- rep(Inf, length(zc))
  for (k in seq_len(K)) {
    old <- k > h
    lo <- if (k == 1L || k == h + 1L) -inf else as.vector((thresh[, , k - 1L] - mu) / tau)
    hi <- if (k == K || k == h) inf else as.vector((thresh[, , k] - mu) / tau)
    out[, , k] <- if (old) .bvn_rect_v(zc, inf, lo, hi, rho) else .bvn_rect_v(-inf, zc, lo, hi, rho)
  }
  out
}

.lnoise_cells <- function(d, sm, thresh, a, K) {
  h <- K %/% 2L
  mu <- as.vector(a * d)
  c0 <- as.vector(thresh[, , h])
  s <- as.vector(sm)
  n <- length(c0)
  out <- array(0, c(dim(d), K))
  for (k in seq_len(K)) {
    if (k > h) {
      Dn <- if (k == h + 1L) rep(0, n) else as.vector(thresh[, , k - 1L]) - c0
      Df <- if (k == K) rep(Inf, n) else as.vector(thresh[, , k]) - c0
      A <- mu - c0
    } else {
      Dn <- if (k == h) rep(0, n) else c0 - as.vector(thresh[, , k])
      Df <- if (k == 1L) rep(Inf, n) else c0 - as.vector(thresh[, , k - 1L])
      A <- c0 - mu
    }
    out[, , k] <- .lnoise_cell_v(A, log(Dn), log(Df), s)
  }
  out
}

# P(D_near e^(sZ) < T <= D_far e^(sZ)), T ~ N(a, 1), Z ~ N(0, 1), elementwise, from ln = log D_near
# (-Inf: D = 0) and lf = log D_far (Inf: open); the native kernel when it is there
.lnoise_cell_v <- function(a, ln, lf, s) {
  n <- length(a)
  if (n && .bvn_native_ready() && !is.null(.bvn_native$sym_lnoise)) {
    return(exp(.C(.bvn_native$sym_lnoise, as.double(a), as.double(ln), as.double(lf), as.double(s),
                  as.integer(n), .native_threads(), lp = double(n), NAOK = TRUE)$lp))
  }
  .lnoise_cell_r(a, ln, lf, s)
}

.lnoise_cell_r <- function(a, ln, lf, s) {
  R <- 8.5
  gl <- .gauss_legendre(56L)
  gh <- .gauss_hermite_e(24L)
  ghlw <- log(gh$w) + 0.5 * gh$x^2
  n <- length(a)
  Pa <- stats::pnorm(a)
  Qa <- stats::pnorm(a, lower.tail = FALSE)
  lam <- function(x) exp(stats::dnorm(x, log = TRUE) - stats::pnorm(x, log.p = TRUE))
  kd <- function(u, Pa, Qa) ifelse(u >= 0, stats::pnorm(u, lower.tail = FALSE) - Qa, Pa - stats::pnorm(u))
  pint <- function(lo, hi) ifelse(lo >= 0, stats::pnorm(lo, lower.tail = FALSE) - stats::pnorm(hi, lower.tail = FALSE),
                                  stats::pnorm(hi) - stats::pnorm(lo))
  cum <- function(l) {
    w <- exp(l); x <- a - w; L <- lam(x); sw <- s * w
    zl <- -L * sw; zr <- rep(0, n)
    z <- -zl / (-1 - L * s * sw - L * (x + L) * sw * sw)
    for (it in seq_len(40L)) {
      w <- exp(l + s * z); x <- a - w; L <- lam(x); sw <- s * w
      g <- -z - L * sw
      d2 <- -1 - L * s * sw - L * (x + L) * sw * sw
      zl <- ifelse(g > 0, z, zl); zr <- ifelse(g > 0, zr, z)
      zn <- z - g / d2
      z <- ifelse(zn > zl & zn < zr, zn, 0.5 * (zl + zr))
    }
    w <- exp(l + s * z); x <- a - w; L <- lam(x); sw <- s * w
    d2 <- -1 - L * s * sw - L * (x + L) * sw * sw
    zs <- (-log(s) - l) / s
    agh <- zs >= z + R
    C <- K <- numeric(n)
    i <- which(agh)
    if (length(i)) {
      sc <- 1 / sqrt(-d2[i])
      h0 <- -0.5 * z[i]^2 + stats::pnorm(a[i] - exp(l[i] + s[i] * z[i]), log.p = TRUE)
      zg <- z[i] + outer(sc, gh$x)
      ug <- a[i] - exp(l[i] + s[i] * zg)
      eg <- exp(sweep(-0.5 * zg^2, 2, ghlw, "+") - h0)
      cg <- sc * exp(h0) / sqrt(2 * pi)
      C[i] <- cg * rowSums(eg * stats::pnorm(ug))
      K[i] <- cg * rowSums(eg * kd(ug, Pa[i], Qa[i]))
    }
    i <- which(!agh)
    if (length(i)) {
      zc <- pmin(z[i], zs[i]); lo <- zc - R; hi <- pmin(zc + R, zs[i])
      hz <- 0.5 * pmax(hi - lo, 0)
      zz <- (lo + hz) + outer(hz, gl$x)
      u <- a[i] - exp(l[i] + s[i] * zz)
      pz <- sweep(stats::dnorm(zz), 2, gl$w, "*") * hz
      es <- 1 / s[i] - a[i]
      elo <- pmax(es, -R)
      net <- es - ifelse(zs[i] < 0, lam(zs[i]), 0)
      ehi <- ifelse(es > 0 & net * R > 40, es + 40 / ifelse(net > 0, net, 1), pmax(es, 0) + R)
      he <- 0.5 * pmax(ehi - elo, 0)
      ee <- (elo + he) + outer(he, gl$x)
      Z <- (log(pmax(a[i] + ee, 1e-300)) - l[i]) / s[i]
      pe <- sweep(stats::dnorm(ee), 2, gl$w, "*") * he
      zsm <- matrix(zs[i], length(i), ncol(Z))
      diff <- ifelse(zsm >= 0, stats::pnorm(zsm, lower.tail = FALSE) - stats::pnorm(Z, lower.tail = FALSE),
                     stats::pnorm(Z) - stats::pnorm(zsm))
      Qzs <- stats::pnorm(zs[i], lower.tail = FALSE)
      C[i] <- rowSums(pz * stats::pnorm(u)) + rowSums(pe * diff)
      K[i] <- rowSums(pz * kd(u, Pa[i], Qa[i])) + rowSums(pe * stats::pnorm(Z, lower.tail = FALSE)) +
        pint(-a[i], elo) * Qzs
    }
    list(C = C, K = K)
  }
  on <- ln == -Inf
  of <- lf == Inf
  cn <- cum(ifelse(on, 0, ln))
  cf <- cum(ifelse(of, 0, lf))
  Cn <- ifelse(on, Pa, cn$C); Kn <- ifelse(on, 0, cn$K)
  Cf <- ifelse(of, 0, cf$C); Kf <- ifelse(of, Pa, cf$K)
  pmax(ifelse(Kf < Cn, Kf - Kn, Cn - Cf), 0)
}

.quad_rules <- new.env(parent = emptyenv())

.golub_welsch <- function(b, mass) {
  n <- length(b) + 1L
  J <- matrix(0, n, n)
  J[cbind(seq_len(n - 1L), 2:n)] <- b
  J[cbind(2:n, seq_len(n - 1L))] <- b
  e <- eigen(J, symmetric = TRUE)
  o <- order(e$values)
  list(x = e$values[o], w = mass * e$vectors[1, o]^2)
}

.gauss_legendre <- function(n) {
  key <- paste0("gl", n)
  if (is.null(.quad_rules[[key]])) {
    k <- seq_len(n - 1L)
    .quad_rules[[key]] <- .golub_welsch(k / sqrt(4 * k^2 - 1), 2)
  }
  .quad_rules[[key]]
}

.gauss_hermite_e <- function(n) {
  key <- paste0("gh", n)
  if (is.null(.quad_rules[[key]])) .quad_rules[[key]] <- .golub_welsch(sqrt(seq_len(n - 1L)), sqrt(2 * pi))
  .quad_rules[[key]]
}

#' UVSDT probabilities
#' @noRd
probs_uvsdt <- function(params, thresh, is_old, link_name, S, N, K) {
  s <- params[["sigma"]]
  s[, is_old <= 0] <- 1
  .ordinal_cells(thresh, sweep(params[["dprime"]], 2, is_old, "*"), s, link_name, K)
}

#' DPSDT probabilities
#' @noRd
probs_dpsdt <- function(params, thresh, is_old, link_name, S, N, K, has_sigma) {
  old <- is_old > 0
  s <- if (has_sigma) params[["sigma"]] else matrix(1, S, N)
  s[, !old] <- 1
  probs <- .ordinal_cells(thresh, sweep(params[["dprime"]], 2, is_old, "*"), s, link_name, K)
  if (any(old)) {
    lam <- params[["lambda"]][, old, drop = FALSE]
    p <- probs[, old, , drop = FALSE] * as.vector(1 - lam)
    p[, , K] <- p[, , K] + lam
    probs[, old, ] <- p
  }
  if (!is.null(params[["lambda_L"]]) && any(!old)) {
    lam <- params[["lambda_L"]][, !old, drop = FALSE]
    p <- probs[, !old, , drop = FALSE] * as.vector(1 - lam)
    p[, , 1] <- p[, , 1] + lam
    probs[, !old, ] <- p
  }
  pmax(probs, 0)
}

#' Mixture SDT probabilities
#' @noRd
probs_mixture <- function(params, thresh, is_old, link_name, S, N, K, model) {
  md <- model$model_data
  dprime <- params[["dprime"]]
  sigma <- if (!is.null(params[["sigma"]])) params[["sigma"]] else matrix(1, S, N)
  dprime2 <- if (!is.null(params[["dprime2"]])) params[["dprime2"]] else matrix(0, S, N)
  sigma2 <- if (!is.null(params[["sigma2"]])) params[["sigma2"]] else matrix(1, S, N)
  lure <- isTRUE(md$has_lure_mixture) && !is.null(params[["dprime_L"]]) && !is.null(params[["lambda_L"]])
  probs <- array(0, dim = c(S, N, K))
  cols <- function(m, j) m[, j, drop = FALSE]
  old <- which(is_old > 0)
  if (length(old)) {
    th <- thresh[, old, , drop = FALSE]
    io <- matrix(is_old[old], S, length(old), byrow = TRUE)
    p1 <- .ordinal_cells(th, io * cols(dprime, old), cols(sigma, old), link_name, K)
    probs[, old, ] <- if (is.null(params[["lambda"]])) p1 else {
      lam <- cols(params[["lambda"]], old)
      p2 <- .ordinal_cells(th, cols(dprime2, old) - (1 - io) * cols(dprime, old), cols(sigma2, old), link_name, K)
      p1 * as.vector(lam) + p2 * as.vector(1 - lam)
    }
  }
  new <- which(is_old <= 0)
  if (length(new)) {
    th <- thresh[, new, , drop = FALSE]
    if (lure) {
      lamL <- cols(params[["lambda_L"]], new)
      sL <- if (!is.null(params[["sigma_L"]])) cols(params[["sigma_L"]], new) else 1
      probs[, new, ] <- .ordinal_cells(th, -cols(params[["dprime_L"]], new), sL, link_name, K) * as.vector(lamL) +
        .ordinal_cells(th, 0, 1, link_name, K) * as.vector(1 - lamL)
    } else {
      io <- matrix(is_old[new], S, length(new), byrow = TRUE)
      probs[, new, ] <- .ordinal_cells(th, io * cols(dprime, new), 1, link_name, K)
    }
  }
  pmax(probs, 0)
}

#' Cumulative ordinal probabilities
#' @noRd
probs_cumulative <- function(params, thresh, link_name, S, N, K) {
  .ordinal_cells(thresh, params[["dprime"]], 1, link_name, K)
}

#' Source mixture SDT probabilities
#' @noRd
probs_source_mixture <- function(params, thresh, source, link_name, S, N, K, model) {
  md <- model$model_data
  B <- source != 0
  d <- params[["dprime"]]
  if (any(B)) d[, B] <- if (isTRUE(md$has_dprime_B)) params[["dprime_B"]][, B] else -params[["dprime"]][, B]
  lam <- params[["lambda"]]
  if (any(B) && isTRUE(md$has_lambda_B)) lam[, B] <- params[["lambda_B"]][, B]
  probs <- .ordinal_cells(thresh, d, 1, link_name, K) * as.vector(lam) +
    .ordinal_cells(thresh, 0, 1, link_name, K) * as.vector(1 - lam)
  pmax(probs, 0)
}

.link_cdf_v <- function(x, link_name) {
  switch(link_name,
    probit = .pnorm_v(x),
    logit = 1 / (1 + .exp_v(-x)),
    cloglog = -expm1(-.exp_v(pmin(x, 700))),
    stop("Unknown link: ", link_name)
  )
}

.link_ccdf_v <- function(x, link_name) {
  switch(link_name,
    probit = .pnorm_v(-x),
    logit = 1 / (1 + .exp_v(x)),
    cloglog = .exp_v(-.exp_v(pmin(x, 700))),
    stop("Unknown link: ", link_name)
  )
}

.cell_v <- function(lo, hi, link_name = "probit") {
  Flo <- .link_cdf_v(lo, link_name)
  out <- .link_cdf_v(hi, link_name) - Flo
  up <- which(Flo > 0.5)
  if (length(up)) out[up] <- .link_ccdf_v(lo[up], link_name) - .link_ccdf_v(hi[up], link_name)
  pmax(out, 0)
}

.ordinal_cells <- function(thresh, mu, s, link_name, K) {
  d <- dim(thresh)
  z <- (thresh - as.vector(mu)) / as.vector(s)
  dim(z) <- d
  nb <- d[1] * d[2]
  lo <- c(rep(-Inf, nb), as.vector(z[, , seq_len(K - 1L), drop = FALSE]))
  hi <- c(as.vector(z[, , seq_len(K - 1L), drop = FALSE]), rep(Inf, nb))
  array(.cell_v(lo, hi, link_name), c(d[1], d[2], K))
}

#' VRDP2D probabilities (2D Variable Recollection Dual-Process)
#'
#' Returns S x N x K1 x K2 array of joint response probabilities.
#' Since rho=0 for all VRDP2D distributions, bivariate normal factors into
#' products of univariates, making this the simplest bivariate family.
#' @noRd
# The per-trial loop implementation, the reference for .probs_vrdp2d_vec (options(bayesroc.vrdp2d_joint_loop = TRUE)
# selects it).
.probs_vrdp2d_loop <- function(params, thresh, thresh2, item_type, S, N, K1, K2, model) {
  md <- model$model_data
  dprime <- params[["dprime"]]
  dprime2 <- params[["dprime2"]]
  discrim <- params[["discrim"]]
  lambda <- params[["lambda"]]
  sigma <- if (!is.null(params[["sigma"]])) params[["sigma"]] else matrix(1, S, N)
  sigma2 <- if (!is.null(params[["sigma2"]])) params[["sigma2"]] else matrix(1, S, N)

  has_discrim_B <- isTRUE(md$has_discrim_B)
  discrim_B <- if (has_discrim_B) params[["discrim_B"]] else NULL

  varying <- isTRUE(md$varying_source_criteria)

  probs <- array(0, dim = c(S, N, K1, K2))

  univar_cell_vec <- function(j, K, mu_s, sig_s, thresh_sn) {
    lo <- if (j == 1) -Inf else (thresh_sn[, j - 1] - mu_s) / sig_s
    hi <- if (j == K) Inf else (thresh_sn[, j] - mu_s) / sig_s
    .cell_v(rep_len(lo, length(mu_s)), rep_len(hi, length(mu_s)))
  }

  for (n in seq_len(N)) {
    it <- item_type[n]

    c1_sn <- thresh[, n, , drop = FALSE]
    dim(c1_sn) <- c(S, K1 - 1)

    if (it == 1) {
      mu1_s <- rep(0, S); sig1_s <- rep(1, S)
      mu2_s <- rep(0, S); sig2_s <- rep(1, S)

      for (j in seq_len(K1)) {
        p1_s <- univar_cell_vec(j, K1, mu1_s, sig1_s, c1_sn)
        c2_sn <- if (varying) {
          m <- thresh2[, n, j, , drop = FALSE]; dim(m) <- c(S, K2 - 1); m
        } else {
          m <- thresh2[, n, , drop = FALSE]; dim(m) <- c(S, K2 - 1); m
        }
        for (k in seq_len(K2)) {
          p2_s <- univar_cell_vec(k, K2, mu2_s, sig2_s, c2_sn)
          probs[, n, j, k] <- p1_s * p2_s
        }
      }
    } else {
      d_F_s <- dprime[, n]
      d_R_s <- dprime2[, n]
      R_s <- lambda[, n]
      s_item_s <- sigma[, n]
      s_S_s <- sigma2[, n]
      source_d_s <- if (it == 2) {
        discrim[, n]
      } else {
        if (has_discrim_B) discrim_B[, n] else -discrim[, n]
      }

      for (j in seq_len(K1)) {
        p1_fam_s <- univar_cell_vec(j, K1, d_F_s, s_item_s, c1_sn)
        p1_rec_s <- univar_cell_vec(j, K1, d_F_s + d_R_s, s_item_s, c1_sn)

        c2_sn <- if (varying) {
          m <- thresh2[, n, j, , drop = FALSE]; dim(m) <- c(S, K2 - 1); m
        } else {
          m <- thresh2[, n, , drop = FALSE]; dim(m) <- c(S, K2 - 1); m
        }

        for (k in seq_len(K2)) {
          p2_fam_s <- univar_cell_vec(k, K2, rep(0, S), rep(1, S), c2_sn)
          p2_rec_s <- univar_cell_vec(k, K2, source_d_s, s_S_s, c2_sn)

          probs[, n, j, k] <- (1 - R_s) * p1_fam_s * p2_fam_s +
                                R_s * p1_rec_s * p2_rec_s
        }
      }
    }
  }

  probs <- pmax(probs, 0)
  probs
}

#' Bivariate Cumulative probabilities
#' @noRd
probs_bivariate_cumulative <- function(params, thresh, thresh2, S, N, K1, K2, model) {
  dprime <- params[["dprime"]]
  discrim <- params[["discrim"]]
  rho_vals <- params[["rho"]]

  probs <- array(0, dim = c(S, N, K1, K2))

  for (n in seq_len(N)) {
    mu1_s <- if (!is.null(dprime)) dprime[, n] else rep(0, S)
    mu2_s <- if (!is.null(discrim)) discrim[, n] else rep(0, S)
    rho_s <- if (!is.null(rho_vals)) rho_vals[, n] else rep(0, S)

    for (j in seq_len(K1)) {
      c1_lo_z <- if (j > 1) thresh[, n, j - 1] - mu1_s else rep(-Inf, S)
      c1_hi_z <- if (j < K1) thresh[, n, j] - mu1_s else rep(Inf, S)

      for (k in seq_len(K2)) {
        c2_lo_z <- if (k > 1) thresh2[, n, k - 1] - mu2_s else rep(-Inf, S)
        c2_hi_z <- if (k < K2) thresh2[, n, k] - mu2_s else rep(Inf, S)

        probs[, n, j, k] <- binormal_cdf_r(c1_hi_z, c2_hi_z, rho_s) -
                             binormal_cdf_r(c1_hi_z, c2_lo_z, rho_s) -
                             binormal_cdf_r(c1_lo_z, c2_hi_z, rho_s) +
                             binormal_cdf_r(c1_lo_z, c2_lo_z, rho_s)
      }
    }
  }

  pmax(probs, 0)
}

#' Bivariate SDT probabilities
#'
#' Returns S x N x K1 x K2 array of joint response probabilities.
#' Uses bivariate normal CDF for correlated dimensions.
#' @noRd
probs_bivariate_sdt <- function(params, thresh, thresh2, item_type, S, N, K1, K2, model) {
  md <- model$model_data
  dprime <- params[["dprime"]]
  discrim <- params[["discrim"]]
  rho_A <- params[["rho"]]
  has_dprime_B <- isTRUE(md$has_dprime_B)
  has_discrim_B <- isTRUE(md$has_discrim_B)
  varying <- isTRUE(md$varying_source_criteria)
  bounded <- isTRUE(md$bounded)

  probs <- array(0, dim = c(S, N, K1, K2))
  per <- max(1L, as.integer(2e6 %/% S))
  for (b0 in seq(1L, N, by = per)) {
    cols <- b0:min(N, b0 + per - 1L)
    nc <- length(cols)
    L <- S * nc
    it <- rep(item_type[cols], each = S)
    iA <- which(it == 2L); iB <- which(it == 3L); iold <- which(it != 1L); inew <- which(it == 1L)
    sub <- function(m) as.vector(m[, cols, drop = FALSE])
    opt <- function(nm, alt) if (!is.null(params[[nm]])) sub(params[[nm]]) else alt
    by_type <- function(new, A, B) {
      v <- rep_len(new, L)
      v[iA] <- A[iA]
      v[iB] <- B[iB]
      v
    }
    one <- rep(1, L)
    dA <- sub(dprime)
    mu1 <- by_type(0, dA, if (has_dprime_B) sub(params[["dprime_B"]]) else dA)
    qA <- sub(discrim)
    sgn <- if (isTRUE(md$bounded)) -1 else 1
    mu2 <- by_type(0, qA, if (has_discrim_B) sgn * sub(params[["discrim_B"]]) else -qA)
    s1A <- opt("sigma", one)
    sig1 <- by_type(1, s1A, opt("sigma_B", s1A))
    s2A <- opt("sigma2", one)
    sig2 <- by_type(1, s2A, opt("sigma2_B", s2A))
    rA <- sub(rho_A)
    rho <- by_type(opt("rho_N", 0), rA, if (!is.null(params[["rho_B"]])) sgn * sub(params[["rho_B"]]) else -rA)
    if (bounded) {
      mu2[iold] <- -mu2[iold]
      rho[iold] <- -rho[iold]
    }
    edge2 <- function(j, k) {
      if (k < 1L) return(rep(-Inf, L))
      if (k > K2 - 1L) return(rep(Inf, L))
      if (varying) as.vector(thresh2[, cols, j, k])
      else as.vector(thresh2[, cols, k])
    }
    rect <- function(i, y_lo, y_hi, c_lo, c_hi) {
      .bvn_rect_v((y_lo[i] - mu1[i]) / sig1[i], (y_hi[i] - mu1[i]) / sig1[i],
                  (c_lo[i] - mu2[i]) / sig2[i], (c_hi[i] - mu2[i]) / sig2[i], rho[i])
    }
    familiarity <- function(i, y_lo, y_hi, c_lo, c_hi) {
      if (!bounded) return(rect(i, y_lo, y_hi, c_lo, c_hi))
      .bounded_cell_v(y_lo[i], y_hi[i], c_lo[i], c_hi[i], mu1[i], mu2[i], sig1[i], sig2[i], rho[i])
    }

    for (j in seq_len(K1)) {
      y_lo <- if (j > 1L) as.vector(thresh[, cols, j - 1L]) else rep(-Inf, L)
      y_hi <- if (j < K1) as.vector(thresh[, cols, j]) else rep(Inf, L)
      for (k in seq_len(K2)) {
        c_lo <- edge2(j, k - 1L)
        c_hi <- edge2(j, k)
        p <- numeric(L)
        if (length(inew)) p[inew] <- rect(inew, y_lo, y_hi, c_lo, c_hi)
        if (length(iold)) p[iold] <- familiarity(iold, y_lo, y_hi, c_lo, c_hi)
        probs[, cols, j, k] <- p
      }
    }
  }

  pmax(probs, 0)
}

#' Bivariate Dual-Process probabilities
#'
#' Implements p1 (familiarity) + p2 (item recollection) + p3 (both recollected)
#' with optional bounding.
#' @noRd
probs_bivariate_dp <- function(params, thresh, thresh2, item_type, S, N, K1, K2, model) {
  md <- model$model_data
  dprime <- params[["dprime"]]
  discrim <- params[["discrim"]]
  rho_A <- params[["rho"]]
  has_dprime_B <- isTRUE(md$has_dprime_B)
  has_discrim_B <- isTRUE(md$has_discrim_B)
  varying <- isTRUE(md$varying_source_criteria)
  bounded <- isTRUE(md$bounded)

  probs <- array(0, dim = c(S, N, K1, K2))
  per <- max(1L, as.integer(2e6 %/% S))
  for (b0 in seq(1L, N, by = per)) {
    cols <- b0:min(N, b0 + per - 1L)
    nc <- length(cols)
    L <- S * nc
    it <- rep(item_type[cols], each = S)
    iA <- which(it == 2L); iB <- which(it == 3L); iold <- which(it != 1L); inew <- which(it == 1L)
    sub <- function(m) as.vector(m[, cols, drop = FALSE])
    opt <- function(nm, alt) if (!is.null(params[[nm]])) sub(params[[nm]]) else alt
    by_type <- function(new, A, B) {
      v <- rep_len(new, L)
      v[iA] <- A[iA]
      v[iB] <- B[iB]
      v
    }
    one <- rep(1, L)
    dA <- sub(dprime)
    mu1 <- by_type(0, dA, if (has_dprime_B) sub(params[["dprime_B"]]) else dA)
    qA <- sub(discrim)
    sgn <- if (isTRUE(md$bounded)) -1 else 1
    mu2 <- by_type(0, qA, if (has_discrim_B) sgn * sub(params[["discrim_B"]]) else -qA)
    s1A <- opt("sigma", one)
    sig1 <- by_type(1, s1A, opt("sigma_B", s1A))
    s2A <- opt("sigma2", one)
    sig2 <- by_type(1, s2A, opt("sigma2_B", s2A))
    rA <- sub(rho_A)
    rho <- by_type(opt("rho_N", 0), rA, if (!is.null(params[["rho_B"]])) sgn * sub(params[["rho_B"]]) else -rA)
    if (bounded) {
      mu2[iold] <- -mu2[iold]
      rho[iold] <- -rho[iold]
    }
    edge2 <- function(j, k) {
      if (k < 1L) return(rep(-Inf, L))
      if (k > K2 - 1L) return(rep(Inf, L))
      if (varying) as.vector(thresh2[, cols, j, k])
      else as.vector(thresh2[, cols, k])
    }
    rect <- function(i, y_lo, y_hi, c_lo, c_hi) {
      .bvn_rect_v((y_lo[i] - mu1[i]) / sig1[i], (y_hi[i] - mu1[i]) / sig1[i],
                  (c_lo[i] - mu2[i]) / sig2[i], (c_hi[i] - mu2[i]) / sig2[i], rho[i])
    }
    familiarity <- function(i, y_lo, y_hi, c_lo, c_hi) {
      if (!bounded) return(rect(i, y_lo, y_hi, c_lo, c_hi))
      .bounded_cell_v(y_lo[i], y_hi[i], c_lo[i], c_hi[i], mu1[i], mu2[i], sig1[i], sig2[i], rho[i])
    }

    lam <- by_type(0, sub(params[["lambda"]]), opt("lambda_B", sub(params[["lambda"]])))
    lam2 <- by_type(0, sub(params[["lambda2"]]), opt("lambda2_B", sub(params[["lambda2"]])))
    for (j in seq_len(K1)) {
      y_lo <- if (j > 1L) as.vector(thresh[, cols, j - 1L]) else rep(-Inf, L)
      y_hi <- if (j < K1) as.vector(thresh[, cols, j]) else rep(Inf, L)
      for (k in seq_len(K2)) {
        c_lo <- edge2(j, k - 1L)
        c_hi <- edge2(j, k)
        p <- numeric(L)
        if (length(inew)) p[inew] <- rect(inew, y_lo, y_hi, c_lo, c_hi)
        if (length(iold)) {
          i <- iold
          po <- (1 - lam[i]) * familiarity(i, y_lo, y_hi, c_lo, c_hi)
          if (j == K1) {
            src <- if (bounded) {
              .bounded_marginal_source_v(c_lo[i], c_hi[i], mu1[i], mu2[i], sig1[i], sig2[i], rho[i])
            } else {
              .pnorm_v((c_hi[i] - mu2[i]) / sig2[i]) - .pnorm_v((c_lo[i] - mu2[i]) / sig2[i])
            }
            po <- po + lam[i] * (1 - lam2[i]) * src
            corner <- (it[i] == 2L & k == 1L) | (it[i] == 3L & k == K2)
            po <- po + corner * lam[i] * lam2[i]
          }
          p[i] <- po
        }
        probs[, cols, j, k] <- p
      }
    }
  }

  pmax(probs, 0)
}


probs_vrdp2d <- function(params, thresh, thresh2, item_type, S, N, K1, K2, model) {
  if (isTRUE(getOption("bayesroc.vrdp2d_joint_loop", FALSE)))
    return(.probs_vrdp2d_loop(params, thresh, thresh2, item_type, S, N, K1, K2, model))
  md <- model$model_data
  varying <- isTRUE(md$varying_source_criteria)
  probs <- array(0, dim = c(S, N, K1, K2))
  per <- max(1L, as.integer(2e6 %/% S))
  cell <- function(mu, sig, lo, hi) .cell_v((lo - mu) / sig, (hi - mu) / sig)
  for (b0 in seq(1L, N, by = per)) {
    cols <- b0:min(N, b0 + per - 1L); nc <- length(cols); L <- S * nc
    it <- rep(item_type[cols], each = S); isB <- it == 3L; isnew <- it == 1L
    sub <- function(m) as.vector(m[, cols, drop = FALSE])
    zero <- rep(0, L); one <- rep(1, L)
    dF <- sub(params[["dprime"]]); dR <- dF + sub(params[["dprime2"]]); lam <- sub(params[["lambda"]])
    s1 <- if (!is.null(params[["sigma"]])) sub(params[["sigma"]]) else one
    s2 <- if (!is.null(params[["sigma2"]])) sub(params[["sigma2"]]) else one
    qA <- sub(params[["discrim"]])
    src <- if (isTRUE(md$has_discrim_B)) ifelse(isB, sub(params[["discrim_B"]]), qA) else ifelse(isB, -qA, qA)
    muF1 <- ifelse(isnew, 0, dF); sF1 <- ifelse(isnew, 1, s1)
    muR1 <- ifelse(isnew, 0, dR)
    muR2 <- ifelse(isnew, 0, src); sR2 <- ifelse(isnew, 1, s2)
    lam[isnew] <- 0
    edge <- function(j, k) {
      if (k < 1L) return(rep(-Inf, L))
      if (k > K2 - 1L) return(rep(Inf, L))
      if (varying) as.vector(thresh2[, cols, j, k]) else as.vector(thresh2[, cols, k])
    }
    for (j in seq_len(K1)) {
      y_lo <- if (j > 1L) as.vector(thresh[, cols, j - 1L]) else rep(-Inf, L)
      y_hi <- if (j < K1) as.vector(thresh[, cols, j]) else rep(Inf, L)
      p1F <- cell(muF1, sF1, y_lo, y_hi); p1R <- cell(muR1, sF1, y_lo, y_hi)
      for (k in seq_len(K2)) {
        c_lo <- edge(j, k - 1L); c_hi <- edge(j, k)
        probs[, cols, j, k] <- (1 - lam) * p1F * cell(zero, one, c_lo, c_hi) + lam * p1R * cell(muR2, sR2, c_lo, c_hi)
      }
    }
  }
  pmax(probs, 0)
}

#' CDP (Continuous Dual-Process) probabilities
#'
#' Returns S x N x K array of marginal confidence probabilities (marginalizing over R/K or R/K/G).
#' Also stores R/K(/G) conditional probabilities as attributes for prediction sampling.
#' @noRd
probs_cdp <- function(params, thresh, is_old, S, N, K, model) {
  family <- model$family
  old_levels <- family$old_levels
  J <- family$J
  new_levels <- setdiff(seq_len(K), old_levels)
  n_rkg <- if (!is.null(family$n_rkg)) family$n_rkg else 2L
  if (isTRUE(model$model_data$cdp_rho_shared) && is.null(params[["rho_N"]])) params[["rho_N"]] <- params[["rho"]]

  nat <- .probs_cdp_native(params, thresh, is_old, S, N, K, J, old_levels, new_levels, n_rkg)
  if (!is.null(nat)) return(nat)
  probs <- array(0, dim = c(S, N, K))
  p_remember <- array(0, dim = c(S, N, K))
  p_know <- if (n_rkg == 3) array(0, dim = c(S, N, K)) else NULL

  band <- function(mu_R, sigma_R, mu_F, sigma_F, c_R, c_K, tau_lo, tau_hi, corr) {
    if (n_rkg == 3) return(compute_rkg_probs_r(mu_R, sigma_R, mu_F, sigma_F, c_R, c_K, tau_lo, tau_hi, corr))
    sigma_M <- sqrt(sigma_R^2 + sigma_F^2 + 2 * corr * sigma_R * sigma_F)
    rho_RM <- (sigma_R + corr * sigma_F) / sigma_M
    z_cR <- (c_R - mu_R) / sigma_R
    z_lo <- (tau_lo - mu_R - mu_F) / sigma_M
    z_hi <- (tau_hi - mu_R - mu_F) / sigma_M
    c(binormal_strip_upper_r(z_cR, z_lo, z_hi, rho_RM),
      binormal_strip_lower_r(z_cR, z_lo, z_hi, rho_RM))
  }

  for (n in seq_len(N)) {
    target <- is_old[n] == 1
    for (s in seq_len(S)) {
      mu_R <- if (target) params[["dprime"]][s, n] else 0
      mu_F <- if (target && !is.null(params[["dprime2"]])) params[["dprime2"]][s, n] else 0
      sigma_R <- if (target && !is.null(params[["sigma"]])) params[["sigma"]][s, n] else 1
      sigma_F <- if (target && !is.null(params[["sigma2"]])) params[["sigma2"]][s, n] else 1
      c_R <- if (!is.null(params[["rec_crit"]])) params[["rec_crit"]][s, n] else 1.5
      c_K <- if (!is.null(params[["know_crit"]])) params[["know_crit"]][s, n] else 1.0
      corr <- if (target) {
        if (!is.null(params[["rho"]])) params[["rho"]][s, n] else 0
      } else if (!is.null(params[["rho_N"]])) params[["rho_N"]][s, n] else 0
      tau <- c(-Inf, thresh[s, n, seq_len(J)], Inf)
      for (b in 0:J) {
        cell <- band(mu_R, sigma_R, mu_F, sigma_F, c_R, c_K, tau[b + 1L], tau[b + 2L], corr)
        total <- sum(cell)
        ks <- if (b == 0L) new_levels else old_levels[b]
        probs[s, n, ks] <- total / length(ks)
        p_remember[s, n, ks] <- cell[1] / total
        if (n_rkg == 3) p_know[s, n, ks] <- cell[2] / total
      }
    }
  }

  probs <- pmax(probs, 0)
  probs <- probs / as.vector(apply(probs, c(1, 2), sum))
  attr(probs, "p_remember") <- p_remember
  if (n_rkg == 3) attr(probs, "p_know") <- p_know
  probs
}

# probs_cdp through the native band cells (the likelihood's own one-dimensional integrals), all draws and
# trials at once; NULL without the library. options(bayesroc.cdp_native = FALSE) forces the R loop.
.probs_cdp_native <- function(params, thresh, is_old, S, N, K, J, old_levels, new_levels, n_rkg) {
  if (!isTRUE(getOption("bayesroc.cdp_native", TRUE)) || !.bvn_native_ready() || is.null(.bvn_native$sym_cdp)) return(NULL)
  target <- matrix(rep(is_old == 1, each = S), S, N)
  pick <- function(nm, alt) if (!is.null(params[[nm]])) ifelse(target, params[[nm]], alt) else matrix(alt, S, N)
  mu_R <- pick("dprime", 0); mu_F <- pick("dprime2", 0)
  sigma_R <- pick("sigma", 1); sigma_F <- pick("sigma2", 1)
  c_R <- if (!is.null(params[["rec_crit"]])) params[["rec_crit"]] else matrix(1.5, S, N)
  c_K <- if (!is.null(params[["know_crit"]])) params[["know_crit"]] else matrix(1.0, S, N)
  corr <- ifelse(target, if (!is.null(params[["rho"]])) params[["rho"]] else 0,
                 if (!is.null(params[["rho_N"]])) params[["rho_N"]] else 0)
  res <- .C(.bvn_native$sym_cdp, as.double(mu_R), as.double(sigma_R), as.double(mu_F), as.double(sigma_F),
            as.double(c_R), as.double(c_K), as.double(corr), as.double(thresh[, , seq_len(J), drop = FALSE]),
            as.integer(c(S, N, J, K, n_rkg, .native_threads(), length(new_levels))),
            as.integer(old_levels), as.integer(new_levels),
            probs = double(S * N * K), prem = double(S * N * K), pknow = double(S * N * K), NAOK = TRUE)
  probs <- array(res$probs, c(S, N, K))
  probs <- pmax(probs, 0)
  probs <- probs / as.vector(rowSums(probs, dims = 2L))
  attr(probs, "p_remember") <- array(res$prem, c(S, N, K))
  if (n_rkg == 3) attr(probs, "p_know") <- array(res$pknow, c(S, N, K))
  probs
}

# =============================================================================
# Newdata Support
# =============================================================================

#' Rebuild design matrices for new data
#' @noRd
rebuild_design_matrices <- function(model, newdata) {
  md <- model$model_data
  parsed <- model$parsed
  family <- model$family
  family_name <- family$family
  is_cumulative <- family_name == "cumulative"
  is_source_mixture <- family_name == "source_mixture"
  is_bivariate <- family_name %in% c("bivariate_sdt", "bivariate_dp", "bivariate_cumulative")
  is_vrdp2d <- family_name == "vrdp2d"
  is_cdp <- family_name == "cdp"
  train_data <- model$data

  required_cols <- character(0)
  if (!is_cumulative && !is_bivariate && !is_vrdp2d) {
    is_old_var <- parsed$response$is_old
    if (!is.null(is_old_var)) required_cols <- c(required_cols, is_old_var)
  }
  if (is_source_mixture) {
    source_var <- parsed$response$source
    if (!is.null(source_var)) required_cols <- c(required_cols, source_var)
  }
  if (is_bivariate || is_vrdp2d) {
    item_type_var <- parsed$response$item_type
    if (!is.null(item_type_var)) required_cols <- c(required_cols, item_type_var)
    resp2_var <- parsed$response2
    if (!is.null(resp2_var)) required_cols <- c(required_cols, resp2_var)
  }
  if (is_cdp) {
    rk_var <- parsed$response$rk
    if (!is.null(rk_var)) required_cols <- c(required_cols, rk_var)
  }

  if (family_name == "bivariate_cumulative" && is.null(newdata$.bivar_cum_item_type)) {
    newdata$.bivar_cum_item_type <- factor(rep("A", nrow(newdata)), levels = c("new", "A"))
  }
  newdata <- .add_group_columns(newdata, md$group_defs %||% attr(model$data, "bayesroc_group_defs"))
  if ((isTRUE(md$has_criterion2_scale) || !is.null(md$smooth_data[["criterion2"]])) &&
      isTRUE(md$varying_source_criteria) && is.null(newdata$.c2bin)) {
    lab <- .vbin_labels(md)
    y1 <- newdata[[parsed$response$response]]
    newdata$.c2bin <- factor(if (!is.null(y1)) lab[.vbin_of(md)[as.integer(y1)]] else rep(lab[1], nrow(newdata)),
                             levels = lab)
  }
  missing_cols <- setdiff(required_cols, names(newdata))
  if (length(missing_cols) > 0) {
    stop("newdata is missing required columns: ", paste(missing_cols, collapse = ", "))
  }

  N <- nrow(newdata)
  src <- list(N = N)
  src$.frame <- newdata

  if (family_name == "mafc") {
    if (!is.null(family$alternatives) && !family$alternatives %in% names(newdata)) {
      stop("newdata is missing required columns: ", family$alternatives, call. = FALSE)
    }
    src <- .mafc_newdata(model, newdata, src)
  } else if (!is_cumulative && !is_bivariate && !is_vrdp2d && !is.null(parsed$response$is_old)) {
    src$is_old <- .code_is_old(newdata[[parsed$response$is_old]], parsed$response$is_old, allow_centered = TRUE)
  } else if (is_cumulative) {
    src$is_old <- rep(1L, N)
  }

  if (is_source_mixture && !is.null(parsed$response$source)) {
    src$source <- .code_like_training(newdata[[parsed$response$source]], train_data[[parsed$response$source]],
                                      parsed$response$source, .code_source)
  }

  if ((is_bivariate || is_vrdp2d) && !is.null(parsed$response$item_type)) {
    it <- parsed$response$item_type
    src$item_type <- if (startsWith(it, ".")) as.integer(newdata[[it]])
                     else .code_like_training(newdata[[it]], train_data[[it]], it, .code_item_type)
    if (!is.null(parsed$response2)) {
      src$y2 <- as.integer(newdata[[parsed$response2]])
    }
  }

  if (is_cdp && !is.null(parsed$response$rk)) {
    rk_values <- newdata[[parsed$response$rk]]
    if (is.character(rk_values) || is.factor(rk_values)) {
      rk_values <- as.character(rk_values)
      rk_numeric <- ifelse(toupper(rk_values) %in% c("R", "REMEMBER", "1"), 1L,
                            ifelse(toupper(rk_values) %in% c("K", "KNOW", "2"), 2L,
                                   ifelse(toupper(rk_values) %in% c("G", "GUESS", "3"), 3L, NA_integer_)))
    } else {
      rk_numeric <- as.integer(rk_values)
    }
    rk_numeric[is.na(rk_numeric)] <- 0L
    src$rk <- rk_numeric
  }

  encoding_vars <- parsed$encoding_vars
  is_old_var <- if (!is_cumulative) parsed$response$is_old else ".cumulative_is_old"
  if (is_cumulative) newdata$.cumulative_is_old <- 1L

  if (isTRUE(md$has_dprime_fixed)) {
    src$X_dprime <- rebuild_one_design_matrix(
      md$dprime_fixed$formula, newdata, encoding_vars, is_old_var,
      ref_colnames = drop_smooth_null_space_names(md$dprime_fixed$coef_names),
      conditional_info = md$dprime_fixed$conditional_info,
      train_data = train_data
    )
  }

  crit_ref_names <- drop_smooth_null_space_names(md$criterion$coef_names)
  if (is.null(md$criterion)) {
  } else if (md$criterion$is_intercept_only ||
      identical(crit_ref_names, "(Intercept)")) {
    src$X_criterion <- matrix(1, nrow = N, ncol = 1)
    colnames(src$X_criterion) <- "(Intercept)"
  } else if (!is.null(md$criterion$conditional_info)) {
    src$X_criterion <- rebuild_one_design_matrix(
      md$criterion$fixed_formula, newdata, NULL, NULL,
      ref_colnames = crit_ref_names,
      conditional_info = md$criterion$conditional_info,
      train_data = train_data
    )
  } else if (!is.null(md$criterion$fixed_formula)) {
    src$X_criterion <- rebuild_one_design_matrix(
      md$criterion$fixed_formula, newdata, NULL, NULL,
      ref_colnames = crit_ref_names,
      train_data = train_data
    )
  } else {
    src$X_criterion <- model.matrix(
      as.formula(paste("~", paste(crit_ref_names, collapse = " + "))),
      data = newdata
    )
    src$X_criterion <- align_matrix_columns(src$X_criterion, crit_ref_names)
  }

  param_names <- c("sigma", "lambda", "dprime2", "sigma2",
                    "dprime_B", "lambda_B",
                    "dprime_L", "sigma_L", "lambda_L",
                    "discrim", "discrim_B", "sigma_B", "sigma2_B",
                    "rho", "rho_B", "rho_N", "lambda2", "lambda2_B", "mratio", "mratio_N", "sigma_meta", "rec_crit", "know_crit",
                    "criterion_scale", "criterion2_scale")
  for (pname in param_names) {
    if (isTRUE(md[[paste0("has_", pname)]])) {
      fixed_info <- md[[paste0(pname, "_fixed")]]
      src[[paste0("X_", pname)]] <- rebuild_one_design_matrix(
        fixed_info$formula, newdata, encoding_vars, is_old_var,
        ref_colnames = drop_smooth_null_space_names(fixed_info$coef_names),
        conditional_info = fixed_info$conditional_info,
        train_data = train_data
      )
    }
  }

  all_re_params <- c("dprime", "sigma", "lambda", "dprime2", "sigma2",
                      "dprime_B", "lambda_B",
                      "dprime_L", "sigma_L", "lambda_L", "rec_crit", "know_crit",
                      "discrim", "discrim_B", "sigma_B", "sigma2_B",
                      "rho", "rho_B", "rho_N", "lambda2", "lambda2_B", "mratio", "mratio_N", "sigma_meta", "criterion_scale", "criterion2_scale")
  for (pname in all_re_params) {
    re_list <- md[[paste0(pname, "_random")]]
    if (is.null(re_list)) next

    new_re_list <- list()
    for (group in names(re_list)) {
      re <- re_list[[group]]
      new_re <- re

      remap <- remap_group_indices(newdata, group, model$data, re)
      new_re$group_idx <- remap$group_idx
      new_re$new_level_idx <- as.integer(remap$new_level_idx)
      if (!is.null(remap$warning)) warning(remap$warning)

      new_re <- .rebuild_re_design(new_re, newdata, encoding_vars, is_old_var)

      new_re_list[[group]] <- new_re
    }
    src[[paste0(pname, "_random")]] <- new_re_list
  }

  if (!is.null(md$criterion$random)) {
    new_crit_re <- list()
    for (group in names(md$criterion$random)) {
      re <- md$criterion$random[[group]]
      new_re <- re

      remap <- remap_group_indices(newdata, group, model$data, re)
      new_re$group_idx <- remap$group_idx
      new_re$new_level_idx <- as.integer(remap$new_level_idx)
      if (!is.null(remap$warning)) warning(remap$warning)

      new_re <- .rebuild_re_design(new_re, newdata, NULL, NULL)

      new_crit_re[[group]] <- new_re
    }
    src$criterion_random <- new_crit_re
  }

  if (isTRUE(md$has_criterion2) && !is.null(md$criterion2)) {
    crit2_ref_names <- drop_smooth_null_space_names(md$criterion2$coef_names)
    if (md$criterion2$is_intercept_only ||
        identical(crit2_ref_names, "(Intercept)")) {
      src$X_criterion2 <- matrix(1, nrow = N, ncol = 1)
      colnames(src$X_criterion2) <- "(Intercept)"
    } else if (!is.null(md$criterion2$conditional_info)) {
      src$X_criterion2 <- rebuild_one_design_matrix(
        md$criterion2$fixed_formula, newdata, NULL, NULL,
        ref_colnames = crit2_ref_names,
        conditional_info = md$criterion2$conditional_info,
        train_data = train_data
      )
    } else if (!is.null(md$criterion2$fixed_formula)) {
      src$X_criterion2 <- rebuild_one_design_matrix(
        md$criterion2$fixed_formula, newdata, NULL, NULL,
        ref_colnames = crit2_ref_names,
        train_data = train_data
      )
    } else {
      src$X_criterion2 <- matrix(1, nrow = N, ncol = 1)
      colnames(src$X_criterion2) <- "(Intercept)"
    }

    if (!is.null(md$criterion2$random)) {
      new_crit2_re <- list()
      for (group in names(md$criterion2$random)) {
        re <- md$criterion2$random[[group]]
        new_re <- re
        remap <- remap_group_indices(newdata, group, model$data, re)
        new_re$group_idx <- remap$group_idx
        new_re$new_level_idx <- as.integer(remap$new_level_idx)
        if (!is.null(remap$warning)) warning(remap$warning)

        new_re <- .rebuild_re_design(new_re, newdata, NULL, NULL)
new_crit2_re[[group]] <- new_re
      }
      src$criterion2_random <- new_crit2_re
    }
  }

  src$cross_cor <- md$cross_cor

  if (!is.null(md$smooth_data)) {
    new_smooth_data <- list()
    for (pname in names(md$smooth_data)) {
      new_smooth_data[[pname]] <- list()
      for (i in seq_along(md$smooth_data[[pname]])) {
        sm <- md$smooth_data[[pname]][[i]]
        new_sm <- sm
        new_sm$components <- list()
        for (j in seq_along(sm$components)) {
          comp <- sm$components[[j]]
          pred <- build_smooth_prediction_basis(comp, newdata)
          new_comp <- comp
          new_comp$Xs <- pred$Xs
          new_comp$Zs_list <- pred$Zs_list
          new_sm$components[[j]] <- new_comp
        }
        if (!is.null(sm$by) && !is.na(sm$by) && sm$by %in% names(newdata) &&
            length(sm$components) > 0) {
          nd_by <- as.character(newdata[[sm$by]])
          for (nl in setdiff(unique(nd_by), sm$bylevels)) {
            ref <- sm$components[[1]]
            syn <- ref
            syn$new_level <- TRUE
            syn$Zs_list <- build_new_bylevel_basis(ref, newdata, sm$by, nl)
            syn$Xs <- matrix(0, nrow = nrow(newdata), ncol = 0)
            syn$Xs_colnames <- character(0)
            new_sm$components[[length(new_sm$components) + 1L]] <- syn
          }
        }
        new_smooth_data[[pname]][[i]] <- new_sm
      }

      x_name <- paste0("X_", pname)
      if (!is.null(src[[x_name]])) {
        for (sm in new_smooth_data[[pname]]) {
          for (comp in sm$components) {
            if (NCOL(comp$Xs) > 0) {
              src[[x_name]] <- cbind(src[[x_name]], comp$Xs)
            }
          }
        }
      }
    }
    src$smooth_data <- new_smooth_data
  }

  src
}

#' Align matrix columns to match a reference set of column names
#'
#' Ensures the output matrix has exactly the columns in ref_colnames, in order.
#' Missing columns are filled with zeros; extra columns are dropped.
#' @noRd
align_matrix_columns <- function(X, ref_colnames) {
  current_cols <- colnames(X)
  if (identical(current_cols, ref_colnames)) return(X)

  N <- nrow(X)
  X_aligned <- matrix(0, nrow = N, ncol = length(ref_colnames))
  colnames(X_aligned) <- ref_colnames

  shared <- intersect(ref_colnames, current_cols)
  if (length(shared) > 0) {
    X_aligned[, shared] <- X[, shared, drop = FALSE]
  }

  X_aligned
}

#' Safely drop unused factor levels for encoding variables in old-item data.
#' Only drops levels if the result still has 2+ levels, avoiding the
#' "contrasts can be applied only to factors with 2 or more levels" error
#' that occurs when non-focal variables are held at a single value
#' (e.g., in prediction grids).
#' @noRd
safe_droplevels_old <- function(old_data, formula_vars, encoding_vars = NULL) {
  for (v in formula_vars) {
    if (v %in% names(old_data) && is.factor(old_data[[v]])) {
      if (!is.null(encoding_vars) && v %in% encoding_vars) {
        dropped <- droplevels(old_data[[v]])
        if (nlevels(dropped) >= 2) {
          old_data[[v]] <- dropped
        }
      }
    }
  }
  old_data
}

#' Rebuild a single fixed effects design matrix for newdata
#' @param ref_colnames Column names from the training design matrix. If provided,
#'   output is aligned to match (missing cols zeroed, extra cols dropped).
#' @noRd
rebuild_one_design_matrix <- function(formula, newdata, encoding_vars, is_old_var,
                                       ref_colnames = NULL, conditional_info = NULL,
                                       train_data = NULL) {
  if (!is.null(conditional_info)) {
    N <- nrow(newdata)
    blocks <- list()

    for (ci in conditional_info) {
      X_cond <- rebuild_conditional_columns(ci, newdata)
      blocks[[length(blocks) + 1]] <- X_cond
    }

    if (length(blocks) == 1) {
      X <- blocks[[1]]
    } else {
      X <- do.call(cbind, blocks)
    }

    if (!is.null(ref_colnames)) {
      cond_cols <- colnames(X)
      missing_cols <- setdiff(ref_colnames, cond_cols)
      if (length(missing_cols) > 0) {
        if (!is.null(formula) && inherits(formula, "formula")) {
          X_reg <- tryCatch({
            f <- as.formula(paste(deparse(formula), collapse = " "))
            model.matrix(f, data = newdata)
          }, error = function(e) NULL)
          if (!is.null(X_reg)) {
            X <- cbind(X_reg[, intersect(colnames(X_reg), missing_cols), drop = FALSE], X)
          }
        }
      }
      X <- align_matrix_columns(X, ref_colnames)
    }
    return(X)
  }

  if (is.null(formula) || !inherits(formula, "formula")) return(NULL)

  formula <- as.formula(paste(deparse(formula), collapse = " "))

  formula_vars <- all.vars(formula)
  has_encoding <- !is.null(encoding_vars) && any(encoding_vars %in% formula_vars)

  if (has_encoding && !is.null(is_old_var)) {
    old_idx <- which(newdata[[is_old_var]] == 1)
    old_data <- newdata[old_idx, , drop = FALSE]
    old_data <- safe_droplevels_old(old_data, formula_vars, encoding_vars)
    if (!is.null(train_data)) {
      for (v in formula_vars) {
        if (v %in% names(old_data) && v %in% names(train_data)) {
          col <- old_data[[v]]
          tcol <- train_data[[v]]
          if (is.character(col) || (is.factor(col) && nlevels(col) < 2)) {
            if (is.factor(tcol)) {
              old_data[[v]] <- factor(old_data[[v]], levels = levels(tcol))
            } else if (is.character(tcol)) {
              old_data[[v]] <- factor(old_data[[v]], levels = sort(unique(tcol)))
            }
          }
        }
      }
    }
    X_old <- model.matrix(formula, data = old_data)
    N <- nrow(newdata)
    X <- matrix(0, nrow = N, ncol = ncol(X_old))
    colnames(X) <- colnames(X_old)
    X[old_idx, ] <- X_old
  } else {
    if (!is.null(train_data)) {
      for (v in formula_vars) {
        if (v %in% names(newdata) && v %in% names(train_data)) {
          col <- newdata[[v]]
          tcol <- train_data[[v]]
          needs_fix <- is.character(col) ||
                       (is.factor(col) && nlevels(col) < 2)
          if (needs_fix) {
            if (is.factor(tcol)) {
              newdata[[v]] <- factor(newdata[[v]], levels = levels(tcol))
            } else if (is.character(tcol)) {
              newdata[[v]] <- factor(newdata[[v]], levels = sort(unique(tcol)))
            }
          }
        }
      }
    }
    X <- model.matrix(formula, data = newdata)
  }

  if (!is.null(ref_colnames)) {
    X <- align_matrix_columns(X, ref_colnames)
  }

  X
}

#' Remap group indices from training data levels to newdata
#' @noRd
remap_group_indices <- function(newdata, group, train_data, re) {
  if (!group %in% names(newdata)) {
    stop("newdata is missing grouping variable: ", group)
  }

  if (is.factor(train_data[[group]])) {
    train_levels <- levels(train_data[[group]])
  } else {
    train_levels <- levels(as.factor(train_data[[group]]))
  }

  new_vals <- as.character(newdata[[group]])
  new_in_train <- new_vals %in% train_levels

  group_idx <- integer(length(new_vals))
  for (i in seq_along(new_vals)) {
    if (new_in_train[i]) {
      group_idx[i] <- match(new_vals[i], train_levels)
    } else {
      group_idx[i] <- 0L
    }
  }

  warn_msg <- NULL
  new_levels <- unique(new_vals[!new_in_train])
  if (length(new_levels) > 0) {
    warn_msg <- paste0("New levels in grouping variable '", group,
                        "' not in training data: ",
                        paste(new_levels, collapse = ", "),
                        ". Random effects set to 0 for these levels.")
  }

  list(group_idx = group_idx, warning = warn_msg,
       new_level_idx = ifelse(new_in_train, 0L, match(new_vals, new_levels)))
}

#' Rebuild term index for newdata
#' @noRd
rebuild_term_idx <- function(re, newdata, encoding_vars, is_old_var) {
  if (is.null(re$level_names) || length(re$level_names) == 0) {
    return(rep(0L, nrow(newdata)))
  }

  re_formula <- re$re_formula
  if (is.null(re_formula)) {
    return(rep(1L, nrow(newdata)))
  }

  is_encoding <- isTRUE(re$is_encoding)
  if (is_encoding && !is.null(is_old_var)) {
    old_idx <- which(newdata[[is_old_var]] == 1)
    old_data <- newdata[old_idx, , drop = FALSE]
    old_data <- safe_droplevels_old(old_data, all.vars(re_formula), all.vars(re_formula))
    Z_old <- tryCatch(model.matrix(re_formula, data = old_data), error = function(e) NULL)
    if (is.null(Z_old)) return(rep(0L, nrow(newdata)))

    Z_old <- align_matrix_columns(Z_old, re$level_names)

    Z <- matrix(0, nrow = nrow(newdata), ncol = ncol(Z_old))
    Z[old_idx, ] <- Z_old
  } else {
    Z <- tryCatch(model.matrix(re_formula, data = newdata), error = function(e) NULL)
    if (is.null(Z)) return(rep(0L, nrow(newdata)))

    Z <- align_matrix_columns(Z, re$level_names)
  }

  build_term_index(Z)
}

#' Rebuild Z matrix for newdata
#' @noRd
rebuild_z_matrix <- function(re, newdata, encoding_vars, is_old_var) {
  re_formula <- re$re_formula
  if (is.null(re_formula)) return(NULL)

  is_encoding <- isTRUE(re$is_encoding)
  if (is_encoding && !is.null(is_old_var)) {
    old_idx <- which(newdata[[is_old_var]] == 1)
    old_data <- newdata[old_idx, , drop = FALSE]
    old_data <- safe_droplevels_old(old_data, all.vars(re_formula), all.vars(re_formula))
    Z_old <- model.matrix(re_formula, data = old_data)
    Z_old <- align_matrix_columns(Z_old, re$level_names)
    Z <- matrix(0, nrow = nrow(newdata), ncol = ncol(Z_old))
    Z[old_idx, ] <- Z_old
  } else {
    Z <- model.matrix(re_formula, data = newdata)
    Z <- align_matrix_columns(Z, re$level_names)
  }

  Z
}

.seed_scope <- function(seed) {
  if (is.null(seed)) return(function() invisible(NULL))
  old <- if (exists(".Random.seed", envir = .GlobalEnv)) get(".Random.seed", envir = .GlobalEnv) else NULL
  set.seed(seed)
  function() {
    if (is.null(old)) {
      if (exists(".Random.seed", envir = .GlobalEnv)) rm(".Random.seed", envir = .GlobalEnv)
    } else {
      assign(".Random.seed", old, envir = .GlobalEnv)
    }
    invisible(NULL)
  }
}

.crit2_varying_u <- function(draws, re, var_base, base_col, S, N, u_new, new_pos) {
  R <- re$n_re_terms %||% 1L
  W <- if (isTRUE(re$use_z_matrix) && !is.null(re$Z)) {
    as.matrix(re$Z)
  } else if (!is.null(re$term_idx)) {
    outer(re$term_idx, seq_len(R), "==") * 1
  } else {
    matrix(1, N, 1L)
  }
  g_idx <- re$group_idx
  ok <- which(g_idx > 0 & g_idx <= re$n_groups)
  nw <- which(new_pos > 0)
  out <- matrix(0, S, N)
  for (r in seq_len(R)) {
    col <- (base_col - 1L) * R + r
    nm <- if (re$dim == 1L) paste0(var_base, "[", seq_len(re$n_groups), "]")
          else paste0(var_base, "[", seq_len(re$n_groups), ",", col, "]")
    u <- draws[, nm, drop = FALSE]
    if (length(ok)) out[, ok] <- out[, ok] + sweep(u[, g_idx[ok], drop = FALSE], 2, W[ok, r], "*")
    if (!is.null(u_new) && length(nw)) {
      un <- matrix(u_new[, new_pos[nw], col], nrow = S)
      out[, nw] <- out[, nw] + sweep(un, 2, W[nw, r], "*")
    }
  }
  out
}

.rebuild_re_design <- function(re, newdata, encoding_vars, is_old_var) {
  if (!is.null(re$part_structs)) {
    parts <- lapply(re$part_structs, .rebuild_re_design, newdata = newdata,
                    encoding_vars = encoding_vars, is_old_var = is_old_var)
    Z <- do.call(cbind, lapply(parts, .re_part_Z, N = nrow(newdata)))
    re$Z <- Z
    re$use_z_matrix <- TRUE
    re$term_idx <- NULL
    return(re)
  }
  if (!is.null(re$conditional_info)) {
    cond_result <- build_conditional_columns(
      term = re$conditional_info$term,
      where_str = re$conditional_info$where_str,
      data = newdata,
      interaction_partners = re$conditional_info$interaction_partners
    )
    Z_new <- align_matrix_columns(cond_result$X, re$conditional_info$col_names)
    if (!isTRUE(re$use_z_matrix) && check_simple_categorical(Z_new)) {
      re$term_idx <- build_term_index(Z_new)
      re$Z <- NULL
      re$use_z_matrix <- FALSE
    } else {
      re$term_idx <- NULL
      re$Z <- Z_new
      re$use_z_matrix <- TRUE
    }
  } else if (!is.null(re$term_idx) && !isTRUE(re$use_z_matrix)) {
    re$term_idx <- rebuild_term_idx(re, newdata, encoding_vars, is_old_var)
  } else if (isTRUE(re$use_z_matrix)) {
    re$Z <- rebuild_z_matrix(re, newdata, encoding_vars, is_old_var)
  }
  re
}

.new_level_pos <- function(re, N) {
  if (!is.null(re$new_level_idx)) return(as.integer(re$new_level_idx))
  pos <- integer(N)
  w <- which(re$group_idx == 0L)
  pos[w] <- seq_along(w)
  pos
}

.draw_new_u <- function(draws, re, pname, stan_pname, group, n_new, S, md, data_src) {
  if (!is.null(re$cor_id)) {
    return(.sample_new_u_cross(draws, md, re$cor_id, group, pname, n_new, S,
                               data_src$.new_u_cache %||% new.env(parent = emptyenv())))
  }
  sample_new_u_correlated(draws, stan_pname, group, dim = re$dim, n_new = n_new, S = S,
                          correlated = isTRUE(re$correlated) && re$dim > 1)
}

.sample_new_u_cross <- function(draws, md, cor_id, group, member_param, n_new, S, cache) {
  cc <- md$cross_cor[[cor_id]]
  dims <- vapply(cc$members, function(m) as.integer(m$dim), integer(1))
  D <- sum(dims)
  key <- paste(cor_id, group, n_new, sep = "|")
  joint <- cache[[key]]
  if (is.null(joint)) {
    base <- paste0("cross_", cor_id, "_", group)
    sig_names <- paste0("sigma_", base, "[", seq_len(D), "]")
    L_names <- paste0("L_corr_", base, "[", rep(seq_len(D), each = D), ",", rep(seq_len(D), times = D), "]")
    miss <- setdiff(c(sig_names, L_names), colnames(draws))
    if (length(miss)) {
      stop("allow_new_levels: cross-parameter group '", cor_id, "' is missing ", miss[1],
           " in the fit draws.", call. = FALSE)
    }
    sig <- as.matrix(draws[, sig_names, drop = FALSE])
    L_arr <- array(as.numeric(draws[, L_names, drop = FALSE]), dim = c(S, D, D))
    joint <- array(NA_real_, dim = c(S, n_new, D))
    for (s in seq_len(S)) {
      chol_s <- diag(as.numeric(sig[s, ]), nrow = D) %*% t(matrix(L_arr[s, , ], D, D))
      joint[s, , ] <- matrix(rnorm(n_new * D), n_new, D) %*% t(chol_s)
    }
    assign(key, joint, envir = cache)
  }
  off <- 0L
  for (i in seq_along(cc$members)) {
    if (identical(cc$members[[i]]$param, member_param)) {
      return(joint[, , off + seq_len(dims[i]), drop = FALSE])
    }
    off <- off + dims[i]
  }
  stop("allow_new_levels: parameter '", member_param, "' is not a member of cross-parameter group '",
       cor_id, "'.", call. = FALSE)
}

.crit2_cov_cols <- function(md) {
  nm <- md$criterion2$coef_names
  match(md$criterion2$cov_names %||% setdiff(nm, "(Intercept)"), nm)
}

.vcell_rows <- function(md, X) {
  cl <- md$criterion2$vcell_names
  if (length(cl) <= 1L) return(rep(1L, nrow(X)))
  loc <- setdiff(cl, "(none)")
  L <- X[, loc, drop = FALSE]
  of <- max.col(L, ties.method = "first")
  of[rowSums(L) == 0] <- match("(none)", cl)
  as.integer(of)
}

# =============================================================================
# m-AFC predictions
# =============================================================================

.rebuild_vec_design <- function(spec, newdata, train_data) {
  N <- nrow(newdata)
  ref <- drop_smooth_null_space_names(spec$coef_names)
  if (isTRUE(spec$is_intercept_only) || identical(ref, "(Intercept)")) {
    X <- matrix(1, nrow = N, ncol = 1)
    colnames(X) <- "(Intercept)"
    return(X)
  }
  if (!is.null(spec$fixed_formula)) {
    return(rebuild_one_design_matrix(spec$fixed_formula, newdata, NULL, NULL, ref_colnames = ref,
                                     conditional_info = spec$conditional_info, train_data = train_data))
  }
  align_matrix_columns(model.matrix(as.formula(paste("~", paste(ref, collapse = " + "))), data = newdata), ref)
}

.mafc_newdata <- function(model, newdata, src) {
  md <- model$model_data
  parsed <- model$parsed
  alt <- model$family$alternatives
  M <- md$K
  src$target_pos <- .code_position(newdata[[parsed$response$is_old]], parsed$response$is_old)
  src$m_alt <- if (is.null(alt)) rep(M, nrow(newdata)) else .code_position(newdata[[alt]], alt)
  bad <- which(src$target_pos > src$m_alt | src$m_alt > M)
  if (length(bad)) {
    stop("newdata row ", bad[1], ": target position ", src$target_pos[bad[1]], " with ", src$m_alt[bad[1]],
         " alternatives does not fit the model's ", M, " positions.", call. = FALSE)
  }
  if (!is.null(md$bias)) {
    src$X_bias <- .rebuild_vec_design(md$bias, newdata, model$data)
    if (!is.null(md$bias$random)) {
      src$bias_random <- lapply(stats::setNames(names(md$bias$random), names(md$bias$random)), function(group) {
        re <- md$bias$random[[group]]
        remap <- remap_group_indices(newdata, group, model$data, re)
        re$group_idx <- remap$group_idx
        re$new_level_idx <- as.integer(remap$new_level_idx)
        if (!is.null(remap$warning)) warning(remap$warning)
        .rebuild_re_design(re, newdata, NULL, NULL)
      })
    }
  }
  src
}

#' Position biases for all draws: [S, N, M], the last position 0
#' @noRd
compute_bias <- function(draws, model, data_src, re_formula, S, N, M) {
  md <- model$model_data
  out <- array(0, dim = c(S, N, M))
  if (is.null(md$bias)) return(out)
  P <- md$bias$n_coef
  X <- data_src$X_bias
  include_re <- !identical(re_formula, NA)
  rand <- data_src$bias_random
  new_u <- if (include_re && !is.null(rand) && isTRUE(data_src$.allow_new_levels)) {
    sample_criterion_new_u(draws, rand, "bias", S, md, data_src, "bias")
  } else NULL
  for (k in seq_len(M - 1L)) {
    beta_k <- draws[, paste0("beta_bias[", k, ",", seq_len(P), "]"), drop = FALSE]
    b <- X %*% t(beta_k)
    if (include_re && !is.null(rand)) {
      b <- b + compute_criterion_re(draws, rand, data_src, "bias", S, N, thresh_k = k, md = md, new_u_cache = new_u)
    }
    out[, , k] <- t(b)
  }
  out
}

#' m-AFC choice probabilities [S, N, M]
#' @noRd
probs_mafc <- function(params, bias, ds, S, N, M) {
  d <- as.vector(params[["dprime"]])
  sg <- if (!is.null(params[["sigma"]])) as.vector(params[["sigma"]]) else rep(1, S * N)
  m <- rep(as.integer(ds$m_alt), each = S)
  tgt <- rep(as.integer(ds$target_pos), each = S)
  b <- matrix(bias, S * N, M)
  p <- if (.bvn_native_ready() && !is.null(.bvn_native$sym_mafc)) {
    .C(.bvn_native$sym_mafc, as.double(d), as.double(sg), as.double(b), as.integer(m), as.integer(tgt),
       as.integer(M), as.integer(S * N), .native_threads(), out = double(S * N * M), NAOK = TRUE)$out
  } else {
    .mafc_probs_r(d, sg, b, m, tgt, M)
  }
  pmax(array(p, c(S, N, M)), 0)
}

.mafc_probs_r <- function(d, sg, b, m, tgt, M) {
  n <- length(d)
  out <- matrix(0, n, M)
  for (k in seq_len(M)) {
    rows <- which(k <= m)
    if (!length(rows)) next
    a <- matrix(0, length(rows), M - 1L)
    cc <- matrix(40, length(rows), M - 1L)
    for (r in seq_along(rows)) {
      i <- rows[r]
      s <- tgt[i]
      pos <- setdiff(seq_len(M), k)
      for (t in seq_along(pos)) {
        j <- pos[t]
        if (j > m[i]) next
        if (k == s) {
          a[r, t] <- sg[i]; cc[r, t] <- d[i] + b[i, s] - b[i, j]
        } else if (j == s) {
          a[r, t] <- 1 / sg[i]; cc[r, t] <- (b[i, k] - b[i, s] - d[i]) / sg[i]
        } else {
          a[r, t] <- 1; cc[r, t] <- b[i, k] - b[i, j]
        }
      }
    }
    out[rows, k] <- exp(.pcq_r(a, cc))
  }
  out
}

# log of the integral of phi(z) prod_j Phi(a_j z + c_j) over rows of a, c: the integrand split at its
# mode and each side integrated by Gauss-Legendre to where it has fallen by e^-40, as the native kernel
.pcq_r <- function(a, cc) {
  gl <- .gauss_legendre(64L)
  n <- nrow(a)
  lam <- function(x) exp(stats::dnorm(x, log = TRUE) - stats::pnorm(x, log.p = TRUE))
  h <- function(z) -0.5 * z^2 + rowSums(stats::pnorm(a * z + cc, log.p = TRUE))
  g <- function(z) -z + rowSums(a * lam(a * z + cc))
  zl <- rep(0, n); zr <- pmax(g(rep(0, n)), 0); z <- 0.5 * zr
  for (it in seq_len(80L)) {
    x <- a * z + cc
    l <- lam(x)
    gz <- -z + rowSums(a * l)
    h2 <- -1 - rowSums(a * a * l * (x + l))
    zl <- ifelse(gz > 0, z, zl); zr <- ifelse(gz > 0, zr, z)
    zn <- z - gz / h2
    z <- ifelse(zn > zl & zn < zr, zn, 0.5 * (zl + zr))
  }
  hm <- h(z)
  edge <- function(side) {
    zo <- z + side * sqrt(80); zi <- z; e <- zo
    for (it in seq_len(60L)) {
      he <- h(e)
      en <- e - (he - (hm - 40)) / g(e)
      zo <- ifelse(he < hm - 40, e, zo); zi <- ifelse(he < hm - 40, zi, e)
      e <- ifelse((en - zi) * (en - zo) < 0, en, 0.5 * (zi + zo))
    }
    e
  }
  lo <- cbind(edge(-1), z); hi <- cbind(z, edge(1))
  tot <- numeric(n)
  for (p in 1:2) {
    hh <- 0.5 * (hi[, p] - lo[, p])
    for (i in seq_along(gl$x)) tot <- tot + gl$w[i] * hh * exp(h(lo[, p] + hh * (gl$x[i] + 1)) - hm)
  }
  hm + log(tot) - 0.5 * log(2 * pi)
}
