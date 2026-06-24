#' Summary Methods for SDT Model Fits
#'
#' Extract and label parameter estimates from fitted SDT models

#' Add ess_tail column to a summary data.frame if available
#' @noRd
.add_ess_tail <- function(df, draws_df, row_indices) {
  if ("ess_tail" %in% names(draws_df)) {
    df$ess_tail <- round(draws_df$ess_tail[row_indices], 0)
  }
  df
}

#' Summarize a Fitted SDT Model
#'
#' Produce a labeled table of posterior estimates for a fitted model. The printed
#' summary groups parameters into sections: fixed effects (per distributional
#' parameter, e.g. d', criterion, sigma), random-effect standard deviations,
#' random-effect correlations, and population-level thresholds. The print method
#' also reports sampler diagnostics.
#'
#' @param object A `broc_fit` object from [fit_broc()].
#' @param prob Probability mass for credible intervals (default 0.95)
#' @param digits Number of digits for rounding (default 3)
#' @param threshold_natural If `TRUE`, additionally report criterion random-effect
#'   SDs and correlations on the natural threshold scale rather than the internal
#'   log-gap scale.
#' @param ... Unused; present for S3 generic compatibility.
#' @return A structured summary object of class `broc_summary`. Its `print`
#'   method also shows a sampler-diagnostics banner (number of divergent
#'   transitions, max-treedepth hits, and minimum E-BFMI across chains).
#' @method summary broc_fit
#' @export
summary.broc_fit <- function(object, prob = 0.95, digits = 3, threshold_natural = FALSE, ...) {
  fit <- object
  
  model <- attr(fit, "broc_model")
  if (is.null(model)) {
    stop("Fit object must have 'broc_model' attribute. Use fit_broc() to fit the model.")
  }
  
  probs <- c((1 - prob) / 2, 1 - (1 - prob) / 2)
  
  # Get only the needed variables - much faster than getting everything
  vars_needed <- get_summary_variables(model)
  
  # Get summary for only needed variables
  draws_summary <- fit$summary(variables = vars_needed)
  
  # Convert to data.frame
  draws_df <- as.data.frame(draws_summary)
  rownames(draws_df) <- draws_df$variable
  
  # Find quantile columns
  col_names <- names(draws_df)
  pct_cols <- grep("%$", col_names, value = TRUE)
  if (length(pct_cols) >= 2) {
    pct_vals <- as.numeric(gsub("%", "", pct_cols))
    lower_col <- pct_cols[which.min(pct_vals)]
    upper_col <- pct_cols[which.max(pct_vals)]
  } else {
    q_cols <- grep("^q\\d", col_names, value = TRUE)
    if (length(q_cols) >= 2) {
      q_vals <- as.numeric(gsub("^q", "", q_cols))
      lower_col <- q_cols[which.min(q_vals)]
      upper_col <- q_cols[which.max(q_vals)]
    } else {
      warning("Could not find quantile columns. CIs will be NA.")
      draws_df$lower_ci <- NA
      draws_df$upper_ci <- NA
      lower_col <- "lower_ci"
      upper_col <- "upper_ci"
    }
  }
  
  result <- list()
  result$fixed <- summarize_fixed_effects_fast(draws_df, model, lower_col, upper_col, digits)
  result$random_sd <- summarize_random_sds_fast(draws_df, model, lower_col, upper_col, digits)
  result$correlations <- summarize_correlations_fast(draws_df, model, lower_col, upper_col, digits)
  result$thresholds <- compute_population_thresholds_fast(fit, model, probs, digits)

  if (isTRUE(threshold_natural)) {
    result$threshold_natural <- compute_natural_threshold_stats(fit, model, probs, digits)
  }

  # Smooth term penalty SDs
  result$smooth_sds <- summarize_smooth_sds(draws_df, model, lower_col, upper_col, digits)

  result$model_info <- list(
    family = model$family$family,
    N = model$model_data$N,
    K = model$model_data$K,
    n_chains = fit$num_chains(),
    n_iter = fit$metadata()$iter_sampling,
    digits = digits
  )

  # Sampler diagnostics (divergences / treedepth / E-BFMI), shown by print().
  result$diagnostics <- tryCatch(fit$diagnostic_summary(),
                                 error = function(e) NULL)

  class(result) <- "broc_summary"
  result
}

#' Format a summary table's numeric columns with a fixed number of decimals so
#' trailing zeros are not dropped (e.g. 0.500 not 0.5, 1.000 not 1). Estimate-
#' style columns use `digits`; rhat always uses 3; integer ess columns are left
#' as-is. Keeps right-alignment by padding to a common width per column.
#' @noRd
.fmt_summary_tbl <- function(df, digits = 3L) {
  if (!is.data.frame(df) || nrow(df) == 0) return(df)
  rjust <- function(v, d) {
    s <- formatC(as.numeric(v), format = "f", digits = d)
    formatC(s, width = max(nchar(s)))
  }
  for (col in intersect(c("estimate", "sd", "lower", "upper", "mean", "median"), names(df))) {
    if (is.numeric(df[[col]])) df[[col]] <- rjust(df[[col]], digits)
  }
  if ("rhat" %in% names(df) && is.numeric(df$rhat)) df$rhat <- rjust(df$rhat, 3L)
  df
}


#' Get list of variables needed for summary
#' @noRd
get_summary_variables <- function(model) {
  vars <- character(0)
  
  # Check if cumulative model (uses mu/cutpoints naming)
  is_cumulative <- model$family$family == "cumulative"
  is_bivariate_cumulative <- model$family$family == "bivariate_cumulative"
  is_cdp <- model$family$family == "cdp"

  # Parameter name mappings for cumulative vs SDT
  dprime_name <- if (is_cumulative) "mu" else if (is_bivariate_cumulative) "mu1" else "dprime"
  criterion_name <- if (is_cumulative) "cutpoints" else if (is_bivariate_cumulative) "cutpoints1" else "criterion"
  
  # Fixed effects - handle case where there are no fixed effects (cumulative with only RE)
  has_dprime_fixed <- isTRUE(model$model_data$has_dprime_fixed)
  if (has_dprime_fixed) {
    P_dprime <- model$model_data$dprime_fixed$n_coef
    if (P_dprime > 0) {
      vars <- c(vars, paste0("beta_", dprime_name, "[", 1:P_dprime, "]"))
    }
  }
  
  P_crit <- model$model_data$criterion$n_coef
  vars <- c(vars, paste0("beta_thresh_mid[", 1:P_crit, "]"))
  
  # Number of log_gaps: J-1 for CDP, K-2 for standard models
  K <- model$model_data$K
  if (is_cdp) {
    J <- model$family$J
    n_gaps <- J - 1
  } else {
    n_gaps <- K - 2
  }
  
  if (n_gaps > 0) {
    gap_param <- if (identical(model$gap_link, "softplus")) "beta_raw_gaps" else "beta_log_gaps"
    # Use prefix-only so both 1D (JAX P_crit==1) and 2D (Stan) naming works
    # via the fuzzy prefix matching in broc_fit$summary()/$draws()
    vars <- c(vars, gap_param)
  }

  if (model$model_data$has_sigma) {
    P_sigma <- model$model_data$sigma_fixed$n_coef
    vars <- c(vars, paste0("beta_sigma[", 1:P_sigma, "]"))
  }
  
  if (model$model_data$has_lambda) {
    P_lambda <- model$model_data$lambda_fixed$n_coef
    vars <- c(vars, paste0("beta_lambda[", 1:P_lambda, "]"))
  }
  
  if (model$model_data$has_dprime2 || isTRUE(model$model_data$needs_ordered_dprime)) {
    P_dprime2 <- model$model_data$dprime2_fixed$n_coef
    vars <- c(vars, paste0("beta_dprime2[", 1:P_dprime2, "]"))
  }
  
  if (model$model_data$has_sigma2) {
    P_sigma2 <- model$model_data$sigma2_fixed$n_coef
    vars <- c(vars, paste0("beta_sigma2[", 1:P_sigma2, "]"))
  }
  
  if (isTRUE(model$model_data$has_dprime_B)) {
    P_dprime_B <- model$model_data$dprime_B_fixed$n_coef
    vars <- c(vars, paste0("beta_dprime_B[", 1:P_dprime_B, "]"))
  }
  
  if (isTRUE(model$model_data$has_lambda_B)) {
    P_lambda_B <- model$model_data$lambda_B_fixed$n_coef
    vars <- c(vars, paste0("beta_lambda_B[", 1:P_lambda_B, "]"))
  }
  
  # Bivariate SDT fixed effects
  if (isTRUE(model$model_data$has_discrim)) {
    P_discrim <- model$model_data$discrim_fixed$n_coef
    if (P_discrim > 0) {
      discrim_var_name <- if (is_bivariate_cumulative) "mu2" else "discrim"
      vars <- c(vars, paste0("beta_", discrim_var_name, "[", 1:P_discrim, "]"))
    }
  }

  if (isTRUE(model$model_data$has_discrim_B)) {
    P_discrim_B <- model$model_data$discrim_B_fixed$n_coef
    vars <- c(vars, paste0("beta_discrim_B[", 1:P_discrim_B, "]"))
  }
  
  if (isTRUE(model$model_data$has_sigma_B)) {
    P_sigma_B <- model$model_data$sigma_B_fixed$n_coef
    vars <- c(vars, paste0("beta_sigma_B[", 1:P_sigma_B, "]"))
  }
  
  if (isTRUE(model$model_data$has_sigma2_B)) {
    P_sigma2_B <- model$model_data$sigma2_B_fixed$n_coef
    vars <- c(vars, paste0("beta_sigma2_B[", 1:P_sigma2_B, "]"))
  }
  
  if (isTRUE(model$model_data$has_rho)) {
    P_rho <- model$model_data$rho_fixed$n_coef
    vars <- c(vars, paste0("beta_rho[", 1:P_rho, "]"))
  }
  
  if (isTRUE(model$model_data$has_rho_B)) {
    P_rho_B <- model$model_data$rho_B_fixed$n_coef
    vars <- c(vars, paste0("beta_rho_B[", 1:P_rho_B, "]"))
  }
  
  if (isTRUE(model$model_data$has_rho_N)) {
    P_rho_N <- model$model_data$rho_N_fixed$n_coef
    vars <- c(vars, paste0("beta_rho_N[", 1:P_rho_N, "]"))
  }
  
  if (isTRUE(model$model_data$has_rec_crit)) {
    P_rec_crit <- model$model_data$rec_crit_fixed$n_coef
    vars <- c(vars, paste0("beta_rec_crit[", 1:P_rec_crit, "]"))
  }
  if (isTRUE(model$model_data$has_know_crit)) {
    P_know_crit <- model$model_data$know_crit_fixed$n_coef
    vars <- c(vars, paste0("beta_know_crit[", 1:P_know_crit, "]"))
  }

  if (isTRUE(model$model_data$has_dprime_L)) {
    P_dprime_L <- model$model_data$dprime_L_fixed$n_coef
    vars <- c(vars, paste0("beta_dprime_L[", 1:P_dprime_L, "]"))
  }
  
  if (isTRUE(model$model_data$has_sigma_L)) {
    P_sigma_L <- model$model_data$sigma_L_fixed$n_coef
    vars <- c(vars, paste0("beta_sigma_L[", 1:P_sigma_L, "]"))
  }
  
  if (isTRUE(model$model_data$has_lambda_L)) {
    P_lambda_L <- model$model_data$lambda_L_fixed$n_coef
    vars <- c(vars, paste0("beta_lambda_L[", 1:P_lambda_L, "]"))
  }
  if (isTRUE(model$model_data$has_lambda2)) {
    P_lambda2 <- model$model_data$lambda2_fixed$n_coef
    vars <- c(vars, paste0("beta_lambda2[", 1:P_lambda2, "]"))
  }
  if (isTRUE(model$model_data$has_lambda2_B)) {
    P_lambda2_B <- model$model_data$lambda2_B_fixed$n_coef
    vars <- c(vars, paste0("beta_lambda2_B[", 1:P_lambda2_B, "]"))
  }

  if (isTRUE(model$model_data$has_criterion2)) {
    K2 <- model$model_data$K2

    if (isTRUE(model$model_data$varying_source_criteria)) {
      K1 <- model$model_data$K
      n_vary <- K1
      vars <- c(vars, paste0("beta_thresh_mid_2_varying[", 1:n_vary, "]"))
      for (k1 in 1:n_vary) {
        for (k2 in 1:(K2-2)) {
          vars <- c(vars, paste0(gap_param, "_2_varying[", k1, ",", k2, "]"))
        }
      }
      # Shared new-item source thresholds
      if (identical(model$model_data$new_source_criteria, "shared")) {
        vars <- c(vars, "beta_thresh_mid_2_new")
        if (K2 > 2) {
          vars <- c(vars, paste0(gap_param, "_2_new[", 1:(K2-2), "]"))
        }
      }
    } else {
      # Standard: single set of source thresholds
      P_crit2 <- model$model_data$criterion2$n_coef
      vars <- c(vars, paste0("beta_thresh_mid_2[", 1:P_crit2, "]"))

      if (K2 > 2) {
        # Use prefix-only so both 1D and 2D naming works
        vars <- c(vars, paste0(gap_param, "_2"))
      }
    }
  }
  
  # Random effect SDs - get from model structure
  # For cumulative, dprime -> mu in Stan variable names
  # Include bivariate and CDP parameters
  param_names <- c("dprime", "sigma", "lambda", "dprime2", "sigma2", "dprime_B", "lambda_B",
                   "discrim", "discrim_B", "sigma_B", "sigma2_B", "rho", "rho_B", "rho_N", "lambda2", "lambda2_B",
                   "rec_crit", "know_crit",
                   "dprime_L", "sigma_L", "lambda_L")
  for (pname in param_names) {
    re_list <- model$model_data[[paste0(pname, "_random")]]
    if (is.null(re_list)) next
    
    # Get the Stan variable name (mu for cumulative dprime)
    stan_pname <- if (is_cumulative && pname == "dprime") "mu" else
      if (is_bivariate_cumulative && pname == "dprime") "mu1" else
      if (is_bivariate_cumulative && pname == "discrim") "mu2" else pname
    
    for (group in names(re_list)) {
      re <- re_list[[group]]
      if (is.null(re$cor_id)) {
        if (re$dim == 1) {
          vars <- c(vars, paste0("sigma_", stan_pname, "_", group))
        } else {
          vars <- c(vars, paste0("sigma_", stan_pname, "_", group, "[", 1:re$dim, "]"))
        }
      }
    }
  }
  
  # Criterion SDs (cutpoints for cumulative)
  for (group in names(model$model_data$criterion$random)) {
    re <- model$model_data$criterion$random[[group]]
    if (is.null(re$cor_id)) {
      vars <- c(vars, paste0("sigma_", criterion_name, "_", group, "[", 1:re$dim, "]"))
    }
  }
  
  # Criterion2 SDs (for bivariate_sdt/bivariate_cumulative)
  crit2_stan_name <- if (is_bivariate_cumulative) "cutpoints2" else "criterion2"
  if (isTRUE(model$model_data$has_criterion2) && !is.null(model$model_data$criterion2$random)) {
    for (group in names(model$model_data$criterion2$random)) {
      re <- model$model_data$criterion2$random[[group]]
      if (is.null(re$cor_id)) {
        vars <- c(vars, paste0("sigma_", crit2_stan_name, "_", group, "[", 1:re$dim, "]"))
      }
    }
  }
  
  # Cross-parameter SDs
  for (cor_id in names(model$model_data$cross_cor)) {
    cc <- model$model_data$cross_cor[[cor_id]]
    vars <- c(vars, paste0("sigma_cross_", cor_id, "_", cc$group, "[", 1:cc$total_dim, "]"))
  }
  
  # Correlations (only off-diagonal elements)
  for (pname in param_names) {
    re_list <- model$model_data[[paste0(pname, "_random")]]
    if (is.null(re_list)) next
    
    # Get Stan variable name (mu for cumulative dprime)
    stan_pname <- if (is_cumulative && pname == "dprime") "mu" else
      if (is_bivariate_cumulative && pname == "dprime") "mu1" else
      if (is_bivariate_cumulative && pname == "discrim") "mu2" else pname
    
    for (group in names(re_list)) {
      re <- re_list[[group]]
      if (is.null(re$cor_id) && re$dim > 1) {
        for (i in 1:(re$dim-1)) {
          for (j in (i+1):re$dim) {
            vars <- c(vars, paste0("corr_", stan_pname, "_", group, "[", i, ",", j, "]"))
          }
        }
      }
    }
  }
  
  # Criterion correlations (cutpoints for cumulative)
  crit_corr_name <- if (is_cumulative) "cutpoints" else if (is_bivariate_cumulative) "cutpoints1" else "criterion"
  for (group in names(model$model_data$criterion$random)) {
    re <- model$model_data$criterion$random[[group]]
    if (is.null(re$cor_id) && re$dim > 1) {
      for (i in 1:(re$dim-1)) {
        for (j in (i+1):re$dim) {
          vars <- c(vars, paste0("corr_", crit_corr_name, "_", group, "[", i, ",", j, "]"))
        }
      }
    }
  }
  
  # Criterion2 correlations (for bivariate_sdt/bivariate_cumulative)
  if (isTRUE(model$model_data$has_criterion2) && !is.null(model$model_data$criterion2$random)) {
    for (group in names(model$model_data$criterion2$random)) {
      re <- model$model_data$criterion2$random[[group]]
      if (is.null(re$cor_id) && re$dim > 1) {
        for (i in 1:(re$dim-1)) {
          for (j in (i+1):re$dim) {
            vars <- c(vars, paste0("corr_", crit2_stan_name, "_", group, "[", i, ",", j, "]"))
          }
        }
      }
    }
  }
  
  # Cross correlations
  for (cor_id in names(model$model_data$cross_cor)) {
    cc <- model$model_data$cross_cor[[cor_id]]
    for (i in 1:(cc$total_dim-1)) {
      for (j in (i+1):cc$total_dim) {
        vars <- c(vars, paste0("corr_cross_", cor_id, "_", cc$group, "[", i, ",", j, "]"))
      }
    }
  }

  # Smooth penalty SDs
  if (!is.null(model$model_data$smooth_data)) {
    for (pname in names(model$model_data$smooth_data)) {
      for (sm in model$model_data$smooth_data[[pname]]) {
        n_thresh <- if (!is.null(sm$n_thresh)) sm$n_thresh else 0L
        for (comp in sm$components) {
          for (k in seq_along(comp$Zs_list)) {
            base <- paste0("sds_", pname, "_", comp$san_label, "_", k)
            if (n_thresh > 0) {
              vars <- c(vars, paste0(base, "[", seq_len(n_thresh), "]"))
            } else {
              vars <- c(vars, base)
            }
          }
        }
      }
    }
  }

  unique(vars)
}


#' Fast Fixed Effects Summary
#' @noRd
summarize_fixed_effects_fast <- function(draws_df, model, lower_col, upper_col, digits) {
  
  result <- list()
  
  # Check if cumulative model
  is_cumulative <- model$family$family == "cumulative"
  is_bivariate_cumulative <- model$family$family == "bivariate_cumulative"
  dprime_beta_pattern <- if (is_cumulative) "^beta_mu\\[" else if (is_bivariate_cumulative) "^beta_mu1\\[" else "^beta_dprime\\["
  
  build_summary_df <- function(pattern, coef_names) {
    rows <- grep(pattern, draws_df$variable)
    if (length(rows) == 0) return(NULL)

    if (is.null(coef_names) || length(coef_names) != length(rows)) {
      coef_names <- draws_df$variable[rows]
    }
    # Clean up R model.matrix naming conventions
    coef_names <- gsub("\\(Intercept\\)", "intercept", coef_names)
    
    df <- data.frame(
      parameter = coef_names,
      estimate = round(draws_df$mean[rows], digits),
      sd = round(draws_df$sd[rows], digits),
      lower = round(draws_df[[lower_col]][rows], digits),
      upper = round(draws_df[[upper_col]][rows], digits),
      rhat = round(draws_df$rhat[rows], 3),
      ess_bulk = round(draws_df$ess_bulk[rows], 0),
      stringsAsFactors = FALSE
    )
    if ("ess_tail" %in% names(draws_df)) {
      df$ess_tail <- round(draws_df$ess_tail[rows], 0)
    }
    df
  }
  
  # For cumulative, only include dprime/mu if there are fixed effects
  has_dprime_fixed <- isTRUE(model$model_data$has_dprime_fixed)
  if (has_dprime_fixed && model$model_data$dprime_fixed$n_coef > 0) {
    dprime_summary <- build_summary_df(dprime_beta_pattern, model$model_data$dprime_fixed$coef_names)
    # Annotate scale if dprime uses a transforming link (e.g., log for bounded SDT)
    dprime_link <- model$family$params$dprime$link
    if (identical(dprime_link, "log")) dprime_summary$scale <- "(log)"
    else if (identical(dprime_link, "logit")) dprime_summary$scale <- "(logit)"
    # Use appropriate name in result
    if (is_cumulative) {
      result$mu <- dprime_summary
    } else if (is_bivariate_cumulative) {
      result$mu1 <- dprime_summary
    } else {
      result$dprime <- dprime_summary
    }
  }
  
  crit_names <- model$model_data$criterion$coef_names
  result$criterion_thresh_mid <- build_summary_df("^beta_thresh_mid\\[", crit_names)
  
  gap_pattern <- if (identical(model$gap_link, "softplus")) "^beta_raw_gaps\\[" else "^beta_log_gaps\\["
  gap_label_prefix <- if (identical(model$gap_link, "softplus")) "raw_gap" else "log_gap"
  log_gaps_rows <- grep(gap_pattern, draws_df$variable)
  if (length(log_gaps_rows) > 0) {
    K <- model$model_data$K
    P_crit <- model$model_data$criterion$n_coef
    is_cdp_model <- isTRUE(model$family$family == "cdp")
    n_gaps_label <- if (is_cdp_model) model$family$J - 1 else K - 2
    gap_names <- character(length(log_gaps_rows))
    idx <- 1
    for (k in 1:n_gaps_label) {
      for (p in 1:P_crit) {
        cond_name <- if (!is.null(crit_names) && p <= length(crit_names)) crit_names[p] else paste0("cond", p)
        cond_name <- gsub("\\(Intercept\\)", "intercept", cond_name)
        gap_names[idx] <- paste0(gap_label_prefix, "[", k, ",", cond_name, "]")
        idx <- idx + 1
      }
    }
    result$criterion_gaps <- data.frame(
      parameter = gap_names,
      estimate = round(draws_df$mean[log_gaps_rows], digits),
      sd = round(draws_df$sd[log_gaps_rows], digits),
      lower = round(draws_df[[lower_col]][log_gaps_rows], digits),
      upper = round(draws_df[[upper_col]][log_gaps_rows], digits),
      rhat = round(draws_df$rhat[log_gaps_rows], 3),
      ess_bulk = round(draws_df$ess_bulk[log_gaps_rows], 0),
      stringsAsFactors = FALSE
    )
    result$criterion_gaps <- .add_ess_tail(result$criterion_gaps, draws_df, log_gaps_rows)
  }

  if (model$model_data$has_sigma) {
    result$sigma <- build_summary_df("^beta_sigma\\[", model$model_data$sigma_fixed$coef_names)
    if (!is.null(result$sigma)) result$sigma$scale <- "(log)"
  }
  
  if (model$model_data$has_lambda) {
    result$lambda <- build_summary_df("^beta_lambda\\[", model$model_data$lambda_fixed$coef_names)
    if (!is.null(result$lambda)) result$lambda$scale <- "(logit)"
  }
  
  if (model$model_data$has_dprime2 || isTRUE(model$model_data$needs_ordered_dprime)) {
    result$dprime2 <- build_summary_df("^beta_dprime2\\[", model$model_data$dprime2_fixed$coef_names)
    if (!is.null(result$dprime2)) {
      d2_link <- model$family$params$dprime2$link
      if (identical(d2_link, "log")) result$dprime2$scale <- "(log)"
      else if (identical(d2_link, "logit")) result$dprime2$scale <- "(logit)"
    }
  }
  
  if (model$model_data$has_sigma2) {
    result$sigma2 <- build_summary_df("^beta_sigma2\\[", model$model_data$sigma2_fixed$coef_names)
    if (!is.null(result$sigma2)) result$sigma2$scale <- "(log)"
  }
  
  if (isTRUE(model$model_data$has_dprime_B)) {
    result$dprime_B <- build_summary_df("^beta_dprime_B\\[", model$model_data$dprime_B_fixed$coef_names)
    if (!is.null(result$dprime_B)) {
      dB_link <- model$family$params$dprime_B$link
      if (identical(dB_link, "log")) result$dprime_B$scale <- "(log)"
      else if (identical(dB_link, "logit")) result$dprime_B$scale <- "(logit)"
    }
  }
  
  if (isTRUE(model$model_data$has_lambda_B)) {
    result$lambda_B <- build_summary_df("^beta_lambda_B\\[", model$model_data$lambda_B_fixed$coef_names)
    if (!is.null(result$lambda_B)) result$lambda_B$scale <- "(logit)"
  }
  
  # Bivariate SDT fixed effects
  if (isTRUE(model$model_data$has_discrim) && model$model_data$discrim_fixed$n_coef > 0) {
    discrim_name <- if (is_bivariate_cumulative) "mu2" else "discrim"
    discrim_pattern <- paste0("^beta_", discrim_name, "\\[")
    result[[discrim_name]] <- build_summary_df(discrim_pattern, model$model_data$discrim_fixed$coef_names)
    if (!is.null(result[[discrim_name]])) {
      di_link <- model$family$params$discrim$link
      if (identical(di_link, "log")) result[[discrim_name]]$scale <- "(log)"
      else if (identical(di_link, "logit")) result[[discrim_name]]$scale <- "(logit)"
    }
  }

  if (isTRUE(model$model_data$has_discrim_B)) {
    result$discrim_B <- build_summary_df("^beta_discrim_B\\[", model$model_data$discrim_B_fixed$coef_names)
    if (!is.null(result$discrim_B)) {
      diB_link <- model$family$params$discrim_B$link
      if (identical(diB_link, "log")) result$discrim_B$scale <- "(log)"
      else if (identical(diB_link, "logit")) result$discrim_B$scale <- "(logit)"
    }
  }
  
  if (isTRUE(model$model_data$has_sigma_B)) {
    result$sigma_B <- build_summary_df("^beta_sigma_B\\[", model$model_data$sigma_B_fixed$coef_names)
    if (!is.null(result$sigma_B)) result$sigma_B$scale <- "(log)"
  }
  
  if (isTRUE(model$model_data$has_sigma2_B)) {
    result$sigma2_B <- build_summary_df("^beta_sigma2_B\\[", model$model_data$sigma2_B_fixed$coef_names)
    if (!is.null(result$sigma2_B)) result$sigma2_B$scale <- "(log)"
  }
  
  if (isTRUE(model$model_data$has_rho)) {
    result$rho <- build_summary_df("^beta_rho\\[", model$model_data$rho_fixed$coef_names)
    if (!is.null(result$rho)) {
      rho_link <- model$family$params$rho$link
      result$rho$scale <- if (identical(rho_link, "logis")) "(logit)" else "(Fisher z)"
    }
  }

  if (isTRUE(model$model_data$has_rho_B)) {
    result$rho_B <- build_summary_df("^beta_rho_B\\[", model$model_data$rho_B_fixed$coef_names)
    if (!is.null(result$rho_B)) {
      rho_B_link <- model$family$params$rho_B$link
      result$rho_B$scale <- if (identical(rho_B_link, "logis")) "(logit)" else "(Fisher z)"
    }
  }

  if (isTRUE(model$model_data$has_rho_N)) {
    result$rho_N <- build_summary_df("^beta_rho_N\\[", model$model_data$rho_N_fixed$coef_names)
    if (!is.null(result$rho_N)) {
      rho_N_link <- model$family$params$rho_N$link
      result$rho_N$scale <- if (identical(rho_N_link, "logis")) "(logit)" else "(Fisher z)"
    }
  }
  
  if (isTRUE(model$model_data$has_rec_crit)) {
    rc_coef <- model$model_data$rec_crit_fixed$coef_names
    result$rec_crit <- build_summary_df("^beta_rec_crit\\[", rc_coef)
  }

  if (isTRUE(model$model_data$has_know_crit)) {
    kc_coef <- model$model_data$know_crit_fixed$coef_names
    result$know_crit <- build_summary_df("^beta_know_crit\\[", kc_coef)
  }

  # Lure mixture fixed effects
  if (isTRUE(model$model_data$has_dprime_L)) {
    result$dprime_L <- build_summary_df("^beta_dprime_L\\[", model$model_data$dprime_L_fixed$coef_names)
  }
  
  if (isTRUE(model$model_data$has_sigma_L)) {
    result$sigma_L <- build_summary_df("^beta_sigma_L\\[", model$model_data$sigma_L_fixed$coef_names)
    if (!is.null(result$sigma_L)) result$sigma_L$scale <- "(log)"
  }
  
  if (isTRUE(model$model_data$has_lambda_L)) {
    result$lambda_L <- build_summary_df("^beta_lambda_L\\[", model$model_data$lambda_L_fixed$coef_names)
    if (!is.null(result$lambda_L)) result$lambda_L$scale <- "(logit)"
  }

  if (isTRUE(model$model_data$has_lambda2)) {
    result$lambda2 <- build_summary_df("^beta_lambda2\\[", model$model_data$lambda2_fixed$coef_names)
    if (!is.null(result$lambda2)) result$lambda2$scale <- "(logit)"
  }

  if (isTRUE(model$model_data$has_lambda2_B)) {
    result$lambda2_B <- build_summary_df("^beta_lambda2_B\\[", model$model_data$lambda2_B_fixed$coef_names)
    if (!is.null(result$lambda2_B)) result$lambda2_B$scale <- "(logit)"
  }

  if (isTRUE(model$model_data$has_criterion2)) {
    K2 <- model$model_data$K2
    gap_param_base <- if (identical(model$gap_link, "softplus")) "beta_raw_gaps" else "beta_log_gaps"
    gap_label <- if (identical(model$gap_link, "softplus")) "raw_gap" else "log_gap"
    if (isTRUE(model$model_data$varying_source_criteria)) {
      # Varying source criteria: thresholds indexed by item response level
      K1 <- model$model_data$K
      n_vary <- K1
      level_labels <- 1:K1

      # First thresholds for each item level
      thresh_rows <- grep("^beta_thresh_mid_2_varying\\[", draws_df$variable)
      if (length(thresh_rows) > 0) {
        thresh_names <- paste0("thresh_mid_2[item_level=", level_labels, "]")
        result$criterion2_thresh_mid_varying <- data.frame(
          parameter = thresh_names,
          estimate = round(draws_df$mean[thresh_rows], digits),
          sd = round(draws_df$sd[thresh_rows], digits),
          lower = round(draws_df[[lower_col]][thresh_rows], digits),
          upper = round(draws_df[[upper_col]][thresh_rows], digits),
          rhat = round(draws_df$rhat[thresh_rows], 3),
          ess_bulk = round(draws_df$ess_bulk[thresh_rows], 0),
          stringsAsFactors = FALSE
        )
        result$criterion2_thresh_mid_varying <- .add_ess_tail(result$criterion2_thresh_mid_varying, draws_df, thresh_rows)
      }

      # Gaps for each item level
      gap_2_vary_pattern <- paste0("^", gap_param_base, "_2_varying\\[")
      log_gaps_rows <- grep(gap_2_vary_pattern, draws_df$variable)
      if (length(log_gaps_rows) > 0) {
        gap_names <- character(length(log_gaps_rows))
        idx <- 1
        for (i in seq_along(level_labels)) {
          for (k2 in 1:(K2-2)) {
            gap_names[idx] <- paste0(gap_label, "_2[item_level=", level_labels[i], ",gap=", k2, "]")
            idx <- idx + 1
          }
        }
        result$criterion2_log_gaps_varying <- data.frame(
          parameter = gap_names,
          estimate = round(draws_df$mean[log_gaps_rows], digits),
          sd = round(draws_df$sd[log_gaps_rows], digits),
          lower = round(draws_df[[lower_col]][log_gaps_rows], digits),
          upper = round(draws_df[[upper_col]][log_gaps_rows], digits),
          rhat = round(draws_df$rhat[log_gaps_rows], 3),
          ess_bulk = round(draws_df$ess_bulk[log_gaps_rows], 0),
          stringsAsFactors = FALSE
        )
        result$criterion2_log_gaps_varying <- .add_ess_tail(result$criterion2_log_gaps_varying, draws_df, log_gaps_rows)
      }

      # Shared new-item source thresholds
      if (identical(model$model_data$new_source_criteria, "shared")) {
        new_thresh_row <- grep("^beta_thresh_mid_2_new$", draws_df$variable)
        if (length(new_thresh_row) > 0) {
          gap_2_new_pattern <- paste0("^", gap_param_base, "_2_new\\[")
          new_rows <- c(new_thresh_row, grep(gap_2_new_pattern, draws_df$variable))
          new_names <- c("thresh_mid_2_new", paste0(gap_label, "_2_new[", 1:(K2-2), "]"))
          if (length(new_rows) > 0) {
            result$criterion2_new <- data.frame(
              parameter = new_names[seq_along(new_rows)],
              estimate = round(draws_df$mean[new_rows], digits),
              sd = round(draws_df$sd[new_rows], digits),
              lower = round(draws_df[[lower_col]][new_rows], digits),
              upper = round(draws_df[[upper_col]][new_rows], digits),
              rhat = round(draws_df$rhat[new_rows], 3),
              ess_bulk = round(draws_df$ess_bulk[new_rows], 0),
              stringsAsFactors = FALSE
            )
            result$criterion2_new <- .add_ess_tail(result$criterion2_new, draws_df, new_rows)
          }
        }
      }

    } else {
      # Standard: single set of source thresholds
      crit2_names <- model$model_data$criterion2$coef_names
      result$criterion2_thresh_mid <- build_summary_df("^beta_thresh_mid_2\\[", crit2_names)

      gap_2_pattern <- paste0("^", gap_param_base, "_2\\[")
      log_gaps_2_rows <- grep(gap_2_pattern, draws_df$variable)
      if (length(log_gaps_2_rows) > 0) {
        P_crit2 <- model$model_data$criterion2$n_coef
        gap_names <- character(length(log_gaps_2_rows))
        idx <- 1
        for (k in 1:(K2-2)) {
          for (p in 1:P_crit2) {
            cond_name <- if (!is.null(crit2_names) && p <= length(crit2_names)) crit2_names[p] else paste0("cond", p)
            gap_names[idx] <- paste0(gap_label, "_2[", k, ",", cond_name, "]")
            idx <- idx + 1
          }
        }
        result$criterion2_log_gaps <- data.frame(
          parameter = gap_names,
          estimate = round(draws_df$mean[log_gaps_2_rows], digits),
          sd = round(draws_df$sd[log_gaps_2_rows], digits),
          lower = round(draws_df[[lower_col]][log_gaps_2_rows], digits),
          upper = round(draws_df[[upper_col]][log_gaps_2_rows], digits),
          rhat = round(draws_df$rhat[log_gaps_2_rows], 3),
          ess_bulk = round(draws_df$ess_bulk[log_gaps_2_rows], 0),
          stringsAsFactors = FALSE
        )
        result$criterion2_log_gaps <- .add_ess_tail(result$criterion2_log_gaps, draws_df, log_gaps_2_rows)
      }
    }
  }

  # Rename parameters for CDP models to match formula names
  if (model$family$family %in% "cdp") {
    if (!is.null(result$dprime)) {
      names(result)[names(result) == "dprime"] <- "rec"
    }
    if (!is.null(result$dprime2)) {
      names(result)[names(result) == "dprime2"] <- "fam"
    }
    if (!is.null(result$sigma)) {
      names(result)[names(result) == "sigma"] <- "sigma_R"
    }
    if (!is.null(result$sigma2)) {
      names(result)[names(result) == "sigma2"] <- "sigma_F"
    }
  }

  # Canonical display order for fixed-effects sections (cosmetic). Aliases
  # (mu/mu1 for cumulative, rec/fam/sigma_R/sigma_F for cdp) share the rank of
  # the parameter they rename. Any slot not listed keeps its position at the end.
  .fixed_order <- c(
    "dprime", "mu", "mu1", "rec",
    "dprime_B", "mu2",
    "dprime2", "fam",
    "dprime_L",
    "discrim", "discrim_B",
    "sigma", "sigma_R", "sigma_B", "sigma2", "sigma_F", "sigma2_B", "sigma_L",
    "lambda", "lambda_B", "lambda2", "lambda2_B", "lambda_L",
    "rho", "rho_B", "rho_N",
    "rec_crit", "know_crit",
    "criterion_thresh_mid", "criterion_gaps",
    "criterion2_thresh_mid", "criterion2_log_gaps",
    "criterion2_thresh_mid_varying", "criterion2_log_gaps_varying", "criterion2_new"
  )
  result <- result[c(intersect(.fixed_order, names(result)),
                     setdiff(names(result), .fixed_order))]

  result
}


#' Fast Random Effects SD Summary
#' @noRd
summarize_random_sds_fast <- function(draws_df, model, lower_col, upper_col, digits) {
  
  result <- list()
  
  sigma_rows <- grep("^sigma_", draws_df$variable)
  if (length(sigma_rows) == 0) return(result)
  
  sigma_vars <- draws_df$variable[sigma_rows]
  is_cross <- grepl("^sigma_cross_", sigma_vars)
  
  non_cross_idx <- sigma_rows[!is_cross]
  if (length(non_cross_idx) > 0) {
    labels <- vapply(draws_df$variable[non_cross_idx], function(v) {
      label_sigma_parameter(v, model)
    }, character(1))
    
    result$individual <- data.frame(
      parameter = labels,
      estimate = round(draws_df$mean[non_cross_idx], digits),
      sd = round(draws_df$sd[non_cross_idx], digits),
      lower = round(draws_df[[lower_col]][non_cross_idx], digits),
      upper = round(draws_df[[upper_col]][non_cross_idx], digits),
      rhat = round(draws_df$rhat[non_cross_idx], 3),
      ess_bulk = round(draws_df$ess_bulk[non_cross_idx], 0),
      stringsAsFactors = FALSE,
      row.names = NULL
    )
    result$individual <- .add_ess_tail(result$individual, draws_df, non_cross_idx)
  }

  cross_idx <- sigma_rows[is_cross]
  if (length(cross_idx) > 0) {
    labels <- vapply(draws_df$variable[cross_idx], function(v) {
      label_cross_sigma_parameter(v, model)
    }, character(1))
    
    result$cross_parameter <- data.frame(
      parameter = labels,
      estimate = round(draws_df$mean[cross_idx], digits),
      sd = round(draws_df$sd[cross_idx], digits),
      lower = round(draws_df[[lower_col]][cross_idx], digits),
      upper = round(draws_df[[upper_col]][cross_idx], digits),
      rhat = round(draws_df$rhat[cross_idx], 3),
      ess_bulk = round(draws_df$ess_bulk[cross_idx], 0),
      stringsAsFactors = FALSE,
      row.names = NULL
    )
    result$cross_parameter <- .add_ess_tail(result$cross_parameter, draws_df, cross_idx)
  }

  attr(result, "digits") <- digits
  class(result) <- "broc_random_sd"
  result
}

#' @export
print.broc_random_sd <- function(x, ...) {
  digits <- attr(x, "digits") %||% 3L
  .print_re_sd_grouped <- function(df) {
    if (is.null(df) || nrow(df) == 0) return()
    params <- df$parameter
    pg <- sub("^sd\\((.+)\\).*$", "\\1", params)
    level <- ifelse(grepl("\\[", params), sub("^.*\\[(.+)\\]$", "\\1", params), "")
    for (g in unique(pg)) {
      mask <- pg == g
      sub_df <- df[mask, , drop = FALSE]
      lvls <- level[mask]
      cat("\n", g, ":\n", sep = "")
      if (all(lvls == "")) {
        out <- sub_df[, !names(sub_df) %in% "parameter", drop = FALSE]
        print(.fmt_summary_tbl(out, digits), row.names = FALSE)
      } else {
        out <- sub_df
        out$parameter <- lvls
        names(out)[names(out) == "parameter"] <- "level"
        print(.fmt_summary_tbl(out, digits), row.names = FALSE)
      }
    }
  }
  if (!is.null(x$individual) && nrow(x$individual) > 0) {
    .print_re_sd_grouped(x$individual)
  }
  if (!is.null(x$cross_parameter) && nrow(x$cross_parameter) > 0) {
    cat("\nCross-parameter correlated:\n")
    .print_re_sd_grouped(x$cross_parameter)
  }
  invisible(x)
}


#' Fast Correlation Summary
#' @noRd
summarize_correlations_fast <- function(draws_df, model, lower_col, upper_col, digits) {
  
  result <- list()
  
  corr_rows <- grep("^corr_", draws_df$variable)
  if (length(corr_rows) == 0) return(result)
  
  corr_vars <- draws_df$variable[corr_rows]
  matrix_bases <- unique(sub("\\[.*", "", corr_vars))
  
  for (mat_name in matrix_bases) {
    mat_pattern <- paste0("^", mat_name, "\\[")
    mat_rows <- grep(mat_pattern, draws_df$variable)
    if (length(mat_rows) == 0) next
    
    vars <- draws_df$variable[mat_rows]
    
    # Parse indices efficiently
    idx_matches <- regmatches(vars, regexec("\\[(\\d+),(\\d+)\\]", vars))
    indices <- do.call(rbind, lapply(idx_matches, function(m) {
      if (length(m) == 3) as.integer(m[2:3]) else c(NA, NA)
    }))
    
    dim_size <- max(indices, na.rm = TRUE)
    labels <- get_correlation_labels(mat_name, dim_size, model)
    
    # Off-diagonal only (already filtered by get_summary_variables)
    off_diag <- which(indices[, 1] < indices[, 2])
    
    if (length(off_diag) > 0) {
      pair_labels <- vapply(off_diag, function(k) {
        paste0(labels[indices[k, 1]], " ~ ", labels[indices[k, 2]])
      }, character(1))
      
      result[[mat_name]] <- data.frame(
        pair = pair_labels,
        estimate = round(draws_df$mean[mat_rows[off_diag]], digits),
        sd = round(draws_df$sd[mat_rows[off_diag]], digits),
        lower = round(draws_df[[lower_col]][mat_rows[off_diag]], digits),
        upper = round(draws_df[[upper_col]][mat_rows[off_diag]], digits),
        rhat = round(draws_df$rhat[mat_rows[off_diag]], 3),
        ess_bulk = round(draws_df$ess_bulk[mat_rows[off_diag]], 0),
        stringsAsFactors = FALSE,
        row.names = NULL
      )
      result[[mat_name]] <- .add_ess_tail(result[[mat_name]], draws_df, mat_rows[off_diag])
    }
  }

  attr(result, "digits") <- digits
  class(result) <- "broc_correlations"
  result
}

#' @export
print.broc_correlations <- function(x, ...) {
  digits <- attr(x, "digits") %||% 3L
  known_params <- c("rec_crit", "know_crit", "criterion2", "dprime_B",
                      "dprime_L", "lambda_B", "lambda_L", "discrim_B",
                      "sigma_B", "sigma_L", "sigma2_B", "rho_B", "rho_N")
  for (name in names(x)) {
    df <- x[[name]]
    if (!is.null(df) && nrow(df) > 0) {
      remainder <- gsub("^corr_", "", name)
      nice_name <- NULL
      for (kp in known_params) {
        prefix <- paste0(kp, "_")
        if (startsWith(remainder, prefix)) {
          group <- sub(paste0("^", gsub("([.|()\\^{}+$*?\\[\\]])", "\\\\\\1", prefix)), "", remainder)
          nice_name <- paste0(kp, " | ", group)
          break
        }
      }
      if (is.null(nice_name)) {
        parts <- strsplit(remainder, "_")[[1]]
        if (length(parts) >= 2) {
          nice_name <- paste0(parts[1], " | ", paste(parts[-1], collapse = "_"))
        } else {
          nice_name <- remainder
        }
      }
      cat("\n", nice_name, ":\n", sep = "")
      print(.fmt_summary_tbl(df, digits), row.names = FALSE)
    }
  }
  invisible(x)
}


#' Compute Population Thresholds - Fast Version
#' 
#' Only computed for models with simple categorical criterion predictors.
#' For models with continuous predictors or complex interactions, 
#' the fixed effects coefficients should be interpreted directly.
#' @noRd
compute_population_thresholds_fast <- function(fit, model, probs, digits) {

  gap_link <- if (!is.null(model$gap_link)) model$gap_link else "log"
  gap_transform <- if (gap_link == "softplus") function(x) log1p(exp(x)) else exp
  gap_param_name <- if (gap_link == "softplus") "beta_raw_gaps" else "beta_log_gaps"

  K <- model$model_data$K
  P_crit <- model$model_data$criterion$n_coef
  X_crit <- model$model_data$criterion$X
  crit_names <- model$model_data$criterion$coef_names
  
  # Check for continuous predictors by looking at the range of values
  # A continuous predictor will have many unique non-0/1 values
  has_continuous <- FALSE
  for (j in 1:P_crit) {
    unique_vals <- unique(X_crit[, j])
    # If more than a few unique values and not all 0/1, it's continuous
    if (length(unique_vals) > 10 && !all(unique_vals %in% c(0, 1))) {
      has_continuous <- TRUE
      break
    }
  }
  
  if (has_continuous) {
    # Return NULL - population thresholds don't make sense with continuous predictors
    # The user should interpret the fixed effects directly
    return(NULL)
  }
  
  # Original behavior for categorical-only models
  X_unique <- unique(X_crit)
  n_conditions <- nrow(X_unique)
  
  if (n_conditions == 1 && P_crit == 1) {
    condition_labels <- "intercept"
  } else {
    condition_labels <- apply(X_unique, 1, function(row) {
      active <- which(row != 0)
      if (length(active) == 0) return("baseline")
      if (length(active) == 1 && !is.null(crit_names)) return(crit_names[active])
      if (!is.null(crit_names)) {
        parts <- vapply(active, function(i) {
          if (row[i] == 1) crit_names[i] else paste0(round(row[i], 2), "*", crit_names[i])
        }, character(1))
        return(paste(parts, collapse = "+"))
      }
      paste0("cond", paste(active, collapse = "_"))
    })
  }
  
  # Get only the draws needed for thresholds
  thresh_vars <- paste0("beta_thresh_mid[", 1:P_crit, "]")
  gap_vars <- character(0)
  
  # Number of thresholds and gaps depends on family
  is_cdp <- model$family$family == "cdp"
  if (is_cdp) {
    J <- model$family$J
    n_thresh <- J
    n_gaps <- J - 1
  } else {
    n_thresh <- K - 1
    n_gaps <- K - 2
  }
  
  # Detect gap variable naming convention: JAX backend with P_crit==1
  # uses 1D names (beta_log_gaps[k]) while Stan always uses 2D (beta_log_gaps[k,j])
  use_1d_gap_names <- FALSE
  if (n_gaps > 0) {
    test_2d <- paste0(gap_param_name, "[1,1]")
    has_2d <- tryCatch({
      fit$draws(variables = test_2d, format = "matrix")
      TRUE
    }, error = function(e) FALSE)
    if (!has_2d) {
      test_1d <- paste0(gap_param_name, "[1]")
      has_1d <- tryCatch({
        fit$draws(variables = test_1d, format = "matrix")
        TRUE
      }, error = function(e) FALSE)
      if (has_1d) use_1d_gap_names <- TRUE
    }
    for (k in 1:n_gaps) {
      if (use_1d_gap_names) {
        gap_vars <- c(gap_vars, paste0(gap_param_name, "[", k, "]"))
      } else {
        gap_vars <- c(gap_vars, paste0(gap_param_name, "[", k, ",", 1:P_crit, "]"))
      }
    }
  }

  all_vars <- c(thresh_vars, gap_vars)
  draws <- fit$draws(variables = all_vars, format = "matrix")
  
  n_draws <- nrow(draws)
  results_list <- vector("list", n_conditions * n_thresh)
  list_idx <- 1
  
  for (c in 1:n_conditions) {
    x_row <- X_unique[c, , drop = FALSE]

    mid <- ceiling(n_thresh / 2)
    n_upper <- n_thresh - mid

    thresh_draws <- matrix(NA_real_, n_draws, n_thresh)

    # Mid-anchor threshold
    thresh_mid_draws <- as.matrix(draws[, thresh_vars, drop = FALSE]) %*% t(x_row)
    thresh_draws[, mid] <- thresh_mid_draws

    # Helper to get gap column names
    .gap_cols <- function(gap_idx) {
      if (use_1d_gap_names) {
        paste0(gap_param_name, "[", gap_idx, "]")
      } else {
        paste0(gap_param_name, "[", gap_idx, ",", 1:P_crit, "]")
      }
    }

    # Upper gaps: mid+1 to n_thresh
    if (n_upper > 0) {
      for (k in (mid + 1):n_thresh) {
        gap_cols <- .gap_cols(k - mid)
        gap_draws <- as.matrix(draws[, gap_cols, drop = FALSE]) %*% t(x_row)
        thresh_draws[, k] <- thresh_draws[, k - 1] + gap_transform(gap_draws)
      }
    }

    # Lower gaps: mid-1 down to 1
    if (mid > 1) {
      for (k_down in seq_len(mid - 1)) {
        k <- mid - k_down
        gap_cols <- .gap_cols(n_upper + k_down)
        gap_draws <- as.matrix(draws[, gap_cols, drop = FALSE]) %*% t(x_row)
        thresh_draws[, k] <- thresh_draws[, k + 1] - gap_transform(gap_draws)
      }
    }
    
    for (k in 1:n_thresh) {
      results_list[[list_idx]] <- data.frame(
        condition = condition_labels[c],
        threshold = k,
        estimate = round(mean(thresh_draws[, k]), digits),
        sd = round(sd(thresh_draws[, k]), digits),
        lower = round(quantile(thresh_draws[, k], probs[1]), digits),
        upper = round(quantile(thresh_draws[, k], probs[2]), digits),
        stringsAsFactors = FALSE,
        row.names = NULL
      )
      list_idx <- list_idx + 1
    }
  }
  
  do.call(rbind, results_list)
}


#' Label a sigma parameter
#' @noRd
label_sigma_parameter <- function(var_name, model) {
  # Check if cumulative model
  is_cumulative <- model$family$family == "cumulative"
  is_cdp <- model$family$family == "cdp"
  
  if (grepl("\\[", var_name)) {
    base <- sub("\\[.*", "", var_name)
    idx <- as.integer(gsub(".*\\[(\\d+)\\].*", "\\1", var_name))
  } else {
    base <- var_name
    idx <- NULL
  }
  
  # Strip "sigma_" prefix
  remainder <- sub("^sigma_", "", base)
  
  # Known parameter names (ordered longest first so greedy matching works)
  known_params <- c("dprime_B", "dprime_L", "lambda_B", "lambda_L", "discrim_B",
                    "sigma_B", "sigma_L", "sigma2_B", "rho_B", "rho_N",
                    "rec_crit", "know_crit", "criterion2",
                    "dprime", "sigma", "lambda", "dprime2", "sigma2",
                    "discrim", "rho", "criterion",
                    "mu", "cutpoints")
  
  param <- NULL
  group <- NULL
  for (kp in known_params) {
    prefix <- paste0(kp, "_")
    if (startsWith(remainder, prefix)) {
      param <- kp
      group <- sub(paste0("^", gsub("([.|()\\^{}+$*?\\[\\]])", "\\\\\\1", prefix)), "", remainder)
      break
    }
  }
  
  # Fallback: original split behavior
  if (is.null(param)) {
    parts <- strsplit(remainder, "_")[[1]]
    if (length(parts) >= 2) {
      param <- parts[1]
      group <- paste(parts[-1], collapse = "_")
    } else {
      return(var_name)
    }
  }
  
  # Map internal names to display names for cumulative
  display_param <- param
  if (is_cumulative) {
    if (param == "dprime") display_param <- "mu"
    if (param == "criterion") display_param <- "cutpoints"
  }

  # Map internal names to display names for CDP
  if (is_cdp) {
    if (param == "dprime") display_param <- "rec"
    if (param == "dprime2") display_param <- "fam"
    if (param == "sigma") display_param <- "sigma_R"
    if (param == "sigma2") display_param <- "sigma_F"
  }
  
  # For looking up in model_data, use internal names
  internal_param <- if (param %in% c("mu", "dprime")) "dprime" else if (param %in% c("cutpoints", "criterion")) "criterion" else param
  
  if (!is.null(idx)) {
    # For dim=1 REs, the [1] index is just an artifact of NumPyro naming -- treat as scalar
    re_list <- model$model_data[[paste0(internal_param, "_random")]]
    re <- NULL
    if (!is.null(re_list) && group %in% names(re_list)) {
      re <- re_list[[group]]
    }
    if (!is.null(re) && re$dim == 1) {
      # Scalar RE -- no level label needed, just "param | group"
      return(paste0(display_param, " | ", group))
    }
    level_label <- as.character(idx)
    if (!is.null(re) && !is.null(re$level_names) && idx <= length(re$level_names)) {
      level_label <- re$level_names[idx]
    }
    if (internal_param == "criterion") {
      # Criterion has (K-1) x n_re_terms dimensions
      re_crit <- model$model_data$criterion$random[[group]]
      K <- model$model_data$K
      if (!is.null(re_crit) && !is.null(re_crit$n_re_terms) && re_crit$n_re_terms > 1) {
        n_re_terms <- re_crit$n_re_terms
        re_labels <- if (!is.null(re_crit$level_names)) re_crit$level_names else paste0("term", 1:n_re_terms)
        thresh_idx <- ((idx - 1) %/% n_re_terms) + 1
        term_idx <- ((idx - 1) %% n_re_terms) + 1
        level_label <- paste0("thresh", thresh_idx, ":", re_labels[term_idx])
      } else {
        level_label <- paste0("thresh", idx)
      }
    } else if (!is.null(re) && !is.null(re$varying_re_mode) &&
               re$varying_re_mode %in% c("per_bin", "full") &&
               re$dim > length(re$level_names %||% character(0))) {
      # Per-bin varying RE: dim was overridden to K but level_names weren't updated
      level_label <- paste0("bin", idx)
    }
    return(sprintf("sd(%s | %s)[%s]", display_param, group, level_label))
  } else {
    return(sprintf("sd(%s | %s)", display_param, group))
  }
}


#' Label a cross-parameter sigma
#' @noRd
label_cross_sigma_parameter <- function(var_name, model) {
  if (!grepl("\\[", var_name)) return(var_name)
  
  base <- sub("\\[.*", "", var_name)
  idx <- as.integer(gsub(".*\\[(\\d+)\\].*", "\\1", var_name))
  
  parts <- strsplit(sub("^sigma_cross_", "", base), "_")[[1]]
  if (length(parts) < 2) return(var_name)
  
  cor_id <- parts[1]
  group <- paste(parts[-1], collapse = "_")
  
  cc <- model$model_data$cross_cor[[cor_id]]
  if (is.null(cc)) return(var_name)
  
  col_start <- 1
  for (member in cc$members) {
    col_end <- col_start + member$dim - 1
    if (idx >= col_start && idx <= col_end) {
      local_idx <- idx - col_start + 1
      param <- member$param
      
      level_label <- as.character(local_idx)
      re_list <- model$model_data[[paste0(param, "_random")]]
      if (!is.null(re_list) && group %in% names(re_list)) {
        re <- re_list[[group]]
        if (!is.null(re$level_names) && local_idx <= length(re$level_names)) {
          level_label <- re$level_names[local_idx]
        }
      }
      level_label <- gsub("\\(Intercept\\)", "intercept", level_label)
      # Map display names
      is_cumulative <- model$family$family == "cumulative"
      is_cdp_family <- model$family$family %in% "cdp"
      display_param <- param
      if (is_cumulative && param == "dprime") display_param <- "mu"
      else if (is_cumulative && param == "criterion") display_param <- "cutpoints"
      else if (is_cdp_family && param == "dprime") display_param <- "rec"
      else if (is_cdp_family && param == "dprime2") display_param <- "fam"
      else if (is_cdp_family && param == "sigma") display_param <- "sigma_R"
      else if (is_cdp_family && param == "sigma2") display_param <- "sigma_F"
      return(sprintf("sd(%s | %s)[%s]", display_param, group, level_label))
    }
    col_start <- col_end + 1
  }
  var_name
}


#' Get labels for correlation matrix dimensions
#' @noRd
get_correlation_labels <- function(mat_name, dim_size, model) {
  
  # Check if cumulative model
  is_cumulative <- model$family$family == "cumulative"
  
  if (grepl("^corr_cross_", mat_name)) {
    parts <- strsplit(sub("^corr_cross_", "", mat_name), "_")[[1]]
    cor_id <- parts[1]
    group <- paste(parts[-1], collapse = "_")
    
    cc <- model$model_data$cross_cor[[cor_id]]
    if (!is.null(cc)) {
      labels <- character(0)
      for (member in cc$members) {
        param <- member$param
        # Map to display name
        is_cdp_family <- model$family$family %in% "cdp"
        display_param <- param
        if (is_cumulative && param == "dprime") display_param <- "mu"
        else if (is_cumulative && param == "criterion") display_param <- "cutpoints"
        else if (is_cdp_family && param == "dprime") display_param <- "rec"
        else if (is_cdp_family && param == "dprime2") display_param <- "fam"
        else if (is_cdp_family && param == "sigma") display_param <- "sigma_R"
        else if (is_cdp_family && param == "sigma2") display_param <- "sigma_F"
        
        re_list <- model$model_data[[paste0(param, "_random")]]
        
        level_names <- NULL
        if (!is.null(re_list) && group %in% names(re_list)) {
          re <- re_list[[group]]
          if (!is.null(re$level_names) && length(re$level_names) == member$dim) {
            level_names <- re$level_names
          }
        }
        if (is.null(level_names)) level_names <- seq_len(member$dim)
        level_names <- gsub("\\(Intercept\\)", "intercept", level_names)
        labels <- c(labels, paste0(display_param, ":", level_names))
      }
      return(labels)
    }
  } else if (grepl("^corr_(criterion|cutpoints)_", mat_name)) {
    # Handle both criterion and cutpoints naming
    group <- sub("^corr_(criterion|cutpoints)_", "", mat_name)
    re <- model$model_data$criterion$random[[group]]
    K <- model$model_data$K
    
    if (!is.null(re) && !is.null(re$n_re_terms) && re$n_re_terms > 1) {
      # Has random slopes - labels are thresh x re_term combinations
      n_re_terms <- re$n_re_terms
      re_labels <- if (!is.null(re$level_names)) re$level_names else paste0("term", 1:n_re_terms)
      
      labels <- character(0)
      for (k in 1:(K-1)) {
        for (j in 1:n_re_terms) {
          labels <- c(labels, paste0("thresh", k, ":", re_labels[j]))
        }
      }
      return(labels)
    } else {
      # Intercept only - just threshold labels
      return(paste0("thresh", 1:(K - 1)))
    }
  } else {
    remainder <- sub("^corr_", "", mat_name)
    
    # Known parameter names (ordered longest first for greedy matching)
    known_params <- c("dprime_B", "dprime_L", "lambda_B", "lambda_L", "discrim_B",
                      "sigma_B", "sigma_L", "sigma2_B", "rho_B", "rho_N",
                      "rec_crit", "know_crit", "criterion2",
                      "dprime", "sigma", "lambda", "dprime2", "sigma2",
                      "discrim", "rho", "criterion",
                      "mu", "cutpoints")
    
    param <- NULL
    group <- NULL
    for (kp in known_params) {
      prefix <- paste0(kp, "_")
      if (startsWith(remainder, prefix)) {
        param <- kp
        group <- sub(paste0("^", gsub("([.|()\\^{}+$*?\\[\\]])", "\\\\\\1", prefix)), "", remainder)
        break
      }
    }
    
    # Fallback
    if (is.null(param)) {
      parts <- strsplit(remainder, "_")[[1]]
      param <- parts[1]
      group <- paste(parts[-1], collapse = "_")
    }
    
    # Map from Stan name to internal name for lookup
    internal_param <- if (param == "mu") "dprime" else if (param == "cutpoints") "criterion" else param
    
    re_list <- model$model_data[[paste0(internal_param, "_random")]]
    if (!is.null(re_list) && group %in% names(re_list)) {
      re <- re_list[[group]]
      if (!is.null(re$level_names) && length(re$level_names) == dim_size) {
        return(re$level_names)
      }
    }
  }
  # Fallback: use bin numbers for per-bin REs, generic dim labels otherwise
  if (!is.null(param) && param %in% c("rec_crit", "know_crit")) {
    paste0("bin", 1:dim_size)
  } else {
    paste0("dim", 1:dim_size)
  }
}


# =============================================================================
# Natural-Scale Threshold Statistics
# =============================================================================

#' Compute natural-scale threshold correlations from posterior draws.
#'
#' For each posterior draw, reconstructs natural-scale thresholds for each
#' participant by applying the gap->threshold transformation to the per-draw
#' RE vectors, then computes correlations directly from these natural-scale
#' draws. This is unbiased (no Jacobian approximation).
#'
#' Only outputs correlations (not SDs) since natural-scale SDs depend on
#' the specific gap values and are less interpretable.
#'
#' @noRd
compute_natural_threshold_stats <- function(fit, model, probs, digits) {
  gap_link <- if (!is.null(model$gap_link)) model$gap_link else "log"
  gap_fn <- if (gap_link == "softplus") function(x) log1p(exp(x)) else exp
  gap_param_name <- if (gap_link == "softplus") "beta_raw_gaps" else "beta_log_gaps"
  lower_q <- probs[1]
  upper_q <- probs[2]

  results <- list()

  # Process each criterion dimension (criterion, criterion2)
  for (crit_name in c("criterion", "criterion2")) {
    crit_data <- model$model_data[[crit_name]]
    if (is.null(crit_data)) next

    crit_random <- crit_data$random
    if (is.null(crit_random) || length(crit_random) == 0) next

    P_crit <- crit_data$n_coef
    X_crit <- crit_data$X

    # Threshold dimensions
    if (crit_name == "criterion") {
      K <- model$model_data$K
      n_thresh <- K - 1
      gap_base <- gap_param_name
      mid_base <- "beta_thresh_mid"
    } else {
      K2 <- model$model_data$K2
      if (is.null(K2)) next
      n_thresh <- K2 - 1
      gap_base <- paste0(gap_param_name, "_2")
      mid_base <- "beta_thresh_mid_2"
    }
    if (n_thresh < 2) next

    n_gaps <- n_thresh - 1
    mid <- ceiling(n_thresh / 2)
    n_upper <- n_thresh - mid

    for (group_name in names(crit_random)) {
      re <- crit_random[[group_name]]
      D <- re$dim
      if (D < 2) next
      if (!isTRUE(re$correlated)) next
      if (!is.null(re$cor_id)) next

      n_groups <- re$n_groups
      n_re_terms <- if (!is.null(re$n_re_terms)) re$n_re_terms else 1
      if (n_groups < 3) next

      # Collect draws as matrices
      u_var <- paste0("u_", crit_name, "_", group_name)
      u_col_names <- as.vector(outer(
        paste0(u_var, "["),
        paste0(seq_len(n_groups), ","),
        FUN = paste0
      ))
      u_col_names <- as.vector(outer(
        u_col_names,
        paste0(seq_len(D), "]"),
        FUN = paste0
      ))

      gap_col_names <- character(0)
      use_1d_gaps <- FALSE
      if (n_gaps > 0) {
        # Detect naming convention: JAX with P_crit==1 uses 1D (gap[k]),
        # Stan always uses 2D (gap[k,j])
        test_2d <- paste0(gap_base, "[1,1]")
        has_2d <- tryCatch({ fit$draws(variables = test_2d, format = "matrix"); TRUE },
                           error = function(e) FALSE)
        if (!has_2d) {
          test_1d <- paste0(gap_base, "[1]")
          has_1d <- tryCatch({ fit$draws(variables = test_1d, format = "matrix"); TRUE },
                             error = function(e) FALSE)
          if (has_1d) use_1d_gaps <- TRUE
        }
        if (use_1d_gaps) {
          gap_col_names <- paste0(gap_base, "[", seq_len(n_gaps), "]")
        } else {
          gap_col_names <- as.vector(outer(
            paste0(gap_base, "[", seq_len(n_gaps), ","),
            paste0(seq_len(P_crit), "]"),
            FUN = paste0
          ))
        }
      }
      mid_vars <- paste0(mid_base, "[", seq_len(P_crit), "]")

      all_vars <- c(u_col_names, gap_col_names, mid_vars)
      draws <- tryCatch(
        fit$draws(variables = all_vars, format = "matrix"),
        error = function(e) NULL
      )
      if (is.null(draws)) next
      n_draws <- nrow(draws)

      # u_arr[s, g, j]
      u_arr <- array(draws[, u_col_names], dim = c(n_draws, n_groups, D))

      # gap_arr[s, gap, p]
      gap_arr <- if (n_gaps > 0) {
        if (use_1d_gaps) {
          # 1D gaps: reshape [n_draws, n_gaps] -> [n_draws, n_gaps, 1]
          array(draws[, gap_col_names], dim = c(n_draws, n_gaps, 1))
        } else {
          array(draws[, gap_col_names], dim = c(n_draws, n_gaps, P_crit))
        }
      } else NULL

      # mid_mat[s, p]
      mid_mat <- draws[, mid_vars, drop = FALSE]

      # Per-term X rows (for pop-level mid/gaps in each condition)
      X_per_term <- matrix(0, n_re_terms, P_crit)
      if (n_re_terms == 1) {
        X_per_term[1, ] <- colMeans(X_crit)
      } else if (!is.null(re$term_idx)) {
        for (t in seq_len(n_re_terms)) {
          obs_t <- which(re$term_idx == t)
          if (length(obs_t) > 0) X_per_term[t, ] <- colMeans(X_crit[obs_t, , drop = FALSE])
        }
      } else if (!is.null(re$Z)) {
        for (t in seq_len(n_re_terms)) {
          obs_t <- which(re$Z[, t] != 0)
          if (length(obs_t) > 0) X_per_term[t, ] <- colMeans(X_crit[obs_t, , drop = FALSE])
        }
      } else {
        for (t in seq_len(n_re_terms)) X_per_term[t, ] <- colMeans(X_crit)
      }

      term_labels <- if (!is.null(re$level_names)) {
        gsub("\\(Intercept\\)", "intercept", re$level_names)
      } else {
        if (n_re_terms == 1) "intercept" else paste0("term", seq_len(n_re_terms))
      }

      # Build natural thresholds for every (threshold k, term t):
      # thresh_nat[s, g, (k-1)*n_re_terms + t] using condition-t's pop mid/gaps and
      # the matching RE slice. This preserves the same dimension ordering as the
      # raw (gap-scale) RE correlation matrix.
      D_total <- n_thresh * n_re_terms
      thresh_nat <- array(NA_real_, dim = c(n_draws, n_groups, D_total))

      col_of <- function(k, t) (k - 1L) * n_re_terms + t

      for (t in seq_len(n_re_terms)) {
        xt <- X_per_term[t, ]
        pop_mid_draws <- as.numeric(mid_mat %*% xt)  # [n_draws]
        pop_gap_draws <- if (n_gaps > 0) {
          m <- matrix(gap_arr, n_draws * n_gaps, P_crit) %*% xt
          dim(m) <- c(n_draws, n_gaps)
          m
        } else NULL

        mid_col <- col_of(mid, t)
        thresh_nat[, , mid_col] <- pop_mid_draws + u_arr[, , mid_col]

        if (n_upper > 0) {
          for (k in (mid + 1):n_thresh) {
            gap_idx <- k - mid
            this_col <- col_of(k, t)
            prev_col <- col_of(k - 1L, t)
            eff <- pop_gap_draws[, gap_idx] + u_arr[, , this_col]
            thresh_nat[, , this_col] <- thresh_nat[, , prev_col] + gap_fn(eff)
          }
        }
        if (mid > 1) {
          for (k_down in seq_len(mid - 1)) {
            k <- mid - k_down
            gap_idx <- n_upper + k_down
            this_col <- col_of(k, t)
            next_col <- col_of(k + 1L, t)
            eff <- pop_gap_draws[, gap_idx] + u_arr[, , this_col]
            thresh_nat[, , this_col] <- thresh_nat[, , next_col] - gap_fn(eff)
          }
        }
      }

      # Labels: matches raw gap-scale ordering (k varies slow, t fast)
      dim_labels <- character(D_total)
      for (k in seq_len(n_thresh)) {
        for (t in seq_len(n_re_terms)) {
          dim_labels[col_of(k, t)] <- if (n_re_terms == 1) {
            paste0("thresh", k)
          } else {
            paste0("thresh", k, ":", term_labels[t])
          }
        }
      }

      # Vectorized Pearson correlation across groups, per draw, all pairs.
      row_mean <- apply(thresh_nat, c(1, 3), mean)            # [n_draws, D_total]
      centered <- sweep(thresh_nat, c(1, 3), row_mean, FUN = "-")
      var_d <- apply(centered * centered, c(1, 3), sum)       # [n_draws, D_total]

      n_pairs <- D_total * (D_total - 1L) / 2L
      corr_nat_draws <- matrix(NA_real_, n_draws, n_pairs)
      pair_labels <- character(n_pairs)
      idx <- 1L
      for (i in seq_len(D_total - 1L)) {
        ci <- centered[, , i]
        vi <- var_d[, i]
        for (j in (i + 1L):D_total) {
          cj <- centered[, , j]
          denom <- sqrt(vi * var_d[, j])
          corr_nat_draws[, idx] <- ifelse(denom > 0, rowSums(ci * cj) / denom, NA_real_)
          pair_labels[idx] <- paste0(dim_labels[i], " ~ ", dim_labels[j])
          idx <- idx + 1L
        }
      }

      valid <- complete.cases(corr_nat_draws)
      if (sum(valid) < 10) next
      cv <- corr_nat_draws[valid, , drop = FALSE]

      corr_summary <- data.frame(
        pair = pair_labels,
        estimate = round(colMeans(cv), digits),
        sd = round(apply(cv, 2, sd), digits),
        lower = round(apply(cv, 2, quantile, lower_q), digits),
        upper = round(apply(cv, 2, quantile, upper_q), digits),
        stringsAsFactors = FALSE
      )

      display_crit <- if (crit_name == "criterion2") "criterion2" else "criterion"
      results[[paste0("corr_", display_crit, "_", group_name)]] <- corr_summary
    }
  }

  if (length(results) == 0) return(NULL)
  attr(results, "digits") <- digits
  class(results) <- "threshold_natural_stats"
  results
}

#' @export
print.threshold_natural_stats <- function(x, ...) {
  digits <- attr(x, "digits") %||% 3L
  for (name in names(x)) {
    crit_label <- if (grepl("^corr_criterion2_", name)) "criterion2" else "criterion"
    group <- sub("^corr_criterion2?_", "", name)
    df <- x[[name]]
    cat("\n", crit_label, " | ", group, ":\n", sep = "")
    print(.fmt_summary_tbl(df, digits), row.names = FALSE)
  }
  invisible(x)
}

#' Summarize Smooth Penalty SDs
#' @noRd
summarize_smooth_sds <- function(draws_df, model, lower_col, upper_col, digits) {
  sd <- model$model_data$smooth_data
  if (is.null(sd)) return(NULL)

  rows <- list()
  for (pname in names(sd)) {
    for (sm in sd[[pname]]) {
      n_thresh <- if (!is.null(sm$n_thresh)) sm$n_thresh else 0L
      for (comp in sm$components) {
        for (k in seq_along(comp$Zs_list)) {
          comp_label <- comp$sm_obj$label
          base_label <- paste0("sds(", pname, ", ", comp_label, ")")
          if (length(comp$Zs_list) > 1) {
            base_label <- paste0(base_label, "[", k, "]")
          }
          base_var <- paste0("sds_", pname, "_", comp$san_label, "_", k)

          if (n_thresh > 0) {
            # Per-threshold: one row per threshold
            for (t in seq_len(n_thresh)) {
              var_name <- paste0(base_var, "[", t, "]")
              row_idx <- which(draws_df$variable == var_name)
              if (length(row_idx) == 0) next
              label <- paste0(base_label, " thresh", t)
              r <- draws_df[row_idx, ]
              rows[[length(rows) + 1]] <- data.frame(
                parameter = label,
                mean = round(r$mean, digits),
                sd = round(r$sd, digits),
                lower = round(r[[lower_col]], digits),
                upper = round(r[[upper_col]], digits),
                rhat = round(r$rhat, 3),
                stringsAsFactors = FALSE
              )
            }
          } else {
            var_name <- base_var
            row_idx <- which(draws_df$variable == var_name)
            if (length(row_idx) == 0) next
            r <- draws_df[row_idx, ]
            rows[[length(rows) + 1]] <- data.frame(
              parameter = base_label,
              mean = round(r$mean, digits),
              sd = round(r$sd, digits),
              lower = round(r[[lower_col]], digits),
              upper = round(r[[upper_col]], digits),
              rhat = round(r$rhat, 3),
              stringsAsFactors = FALSE
            )
          }
        }
      }
    }
  }

  if (length(rows) == 0) return(NULL)
  result <- do.call(rbind, rows)
  names(result)[names(result) == "lower"] <- lower_col
  names(result)[names(result) == "upper"] <- upper_col
  result
}


#' @export
print.broc_summary <- function(x, ...) {
  digits <- x$model_info$digits %||% 3L
  cat("\nbayesroc model summary\n")
  cat("======================\n")
  cat("Family:", family_display_name(x$model_info$family), "\n")
  cat("N:", x$model_info$N, "| K:", x$model_info$K, "| Chains:",
      x$model_info$n_chains, "x", x$model_info$n_iter, "\n")

  # Sampler diagnostics banner
  d <- x$diagnostics
  if (!is.null(d)) {
    ndiv  <- if (!is.null(d$num_divergent)) sum(d$num_divergent) else 0L
    ntd   <- if (!is.null(d$num_max_treedepth)) sum(d$num_max_treedepth) else 0L
    ebfmi <- if (!is.null(d$ebfmi)) suppressWarnings(min(d$ebfmi, na.rm = TRUE)) else NA_real_
    cat(sprintf("Sampler: %d divergent | %d max-treedepth | min E-BFMI %s\n",
                as.integer(ndiv), as.integer(ntd),
                if (is.finite(ebfmi)) sprintf("%.2f", ebfmi) else "NA"))
    if (ndiv > 0) {
      cat("  ** WARNING: divergent transitions -- treat inference as unreliable.\n")
    } else if (is.finite(ebfmi) && ebfmi < 0.3) {
      cat("  ** WARNING: low E-BFMI -- sampler may not have explored the posterior.\n")
    }
  }

  cat("\nFIXED EFFECTS\n")
  cat("-------------\n")
  
  # Display name mapping for fixed effect section headers. For aliased
  # families (cumulative / bivariate_cumulative), the criterion thresholds
  # are user-facing "cutpoints" -- relabel accordingly so the printed output
  # matches the names users wrote in their formulas.
  fname <- x$model_info$family
  is_cumulative_like <- fname %in% c("cumulative", "bivariate_cumulative")
  thresh_label <- if (is_cumulative_like) "CUTPOINTS" else "CRITERION"
  thresh2_label <- if (fname == "bivariate_cumulative") "CUTPOINTS2" else "CRITERION2"

  header_map <- c(
    rec = "REC (Recollection d')",
    fam = "FAM (Familiarity d')",
    dprime = if (is_cumulative_like) "MU" else "DPRIME",
    dprime2 = if (fname == "bivariate_cumulative") "MU2" else "DPRIME2",
    sigma = "SIGMA", sigma2 = "SIGMA2",
    sigma_R = "SIGMA_R", sigma_F = "SIGMA_F",
    lambda = "LAMBDA", lambda2 = "LAMBDA2", lambda2_B = "LAMBDA2 (B)",
    rec_crit = "RECOLLECTION CRITERION",
    know_crit = "KNOW CRITERION",
    discrim = if (fname == "bivariate_cumulative") "MU2" else "SOURCE DISCRIMINABILITY",
    discrim_B = "SOURCE DISCRIMINABILITY (B)",
    rho = "RHO", rho_B = "RHO (B)", rho_N = "RHO (NEW)",
    criterion_thresh_mid = paste(thresh_label, "THRESH MID"),
    criterion_gaps       = paste(thresh_label, "GAPS"),
    criterion2_thresh_mid = paste(thresh2_label, "THRESH MID"),
    criterion2_log_gaps   = paste(thresh2_label, "GAPS"),
    dprime_L = "DPRIME (LURE)", sigma_L = "SIGMA (LURE)", lambda_L = "LAMBDA (LURE)"
  )
  for (name in names(x$fixed)) {
    df <- x$fixed[[name]]
    if (!is.null(df) && is.data.frame(df) && nrow(df) > 0) {
      header <- if (name %in% names(header_map)) header_map[[name]] else toupper(gsub("_", " ", name))
      cat("\n", header, sep = "")
      if ("scale" %in% names(df)) {
        cat(" ", df$scale[1], sep = "")
        df$scale <- NULL
      }
      cat(":\n")
      print(.fmt_summary_tbl(df, digits), row.names = FALSE)
    }
  }
  
  pop_label <- if (is_cumulative_like) "POPULATION CUTPOINTS" else "POPULATION THRESHOLDS"
  cat("\n", pop_label, "\n", sep = "")
  cat(strrep("-", nchar(pop_label)), "\n", sep = "")
  
  if (!is.null(x$thresholds) && nrow(x$thresholds) > 0) {
    conditions <- unique(x$thresholds$condition)
    for (cond in conditions) {
      cat("\n", cond, ":\n", sep = "")
      cond_df <- x$thresholds[x$thresholds$condition == cond, ]
      cond_df$condition <- NULL
      print(.fmt_summary_tbl(cond_df, digits), row.names = FALSE)
    }
  } else {
    cat("\n(Skipped - criterion has continuous predictors. Interpret fixed effects directly.)\n")
  }
  
  cat("\nRANDOM EFFECTS (Standard Deviations)\n")
  cat("------------------------------------\n")

  # Helper: print RE SDs grouped by param|group with subheaders
  print_re_sd_grouped <- function(df) {
    if (is.null(df) || nrow(df) == 0) return()
    # Parse "sd(param | group)[level]" or "sd(param | group)" from parameter column
    params <- df$parameter
    # Extract param|group and level
    pg <- sub("^sd\\((.+)\\).*$", "\\1", params)
    level <- ifelse(grepl("\\[", params), sub("^.*\\[(.+)\\]$", "\\1", params), "")
    groups <- unique(pg)
    for (g in groups) {
      mask <- pg == g
      sub_df <- df[mask, , drop = FALSE]
      lvls <- level[mask]
      cat("\n", g, ":\n", sep = "")
      if (all(lvls == "")) {
        # No level index -- single dim RE
        out <- sub_df[, !names(sub_df) %in% "parameter", drop = FALSE]
        print(.fmt_summary_tbl(out, digits), row.names = FALSE)
      } else {
        # Replace parameter column with just the level
        out <- sub_df
        out$parameter <- lvls
        names(out)[names(out) == "parameter"] <- "level"
        print(.fmt_summary_tbl(out, digits), row.names = FALSE)
      }
    }
  }

  print_re_sd_grouped(x$random_sd$individual)

  if (!is.null(x$random_sd$cross_parameter) && nrow(x$random_sd$cross_parameter) > 0) {
    cat("\nCross-parameter correlated:\n")
    print_re_sd_grouped(x$random_sd$cross_parameter)
  }
  
  if (!is.null(x$smooth_sds) && nrow(x$smooth_sds) > 0) {
    cat("\nSMOOTH TERMS (Penalty SDs)\n")
    cat("--------------------------\n")
    print(.fmt_summary_tbl(x$smooth_sds, digits), row.names = FALSE)
  }

  cat("\nCORRELATIONS\n")
  cat("------------\n")

  # Known multi-word parameter names (must not be split on _)
  known_params_corr <- c("rec_crit", "know_crit", "criterion2", "dprime_B",
                          "dprime_L", "lambda_B", "lambda_L", "discrim_B",
                          "sigma_B", "sigma_L", "sigma2_B", "rho_B", "rho_N")
  for (name in names(x$correlations)) {
    df <- x$correlations[[name]]
    if (!is.null(df) && nrow(df) > 0) {
      remainder <- gsub("^corr_", "", name)
      # Find which known param prefix matches, then extract group
      nice_name <- NULL
      for (kp in known_params_corr) {
        prefix <- paste0(kp, "_")
        if (startsWith(remainder, prefix)) {
          group <- sub(paste0("^", gsub("([.|()\\^{}+$*?\\[\\]])", "\\\\\\1", prefix)), "", remainder)
          nice_name <- paste0(kp, " | ", group)
          break
        }
      }
      if (is.null(nice_name)) {
        # Simple params: first segment is param, rest is group
        parts <- strsplit(remainder, "_")[[1]]
        if (length(parts) >= 2) {
          nice_name <- paste0(parts[1], " | ", paste(parts[-1], collapse = "_"))
        } else {
          nice_name <- remainder
        }
      }
      cat("\n", nice_name, ":\n", sep = "")
      print(.fmt_summary_tbl(df, digits), row.names = FALSE)
    }
  }

  if (!is.null(x$threshold_natural)) {
    cat("\nNATURAL-SCALE THRESHOLD CORRELATIONS\n")
    cat("------------------------------------\n")
    print(x$threshold_natural)
  }

  invisible(x)
}
