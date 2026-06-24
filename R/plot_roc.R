#' Plot Model-Implied ROC Curves
#'
#' Generates receiver operating characteristic (ROC) curves from a fitted
#' Bayesian SDT model, with posterior credible bands and optional empirical ROC
#' points, on standard or zROC (probit-transformed) scales.
#'
#' @param fit A `broc_fit` object from [fit_broc()].
#' @param response Which response dimension to plot for bivariate families:
#'   `1` (default) for detection, `2` for source. Ignored for univariate families.
#'   For `response = 2`, two ROC curves are plotted (Source A vs New, Source B vs
#'   New).
#' @param group Optional character vector of column name(s) from the original data
#'   for per-condition ROC curves (faceted). E.g., `group = "condition"`.
#' @param empirical If `TRUE` (default), overlay empirical ROC points computed
#'   from observed data.
#' @param scale `"probability"` (default) for standard ROC, `"z"` for zROC with
#'   qnorm-transformed axes (linear zROC = Gaussian SDT; slope = 1/sigma for UVSDT).
#' @param ndraws Number of posterior draws to use (default 200).
#' @param prob Width of credible interval for the ribbon (default 0.95).
#' @param re_formula `NULL` (default) includes all random effects;
#'   `NA` for population-level ROC only.
#' @param cores Number of cores for parallel computation.
#' @return A ggplot2 object.
#' @export
#'
#' @examples
#' \dontrun{
#' # Standard ROC
#' plot_roc_curve(fit)
#'
#' # zROC (probit-transformed)
#' plot_roc_curve(fit, scale = "z")
#'
#' # Bivariate: source dimension ROC
#' plot_roc_curve(fit_bv, response = 2)
#'
#' # Per-condition ROC curves
#' plot_roc_curve(fit, group = "condition")
#' }
plot_roc_curve <- function(fit, response = 1, group = NULL, empirical = TRUE,
                           scale = c("probability", "z"), ndraws = 200,
                           prob = 0.95, re_formula = NULL, cores = 1) {

  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required for plot_roc_curve().")
  }

  scale <- match.arg(scale)

  model <- attr(fit, "broc_model")
  if (is.null(model)) {
    stop("Fit object must have 'broc_model' attribute. Use fit_broc() to fit the model.")
  }

  family_name <- model$family$family
  is_bivariate <- family_name %in% c("bivariate_sdt", "bivariate_dp", "vrdp2d")

  # --- Validation ---
  unsupported <- c("cumulative", "bivariate_cumulative")
  if (family_name %in% unsupported) {
    stop("plot_roc_curve() is not supported for '", family_display_name(family_name),
         "'. Cumulative ordinal models have no SDT structure and no meaningful ROC curve.")
  }
  if (!is_bivariate && response != 1) {
    stop("response = 2 is only valid for bivariate families (bivariate_gaussian, bivariate_dp, vrdp2d).")
  }

  # Determine K for the plotted dimension
  if (is_bivariate && response == 2) {
    K <- model$model_data$K2
  } else {
    K <- model$model_data$K
  }
  if (K < 3) {
    stop("ROC curves require ordinal responses with K >= 3 categories. ",
         "This model has K = ", K, ".")
  }

  # --- Get posterior category probabilities ---
  probs_raw <- predict(fit, type = "response", ndraws = ndraws,
                       summary = FALSE, re_formula = re_formula, cores = cores)
  S <- dim(probs_raw)[1]
  N <- dim(probs_raw)[2]

  # Marginalize bivariate to selected dimension
  # Note: varying_source_criteria is handled transparently -- the joint probs
  # from predict_broc already incorporate per-detection-level source criteria,
  # and marginalization correctly averages over them.
  if (is_bivariate) {
    if (response == 1) {
      probs <- apply(probs_raw, c(1, 2, 3), sum)  # S x N x K1
    } else {
      probs <- apply(probs_raw, c(1, 2, 4), sum)  # S x N x K2
    }
  } else {
    probs <- probs_raw
  }

  # --- Counts ---
  use_counts <- isTRUE(model$model_data$has_counts)
  counts <- if (use_counts) model$stan_data$counts else rep(1L, N)

  # --- Signal/noise comparisons ---
  comparisons <- .roc_signal_indicator(model, family_name, response)

  # --- Grouping ---
  group_vec <- NULL
  group_levels <- NULL
  if (!is.null(group)) {
    if (is.null(model$data)) {
      stop("Original data not stored on model. Cannot use 'group' argument.")
    }
    if (length(group) == 1) {
      group_vec <- model$data[[group]]
    } else {
      group_vec <- interaction(model$data[group], sep = " : ")
    }
    if (is.null(group_vec)) {
      stop("Group variable '", paste(group, collapse = "', '"), "' not found in data.")
    }
    group_levels <- levels(as.factor(group_vec))
  }

  # --- Observed responses (for empirical ROC) ---
  if (is_bivariate && response == 2) {
    y_obs <- model$stan_data$y2
  } else {
    y_obs <- model$stan_data$y
  }

  # --- Compute ROC for each comparison x group ---
  all_model_dfs <- list()
  all_emp_dfs <- list()

  for (comp in comparisons) {
    sig_idx <- comp$signal_idx
    noi_idx <- comp$noise_idx
    curve_label <- comp$label
    reverse <- isTRUE(comp$reverse)

    if (is.null(group_vec)) {
      # No grouping: single ROC per comparison
      model_roc <- .compute_model_roc(probs, sig_idx, noi_idx, K, counts, S, reverse)
      df <- .summarize_roc(model_roc, prob)
      df$curve <- curve_label
      all_model_dfs[[length(all_model_dfs) + 1]] <- df

      if (empirical) {
        emp_df <- .compute_empirical_roc(y_obs, sig_idx, noi_idx, K, counts, reverse)
        emp_df$curve <- curve_label
        all_emp_dfs[[length(all_emp_dfs) + 1]] <- emp_df
      }
    } else {
      # Per-group ROC. If group variable is an encoding factor, noise items
      # don't have that variable -- use ALL noise as shared FAR baseline.
      # Otherwise, each group has its own signal AND noise observations.
      enc_vars <- model$model_data$encoding_vars
      group_is_encoding <- any(group %in% enc_vars)

      for (g in group_levels) {
        g_mask <- group_vec == g
        g_sig <- intersect(sig_idx, which(g_mask))

        if (length(g_sig) == 0) next

        if (group_is_encoding) {
          g_noi <- noi_idx
        } else {
          g_noi <- intersect(noi_idx, which(g_mask))
          if (length(g_noi) == 0) next
        }

        model_roc <- .compute_model_roc(probs, g_sig, g_noi, K, counts, S, reverse)
        df <- .summarize_roc(model_roc, prob)
        df$curve <- curve_label
        df$group <- g
        all_model_dfs[[length(all_model_dfs) + 1]] <- df

        if (empirical) {
          emp_df <- .compute_empirical_roc(y_obs, g_sig, g_noi, K, counts, reverse)
          emp_df$curve <- curve_label
          emp_df$group <- g
          all_emp_dfs[[length(all_emp_dfs) + 1]] <- emp_df
        }
      }
    }
  }

  if (length(all_model_dfs) == 0) {
    stop("No ROC curves could be computed. Check that each group has both ",
         "signal and noise observations.")
  }
  df_model <- do.call(rbind, all_model_dfs)
  df_emp <- if (length(all_emp_dfs) > 0) do.call(rbind, all_emp_dfs) else NULL
  rownames(df_model) <- NULL
  if (!is.null(df_emp)) rownames(df_emp) <- NULL

  # Ensure is_anchor is logical after rbind
  df_model$is_anchor <- as.logical(df_model$is_anchor)

  # --- zROC transform ---
  if (scale == "z") {
    # Remove anchor points (0,0) and (1,1) -- they map to +/-Inf on z-scale
    df_model <- df_model[!df_model$is_anchor, ]
    df_model <- .zroc_transform(df_model)
    if (!is.null(df_emp)) df_emp <- .zroc_transform_emp(df_emp)
  }

  # --- Build plot ---
  has_multiple_curves <- length(comparisons) > 1
  has_groups <- !is.null(group_vec)

  p <- ggplot2::ggplot()

  # Diagonal reference line
  p <- p + ggplot2::geom_abline(intercept = 0, slope = 1, linetype = "dashed",
                                 color = "gray50", linewidth = 0.5)

  if (has_multiple_curves) {
    # Multiple curves: color by curve label
    p <- p +
      ggplot2::geom_ribbon(
        data = df_model,
        ggplot2::aes(x = .data$FAR, ymin = .data$HR_lower, ymax = .data$HR_upper,
                     fill = .data$curve),
        alpha = 0.15
      ) +
      ggplot2::geom_line(
        data = df_model,
        ggplot2::aes(x = .data$FAR, y = .data$HR, color = .data$curve),
        linewidth = 1
      ) +
      ggplot2::geom_point(
        data = df_model[which(!df_model$is_anchor), ],
        ggplot2::aes(x = .data$FAR, y = .data$HR, color = .data$curve),
        size = 2
      )

    if (empirical && !is.null(df_emp)) {
      p <- p + ggplot2::geom_point(
        data = df_emp,
        ggplot2::aes(x = .data$FAR, y = .data$HR, color = .data$curve),
        size = 3, shape = 17
      )
    }
  } else {
    # Single curve: fixed color
    p <- p +
      ggplot2::geom_ribbon(
        data = df_model,
        ggplot2::aes(x = .data$FAR, ymin = .data$HR_lower, ymax = .data$HR_upper),
        fill = "#3366CC", alpha = 0.2
      ) +
      ggplot2::geom_line(
        data = df_model,
        ggplot2::aes(x = .data$FAR, y = .data$HR),
        color = "#3366CC", linewidth = 1
      ) +
      ggplot2::geom_point(
        data = df_model[which(!df_model$is_anchor), ],
        ggplot2::aes(x = .data$FAR, y = .data$HR),
        color = "#3366CC", size = 2
      )

    if (empirical && !is.null(df_emp)) {
      p <- p + ggplot2::geom_point(
        data = df_emp,
        ggplot2::aes(x = .data$FAR, y = .data$HR),
        color = "gray30", size = 3, shape = 17
      )
    }
  }

  # Facet by group if provided
  if (has_groups) {
    p <- p + ggplot2::facet_wrap(~ group)
  }

  # Axes and labels
  if (scale == "probability") {
    p <- p + ggplot2::coord_fixed(ratio = 1, xlim = c(0, 1), ylim = c(0, 1))
    x_lab <- "False Alarm Rate"
    y_lab <- "Hit Rate"
  } else {
    # Symmetric axes centered on 0 so the diagonal is visually centered
    z_range <- max(abs(c(df_model$FAR, df_model$HR,
                         df_model$HR_lower, df_model$HR_upper)), na.rm = TRUE)
    z_lim <- c(-z_range, z_range) * 1.05
    p <- p + ggplot2::coord_fixed(ratio = 1, xlim = z_lim, ylim = z_lim)
    x_lab <- "z(False Alarm Rate)"
    y_lab <- "z(Hit Rate)"
  }

  dim_label <- if (is_bivariate) {
    if (response == 1) " (Detection)" else " (Source)"
  } else ""

  roc_label <- if (scale == "z") "zROC" else "ROC Curve"

  p <- p +
    ggplot2::labs(
      x = x_lab, y = y_lab,
      title = paste0(roc_label, dim_label),
      subtitle = paste0("Posterior mean + ", round(prob * 100), "% CI")
    ) +
    broc_pp_theme()

  p
}


# =============================================================================
# Internal Helpers
# =============================================================================

#' Determine signal/noise observation indices for ROC
#' @return List of comparisons, each with signal_idx, noise_idx, label
#' @noRd
.roc_signal_indicator <- function(model, family_name, response) {
  is_bivariate <- family_name %in% c("bivariate_sdt", "bivariate_dp", "vrdp2d")

  if (family_name %in% c("evsdt", "uvsdt", "dpsdt", "mixture", "cdp")) {
    is_old <- model$stan_data$is_old
    return(list(list(
      signal_idx = which(is_old == 1),
      noise_idx = which(is_old == 0),
      label = "ROC"
    )))
  }

  if (family_name == "source_mixture") {
    source_vec <- model$stan_data$source
    return(list(list(
      signal_idx = which(source_vec == 2L),
      noise_idx = which(source_vec == 1L),
      label = "Source Discrimination"
    )))
  }

  if (is_bivariate && response == 1) {
    item_type <- model$stan_data$item_type
    return(list(list(
      signal_idx = which(item_type >= 2L),
      noise_idx = which(item_type == 1L),
      label = "Detection"
    )))
  }

  if (is_bivariate && response == 2) {
    item_type <- model$stan_data$item_type
    new_idx <- which(item_type == 1L)
    src_a_idx <- which(item_type == 2L)
    src_b_idx <- which(item_type == 3L)
    # Source A has positive discrim -> high source responses -> standard P(Y >= k).
    # Source B has negative discrim -> low source responses -> flip scale.
    return(list(
      list(signal_idx = src_a_idx, noise_idx = new_idx, label = "Source A", reverse = FALSE),
      list(signal_idx = src_b_idx, noise_idx = new_idx, label = "Source B", reverse = TRUE)
    ))
  }

  stop("Unsupported family/response combination for ROC: ",
       family_display_name(family_name), ", response = ", response)
}


#' Compute model-implied ROC curves from posterior draws
#' @return List with HR (S x n_points) and FAR (S x n_points) matrices
#' @noRd
.compute_model_roc <- function(probs, sig_idx, noi_idx, K, counts, S, reverse = FALSE) {
  n_points <- K - 1

  # Weighted mean P(Y=k) per group per draw
  sig_wt <- counts[sig_idx] / sum(counts[sig_idx])
  noi_wt <- counts[noi_idx] / sum(counts[noi_idx])

  # Compute weighted average category probs: S x K
  sig_mean <- matrix(0, S, K)
  noi_mean <- matrix(0, S, K)
  for (s in seq_len(S)) {
    sp <- matrix(probs[s, sig_idx, ], nrow = length(sig_idx), ncol = K)
    np <- matrix(probs[s, noi_idx, ], nrow = length(noi_idx), ncol = K)
    sig_mean[s, ] <- colSums(sp * sig_wt)
    noi_mean[s, ] <- colSums(np * noi_wt)
  }

  # If reverse, flip the probability columns (mirror the response scale)
  # so that P(Y >= k) on the flipped scale = P(Y_orig <= K+1-k)
  if (reverse) {
    sig_mean <- sig_mean[, K:1, drop = FALSE]
    noi_mean <- noi_mean[, K:1, drop = FALSE]
  }

  # Cumulative from right: P(Y >= k) for k = 2, ..., K
  HR <- matrix(0, S, n_points)
  FAR <- matrix(0, S, n_points)
  for (j in seq_len(n_points)) {
    k <- j + 1
    HR[, j] <- rowSums(sig_mean[, k:K, drop = FALSE])
    FAR[, j] <- rowSums(noi_mean[, k:K, drop = FALSE])
  }

  list(HR = HR, FAR = FAR)
}


#' Summarize posterior ROC into data frame with mean + CI
#' @noRd
.summarize_roc <- function(model_roc, prob) {
  probs_lower <- (1 - prob) / 2
  probs_upper <- 1 - probs_lower

  df <- data.frame(
    FAR = colMeans(model_roc$FAR),
    HR = colMeans(model_roc$HR),
    HR_lower = apply(model_roc$HR, 2, quantile, probs = probs_lower),
    HR_upper = apply(model_roc$HR, 2, quantile, probs = probs_upper),
    is_anchor = FALSE
  )

  # Add anchor points (0,0) and (1,1)
  anchors <- data.frame(
    FAR = c(0, 1), HR = c(0, 1),
    HR_lower = c(0, 1), HR_upper = c(0, 1),
    is_anchor = TRUE
  )
  df <- rbind(anchors[1, ], df, anchors[2, ])
  df <- df[order(df$FAR), ]
  rownames(df) <- NULL
  df
}


#' Compute empirical ROC from observed data
#' @noRd
.compute_empirical_roc <- function(y_obs, sig_idx, noi_idx, K, counts, reverse = FALSE) {
  n_points <- K - 1
  emp_HR <- numeric(n_points)
  emp_FAR <- numeric(n_points)

  sig_total <- sum(counts[sig_idx])
  noi_total <- sum(counts[noi_idx])

  # If reverse, flip the response scale so "high" means Source A direction
  y_sig <- y_obs[sig_idx]
  y_noi <- y_obs[noi_idx]
  if (reverse) {
    y_sig <- K + 1L - y_sig
    y_noi <- K + 1L - y_noi
  }

  # P(Y >= k) for k = 2, ..., K (on possibly flipped scale)
  for (j in seq_len(n_points)) {
    k <- j + 1
    emp_HR[j] <- sum(counts[sig_idx][y_sig >= k]) / sig_total
    emp_FAR[j] <- sum(counts[noi_idx][y_noi >= k]) / noi_total
  }

  data.frame(FAR = emp_FAR, HR = emp_HR)
}


#' Transform ROC data frame to zROC scale
#' @noRd
.zroc_transform <- function(df) {
  # Clip to avoid Inf from qnorm(0) or qnorm(1)
  clip <- function(x) pmin(pmax(x, 0.001), 0.999)
  df$FAR <- qnorm(clip(df$FAR))
  df$HR <- qnorm(clip(df$HR))
  df$HR_lower <- qnorm(clip(df$HR_lower))
  df$HR_upper <- qnorm(clip(df$HR_upper))
  df
}


#' Transform empirical ROC to zROC scale
#' @noRd
.zroc_transform_emp <- function(df) {
  clip <- function(x) pmin(pmax(x, 0.001), 0.999)
  df$FAR <- qnorm(clip(df$FAR))
  df$HR <- qnorm(clip(df$HR))
  df
}
