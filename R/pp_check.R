# =============================================================================
# Main Function
# =============================================================================

#' Posterior Predictive Check for SDT Models
#'
#' Compare observed response distributions against draws from the posterior
#' predictive distribution.
#'
#' @param object A `broc_fit` object from [fit_broc()].
#' @param type Plot type. `"bars"`: observed vs predicted counts per response
#'   category, with predictive intervals. `"bars_grouped"`: the same, split by
#'   `group`. `"rootogram"`: a hanging rootogram, emphasizing fit in low-count
#'   categories. `"stat"`: posterior distribution of a summary statistic
#'   (`stat_fun`) against its observed value. `"dens_overlay"`: observed density
#'   with overlaid predictive densities.
#' @param group Column name (from the original data) to facet by; `NULL` for no
#'   grouping. `type = "bars_grouped"` groups by this column.
#' @param ndraws Number of replicated datasets (default 200)
#' @param prob CI width for intervals on predicted counts (default 0.95)
#' @param stat_fun Summary statistic function for `type = "stat"` (default: mean)
#' @param re_formula Which random effects to include in prediction. `NULL`
#'   (default) includes all random effects; `NA` excludes them, giving
#'   population-level predictions.
#' @param response For bivariate families (bivariate_gaussian, vrdp2d): which dimension
#'   to check (1 = item/detection, 2 = source/discrimination). Default 1.
#' @param cores Number of cores used for posterior prediction (default 1).
#' @param ... Additional arguments (unused).
#' @return A ggplot object.
#' @examples
#' \dontrun{
#' fit <- fit_broc(model)
#' pp_check(fit, type = "bars")
#' pp_check(fit, type = "bars_grouped", group = "condition")
#' pp_check(fit, type = "rootogram")
#' }
#' @importFrom bayesplot pp_check
#' @method pp_check broc_fit
#' @export
pp_check.broc_fit <- function(object, type = c("bars", "bars_grouped", "dens_overlay", "stat", "rootogram"),
                              group = NULL, ndraws = 200, prob = 0.95,
                              stat_fun = mean, re_formula = NULL, response = 1,
                              cores = 1, ...) {
  fit <- object

  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required for pp_check().")
  }

  type <- match.arg(type)

  model <- attr(fit, "broc_model")
  if (is.null(model)) {
    stop("Fit object must have 'broc_model' attribute. Use fit_broc() to fit the model.")
  }

  family_name <- model$family$family
  is_bivariate <- family_name %in% c("bivariate_sdt", "bivariate_dp", "vrdp2d", "bivariate_cumulative")
  is_cdp <- family_name == "cdp"

  old_mask <- NULL  # used for CDP response=2 filtering

  # Handle CDP response=2: R/K(/G) dimension
  if (is_cdp && response == 2) {
    probs_all <- predict(fit, type = "response", ndraws = ndraws,
                              summary = FALSE, re_formula = re_formula, cores = cores,
                              response = 2)
    # probs_all is S x N x n_rkg -- filter to trials with old responses only
    n_rkg <- if (!is.null(model$family$n_rkg)) model$family$n_rkg else 2L
    K <- n_rkg
    rk_obs <- model$stan_data$rk
    old_levels <- model$family$old_levels
    y_conf <- model$stan_data$y
    # Only keep trials where observed confidence was an "old" response
    old_mask <- y_conf %in% old_levels & !is.na(rk_obs)
    probs <- probs_all[, old_mask, , drop = FALSE]
    y_obs <- rk_obs[old_mask]
  } else {
    # Get posterior predictive probabilities (not single-sample predictions)
    # This gives an S x N x K array (or S x N x K1 x K2 for bivariate)
    probs_raw <- predict(fit, type = "response", ndraws = ndraws,
                              summary = FALSE, re_formula = re_formula, cores = cores)

    # Handle bivariate: marginalize 4D probs to selected response dimension
    if (is_bivariate) {
      if (response == 1) {
        K <- model$model_data$K   # K1
        y_obs <- model$stan_data$y
        # Sum over K2 (dim 4) to get marginal P(Y1=j)
        probs <- apply(probs_raw, c(1, 2, 3), sum)  # S x N x K1
      } else {
        K <- model$model_data$K2
        y_obs <- model$stan_data$y2
        # Sum over K1 (dim 3) to get marginal P(Y2=k)
        probs <- apply(probs_raw, c(1, 2, 4), sum)  # S x N x K2
      }
    } else {
      K <- model$model_data$K
      y_obs <- model$stan_data$y
      probs <- probs_raw  # S x N x K
    }
  }

  S <- dim(probs)[1]
  N <- dim(probs)[2]

  # Counts support
  use_counts <- isTRUE(model$model_data$has_counts)
  counts_all <- if (use_counts) model$stan_data$counts else rep(1L, length(model$stan_data$y))
  # For CDP response=2, filter counts to old-response trials
  if (!is.null(old_mask)) {
    counts <- counts_all[old_mask]
  } else {
    counts <- counts_all
  }

  # Observed counts per category
  obs_counts <- integer(K)
  for (k in seq_len(K)) obs_counts[k] <- sum(counts[y_obs == k])

  # Auto-detect grouping for bars_grouped
  if (type == "bars_grouped" && is.null(group)) {
    if (is_bivariate && !is.null(model$parsed$response$item_type)) {
      group <- model$parsed$response$item_type
    } else if (is_cdp && !is.null(model$parsed$response$is_old)) {
      group <- model$parsed$response$is_old
    } else {
      is_sdt <- !family_name %in% c("cumulative")
      if (is_sdt && !is.null(model$parsed$response$is_old)) {
        group <- model$parsed$response$is_old
      } else if (family_name == "source_mixture" && !is.null(model$parsed$response$source)) {
        group <- model$parsed$response$source
      }
    }
  }

  # Get grouping vector
  group_vec <- NULL
  if (!is.null(group)) {
    if (is.null(model$data)) {
      stop("Original data not stored on model. Cannot use 'group' argument.")
    }
    missing_vars <- group[!group %in% names(model$data)]
    if (length(missing_vars) > 0) {
      stop("Group variable(s) '", paste(missing_vars, collapse = "', '"),
           "' not found in original data. ",
           "Available columns: ", paste(names(model$data), collapse = ", "))
    }
    if (length(group) == 1) {
      group_vec <- model$data[[group]]
    } else {
      # Multiple grouping variables: combine into single interaction factor
      group_vec <- interaction(model$data[group], sep = " : ")
    }
    # CDP response=2 filters probs/counts/y_obs to old-response trials only;
    # the grouping vector must be filtered to match, otherwise indices fall
    # out of alignment with the post-filter N.
    if (!is.null(old_mask)) {
      group_vec <- group_vec[old_mask]
    }
  }

  # --- Grouped bar chart: compute per-group pred counts via multinomial sampling ---
  if (type == "bars_grouped") {
    if (is.null(group_vec)) {
      # No group available: fall back to ungrouped bars
      pred_counts <- .pp_sample_pred_counts(probs, counts, S, N, K)
      return(pp_bars(obs_counts, pred_counts, K, prob))
    }

    group_levels <- if (is.factor(group_vec)) levels(group_vec) else sort(unique(group_vec))
    n_groups <- length(group_levels)
    group_assign <- match(as.character(group_vec), as.character(group_levels))

    # Per-group observed counts: n_groups x K
    group_obs <- matrix(0L, nrow = n_groups, ncol = K)
    for (g in seq_len(n_groups)) {
      idx <- which(group_assign == g)
      for (k in seq_len(K)) group_obs[g, k] <- sum(counts[idx][y_obs[idx] == k])
    }

    # Per-group predicted counts via multinomial sampling: S x n_groups x K.
    # Vectorized: draw one category per trial, then tabulate by (group, category).
    row_of_trial   <- rep.int(seq_len(N), counts)   # length T = sum(counts)
    group_of_trial <- group_assign[row_of_trial]
    Tt <- length(row_of_trial)
    group_pred <- array(0L, dim = c(S, n_groups, K))
    for (s in seq_len(S)) {
      P <- matrix(probs[s, row_of_trial, ], nrow = Tt, ncol = K)
      cat <- .pp_draw_categories(P, K)
      # combined (group, category) bin index, category varying fastest
      tab <- tabulate((group_of_trial - 1L) * K + cat, nbins = n_groups * K)
      group_pred[s, , ] <- matrix(tab, nrow = n_groups, ncol = K, byrow = TRUE)
    }

    return(pp_bars_grouped(group_obs, group_pred, K, group_levels, group, prob))
  }

  # --- For all other types: compute total pred counts (S x K) ---
  pred_counts <- .pp_sample_pred_counts(probs, counts, S, N, K)

  switch(type,
    bars         = pp_bars(obs_counts, pred_counts, K, prob),
    dens_overlay = pp_dens_overlay(obs_counts, pred_counts, K),
    stat         = pp_stat(obs_counts, pred_counts, K, stat_fun),
    rootogram    = pp_rootogram(obs_counts, pred_counts, K, prob)
  )
}


# =============================================================================
# Multinomial Sampling Helper
# =============================================================================

#' Sample total predicted counts per category via multinomial sampling
#'
#' For each posterior draw s and data row n, draws counts[n] responses from
#' the categorical distribution probs[s,n,], then sums across rows to get
#' total predicted counts per category.
#'
#' @param probs S x N x K array of category probabilities
#' @param counts Integer vector of length N (trial counts per row)
#' @param S Number of posterior draws
#' @param N Number of data rows
#' @param K Number of response categories
#' @return S x K matrix of total predicted counts per category per draw
#' @noRd
.pp_sample_pred_counts <- function(probs, counts, S, N, K) {
  row_of_trial <- rep.int(seq_len(N), counts)   # length T = sum(counts)
  Tt <- length(row_of_trial)
  pred_counts <- matrix(0L, nrow = S, ncol = K)
  for (s in seq_len(S)) {
    P <- matrix(probs[s, row_of_trial, ], nrow = Tt, ncol = K)
    pred_counts[s, ] <- tabulate(.pp_draw_categories(P, K), nbins = K)
  }
  pred_counts
}

#' Vectorized categorical draw from a matrix of per-row probabilities
#'
#' A size-c multinomial is the sum of c categoricals, so total predicted counts
#' come from drawing one category per trial. `P` holds one (possibly unnormalized
#' or degenerate) probability row per trial; this draws a category for every row
#' at once via inverse-CDF sampling, replacing a per-row rmultinom loop.
#'
#' @param P T x K matrix of per-row category probabilities
#' @param K Number of response categories
#' @return Integer vector of length T of sampled categories (1..K)
#' @noRd
.pp_draw_categories <- function(P, K) {
  rs <- rowSums(P)
  bad <- !is.finite(rs) | rs <= 0
  P <- P / rs
  if (any(bad)) P[bad, ] <- 1 / K
  cumP <- P
  for (k in 2:K) cumP[, k] <- cumP[, k - 1] + P[, k]
  u <- runif(nrow(P))
  cat <- rowSums(cumP < u) + 1L
  cat[cat > K] <- K
  cat
}


# =============================================================================
# Plot Type: bars
# =============================================================================

#' Bar chart of observed vs predicted response distribution
#' @param obs_counts K-length vector of observed counts per category
#' @param pred_counts S x K matrix of predicted counts per category per draw
#' @param K Number of response categories
#' @param prob CI width
#' @noRd
pp_bars <- function(obs_counts, pred_counts, K, prob) {
  probs_lower <- (1 - prob) / 2
  probs_upper <- 1 - probs_lower

  df_obs <- data.frame(
    category = factor(seq_len(K)),
    count = obs_counts,
    type = "Observed"
  )

  df_pred <- data.frame(
    category = factor(seq_len(K)),
    median = apply(pred_counts, 2, median),
    lower = apply(pred_counts, 2, quantile, probs = probs_lower),
    upper = apply(pred_counts, 2, quantile, probs = probs_upper)
  )

  ggplot2::ggplot() +
    ggplot2::geom_col(
      data = df_obs,
      ggplot2::aes(x = .data$category, y = .data$count),
      fill = "gray70", color = "gray40", width = 0.7
    ) +
    ggplot2::geom_pointrange(
      data = df_pred,
      ggplot2::aes(x = .data$category, y = .data$median,
                   ymin = .data$lower, ymax = .data$upper),
      color = "#3366CC", size = 0.8, linewidth = 0.8
    ) +
    ggplot2::labs(
      x = "Response Category", y = "Count",
      title = "Posterior Predictive Check",
      subtitle = "Bars = observed, points = predicted (median + CI)"
    ) +
    broc_pp_theme()
}


# =============================================================================
# Plot Type: bars_grouped
# =============================================================================

#' Faceted bar chart by group
#' @param group_obs n_groups x K matrix of observed counts per group and category
#' @param group_pred S x n_groups x K array of predicted counts
#' @param K Number of response categories
#' @param group_levels Character vector of group level names
#' @param group_name Name of the grouping variable
#' @param prob CI width
#' @noRd
pp_bars_grouped <- function(group_obs, group_pred, K, group_levels, group_name, prob) {
  n_groups <- length(group_levels)
  S <- dim(group_pred)[1]

  probs_lower <- (1 - prob) / 2
  probs_upper <- 1 - probs_lower

  obs_list <- list()
  pred_list <- list()

  for (g in seq_len(n_groups)) {
    gl <- group_levels[g]

    obs_list[[g]] <- data.frame(
      category = factor(seq_len(K)),
      count = group_obs[g, ],
      group = gl
    )

    pred_cts <- group_pred[, g, ]  # S x K
    if (S == 1L) pred_cts <- matrix(pred_cts, nrow = 1)

    pred_list[[g]] <- data.frame(
      category = factor(seq_len(K)),
      median = apply(pred_cts, 2, median),
      lower = apply(pred_cts, 2, quantile, probs = probs_lower),
      upper = apply(pred_cts, 2, quantile, probs = probs_upper),
      group = gl
    )
  }

  df_obs <- do.call(rbind, obs_list)
  df_pred <- do.call(rbind, pred_list)
  rownames(df_obs) <- rownames(df_pred) <- NULL

  df_obs$group <- factor(df_obs$group, levels = group_levels)
  df_pred$group <- factor(df_pred$group, levels = group_levels)

  ggplot2::ggplot() +
    ggplot2::geom_col(
      data = df_obs,
      ggplot2::aes(x = .data$category, y = .data$count),
      fill = "gray70", color = "gray40", width = 0.7
    ) +
    ggplot2::geom_pointrange(
      data = df_pred,
      ggplot2::aes(x = .data$category, y = .data$median,
                   ymin = .data$lower, ymax = .data$upper),
      color = "#3366CC", size = 0.6, linewidth = 0.7
    ) +
    ggplot2::facet_wrap(~ group, scales = "free_y") +
    ggplot2::labs(
      x = "Response Category", y = "Count",
      title = "Posterior Predictive Check (grouped)",
      subtitle = paste0("Grouped by: ", if (!is.null(group_name)) group_name else "group")
    ) +
    broc_pp_theme()
}


# =============================================================================
# Plot Type: dens_overlay
# =============================================================================

#' Overlay density of replicated datasets on observed
#' @param obs_counts K-length vector of observed counts per category
#' @param pred_counts S x K matrix of predicted counts per category per draw
#' @param K Number of response categories
#' @noRd
pp_dens_overlay <- function(obs_counts, pred_counts, K) {
  S <- nrow(pred_counts)
  total_obs <- sum(obs_counts)

  # Observed proportions
  obs_props <- obs_counts / total_obs

  # Replicated proportions (subset for performance)
  n_show <- min(S, 50L)
  show_idx <- if (n_show < S) sample.int(S, n_show) else seq_len(S)

  rep_list <- vector("list", n_show)
  for (i in seq_along(show_idx)) {
    s <- show_idx[i]
    total_pred <- sum(pred_counts[s, ])
    props <- if (total_pred > 0) pred_counts[s, ] / total_pred else rep(1 / K, K)
    rep_list[[i]] <- data.frame(
      category = seq_len(K),
      proportion = props,
      draw = i
    )
  }

  df_rep <- do.call(rbind, rep_list)
  df_obs <- data.frame(category = seq_len(K), proportion = obs_props)

  ggplot2::ggplot() +
    ggplot2::geom_line(
      data = df_rep,
      ggplot2::aes(x = .data$category, y = .data$proportion, group = .data$draw),
      color = "#3366CC", alpha = 0.15, linewidth = 0.4
    ) +
    ggplot2::geom_line(
      data = df_obs,
      ggplot2::aes(x = .data$category, y = .data$proportion),
      color = "black", linewidth = 1.2
    ) +
    ggplot2::geom_point(
      data = df_obs,
      ggplot2::aes(x = .data$category, y = .data$proportion),
      color = "black", size = 2.5
    ) +
    ggplot2::scale_x_continuous(breaks = seq_len(K)) +
    ggplot2::labs(
      x = "Response Category", y = "Proportion",
      title = "Posterior Predictive Check (density overlay)",
      subtitle = "Thin lines = replicated datasets, thick line = observed"
    ) +
    broc_pp_theme()
}


# =============================================================================
# Plot Type: stat
# =============================================================================

#' Compare summary statistic across replicated vs observed
#' @param obs_counts K-length vector of observed counts per category
#' @param pred_counts S x K matrix of predicted counts per category per draw
#' @param K Number of response categories
#' @param stat_fun Summary statistic function (applied to expanded response vector)
#' @noRd
pp_stat <- function(obs_counts, pred_counts, K, stat_fun) {
  S <- nrow(pred_counts)

  # Observed statistic: expand counts to individual responses
  obs_stat <- stat_fun(rep(seq_len(K), obs_counts))

  # Replicated statistics: expand predicted counts to individual responses
  rep_stats <- numeric(S)
  for (s in seq_len(S)) {
    rep_stats[s] <- stat_fun(rep(seq_len(K), pred_counts[s, ]))
  }

  df_rep <- data.frame(stat = rep_stats)

  ggplot2::ggplot(df_rep, ggplot2::aes(x = .data$stat)) +
    ggplot2::geom_histogram(
      fill = "#3366CC", color = "white", alpha = 0.7, bins = 30
    ) +
    ggplot2::geom_vline(
      xintercept = obs_stat, color = "black", linewidth = 1.2, linetype = "dashed"
    ) +
    ggplot2::labs(
      x = "Test Statistic", y = "Count",
      title = "Posterior Predictive Check (statistic)",
      subtitle = "Histogram = replicated, dashed line = observed"
    ) +
    broc_pp_theme()
}


# =============================================================================
# Plot Type: rootogram
# =============================================================================

#' Hanging rootogram
#' @param obs_counts K-length vector of observed counts per category
#' @param pred_counts S x K matrix of predicted counts per category per draw
#' @param K Number of response categories
#' @param prob CI width (unused, kept for interface consistency)
#' @noRd
pp_rootogram <- function(obs_counts, pred_counts, K, prob) {
  exp_counts <- colMeans(pred_counts)

  # Hanging rootogram: bars hang from sqrt(expected)
  sqrt_obs <- sqrt(obs_counts)
  sqrt_exp <- sqrt(exp_counts)

  df <- data.frame(
    x = seq_len(K),
    ymin = sqrt_exp - sqrt_obs,
    ymax = sqrt_exp
  )

  ggplot2::ggplot(df) +
    ggplot2::geom_rect(
      ggplot2::aes(
        xmin = .data$x - 0.35, xmax = .data$x + 0.35,
        ymin = .data$ymin, ymax = .data$ymax
      ),
      fill = "#3366CC", alpha = 0.7
    ) +
    ggplot2::geom_hline(yintercept = 0, linewidth = 0.5, linetype = "dashed") +
    ggplot2::scale_x_continuous(breaks = seq_len(K), labels = seq_len(K)) +
    ggplot2::labs(
      x = "Response Category", y = expression(sqrt("Count")),
      title = "Hanging Rootogram",
      subtitle = "Bar tops = sqrt(expected); bar bottoms deviating from zero indicate misfit"
    ) +
    broc_pp_theme()
}


# =============================================================================
# Theme
# =============================================================================

#' Standard theme for pp_check plots (matches plot.R conventions)
#' @noRd
broc_pp_theme <- function() {
  ggplot2::theme_minimal(base_size = 11) +
    ggplot2::theme(
      panel.grid.minor = ggplot2::element_blank(),
      panel.grid.major.x = ggplot2::element_blank(),
      strip.text = ggplot2::element_text(face = "bold")
    )
}
