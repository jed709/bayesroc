#' Conditional Smooth Effects
#'
#' Evaluate smooth terms at a grid of covariate values using posterior draws.
#' Returns a data frame with posterior mean and credible intervals for each
#' smooth, suitable for plotting with ggplot2. Requires a model fit with
#' `s()`/`t2()` smooth terms in a parameter formula.
#'
#' @param fit A `broc_fit` object from [fit_broc()].
#' @param smooths Character vector of smooth terms to evaluate (e.g., "s(age)").
#'   If NULL (default), all smooth terms are evaluated.
#' @param dpar Optional distributional parameter to filter to (e.g., `"dprime"`).
#' @param resolution Number of grid points per covariate (default 100).
#' @param prob Credible interval width (default 0.95).
#' @param ndraws Number of posterior draws to use (default: all).
#' @return A data frame of class `conditional_smooths` with covariate value(s),
#'   parameter, smooth label, estimate, lower, upper.
#' @export
conditional_smooths <- function(fit, smooths = NULL, dpar = NULL, resolution = 100,
                                prob = 0.95, ndraws = NULL) {
  model <- attr(fit, "broc_model")
  if (is.null(model)) {
    stop("Fit object must have 'broc_model' attribute.")
  }

  md <- model$model_data
  smooth_data <- md$smooth_data
  if (is.null(smooth_data)) {
    stop("Model does not contain smooth terms.")
  }

  probs <- c((1 - prob) / 2, 1 - (1 - prob) / 2)

  # Get posterior draws
  all_draws <- as.matrix(fit$draws(format = "matrix"))
  S <- nrow(all_draws)
  if (!is.null(ndraws) && ndraws < S) {
    idx <- sample.int(S, ndraws)
    all_draws <- all_draws[idx, , drop = FALSE]
    S <- ndraws
  }

  results <- list()

  for (pname in names(smooth_data)) {
    # Filter by distributional parameter
    if (!is.null(dpar) && !pname %in% dpar) next

    for (sm in smooth_data[[pname]]) {
      # Filter by user-specified smooths
      if (!is.null(smooths) && !sm$term_str %in% smooths) next

      for (j in seq_along(sm$components)) {
        comp <- sm$components[[j]]

        # Build grid of covariate values from the original data range
        covars <- sm$covars
        grid_data <- build_smooth_grid(covars, model$data, resolution,
                                       by_var = sm$by, bylevel = comp$bylevel)

        # Compute smooth prediction basis on the grid
        pred <- build_smooth_prediction_basis(comp, grid_data)

        # Extract unpenalized (Xs) and penalized (s_ = sds * zs) draws
        smooth_contribution <- matrix(0, nrow = nrow(grid_data), ncol = S)

        # Unpenalized part: Xs * bs
        if (NCOL(pred$Xs) > 0) {
          # Find which beta indices correspond to the Xs columns
          all_coef_names <- md[[paste0(pname, "_fixed")]]$coef_names
          xs_colnames <- comp$Xs_colnames
          xs_idx <- match(xs_colnames, all_coef_names)
          xs_idx <- xs_idx[!is.na(xs_idx)]
          if (length(xs_idx) > 0) {
            beta_names <- paste0("beta_", pname, "[", xs_idx, "]")
            beta_draws <- all_draws[, beta_names, drop = FALSE]  # S x n_Xs
            smooth_contribution <- smooth_contribution + pred$Xs %*% t(beta_draws)
          }
        }

        # Penalized part: Zs * s (where s = sds * zs)
        for (k in seq_along(pred$Zs_list)) {
          Zs <- pred$Zs_list[[k]]
          nbasis <- ncol(Zs)
          s_name <- paste0("s_", pname, "_", comp$san_label, "_", k)
          s_vars <- paste0(s_name, "[", 1:nbasis, "]")
          s_draws <- all_draws[, s_vars, drop = FALSE]  # S x nbasis
          smooth_contribution <- smooth_contribution + Zs %*% t(s_draws)  # N_grid x S
        }

        # Summarize: mean and CIs across draws
        est <- rowMeans(smooth_contribution)
        lower <- apply(smooth_contribution, 1, quantile, probs = probs[1])
        upper <- apply(smooth_contribution, 1, quantile, probs = probs[2])

        # Build label
        comp_label <- comp$sm_obj$label

        # Assemble result
        res <- grid_data
        res$estimate <- est
        res$lower <- lower
        res$upper <- upper
        res$parameter <- pname
        res$smooth <- comp_label

        results[[length(results) + 1]] <- res
      }
    }
  }

  if (length(results) == 0) {
    stop("No matching smooth terms found.")
  }

  # Unify column names across all results (handles mismatched columns),
  # fill missing with NA, then row-bind.
  all_cols <- unique(unlist(lapply(results, names)))
  results <- lapply(results, function(df) {
    missing <- setdiff(all_cols, names(df))
    for (col in missing) df[[col]] <- NA
    df[all_cols]
  })
  out <- do.call(rbind, results)
  rownames(out) <- NULL
  class(out) <- c("conditional_smooths", "data.frame")
  out
}


#' Build Grid for Smooth Evaluation
#'
#' Creates a grid of covariate values spanning the range observed in the data.
#'
#' @param covars Character vector of covariate names
#' @param data Training data
#' @param resolution Number of grid points per covariate
#' @param by_var By-variable name (or NA)
#' @param bylevel By-level value (or NULL)
#' @return Data frame with grid values
#' @noRd
build_smooth_grid <- function(covars, data, resolution, by_var = NA, bylevel = NULL) {
  grid_vals <- list()
  for (cv in covars) {
    vals <- data[[cv]]
    if (is.numeric(vals)) {
      grid_vals[[cv]] <- seq(min(vals), max(vals), length.out = resolution)
    } else {
      grid_vals[[cv]] <- sort(unique(vals))
    }
  }

  grid <- expand.grid(grid_vals, KEEP.OUT.ATTRS = FALSE)

  # If by-variable is a factor, add the by-level column
  if (!is.na(by_var) && !is.null(bylevel)) {
    grid[[by_var]] <- factor(bylevel, levels = levels(data[[by_var]]))
  }

  grid
}


#' Plot Conditional Smooths
#'
#' @param x A `conditional_smooths` object
#' @param ... Additional arguments passed to ggplot
#' @return A ggplot object
#' @keywords internal
#' @export
plot.conditional_smooths <- function(x, ...) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required for smooth plots.")
  }

  # Determine the covariate column(s) -- everything except estimate/lower/upper/parameter/smooth
  meta_cols <- c("estimate", "lower", "upper", "parameter", "smooth")
  covar_cols <- setdiff(names(x), meta_cols)

  # For 1D smooths: single covariate
  smooths <- unique(x$smooth)
  params <- unique(x$parameter)

  # Identify the primary numeric covariate
  numeric_covars <- covar_cols[vapply(x[covar_cols], is.numeric, logical(1))]
  if (length(numeric_covars) == 0) {
    stop("No numeric covariates found for smooth plotting.")
  }

  xvar <- numeric_covars[1]

  # Rename covariate column to a fixed name for ggplot mapping
  plot_df <- x
  plot_df$.xvar <- plot_df[[xvar]]

  p <- ggplot2::ggplot(plot_df, ggplot2::aes(x = .xvar, y = estimate)) +
    ggplot2::geom_ribbon(ggplot2::aes(ymin = lower, ymax = upper),
                         alpha = 0.2, fill = "steelblue") +
    ggplot2::geom_line(color = "steelblue", linewidth = 1) +
    ggplot2::geom_hline(yintercept = 0, linetype = "dashed", alpha = 0.5) +
    ggplot2::labs(x = xvar, y = "Smooth effect") +
    ggplot2::theme_minimal()

  # Facet by smooth term if multiple
  if (length(smooths) > 1 || length(params) > 1) {
    p <- p + ggplot2::facet_wrap(~ smooth, scales = "free")
  } else {
    p <- p + ggplot2::labs(title = smooths[1])
  }

  p
}
