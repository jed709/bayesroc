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
#'   parameter, smooth label, estimate, lower, upper. Smooths on `criterion` /
#'   `criterion2` have one set of rows per threshold component, identified by the
#'   `component` column (`"location"`, `"gap1"`, ...).
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

  all_draws <- as.matrix(fit$draws(format = "matrix"))
  S <- nrow(all_draws)
  if (!is.null(ndraws) && ndraws < S) {
    idx <- sample.int(S, ndraws)
    all_draws <- all_draws[idx, , drop = FALSE]
    S <- ndraws
  }

  results <- list()

  for (pname in names(smooth_data)) {
    if (!is.null(dpar) && !pname %in% dpar && !.smooth_display_pname(model, pname) %in% dpar) next

    for (sm in smooth_data[[pname]]) {
      if (!is.null(smooths) && !sm$term_str %in% smooths) next

      for (j in seq_along(sm$components)) {
        comp <- sm$components[[j]]

        covars <- sm$covars
        grid_data <- build_smooth_grid(covars, model$data, resolution,
                                       by_var = sm$by, bylevel = comp$bylevel)

        pred <- build_smooth_prediction_basis(comp, grid_data)
        stan_p <- .smooth_stan_pname(model, pname)
        is_crit <- pname %in% c("criterion", "criterion2")
        rows <- if (is_crit) seq_len(1L + .smooth_n_gaps(md, pname)) else 1L

        for (r in rows) {
          smooth_contribution <- matrix(0, nrow = nrow(grid_data), ncol = S)

          if (NCOL(pred$Xs) > 0) {
            beta_names <- .smooth_null_space_names(model, pname, comp$Xs_colnames, r)
            if (length(beta_names) > 0) {
              beta_draws <- all_draws[, beta_names, drop = FALSE]
              smooth_contribution <- smooth_contribution + pred$Xs %*% t(beta_draws)
            }
          }

          for (k in seq_along(pred$Zs_list)) {
            Zs <- pred$Zs_list[[k]]
            nbasis <- ncol(Zs)
            s_name <- paste0("s_", stan_p, "_", comp$san_label, "_", k)
            s_vars <- if (is_crit) paste0(s_name, "[", r, ",", 1:nbasis, "]") else paste0(s_name, "[", 1:nbasis, "]")
            s_draws <- all_draws[, s_vars, drop = FALSE]
            smooth_contribution <- smooth_contribution + Zs %*% t(s_draws)
          }

          res <- grid_data
          res$estimate <- rowMeans(smooth_contribution)
          res$lower <- apply(smooth_contribution, 1, quantile, probs = probs[1])
          res$upper <- apply(smooth_contribution, 1, quantile, probs = probs[2])
          res$parameter <- .smooth_display_pname(model, pname)
          res$smooth <- comp$sm_obj$label
          if (is_crit) res$component <- if (r == 1L) "location" else paste0("gap", r - 1L)

          results[[length(results) + 1]] <- res
        }
      }
    }
  }

  if (length(results) == 0) {
    stop("No matching smooth terms found.")
  }

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

  meta_cols <- c("estimate", "lower", "upper", "parameter", "smooth", "component")
  covar_cols <- setdiff(names(x), meta_cols)

  numeric_covars <- covar_cols[vapply(x[covar_cols], is.numeric, logical(1))]
  if (length(numeric_covars) == 0) {
    stop("No numeric covariates found for smooth plotting.")
  }

  xvar <- numeric_covars[1]

  plot_df <- x
  plot_df$.xvar <- plot_df[[xvar]]
  comp <- if ("component" %in% names(plot_df)) ifelse(is.na(plot_df$component), "", paste0(" [", plot_df$component, "]")) else ""
  plot_df$.panel <- paste0(plot_df$parameter, ": ", plot_df$smooth, comp)
  panels <- unique(plot_df$.panel)

  p <- ggplot2::ggplot(plot_df, ggplot2::aes(x = .xvar, y = estimate, group = .panel)) +
    ggplot2::geom_ribbon(ggplot2::aes(ymin = lower, ymax = upper),
                         alpha = 0.2, fill = "steelblue") +
    ggplot2::geom_line(color = "steelblue", linewidth = 1) +
    ggplot2::geom_hline(yintercept = 0, linetype = "dashed", alpha = 0.5) +
    ggplot2::labs(x = xvar, y = "Smooth effect") +
    ggplot2::theme_minimal()

  if (length(panels) > 1) {
    p <- p + ggplot2::facet_wrap(~ .panel, scales = "free")
  } else {
    p <- p + ggplot2::labs(title = panels[1])
  }

  p
}

.smooth_display_pname <- function(model, pname) .param_name_map(model$family$family)[[pname]] %||% pname

.smooth_stan_pname <- function(model, pname) {
  if (model$family$family %in% c("cumulative", "bivariate_cumulative")) .smooth_display_pname(model, pname) else pname
}

.smooth_n_gaps <- function(md, pname) {
  K <- if (pname == "criterion2") md$K2 else md$K
  max(0L, K - 2L)
}

.smooth_null_space_names <- function(model, pname, xs_colnames, r) {
  md <- model$model_data
  gl <- model$gap_link %||% "log"
  if (pname %in% c("criterion", "criterion2")) {
    crit <- md[[pname]]
    sfx <- if (pname == "criterion2") "_2" else ""
    if (pname == "criterion2" && isTRUE(md$varying_source_criteria)) {
      idx <- match(xs_colnames, crit$cov_names)
      idx <- idx[!is.na(idx)]
      if (!length(idx)) return(character(0))
      return(if (r == 1L) paste0("beta_thresh_mid_2_cov[", idx, "]")
             else paste0(.gap_param(gl, "_2_cov"), "[", r - 1L, ",", idx, "]"))
    }
    idx <- match(xs_colnames, crit$coef_names)
    idx <- idx[!is.na(idx)]
    if (!length(idx)) return(character(0))
    return(if (r == 1L) paste0("beta_thresh_mid", sfx, "[", idx, "]")
           else paste0(.gap_param(gl, sfx), "[", r - 1L, ",", idx, "]"))
  }
  idx <- match(xs_colnames, md[[paste0(pname, "_fixed")]]$coef_names)
  idx <- idx[!is.na(idx)]
  if (!length(idx)) return(character(0))
  paste0("beta_", .smooth_stan_pname(model, pname), "[", idx, "]")
}
