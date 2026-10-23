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
#'   `component` column (`"c3 (midpoint)"`, `"c3 to c4"`, ... for six response categories).
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
        n_comp <- if (is_crit) 1L + .smooth_n_gaps(md, pname) else 1L
        rows <- seq_len(n_comp)

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
          if (is_crit) res$component <- if (r == 1L) .crit_mid_label(n_comp) else .crit_gap_label(r - 1L, n_comp)

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
  attr(out, "param_labels") <- vapply(stats::setNames(names(smooth_data), vapply(names(smooth_data), function(p)
    .smooth_display_pname(model, p), character(1))), function(p) .param_label(model, p), character(1))
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
#' Plots smooth terms evaluated by [conditional_smooths()]: the posterior mean
#' with a credible band, one panel per smooth (and per threshold component for
#' criterion smooths). A two-dimensional smooth is drawn against its first
#' covariate at five values of the second.
#'
#' @param x A `conditional_smooths` object.
#' @param ... Unused.
#' @return A ggplot object.
#' @export
plot.conditional_smooths <- function(x, ...) {
  .need_ggplot2()
  meta_cols <- c("estimate", "lower", "upper", "parameter", "smooth", "component")
  covar_cols <- setdiff(names(x), meta_cols)
  numeric_covars <- covar_cols[vapply(x[covar_cols], is.numeric, logical(1))]
  if (length(numeric_covars) == 0) stop("No numeric covariates found for smooth plotting.", call. = FALSE)
  labs_map <- attr(x, "param_labels")
  df <- as.data.frame(x)
  plab <- if (!is.null(labs_map)) unname(ifelse(is.na(labs_map[df$parameter]), df$parameter, labs_map[df$parameter])) else df$parameter
  comp <- if ("component" %in% names(df)) ifelse(is.na(df$component), "", paste0(", ", df$component)) else ""
  df$.panel <- paste0(plab, ": ", df$smooth, comp)
  parts <- list()
  for (pn in unique(df$.panel)) {
    d <- df[df$.panel == pn, , drop = FALSE]
    used <- numeric_covars[vapply(numeric_covars, function(v) any(!is.na(d[[v]])) && length(unique(d[[v]])) > 1L, logical(1))]
    d$.x <- d[[used[1]]]
    d$.z <- NA_real_
    if (length(used) > 1L) {
      zv <- sort(unique(d[[used[2]]]))
      pick <- zv[unique(round(seq(1, length(zv), length.out = 5)))]
      d <- d[d[[used[2]]] %in% pick, , drop = FALSE]
      d$.z <- d[[used[2]]]
      d$.zname <- used[2]
    }
    d$.xname <- used[1]
    parts[[pn]] <- d
  }
  df <- do.call(rbind, lapply(parts, function(d) { if (is.null(d$.zname)) d$.zname <- NA_character_; d }))
  df$.panel <- factor(df$.panel, levels = names(parts))
  df$.grp <- interaction(df$.panel, df$.z, drop = TRUE)
  xname <- unique(df$.xname)
  two_d <- any(!is.na(df$.z))
  p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$.x, y = .data$estimate, group = .data$.grp)) +
    ggplot2::geom_hline(yintercept = 0, colour = "grey60", linetype = "22", linewidth = 0.4)
  if (two_d) {
    p <- p + ggplot2::geom_ribbon(ggplot2::aes(ymin = .data$lower, ymax = .data$upper, fill = .data$.z), alpha = 0.12) +
      ggplot2::geom_line(ggplot2::aes(colour = .data$.z), linewidth = 0.9) +
      ggplot2::scale_colour_gradient(low = "#56B4E9", high = "#0B3C5D", name = paste(unique(stats::na.omit(df$.zname)), collapse = " / ")) +
      ggplot2::scale_fill_gradient(low = "#56B4E9", high = "#0B3C5D", guide = "none")
  } else {
    p <- p + ggplot2::geom_ribbon(ggplot2::aes(ymin = .data$lower, ymax = .data$upper), fill = .broc_accent, alpha = 0.15) +
      ggplot2::geom_line(colour = .broc_accent, linewidth = 0.9)
  }
  if (length(parts) > 1L) p <- p + ggplot2::facet_wrap(~ .panel, scales = "free")
  p + ggplot2::labs(x = if (length(xname) == 1L) xname else NULL, y = "Smooth effect",
                    title = if (length(parts) == 1L) names(parts) else "Smooth terms",
                    subtitle = "Posterior mean with credible band") +
    theme_broc()
}

#' Plot Smooth Terms
#'
#' Convenience wrapper: `plot(conditional_smooths(fit, ...))`.
#'
#' @inheritParams conditional_smooths
#' @return A ggplot object.
#' @export
#' @examples
#' \dontrun{
#' plot_smooths(fit)
#' }
plot_smooths <- function(fit, smooths = NULL, dpar = NULL, resolution = 100, prob = 0.95, ndraws = NULL) {
  plot(conditional_smooths(fit, smooths = smooths, dpar = dpar, resolution = resolution, prob = prob,
                           ndraws = ndraws))
}

.smooth_display_pname <- function(model, pname) {
  (model$family$external_aliases %||% .param_name_map(model$family$family))[[pname]] %||% pname
}

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
