#' Plot Model Parameters
#'
#' Create forest plots of parameter estimates with credible intervals from fitted
#' models, using ggdist.
#'
#' @param x A `broc_fit` object from [fit_broc()].
#' @param class Parameter class to plot: "b" (fixed effects), "sd" (random effect SDs),
#'              or "cor" (correlations)
#' @param dpar Character vector of distributional parameters to plot (default
#'             `"dprime"`). The valid set depends on the family (see the family
#'             constructor).
#' @param group Optional: filter to specific grouping factor(s) for sd/cor plots
#' @param type Plot type: "interval" (default, point + interval), "halfeye" (density + interval),
#'             "gradient" (gradient interval), "dots" (quantile dot plot)
#' @param prob Probability mass for outer credible interval (default 0.95)
#' @param prob_inner Probability mass for inner credible interval (default 0.5)
#' @param ... Ignored.
#' @return A ggplot object
#' @method plot broc_fit
#' @export
#'
#' @examples
#' \dontrun{
#' # Plot d' fixed effects with default interval style
#' plot(fit, class = "b", dpar = "dprime")
#'
#' # Plot with halfeye (density + interval)
#' plot(fit, class = "b", dpar = "dprime", type = "halfeye")
#'
#' # Plot multiple fixed effects, faceted by dpar
#' plot(fit, class = "b", dpar = c("dprime", "sigma"))
#'
#' # Plot random effect SDs
#' plot(fit, class = "sd", dpar = "dprime")
#'
#' # Plot correlations with gradient intervals
#' plot(fit, class = "cor", dpar = "dprime", type = "gradient")
#' }
plot.broc_fit <- function(x, class = "b", dpar = "dprime", group = NULL,
                          type = c("interval", "halfeye", "gradient", "dots"),
                          prob = 0.95, prob_inner = 0.5, ...) {
  fit <- x

  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required for plotting. Please install it.")
  }
  
  if (!requireNamespace("ggdist", quietly = TRUE)) {
    stop("Package 'ggdist' is required for plotting. Please install it.")
  }
  
  # Match type argument
  type <- match.arg(type)
  
  # Validate class
  valid_classes <- c("b", "sd", "cor")
  if (!class %in% valid_classes) {
    stop("class must be one of: ", paste(valid_classes, collapse = ", "))
  }
  
  if (is.null(dpar)) {
    stop("dpar must not be NULL")
  }
  
  # Get the model object from attribute (set by fit_broc)
  model <- attr(fit, "broc_model")
  if (is.null(model)) {
    stop("Fit object must have 'broc_model' attribute. Use fit_broc() to fit the model.")
  }
  
  # Handle criterion -> thresh_mid + log_gaps expansion (fixed effects only)
  if (class == "b") {
    dpar <- expand_criterion_dpar(dpar)
  }

  # Get draws data for plotting based on class
  plot_data <- switch(class,
                      "b" = get_fixed_effects_draws(fit, model, dpar),
                      "sd" = get_sd_draws(fit, model, dpar, group),
                      "cor" = get_cor_draws(fit, model, dpar, group)
  )
  
  if (nrow(plot_data) == 0) {
    stop("No parameters found matching the specified dpar and group")
  }
  
  # Build the plot
  p <- build_ggdist_plot(plot_data, class, type, prob, prob_inner)
  
  p
}


#' Expand criterion to thresh_mid and log_gaps
#' @noRd
expand_criterion_dpar <- function(dpar) {
  if ("criterion" %in% dpar) {
    dpar <- unique(c(setdiff(dpar, "criterion"), "thresh_mid", "log_gaps"))
  }
  if ("criterion2" %in% dpar) {
    dpar <- unique(c(setdiff(dpar, "criterion2"), "thresh_mid2", "log_gaps2"))
  }
  if ("lure" %in% dpar) {
    dpar <- unique(c(setdiff(dpar, "lure"), "dprime_L", "sigma_L", "lambda_L"))
  }
  dpar
}


#' Get fixed effects draws for plotting
#' @noRd
get_fixed_effects_draws <- function(fit, model, dpar) {
  
  draws <- fit$draws()
  var_names <- dimnames(draws)[[3]]
  
  results <- list()
  is_cdp <- model$family$family == "cdp"
  
  for (d in dpar) {
    # Get internal dpar name
    internal_dpar <- d
    if (is_cdp) {
      if (d == "rec") internal_dpar <- "dprime"
      if (d == "fam") internal_dpar <- "dprime2"
      if (d == "sigma_R") internal_dpar <- "sigma"
      if (d == "sigma_F") internal_dpar <- "sigma2"
    }
    
    # Handle thresh_mid
    if (d == "thresh_mid") {
      var_name <- "beta_thresh_mid"
      if (!any(grepl(paste0("^", var_name, "\\["), var_names))) next

      coef_names <- model$model_data$criterion$coef_names
      for (i in seq_along(coef_names)) {
        col_name <- paste0(var_name, "[", i, "]")
        if (!col_name %in% var_names) next
        draws_vec <- as.vector(draws[, , col_name])
        results[[length(results) + 1]] <- data.frame(
          dpar = "thresh_mid",
          coef = coef_names[i],
          value = draws_vec,
          stringsAsFactors = FALSE
        )
      }
      next
    }
    
    # Handle thresh_mid2 (source/dim2 thresholds)
    if (d == "thresh_mid2") {
      # Try varying, then non-varying, then per-coef
      for (var_name in c("beta_thresh_mid_2_varying", "beta_thresh_mid_2")) {
        if (any(grepl(paste0("^", var_name), var_names))) {
          matched <- var_names[grepl(paste0("^", var_name, "(\\[|$)"), var_names)]
          for (col_name in matched) {
            draws_vec <- as.vector(draws[, , col_name])
            results[[length(results) + 1]] <- data.frame(
              dpar = "thresh_mid2", coef = col_name,
              value = draws_vec, stringsAsFactors = FALSE
            )
          }
          break
        }
      }
      next
    }

    # Handle log_gaps2 (source/dim2 gaps)
    if (d == "log_gaps2") {
      gap_prefix <- if (identical(model$gap_link, "softplus")) "beta_raw_gaps_2" else "beta_log_gaps_2"
      for (var_name in c(paste0(gap_prefix, "_varying"), gap_prefix)) {
        if (any(grepl(paste0("^", var_name), var_names))) {
          matched <- var_names[grepl(paste0("^", var_name, "(\\[|$)"), var_names)]
          for (col_name in matched) {
            draws_vec <- as.vector(draws[, , col_name])
            results[[length(results) + 1]] <- data.frame(
              dpar = "log_gaps2", coef = col_name,
              value = draws_vec, stringsAsFactors = FALSE
            )
          }
          break
        }
      }
      next
    }

    # Handle log_gaps / raw_gaps
    if (d == "log_gaps") {
      var_name <- if (identical(model$gap_link, "softplus")) "beta_raw_gaps" else "beta_log_gaps"
      if (!any(grepl(paste0("^", var_name, "\\["), var_names))) next

      coef_names <- model$model_data$criterion$coef_names
      K <- model$model_data$K
      
      # For CDP, use J-1 gaps; for standard models use K-2
      if (is_cdp) {
        J <- model$family$J
        n_gaps <- J - 1
      } else {
        n_gaps <- K - 2
      }
      
      # Gaps are named 2D (beta_log_gaps[k,i], per coefficient) or 1D
      # (beta_log_gaps[k], single intercept) depending on backend/design.
      is_2d <- any(grepl(paste0("^", var_name, "\\[1,"), var_names))
      for (k in 1:n_gaps) {
        cols <- if (is_2d) {
          stats::setNames(paste0(var_name, "[", k, ",", seq_along(coef_names), "]"), coef_names)
        } else {
          stats::setNames(paste0(var_name, "[", k, "]"), coef_names[1])
        }
        for (i in seq_along(cols)) {
          col_name <- cols[i]
          if (!col_name %in% var_names) next
          draws_vec <- as.vector(draws[, , col_name])
          results[[length(results) + 1]] <- data.frame(
            dpar = "log_gaps",
            coef = paste0("gap", k, "_", names(cols)[i]),
            value = draws_vec,
            stringsAsFactors = FALSE
          )
        }
      }
      next
    }
    
    # Standard fixed effects
    var_name <- paste0("beta_", internal_dpar)
    if (!any(grepl(paste0("^", var_name, "\\["), var_names))) next
    
    # Get coefficient names
    fixed_info <- model$model_data[[paste0(internal_dpar, "_fixed")]]
    if (is.null(fixed_info)) next
    coef_names <- fixed_info$coef_names
    
    # Display dpar name
    display_dpar <- d
    if (is_cdp) {
      if (internal_dpar == "dprime") display_dpar <- "rec"
      if (internal_dpar == "dprime2") display_dpar <- "fam"
      if (internal_dpar == "sigma") display_dpar <- "sigma_R"
      if (internal_dpar == "sigma2") display_dpar <- "sigma_F"
    }
    
    for (i in seq_along(coef_names)) {
      col_name <- paste0(var_name, "[", i, "]")
      draws_vec <- as.vector(draws[, , col_name])
      results[[length(results) + 1]] <- data.frame(
        dpar = display_dpar,
        coef = coef_names[i],
        value = draws_vec,
        stringsAsFactors = FALSE
      )
    }
  }
  
  if (length(results) == 0) {
    return(data.frame(dpar = character(), coef = character(), value = numeric()))
  }
  
  do.call(rbind, results)
}


#' Get SD draws for plotting
#' @noRd
get_sd_draws <- function(fit, model, dpar, group_filter) {

  draws <- fit$draws()
  var_names <- dimnames(draws)[[3]]

  results <- list()
  is_cdp <- model$family$family == "cdp"

  # Helper to map internal param name to display name
  display_name <- function(d, internal) {
    if (is_cdp) {
      if (internal == "dprime") return("rec")
      if (internal == "dprime2") return("fam")
      if (internal == "sigma") return("sigma_R")
      if (internal == "sigma2") return("sigma_F")
    }
    d
  }

  # Helper to map user dpar to internal name
  to_internal <- function(d) {
    if (is_cdp) {
      if (d == "rec") return("dprime")
      if (d == "fam") return("dprime2")
      if (d == "sigma_R") return("sigma")
      if (d == "sigma_F") return("sigma2")
    }
    d
  }

  for (d in dpar) {
    internal_dpar <- to_internal(d)

    # Find random effects for this dpar
    if (internal_dpar %in% c("criterion", "criterion2")) {
      re_list <- model$model_data[[internal_dpar]]$random
    } else {
      re_list <- model$model_data[[paste0(internal_dpar, "_random")]]
    }

    if (is.null(re_list)) next

    for (grp in names(re_list)) {
      if (!is.null(group_filter) && !grp %in% group_filter) next

      re <- re_list[[grp]]

      # Skip cross-correlated REs here; handled separately below
      if (!is.null(re$cor_id)) next

      var_name <- paste0("sigma_", internal_dpar, "_", grp)

      # Get level names
      level_names <- re$level_names
      if (is.null(level_names) || length(level_names) == 0) {
        level_names <- "(Intercept)"
      }

      # Criterion/criterion2 SDs are a per-threshold vector
      # sigma_<crit>_<grp>[1..dim] -- always indexed, even when intercept-only.
      is_criterion <- internal_dpar %in% c("criterion", "criterion2")
      if (is_criterion) {
        n_re_terms <- if (!is.null(re$n_re_terms)) re$n_re_terms else 1
        if (n_re_terms == 1) {
          n_dim <- if (!is.null(re$dim)) re$dim else length(level_names)
          level_names <- paste0("thresh", seq_len(n_dim))
        }
      }

      disp <- display_name(d, internal_dpar)

      for (i in seq_along(level_names)) {
        col_name <- if (is_criterion || length(level_names) > 1) {
          paste0(var_name, "[", i, "]")
        } else {
          var_name
        }
        if (!col_name %in% var_names) next

        draws_vec <- as.vector(draws[, , col_name])
        results[[length(results) + 1]] <- data.frame(
          dpar = disp,
          group = grp,
          coef = level_names[i],
          value = draws_vec,
          stringsAsFactors = FALSE
        )
      }
    }
  }

  # Handle cross-parameter correlated SDs
  internal_dpars <- vapply(dpar, to_internal, character(1))

  for (cor_id in names(model$model_data$cross_cor)) {
    cc <- model$model_data$cross_cor[[cor_id]]
    grp <- cc$group
    if (!is.null(group_filter) && !grp %in% group_filter) next

    col_start <- 1
    for (member in cc$members) {
      param <- member$param

      # Only include if user requested this dpar
      if (!param %in% internal_dpars) {
        col_start <- col_start + member$dim
        next
      }

      # Find the user-facing dpar name for display
      disp <- display_name(dpar[internal_dpars == param][1], param)

      # Get level names
      if (param %in% c("criterion", "criterion2")) {
        re <- model$model_data[[param]]$random[[grp]]
      } else {
        re_list_tmp <- model$model_data[[paste0(param, "_random")]]
        re <- if (!is.null(re_list_tmp)) re_list_tmp[[grp]] else NULL
      }

      level_names <- "(Intercept)"
      if (!is.null(re)) {
        if (!is.null(re$level_names) && length(re$level_names) == member$dim) {
          level_names <- re$level_names
        } else if (member$dim > 1) {
          level_names <- paste0("dim", 1:member$dim)
        }
      }

      for (i in 1:member$dim) {
        idx <- col_start + i - 1
        col_name <- paste0("sigma_cross_", cor_id, "_", grp, "[", idx, "]")
        if (!col_name %in% var_names) next

        draws_vec <- as.vector(draws[, , col_name])
        results[[length(results) + 1]] <- data.frame(
          dpar = disp,
          group = grp,
          coef = if (length(level_names) == 1) level_names else level_names[i],
          value = draws_vec,
          stringsAsFactors = FALSE
        )
      }

      col_start <- col_start + member$dim
    }
  }

  if (length(results) == 0) {
    return(data.frame(dpar = character(), group = character(), coef = character(), value = numeric()))
  }

  do.call(rbind, results)
}


#' Get correlation draws for plotting
#' @noRd
get_cor_draws <- function(fit, model, dpar, group_filter) {

  draws <- fit$draws()
  var_names <- dimnames(draws)[[3]]

  results <- list()
  is_cdp <- model$family$family == "cdp"

  # Helper to map user dpar to internal name
  to_internal <- function(d) {
    if (is_cdp) {
      if (d == "rec") return("dprime")
      if (d == "fam") return("dprime2")
      if (d == "sigma_R") return("sigma")
      if (d == "sigma_F") return("sigma2")
    }
    d
  }

  # Helper to map internal param name to display name
  display_name <- function(d, internal) {
    if (is_cdp) {
      if (internal == "dprime") return("rec")
      if (internal == "dprime2") return("fam")
      if (internal == "sigma") return("sigma_R")
      if (internal == "sigma2") return("sigma_F")
    }
    d
  }

  for (d in dpar) {
    internal_dpar <- to_internal(d)

    # Find random effects for this dpar
    if (internal_dpar %in% c("criterion", "criterion2")) {
      re_list <- model$model_data[[internal_dpar]]$random
    } else {
      re_list <- model$model_data[[paste0(internal_dpar, "_random")]]
    }

    if (is.null(re_list)) next

    for (grp in names(re_list)) {
      if (!is.null(group_filter) && !grp %in% group_filter) next

      re <- re_list[[grp]]

      # Skip cross-correlated REs; handled separately below
      if (!is.null(re$cor_id)) next
      if (re$dim <= 1) next

      var_name <- paste0("corr_", internal_dpar, "_", grp)

      # Get level names for labeling
      level_names <- re$level_names
      if (is.null(level_names) || length(level_names) == 0) {
        level_names <- paste0("dim", 1:re$dim)
      }
      # Criterion REs correlate across thresholds; label by threshold rather
      # than the (shorter) term-level names, which would index out to NA.
      if (internal_dpar %in% c("criterion", "criterion2") &&
          (is.null(re$n_re_terms) || re$n_re_terms == 1)) {
        level_names <- paste0("thresh", seq_len(re$dim))
      }

      disp <- display_name(d, internal_dpar)

      # Extract correlations (lower triangle)
      for (i in 2:re$dim) {
        for (j in 1:(i-1)) {
          col_name <- paste0(var_name, "[", i, ",", j, "]")
          if (!col_name %in% var_names) next

          draws_vec <- as.vector(draws[, , col_name])
          pair_label <- paste0(level_names[j], " ~ ", level_names[i])

          results[[length(results) + 1]] <- data.frame(
            dpar = disp,
            group = grp,
            coef = pair_label,
            value = draws_vec,
            stringsAsFactors = FALSE
          )
        }
      }
    }
  }

  # Handle cross-parameter correlations
  internal_dpars <- vapply(dpar, to_internal, character(1))

  for (cor_id in names(model$model_data$cross_cor)) {
    cc <- model$model_data$cross_cor[[cor_id]]
    grp <- cc$group
    if (!is.null(group_filter) && !grp %in% group_filter) next
    if (cc$total_dim <= 1) next

    # Check if any requested dpar is in this cross-correlation
    member_params <- vapply(cc$members, function(m) m$param, character(1))
    if (!any(member_params %in% internal_dpars)) next

    # Build dimension labels
    labels <- character(0)
    for (member in cc$members) {
      param <- member$param
      disp <- display_name(param, param)

      if (param %in% c("criterion", "criterion2")) {
        re <- model$model_data[[param]]$random[[grp]]
      } else {
        re_list_tmp <- model$model_data[[paste0(param, "_random")]]
        re <- if (!is.null(re_list_tmp)) re_list_tmp[[grp]] else NULL
      }

      level_names <- NULL
      if (!is.null(re) && !is.null(re$level_names) && length(re$level_names) == member$dim) {
        level_names <- re$level_names
      }
      if (is.null(level_names)) level_names <- seq_len(member$dim)
      labels <- c(labels, paste0(disp, ":", level_names))
    }

    # Extract lower triangle correlations
    for (i in 2:cc$total_dim) {
      for (j in 1:(i-1)) {
        col_name <- paste0("corr_cross_", cor_id, "_", grp, "[", i, ",", j, "]")
        if (!col_name %in% var_names) next

        draws_vec <- as.vector(draws[, , col_name])
        pair_label <- paste0(labels[j], " ~ ", labels[i])

        results[[length(results) + 1]] <- data.frame(
          dpar = paste0("cross(", cor_id, ")"),
          group = grp,
          coef = pair_label,
          value = draws_vec,
          stringsAsFactors = FALSE
        )
      }
    }
  }

  if (length(results) == 0) {
    return(data.frame(dpar = character(), group = character(), coef = character(), value = numeric()))
  }

  do.call(rbind, results)
}


#' Build ggdist plot
#' @noRd
build_ggdist_plot <- function(data, class, type, prob, prob_inner) {
  
  # Determine faceting structure
  n_dpars <- length(unique(data$dpar))
  has_group <- "group" %in% names(data)
  n_groups <- if (has_group) length(unique(data$group)) else 1
  
  # Create coefficient factor with order preserved
  data$coef <- factor(data$coef, levels = rev(unique(data$coef)))
  
  # Base plot
  p <- ggplot2::ggplot(data, ggplot2::aes(x = value, y = coef))
  
  # Add the appropriate ggdist geom based on type
  probs <- c(prob_inner, prob)
  
  if (type == "interval") {
    p <- p + ggdist::stat_pointinterval(
      .width = probs,
      interval_size_range = c(0.8, 2),
      point_size = 2,
      shape = 21,
      fill = "white",
      color = "black"
    )
  } else if (type == "halfeye") {
    p <- p + ggdist::stat_halfeye(
      .width = probs,
      point_size = 2,
      interval_size_range = c(0.8, 2),
      shape = 21,
      fill = "gray70",
      slab_fill = "gray85",
      slab_alpha = 0.8,
      point_fill = "white"
    )
  } else if (type == "gradient") {
    p <- p + ggdist::stat_gradientinterval(
      .width = probs,
      point_size = 2,
      shape = 21,
      point_fill = "white",
      fill_type = "segments"
    )
  } else if (type == "dots") {
    p <- p + ggdist::stat_dotsinterval(
      .width = probs,
      point_size = 2,
      shape = 21,
      fill = "white",
      slab_fill = "gray60",
      quantiles = 100
    )
  }
  
  # Theme
  p <- p + ggplot2::theme_minimal(base_size = 11) +
    ggplot2::theme(
      panel.grid.minor = ggplot2::element_blank(),
      panel.grid.major.y = ggplot2::element_blank(),
      strip.text = ggplot2::element_text(face = "bold"),
      axis.title.y = ggplot2::element_blank()
    )
  
  # Labels
  class_labels <- c(b = "Fixed Effects", sd = "Random Effect SDs", cor = "Correlations")
  ci_label <- paste0(round(prob_inner * 100), "% / ", round(prob * 100), "% CI")
  
  p <- p + ggplot2::labs(
    x = paste0("Estimate (", ci_label, ")"),
    title = class_labels[class]
  )
  
  # Add faceting
  if (n_dpars > 1 && has_group && n_groups > 1) {
    p <- p + ggplot2::facet_grid(group ~ dpar, scales = "free_y", space = "free_y")
  } else if (n_dpars > 1) {
    p <- p + ggplot2::facet_wrap(~ dpar, scales = "free_y", ncol = 1)
  } else if (has_group && n_groups > 1) {
    p <- p + ggplot2::facet_wrap(~ group, scales = "free_y", ncol = 1)
  }
  
  p
}
