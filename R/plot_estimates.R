#' Plot Parameter Estimates
#'
#' Plots posterior estimates of the model parameters with 50% and 95% intervals.
#'
#' By default (`scale = "natural"`) each parameter is shown on its natural scale
#' (d', standard deviations, probabilities, criterion locations) for every design
#' cell, averaging over the cell's observed rows so that continuous covariates
#' are handled without picking a reference value. Signal parameters are
#' summarised over target items and lure parameters over lures. With
#' `scale = "coef"` the regression coefficients are shown instead, with readable
#' labels.
#'
#' @param fit A `broc_fit` object from [fit_broc()].
#' @param pars Optional character vector of parameters to show, using the names
#'   of the family (e.g. `"dprime"`, `"sigma"`, `"lambda"`, `"rec"`, `"mu"`) or
#'   `"criteria"` / `"criteria2"` for the response criteria. `NULL` shows all.
#' @param by Optional character vector of data columns defining the design cells
#'   (natural scale only). `NULL` uses the factors in the fixed-effect formulas.
#' @param scale `"natural"` (default) or `"coef"`.
#' @param type `"interval"` (default; point with 50% and 95% intervals) or
#'   `"halfeye"` (posterior density with intervals; needs the ggdist package).
#' @param re_formula `NA` (default) gives population-level estimates; `NULL`
#'   averages over the observed participants and items as well.
#' @param ndraws Optional number of posterior draws to use.
#' @param prob Width of the outer interval (default 0.95).
#' @param seed Optional seed for the draw subset.
#' @return A ggplot object.
#' @seealso [plot_random_effects()], [plot_roc()], [plot_distributions()].
#' @export
#' @examples
#' \dontrun{
#' plot_estimates(fit)
#' plot_estimates(fit, pars = c("dprime", "criteria"))
#' plot_estimates(fit, scale = "coef")
#' }
plot_estimates <- function(fit, pars = NULL, by = NULL, scale = c("natural", "coef"),
                           type = c("interval", "halfeye"), re_formula = NA, ndraws = NULL,
                           prob = 0.95, seed = NULL) {
  .need_ggplot2()
  restore_seed <- .seed_scope(seed)
  on.exit(restore_seed(), add = TRUE)
  scale <- match.arg(scale)
  type <- match.arg(type)
  if (type == "halfeye" && !requireNamespace("ggdist", quietly = TRUE)) {
    stop("type = \"halfeye\" needs the ggdist package.", call. = FALSE)
  }
  model <- .plot_model(fit)
  long <- if (scale == "natural") .natural_draws(fit, model, pars, by, re_formula, ndraws)
          else .coef_draws(fit, model, pars, ndraws)
  if (!nrow(long)) stop("No parameters matched 'pars'.", call. = FALSE)
  .estimates_plot(long, type, prob, scale, re_formula)
}

#' @rdname plot_estimates
#' @param x A `broc_fit` object.
#' @param ... Passed to [plot_estimates()].
#' @method plot broc_fit
#' @export
plot.broc_fit <- function(x, ...) plot_estimates(x, ...)

.signal_param_set <- c("dprime", "dprime2", "sigma", "sigma2", "lambda", "lambda2", "discrim", "rho",
                       "dprime_B", "lambda_B", "discrim_B", "sigma_B", "sigma2_B", "rho_B", "lambda2_B")
.lure_param_set <- c("dprime_L", "sigma_L", "lambda_L", "rho_N")

.default_by <- function(model) {
  md <- model$model_data
  fx <- grep("_fixed$", names(md), value = TRUE)
  vars <- unique(unlist(lapply(fx, function(n) {
    fe <- md[[n]]
    if (is.list(fe) && !is.null(fe$formula)) all.vars(fe$formula) else character(0)
  })))
  vars <- c(vars, if (!is.null(md$criterion$fixed_formula)) all.vars(md$criterion$fixed_formula))
  vars <- intersect(unique(vars), names(model$data))
  vars[vapply(vars, function(v) is.factor(model$data[[v]]) || is.character(model$data[[v]]), logical(1))]
}

# the fixed-effect and smooth predictors that are numeric: natural-scale estimates average over
# their observed values, and estimates() adds their average change per standard deviation
.continuous_predictors <- function(model) {
  md <- model$model_data
  fx <- grep("_fixed$", names(md), value = TRUE)
  vars <- unique(unlist(lapply(fx, function(n) {
    fe <- md[[n]]
    if (is.list(fe) && !is.null(fe$formula)) all.vars(fe$formula) else character(0)
  })))
  vars <- c(vars, if (!is.null(md$criterion$fixed_formula)) all.vars(md$criterion$fixed_formula),
            if (!is.null(md$criterion2$fixed_formula)) all.vars(md$criterion2$fixed_formula),
            unlist(lapply(md$smooth_data, function(l) lapply(l, function(sm)
              tryCatch(all.vars(str2lang(sm$sm_obj$term %||% sm$term)), error = function(e) character(0))))))
  vars <- intersect(unique(vars), names(model$data))
  vars[vapply(vars, function(v) is.numeric(model$data[[v]]), logical(1))]
}

.natural_draws <- function(fit, model, pars, by, re_formula, ndraws) {
  ed <- .estimate_draws(fit, model, pars, by, re_formula, ndraws)
  if (!ncol(ed$draws)) return(data.frame())
  S <- nrow(ed$draws)
  cell <- as.character(ed$cells)
  cell[is.na(cell)] <- "All"
  crit <- ed$info$criterion %||% rep(NA_character_, nrow(ed$info))
  item <- ifelse(is.na(crit), cell, crit)
  res <- data.frame(param = rep(ed$info$parameter, each = S), item = rep(item, each = S),
                    colour = rep(cell, each = S), value = as.vector(ed$draws))
  res$param <- factor(res$param, levels = unique(ed$info$parameter))
  by_used <- setdiff(names(ed$info), c("parameter", "name", "criterion"))
  if (length(by_used)) res$by <- paste(by_used, collapse = " / ")
  res
}

.coef_map <- function(model, vars, pars = NULL) {
  md <- model$model_data
  fam <- model$family$family
  want <- .pars_internal(pars, fam)
  rows <- list()
  add <- function(label, names, terms, link) {
    ok <- names %in% vars
    if (!any(ok)) return()
    lab <- if (identical(link, "identity")) label else paste0(label, " (", link, ")")
    rows[[length(rows) + 1L]] <<- data.frame(variable = names[ok], parameter = lab, term = terms[ok])
  }
  for (nm in grep("_fixed$", names(md), value = TRUE)) {
    fe <- md[[nm]]
    p <- sub("_fixed$", "", nm)
    if (!is.list(fe) || !isTRUE(fe$n_coef > 0) || (!is.null(want) && !p %in% want)) next
    add(.param_label(model, p), paste0("beta_", .smooth_stan_pname(model, p), "[", seq_len(fe$n_coef), "]"),
        .pretty_coef(fe$coef_names, model$data), model$family$params[[p]]$link %||% "identity")
  }
  gl <- model$gap_link %||% "log"
  for (cn in c("criterion", "criterion2")) {
    crit <- md[[cn]]
    if (is.null(crit) || (!is.null(want) && !cn %in% want)) next
    sfx <- if (cn == "criterion2") "_2" else ""
    coefs <- .pretty_coef(crit$coef_names %||% "(Intercept)", model$data)
    lab <- .param_label(model, cn)
    J <- .n_thresh_of(model, cn)
    mids <- grep(paste0("^beta_thresh_mid", sfx, "(_varying|_cov|_new)?(\\[|$)"), vars, value = TRUE)
    if (length(mids)) {
      add(paste0(lab, ": ", if (isTRUE(crit$location_scale)) "location (mean)" else .crit_mid_label(J)), mids,
          if (length(mids) == length(coefs)) coefs else sub("^beta_thresh_mid", "", mids), "identity")
    }
    gp <- .gap_param(gl, sfx)
    gaps <- grep(paste0("^", gp, "(_varying|_cov|_new)?(\\[|$)"), vars, value = TRUE)
    if (length(gaps)) {
      ix <- lapply(gaps, function(g) suppressWarnings(as.integer(strsplit(sub(".*\\[(.*)\\]$", "\\1", g), ",")[[1]])))
      gi <- vapply(seq_along(gaps), function(i) {
        if (grepl("_varying|_cov|_new", gaps[i]) || anyNA(ix[[i]])) NA_integer_ else ix[[i]][1]
      }, integer(1))
      glab <- vapply(seq_along(gaps), function(i) {
        if (is.na(gi[i])) return(gaps[i])
        k <- ix[[i]]
        if (length(k) == 2L && k[2] <= length(coefs)) paste0(.crit_gap_label(gi[i], J), ": ", coefs[k[2]])
        else .crit_gap_label(gi[i], J)
      }, character(1))
      o <- order(ifelse(is.na(gi), Inf, .crit_gap_hi(gi, J)), seq_along(gaps))
      add(paste0(lab, ": gaps"), gaps[o], glab[o], if (gl == "softplus") "softplus" else "log")
    }
  }
  out <- do.call(rbind, rows)
  if (is.null(out)) data.frame(variable = character(0), parameter = character(0), term = character(0)) else out
}

.coef_draws <- function(fit, model, pars, ndraws) {
  draws <- fit$draws(format = "matrix")
  if (!is.null(ndraws) && ndraws < nrow(draws)) draws <- draws[sample.int(nrow(draws), ndraws), , drop = FALSE]
  map <- .coef_map(model, colnames(draws), pars)
  if (!nrow(map)) return(data.frame())
  S <- nrow(draws)
  res <- data.frame(param = rep(map$parameter, each = S), item = rep(map$term, each = S), colour = NA_character_,
                    value = as.vector(draws[, map$variable, drop = FALSE]))
  res$param <- factor(res$param, levels = unique(map$parameter))
  res
}

.estimates_plot <- function(long, type, prob, scale, re_formula) {
  long$key <- paste(long$param, long$item, long$colour, sep = "\r")
  keys <- unique(long$key)
  summ <- do.call(rbind, lapply(keys, function(k) {
    v <- long$value[long$key == k]
    first <- long[match(k, long$key), c("param", "item", "colour")]
    cbind(first, .interval_summary(matrix(v, ncol = 1), prob))
  }))
  summ$item <- factor(summ$item, levels = rev(unique(summ$item)))
  long$item <- factor(long$item, levels = levels(summ$item))
  colours <- setdiff(unique(summ$colour[!is.na(summ$colour)]), "All")
  multi <- length(colours) > 1L
  summ$colour[is.na(summ$colour) | summ$colour == "All"] <- if (multi) "All" else ""
  long$colour[is.na(long$colour) | long$colour == "All"] <- if (multi) "All" else ""
  pos <- if (multi) ggplot2::position_dodge(width = 0.6) else "identity"
  p <- ggplot2::ggplot(summ, ggplot2::aes(y = .data$item, colour = .data$colour, fill = .data$colour))
  if (type == "halfeye") {
    p <- p + ggdist::stat_slab(data = long, ggplot2::aes(x = .data$value), alpha = 0.25, colour = NA,
                               position = pos, height = 0.85)
  }
  p <- p +
    ggplot2::geom_linerange(ggplot2::aes(xmin = .data$lower, xmax = .data$upper), linewidth = 0.5, position = pos) +
    ggplot2::geom_linerange(ggplot2::aes(xmin = .data$lower50, xmax = .data$upper50), linewidth = 1.6, position = pos) +
    ggplot2::geom_point(ggplot2::aes(x = .data$mean), shape = 21, fill = "white", size = 2.2, stroke = 1, position = pos) +
    ggplot2::facet_wrap(~ param, scales = "free") +
    ggplot2::labs(x = NULL, y = NULL, title = if (scale == "coef") "Coefficients" else "Estimates",
                  subtitle = paste0("Posterior mean with 50% and ", round(100 * prob), "% intervals",
                                    if (scale == "natural") {
                                      if (identical(re_formula, NA)) "; population level" else "; averaged over the data"
                                    } else "")) +
    theme_broc() + ggplot2::theme(panel.grid.major.y = ggplot2::element_blank())
  if (multi) {
    by_name <- unique(long$by)
    vals <- c(.broc_palette(length(colours)), All = "grey40")
    names(vals)[seq_along(colours)] <- colours
    p <- p + ggplot2::scale_colour_manual(values = vals, breaks = colours, name = if (length(by_name)) by_name else NULL) +
      ggplot2::scale_fill_manual(values = vals, breaks = colours, guide = "none")
  } else {
    p <- p + ggplot2::scale_colour_manual(values = .broc_accent, guide = "none") +
      ggplot2::scale_fill_manual(values = .broc_accent, guide = "none")
  }
  p
}
