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

.natural_draws <- function(fit, model, pars, by, re_formula, ndraws) {
  by <- .check_by(model, by %||% .default_by(model))
  if (!length(by)) by <- NULL
  setup <- .plot_setup(fit, model, ndraws, re_formula)
  w <- .row_weights(model)
  cells <- .cell_of_rows(model, by)
  fam <- model$family$family
  roles <- tryCatch(.roc_roles(model, 1), error = function(e) NULL)
  internal <- if (is.null(pars)) NULL else vapply(pars, function(p) {
    if (p %in% c("criteria", "criterion", "cutpoints", "cutpoints1")) return("criterion")
    if (p %in% c("criteria2", "criterion2", "cutpoints2")) return("criterion2")
    external_to_internal_param(p, fam)
  }, character(1))
  out <- list()
  cell_mean <- function(mat, rows) {
    vals <- vapply(levels(cells), function(lev) {
      r <- intersect(rows, which(cells == lev))
      if (!length(r)) return(rep(NA_real_, nrow(mat)))
      as.vector(mat[, r, drop = FALSE] %*% w[r]) / sum(w[r])
    }, numeric(nrow(mat)))
    matrix(vals, nrow = nrow(mat), dimnames = list(NULL, levels(cells)))
  }
  add <- function(label, item, colour, vals) {
    keep <- colSums(is.na(vals)) == 0
    if (!any(keep)) return()
    vals <- vals[, keep, drop = FALSE]
    item <- item[keep]; colour <- colour[keep]
    if (ncol(vals) > 1L && max(abs(sweep(vals, 1, vals[, 1]))) < 1e-10) {
      vals <- vals[, 1, drop = FALSE]
      item <- if (all(item == item[1])) item[1] else "All"
      colour <- "All"
    }
    out[[length(out) + 1L]] <<- data.frame(param = label, item = rep(item, each = nrow(vals)),
                                           colour = rep(colour, each = nrow(vals)), value = as.vector(vals))
  }
  for (p in names(setup$tp)) {
    if (!is.null(internal) && !p %in% internal) next
    rows <- if (p %in% .lure_param_set && !is.null(roles)) roles$noise
            else if (p %in% .signal_param_set && !is.null(roles) && fam != "source_mixture") roles$signal
            else seq_len(setup$N)
    vals <- cell_mean(setup$tp[[p]], rows)
    add(.param_label(model, p), colnames(vals), colnames(vals), vals)
  }
  crit_sets <- list(criterion = setup$th, criterion2 = if (length(dim(setup$th2)) == 3L) setup$th2)
  for (cn in names(crit_sets)) {
    th <- crit_sets[[cn]]
    if (is.null(th) || (!is.null(internal) && !cn %in% internal)) next
    J <- dim(th)[3]
    allv <- list(); items <- character(0); cols <- character(0)
    for (k in seq_len(J)) {
      v <- cell_mean(matrix(th[, , k], nrow = setup$S), seq_len(setup$N))
      allv[[k]] <- v
      items <- c(items, rep(paste0("c", k), ncol(v)))
      cols <- c(cols, colnames(v))
    }
    vals <- do.call(cbind, allv)
    distinct <- ncol(allv[[1]]) > 1L &&
      max(vapply(allv, function(v) max(abs(sweep(v, 1, v[, 1]))), numeric(1))) > 1e-10
    if (!distinct) {
      vals <- do.call(cbind, lapply(allv, function(v) v[, 1, drop = FALSE]))
      items <- paste0("c", seq_len(J)); cols <- rep("All", J)
    }
    out[[length(out) + 1L]] <- data.frame(param = .param_label(model, cn), item = rep(items, each = setup$S),
                                          colour = rep(cols, each = setup$S), value = as.vector(vals))
  }
  res <- do.call(rbind, out)
  if (is.null(res)) return(data.frame())
  res$param <- factor(res$param, levels = unique(res$param))
  res$by <- if (is.null(by)) NULL else paste(by, collapse = " / ")
  res
}

.coef_draws <- function(fit, model, pars, ndraws) {
  md <- model$model_data
  fam <- model$family$family
  vars <- posterior::variables(fit$draws())
  draws <- fit$draws(format = "matrix")
  if (!is.null(ndraws) && ndraws < nrow(draws)) draws <- draws[sample.int(nrow(draws), ndraws), , drop = FALSE]
  want <- if (is.null(pars)) NULL else vapply(pars, function(p) {
    if (p %in% c("criteria", "criterion", "cutpoints", "cutpoints1")) return("criterion")
    if (p %in% c("criteria2", "criterion2", "cutpoints2")) return("criterion2")
    external_to_internal_param(p, fam)
  }, character(1))
  out <- list()
  add <- function(label, names, items) {
    ok <- names %in% colnames(draws)
    if (!any(ok)) return()
    out[[length(out) + 1L]] <<- data.frame(param = label, item = rep(items[ok], each = nrow(draws)),
                                           colour = NA_character_,
                                           value = as.vector(draws[, names[ok], drop = FALSE]))
  }
  for (nm in grep("_fixed$", names(md), value = TRUE)) {
    fe <- md[[nm]]
    p <- sub("_fixed$", "", nm)
    if (!is.list(fe) || !isTRUE(fe$n_coef > 0) || (!is.null(want) && !p %in% want)) next
    stan_p <- .smooth_stan_pname(model, p)
    add(.param_label(model, p), paste0("beta_", stan_p, "[", seq_len(fe$n_coef), "]"),
        .pretty_coef(fe$coef_names, model$data))
  }
  gl <- model$gap_link %||% "log"
  for (cn in c("criterion", "criterion2")) {
    crit <- md[[cn]]
    if (is.null(crit) || (!is.null(want) && !cn %in% want)) next
    sfx <- if (cn == "criterion2") "_2" else ""
    coefs <- .pretty_coef(crit$coef_names %||% "(Intercept)", model$data)
    lab <- .param_label(model, cn)
    mids <- grep(paste0("^beta_thresh_mid", sfx, "(_varying|_cov|_new)?(\\[|$)"), colnames(draws), value = TRUE)
    if (length(mids)) add(paste0(lab, ": location"), mids, if (length(mids) == length(coefs)) coefs else mids)
    gp <- .gap_param(gl, sfx)
    gaps <- grep(paste0("^", gp, "(_varying|_cov|_new)?(\\[|$)"), colnames(draws), value = TRUE)
    if (length(gaps)) {
      glab <- vapply(gaps, function(g) {
        ix <- as.integer(strsplit(sub(".*\\[(.*)\\]$", "\\1", g), ",")[[1]])
        if (length(ix) == 2L && !grepl("_varying", g) && ix[2] <= length(coefs)) paste0("gap ", ix[1], ": ", coefs[ix[2]])
        else g
      }, character(1))
      add(paste0(lab, ": ", if (gl == "softplus") "raw gaps" else "log gaps"), gaps, unname(glab))
    }
  }
  res <- do.call(rbind, out)
  if (is.null(res)) return(data.frame())
  res$param <- factor(res$param, levels = unique(res$param))
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
