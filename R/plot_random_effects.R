#' Plot Random-Effect Standard Deviations and Correlations
#'
#' `type = "sd"` plots the posterior of each random-effect standard deviation
#' with 50% and 95% intervals, one panel per grouping factor. `type = "cor"`
#' plots each correlation matrix as a heatmap of posterior means; correlations
#' whose interval includes zero are faded.
#'
#' @param fit A `broc_fit` object from [fit_broc()].
#' @param type `"sd"` (default) or `"cor"`.
#' @param group Optional character vector of grouping factors to show.
#' @param pars Optional character vector of parameters to show (family names, or
#'   `"criteria"` / `"criteria2"`).
#' @param prob Width of the outer interval (default 0.95).
#' @return A ggplot object.
#' @export
#' @examples
#' \dontrun{
#' plot_random_effects(fit)
#' plot_random_effects(fit, type = "cor", group = "participant")
#' }
plot_random_effects <- function(fit, type = c("sd", "cor"), group = NULL, pars = NULL, prob = 0.95) {
  .need_ggplot2()
  type <- match.arg(type)
  model <- .plot_model(fit)
  blocks <- .re_blocks(model)
  fam <- model$family$family
  if (!is.null(group)) blocks <- Filter(function(b) b$group %in% group, blocks)
  if (!is.null(pars)) {
    want <- vapply(pars, function(p) {
      if (p %in% c("criteria", "cutpoints")) return("criterion")
      if (p %in% c("criteria2", "cutpoints2")) return("criterion2")
      external_to_internal_param(p, fam)
    }, character(1))
    blocks <- Filter(function(b) any(b$params %in% want), blocks)
  }
  if (!length(blocks)) stop("The model has no matching random effects.", call. = FALSE)
  draws <- fit$draws(format = "matrix")
  if (type == "sd") .re_sd_plot(blocks, draws, prob) else .re_cor_plot(blocks, draws, prob)
}

.re_dim_labels <- function(model, p, re) {
  lab <- .param_label(model, p)
  if (p %in% c("criterion", "criterion2")) {
    terms <- .pretty_coef(.crit_re_coef_names(re), model$data)
    R <- length(terms)
    D <- re$dim
    base <- D %/% R
    plain <- R == 1L && identical(terms, "Intercept")
    mode <- re$varying_re_mode %||% "none"
    J <- .n_thresh_of(model, p)
    comp <- function(k) .crit_comp_label(k, J, model$gap_link)
    bin <- function(i) {
      lab <- (re$bin_levels %||% seq_len(i))[i]
      if (identical(as.character(lab), "new")) "new" else paste0("bin ", lab)
    }
    if (isTRUE(model$model_data[[p]]$location_scale) && is.null(re$varying_re_mode)) mode <- "shared"
    vapply(seq_len(D), function(d) {
      b <- (d - 1L) %/% R + 1L
      j <- (d - 1L) %% R + 1L
      bl <- switch(mode, shared = "location", per_bin = paste0(bin(b), " (location)"),
                   full = { k <- (b - 1L) %% J + 1L; paste0(bin((b - 1L) %/% J + 1L), ", ", comp(k)) },
                   comp(b))
      paste0(lab, ": ", paste(c(bl, if (!plain) terms[j]), collapse = ", "))
    }, character(1))
  } else {
    lv <- re$level_names
    if (is.null(lv) || !length(lv)) lv <- "(Intercept)"
    paste0(lab, ": ", .pretty_coef(rep_len(lv, re$dim), model$data))
  }
}

.re_blocks <- function(model) {
  md <- model$model_data
  out <- list()
  for (p in .re_param_names()) {
    res <- .re_list_of(md, p)
    for (g in names(res)) {
      re <- res[[g]]
      if (!is.null(re$cor_id)) next
      stan <- paste0(.smooth_stan_pname(model, p), "_", g)
      out[[length(out) + 1L]] <- list(group = g, params = p, dim = re$dim, labels = .re_dim_labels(model, p, re),
                                      sd = paste0("sigma_", stan), cor = paste0("corr_", stan),
                                      title = paste0(.param_label(model, p), " | ", g))
    }
  }
  for (cid in names(md$cross_cor)) {
    cc <- md$cross_cor[[cid]]
    labs <- unlist(lapply(cc$members, function(m) {
      re <- .re_list_of(md, m$param)[[cc$group]]
      rep_len(.re_dim_labels(model, m$param, re), m$dim)
    }))
    out[[length(out) + 1L]] <- list(group = cc$group, params = vapply(cc$members, function(m) m$param, ""),
                                    dim = length(labs), labels = labs, sd = paste0("sigma_cross_", cid, "_", cc$group),
                                    cor = paste0("corr_cross_", cid, "_", cc$group),
                                    title = paste0(cid, " | ", cc$group))
  }
  out
}

.re_sd_plot <- function(blocks, draws, prob) {
  rows <- list()
  for (b in blocks) {
    nm <- if (b$dim == 1L && b$sd %in% colnames(draws)) b$sd else paste0(b$sd, "[", seq_len(b$dim), "]")
    ok <- nm %in% colnames(draws)
    if (!any(ok)) next
    s <- .interval_summary(as.matrix(draws[, nm[ok], drop = FALSE]), prob)
    rows[[length(rows) + 1L]] <- cbind(s, label = b$labels[ok], group = b$group)
  }
  df <- do.call(rbind, rows)
  df$label <- factor(df$label, levels = rev(unique(df$label)))
  ggplot2::ggplot(df, ggplot2::aes(y = .data$label)) +
    ggplot2::geom_linerange(ggplot2::aes(xmin = .data$lower, xmax = .data$upper), linewidth = 0.5, colour = .broc_accent) +
    ggplot2::geom_linerange(ggplot2::aes(xmin = .data$lower50, xmax = .data$upper50), linewidth = 1.6, colour = .broc_accent) +
    ggplot2::geom_point(ggplot2::aes(x = .data$mean), shape = 21, fill = "white", colour = .broc_accent, size = 2.2, stroke = 1) +
    ggplot2::facet_wrap(~ group, scales = "free_y") +
    ggplot2::scale_x_continuous(limits = c(0, NA), expand = ggplot2::expansion(c(0, 0.05))) +
    ggplot2::labs(x = "Standard deviation", y = NULL, title = "Random-effect standard deviations",
                  subtitle = paste0("Posterior mean with 50% and ", round(100 * prob), "% intervals")) +
    theme_broc() + ggplot2::theme(panel.grid.major.y = ggplot2::element_blank())
}

.re_cor_plot <- function(blocks, draws, prob) {
  lo <- (1 - prob) / 2
  rows <- list()
  for (b in blocks) {
    if (b$dim < 2L) next
    for (i in 2:b$dim) for (j in 1:(i - 1L)) {
      nm <- paste0(b$cor, "[", i, ",", j, "]")
      if (!nm %in% colnames(draws)) next
      v <- as.numeric(draws[, nm])
      if (stats::sd(v) == 0 && all(v == 0)) next
      q <- stats::quantile(v, c(lo, 1 - lo), names = FALSE)
      rows[[length(rows) + 1L]] <- data.frame(panel = b$title, row = b$labels[i], col = b$labels[j], mean = mean(v),
                                              excludes0 = q[1] > 0 || q[2] < 0,
                                              ord_r = i, ord_c = j)
    }
  }
  if (!length(rows)) stop("No estimated correlations among the selected random effects.", call. = FALSE)
  df <- do.call(rbind, rows)
  short <- function(x) sub("^[^:]+: ", "", x)
  df$rowl <- factor(df$row, levels = unique(df$row[order(df$ord_r)]))
  df$coll <- factor(df$col, levels = unique(df$col[order(df$ord_c)]))
  ggplot2::ggplot(df, ggplot2::aes(x = .data$coll, y = .data$rowl, fill = .data$mean)) +
    ggplot2::geom_tile(ggplot2::aes(alpha = .data$excludes0), colour = "white", linewidth = 0.8) +
    ggplot2::geom_text(ggplot2::aes(label = formatC(.data$mean, format = "f", digits = 2),
                                    colour = abs(.data$mean) > 0.55), size = 3) +
    ggplot2::scale_fill_gradient2(low = "#0072B2", mid = "grey97", high = "#D55E00", limits = c(-1, 1),
                                  name = "Posterior\nmean r") +
    ggplot2::scale_alpha_manual(values = c(`TRUE` = 1, `FALSE` = 0.35), guide = "none") +
    ggplot2::scale_colour_manual(values = c(`TRUE` = "white", `FALSE` = "grey15"), guide = "none") +
    ggplot2::scale_x_discrete(labels = short, position = "top") +
    ggplot2::scale_y_discrete(labels = short, limits = rev) +
    ggplot2::facet_wrap(~ panel, scales = "free") +
    ggplot2::labs(x = NULL, y = NULL, title = "Random-effect correlations",
                  subtitle = paste0("Faded cells: ", round(100 * prob), "% interval includes zero")) +
    theme_broc() +
    ggplot2::theme(panel.grid = ggplot2::element_blank(), axis.line = ggplot2::element_blank(),
                   axis.ticks = ggplot2::element_blank(), legend.position = "right",
                   axis.text.x.top = if (max(nchar(short(as.character(df$col)))) > 6)
                     ggplot2::element_text(angle = 40, hjust = 0, vjust = 0) else ggplot2::element_text())
}
