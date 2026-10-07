#' Plot Latent Strength Distributions
#'
#' Draws the model-implied latent strength distributions of lures and targets
#' with the response criteria, the classic signal detection picture.
#'
#' Densities are derived from the fitted response probabilities (the rate of
#' change of the probability of a response above a criterion placed at each
#' strength), so they follow each family's own structure: unequal variances,
#' mixtures, and summed recollection and familiarity in the continuous
#' dual-process model. In the dual-process signal detection model the curve is
#' the familiarity distribution scaled by the probability of not recollecting.
#' Densities are averaged over the observed rows of each cell (random effects
#' included) by default, or describe a typical participant with
#' `re_formula = NA`. Criteria are drawn at their mean locations.
#'
#' @inheritParams plot_roc
#' @param response For bivariate families, `1` (default) shows the detection
#'   dimension and `2` the source dimension.
#' @param ndraws Number of posterior draws averaged (default 100).
#' @return A ggplot object.
#' @export
#' @examples
#' \dontrun{
#' plot_distributions(fit)
#' plot_distributions(fit, by = "condition")
#' }
plot_distributions <- function(fit, by = NULL, response = 1, re_formula = NULL, ndraws = 100, seed = NULL,
                               cores = 1) {
  .need_ggplot2()
  restore_seed <- .seed_scope(seed)
  on.exit(restore_seed(), add = TRUE)
  model <- .plot_model(fit)
  by <- .check_by(model, by)
  fam <- model$family$family
  bivariate <- .is_bivariate_family(fam)
  if (cores > 1 && bivariate && .bvn_native_ready()) {
    old_threads <- .bvn_native$threads
    .bvn_native$threads <- as.integer(cores)
    on.exit(.bvn_native$threads <- old_threads, add = TRUE)
  }
  if (!bivariate && !identical(as.numeric(response), 1)) {
    stop("response = 2 is only available for bivariate families.", call. = FALSE)
  }
  roles <- if (fam %in% c("cumulative", "bivariate_cumulative")) {
    list(signal = seq_along(model$stan_data$y), noise = integer(0), what = "ordinal")
  } else .roc_roles(model, response)
  cells <- .cell_of_rows(model, by)
  setup <- .plot_setup(fit, model, ndraws, re_formula, thresholds2 = !identical(as.numeric(response), 1))
  setup$cache <- new.env(parent = emptyenv())
  w <- .row_weights(model)
  th_all <- if (bivariate && response == 2) setup$th2 else setup$th
  if (length(dim(th_all)) == 4L) th_all <- apply(th_all, c(1, 2, 4), mean)
  rng <- range(th_all)
  grid <- seq(rng[1] - 3.5, rng[2] + 3.5, length.out = 220)
  step <- diff(grid[1:2])
  dens <- function(rows) {
    sv <- .roc_survival(model, setup, rows, w, response, grid, mode = "grid")
    d <- -t(apply(sv, 1, diff)) / step
    data.frame(x = grid[-1] - step / 2, y = pmax(colMeans(matrix(d, nrow = nrow(sv))), 0))
  }

  pairs <- if (roles$what == "ordinal") {
    stats::setNames(lapply(levels(cells), function(l) list(signal = which(cells == l), noise = integer(0))), levels(cells))
  } else .pair_cells(model, by, cells, roles$signal, roles$noise)
  if (!length(pairs)) stop("No cell has the items needed for this plot.", call. = FALSE)
  noise_sets <- unique(lapply(pairs, function(p) sort(p$noise)))
  shared <- length(noise_sets) == 1L
  noise_name <- switch(roles$what, source = "Source A", detection = "Lures", ordinal = NULL)
  signal_name <- switch(roles$what, source = "Source B", detection = "Targets", ordinal = "Responses")

  curves <- list()
  for (lev in names(pairs)) {
    pr <- pairs[[lev]]
    curves[[length(curves) + 1L]] <- cbind(dens(pr$signal), item = signal_name, cell = lev)
    if (length(pr$noise) && (!shared || length(curves) == 1L)) {
      curves[[length(curves) + 1L]] <- cbind(dens(pr$noise), item = noise_name, cell = if (shared) noise_name else lev)
    }
  }
  df <- do.call(rbind, curves)
  crit <- .criteria_by_cell(th_all, cells, w, names(pairs))
  .distributions_plot(df, crit, by, shared, names(pairs), noise_name, signal_name, roles$what, re_formula, fam)
}

.criteria_by_cell <- function(th, cells, w, levs) {
  J <- dim(th)[3]
  out <- do.call(rbind, lapply(levs, function(l) {
    r <- which(cells == l)
    data.frame(cell = l, k = seq_len(J),
               x = vapply(seq_len(J), function(k) sum(colMeans(matrix(th[, r, k], ncol = length(r))) * w[r]) / sum(w[r]),
                          numeric(1)))
  }))
  if (max(tapply(out$x, out$k, function(v) diff(range(v)))) < 1e-8) {
    out <- out[out$cell == levs[1], ]
    out$cell <- "All"
  }
  out
}

.distributions_plot <- function(df, crit, by, shared, levs, noise_name, signal_name, what, re_formula, fam) {
  multi <- length(levs) > 1L
  if (!multi) df$cell <- ifelse(df$item == signal_name, signal_name, df$cell)
  if (!multi) levs <- signal_name
  cols <- if (multi) stats::setNames(.broc_palette(length(levs)), levs) else stats::setNames(.broc_accent, levs)
  if (!is.null(noise_name)) cols <- c(cols, stats::setNames("grey45", noise_name))
  df$key <- if (is.null(noise_name)) df$cell else ifelse(df$item == noise_name, noise_name, df$cell)
  df$panel <- if (shared || !multi) "" else df$cell
  df$key <- factor(df$key, levels = intersect(c(noise_name, levs), unique(df$key)))
  p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$x, y = .data$y, colour = .data$key, fill = .data$key,
                                        group = interaction(.data$key, .data$panel))) +
    ggplot2::geom_area(position = "identity", alpha = 0.12, linewidth = 0.8)
  sec <- ggplot2::waiver()
  if (crit$cell[1] == "All" || !multi) {
    cr <- crit[!duplicated(crit$k), ]
    p <- p + ggplot2::geom_vline(data = cr, ggplot2::aes(xintercept = .data$x), colour = "grey30",
                                 linetype = "22", linewidth = 0.4, inherit.aes = FALSE)
    sec <- ggplot2::dup_axis(name = NULL, breaks = cr$x, labels = paste0("c", cr$k))
  } else {
    crit$key <- factor(crit$cell, levels = levels(df$key))
    if (!shared) crit$panel <- crit$cell
    p <- p + ggplot2::geom_vline(data = crit, ggplot2::aes(xintercept = .data$x, colour = .data$key),
                                 linetype = "22", linewidth = 0.4, show.legend = FALSE)
  }
  if (!shared && multi) p <- p + ggplot2::facet_wrap(~ panel, ncol = 1)
  xlab <- switch(what, source = "Source strength (A to B)", ordinal = "Latent strength", "Memory strength")
  sub <- if (fam == "dpsdt") "Familiarity, scaled by the probability of not recollecting; dashed lines are criteria"
         else "Dashed lines are the response criteria"
  p + ggplot2::scale_x_continuous(sec.axis = sec) +
    ggplot2::scale_colour_manual(values = cols, name = if (multi && length(by)) by[1] else NULL) +
    ggplot2::scale_fill_manual(values = cols, name = if (multi && length(by)) by[1] else NULL) +
    ggplot2::scale_y_continuous(expand = ggplot2::expansion(c(0, 0.12))) +
    ggplot2::labs(x = xlab, y = NULL, title = "Latent distributions", subtitle = sub,
                  caption = if (identical(re_formula, NA)) "Typical participant (random effects at zero)."
                            else "Averaged over the observed data.") +
    theme_broc() +
    ggplot2::theme(axis.text.y = ggplot2::element_blank(), axis.ticks.y = ggplot2::element_blank(),
                   axis.line.y = ggplot2::element_blank(), panel.grid.major.y = ggplot2::element_blank(),
                   axis.text.x.top = ggplot2::element_text(colour = "grey30", size = ggplot2::rel(0.85)),
                   axis.ticks.x.top = ggplot2::element_blank(), axis.line.x.top = ggplot2::element_blank())
}
