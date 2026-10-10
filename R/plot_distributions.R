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
#' @param response For bivariate families, which dimension to draw: `1` the detection dimension, `2` the
#'   source dimension, or both (the default, one panel each). Other families have one dimension.
#' @param type `"marginal"` (default) draws one latent dimension per panel. `"joint"` draws the latent plane
#'   of the two dimensions: for the bivariate families the (detection, source) evidence densities of new,
#'   source-A and source-B items as contours with the detection criteria (vertical) and the source criteria
#'   (horizontal; per detection response when they vary); for the CDP the recollection and familiarity
#'   densities of lures and targets with the recollection criterion, the know criterion and the confidence
#'   criteria on their sum. Densities come from the posterior-mean parameters of each distinct row of a cell,
#'   averaged over the rows with their weights. The recollected mass of the bivariate_dp model is not a
#'   density (it sits in the top detection category) and is left out, so its old-item contours are the
#'   familiarity component scaled by 1 - lambda; the vrdp2d mixture is drawn in full.
#' @param ndraws Number of posterior draws averaged (default 100).
#' @return A ggplot object.
#' @export
#' @examples
#' \dontrun{
#' plot_distributions(fit)
#' plot_distributions(fit, by = "condition")
#' }
plot_distributions <- function(fit, by = NULL, response = NULL, type = c("marginal", "joint"), re_formula = NULL,
                               ndraws = 100, seed = NULL, cores = 1) {
  .need_ggplot2()
  restore_seed <- .seed_scope(seed)
  on.exit(restore_seed(), add = TRUE)
  type <- match.arg(type)
  model <- .plot_model(fit)
  by <- .check_by(model, by)
  fam <- model$family$family
  bivariate <- .is_bivariate_family(fam)
  if (type == "joint") {
    if (!bivariate && fam != "cdp") stop("type = \"joint\" needs a bivariate family or the CDP; ",
                                         family_display_name(fam), " has one latent dimension.", call. = FALSE)
    cells <- .cell_of_rows(model, by)
    cf <- .cell_frame(model, by, cells)
    setup <- .plot_setup(fit, model, ndraws, re_formula)
    return(.latent_joint_plot(.latent_joint(model, setup, cells, cf, .row_weights(model), re_formula), by, re_formula, fam))
  }
  responses <- .plot_responses(response, bivariate)
  if (cores > 1 && .bvn_native_ready()) {
    old_threads <- .bvn_native$threads
    .bvn_native$threads <- as.integer(cores)
    on.exit(.bvn_native$threads <- old_threads, add = TRUE)
  }
  cells <- .cell_of_rows(model, by)
  setup <- .plot_setup(fit, model, ndraws, re_formula, thresholds2 = any(responses == 2L))
  setup$cache <- new.env(parent = emptyenv())
  w <- .row_weights(model)
  parts <- lapply(responses, function(resp) .distribution_curves(model, setup, cells, w, by, resp, fam, bivariate))
  df <- do.call(rbind, lapply(parts, `[[`, "df"))
  crit <- do.call(rbind, lapply(parts, `[[`, "crit"))
  .distributions_plot(df, crit, by, parts, re_formula, fam)
}

# Which response dimensions a plot draws: both for a bivariate family unless one is asked for.
.plot_responses <- function(response, bivariate) {
  if (is.null(response)) return(if (bivariate) c(1L, 2L) else 1L)
  r <- unique(as.integer(response))
  if (!bivariate && !identical(r, 1L)) stop("response = 2 is only available for bivariate families.", call. = FALSE)
  if (!all(r %in% c(1L, 2L))) stop("response must be 1, 2 or c(1, 2).", call. = FALSE)
  r
}

# The latent densities and criteria of one response dimension: the density is the rate of change of the
# survival over a grid spanning the drawn criteria, widened until the mass beyond its ends is negligible.
.distribution_curves <- function(model, setup, cells, w, by, response, fam, bivariate) {
  roles <- if (fam %in% c("cumulative", "bivariate_cumulative")) {
    list(signal = seq_along(model$stan_data$y), noise = integer(0), what = "ordinal")
  } else .roc_roles(model, response)
  pairs <- if (roles$what == "ordinal") {
    stats::setNames(lapply(levels(cells), function(l) list(signal = which(cells == l), noise = integer(0))), levels(cells))
  } else .pair_cells(model, by, cells, roles$signal, roles$noise)
  if (!length(pairs)) stop("No cell has the items needed for this plot.", call. = FALSE)
  th_all <- if (bivariate && response == 2L) setup$th2 else setup$th
  crit <- .criteria_by_cell(th_all, cells, w, names(pairs))
  all_rows <- sort(unique(unlist(lapply(pairs, function(p) c(p$signal, p$noise)))))
  rng <- range(crit$x) + c(-4, 4)
  for (i in 1:4) {
    sv <- .roc_survival(model, setup, all_rows, w, response, rng, mode = "grid")
    ends <- colMeans(sv)
    wider <- c(ends[1] < 0.998, ends[2] > 0.002)
    if (!any(wider)) break
    rng <- rng + c(-2, 2) * wider
  }
  grid <- seq(rng[1], rng[2], length.out = 220)
  step <- diff(grid[1:2])
  dens <- function(rows) {
    sv <- .roc_survival(model, setup, rows, w, response, grid, mode = "grid")
    d <- -t(apply(sv, 1, diff)) / step
    data.frame(x = grid[-1] - step / 2, y = pmax(colMeans(matrix(d, nrow = nrow(sv))), 0))
  }
  noise_sets <- unique(lapply(pairs, function(p) sort(p$noise)))
  shared <- length(noise_sets) == 1L
  noise_name <- switch(roles$what, source = "Source A", detection = "Lures", ordinal = NULL)
  signal_name <- switch(roles$what, source = "Source B", detection = "Targets", ordinal = "Responses")
  dim_label <- switch(roles$what, source = "Source strength (A to B)",
                      ordinal = if (bivariate) paste("Latent dimension", response) else "Latent strength", "Memory strength")
  curves <- list()
  for (lev in names(pairs)) {
    pr <- pairs[[lev]]
    curves[[length(curves) + 1L]] <- cbind(dens(pr$signal), item = signal_name, cell = lev)
    if (length(pr$noise) && (!shared || length(curves) == 1L)) {
      curves[[length(curves) + 1L]] <- cbind(dens(pr$noise), item = noise_name, cell = if (shared) noise_name else lev)
    }
  }
  df <- do.call(rbind, curves)
  df$dim <- dim_label
  crit$dim <- dim_label
  list(df = df, crit = crit, shared = shared, levs = names(pairs), noise_name = noise_name,
       signal_name = signal_name, what = roles$what, dim = dim_label)
}

# Mean criterion locations per cell (data-weighted over rows, averaged over draws). Criteria that vary with
# the detection response ([S, N, K1, J] thresholds) give one set per detection response (`set`); otherwise
# `set` is 0.
.criteria_by_cell <- function(th, cells, w, levs) {
  sets <- if (length(dim(th)) == 4L) seq_len(dim(th)[3]) else 0L
  J <- dim(th)[length(dim(th))]
  out <- do.call(rbind, lapply(sets, function(j) {
    th3 <- if (j > 0L) th[, , j, , drop = TRUE] else th
    dim(th3) <- c(dim(th)[1:2], J)
    do.call(rbind, lapply(levs, function(l) {
      r <- which(cells == l)
      data.frame(cell = l, set = j, k = seq_len(J),
                 x = vapply(seq_len(J), function(k) sum(colMeans(matrix(th3[, r, k], ncol = length(r))) * w[r]) / sum(w[r]),
                            numeric(1)))
    }))
  }))
  if (max(tapply(out$x, interaction(out$set, out$k), function(v) diff(range(v)))) < 1e-8) {
    out <- out[out$cell == levs[1], ]
    out$cell <- "All"
  }
  out
}

.distributions_plot <- function(df, crit, by, parts, re_formula, fam) {
  p1 <- parts[[1]]
  levs <- p1$levs
  shared <- all(vapply(parts, `[[`, logical(1), "shared"))
  two <- length(parts) > 1L
  multi <- length(levs) > 1L
  noise_names <- unlist(lapply(parts, `[[`, "noise_name"))
  signal_names <- vapply(parts, `[[`, character(1), "signal_name")
  if (!multi) df$cell <- ifelse(df$item %in% signal_names, df$item, df$cell)
  keys <- if (multi) levs else signal_names
  cols <- if (multi) stats::setNames(.broc_palette(length(levs)), levs) else stats::setNames(rep(.broc_accent, length(keys)), keys)
  if (length(noise_names)) cols <- c(cols, stats::setNames(rep("grey45", length(noise_names)), noise_names))
  df$key <- if (!length(noise_names)) df$cell else ifelse(df$item %in% noise_names, df$item, df$cell)
  df$panel <- if (shared || !multi) "" else df$cell
  key_levels <- if (two) unlist(lapply(parts, function(p) c(p$noise_name, if (multi) levs else p$signal_name)))
                else c(p1$noise_name, keys)
  df$key <- factor(df$key, levels = intersect(unique(key_levels), unique(df$key)))
  dims <- vapply(parts, `[[`, character(1), "dim")
  df$dim <- factor(df$dim, levels = dims)
  crit$dim <- factor(crit$dim, levels = dims)
  p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$x, y = .data$y, colour = .data$key, fill = .data$key,
                                        group = interaction(.data$key, .data$panel, .data$dim))) +
    ggplot2::geom_area(position = "identity", alpha = 0.12, linewidth = 0.8)
  sec <- ggplot2::waiver()
  one_set <- all(crit$cell == "All") || !multi
  varying <- any(crit$set > 0L)
  if (one_set) {
    cr <- crit[!duplicated(crit[, c("set", "k", "dim")]), ]
    if (varying) {
      # one set of criteria per detection response, light (lowest response) to dark (highest)
      cv <- cr[cr$set > 0L, ]
      cv$set <- factor(cv$set, levels = sort(unique(cv$set)))
      p <- p + ggplot2::geom_vline(data = cv, ggplot2::aes(xintercept = .data$x, alpha = .data$set), colour = "grey20",
                                   linetype = "22", linewidth = 0.4, inherit.aes = FALSE) +
        ggplot2::scale_alpha_manual(values = seq(0.2, 1, length.out = nlevels(cv$set)), name = "Detection response",
                                    guide = ggplot2::guide_legend(nrow = 1, override.aes = list(linetype = "22", colour = "grey20")))
      cr <- cr[cr$set == 0L, ]
    }
    if (nrow(cr)) {
      p <- p + ggplot2::geom_vline(data = cr, ggplot2::aes(xintercept = .data$x), colour = "grey30",
                                   linetype = "22", linewidth = 0.4, inherit.aes = FALSE)
      if (two || varying) {
        p <- p + ggplot2::geom_text(data = cr, ggplot2::aes(x = .data$x, label = paste0("c", .data$k)), y = Inf,
                                    vjust = 1.4, size = 2.8, colour = "grey30", inherit.aes = FALSE)
      } else {
        sec <- ggplot2::dup_axis(name = NULL, breaks = cr$x, labels = paste0("c", cr$k))
      }
    }
  } else {
    crit$key <- factor(crit$cell, levels = levels(df$key))
    if (!shared) crit$panel <- crit$cell
    if (varying) crit$alpha <- 0.25 + 0.75 * (crit$set - 1) / max(1, max(crit$set) - 1) else crit$alpha <- 1
    p <- p + ggplot2::geom_vline(data = crit, ggplot2::aes(xintercept = .data$x, colour = .data$key, alpha = .data$alpha),
                                 linetype = "22", linewidth = 0.4, show.legend = FALSE) +
      ggplot2::scale_alpha_identity()
  }
  panels <- !shared && multi
  if (two && panels) p <- p + ggplot2::facet_grid(panel ~ dim, scales = "free_x")
  else if (two) p <- p + ggplot2::facet_wrap(~ dim, scales = "free_x")
  else if (panels) p <- p + ggplot2::facet_wrap(~ panel, ncol = 1)
  xlab <- if (two) "Latent strength" else p1$dim
  sub <- if (fam == "dpsdt") "Familiarity, scaled by the probability of not recollecting; dashed lines are criteria"
         else if (varying) "Dashed lines are the response criteria; the source criteria are one set per detection response"
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
