#' Posterior Predictive Check for SDT Models
#'
#' Compares the observed response distributions with data sets replicated from
#' the posterior predictive distribution, separately for each design cell.
#'
#' @param object A `broc_fit` object from [fit_broc()].
#' @param type `"bars"` (default): observed response proportions with the
#'   predictive mean and interval. `"rootogram"`: a hanging rootogram, where
#'   observed bars hang from the expected counts and misfit shows as bar bottoms
#'   away from zero. `"stat"`: predictive distribution of a summary statistic
#'   (`stat_fun`) with its observed value. `"lines"`: replicated response
#'   proportions as thin lines with the observed proportions.
#' @param by Optional character vector of data columns defining the cells.
#'   `NULL` (default) splits by item type (lures and targets, sources, or new,
#'   source A and source B items). Cells that mix item types are also split by
#'   item type.
#' @param ndraws Number of replicated data sets (default 200).
#' @param prob Width of the predictive intervals (default 0.95).
#' @param stat_fun Statistic for `type = "stat"`, applied to the vector of
#'   responses (default: mean).
#' @param re_formula `NULL` (default) includes all random effects; `NA` excludes
#'   them.
#' @param response For bivariate families, `1` (default) checks the first
#'   response and `2` the second. For the continuous dual-process model, `2`
#'   checks the remember/know/guess responses.
#' @param cores Number of cores used for posterior prediction (default 1).
#' @param seed Optional seed for the draw subset and replicated data; the global
#'   random-number state is restored afterwards.
#' @param ... Unused.
#' @return A ggplot object.
#' @examples
#' \dontrun{
#' pp_check(fit)
#' pp_check(fit, by = "condition")
#' pp_check(fit, type = "rootogram")
#' }
#' @importFrom bayesplot pp_check
#' @method pp_check broc_fit
#' @export
pp_check.broc_fit <- function(object, type = c("bars", "rootogram", "stat", "lines"), by = NULL,
                              ndraws = 200, prob = 0.95, stat_fun = mean, re_formula = NULL,
                              response = 1, cores = 1, seed = NULL, ...) {
  .need_ggplot2()
  restore_seed <- .seed_scope(seed)
  on.exit(restore_seed(), add = TRUE)
  type <- match.arg(type)
  fit <- object
  model <- .plot_model(fit)
  by <- .check_by(model, by)
  fam <- model$family$family
  bivariate <- .is_bivariate_family(fam)

  rows <- seq_along(model$stan_data$y)
  if (fam == "cdp" && response == 2) {
    probs <- predict(fit, type = "response", ndraws = ndraws, summary = FALSE, re_formula = re_formula,
                     cores = cores, response = 2)
    rk <- model$stan_data$rk
    rows <- which(model$stan_data$y %in% model$family$old_levels & !is.na(rk))
    probs <- probs[, rows, , drop = FALSE]
    y_obs <- rk[rows]
    K <- if (!is.null(model$family$n_rkg)) model$family$n_rkg else 2L
    cat_labels <- c("Remember", "Know", "Guess")[seq_len(K)]
  } else {
    probs <- .predict_broc(fit, type = "response", ndraws = ndraws, summary = FALSE, re_formula = re_formula,
                           cores = cores, margin = if (bivariate) as.integer(response))
    y_obs <- if (bivariate && response == 2) model$stan_data$y2 else model$stan_data$y
    K <- dim(probs)[3]
    labs <- if (bivariate && response == 2) model$model_data$response2_labels else model$model_data$response_labels
    cat_labels <- if (length(labs) == K) as.character(labs) else as.character(seq_len(K))
  }
  counts <- .row_weights(model)[rows]
  cells <- .ppc_cells(model, by)[rows]
  cells <- droplevels(cells)
  levs <- levels(cells)
  S <- dim(probs)[1]
  obs <- t(vapply(levs, function(l) {
    i <- which(cells == l)
    vapply(seq_len(K), function(k) sum(counts[i][y_obs[i] == k]), numeric(1))
  }, numeric(K)))
  obs <- matrix(obs, nrow = length(levs), dimnames = list(levs, NULL))
  pred <- .ppc_replicate(probs, counts, as.integer(cells), length(levs), K)

  switch(type,
    bars = .ppc_bars(obs, pred, levs, cat_labels, prob),
    rootogram = .ppc_rootogram(obs, pred, levs, cat_labels),
    stat = .ppc_stat(obs, pred, levs, stat_fun, K),
    lines = .ppc_lines(obs, pred, levs, cat_labels))
}

.ppc_cells <- function(model, by) {
  sd <- model$stan_data
  fam <- model$family$family
  role <- if (.is_bivariate_family(fam) && !is.null(sd$item_type)) {
    factor(c("New", "Source A", "Source B")[sd$item_type], levels = c("New", "Source A", "Source B"))
  } else if (fam == "source_mixture") {
    factor(ifelse(sd$source == max(sd$source), "Source B", "Source A"), levels = c("Source A", "Source B"))
  } else if (!is.null(sd$is_old) && fam != "cumulative") {
    factor(ifelse(sd$is_old == 1, "Targets", "Lures"), levels = c("Lures", "Targets"))
  } else NULL
  if (is.null(by)) return(if (is.null(role)) factor(rep("All", length(sd$y))) else droplevels(role))
  cells <- .cell_of_rows(model, by)
  if (is.null(role)) return(cells)
  mixed <- any(tapply(as.integer(role), cells, function(v) length(unique(v)) > 1L))
  if (!mixed) return(cells)
  interaction(cells, role, sep = " / ", drop = TRUE, lex.order = TRUE)
}

.ppc_replicate <- function(probs, counts, cell, n_cells, K) {
  S <- dim(probs)[1]
  trial_row <- rep.int(seq_along(counts), counts)
  trial_cell <- cell[trial_row]
  out <- array(0L, c(S, n_cells, K))
  for (s in seq_len(S)) {
    P <- matrix(probs[s, trial_row, ], nrow = length(trial_row), ncol = K)
    k <- .pp_draw_categories(P, K)
    out[s, , ] <- matrix(tabulate((trial_cell - 1L) * K + k, nbins = n_cells * K), n_cells, K, byrow = TRUE)
  }
  out
}

.pp_draw_categories <- function(P, K) {
  rs <- rowSums(P)
  bad <- !is.finite(rs) | rs <= 0
  P <- P / rs
  if (any(bad)) P[bad, ] <- 1 / K
  cumP <- P
  for (k in 2:K) cumP[, k] <- cumP[, k - 1] + P[, k]
  u <- stats::runif(nrow(P))
  cat <- rowSums(cumP < u) + 1L
  cat[cat > K] <- K
  cat
}

.ppc_long <- function(obs, pred, levs, cat_labels, prob) {
  lo <- (1 - prob) / 2
  do.call(rbind, lapply(seq_along(levs), function(g) {
    n <- sum(obs[g, ])
    pp <- matrix(pred[, g, ], nrow = dim(pred)[1]) / n
    data.frame(cell = levs[g], k = factor(cat_labels, levels = cat_labels), obs = obs[g, ] / n,
               mean = colMeans(pp), lower = apply(pp, 2, stats::quantile, lo, names = FALSE),
               upper = apply(pp, 2, stats::quantile, 1 - lo, names = FALSE))
  }))
}

.ppc_facets <- function(levs) {
  if (length(levs) > 1L) ggplot2::facet_wrap(~ cell, nrow = if (length(levs) <= 4L) 1L else NULL)
}

.ppc_bars <- function(obs, pred, levs, cat_labels, prob) {
  df <- .ppc_long(obs, pred, levs, cat_labels, prob)
  df$cell <- factor(df$cell, levels = levs)
  ggplot2::ggplot(df, ggplot2::aes(x = .data$k)) +
    ggplot2::geom_col(ggplot2::aes(y = .data$obs), fill = "grey85", width = 0.75) +
    ggplot2::geom_linerange(ggplot2::aes(ymin = .data$lower, ymax = .data$upper), colour = .broc_accent, linewidth = 0.9) +
    ggplot2::geom_point(ggplot2::aes(y = .data$mean), colour = .broc_accent, size = 1.9) +
    .ppc_facets(levs) +
    ggplot2::scale_y_continuous(expand = ggplot2::expansion(c(0, 0.05))) +
    ggplot2::labs(x = "Response", y = "Proportion of responses", title = "Posterior predictive check",
                  subtitle = paste0("Bars: observed. Points and lines: predicted mean and ", round(100 * prob),
                                    "% interval")) +
    theme_broc() + ggplot2::theme(panel.grid.major.x = ggplot2::element_blank())
}

.ppc_rootogram <- function(obs, pred, levs, cat_labels) {
  df <- do.call(rbind, lapply(seq_along(levs), function(g) {
    e <- colMeans(matrix(pred[, g, ], nrow = dim(pred)[1]))
    data.frame(cell = levs[g], k = seq_along(cat_labels), expected = sqrt(e), top = sqrt(e),
               bottom = sqrt(e) - sqrt(obs[g, ]))
  }))
  df$cell <- factor(df$cell, levels = levs)
  ggplot2::ggplot(df, ggplot2::aes(x = .data$k)) +
    ggplot2::geom_rect(ggplot2::aes(xmin = .data$k - 0.375, xmax = .data$k + 0.375, ymin = .data$bottom,
                                    ymax = .data$top), fill = "grey80") +
    ggplot2::geom_hline(yintercept = 0, colour = "grey25", linewidth = 0.4) +
    ggplot2::geom_line(ggplot2::aes(y = .data$expected), colour = .broc_accent, linewidth = 0.9) +
    ggplot2::geom_point(ggplot2::aes(y = .data$expected), colour = .broc_accent, size = 1.9) +
    .ppc_facets(levs) +
    ggplot2::scale_x_continuous(breaks = seq_along(cat_labels), labels = cat_labels) +
    ggplot2::labs(x = "Response", y = "Square root of count", title = "Hanging rootogram",
                  subtitle = "Observed bars hang from the expected counts (line); bottoms away from zero show misfit") +
    theme_broc() + ggplot2::theme(panel.grid.major.x = ggplot2::element_blank())
}

.ppc_stat <- function(obs, pred, levs, stat_fun, K) {
  df <- do.call(rbind, lapply(seq_along(levs), function(g) {
    rep_stat <- apply(matrix(pred[, g, ], nrow = dim(pred)[1]), 1, function(cn) stat_fun(rep(seq_len(K), cn)))
    data.frame(cell = levs[g], value = rep_stat)
  }))
  ob <- data.frame(cell = levs, value = vapply(seq_along(levs), function(g) stat_fun(rep(seq_len(K), obs[g, ])), numeric(1)))
  df$cell <- factor(df$cell, levels = levs); ob$cell <- factor(ob$cell, levels = levs)
  ggplot2::ggplot(df, ggplot2::aes(x = .data$value)) +
    ggplot2::geom_histogram(bins = 30, fill = "grey80", colour = "white", linewidth = 0.2) +
    ggplot2::geom_vline(data = ob, ggplot2::aes(xintercept = .data$value), colour = .broc_accent, linewidth = 1) +
    (if (length(levs) > 1L) ggplot2::facet_wrap(~ cell, scales = "free_x")) +
    ggplot2::scale_y_continuous(expand = ggplot2::expansion(c(0, 0.05))) +
    ggplot2::labs(x = "Statistic", y = NULL, title = "Posterior predictive check: statistic",
                  subtitle = "Histogram: replicated data sets. Line: observed") +
    theme_broc() + ggplot2::theme(axis.text.y = ggplot2::element_blank(), panel.grid.major.y = ggplot2::element_blank())
}

.ppc_lines <- function(obs, pred, levs, cat_labels) {
  S <- dim(pred)[1]
  keep <- seq_len(min(S, 60L))
  reps <- do.call(rbind, lapply(seq_along(levs), function(g) {
    n <- sum(obs[g, ])
    do.call(rbind, lapply(keep, function(s) data.frame(cell = levs[g], draw = s, k = seq_along(cat_labels),
                                                        p = pred[s, g, ] / n)))
  }))
  ob <- do.call(rbind, lapply(seq_along(levs), function(g)
    data.frame(cell = levs[g], k = seq_along(cat_labels), p = obs[g, ] / sum(obs[g, ]))))
  reps$cell <- factor(reps$cell, levels = levs); ob$cell <- factor(ob$cell, levels = levs)
  ggplot2::ggplot(reps, ggplot2::aes(x = .data$k, y = .data$p)) +
    ggplot2::geom_line(ggplot2::aes(group = .data$draw), colour = .broc_accent, alpha = 0.12, linewidth = 0.4) +
    ggplot2::geom_line(data = ob, colour = "grey10", linewidth = 0.9) +
    ggplot2::geom_point(data = ob, colour = "grey10", size = 1.8) +
    .ppc_facets(levs) +
    ggplot2::scale_x_continuous(breaks = seq_along(cat_labels), labels = cat_labels) +
    ggplot2::labs(x = "Response", y = "Proportion of responses", title = "Posterior predictive check",
                  subtitle = "Thin lines: replicated data sets. Black: observed") +
    theme_broc() + ggplot2::theme(panel.grid.major.x = ggplot2::element_blank())
}
