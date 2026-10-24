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
#'   For the bivariate families `"joint"` checks the K1 x K2 table of the two responses together:
#'   one tile per cell, coloured by the standardized residual of the observed count against the
#'   replicated counts and labelled with the observed count. A misfit confined to the joint cells
#'   (both margins fine) is visible only here.
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
#' @param response Which response to check. Bivariate families have the
#'   detection (`1`) and source (`2`) responses, the continuous dual-process
#'   model the confidence (`1`) and remember/know/guess (`2`) responses; by
#'   default both are shown, one row each. Other families have one response.
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
pp_check.broc_fit <- function(object, type = c("bars", "rootogram", "stat", "lines", "joint"), by = NULL,
                              ndraws = 200, prob = 0.95, stat_fun = mean, re_formula = NULL,
                              response = NULL, cores = 1, seed = NULL, ...) {
  .need_ggplot2()
  restore_seed <- .seed_scope(seed)
  on.exit(restore_seed(), add = TRUE)
  type <- match.arg(type)
  fit <- object
  model <- .plot_model(fit)
  by <- .check_by(model, by)
  fam <- model$family$family
  bivariate <- .is_bivariate_family(fam)
  two <- bivariate || fam == "cdp"
  responses <- if (is.null(response)) (if (two) c(1L, 2L) else 1L) else unique(as.integer(response))
  if (!two && !identical(responses, 1L)) {
    stop("response = 2 is only available for bivariate families and the continuous dual-process model.", call. = FALSE)
  }
  if (!all(responses %in% c(1L, 2L))) stop("response must be 1, 2 or c(1, 2).", call. = FALSE)
  if (type == "joint") {
    if (!bivariate) stop("type = \"joint\" needs a bivariate family: it checks the joint table of the two responses.",
                         call. = FALSE)
    return(.ppc_joint_plot(.ppc_joint_block(fit, model, by, ndraws, re_formula, cores, fam), by, re_formula))
  }
  blocks <- lapply(responses, function(resp) .ppc_block(fit, model, by, ndraws, re_formula, resp, cores, fam, bivariate))
  switch(type,
    bars = .ppc_bars(blocks, prob),
    rootogram = .ppc_rootogram(blocks),
    stat = .ppc_stat(blocks, stat_fun),
    lines = .ppc_lines(blocks))
}

# Observed and replicated counts per cell and category for one response.
.ppc_block <- function(fit, model, by, ndraws, re_formula, response, cores, fam, bivariate) {
  rows <- seq_along(model$stan_data$y)
  if (fam == "cdp" && response == 2L) {
    probs <- predict(fit, type = "response", ndraws = ndraws, summary = FALSE, re_formula = re_formula,
                     cores = cores, response = 2)
    rk <- model$stan_data$rk
    rk_levels <- if (!is.null(model$family$rk_levels)) model$family$rk_levels else model$family$old_levels
    rows <- which(model$stan_data$y %in% rk_levels & !is.na(rk) & rk > 0)
    probs <- probs[, rows, , drop = FALSE]
    y_obs <- rk[rows]
    K <- if (!is.null(model$family$n_rkg)) model$family$n_rkg else 2L
    cat_labels <- c("Remember", "Know", "Guess")[seq_len(K)]
  } else {
    probs <- .predict_broc(fit, type = "response", ndraws = ndraws, summary = FALSE, re_formula = re_formula,
                           cores = cores, margin = if (bivariate) response)
    y_obs <- if (bivariate && response == 2L) model$stan_data$y2 else model$stan_data$y
    K <- dim(probs)[3]
    labs <- if (bivariate && response == 2L) model$model_data$response2_labels else model$model_data$response_labels
    cat_labels <- if (length(labs) == K) as.character(labs) else as.character(seq_len(K))
  }
  counts <- .row_weights(model)[rows]
  cells <- droplevels(.ppc_cells(model, by)[rows])
  levs <- levels(cells)
  obs <- t(vapply(levs, function(l) {
    i <- which(cells == l)
    vapply(seq_len(K), function(k) sum(counts[i][y_obs[i] == k]), numeric(1))
  }, numeric(K)))
  obs <- matrix(obs, nrow = length(levs), dimnames = list(levs, NULL))
  pred <- .ppc_replicate(probs, counts, as.integer(cells), length(levs), K)
  dim_label <- if (fam == "cdp") c("Confidence", "Judgement")[response]
               else if (fam == "bivariate_cumulative") paste("Response", response)
               else c("Detection", "Source")[response]
  list(obs = obs, pred = pred, levs = levs, cat_labels = cat_labels, K = K, dim = dim_label)
}

.ppc_cells <- function(model, by) {
  sd <- model$stan_data
  fam <- model$family$family
  role <- if (.is_bivariate_family(fam) && !is.null(sd$item_type)) {
    factor(c("New", "Source A", "Source B")[sd$item_type], levels = c("New", "Source A", "Source B"))
  } else if (fam == "source_mixture") {
    factor(ifelse(sd$source == max(sd$source), "Source B", "Source A"), levels = c("Source A", "Source B"))
  } else if (!is.null(sd$is_old) && fam != "cumulative") {
    factor(ifelse(sd$is_old > 0, "Targets", "Lures"), levels = c("Lures", "Targets"))
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

# Long data of every block: one row per cell x category, with the response dimension.
.ppc_long <- function(blocks, prob) {
  lo <- (1 - prob) / 2
  df <- do.call(rbind, lapply(blocks, function(bl) {
    do.call(rbind, lapply(seq_along(bl$levs), function(g) {
      n <- sum(bl$obs[g, ])
      pp <- matrix(bl$pred[, g, ], nrow = dim(bl$pred)[1]) / n
      data.frame(dim = bl$dim, cell = bl$levs[g], k = bl$cat_labels, obs = bl$obs[g, ] / n,
                 mean = colMeans(pp), lower = apply(pp, 2, stats::quantile, lo, names = FALSE),
                 upper = apply(pp, 2, stats::quantile, 1 - lo, names = FALSE), stringsAsFactors = FALSE)
    }))
  }))
  .ppc_factors(df, blocks)
}

.ppc_factors <- function(df, blocks) {
  df$dim <- factor(df$dim, levels = unique(vapply(blocks, `[[`, character(1), "dim")))
  df$cell <- factor(df$cell, levels = unique(unlist(lapply(blocks, `[[`, "levs"))))
  df$k <- factor(df$k, levels = unique(unlist(lapply(blocks, `[[`, "cat_labels"))))
  df
}

# Panels: cells across, response dimensions down (each with its own categories).
.ppc_facets <- function(df, free = TRUE) {
  nd <- nlevels(df$dim); nc <- nlevels(df$cell)
  sc <- if (free) "free_x" else "fixed"
  if (nd > 1L && nc > 1L) return(ggplot2::facet_wrap(~ dim + cell, nrow = nd, scales = sc,
                                                     labeller = ggplot2::labeller(.multi_line = FALSE)))
  if (nd > 1L) return(ggplot2::facet_wrap(~ dim, scales = sc, nrow = 1L))
  if (nc > 1L) return(ggplot2::facet_wrap(~ cell, nrow = if (nc <= 4L) 1L else NULL, scales = sc))
  NULL
}

.ppc_bars <- function(blocks, prob) {
  df <- .ppc_long(blocks, prob)
  ggplot2::ggplot(df, ggplot2::aes(x = .data$k)) +
    ggplot2::geom_col(ggplot2::aes(y = .data$obs), fill = "grey85", width = 0.75) +
    ggplot2::geom_linerange(ggplot2::aes(ymin = .data$lower, ymax = .data$upper), colour = .broc_accent, linewidth = 0.9) +
    ggplot2::geom_point(ggplot2::aes(y = .data$mean), colour = .broc_accent, size = 1.9) +
    .ppc_facets(df) +
    ggplot2::scale_y_continuous(expand = ggplot2::expansion(c(0, 0.05))) +
    ggplot2::labs(x = "Response", y = "Proportion of responses", title = "Posterior predictive check",
                  subtitle = paste0("Bars: observed. Points and lines: predicted mean and ", round(100 * prob),
                                    "% interval")) +
    theme_broc() + ggplot2::theme(panel.grid.major.x = ggplot2::element_blank())
}

.ppc_rootogram <- function(blocks) {
  df <- do.call(rbind, lapply(blocks, function(bl) {
    do.call(rbind, lapply(seq_along(bl$levs), function(g) {
      e <- colMeans(matrix(bl$pred[, g, ], nrow = dim(bl$pred)[1]))
      data.frame(dim = bl$dim, cell = bl$levs[g], k = bl$cat_labels, expected = sqrt(e), top = sqrt(e),
                 bottom = sqrt(e) - sqrt(bl$obs[g, ]), stringsAsFactors = FALSE)
    }))
  }))
  df <- .ppc_factors(df, blocks)
  ggplot2::ggplot(df, ggplot2::aes(x = .data$k)) +
    ggplot2::geom_tile(ggplot2::aes(y = (.data$top + .data$bottom) / 2, height = .data$top - .data$bottom),
                       width = 0.75, fill = "grey80") +
    ggplot2::geom_hline(yintercept = 0, colour = "grey25", linewidth = 0.4) +
    ggplot2::geom_line(ggplot2::aes(y = .data$expected, group = 1), colour = .broc_accent, linewidth = 0.9) +
    ggplot2::geom_point(ggplot2::aes(y = .data$expected), colour = .broc_accent, size = 1.9) +
    .ppc_facets(df) +
    ggplot2::labs(x = "Response", y = "Square root of count", title = "Hanging rootogram",
                  subtitle = "Observed bars hang from the expected counts (line); bottoms away from zero show misfit") +
    theme_broc() + ggplot2::theme(panel.grid.major.x = ggplot2::element_blank())
}

.ppc_stat <- function(blocks, stat_fun) {
  df <- do.call(rbind, lapply(blocks, function(bl) {
    do.call(rbind, lapply(seq_along(bl$levs), function(g) {
      rep_stat <- apply(matrix(bl$pred[, g, ], nrow = dim(bl$pred)[1]), 1, function(cn) stat_fun(rep(seq_len(bl$K), cn)))
      data.frame(dim = bl$dim, cell = bl$levs[g], value = rep_stat, stringsAsFactors = FALSE)
    }))
  }))
  ob <- do.call(rbind, lapply(blocks, function(bl) {
    data.frame(dim = bl$dim, cell = bl$levs, stringsAsFactors = FALSE,
               value = vapply(seq_along(bl$levs), function(g) stat_fun(rep(seq_len(bl$K), bl$obs[g, ])), numeric(1)))
  }))
  df$k <- ""; ob$k <- ""
  df <- .ppc_factors(df, blocks); ob <- .ppc_factors(ob, blocks)
  ggplot2::ggplot(df, ggplot2::aes(x = .data$value)) +
    ggplot2::geom_histogram(bins = 30, fill = "grey80", colour = "white", linewidth = 0.2) +
    ggplot2::geom_vline(data = ob, ggplot2::aes(xintercept = .data$value), colour = .broc_accent, linewidth = 1) +
    .ppc_facets(df) +
    ggplot2::scale_y_continuous(expand = ggplot2::expansion(c(0, 0.05))) +
    ggplot2::labs(x = "Statistic", y = NULL, title = "Posterior predictive check: statistic",
                  subtitle = "Histogram: replicated data sets. Line: observed") +
    theme_broc() + ggplot2::theme(axis.text.y = ggplot2::element_blank(), panel.grid.major.y = ggplot2::element_blank())
}

.ppc_lines <- function(blocks) {
  reps <- do.call(rbind, lapply(blocks, function(bl) {
    S <- dim(bl$pred)[1]
    keep <- seq_len(min(S, 60L))
    do.call(rbind, lapply(seq_along(bl$levs), function(g) {
      n <- sum(bl$obs[g, ])
      do.call(rbind, lapply(keep, function(s) data.frame(dim = bl$dim, cell = bl$levs[g], draw = s, k = bl$cat_labels,
                                                          p = bl$pred[s, g, ] / n, stringsAsFactors = FALSE)))
    }))
  }))
  ob <- do.call(rbind, lapply(blocks, function(bl) do.call(rbind, lapply(seq_along(bl$levs), function(g)
    data.frame(dim = bl$dim, cell = bl$levs[g], k = bl$cat_labels, p = bl$obs[g, ] / sum(bl$obs[g, ]),
               stringsAsFactors = FALSE)))))
  reps <- .ppc_factors(reps, blocks); ob <- .ppc_factors(ob, blocks)
  ggplot2::ggplot(reps, ggplot2::aes(x = .data$k, y = .data$p)) +
    ggplot2::geom_line(ggplot2::aes(group = .data$draw), colour = .broc_accent, alpha = 0.12, linewidth = 0.4) +
    ggplot2::geom_line(data = ob, ggplot2::aes(group = 1), colour = "grey10", linewidth = 0.9) +
    ggplot2::geom_point(data = ob, colour = "grey10", size = 1.8) +
    .ppc_facets(reps) +
    ggplot2::labs(x = "Response", y = "Proportion of responses", title = "Posterior predictive check",
                  subtitle = "Thin lines: replicated data sets. Black: observed") +
    theme_broc() + ggplot2::theme(panel.grid.major.x = ggplot2::element_blank())
}


# ---- joint table of the two responses (bivariate families) ----------------------------------------

.ppc_joint_block <- function(fit, model, by, ndraws, re_formula, cores, fam) {
  probs <- .predict_broc(fit, type = "response", ndraws = ndraws, summary = FALSE, re_formula = re_formula, cores = cores)
  d <- dim(probs)
  if (length(d) != 4L) stop("joint probabilities are not available for this fit.", call. = FALSE)
  S <- d[1]; N <- d[2]; K1 <- d[3]; K2 <- d[4]
  flat <- array(probs, c(S, N, K1 * K2))                       # category (j, k) -> (k - 1) * K1 + j
  y <- (model$stan_data$y2 - 1L) * K1 + model$stan_data$y
  counts <- .row_weights(model)
  cells <- droplevels(.ppc_cells(model, by))
  levs <- levels(cells)
  obs <- t(vapply(levs, function(l) {
    i <- which(cells == l)
    vapply(seq_len(K1 * K2), function(k) sum(counts[i][y[i] == k]), numeric(1))
  }, numeric(K1 * K2)))
  obs <- matrix(obs, nrow = length(levs))
  pred <- .ppc_replicate(flat, counts, as.integer(cells), length(levs), K1 * K2)
  mu <- apply(pred, c(2, 3), mean); sd <- apply(pred, c(2, 3), stats::sd)
  z <- (obs - mu) / pmax(sd, 1)
  df <- data.frame(cell = rep(levs, times = K1 * K2), y1 = rep(rep(seq_len(K1), times = K2), each = length(levs)),
                   y2 = rep(seq_len(K2), each = length(levs) * K1), observed = as.vector(obs),
                   predicted = as.vector(mu), z = as.vector(z))
  df$cell <- factor(df$cell, levels = levs)
  labs <- if (fam == "cdp") c("Confidence", "Judgement") else if (fam == "bivariate_cumulative") c("Response 1", "Response 2")
          else c("Detection confidence", "Source confidence")
  list(df = df, labs = labs, n_cells = length(levs))
}

.ppc_joint_plot <- function(block, by, re_formula) {
  df <- block$df
  df$z <- pmin(pmax(df$z, -3), 3)
  p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$y1, y = .data$y2)) +
    ggplot2::geom_tile(ggplot2::aes(fill = .data$z), colour = "white") +
    ggplot2::geom_text(ggplot2::aes(label = .data$observed), size = 2.8) +
    ggplot2::scale_fill_gradient2(low = "#1d4ed8", mid = "white", high = "#c2410c", limits = c(-3, 3),
                                  name = "(observed - replicated) / sd") +
    ggplot2::scale_x_continuous(breaks = seq_len(max(df$y1))) +
    ggplot2::scale_y_continuous(breaks = seq_len(max(df$y2)))
  if (block$n_cells > 1L) p <- p + ggplot2::facet_wrap(~ cell)
  level <- if (identical(re_formula, NA)) "Replicated for a typical participant (random effects at zero)."
           else "Replicated from the observed rows (random effects included)."
  p + ggplot2::coord_equal() +
    ggplot2::labs(x = block$labs[1], y = block$labs[2], title = "Posterior predictive check: joint responses",
                  subtitle = "Tile: standardized residual of the observed count; label: observed count", caption = level) +
    theme_broc() + ggplot2::theme(panel.grid = ggplot2::element_blank())
}
