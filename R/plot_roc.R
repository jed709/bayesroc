#' Plot Model-Implied ROC Curves
#'
#' Draws the model-implied receiver operating characteristic (ROC) for each
#' design cell, with a posterior band, alongside the observed operating points.
#'
#' The curve traces how hit and false-alarm rates change as the confidence
#' criterion moves continuously through the fitted criteria. By default it is
#' averaged over the observed rows of each cell (random effects included), so it
#' describes the same pooled responses as the observed points; at each fitted
#' criterion it passes through the model's pooled operating point. With
#' `re_formula = NA` it is the ROC of a typical participant instead.
#'
#' @param fit A `broc_fit` object from [fit_broc()].
#' @param by Optional character vector of data columns defining the curves. The
#'   first variable is mapped to colour and any others to panels. Targets of a
#'   cell are compared with the lures that share its non-encoding variables, so
#'   an encoding condition (see `encoding_vars` in [broc()]) is compared with
#'   the common lures.
#' @param response For bivariate families, `1` (default) plots the detection ROC
#'   (old versus new items) and `2` the source ROC (source B versus source A
#'   items, with high source ratings meaning source B).
#' @param scale `"probability"` (default) or `"z"` for the zROC (probit axes).
#' @param observed If `TRUE` (default), overlay the observed operating points
#'   with 95% intervals.
#' @param re_formula `NULL` (default) averages over the observed participants and
#'   items; `NA` gives the population-level (typical participant) curve.
#' @param ndraws Number of posterior draws used (default 200).
#' @param prob Width of the posterior band (default 0.95).
#' @param cores Number of cores for the curves' survival computation (default 1): the cells'
#'   signal and noise curves are independent and are spread over a cluster when `cores > 1`.
#' @param seed Optional seed for the draw subset; the global random-number state
#'   is restored afterwards.
#' @return A ggplot object.
#' @export
#' @examples
#' \dontrun{
#' plot_roc(fit)
#' plot_roc(fit, by = "condition")
#' plot_roc(fit, by = "condition", scale = "z")
#' plot_roc(fit_bivariate, response = 2)
#' }
plot_roc <- function(fit, by = NULL, response = 1, scale = c("probability", "z"), observed = TRUE,
                     re_formula = NULL, ndraws = 200, prob = 0.95, seed = NULL, cores = 1) {
  .need_ggplot2()
  restore_seed <- .seed_scope(seed)
  on.exit(restore_seed(), add = TRUE)
  scale <- match.arg(scale)
  model <- .plot_model(fit)
  by <- .check_by(model, by)
  fam <- model$family$family
  if (fam %in% c("cumulative", "bivariate_cumulative")) {
    stop("ROC curves are not defined for the ", family_display_name(fam), " family: ordinal ",
         "models without signal and noise items have no ROC.", call. = FALSE)
  }
  bivariate <- .is_bivariate_family(fam)
  if (!bivariate && !identical(as.numeric(response), 1)) {
    stop("response = 2 is only available for bivariate families.", call. = FALSE)
  }
  roles <- .roc_roles(model, response)
  K <- if (bivariate && response == 2) model$model_data$K2 else model$model_data$K
  if (K < 3) stop("An ROC needs at least three response categories; this model has ", K, ".", call. = FALSE)

  cells <- .cell_of_rows(model, by)
  pairs <- .pair_cells(model, by, cells, roles$signal, roles$noise)
  if (!length(pairs)) stop("No cell has both signal and noise items.", call. = FALSE)

  setup <- .plot_setup(fit, model, ndraws, re_formula, thresholds2 = !identical(as.numeric(response), 1))
  setup$cache <- new.env(parent = emptyenv())
  w <- .row_weights(model)
  J <- K - 1L
  u <- .roc_u_grid(J)
  y_obs <- if (bivariate && response == 2) model$stan_data$y2 else model$stan_data$y

  surv <- .roc_survival_pairs(model, setup, pairs, w, response, u, cores)
  curves <- list(); points <- list()
  for (lev in names(pairs)) {
    pr <- pairs[[lev]]
    curves[[lev]] <- cbind(.roc_band(surv[[lev]]$noise, surv[[lev]]$signal, scale, prob), .cell = lev)
    if (observed) points[[lev]] <- cbind(.roc_observed(y_obs, w, pr$signal, pr$noise, K, scale), .cell = lev)
  }
  cf <- .cell_frame(model, by, cells)
  curves <- merge(do.call(rbind, curves), cf, by = ".cell", sort = FALSE)
  pts <- if (observed) merge(do.call(rbind, points), cf, by = ".cell", sort = FALSE)
  for (nm in c("curves", "pts")) {
    x <- get(nm)
    if (!is.null(x)) { x$.cell <- factor(x$.cell, levels = levels(cf$.cell)); assign(nm, x) }
  }
  .roc_plot(curves, pts, by, scale, prob, roles$what, length(pairs), re_formula)
}

.roc_band <- function(far, hr, scale, prob) {
  lo <- (1 - prob) / 2
  if (scale == "z") {
    clip <- function(x) stats::qnorm(pmin(pmax(x, 1e-6), 1 - 1e-6))
    far <- clip(far); hr <- clip(hr)
    xg <- seq(max(apply(far, 1, min), -3), min(apply(far, 1, max), 3), length.out = 120)
  } else {
    xg <- seq(0, 1, length.out = 151)
  }
  ys <- t(vapply(seq_len(nrow(far)), function(s) {
    o <- order(far[s, ])
    stats::approx(far[s, o], hr[s, o], xout = xg, ties = mean, rule = 2)$y
  }, numeric(length(xg))))
  if (scale == "probability") {
    ys <- cbind(0, ys[, -c(1, length(xg)), drop = FALSE], 1)
  }
  data.frame(x = xg, y = colMeans(ys), lower = apply(ys, 2, stats::quantile, lo, names = FALSE),
             upper = apply(ys, 2, stats::quantile, 1 - lo, names = FALSE))
}

.roc_observed <- function(y, w, sig, noi, K, scale) {
  tail_rate <- function(idx, k) {
    n <- sum(w[idx]); x <- sum(w[idx][y[idx] >= k])
    c(rate = x / n, lo = stats::qbeta(0.025, x + 0.5, n - x + 0.5), hi = stats::qbeta(0.975, x + 0.5, n - x + 0.5))
  }
  out <- do.call(rbind, lapply(2:K, function(k) {
    h <- tail_rate(sig, k); f <- tail_rate(noi, k)
    data.frame(x = f[["rate"]], y = h[["rate"]], xlo = f[["lo"]], xhi = f[["hi"]], ylo = h[["lo"]], yhi = h[["hi"]])
  }))
  if (scale == "z") {
    z <- function(p) stats::qnorm(pmin(pmax(p, 1e-4), 1 - 1e-4))
    for (v in names(out)) out[[v]] <- z(out[[v]])
  }
  out
}

.criterion_path <- function(th, u, ext = 8) {
  d <- dim(th)
  J <- d[3]
  knots <- array(0, c(d[1], d[2], J + 2L))
  knots[, , 1] <- th[, , 1] - ext
  knots[, , 2:(J + 1L)] <- th
  knots[, , J + 2L] <- th[, , J] + ext
  out <- array(0, c(d[1], d[2], length(u)))
  for (j in seq_along(u)) {
    lo <- min(floor(u[j]), J)
    f <- u[j] - lo
    out[, , j] <- knots[, , lo + 1L] * (1 - f) + knots[, , lo + 2L] * f
  }
  out
}

.roc_survival_pairs <- function(model, setup, pairs, w, response, u, cores = 1) {
  # survival curves for every cell's signal and noise rows: list(cell = list(signal, noise)).
  # Workers get each job's rows only (collapsed to the distinct ones first), not the full
  # per-trial draws.
  jobs <- unlist(lapply(names(pairs), function(lev) list(
    list(cell = lev, role = "signal", rows = pairs[[lev]]$signal),
    list(cell = lev, role = "noise", rows = pairs[[lev]]$noise))), recursive = FALSE)
  cores <- max(1L, as.integer(cores))
  if (cores > 1L && .is_bivariate_family(model$family$family) && .bvn_native_ready()) {
    # the native kernels thread the survival computation; no cluster
    old_threads <- .bvn_native$threads
    .bvn_native$threads <- cores
    on.exit(.bvn_native$threads <- old_threads, add = TRUE)
    cores <- 1L
  }
  if (cores > 1L && length(jobs) > 1L) {
    # cells often share a row set (one noise set for every signal condition): compute each
    # distinct set once
    keys <- vapply(jobs, function(job) digest_string(paste(sort(job$rows), collapse = ",")), "")
    first <- which(!duplicated(keys))
    # the survival of a row set is a weighted sum over its rows, so each distinct set is cut
    # into chunks that spread over the workers and are summed back
    units <- list()
    for (j in first) {
      red <- .collapse_rows(setup, jobs[[j]]$rows, w)
      n_chunk <- max(1L, min(cores * 2L, ceiling(length(red$rows) / 200)))
      parts <- split(seq_along(red$rows), cut(seq_along(red$rows), n_chunk, labels = FALSE))
      for (pt in parts) {
        units[[length(units) + 1L]] <- list(
          key = keys[j], setup = .roc_rows_setup(setup, red$rows[pt]), w = red$w[pt],
          is_old = model$stan_data$is_old[red$rows[pt]])
      }
    }
    slim <- model
    slim$stan_data <- list(is_old = NULL)
    cl <- parallel::makeCluster(min(cores, length(units)))
    on.exit(parallel::stopCluster(cl), add = TRUE)
    .export_pkg_fns(cl, c(".roc_survival_rows", ".criterion_path"))
    part <- parallel::parLapply(cl, units, function(un, model, response, u) {
      model$stan_data$is_old <- un$is_old
      .roc_survival_rows(model, un$setup, seq_along(un$w), un$w, response, length(u),
                         function(th) .criterion_path(th, u))
    }, model = slim, response = response, u = u)
    ukeys <- vapply(units, function(un) un$key, "")
    res <- lapply(keys, function(k) Reduce(`+`, part[ukeys == k]))
  } else {
    res <- lapply(jobs, function(job) .roc_survival(model, setup, job$rows, w, response, u))
  }
  out <- list()
  for (i in seq_along(jobs)) out[[jobs[[i]]$cell]][[jobs[[i]]$role]] <- res[[i]]
  out
}

.roc_rows_setup <- function(setup, rows) {
  # the per-trial parts of a plot setup restricted to `rows`, renumbered 1..length(rows)
  list(S = setup$S, tp = lapply(setup$tp, function(m) m[, rows, drop = FALSE]),
       th = setup$th[, rows, , drop = FALSE],
       th2 = if (is.null(setup$th2)) NULL else .subset_thresh2(setup$th2, rows),
       data_src = lapply(setup$data_src, function(v) if (is.null(v)) NULL else v[rows]),
       cache = NULL)
}

.roc_survival <- function(model, setup, rows, w, response, u, mode = c("path", "grid")) {
  mode <- match.arg(mode)
  # the row set can run to thousands of indices; an environment key must stay short
  key <- digest_string(paste(mode, response, length(u), paste(sort(rows), collapse = ",")))
  cache <- setup$cache
  if (!is.null(cache) && !is.null(cache[[key]])) return(cache[[key]])
  red <- .collapse_rows(setup, rows, w)
  crit <- if (mode == "path") function(th) .criterion_path(th, u)
          else function(th) array(rep(u, each = dim(th)[1] * dim(th)[2]), c(dim(th)[1:2], length(u)))
  attr(crit, "u") <- u
  attr(crit, "mode") <- mode
  out <- .roc_survival_rows(model, setup, red$rows, red$w, response, length(u), crit)
  if (!is.null(cache)) assign(key, out, envir = cache)
  out
}

.collapse_rows <- function(setup, rows, w) {
  pick <- unique(c(1L, ceiling(setup$S / 2), setup$S))
  cols <- list(setup$data_src$is_old[rows], setup$data_src$source[rows], setup$data_src$item_type[rows])
  for (m in setup$tp) cols <- c(cols, list(t(m[pick, rows, drop = FALSE])))
  for (a in list(setup$th, setup$th2)) {
    if (is.null(a)) next
    sub <- if (length(dim(a)) == 4L) a[pick, rows, , , drop = FALSE] else a[pick, rows, , drop = FALSE]
    cols <- c(cols, list(matrix(aperm(sub, c(2, 1, seq_along(dim(sub))[-(1:2)])), nrow = length(rows))))
  }
  cols <- cols[lengths(cols) > 0]
  keymat <- do.call(cbind, lapply(cols, function(x) signif(as.matrix(x), 12)))
  key <- do.call(paste, c(as.data.frame(keymat), sep = "|"))
  grp <- match(key, unique(key))
  list(rows = rows[!duplicated(grp)], w = as.numeric(tapply(w[rows], grp, sum)))
}

.roc_survival_rows <- function(model, setup, rows, w_rows, response, U, crit) {
  w <- numeric(max(rows))
  w[rows] <- w_rows
  fam <- model$family$family
  S <- setup$S
  link <- model$family$link$name
  u <- attr(crit, "u")
  if (.is_bivariate_family(fam) && response == 1 && !is.null(u) && .bvn_native_ready()) {
    # the detection response's survival along the criterion path, accumulated natively
    r <- length(rows)
    pt <- .detection_margin_parts(fam, lapply(setup$tp, function(m) m[, rows, drop = FALSE]),
                                  setup$data_src$item_type[rows], S, r, model)
    th <- setup$th[, rows, , drop = FALSE]
    acc <- .C(.bvn_native$sym_surv, as.double(th), as.double(pt$mu), as.double(pt$sig),
              as.double(pt$lam %||% 0), as.double(pt$mu_rec %||% 0), as.double(u), as.double(w_rows),
              as.integer(c(S, r, dim(th)[3], length(u), pt$kind, identical(attr(crit, "mode"), "path"),
                           .native_threads())),
              8.0, acc = double(S * length(u)), NAOK = TRUE)$acc
    return(matrix(acc, S, length(u)) / sum(w_rows))
  }
  acc <- matrix(0, S, U)
  chunk <- max(1L, floor(4e6 / (S * (U + 1))))
  for (start in seq(1L, length(rows), by = chunk)) {
    idx <- rows[start:min(length(rows), start + chunk - 1L)]
    r <- length(idx)
    tps <- lapply(setup$tp, function(m) m[, idx, drop = FALSE])
    if (fam == "cdp") {
      c_path <- crit(setup$th[, idx, , drop = FALSE])
      old <- model$stan_data$is_old[idx] == 1
      mu <- tps$dprime + (tps$dprime2 %||% 0)
      sg <- sqrt((tps$sigma %||% 1)^2 + (tps$sigma2 %||% 1)^2)
      mu[, !old] <- 0
      sg[, !old] <- sqrt(2)
      surv <- 1 - stats::pnorm((c_path - as.vector(mu)) / as.vector(sg))
      for (j in seq_len(U)) acc[, j] <- acc[, j] + matrix(surv[, , j], S, r) %*% w[idx]
      next
    }
    ds <- list(N = r,
               is_old = setup$data_src$is_old[idx], source = setup$data_src$source[idx],
               item_type = setup$data_src$item_type[idx])
    if (!.is_bivariate_family(fam)) {
      pr <- compute_category_probs(tps, crit(setup$th[, idx, , drop = FALSE]), ds, model,
                                   link, S, r, U + 1L)
    } else if (response == 1) {
      pr <- compute_category_probs(tps, crit(setup$th[, idx, , drop = FALSE]), ds, model,
                                   link, S, r, U + 1L, margin = 1L)
    } else {
      t2 <- .subset_thresh2(setup$th2, idx)
      t2g <- if (length(dim(t2)) == 4L) {
        a <- array(0, c(S, r, dim(t2)[3], U))
        for (b in seq_len(dim(t2)[3])) a[, , b, ] <- crit(array(t2[, , b, ], c(S, r, dim(t2)[4])))
        a
      } else crit(t2)
      nw <- attr(t2, "thresh2_new")
      if (!is.null(nw)) attr(t2g, "thresh2_new") <- crit(array(nw, c(S, 1L, ncol(nw))))[, 1, ]
      m2 <- model
      m2$model_data$K2 <- U + 1L
      pr <- compute_category_probs(tps, setup$th[, idx, , drop = FALSE], ds, m2, link, S, r,
                                   model$model_data$K, thresh2 = t2g, margin = 2L)
    }
    acc <- acc + .tail_by_rows(pr, w[idx])
  }
  acc / sum(w[rows])
}

.subset_thresh2 <- function(t2, idx) {
  out <- if (length(dim(t2)) == 4L) t2[, idx, , , drop = FALSE] else t2[, idx, , drop = FALSE]
  if (!is.null(attr(t2, "thresh2_new"))) attr(out, "thresh2_new") <- attr(t2, "thresh2_new")
  out
}

.tail_by_rows <- function(pr, w) {
  S <- dim(pr)[1]; r <- dim(pr)[2]; C <- dim(pr)[3]
  out <- matrix(0, S, C - 1L)
  cum <- matrix(0, S, r)
  for (j in seq_len(C - 1L)) {
    cum <- cum + matrix(pr[, , j], S, r)
    out[, j] <- pmax(1 - cum, 0) %*% w
  }
  out
}

.roc_plot <- function(curves, pts, by, scale, prob, what, n_cells, re_formula) {
  z <- scale == "z"
  colour_var <- if (length(by)) by[1] else ".cell"
  p <- ggplot2::ggplot(curves, ggplot2::aes(x = .data$x, y = .data$y, colour = .data[[colour_var]],
                                            fill = .data[[colour_var]], group = .data$.cell)) +
    ggplot2::geom_abline(slope = 1, intercept = 0, colour = "grey70", linetype = "22", linewidth = 0.4) +
    ggplot2::geom_ribbon(ggplot2::aes(ymin = .data$lower, ymax = .data$upper), alpha = 0.15, colour = NA) +
    ggplot2::geom_line(linewidth = 0.9)
  if (!is.null(pts)) {
    p <- p +
      ggplot2::geom_linerange(data = pts, ggplot2::aes(ymin = .data$ylo, ymax = .data$yhi), linewidth = 0.35) +
      ggplot2::geom_linerange(data = pts, ggplot2::aes(xmin = .data$xlo, xmax = .data$xhi), orientation = "y", linewidth = 0.35) +
      ggplot2::geom_point(data = pts, shape = 21, fill = "white", size = 2.1, stroke = 0.9)
  }
  if (length(by) > 1L) p <- p + ggplot2::facet_wrap(stats::as.formula(paste("~", paste(by[-1], collapse = " + "))))
  axis_x <- if (what == "source") "Source A items called B" else "False-alarm rate"
  axis_y <- if (what == "source") "Source B items called B" else "Hit rate"
  if (z) {
    rng <- if (!is.null(pts)) range(c(pts$x, pts$y), na.rm = TRUE) + c(-0.75, 0.75)
           else range(c(curves$x, curves$y), na.rm = TRUE)
    p <- p + ggplot2::coord_equal(xlim = rng, ylim = rng) +
      ggplot2::labs(x = paste0("z(", tolower(axis_x), ")"), y = paste0("z(", tolower(axis_y), ")"))
  } else {
    p <- p + ggplot2::coord_equal(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
      ggplot2::scale_x_continuous(breaks = seq(0, 1, 0.25), labels = c("0", ".25", ".5", ".75", "1")) +
      ggplot2::scale_y_continuous(breaks = seq(0, 1, 0.25), labels = c("0", ".25", ".5", ".75", "1")) +
      ggplot2::labs(x = axis_x, y = axis_y)
  }
  level <- if (identical(re_formula, NA)) "Model curve for a typical participant (random effects at zero)."
           else "Model curve averaged over the observed data."
  title <- paste0(if (z) "zROC" else "ROC", if (what == "source") ": source memory" else "")
  p + .colour_scales(n_cells, if (length(by)) by[1] else NULL) +
    ggplot2::labs(title = title,
                  subtitle = paste0("Model with ", round(100 * prob), "% band",
                                    if (!is.null(pts)) "; observed with 95% intervals" else ""),
                  caption = level) +
    theme_broc()
}

.roc_u_grid <- function(J) {
  unique(c(seq(0, 1, length.out = 30), seq(1, J, length.out = 14L * (J - 1L) + 1L), seq(J, J + 1, length.out = 30)))
}
