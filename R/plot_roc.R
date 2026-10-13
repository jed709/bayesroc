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
#' @param response For bivariate families, which ROC to plot: `1` the detection ROC, `2` the
#'   source ROC, or both (the default, one panel each). Other families have one ROC.
#' @param type `"marginal"` (default) draws the ROC of each response dimension. For the bivariate families
#'   two joint displays are available: `"conditional"` draws the source ROC separately for the trials in each
#'   detection response (one panel per detection category), with the model-implied conditional curve, which is
#'   where the model's claim that source accuracy rises with detection confidence is tested; `"joint"` draws
#'   the three-way ROC surface (Mossman, 1999) as a map: for every decision rule "new if detection evidence is
#'   below c, else B if source evidence is above k, else A", the rate P(say B | B) is shown as colour and
#'   contours over P(say new | new) and P(say A | A), with the rules the rating grid realises as labelled
#'   points and the volume under the surface (VUS; chance 1/6) with its posterior interval. The surface is
#'   drawn for a typical participant (random effects at zero) regardless of `re_formula`.
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
plot_roc <- function(fit, by = NULL, response = NULL, type = c("marginal", "conditional", "joint"),
                     scale = c("probability", "z"), observed = TRUE,
                     re_formula = NULL, ndraws = 200, prob = 0.95, seed = NULL, cores = 1) {
  .need_ggplot2()
  restore_seed <- .seed_scope(seed)
  on.exit(restore_seed(), add = TRUE)
  scale <- match.arg(scale)
  type <- match.arg(type)
  model <- .plot_model(fit)
  by <- .check_by(model, by)
  fam <- model$family$family
  if (type != "marginal") {
    .joint_family_fn(fam)
    cells <- .cell_of_rows(model, by)
    cf <- .cell_frame(model, by, cells)
    w <- .row_weights(model)
    if (type == "joint") {
      # the surface scans both axes over a grid, so it is drawn for the typical participant
      setup <- .plot_setup(fit, model, ndraws, NA)
      return(.roc_joint_plot(.roc_joint(model, setup, cells, cf, w, prob, observed), by, prob))
    }
    setup <- .plot_setup(fit, model, ndraws, re_formula)
    return(.roc_conditional_plot(.roc_conditional(model, setup, cells, cf, w, by, scale, prob, observed), by, scale, prob, re_formula))
  }
  if (fam %in% c("cumulative", "bivariate_cumulative")) {
    stop("ROC curves are not defined for the ", family_display_name(fam), " family: ordinal ",
         "models without signal and noise items have no ROC. Use plot_distributions() for the latent ",
         "distribution and cutpoints", if (fam == "bivariate_cumulative") " (type = \"joint\" for the plane)" else "",
         ", and pp_check() for the fit.", call. = FALSE)
  }
  bivariate <- .is_bivariate_family(fam)
  responses <- .plot_responses(response, bivariate)
  cells <- .cell_of_rows(model, by)
  if (cores > 1 && .bvn_native_ready()) {
    # the native kernels thread the threshold construction and the survival computation
    old_threads <- .bvn_native$threads
    .bvn_native$threads <- as.integer(cores)
    on.exit(.bvn_native$threads <- old_threads, add = TRUE)
  }
  setup <- .plot_setup(fit, model, ndraws, re_formula, thresholds2 = any(responses == 2L))
  setup$cache <- new.env(parent = emptyenv())
  w <- .row_weights(model)
  cf <- .cell_frame(model, by, cells)
  parts <- lapply(responses, function(resp) {
    roles <- .roc_roles(model, resp)
    K <- if (bivariate && resp == 2L) model$model_data$K2 else model$model_data$K
    if (K < 3) stop("An ROC needs at least three response categories; this model has ", K, ".", call. = FALSE)
    pairs <- .pair_cells(model, by, cells, roles$signal, roles$noise)
    if (!length(pairs)) stop("No cell has both signal and noise items.", call. = FALSE)
    u <- .roc_u_grid(K - 1L)
    y_obs <- if (bivariate && resp == 2L) model$stan_data$y2 else model$stan_data$y
    surv <- .roc_survival_pairs(model, setup, pairs, w, resp, u, cores)
    curves <- list(); points <- list()
    for (lev in names(pairs)) {
      pr <- pairs[[lev]]
      curves[[lev]] <- cbind(.roc_band(surv[[lev]]$noise, surv[[lev]]$signal, scale, prob), .cell = lev)
      if (observed) points[[lev]] <- cbind(.roc_observed(y_obs, w, pr$signal, pr$noise, K, scale), .cell = lev)
    }
    dim_label <- if (roles$what == "source") "Source" else "Detection"
    curves <- cbind(merge(do.call(rbind, curves), cf, by = ".cell", sort = FALSE), .dim = dim_label)
    pts <- if (observed) cbind(merge(do.call(rbind, points), cf, by = ".cell", sort = FALSE), .dim = dim_label)
    list(curves = curves, pts = pts, what = roles$what, n_cells = length(pairs))
  })
  curves <- do.call(rbind, lapply(parts, `[[`, "curves"))
  pts <- if (observed) do.call(rbind, lapply(parts, `[[`, "pts"))
  dims <- vapply(parts, function(p) as.character(p$curves$.dim[1]), character(1))
  for (nm in c("curves", "pts")) {
    x <- get(nm)
    if (!is.null(x)) {
      x$.cell <- factor(x$.cell, levels = levels(cf$.cell))
      x$.dim <- factor(x$.dim, levels = dims)
      assign(nm, x)
    }
  }
  .roc_plot(curves, pts, by, scale, prob, if (length(parts) > 1L) "both" else parts[[1]]$what,
            max(vapply(parts, `[[`, integer(1), "n_cells")), re_formula)
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
  if (cores > 1L && .bvn_native_ready()) {
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
  pt <- NULL
  if (!is.null(u) && .bvn_native_ready()) {
    r <- length(rows)
    tps_all <- lapply(setup$tp, function(m) m[, rows, drop = FALSE])
    pt <- if (.is_bivariate_family(fam)) {
      if (response == 1) .detection_margin_parts(fam, tps_all, setup$data_src$item_type[rows], S, r, model)
    } else {
      .univariate_margin_parts(fam, link, tps_all, if (fam == "cdp") model$stan_data$is_old[rows] else setup$data_src$is_old[rows],
                               setup$data_src$source[rows], S, r, model)
    }
  }
  if (!is.null(pt)) {
    # a normal (or recollection-mixed normal) response's survival along the criterion path, accumulated natively
    th <- setup$th[, rows, , drop = FALSE]
    if (pt$kind == 3L) {
      acc <- .C(.bvn_native$sym_mixsurv, as.double(th), as.double(pt$mu), as.double(pt$sig), as.double(pt$lam),
                as.double(pt$mu_rec), as.double(pt$sig2), as.double(u), as.double(w_rows),
                as.integer(c(S, r, dim(th)[3], length(u), identical(attr(crit, "mode"), "path"), .native_threads())),
                8.0, acc = double(S * length(u)), NAOK = TRUE)$acc
      return(matrix(acc, S, length(u)) / sum(w_rows))
    }
    acc <- .C(.bvn_native$sym_surv, as.double(th), as.double(pt$mu), as.double(pt$sig),
              as.double(pt$lam %||% 0), as.double(pt$mu_rec %||% 0), as.double(u), as.double(w_rows),
              as.integer(c(S, r, dim(th)[3], length(u), pt$kind, identical(attr(crit, "mode"), "path"),
                           .native_threads())),
              8.0, acc = double(S * length(u)), NAOK = TRUE)$acc
    return(matrix(acc, S, length(u)) / sum(w_rows))
  }
  acc <- matrix(0, S, U)
  fast2 <- .is_bivariate_family(fam) && response == 2 && isTRUE(getOption("bayesroc.source_survival_fast", TRUE))
  # the direct survival holds one [S, r, U] block, so its chunks can be large enough for the threaded pnorm
  chunk <- max(1L, floor((if (fast2) 1.6e7 else 4e6) / (S * (U + 1))))
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
      surv <- 1 - .pnorm_v((c_path - as.vector(mu)) / as.vector(sg))
      dim(surv) <- dim(c_path)
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
    } else if (fam == "vrdp2d" && !is.null(u) && .bvn_native_ready() && !is.null(.bvn_native$sym_srcsurv) &&
               isTRUE(getOption("bayesroc.source_survival_native", TRUE))) {
      # the source response's survival along the criterion path, accumulated natively
      t2 <- .subset_thresh2(setup$th2, idx)
      pt <- .vrdp2d_source_parts(tps, setup$data_src$item_type[idx], S, r, model$model_data)
      th1 <- setup$th[, idx, , drop = FALSE]
      varying <- length(dim(t2)) == 4L
      res <- .C(.bvn_native$sym_srcsurv, as.double(th1), as.double(t2), as.double(pt$muF), as.double(pt$sig1),
                as.double(pt$muR), as.double(pt$lam), as.double(pt$mu2), as.double(pt$sig2), as.double(u),
                as.double(w[idx]),
                as.integer(c(S, r, dim(th1)[3], dim(t2)[length(dim(t2))], length(u), varying,
                             identical(attr(crit, "mode"), "path"), .native_threads(), 0L)),
                8.0, acc = double(S * length(u)), NAOK = TRUE)$acc
      acc <- acc + matrix(res, S, length(u))
      next
    } else if (fam %in% c("bivariate_sdt", "bivariate_dp") && !is.null(u) && .bvn_native_ready() &&
               !is.null(.bvn_native$sym_srcsurv_g) && isTRUE(getOption("bayesroc.source_survival_native", TRUE)) &&
               is.null(attr(.subset_thresh2(setup$th2, idx), "thresh2_new"))) {
      t2 <- .subset_thresh2(setup$th2, idx)
      md <- model$model_data
      st <- .biv_trial_setup(tps, setup$data_src$item_type[idx], seq_len(r), S, md)
      th1 <- setup$th[, idx, , drop = FALSE]
      varying <- length(dim(t2)) == 4L
      oldv <- as.double(st$it != 1L) + as.double(st$it == 3L)
      lam <- if (fam == "bivariate_dp") st$sub(tps$lambda) else double(S * r)
      lam2 <- if (fam == "bivariate_dp") st$sub(tps$lambda2) else double(S * r)
      res <- .C(.bvn_native$sym_srcsurv_g, as.double(th1), as.double(t2), as.double(st$mu1), as.double(st$mu2),
                as.double(st$sig1), as.double(st$sig2), as.double(st$rho), oldv, as.double(lam), as.double(lam2),
                as.double(u), as.double(w[idx]),
                as.integer(c(S, r, dim(th1)[3], dim(t2)[length(dim(t2))], length(u), varying,
                             identical(attr(crit, "mode"), "path"), .native_threads(), isTRUE(md$bounded),
                             fam == "bivariate_dp", 0L)),
                8.0, acc = double(S * length(u)), NAOK = TRUE)$acc
      acc <- acc + matrix(res, S, length(u))
      next
    } else {
      t2 <- .subset_thresh2(setup$th2, idx)
      t2g <- if (length(dim(t2)) == 4L) {
        a <- array(0, c(S, r, dim(t2)[3], U))
        for (b in seq_len(dim(t2)[3])) a[, , b, ] <- crit(array(t2[, , b, ], c(S, r, dim(t2)[4])))
        a
      } else crit(t2)
      nw <- attr(t2, "thresh2_new")
      if (!is.null(nw)) attr(t2g, "thresh2_new") <- crit(array(nw, c(S, 1L, ncol(nw))))[, 1, ]
      if (isTRUE(getOption("bayesroc.source_survival_fast", TRUE))) {
        sv <- probs_source_survival(fam, tps, setup$th[, idx, , drop = FALSE], t2g, ds$item_type, S, r,
                                    model$model_data$K, U, model, thresh2_new = attr(t2g, "thresh2_new"))
        for (j in seq_len(U)) acc[, j] <- acc[, j] + matrix(sv[, , j], S, r) %*% w[idx]
        next
      }
      m2 <- model
      m2$model_data$K2 <- U + 1L
      pr <- compute_category_probs(tps, setup$th[, idx, , drop = FALSE], ds, m2, link, S, r,
                                   model$model_data$K, thresh2 = t2g, margin = 2L)
    }
    acc <- acc + .tail_by_rows(pr, w[idx])
  }
  acc / sum(w[rows])
}

# Per-(draw, trial) parts of the univariate probit families' response distribution in the layout of
# .detection_margin_parts: a normal (kind 0), a normal with a recollected mass in the top category (kind 1), or
# two normals with their own means and SDs, the second with weight lam (kind 3); NULL for the others.
.univariate_margin_parts <- function(fam, link, tps, is_old, source, S, r, model) {
  if (!fam %in% c("evsdt", "uvsdt", "dpsdt", "cumulative", "cdp", "mixture", "source_mixture") || (fam != "cdp" && link != "probit")) return(NULL)
  one <- rep(1, S * r)
  md <- model$model_data
  vec <- function(nm, alt) if (!is.null(tps[[nm]])) as.vector(tps[[nm]]) else alt
  if (fam == "cdp") {
    old <- rep(is_old == 1, each = S)
    mu <- as.vector(tps$dprime + (tps$dprime2 %||% 0))
    sig <- as.vector(sqrt((tps$sigma %||% 1)^2 + (tps$sigma2 %||% 1)^2)) * one
    return(list(mu = ifelse(old, mu, 0), sig = ifelse(old, sig, sqrt(2)), lam = NULL, mu_rec = NULL, kind = 0L))
  }
  if (fam == "cumulative") return(list(mu = as.vector(tps$dprime), sig = one, lam = NULL, mu_rec = NULL, kind = 0L))
  if (fam == "source_mixture") {
    if (is.null(.bvn_native$sym_mixsurv)) return(NULL)
    B <- rep(source != 0, each = S)
    d <- as.vector(tps$dprime)
    dB <- if (isTRUE(md$has_dprime_B)) as.vector(tps$dprime_B) else -d
    lam <- as.vector(tps$lambda)
    lamB <- if (isTRUE(md$has_lambda_B)) as.vector(tps$lambda_B) else lam
    return(list(mu = 0 * one, sig = one, lam = ifelse(B, lamB, lam), mu_rec = ifelse(B, dB, d), sig2 = one, kind = 3L))
  }
  io <- rep(is_old, each = S)
  old <- io > 0
  mu <- io * as.vector(tps$dprime)
  if (fam == "mixture") {
    if (is.null(.bvn_native$sym_mixsurv)) return(NULL)
    lure <- isTRUE(md$has_lure_mixture) && !is.null(tps$dprime_L) && !is.null(tps$lambda_L)
    mu2 <- vec("dprime2", 0 * one) - (1 - io) * as.vector(tps$dprime)
    out <- list(mu = ifelse(old, mu2, if (lure) 0 else mu), sig = ifelse(old, vec("sigma2", one), 1),
                lam = ifelse(old, as.vector(tps$lambda), if (lure) as.vector(tps$lambda_L) else 0),
                mu_rec = ifelse(old, mu, if (lure) -as.vector(tps$dprime_L) else 0),
                sig2 = ifelse(old, vec("sigma", one), if (lure) vec("sigma_L", one) else 1), kind = 3L)
    return(out)
  }
  sig <- one
  if (fam == "uvsdt" || (fam == "dpsdt" && isTRUE(md$has_sigma))) sig <- ifelse(old, as.vector(tps$sigma), 1)
  lam <- if (fam == "dpsdt") ifelse(old, as.vector(tps$lambda), 0) else NULL
  list(mu = mu, sig = sig, lam = lam, mu_rec = NULL, kind = if (fam == "dpsdt") 1L else 0L)
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
  both <- what == "both"
  if (both && length(by) > 1L) {
    p <- p + ggplot2::facet_grid(stats::as.formula(paste(paste(by[-1], collapse = " + "), "~ .dim")))
  } else if (both) {
    p <- p + ggplot2::facet_wrap(~ .dim)
  } else if (length(by) > 1L) {
    p <- p + ggplot2::facet_wrap(stats::as.formula(paste("~", paste(by[-1], collapse = " + "))))
  }
  axis_x <- switch(what, source = "Source A items called B", both = "False-alarm rate (detection); A items called B (source)",
                   "False-alarm rate")
  axis_y <- switch(what, source = "Source B items called B", both = "Hit rate (detection); B items called B (source)",
                   "Hit rate")
  if (z) {
    rng <- if (!is.null(pts)) range(c(pts$x, pts$y), na.rm = TRUE) + c(-0.75, 0.75)
           else range(c(curves$x, curves$y), na.rm = TRUE)
    p <- p + ggplot2::coord_equal(xlim = rng, ylim = rng) +
      ggplot2::labs(x = paste0("z(", if (both) axis_x else tolower(axis_x), ")"),
                    y = paste0("z(", if (both) axis_y else tolower(axis_y), ")"))
  } else {
    p <- p + ggplot2::coord_equal(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
      ggplot2::scale_x_continuous(breaks = seq(0, 1, 0.25), labels = c("0", ".25", ".5", ".75", "1")) +
      ggplot2::scale_y_continuous(breaks = seq(0, 1, 0.25), labels = c("0", ".25", ".5", ".75", "1")) +
      ggplot2::labs(x = axis_x, y = axis_y)
  }
  level <- if (identical(re_formula, NA)) "Model curve for a typical participant (random effects at zero)."
           else "Model curve averaged over the observed data."
  title <- paste0(if (z) "zROC" else "ROC", switch(what, source = ": source memory", both = ": detection and source memory", ""))
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

# Per-(draw, trial) parts of the vrdp2d source survival for the native kernel: familiarity and recollection
# detection means, the shared detection SD, the recollection weight, and the source mean/SD of the
# recollected component (new items: mean 0, SD 1, weight 0), each [S, r].
.vrdp2d_source_parts <- function(tps, item_type, S, r, md) {
  it <- matrix(rep(item_type, each = S), S, r)
  isnew <- it == 1L; isB <- it == 3L
  one <- matrix(1, S, r)
  dF <- tps$dprime; dR <- dF + tps$dprime2; lam <- tps$lambda
  s1 <- if (!is.null(tps$sigma)) tps$sigma else one
  s2 <- if (!is.null(tps$sigma2)) tps$sigma2 else one
  qA <- tps$discrim
  src <- if (isTRUE(md$has_discrim_B)) ifelse(isB, tps$discrim_B, qA) else ifelse(isB, -qA, qA)
  list(muF = ifelse(isnew, 0, dF), sig1 = ifelse(isnew, 1, s1), muR = ifelse(isnew, 0, dR),
       lam = ifelse(isnew, 0, lam), mu2 = ifelse(isnew, 0, src), sig2 = ifelse(isnew, 1, s2))
}

# Native vrdp2d source survival against the R probs_source_survival on rows of a plot setup.
.parity_src_native <- function(model, setup, rows, u) {
  acc_r <- function(on) {
    options(bayesroc.source_survival_native = on)
    .roc_survival_rows(model, setup, rows, rep(1, length(rows)), 2, length(u), structure(function(th) .criterion_path(th, u), u = u, mode = "path"))
  }
  old <- getOption("bayesroc.source_survival_native", TRUE); on.exit(options(bayesroc.source_survival_native = old), add = TRUE)
  a <- acc_r(TRUE); b <- acc_r(FALSE)
  c(max_abs_diff = max(abs(a - b)), cells = length(a))
}

# The same on the absolute criterion grid of the distribution plots.
.parity_src_native_grid <- function(model, setup, rows, u) {
  crit <- function(th) array(rep(u, each = dim(th)[1] * dim(th)[2]), c(dim(th)[1:2], length(u)))
  attr(crit, "u") <- u; attr(crit, "mode") <- "grid"
  acc_r <- function(on) {
    options(bayesroc.source_survival_native = on)
    .roc_survival_rows(model, setup, rows, rep(1, length(rows)), 2, length(u), crit)
  }
  old <- getOption("bayesroc.source_survival_native", TRUE); on.exit(options(bayesroc.source_survival_native = old), add = TRUE)
  a <- acc_r(TRUE); b <- acc_r(FALSE)
  c(max_abs_diff = max(abs(a - b)), cells = length(a))
}
