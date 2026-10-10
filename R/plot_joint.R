# Joint displays for the multi-response families: the (detection, source) latent plane with both
# criterion grids, the source ROC within each detection band, and the three-way ROC surface
# (Mossman 1999) drawn as a map. Every probability goes through the family's own joint cell
# function with custom criterion arrays; nothing here re-derives a cell.

# ---- joint cells with arbitrary criterion arrays --------------------------------------------------

.joint_family_fn <- function(fam) {
  switch(fam, bivariate_sdt = probs_bivariate_sdt, bivariate_dp = probs_bivariate_dp, vrdp2d = probs_vrdp2d,
         stop("Joint displays are available for the bivariate_sdt, bivariate_dp and vrdp2d families, not ",
              family_display_name(fam), ".", call. = FALSE))
}

# P(y1 = j, y2 = k) [S, n, K1, K2] for the given rows with detection criteria th1 [S, n, K1 - 1] and
# source criteria th2 ([S, n, K2 - 1], or [S, n, K1, K2 - 1] when the model's source criteria vary)
.joint_cells <- function(model, setup, rows, th1, th2) {
  f <- .joint_family_fn(model$family$family)
  params <- lapply(setup$tp, function(m) m[, rows, drop = FALSE])
  K1 <- dim(th1)[3] + 1L
  K2 <- dim(th2)[length(dim(th2))] + 1L
  f(params, th1, th2, setup$data_src$item_type[rows], setup$S, length(rows), K1, K2, model)
}

# a source criterion array of U scanning values for these rows, in the model's own layout
.scan_th2 <- function(model, S, n, K1, vals) {
  U <- if (is.matrix(vals)) ncol(vals) else length(vals)
  a <- if (is.matrix(vals)) array(vals[, rep(seq_len(U), each = n)], c(S, n, U)) else array(rep(vals, each = S * n), c(S, n, U))
  if (isTRUE(model$model_data$varying_source_criteria)) array(a[, , rep(seq_len(U), each = K1)], c(S, n, K1, U))
  else a
}

# the three item-type row sets of a cell, collapsed to distinct rows with summed weights
.joint_rows <- function(setup, cells, lev, w) {
  it <- setup$data_src$item_type
  rows <- which(cells == lev & it %in% 1:3)
  if (!length(rows)) return(NULL)
  red <- .collapse_rows(setup, rows, w)
  list(rows = red$rows, w = red$w, it = it[red$rows])
}

.wmean_rows <- function(p, w, idx) {
  # weighted mean over the row axis (2) of an [S, n, ...] array for rows idx
  if (!length(idx)) return(NULL)
  d <- dim(p)
  sub <- p[, idx, , drop = FALSE]
  out <- apply(sub, c(1, 3), function(v) sum(v * w[idx]))
  out / sum(w[idx])
}

# ---- source ROC within detection bands -------------------------------------------------------------

.roc_conditional <- function(model, setup, cells, cf, w, by, scale, prob, observed) {
  md <- model$model_data
  K1 <- md$K; K2 <- md$K2
  if (K2 < 3) stop("A source ROC needs at least three source categories; this model has ", K2, ".", call. = FALSE)
  y1 <- model$stan_data$y; y2 <- model$stan_data$y2
  it_all <- setup$data_src$item_type
  u <- .roc_u_grid(K2 - 1L)
  curves <- list(); pts <- list()
  for (lev in levels(cells)) {
    jr <- .joint_rows(setup, cells, lev, w)
    if (is.null(jr) || !any(jr$it == 2L) || !any(jr$it == 3L)) next
    S <- setup$S; n <- length(jr$rows)
    th1 <- setup$th[, jr$rows, , drop = FALSE]
    th2 <- setup$th2
    if (length(dim(th2)) == 4L) {
      sc <- array(0, c(S, n, K1, length(u)))
      for (j in seq_len(K1)) sc[, , j, ] <- .criterion_path(array(th2[, jr$rows, j, ], c(S, n, K2 - 1L)), u)
    } else {
      sc <- .criterion_path(th2[, jr$rows, , drop = FALSE], u)
    }
    p <- .joint_cells(model, setup, jr$rows, th1, sc)            # [S, n, K1, U + 1]
    U <- length(u)
    # P(band j, S > u_m) = mass in the source categories above m; band mass = all of them
    surv <- array(0, c(S, n, K1, U))
    tot <- p[, , , U + 1L]
    for (m in rev(seq_len(U))) {
      surv[, , , m] <- tot
      tot <- tot + p[, , , m]
    }
    for (j in seq_len(K1)) {
      hr <- .wmean_rows(surv[, , j, , drop = FALSE][, , 1, ], jr$w, which(jr$it == 3L))
      fa <- .wmean_rows(surv[, , j, , drop = FALSE][, , 1, ], jr$w, which(jr$it == 2L))
      band_B <- .wmean_rows(array(tot[, , j], c(S, n, 1)), jr$w, which(jr$it == 3L))[, 1]
      band_A <- .wmean_rows(array(tot[, , j], c(S, n, 1)), jr$w, which(jr$it == 2L))[, 1]
      hr <- hr / band_B; fa <- fa / band_A
      cv <- .roc_band(fa, hr, scale, prob)
      curves[[length(curves) + 1L]] <- cbind(cv, .cell = lev, .band = j)
      if (observed) {
        sig <- which(cells == lev & it_all == 3L & y1 == j); noi <- which(cells == lev & it_all == 2L & y1 == j)
        if (length(sig) && length(noi)) {
          ob <- .roc_observed(y2, w, sig, noi, K2, scale)
          pts[[length(pts) + 1L]] <- cbind(ob, .cell = lev, .band = j, .n = sum(w[c(sig, noi)]))
        }
      }
    }
  }
  if (!length(curves)) stop("No cell has both source-A and source-B items.", call. = FALSE)
  curves <- merge(do.call(rbind, curves), cf, by = ".cell", sort = FALSE)
  pts <- if (length(pts)) merge(do.call(rbind, pts), cf, by = ".cell", sort = FALSE)
  n_by_band <- if (!is.null(pts)) tapply(pts$.n, pts$.band, function(v) v[1]) else NULL
  lab <- vapply(seq_len(K1), function(j) paste0("Detection ", j,
    if (!is.null(n_by_band) && !is.na(n_by_band[as.character(j)])) paste0("  (n = ", round(n_by_band[as.character(j)]), ")") else ""),
    character(1))
  for (nm in c("curves", "pts")) {
    x <- get(nm)
    if (!is.null(x)) {
      x$.cell <- factor(x$.cell, levels = levels(cf$.cell))
      x$.band <- factor(lab[x$.band], levels = lab)
      assign(nm, x)
    }
  }
  list(curves = curves, pts = pts, n_cells = nlevels(cf$.cell))
}

.roc_conditional_plot <- function(parts, by, scale, prob, re_formula) {
  curves <- parts$curves; pts <- parts$pts
  z <- scale == "z"
  colour_var <- if (length(by)) by[1] else ".cell"
  p <- ggplot2::ggplot(curves, ggplot2::aes(x = .data$x, y = .data$y, colour = .data[[colour_var]],
                                            fill = .data[[colour_var]], group = .data$.cell)) +
    ggplot2::geom_abline(slope = 1, intercept = 0, colour = "grey70", linetype = "22", linewidth = 0.4) +
    ggplot2::geom_ribbon(ggplot2::aes(ymin = .data$lower, ymax = .data$upper), alpha = 0.15, colour = NA) +
    ggplot2::geom_line(linewidth = 0.8)
  if (!is.null(pts)) {
    p <- p +
      ggplot2::geom_linerange(data = pts, ggplot2::aes(ymin = .data$ylo, ymax = .data$yhi), linewidth = 0.3) +
      ggplot2::geom_linerange(data = pts, ggplot2::aes(xmin = .data$xlo, xmax = .data$xhi), orientation = "y", linewidth = 0.3) +
      ggplot2::geom_point(data = pts, shape = 21, fill = "white", size = 1.9, stroke = 0.8)
  }
  p <- if (length(by) > 1L) p + ggplot2::facet_grid(stats::as.formula(paste(paste(by[-1], collapse = " + "), "~ .band")))
       else p + ggplot2::facet_wrap(~ .band)
  if (z) {
    rng <- if (!is.null(pts)) range(c(pts$x, pts$y), na.rm = TRUE) + c(-0.75, 0.75) else range(c(curves$x, curves$y), na.rm = TRUE)
    p <- p + ggplot2::coord_equal(xlim = rng, ylim = rng) +
      ggplot2::labs(x = "z(source A items called B)", y = "z(source B items called B)")
  } else {
    p <- p + ggplot2::coord_equal(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
      ggplot2::scale_x_continuous(breaks = seq(0, 1, 0.25), labels = c("0", ".25", ".5", ".75", "1")) +
      ggplot2::scale_y_continuous(breaks = seq(0, 1, 0.25), labels = c("0", ".25", ".5", ".75", "1")) +
      ggplot2::labs(x = "Source A items called B", y = "Source B items called B")
  }
  level <- if (identical(re_formula, NA)) "Model curve for a typical participant (random effects at zero)."
           else "Model curve averaged over the observed data."
  p + .colour_scales(parts$n_cells, if (length(by)) by[1] else NULL) +
    ggplot2::labs(title = paste0(if (z) "zROC" else "ROC", ": source memory within each detection response"),
                  subtitle = paste0("Model with ", round(100 * prob), "% band",
                                    if (!is.null(pts)) "; observed with 95% intervals" else ""),
                  caption = level) +
    theme_broc()
}

# ---- three-way ROC surface as a map -----------------------------------------------------------------

# Rule (c, k): "new" if detection evidence < c, else "B" if source evidence > k, else "A". The surface
# is P(say B | B) over P(say new | new) x P(say A | A); VUS is its volume (chance 1/6).
.roc_joint <- function(model, setup, cells, cf, w, prob, observed, nx = 41L, nk = 121L) {
  md <- model$model_data
  K1 <- md$K; K2 <- md$K2
  y1 <- model$stan_data$y; y2 <- model$stan_data$y2
  it_all <- setup$data_src$item_type
  xg <- seq(0.02, 0.98, length.out = nx); yg <- xg
  cs <- stats::qnorm(xg)
  k_rng <- range(setup$th2, na.rm = TRUE) + c(-3, 3)
  kg <- seq(k_rng[1], k_rng[2], length.out = nk)
  S <- setup$S
  grids <- list(); rules <- list(); vus <- list()
  for (lev in levels(cells)) {
    jr <- .joint_rows(setup, cells, lev, w)
    if (is.null(jr) || !all(1:3 %in% jr$it)) next
    n <- length(jr$rows)
    iN <- which(jr$it == 1L); iA <- which(jr$it == 2L); iB <- which(jr$it == 3L)
    th2 <- .scan_th2(model, S, n, 2L, kg)
    Z <- array(NA_real_, c(S, nx, nx)); X <- matrix(0, S, nx)
    for (i in seq_len(nx)) {
      th1 <- array(cs[i], c(S, n, 1))
      p <- .joint_cells(model, setup, jr$rows, th1, th2)     # [S, n, 2, nk + 1]
      X[, i] <- rowSums(.wmean_rows(array(p[, , 1, ], c(S, n, nk + 1L)), jr$w, iN))
      old <- array(p[, , 2, ], c(S, n, nk + 1L))             # P(F > c, S in source category k)
      ya <- .wmean_rows(old, jr$w, iA); zb <- .wmean_rows(old, jr$w, iB)
      yA <- t(apply(ya, 1, cumsum))[, seq_len(nk), drop = FALSE]                            # P(F > c, S <= k_m | A)
      zB <- t(apply(zb, 1, function(v) rev(cumsum(rev(v)))))[, 1L + seq_len(nk), drop = FALSE]  # P(F > c, S > k_m | B)
      for (s in seq_len(S)) {
        o <- order(yA[s, ])
        Z[s, i, ] <- stats::approx(yA[s, o], zB[s, o], xout = yg, ties = "ordered", rule = 1)$y
      }
    }
    Zf <- Z; Zf[is.na(Zf)] <- 0
    vus[[lev]] <- apply(Zf, 1, mean)                           # volume over the unit square
    zbar <- apply(Z, c(2, 3), function(v) if (mean(is.na(v)) > 0.5) NA_real_ else mean(v, na.rm = TRUE))
    grids[[lev]] <- data.frame(x = rep(xg, times = nx), y = rep(yg, each = nx), z = as.vector(zbar), .cell = lev)
    if (observed) {
      rn <- which(cells == lev & it_all == 1L); ra <- which(cells == lev & it_all == 2L); rb <- which(cells == lev & it_all == 3L)
      rl <- expand.grid(i = seq_len(K1 - 1L), j = seq_len(K2 - 1L))
      rules[[lev]] <- data.frame(
        x = vapply(seq_len(nrow(rl)), function(r) sum(w[rn][y1[rn] <= rl$i[r]]) / sum(w[rn]), 1),
        y = vapply(seq_len(nrow(rl)), function(r) sum(w[ra][y1[ra] > rl$i[r] & y2[ra] <= rl$j[r]]) / sum(w[ra]), 1),
        z = vapply(seq_len(nrow(rl)), function(r) sum(w[rb][y1[rb] > rl$i[r] & y2[rb] > rl$j[r]]) / sum(w[rb]), 1),
        .cell = lev)
    }
  }
  if (!length(grids)) stop("No cell has new, source-A and source-B items.", call. = FALSE)
  lo <- (1 - prob) / 2
  vus_df <- data.frame(.cell = names(vus), vus = vapply(vus, mean, 1),
                       lower = vapply(vus, stats::quantile, 1, probs = lo, names = FALSE),
                       upper = vapply(vus, stats::quantile, 1, probs = 1 - lo, names = FALSE))
  grid <- merge(do.call(rbind, grids), cf, by = ".cell", sort = FALSE)
  pts <- if (observed) merge(do.call(rbind, rules), cf, by = ".cell", sort = FALSE)
  for (nm in c("grid", "pts")) {
    x <- get(nm)
    if (!is.null(x)) { x$.cell <- factor(x$.cell, levels = levels(cf$.cell)); assign(nm, x) }
  }
  list(grid = grid, pts = pts, vus = vus_df, n_cells = nlevels(cf$.cell))
}

.roc_joint_plot <- function(parts, by, prob) {
  grid <- parts$grid; pts <- parts$pts; vus <- parts$vus
  g <- grid[!is.na(grid$z), ]
  p <- ggplot2::ggplot(g, ggplot2::aes(x = .data$x, y = .data$y)) +
    ggplot2::geom_raster(ggplot2::aes(fill = .data$z), interpolate = TRUE) +
    ggplot2::geom_contour(ggplot2::aes(z = .data$z), breaks = seq(0.1, 0.9, 0.1), colour = "white", linewidth = 0.3) +
    ggplot2::geom_abline(slope = -1, intercept = 1, colour = "grey40", linetype = "22", linewidth = 0.4)
  # one label per contour, where the level is crossed along the ray from the origin at 12 degrees, below the lowest row of rules
  lab <- do.call(rbind, lapply(split(g, g$.cell), function(gc) {
    t <- seq(0.02, 1.4, length.out = 400)
    rx <- t * cos(12 * pi / 180); ry <- t * sin(12 * pi / 180)
    xs <- sort(unique(gc$x)); ys <- sort(unique(gc$y))
    zm <- matrix(NA_real_, length(xs), length(ys))
    zm[cbind(match(gc$x, xs), match(gc$y, ys))] <- gc$z
    zr <- vapply(seq_along(t), function(k) {
      ix <- which.min(abs(xs - rx[k])); iy <- which.min(abs(ys - ry[k]))
      if (abs(xs[ix] - rx[k]) > 0.03 || abs(ys[iy] - ry[k]) > 0.03) NA_real_ else zm[ix, iy]
    }, 1)
    ok <- !is.na(zr)
    if (sum(ok) < 3) return(NULL)
    lv <- seq(0.1, 0.9, 0.1)
    lv <- lv[lv > min(zr[ok]) & lv < max(zr[ok])]
    if (!length(lv)) return(NULL)
    tt <- stats::approx(zr[ok], t[ok], xout = lv, ties = mean)$y
    data.frame(x = tt * cos(12 * pi / 180), y = tt * sin(12 * pi / 180), label = sprintf(".%d", round(10 * lv)),
               .cell = gc$.cell[1])
  }))
  if (!is.null(lab)) p <- p + ggplot2::geom_label(data = lab, ggplot2::aes(label = .data$label), size = 2.6,
                                                  label.size = 0, label.padding = ggplot2::unit(0.12, "lines"),
                                                  fill = "white", alpha = 0.75, colour = "grey15")
  if (!is.null(pts)) {
    # dark text where the surface is pale, white where it is dark
    pz <- vapply(seq_len(nrow(pts)), function(r) {
      gc <- g[g$.cell == pts$.cell[r], ]
      d <- (gc$x - pts$x[r])^2 + (gc$y - pts$y[r])^2
      if (min(d) < 0.01) gc$z[which.min(d)] else 0
    }, 1)
    pts$.txt <- ifelse(pz > 0.4, "white", "grey15")
    p <- p + ggplot2::geom_point(data = pts, colour = .broc_accent, size = 1.9) +
      ggplot2::geom_text(data = pts, ggplot2::aes(label = sprintf("%.2f", .data$z), colour = .data$.txt), size = 2.6,
                         nudge_y = 0.035, fontface = "bold", show.legend = FALSE) +
      ggplot2::scale_colour_identity()
  }
  vl <- vus; vl$lab <- sprintf("VUS = %.2f [%.2f, %.2f]", vl$vus, vl$lower, vl$upper)
  p <- p + ggplot2::geom_label(data = vl, ggplot2::aes(x = 0.97, y = 0.97, label = .data$lab), hjust = 1, vjust = 1,
                               size = 3, label.size = 0, fill = "white", alpha = 0.85)
  if (parts$n_cells > 1L) p <- p + ggplot2::facet_wrap(~ .cell)
  p + ggplot2::scale_fill_viridis_c(name = "P(say B | B)", limits = c(0, 1), breaks = seq(0, 1, 0.25),
                                    labels = c("0", ".25", ".5", ".75", "1"), option = "G", direction = -1,
                                    guide = ggplot2::guide_colourbar(barwidth = 12, barheight = 0.6)) +
    ggplot2::coord_equal(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
    ggplot2::scale_x_continuous(breaks = seq(0, 1, 0.25), labels = c("0", ".25", ".5", ".75", "1")) +
    ggplot2::scale_y_continuous(breaks = seq(0, 1, 0.25), labels = c("0", ".25", ".5", ".75", "1")) +
    ggplot2::labs(x = "P(say new | new)", y = "P(say A | A)", title = "Three-way ROC surface",
                  subtitle = paste0("Colour and contours: P(say B | B); VUS with ", round(100 * prob), "% interval (chance 1/6)",
                                    if (!is.null(pts)) "
Points: the rules of the rating grid, labelled with P(say B | B)" else ""),
                  caption = "Population level (random effects at zero).") +
    theme_broc()
}

# ---- the latent plane ------------------------------------------------------------------------------

# Gaussian components of the latent (x, y) evidence of each distinct row, from posterior-mean
# parameters: data.frame(row, weight, mu_x, mu_y, s_x, s_y, rho, type)
.joint_components <- function(model, setup, rows, w_rows) {
  fam <- model$family$family
  md <- model$model_data
  S <- setup$S; n <- length(rows)
  params <- lapply(setup$tp, function(m) matrix(colMeans(m[, rows, drop = FALSE]), 1))
  it <- setup$data_src$item_type[rows]
  if (fam == "cdp") {
    g <- function(nm, alt) if (!is.null(params[[nm]])) as.vector(params[[nm]]) else rep(alt, n)
    old <- setup$data_src$is_old[rows] == 1
    return(data.frame(row = seq_len(n), w_row = w_rows, weight = w_rows, mu_x = ifelse(old, g("dprime2", 0), 0), mu_y = ifelse(old, g("dprime", 0), 0),
                      s_x = ifelse(old, g("sigma2", 1), 1), s_y = ifelse(old, g("sigma", 1), 1), rho = 0,
                      type = ifelse(old, "Targets", "Lures"), component = ""))
  }
  if (fam == "bivariate_cumulative") {
    bs <- .biv_trial_setup(params, it, seq_len(n), 1L, md)
    return(data.frame(row = seq_len(n), w_row = w_rows, weight = w_rows, mu_x = bs$mu1, mu_y = bs$mu2, s_x = bs$sig1,
                      s_y = bs$sig2, rho = bs$rho, type = "Responses", component = ""))
  }
  type_lab <- c("New", "Source A", "Source B")[it]
  if (fam == "vrdp2d") {
    g <- function(nm, alt) if (!is.null(params[[nm]])) as.vector(params[[nm]]) else rep(alt, n)
    dF <- g("dprime", 0); dR <- dF + g("dprime2", 0); lam <- g("lambda", 0)
    s1 <- g("sigma", 1); s2 <- g("sigma2", 1); qA <- g("discrim", 0)
    src <- if (isTRUE(md$has_discrim_B)) ifelse(it == 3L, g("discrim_B", 0), qA) else ifelse(it == 3L, -qA, qA)
    new <- it == 1L
    fam_c <- data.frame(row = seq_len(n), w_row = w_rows, weight = w_rows * ifelse(new, 1, 1 - lam), mu_x = ifelse(new, 0, dF), mu_y = 0,
                        s_x = ifelse(new, 1, s1), s_y = 1, rho = 0, type = type_lab, component = ifelse(new, "", "Familiarity"))
    rec_c <- data.frame(row = seq_len(n), w_row = w_rows, weight = w_rows * ifelse(new, 0, lam), mu_x = dR, mu_y = src,
                        s_x = s1, s_y = s2, rho = 0, type = type_lab, component = "Recollection")
    return(rbind(fam_c, rec_c[!new, , drop = FALSE]))
  }
  bs <- .biv_trial_setup(params, it, seq_len(n), 1L, md)
  wt <- w_rows
  if (fam == "bivariate_dp" && !is.null(params$lambda)) wt <- wt * ifelse(it == 1L, 1, 1 - as.vector(params$lambda))
  data.frame(row = seq_len(n), w_row = w_rows, weight = wt, mu_x = bs$mu1, mu_y = bs$mu2, s_x = bs$sig1, s_y = bs$sig2, rho = bs$rho,
             type = type_lab, component = "")
}

.bvn_density <- function(x, y, mu_x, mu_y, s_x, s_y, rho) {
  zx <- (x - mu_x) / s_x; zy <- (y - mu_y) / s_y
  q <- (zx^2 - 2 * rho * zx * zy + zy^2) / (1 - rho^2)
  exp(-q / 2) / (2 * pi * s_x * s_y * sqrt(1 - rho^2))
}

.latent_joint <- function(model, setup, cells, cf, w, re_formula) {
  fam <- model$family$family
  cdp <- fam == "cdp"
  dens <- list(); crit <- list()
  for (lev in levels(cells)) {
    rows <- which(cells == lev)
    if (!length(rows)) next
    red <- .collapse_rows(setup, rows, w)
    comp <- .joint_components(model, setup, red$rows, red$w)
    # detection criteria (x) and, for the bivariate families, source criteria (y) per detection band
    thx <- apply(setup$th[, red$rows, , drop = FALSE], 3, function(m) sum(colMeans(m) * red$w) / sum(red$w))
    rng_x <- range(c(comp$mu_x - 3 * comp$s_x, comp$mu_x + 3 * comp$s_x))
    rng_x <- range(c(rng_x, thx[thx > rng_x[1] - 1 & thx < rng_x[2] + 1]))
    rng_y <- range(c(comp$mu_y - 3 * comp$s_y, comp$mu_y + 3 * comp$s_y))
    if (cdp) {
      ck <- if (!is.null(setup$tp$know_crit)) mean(colMeans(setup$tp$know_crit[, red$rows, drop = FALSE])) else NA
      cr <- if (!is.null(setup$tp$rec_crit)) mean(colMeans(setup$tp$rec_crit[, red$rows, drop = FALSE])) else 1.5
      crit[[lev]] <- data.frame(.cell = lev, kind = c(rep("tau", length(thx)), "rec_crit", if (!is.na(ck)) "know_crit"),
                                value = c(thx, cr, if (!is.na(ck)) ck), x0 = NA, x1 = NA)
      rng_y <- range(c(rng_y, cr, ck), na.rm = TRUE); rng_x <- range(c(rng_x, ck), na.rm = TRUE)
    } else {
      th2 <- setup$th2
      K1 <- model$model_data$K; K2 <- model$model_data$K2
      md <- model$model_data
      varying <- length(dim(th2)) == 4L
      new_shared <- identical(md$new_source_criteria, "shared")
      is_new <- if (new_shared) !(seq_len(K1) %in% md$old_levels) else rep(FALSE, K1)
      edges <- c(-Inf, thx, Inf)
      seg <- list()
      for (j in seq_len(K1)) for (k in seq_len(K2 - 1L)) {
        v <- .source_edge(th2, red$rows, j, k, K2, setup$S * length(red$rows), length(red$rows), varying, new_shared, is_new,
                          attr(th2, "thresh2_new"))
        v <- sum(colMeans(matrix(v, setup$S)) * red$w) / sum(red$w)
        seg[[length(seg) + 1L]] <- data.frame(.cell = lev, kind = "source", value = v, x0 = edges[j], x1 = edges[j + 1L])
      }
      seg <- do.call(rbind, seg)
      if (!varying && !new_shared) seg <- seg[!duplicated(seg$value), ]
      if (!varying && !new_shared) { seg$x0 <- -Inf; seg$x1 <- Inf }
      crit[[lev]] <- rbind(data.frame(.cell = lev, kind = "detection", value = thx, x0 = NA, x1 = NA), seg)
      rng_y <- range(c(rng_y, seg$value[seg$value > rng_y[1] - 1 & seg$value < rng_y[2] + 1]))
    }
    gx <- seq(rng_x[1], rng_x[2], length.out = 90); gy <- seq(rng_y[1], rng_y[2], length.out = 90)
    G <- expand.grid(x = gx, y = gy)
    # each item type's density, with the vrdp2d mixture components drawn separately; contours sit at
    # fractions of each curve's own peak
    tcs <- unique(comp[c("type", "component")])
    for (r0 in seq_len(nrow(tcs))) {
      tc <- c(tcs$type[r0], tcs$component[r0])
      cc <- comp[comp$type == tc[1] & comp$component == tc[2], , drop = FALSE]
      z <- numeric(nrow(G))
      for (r in seq_len(nrow(cc))) z <- z + cc$weight[r] * .bvn_density(G$x, G$y, cc$mu_x[r], cc$mu_y[r], cc$s_x[r], cc$s_y[r], cc$rho[r])
      z <- z / max(z)
      dens[[length(dens) + 1L]] <- data.frame(G, z = z, type = tc[1], component = tc[2], .cell = lev)
    }
  }
  dens <- do.call(rbind, dens)
  # crop to where some density is visible (the criteria of rarely used bands can sit far outside it)
  keep <- dens$z > 0.01
  xl <- range(dens$x[keep]); yl <- range(dens$y[keep])
  dens <- dens[dens$x >= xl[1] & dens$x <= xl[2] & dens$y >= yl[1] & dens$y <= yl[2], ]
  dens <- merge(dens, cf, by = ".cell", sort = FALSE)
  crit <- merge(do.call(rbind, crit), cf, by = ".cell", sort = FALSE)
  for (nm in c("dens", "crit")) { x <- get(nm); x$.cell <- factor(x$.cell, levels = levels(cf$.cell)); assign(nm, x) }
  dens$type <- factor(dens$type, levels = if (cdp) c("Lures", "Targets") else if (fam == "bivariate_cumulative") "Responses"
                                           else c("New", "Source A", "Source B"))
  dens$component <- factor(dens$component, levels = c("", "Familiarity", "Recollection"))
  list(dens = dens, crit = crit, cdp = cdp, n_cells = nlevels(cf$.cell), mixture = any(dens$component != ""))
}

.latent_joint_plot <- function(parts, by, re_formula, fam) {
  dens <- parts$dens; crit <- parts$crit
  xr <- range(dens$x); yr <- range(dens$y)
  p <- ggplot2::ggplot(dens, ggplot2::aes(x = .data$x, y = .data$y))
  p <- if (parts$mixture) {
    p + ggplot2::geom_contour(ggplot2::aes(z = .data$z, colour = .data$type, linetype = .data$component,
                                           group = interaction(.data$type, .data$component, .data$.cell)),
                              breaks = c(0.05, 0.2, 0.4, 0.6, 0.8), linewidth = 0.55) +
      ggplot2::scale_linetype_manual(values = c("solid", Familiarity = "solid", Recollection = "42"),
                                     breaks = c("Familiarity", "Recollection"), name = NULL,
                                     guide = ggplot2::guide_legend(override.aes = list(colour = "grey35")))
  } else {
    p + ggplot2::geom_contour(ggplot2::aes(z = .data$z, colour = .data$type, group = interaction(.data$type, .data$.cell)),
                              breaks = c(0.05, 0.2, 0.4, 0.6, 0.8), linewidth = 0.55)
  }
  if (parts$cdp) {
    tau <- crit[crit$kind == "tau", ]; cr <- crit[crit$kind == "rec_crit", ]; ck <- crit[crit$kind == "know_crit", ]
    p <- p + ggplot2::geom_abline(data = tau, ggplot2::aes(slope = -1, intercept = .data$value), colour = "grey55", linetype = "33") +
      ggplot2::geom_hline(data = cr, ggplot2::aes(yintercept = .data$value), colour = "grey25", linewidth = 0.5) +
      ggplot2::annotate("text", x = xr[2], y = cr$value[1], label = "c[R]", parse = TRUE, hjust = 1.1, vjust = -0.4, size = 3.2, colour = "grey25")
    if (nrow(ck)) p <- p + ggplot2::geom_vline(data = ck, ggplot2::aes(xintercept = .data$value), colour = "grey25", linetype = "22") +
      ggplot2::annotate("text", x = ck$value[1], y = yr[1], label = "c[K]", parse = TRUE, hjust = -0.3, vjust = 0, size = 3.2, colour = "grey25")
    labs <- ggplot2::labs(x = "Familiarity (F)", y = "Recollection (R)", title = "Latent recollection and familiarity",
                          subtitle = "Remember above the recollection criterion (solid), Know right of the know criterion (dashed);
dotted diagonals: confidence criteria on R + F",
                          colour = NULL)
  } else {
    det <- crit[crit$kind == "detection", ]; src <- crit[crit$kind == "source", ]
    src$x0[!is.finite(src$x0)] <- xr[1]; src$x1[!is.finite(src$x1)] <- xr[2]
    p <- p + ggplot2::geom_vline(data = det, ggplot2::aes(xintercept = .data$value), colour = "grey30", linetype = "22") +
      ggplot2::geom_segment(data = src, ggplot2::aes(x = .data$x0, xend = .data$x1, y = .data$value, yend = .data$value),
                            colour = "grey30", linetype = "22")
    labs <- if (fam == "bivariate_cumulative") {
      ggplot2::labs(x = "Latent dimension 1", y = "Latent dimension 2", title = "Latent bivariate distribution",
                    subtitle = "Dashed: cutpoints of each dimension", colour = NULL)
    } else {
      ggplot2::labs(x = "Detection evidence", y = "Source evidence (A to B)", title = "Latent detection and source evidence",
                    subtitle = "Dashed: detection criteria (vertical) and source criteria (horizontal, per detection response when they vary)",
                    colour = NULL)
    }
  }
  if (parts$n_cells > 1L) p <- p + ggplot2::facet_wrap(~ .cell)
  level <- if (identical(re_formula, NA)) "Typical participant (random effects at zero)." else "Averaged over the observed data."
  p + ggplot2::coord_cartesian(xlim = xr, ylim = yr, expand = FALSE) + scale_colour_broc() + labs +
    ggplot2::labs(caption = level) + theme_broc()
}
