# =============================================================================
# Information-scaled smooth coefficients (broc(smooth_info = TRUE))
# =============================================================================

.smooth_info_eligible <- function(comp) {
  !inherits(comp$sm_obj, c("fs.interaction", "random.effect", "sz.interaction", "mrf.smooth"))
}

.smooth_info_base <- function(pname, comp, k) paste0(pname, "_", comp$san_label, "_", k)

.smooth_info_blocks <- function(model_data) {
  out <- list()
  for (pname in names(model_data$smooth_data)) {
    for (sm in model_data$smooth_data[[pname]]) {
      nth <- sm$n_thresh %||% 0L
      for (comp in sm$components) {
        if (!.smooth_info_eligible(comp)) next
        for (k in seq_along(comp$Zs_list)) {
          out[[.smooth_info_base(pname, comp, k)]] <- list(pname = pname, n_thresh = nth, Zs = comp$Zs_list[[k]])
        }
      }
    }
  }
  out
}

.pilot_variables <- function(model) {
  md <- model$model_data
  gl <- model$gap_link %||% "log"
  vars <- get_predict_variables(model, include_re = FALSE)
  vars <- vars[!grepl("^s_", vars)]
  n_gaps <- if (model$family$family == "cdp") model$family$J - 1 else md$K - 2
  n_gaps2 <- if (!is.null(md$K2)) md$K2 - 2 else 0L
  grid <- function(prefix, rows, cols) {
    as.vector(outer(rows, cols, function(r, c) paste0(prefix, "[", r, ",", c, "]")))
  }
  expand <- list()
  expand[[.gap_param(gl)]] <- grid(.gap_param(gl), seq_len(n_gaps), seq_len(md$criterion$n_coef))
  if (isTRUE(md$has_criterion2)) {
    if (!is.null(md$criterion2)) {
      expand[[.gap_param(gl, "_2")]] <- grid(.gap_param(gl, "_2"), seq_len(n_gaps2), seq_len(md$criterion2$n_coef))
    }
    expand[[.gap_param(gl, "_2_varying")]] <- grid(.gap_param(gl, "_2_varying"), seq_len(md$stan_data$n_vbin %||% md$K), seq_len(n_gaps2))
    expand[[.gap_param(gl, "_2_cov")]] <- grid(.gap_param(gl, "_2_cov"), seq_len(n_gaps2),
                                               seq_len(md$stan_data$P_criterion2_cov %||% 0L))
    expand[[.gap_param(gl, "_2_new")]] <- paste0(.gap_param(gl, "_2_new"), "[", seq_len(n_gaps2), "]")
  }
  unlist(lapply(vars, function(v) if (!is.null(expand[[v]])) expand[[v]] else v), use.names = FALSE)
}

.obs_loglik_r <- function(model, data_src, draws, re_formula = NA) {
  S <- nrow(draws); N <- data_src$N
  md <- model$model_data
  fam <- model$family
  K <- md$K
  n_thresh_K <- if (fam$family == "cdp") fam$J + 1 else K
  tp <- compute_trial_params(draws, model, data_src, re_formula, S, N)
  th <- compute_thresholds(draws, model, data_src, re_formula, S, N, n_thresh_K)
  bivariate <- fam$family %in% c("bivariate_sdt", "bivariate_dp", "vrdp2d", "bivariate_cumulative")
  th2 <- if (bivariate) compute_thresholds_2(draws, model, data_src, re_formula, S, N)
  pr <- compute_category_probs(tp, th, data_src, model, fam$link$name, S, N, K, thresh2 = th2)
  y <- md$stan_data$y
  out <- matrix(0, S, N)
  if (bivariate) {
    y2 <- md$stan_data$y2
    for (n in seq_len(N)) out[, n] <- pr[, n, y[n], y2[n]]
  } else if (fam$family == "cdp") {
    pR <- attr(pr, "p_remember"); pK <- attr(pr, "p_know")
    rk <- md$stan_data$rk
    for (n in seq_len(N)) {
      marg <- pr[, n, y[n]]
      if (y[n] %in% fam$old_levels && !is.na(rk[n])) {
        cond <- if (rk[n] == 1L) pR[, n, y[n]]
                else if (rk[n] == 2L) (if (!is.null(pK)) pK[, n, y[n]] else 1 - pR[, n, y[n]])
                else 1 - pR[, n, y[n]] - (if (!is.null(pK)) pK[, n, y[n]] else 0)
        marg <- marg * cond
      }
      out[, n] <- marg
    }
  } else {
    for (n in seq_len(N)) out[, n] <- pr[, n, y[n]]
  }
  log(pmax(out, 0))
}

.smooth_info_pilot <- function(model) {
  blocks <- .smooth_info_blocks(model$model_data)
  if (!length(blocks)) return(NULL)
  md <- model$model_data
  src <- build_data_source_from_model(model)
  N <- src$N
  cnt <- md$stan_data$counts %||% rep(1, N)
  fx <- .pilot_variables(model)
  svars <- grep("^s_", get_predict_variables(model, include_re = FALSE), value = TRUE)
  base <- matrix(0, 1, length(fx) + length(svars), dimnames = list(NULL, c(fx, svars)))
  total <- function(D, s = src) drop(.obs_loglik_r(model, s, D) %*% cnt)
  nll <- function(theta) {
    b <- base; b[1, fx] <- theta
    v <- -total(b)
    if (is.finite(v)) v else .Machine$double.xmax
  }
  grad <- function(theta, h = 1e-5) {
    P <- length(theta)
    D <- base[rep(1, 2 * P), , drop = FALSE]
    D[, fx] <- matrix(theta, 2 * P, P, byrow = TRUE)
    D[cbind(seq(1, 2 * P, 2), seq_len(P))] <- theta + h
    D[cbind(seq(2, 2 * P, 2), seq_len(P))] <- theta - h
    L <- total(D)
    g <- -(L[seq(1, 2 * P, 2)] - L[seq(2, 2 * P, 2)]) / (2 * h)
    g[!is.finite(g)] <- 0
    g
  }
  theta0 <- rep(0, length(fx))
  kc <- grep("^beta_know_crit\\[", fx)
  at <- function(theta) { b <- base; b[1, fx] <- theta; b }
  if (length(kc) && !is.finite(total(at(theta0)))) {
    v <- md$stan_data$kc_v %||% c(1, rep(0, length(kc) - 1))
    for (step in seq_len(40)) {
      theta0[kc] <- theta0[kc] + 0.5 * v
      if (is.finite(total(at(theta0)))) break
    }
  }
  opt <- stats::optim(theta0, nll, grad, method = "BFGS", control = list(maxit = 1000, reltol = 1e-10))
  base[1, fx] <- opt$par

  h <- 1e-3
  info <- list()
  rows_of <- split(names(blocks), vapply(blocks, function(b) b$pname, character(1)))
  for (pname in names(rows_of)) {
    nth <- blocks[[rows_of[[pname]][1]]]$n_thresh
    R <- if (nth > 0) nth else 1L
    s2 <- src
    s2$smooth_data[[pname]] <- c(src$smooth_data[[pname]],
                                 list(list(n_thresh = nth, components = list(list(
                                   san_label = "pilotshift", Zs_list = list(matrix(1, N, 1)))))))
    sname <- paste0("s_", pname, "_pilotshift_1")
    cols <- if (nth > 0) paste0(sname, "[", seq_len(R), ",1]") else paste0(sname, "[1]")
    D <- cbind(base[rep(1, 1 + 2 * R), , drop = FALSE],
               matrix(0, 1 + 2 * R, R, dimnames = list(NULL, cols)))
    for (r in seq_len(R)) {
      D[2 * r, cols[r]] <- h
      D[2 * r + 1, cols[r]] <- -h
    }
    L <- .obs_loglik_r(model, s2, D)
    dn <- vapply(seq_len(R), function(r) {
      x <- -(L[2 * r, ] - 2 * L[1, ] + L[2 * r + 1, ]) / h^2 * cnt
      x[!is.finite(x)] <- 0
      x
    }, numeric(N))
    dn <- matrix(dn, N, R)
    for (b in rows_of[[pname]]) {
      Zs <- as.matrix(blocks[[b]]$Zs)
      mats <- lapply(seq_len(R), function(r) {
        I <- crossprod(Zs, Zs * dn[, r])
        I <- (I + t(I)) / 2
        e <- eigen(I, symmetric = TRUE)
        e$vectors %*% (pmax(e$values, 0) * t(e$vectors))
      })
      if (nth > 0) {
        arr <- array(0, c(R, ncol(Zs), ncol(Zs)))
        for (r in seq_len(R)) arr[r, , ] <- mats[[r]]
        info[[b]] <- arr
      } else {
        info[[b]] <- mats[[1]]
      }
    }
  }
  list(info = info, pilot = stats::setNames(opt$par, fx), converged = opt$convergence == 0L)
}

.apply_smooth_info <- function(smooth_info, model_data, family, gap_link, data) {
  if (!isTRUE(smooth_info)) return(list(model_data = model_data, pilot = NULL))
  pilot <- .smooth_info_pilot(list(model_data = model_data, family = family,
                                   gap_link = gap_link, data = data))
  if (is.null(pilot)) {
    warning("smooth_info = TRUE has no effect: the model has no population-level smooth terms.",
            call. = FALSE)
    return(list(model_data = model_data, pilot = NULL))
  }
  if (!pilot$converged) {
    warning("smooth_info: the maximum-likelihood pilot did not converge; the information ",
            "scaling is still exact but may sample less efficiently.", call. = FALSE)
  }
  model_data$smooth_info_bases <- names(pilot$info)
  for (b in names(pilot$info)) model_data$stan_data[[paste0("Ism_", b)]] <- pilot$info[[b]]
  list(model_data = model_data, pilot = pilot)
}
