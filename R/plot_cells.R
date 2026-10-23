# =============================================================================
# Shared plotting machinery: posterior setup, design cells, labels
# =============================================================================

.plot_model <- function(fit) {
  model <- attr(fit, "broc_model")
  if (is.null(model)) {
    stop("The fit has no 'broc_model' attribute; fit models with fit_broc().", call. = FALSE)
  }
  model
}

.is_bivariate_family <- function(family_name) {
  family_name %in% c("bivariate_sdt", "bivariate_dp", "vrdp2d", "bivariate_cumulative")
}

.plot_setup <- function(fit, model, ndraws, re_formula, thresholds = TRUE, rows = NULL, draw_idx = NULL,
                        thresholds2 = TRUE, data = NULL, draws = NULL) {
  data_src <- if (!is.null(data)) rebuild_design_matrices(model, data)
              else if (is.null(rows)) build_data_source_from_model(model)
              else rebuild_design_matrices(model, model$data[rows, , drop = FALSE])
  if (is.null(draws)) {
    vars <- get_predict_variables(model, include_re = !identical(re_formula, NA))
    draws <- fit$draws(variables = vars, format = "matrix")
    if (!is.null(draw_idx)) draws <- draws[draw_idx, , drop = FALSE]
    else if (!is.null(ndraws) && ndraws < nrow(draws)) draws <- draws[sample.int(nrow(draws), ndraws), , drop = FALSE]
  }
  S <- nrow(draws)
  N <- data_src$N
  fam <- model$family
  tp <- compute_trial_params(draws, model, data_src, re_formula, S, N)
  th <- th2 <- NULL
  if (thresholds) {
    n_thresh_K <- if (fam$family == "cdp") fam$J + 1 else model$model_data$K
    th <- compute_thresholds(draws, model, data_src, re_formula, S, N, n_thresh_K, crit_scale = tp[["criterion_scale"]])
    if (thresholds2 && .is_bivariate_family(fam$family)) {
      th2 <- compute_thresholds_2(draws, model, data_src, re_formula, S, N)
    }
  }
  list(draws = draws, data_src = data_src, tp = tp, th = th, th2 = th2, S = S, N = N)
}

.design_units <- function(fit, model, re_formula, keys) {
  n <- length(.row_weights(model))
  probe <- .plot_setup(fit, model, NULL, re_formula, draw_idx = unique(c(1L, max(1L, fit$num_chains() * fit$metadata()$iter_sampling %/% 2L))))
  cols <- c(list(probe$data_src$is_old, probe$data_src$source, probe$data_src$item_type), keys)
  for (m in probe$tp) cols <- c(cols, list(t(m)))
  for (a in list(probe$th, probe$th2)) {
    if (!is.null(a)) cols <- c(cols, list(matrix(aperm(a, c(2, 1, seq_along(dim(a))[-(1:2)])), nrow = n)))
  }
  cols <- cols[lengths(cols) > 0]
  keymat <- do.call(cbind, lapply(cols, function(x) if (is.numeric(x) || is.matrix(x)) signif(as.matrix(x), 12) else as.character(x)))
  key <- do.call(paste, c(as.data.frame(keymat), sep = "|"))
  grp <- match(key, unique(key))
  list(rows = which(!duplicated(grp)), unit = grp)
}

.row_weights <- function(model) {
  cnt <- model$stan_data$counts
  if (is.null(cnt)) rep(1, model$model_data$N %||% length(model$stan_data$y)) else as.numeric(cnt)
}

.check_by <- function(model, by) {
  if (is.null(by)) return(NULL)
  if (is.null(model$data)) stop("The fit does not store its data, so 'by' cannot be used.", call. = FALSE)
  miss <- setdiff(by, names(model$data))
  if (length(miss)) {
    stop("'by' variable(s) not found in the data: ", paste(miss, collapse = ", "),
         ". Available: ", paste(setdiff(names(model$data), ".bivar_cum_item_type"), collapse = ", "), call. = FALSE)
  }
  by
}

.cell_of_rows <- function(model, by) {
  n <- length(model$stan_data$y)
  if (is.null(by)) return(factor(rep("All", n)))
  if (length(by) == 1L) {
    v <- model$data[[by]]
    return(if (is.factor(v)) droplevels(v) else factor(v))
  }
  interaction(model$data[by], sep = " / ", drop = TRUE, lex.order = TRUE)
}

.cell_frame <- function(model, by, cells) {
  levs <- levels(cells)
  if (is.null(by)) return(data.frame(.cell = factor(levs, levels = levs)))
  first <- match(levs, as.character(cells))
  out <- model$data[first, by, drop = FALSE]
  for (v in by) out[[v]] <- if (is.factor(model$data[[v]])) factor(out[[v]], levels = levels(droplevels(model$data[[v]]))) else out[[v]]
  out$.cell <- factor(levs, levels = levs)
  rownames(out) <- NULL
  out
}

.pair_cells <- function(model, by, cells, signal, noise) {
  levs <- levels(cells)
  enc <- model$model_data$encoding_vars %||% character(0)
  out <- list()
  for (lev in levs) {
    sig <- intersect(signal, which(cells == lev))
    if (!length(sig)) next
    noi <- intersect(noise, which(cells == lev))
    if (!length(noi) && length(by)) {
      keep <- rep(TRUE, length(noise))
      for (v in setdiff(by, enc)) {
        sv <- unique(as.character(model$data[[v]][sig]))
        nv <- as.character(model$data[[v]][noise])
        if (any(nv %in% unique(as.character(model$data[[v]][signal])))) keep <- keep & nv %in% sv
      }
      noi <- noise[keep]
    }
    if (length(noi)) out[[lev]] <- list(signal = sig, noise = noi)
  }
  out
}

.roc_roles <- function(model, response) {
  fam <- model$family$family
  sd <- model$stan_data
  if (fam %in% c("evsdt", "uvsdt", "dpsdt", "mixture", "cdp")) {
    return(list(signal = which(sd$is_old == 1), noise = which(sd$is_old == 0), what = "detection"))
  }
  if (fam == "source_mixture") {
    return(list(signal = which(sd$source == max(sd$source)), noise = which(sd$source == min(sd$source)),
                what = "source"))
  }
  if (fam %in% c("bivariate_sdt", "bivariate_dp", "vrdp2d")) {
    if (identical(response, 1) || identical(response, 1L)) {
      return(list(signal = which(sd$item_type >= 2L), noise = which(sd$item_type == 1L), what = "detection"))
    }
    return(list(signal = which(sd$item_type == 3L), noise = which(sd$item_type == 2L), what = "source"))
  }
  stop("ROC curves are not defined for the ", family_display_name(fam), " family: it has no ",
       "signal and noise items.", call. = FALSE)
}

# ---- Labels -------------------------------------------------------------------

.param_label <- function(model, p) {
  ext <- (model$family$external_aliases %||% .param_name_map(model$family$family))[[p]]
  if (is.null(ext)) p else ext
}

.n_thresh_of <- function(model, p) {
  if (p == "criterion2") model$model_data$K2 - 1L
  else if (model$family$family == "cdp") model$family$J
  else model$model_data$K - 1L
}

.crit_mid_label <- function(J) sprintf("c%d (midpoint)", as.integer(ceiling(J / 2)))

.crit_gap_hi <- function(g, J) {
  m <- as.integer(ceiling(J / 2))
  n_up <- J - m
  as.integer(ifelse(g <= n_up, m + g, m - (g - n_up) + 1L))
}

.crit_gap_label <- function(g, J) {
  hi <- .crit_gap_hi(g, J)
  sprintf("c%d to c%d", hi - 1L, hi)
}

.crit_comp_label <- function(k, J, gap_link) {
  m <- as.integer(ceiling(J / 2))
  if (k == m) return(.crit_mid_label(J))
  hi <- if (k > m) k else k + 1L
  sprintf("c%d to c%d (%s)", hi - 1L, hi, if (identical(gap_link, "softplus")) "gap" else "log gap")
}

.pretty_coef <- function(coef, data) {
  vapply(coef, function(cf) {
    if (identical(cf, "(Intercept)")) return("Intercept")
    if (grepl("^sXs", cf)) return(paste0(sub("^sXs(.*?)(_[0-9]+)?$", "\\1", cf, perl = TRUE), ", linear"))
    parts <- strsplit(cf, ":", fixed = TRUE)[[1]]
    parts <- vapply(parts, function(pt) {
      for (v in names(data)) {
        x <- data[[v]]
        if ((is.factor(x) || is.character(x)) && startsWith(pt, v)) {
          lev <- substring(pt, nchar(v) + 1L)
          if (nzchar(lev) && lev %in% as.character(unique(x))) return(paste0(v, ": ", lev))
        }
      }
      pt
    }, character(1))
    paste(parts, collapse = " x ")
  }, character(1), USE.NAMES = FALSE)
}

.interval_summary <- function(mat, prob) {
  lo <- (1 - prob) / 2
  data.frame(mean = colMeans(mat), lower = apply(mat, 2, stats::quantile, lo, names = FALSE),
             upper = apply(mat, 2, stats::quantile, 1 - lo, names = FALSE),
             lower50 = apply(mat, 2, stats::quantile, 0.25, names = FALSE),
             upper50 = apply(mat, 2, stats::quantile, 0.75, names = FALSE))
}

.colour_scales <- function(n_cells, name) {
  if (n_cells <= 1L) {
    return(list(ggplot2::scale_colour_manual(values = .broc_accent, guide = "none"),
                ggplot2::scale_fill_manual(values = .broc_accent, guide = "none")))
  }
  list(scale_colour_broc(name = name), scale_fill_broc(name = name))
}
