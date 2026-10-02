#' Summarize a Fitted SDT Model
#'
#' Produces a labelled summary of a fitted model. The printed summary shows the
#' model and sampler status, then the parameter estimates on their natural scale
#' for each design cell (as returned by [estimates()]), the regression
#' coefficients on the model scale, random-effect standard deviations,
#' random-effect correlations and smooth-term penalty standard deviations, each
#' with posterior mean, sd, interval, R-hat and bulk/tail ESS. Correlation blocks
#' with more than 15 pairs print the pairs whose interval excludes 0 and then the
#' largest; every pair is in `summary(fit)$correlations`.
#'
#' @param object A `broc_fit` object from [fit_broc()].
#' @param prob Interval width (default 0.95).
#' @param digits Number of decimals printed (default 3).
#' @param by Optional character vector of data columns defining the cells of the
#'   natural-scale estimates; see [estimates()].
#' @param threshold_natural If `TRUE`, additionally report the criterion
#'   random-effect standard deviations and correlations on the natural threshold
#'   scale rather than the internal location and gap scale.
#' @param ... Unused.
#' @return An object of class `broc_summary`: a list with `estimates`,
#'   `coefficients`, `random_sd`, `correlations` (long table) and
#'   `correlation_matrices`, `smooth_sds`, `model_info` and `diagnostics`.
#' @method summary broc_fit
#' @export
summary.broc_fit <- function(object, prob = 0.95, digits = 3, by = NULL, threshold_natural = FALSE, ...) {
  fit <- object
  model <- .plot_model(fit)
  lo <- (1 - prob) / 2
  vars <- posterior::variables(fit$draws())
  summ <- function(v) {
    if (!length(v)) return(NULL)
    s <- posterior::summarise_draws(fit$draws(variables = v), "mean", "sd",
                                    ~ stats::quantile(.x, probs = c(lo, 1 - lo), names = FALSE),
                                    "rhat", "ess_bulk", "ess_tail")
    s <- as.data.frame(s)
    data.frame(variable = s$variable, estimate = s$mean, sd = s$sd, lower = s[[4]], upper = s[[5]],
               rhat = s$rhat, ess_bulk = s$ess_bulk, ess_tail = s$ess_tail)
  }
  result <- list()
  result$estimates <- tryCatch(estimates(fit, by = by, prob = prob), error = function(e) NULL)

  cmap <- .coef_map(model, vars)
  cs <- summ(cmap$variable)
  result$coefficients <- if (!is.null(cs)) cbind(cmap[c("parameter", "term")], cs[-1]) else NULL

  blocks <- .re_blocks(model)
  sd_rows <- list(); cor_rows <- list(); cor_mats <- list()
  for (b in blocks) {
    nm <- if (b$dim == 1L && b$sd %in% vars) b$sd else paste0(b$sd, "[", seq_len(b$dim), "]")
    ok <- nm %in% vars
    if (any(ok)) {
      s <- summ(nm[ok])
      sd_rows[[length(sd_rows) + 1L]] <- cbind(group = b$group, .re_label_parts(b$labels[ok]),
                                               effect = b$labels[ok], s[-1])
    }
    if (b$dim < 2L) next
    pairs <- do.call(rbind, lapply(2:b$dim, function(i) data.frame(i = i, j = seq_len(i - 1L))))
    cv <- paste0(b$cor, "[", pairs$i, ",", pairs$j, "]")
    keep <- cv %in% vars
    if (!any(keep)) next
    s <- summ(cv[keep])
    est <- s$estimate
    fixed0 <- abs(est) < 1e-12 & s$sd < 1e-12
    pairs <- pairs[keep, , drop = FALSE][!fixed0, , drop = FALSE]
    s <- s[!fixed0, , drop = FALSE]
    if (!nrow(s)) next
    cor_rows[[length(cor_rows) + 1L]] <- cbind(block = b$title, effect_1 = b$labels[pairs$i],
                                               effect_2 = b$labels[pairs$j], s[-1])
    m <- matrix(NA_real_, b$dim, b$dim, dimnames = list(b$labels, b$labels))
    sig <- matrix(FALSE, b$dim, b$dim, dimnames = list(b$labels, b$labels))
    m[cbind(pairs$i, pairs$j)] <- s$estimate
    sig[cbind(pairs$i, pairs$j)] <- s$lower > 0 | s$upper < 0
    used <- sort(unique(c(pairs$i, pairs$j)))
    cor_mats[[b$title]] <- list(mean = m[used, used, drop = FALSE], excludes0 = sig[used, used, drop = FALSE])
  }
  result$random_sd <- if (length(sd_rows)) do.call(rbind, sd_rows)
  result$correlations <- if (length(cor_rows)) do.call(rbind, cor_rows)
  result$correlation_matrices <- cor_mats
  result$smooth_sds <- .smooth_sd_summary(model, vars, summ)
  if (isTRUE(threshold_natural)) {
    result$threshold_natural <- compute_natural_threshold_stats(fit, model, c(lo, 1 - lo), digits)
  }
  groups <- unique(vapply(blocks, function(b) b$group, character(1)))
  result$model_info <- list(
    family = model$family$family, link = model$family$link$name,
    formula = .formula_lines(model), N = model$model_data$N, K = model$model_data$K,
    K2 = model$model_data$K2, counts = !is.null(model$stan_data$counts),
    n_obs = sum(.row_weights(model)),
    groups = stats::setNames(vapply(groups, function(g) length(unique(model$data[[g]])), integer(1)), groups),
    n_chains = fit$num_chains(), n_iter = fit$metadata()$iter_sampling,
    thin_stride = attr(fit, "thin_stride") %||% 1L, draws_per_chain = attr(fit, "draws_per_chain"),
    sampler = fit$.sampler %||% "nuts", digits = digits, prob = prob)
  result$diagnostics <- tryCatch(fit$diagnostic_summary(), error = function(e) NULL)
  class(result) <- "broc_summary"
  result
}

.formula_lines <- function(model) {
  f <- model$formula
  if (is.null(f)) return(character(0))
  dep <- function(x) paste(trimws(deparse(x, width.cutoff = 500L)), collapse = " ")
  c(dep(f$primary), vapply(names(f$params), function(n) {
    d <- dep(f$params[[n]])
    if (startsWith(d, "~")) paste(n, sub("^~[[:space:]]*", "~ ", d)) else d
  }, character(1), USE.NAMES = FALSE))
}

.smooth_sd_summary <- function(model, vars, summ) {
  sd <- model$model_data$smooth_data
  if (is.null(sd)) return(NULL)
  rows <- list()
  for (pname in names(sd)) {
    for (sm in sd[[pname]]) {
      n_thresh <- sm$n_thresh %||% 0L
      for (comp in sm$components) {
        for (k in seq_along(comp$Zs_list)) {
          base <- paste0("sds_", pname, "_", comp$san_label, "_", k)
          lab <- paste0(.param_label(model, pname), ": ", comp$sm_obj$label,
                        if (length(comp$Zs_list) > 1L) paste0(" [", k, "]") else "")
          nm <- if (n_thresh > 0) paste0(base, "[", seq_len(n_thresh), "]") else base
          comp_lab <- if (n_thresh > 0) c("location", paste0("gap", seq_len(n_thresh - 1L))) else ""
          ok <- nm %in% vars
          if (!any(ok)) next
          s <- summ(nm[ok])
          rows[[length(rows) + 1L]] <- cbind(smooth = paste0(lab, ifelse(nzchar(comp_lab[ok]), paste0(", ", comp_lab[ok]), "")), s[-1])
        }
      }
    }
  }
  if (length(rows)) do.call(rbind, rows)
}

.fmt_summary_tbl <- function(df, digits = 3L) {
  if (!is.data.frame(df) || nrow(df) == 0) return(df)
  rjust <- function(v, d) {
    s <- ifelse(is.na(v), "", formatC(as.numeric(v), format = "f", digits = d))
    formatC(s, width = max(nchar(s)))
  }
  for (col in intersect(c("estimate", "sd", "lower", "upper", "mean", "median"), names(df))) {
    if (is.numeric(df[[col]])) df[[col]] <- rjust(df[[col]], digits)
  }
  if ("rhat" %in% names(df) && is.numeric(df$rhat)) df$rhat <- rjust(df$rhat, 3L)
  for (col in intersect(c("ess_bulk", "ess_tail"), names(df))) {
    if (is.numeric(df[[col]])) df[[col]] <- ifelse(is.na(df[[col]]), "", formatC(round(df[[col]]), format = "d"))
  }
  df
}

.print_tbl <- function(df, digits, indent = "  ") {
  old <- options(width = 10000L)
  on.exit(options(old))
  out <- utils::capture.output(print(.fmt_summary_tbl(df, digits), row.names = FALSE))
  cat(paste0(indent, out), sep = "\n")
}

.print_section <- function(title, note = NULL) {
  cat("\n", title, if (!is.null(note)) paste0("  (", note, ")"), "\n", strrep("-", nchar(title)), "\n", sep = "")
}

#' @export
print.broc_summary <- function(x, ...) {
  mi <- x$model_info
  digits <- mi$digits %||% 3L
  pct <- paste0(round(100 * (mi$prob %||% 0.95)), "% intervals")
  cat("\nbayesroc model summary\n")
  cat("======================\n")
  cat("Family:  ", family_display_name(mi$family), if (!is.null(mi$link)) paste0(" (", mi$link, " link)"), "\n", sep = "")
  if (length(mi$formula)) {
    cat("Formula: ", mi$formula[1], "\n", sep = "")
    for (f in mi$formula[-1]) cat("         ", f, "\n", sep = "")
  }
  data_line <- paste0(if (isTRUE(mi$counts)) paste0(mi$n_obs, " responses in ", mi$N, " rows") else paste0(mi$N, " observations"),
                      "; ", mi$K, " response categories", if (!is.null(mi$K2)) paste0(" x ", mi$K2) else "")
  if (length(mi$groups)) data_line <- paste0(data_line, "; ", paste0(names(mi$groups), ": ", mi$groups, " levels", collapse = ", "))
  cat("Data:    ", data_line, "\n", sep = "")
  cat("Draws:   ", mi$n_chains, " chains x ", mi$n_iter, " post-warmup",
      if ((mi$thin_stride %||% 1L) > 1L) paste0(", every ", .ordinal(mi$thin_stride), " of ", mi$draws_per_chain),
      "\n", sep = "")
  .print_sampler_banner(x$diagnostics, mi$sampler %||% "nuts")

  e <- x$estimates
  if (!is.null(e) && nrow(e)) {
    .print_section("ESTIMATES", paste0("natural scale, population level; ", pct))
    by_cols <- setdiff(names(e), c("parameter", "name", "criterion", "estimate", "sd", "lower", "upper",
                                   "rhat", "ess_bulk", "ess_tail"))
    for (p in unique(e$parameter)) {
      d <- e[e$parameter == p, , drop = FALSE]
      cat("\n", p, "\n", sep = "")
      keep_by <- by_cols[vapply(by_cols, function(v) any(!is.na(d[[v]])), logical(1))]
      lab_cols <- c(keep_by, if ("criterion" %in% names(d) && any(!is.na(d$criterion))) "criterion")
      tbl <- d[c(lab_cols, intersect(c("estimate", "sd", "lower", "upper", "rhat", "ess_bulk", "ess_tail"), names(d)))]
      if (!length(lab_cols)) tbl <- cbind(cell = "all", tbl)
      for (v in lab_cols) tbl[[v]] <- ifelse(is.na(tbl[[v]]), "all", as.character(tbl[[v]]))
      .print_tbl(tbl, digits)
    }
  }
  co <- x$coefficients
  if (!is.null(co) && nrow(co)) {
    same <- vapply(unique(co$parameter), function(p) .same_as_estimates(co[co$parameter == p, , drop = FALSE], e[e$parameter == p, , drop = FALSE]), logical(1))
    if (!all(same)) .print_section("COEFFICIENTS", paste0("model scale; ", pct))
    for (p in unique(co$parameter)[!same]) {
      cat("\n", p, "\n", sep = "")
      .print_tbl(co[co$parameter == p, setdiff(names(co), "parameter"), drop = FALSE], digits)
    }
    if (any(same)) {
      cat("\n", paste(unique(co$parameter)[same], collapse = ", "), ": the coefficients equal the estimates above ",
          "(summary(fit)$coefficients).\n", sep = "")
    }
  }
  rs <- x$random_sd
  if (!is.null(rs) && nrow(rs)) {
    .print_section("RANDOM EFFECTS: STANDARD DEVIATIONS", pct)
    stat_cols <- intersect(c("estimate", "sd", "lower", "upper", "rhat", "ess_bulk", "ess_tail"), names(rs))
    for (g in unique(rs$group)) {
      rg <- rs[rs$group == g, , drop = FALSE]
      for (p in unique(rg$parameter)) {
        d <- rg[rg$parameter == p, , drop = FALSE]
        cat("\n", g, ": ", p, "\n", sep = "")
        lab <- c("criterion", "term")[c(any(nzchar(d$criterion)), any(nzchar(d$term)))]
        tbl <- d[c(lab, stat_cols)]
        if (!length(lab)) tbl <- cbind(term = "(Intercept)", tbl)
        .print_tbl(tbl, digits)
      }
    }
  }
  co <- x$correlations
  if (!is.null(co) && nrow(co)) {
    .print_section("RANDOM EFFECTS: CORRELATIONS", pct)
    any_cut <- FALSE
    for (b in unique(co$block)) {
      d <- co[co$block == b, , drop = FALSE]
      tb <- .cor_print_rows(d)
      cut <- nrow(tb$rows) < nrow(d)
      any_cut <- any_cut || cut
      cat("\n", b, if (cut) sprintf("  (%d of %d pairs: the %d whose interval excludes 0, then the largest |estimate|)",
                                    nrow(tb$rows), nrow(d), sum(d$lower > 0 | d$upper < 0)), "\n", sep = "")
      .print_tbl(tb$rows, digits)
    }
    if (any_cut) cat("\n  All pairs: summary(fit)$correlations; as matrices: summary(fit)$correlation_matrices\n")
  }
  if (!is.null(x$smooth_sds) && nrow(x$smooth_sds)) {
    .print_section("SMOOTH TERMS: PENALTY STANDARD DEVIATIONS", pct)
    .print_tbl(x$smooth_sds, digits)
  }
  if (!is.null(x$threshold_natural)) {
    .print_section("NATURAL-SCALE THRESHOLD CORRELATIONS")
    print(x$threshold_natural)
  }
  invisible(x)
}

.re_label_parts <- function(labels) {
  parameter <- sub(": .*$", "", labels)
  rest <- sub("^[^:]+: ", "", labels)
  crit_re <- "^((bin [0-9]+, )?c[0-9]+ \\([^)]*\\)|bin [0-9]+ \\(location\\)|location)(, (.*))?$"
  is_crit <- grepl(crit_re, rest)
  criterion <- ifelse(is_crit, sub(crit_re, "\\1", rest), "")
  term <- ifelse(is_crit, sub(crit_re, "\\4", rest), rest)
  data.frame(parameter = parameter, criterion = criterion, term = term, stringsAsFactors = FALSE)
}

.cor_print_rows <- function(d, max_full = 15L, n_show = 10L) {
  lab <- c(d$effect_1, d$effect_2)
  param <- unique(sub(": .*$", "", lab))
  strip <- if (length(param) == 1L) function(v) sub("^[^:]+: ", "", v) else identity
  rows <- data.frame(effect_1 = strip(d$effect_2), effect_2 = strip(d$effect_1),
                     d[intersect(c("estimate", "sd", "lower", "upper", "rhat", "ess_bulk", "ess_tail"), names(d))],
                     stringsAsFactors = FALSE)
  if (nrow(rows) > max_full) {
    sig <- d$lower > 0 | d$upper < 0
    ord <- order(-sig, -abs(d$estimate))
    rows <- rows[ord[seq_len(max(min(n_show, nrow(rows)), sum(sig)))], , drop = FALSE]
  }
  rownames(rows) <- NULL
  list(rows = rows)
}

.same_as_estimates <- function(co, e) {
  if (is.null(e) || !nrow(e) || nrow(co) != nrow(e)) return(FALSE)
  isTRUE(all(abs(sort(co$estimate) - sort(e$estimate)) < 1e-10)) &&
    isTRUE(all(abs(sort(co$sd) - sort(e$sd)) < 1e-10))
}

.print_sampler_banner <- function(d, smp) {
  if (is.null(d)) return(invisible())
  ebfmi <- if (!is.null(d$ebfmi)) suppressWarnings(min(d$ebfmi, na.rm = TRUE)) else NA_real_
  ebfmi_s <- if (is.finite(ebfmi)) sprintf("%.2f", ebfmi) else "NA"
  if (identical(smp, "drghmc")) {
    st <- d$dr_stage
    cat(sprintf("Sampler (drghmc): acc %s | DR %d/%d/%d/%d (s1/s2/s3/rej)\n",
                if (!is.null(d$accept) && is.finite(d$accept)) sprintf("%.2f", d$accept) else "NA",
                as.integer(round(100 * (st["s1"] %||% 0))), as.integer(round(100 * (st["s2"] %||% 0))),
                as.integer(round(100 * (st["s3"] %||% 0))), as.integer(round(100 * (st["rej"] %||% 0)))))
    if (!is.null(st) && (st["rej"] %||% 0) > 0.30) {
      cat("  ** WARNING: high DR rejection -- adaptation may be insufficient.\n")
    }
  } else {
    ndiv <- if (!is.null(d$num_divergent)) sum(d$num_divergent) else 0L
    ntd <- if (!is.null(d$num_max_treedepth)) sum(d$num_max_treedepth) else 0L
    cat(sprintf("Sampler: %d divergent | %d max-treedepth | min E-BFMI %s\n", as.integer(ndiv), as.integer(ntd), ebfmi_s))
    if (ndiv > 0) {
      cat("  ** WARNING: divergent transitions -- treat inference as unreliable.\n")
    } else if (is.finite(ebfmi) && ebfmi < 0.3) {
      cat("  ** WARNING: low E-BFMI -- sampler may not have explored the posterior.\n")
    }
  }
  invisible()
}

#' Compute natural-scale threshold correlations from posterior draws.
#'
#' For each posterior draw, reconstructs natural-scale thresholds for each
#' participant by applying the gap->threshold transformation to the per-draw
#' RE vectors, then computes correlations directly from these natural-scale
#' draws. This is unbiased (no Jacobian approximation).
#'
#' Only outputs correlations (not SDs) since natural-scale SDs depend on
#' the specific gap values and are less interpretable.
#'
#' @noRd
compute_natural_threshold_stats <- function(fit, model, probs, digits) {
  gap_link <- if (!is.null(model$gap_link)) model$gap_link else "log"
  gap_fn <- if (gap_link == "softplus") function(x) log1p(exp(x)) else exp
  gap_param_name <- if (gap_link == "softplus") "beta_raw_gaps" else "beta_log_gaps"
  lower_q <- probs[1]
  upper_q <- probs[2]

  results <- list()

  for (crit_name in c("criterion", "criterion2")) {
    crit_data <- model$model_data[[crit_name]]
    if (is.null(crit_data)) next

    crit_random <- crit_data$random
    if (is.null(crit_random) || length(crit_random) == 0) next

    P_crit <- crit_data$n_coef
    X_crit <- crit_data$X

    if (crit_name == "criterion") {
      K <- model$model_data$K
      n_thresh <- K - 1
      gap_base <- gap_param_name
      mid_base <- "beta_thresh_mid"
    } else {
      K2 <- model$model_data$K2
      if (is.null(K2)) next
      n_thresh <- K2 - 1
      gap_base <- paste0(gap_param_name, "_2")
      mid_base <- "beta_thresh_mid_2"
    }
    if (n_thresh < 2) next

    n_gaps <- n_thresh - 1
    mid <- ceiling(n_thresh / 2)
    n_upper <- n_thresh - mid

    for (group_name in names(crit_random)) {
      re <- crit_random[[group_name]]
      D <- re$dim
      if (D < 2) next
      if (!isTRUE(re$correlated)) next
      if (!is.null(re$cor_id)) next

      n_groups <- re$n_groups
      n_re_terms <- if (!is.null(re$n_re_terms)) re$n_re_terms else 1
      if (n_groups < 3) next

      u_var <- paste0("u_", crit_name, "_", group_name)
      u_col_names <- as.vector(outer(
        paste0(u_var, "["),
        paste0(seq_len(n_groups), ","),
        FUN = paste0
      ))
      u_col_names <- as.vector(outer(
        u_col_names,
        paste0(seq_len(D), "]"),
        FUN = paste0
      ))

      gap_col_names <- character(0)
      use_1d_gaps <- FALSE
      if (n_gaps > 0) {
        test_2d <- paste0(gap_base, "[1,1]")
        has_2d <- tryCatch({ fit$draws(variables = test_2d, format = "matrix"); TRUE },
                           error = function(e) FALSE)
        if (!has_2d) {
          test_1d <- paste0(gap_base, "[1]")
          has_1d <- tryCatch({ fit$draws(variables = test_1d, format = "matrix"); TRUE },
                             error = function(e) FALSE)
          if (has_1d) use_1d_gaps <- TRUE
        }
        if (use_1d_gaps) {
          gap_col_names <- paste0(gap_base, "[", seq_len(n_gaps), "]")
        } else {
          gap_col_names <- as.vector(outer(
            paste0(gap_base, "[", seq_len(n_gaps), ","),
            paste0(seq_len(P_crit), "]"),
            FUN = paste0
          ))
        }
      }
      mid_vars <- paste0(mid_base, "[", seq_len(P_crit), "]")

      all_vars <- c(u_col_names, gap_col_names, mid_vars)
      draws <- tryCatch(
        fit$draws(variables = all_vars, format = "matrix"),
        error = function(e) NULL
      )
      if (is.null(draws)) next
      n_draws <- nrow(draws)

      u_arr <- array(draws[, u_col_names], dim = c(n_draws, n_groups, D))

      gap_arr <- if (n_gaps > 0) {
        if (use_1d_gaps) {
          array(draws[, gap_col_names], dim = c(n_draws, n_gaps, 1))
        } else {
          array(draws[, gap_col_names], dim = c(n_draws, n_gaps, P_crit))
        }
      } else NULL

      mid_mat <- draws[, mid_vars, drop = FALSE]

      X_per_term <- matrix(0, n_re_terms, P_crit)
      if (n_re_terms == 1) {
        X_per_term[1, ] <- colMeans(X_crit)
      } else if (!is.null(re$term_idx)) {
        for (t in seq_len(n_re_terms)) {
          obs_t <- which(re$term_idx == t)
          if (length(obs_t) > 0) X_per_term[t, ] <- colMeans(X_crit[obs_t, , drop = FALSE])
        }
      } else if (!is.null(re$Z)) {
        for (t in seq_len(n_re_terms)) {
          obs_t <- which(re$Z[, t] != 0)
          if (length(obs_t) > 0) X_per_term[t, ] <- colMeans(X_crit[obs_t, , drop = FALSE])
        }
      } else {
        for (t in seq_len(n_re_terms)) X_per_term[t, ] <- colMeans(X_crit)
      }

      term_labels <- if (!is.null(re$level_names)) {
        gsub("\\(Intercept\\)", "intercept", re$level_names)
      } else {
        if (n_re_terms == 1) "intercept" else paste0("term", seq_len(n_re_terms))
      }

      D_total <- n_thresh * n_re_terms
      thresh_nat <- array(NA_real_, dim = c(n_draws, n_groups, D_total))

      col_of <- function(k, t) (k - 1L) * n_re_terms + t

      for (t in seq_len(n_re_terms)) {
        xt <- X_per_term[t, ]
        pop_mid_draws <- as.numeric(mid_mat %*% xt)
        pop_gap_draws <- if (n_gaps > 0) {
          m <- matrix(gap_arr, n_draws * n_gaps, P_crit) %*% xt
          dim(m) <- c(n_draws, n_gaps)
          m
        } else NULL

        mid_col <- col_of(mid, t)
        thresh_nat[, , mid_col] <- pop_mid_draws + u_arr[, , mid_col]

        if (n_upper > 0) {
          for (k in (mid + 1):n_thresh) {
            gap_idx <- k - mid
            this_col <- col_of(k, t)
            prev_col <- col_of(k - 1L, t)
            eff <- pop_gap_draws[, gap_idx] + u_arr[, , this_col]
            thresh_nat[, , this_col] <- thresh_nat[, , prev_col] + gap_fn(eff)
          }
        }
        if (mid > 1) {
          for (k_down in seq_len(mid - 1)) {
            k <- mid - k_down
            gap_idx <- n_upper + k_down
            this_col <- col_of(k, t)
            next_col <- col_of(k + 1L, t)
            eff <- pop_gap_draws[, gap_idx] + u_arr[, , this_col]
            thresh_nat[, , this_col] <- thresh_nat[, , next_col] - gap_fn(eff)
          }
        }
      }

      dim_labels <- character(D_total)
      for (k in seq_len(n_thresh)) {
        for (t in seq_len(n_re_terms)) {
          dim_labels[col_of(k, t)] <- if (n_re_terms == 1) {
            paste0("thresh", k)
          } else {
            paste0("thresh", k, ":", term_labels[t])
          }
        }
      }

      row_mean <- apply(thresh_nat, c(1, 3), mean)
      centered <- sweep(thresh_nat, c(1, 3), row_mean, FUN = "-")
      var_d <- apply(centered * centered, c(1, 3), sum)

      n_pairs <- D_total * (D_total - 1L) / 2L
      corr_nat_draws <- matrix(NA_real_, n_draws, n_pairs)
      pair_labels <- character(n_pairs)
      idx <- 1L
      for (i in seq_len(D_total - 1L)) {
        ci <- centered[, , i]
        vi <- var_d[, i]
        for (j in (i + 1L):D_total) {
          cj <- centered[, , j]
          denom <- sqrt(vi * var_d[, j])
          corr_nat_draws[, idx] <- ifelse(denom > 0, rowSums(ci * cj) / denom, NA_real_)
          pair_labels[idx] <- paste0(dim_labels[i], " ~ ", dim_labels[j])
          idx <- idx + 1L
        }
      }

      valid <- complete.cases(corr_nat_draws)
      if (sum(valid) < 10) next
      cv <- corr_nat_draws[valid, , drop = FALSE]

      corr_summary <- data.frame(
        pair = pair_labels,
        estimate = round(colMeans(cv), digits),
        sd = round(apply(cv, 2, sd), digits),
        lower = round(apply(cv, 2, quantile, lower_q), digits),
        upper = round(apply(cv, 2, quantile, upper_q), digits),
        stringsAsFactors = FALSE
      )

      display_crit <- if (crit_name == "criterion2") "criterion2" else "criterion"
      results[[paste0("corr_", display_crit, "_", group_name)]] <- corr_summary
    }
  }

  if (length(results) == 0) return(NULL)
  attr(results, "digits") <- digits
  class(results) <- "threshold_natural_stats"
  results
}

#' @export
print.threshold_natural_stats <- function(x, ...) {
  digits <- attr(x, "digits") %||% 3L
  for (name in names(x)) {
    crit_label <- if (grepl("^corr_criterion2_", name)) "criterion2" else "criterion"
    group <- sub("^corr_criterion2?_", "", name)
    df <- x[[name]]
    cat("\n", crit_label, " | ", group, ":\n", sep = "")
    print(.fmt_summary_tbl(df, digits), row.names = FALSE)
  }
  invisible(x)
}

