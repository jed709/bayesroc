#' Natural-Scale Parameter Estimates per Design Cell
#'
#' Computes each parameter on its natural scale (d', standard deviations,
#' probabilities, criterion locations) for every design cell. A cell's value is
#' the average over its observed rows, so continuous covariates need no reference
#' value. Signal parameters are averaged over target items and lure parameters
#' over lures. These are the quantities shown by [plot_estimates()] and in the
#' first section of [summary()][summary.broc_fit].
#'
#' @param fit A `broc_fit` object from [fit_broc()].
#' @param pars Optional character vector of parameters (family names such as
#'   `"dprime"`, `"sigma"`, `"lambda"`, `"rec"`, `"mu"`, or `"criteria"` /
#'   `"criteria2"`). `NULL` returns all.
#' @param by Optional character vector of data columns defining the cells. `NULL`
#'   uses the factors in the fixed-effect formulas.
#' @param re_formula `NA` (default) gives population-level values; `NULL`
#'   averages over the observed participants and items as well.
#' @param summary If `TRUE` (default), one row per quantity with posterior mean,
#'   standard deviation, interval and convergence diagnostics; if `FALSE`, the
#'   posterior draws in long format.
#' @param prob Interval width (default 0.95).
#' @param ndraws Optional number of posterior draws (diagnostics are only
#'   reported when all draws are used).
#' @param seed Optional seed for the draw subset.
#' @return A data frame. Columns: `parameter` (label), `name` (the `pars` name),
#'   the `by` columns, `criterion` (for criteria), then either `estimate`, `sd`,
#'   `lower`, `upper`, `rhat`, `ess_bulk`, `ess_tail`, or `.draw` and `value`.
#'   Quantities that do not differ between cells are reported once, with the
#'   `by` columns set to `NA`.
#' @seealso [compare_estimates()], [plot_estimates()].
#' @export
#' @examples
#' \dontrun{
#' estimates(fit)
#' estimates(fit, pars = "dprime", by = "condition")
#' }
estimates <- function(fit, pars = NULL, by = NULL, re_formula = NA, summary = TRUE, prob = 0.95,
                      ndraws = NULL, seed = NULL) {
  restore_seed <- .seed_scope(seed)
  on.exit(restore_seed(), add = TRUE)
  model <- .plot_model(fit)
  ed <- .estimate_draws(fit, model, pars, by, re_formula, ndraws)
  if (!ncol(ed$draws)) stop("No parameters matched 'pars'.", call. = FALSE)
  if (!summary) {
    S <- nrow(ed$draws)
    out <- ed$info[rep(seq_len(nrow(ed$info)), each = S), , drop = FALSE]
    out$.draw <- rep(seq_len(S), times = nrow(ed$info))
    out$value <- as.vector(ed$draws)
    rownames(out) <- NULL
    return(out)
  }
  cbind(ed$info, .draw_summary(ed$draws, ed$chain, prob))
}

#' Compare Natural-Scale Estimates Between Design Cells
#'
#' Posterior differences of the natural-scale estimates from [estimates()]
#' between design cells, for every pair of cells or against a reference cell.
#'
#' @inheritParams estimates
#' @param ref Optional reference cell (a level of the `by` variable, or of their
#'   interaction separated by `" / "`). `NULL` compares all pairs.
#' @return A data frame with `parameter`, `name`, `criterion`, `contrast`, and
#'   (when `summary = TRUE`) `estimate`, `sd`, `lower`, `upper` and `p_positive`,
#'   the posterior probability that the difference is positive; otherwise
#'   `.draw` and `value`.
#' @export
#' @examples
#' \dontrun{
#' compare_estimates(fit, pars = "dprime", by = "condition")
#' compare_estimates(fit, pars = "dprime", by = "condition", ref = "read")
#' }
compare_estimates <- function(fit, pars = NULL, by = NULL, ref = NULL, re_formula = NA, summary = TRUE,
                              prob = 0.95, ndraws = NULL, seed = NULL) {
  restore_seed <- .seed_scope(seed)
  on.exit(restore_seed(), add = TRUE)
  model <- .plot_model(fit)
  ed <- .estimate_draws(fit, model, pars, by, re_formula, ndraws)
  info <- ed$info
  info$.cell <- ed$cells
  if (is.null(info$criterion)) info$criterion <- NA_character_
  key <- paste(info$name, info$criterion, sep = "\r")
  cols <- list(); rows <- list()
  for (k in unique(key)) {
    idx <- which(key == k & !is.na(info$.cell))
    if (length(idx) < 2L) next
    cells <- as.character(info$.cell[idx])
    pairs <- if (is.null(ref)) utils::combn(seq_along(idx), 2, simplify = FALSE) else {
      r <- match(ref, cells)
      if (is.na(r)) next
      lapply(setdiff(seq_along(idx), r), function(j) c(j, r))
    }
    for (pr in pairs) {
      a <- idx[pr[1]]; b <- idx[pr[2]]
      if (is.null(ref)) { a <- idx[pr[2]]; b <- idx[pr[1]] }
      cols[[length(cols) + 1L]] <- ed$draws[, a] - ed$draws[, b]
      rows[[length(rows) + 1L]] <- data.frame(parameter = info$parameter[a], name = info$name[a],
                                              criterion = info$criterion[a],
                                              contrast = paste(info$.cell[a], "-", info$.cell[b]))
    }
  }
  if (!length(cols)) stop("No parameter differs between cells; nothing to compare.", call. = FALSE)
  d <- do.call(cbind, cols)
  meta <- do.call(rbind, rows)
  if (!summary) {
    S <- nrow(d)
    out <- meta[rep(seq_len(nrow(meta)), each = S), , drop = FALSE]
    out$.draw <- rep(seq_len(S), times = nrow(meta))
    out$value <- as.vector(d)
    rownames(out) <- NULL
    return(out)
  }
  s <- .draw_summary(d, NULL, prob)
  cbind(meta, s[c("estimate", "sd", "lower", "upper")], p_positive = colMeans(d > 0))
}

.draw_summary <- function(mat, chain, prob) {
  lo <- (1 - prob) / 2
  out <- data.frame(estimate = colMeans(mat), sd = apply(mat, 2, stats::sd),
                    lower = apply(mat, 2, stats::quantile, lo, names = FALSE),
                    upper = apply(mat, 2, stats::quantile, 1 - lo, names = FALSE))
  if (!is.null(chain) && length(unique(chain)) > 1L) {
    iter <- stats::ave(seq_along(chain), chain, FUN = seq_along)
    out$rhat <- NA_real_
    out$ess_bulk <- NA_real_
    out$ess_tail <- NA_real_
    for (j in seq_len(ncol(mat))) {
      arr <- tryCatch(posterior::as_draws_array(posterior::as_draws_df(
        data.frame(x = mat[, j], .chain = chain, .iteration = iter, .draw = seq_along(chain)))),
        error = function(e) NULL)
      if (is.null(arr)) next
      out$rhat[j] <- posterior::rhat(posterior::extract_variable_matrix(arr, "x"))
      out$ess_bulk[j] <- posterior::ess_bulk(posterior::extract_variable_matrix(arr, "x"))
      out$ess_tail[j] <- posterior::ess_tail(posterior::extract_variable_matrix(arr, "x"))
    }
  }
  rownames(out) <- NULL
  out
}

.pars_internal <- function(pars, fam) {
  if (is.null(pars)) return(NULL)
  vapply(pars, function(p) {
    if (p %in% c("criteria", "criterion", "cutpoints", "cutpoints1")) return("criterion")
    if (p %in% c("criteria2", "criterion2", "cutpoints2")) return("criterion2")
    external_to_internal_param(p, fam)
  }, character(1))
}

.estimate_draws <- function(fit, model, pars, by, re_formula, ndraws) {
  by <- .check_by(model, by %||% .default_by(model))
  if (!length(by)) by <- NULL
  w <- .row_weights(model)
  cells <- .cell_of_rows(model, by)
  cf <- .cell_frame(model, by, cells)
  fam <- model$family$family
  roles <- tryCatch(.roc_roles(model, 1), error = function(e) NULL)
  n_all <- length(w)
  if (identical(re_formula, NA)) {
    role_key <- if (is.null(roles)) NULL
                else list(seq_len(n_all) %in% roles$signal, seq_len(n_all) %in% roles$noise)
    du <- .design_units(fit, model, re_formula, c(list(as.integer(cells)), role_key))
    setup <- .plot_setup(fit, model, ndraws, re_formula, rows = du$rows)
    w <- as.numeric(tapply(w, du$unit, sum))
    cells <- cells[du$rows]
    if (!is.null(roles)) {
      roles$signal <- which(du$rows %in% roles$signal)
      roles$noise <- which(du$rows %in% roles$noise)
    }
  } else {
    setup <- .plot_setup(fit, model, ndraws, re_formula)
  }
  internal <- .pars_internal(pars, fam)
  ext <- function(p) if (p %in% c("criterion", "criterion2")) sub("criterion", "criteria", p)
                     else .smooth_display_pname(model, p)
  cell_mean <- function(mat, rows) {
    vals <- vapply(levels(cells), function(lev) {
      r <- intersect(rows, which(cells == lev))
      if (!length(r)) return(rep(NA_real_, nrow(mat)))
      as.vector(mat[, r, drop = FALSE] %*% w[r]) / sum(w[r])
    }, numeric(nrow(mat)))
    matrix(vals, nrow = nrow(mat), dimnames = list(NULL, levels(cells)))
  }
  # a parameter is reported over the `by` variables it actually varies with: variables
  # along which every cell agrees are dropped one at a time (a criterion with `exp` in
  # its formula is one row per experiment, not one per condition x experiment)
  reduce <- function(vals) {
    vals <- vals[, colSums(is.na(vals)) == 0, drop = FALSE]
    active <- by
    while (ncol(vals) > 1L && length(active)) {
      dropped <- FALSE
      for (v in active) {
        rest <- setdiff(active, v)
        cfi <- cf[match(colnames(vals), as.character(cf$.cell)), , drop = FALSE]
        key <- if (length(rest)) interaction(cfi[rest], drop = TRUE) else factor(rep(1L, ncol(vals)))
        agree <- all(vapply(split(seq_len(ncol(vals)), key), function(idx)
          max(abs(sweep(vals[, idx, drop = FALSE], 1, vals[, idx[1]]))) < 1e-10, logical(1)))
        if (agree) {
          vals <- vals[, !duplicated(key), drop = FALSE]
          active <- rest
          dropped <- TRUE
          break
        }
      }
      if (!dropped) break
    }
    if (ncol(vals) == 1L && !length(active)) colnames(vals) <- NA_character_
    list(vals = vals, active = active)
  }
  mats <- list(); infos <- list()
  push <- function(p, red, crit = NA_character_) {
    if (!ncol(red$vals)) return()
    mats[[length(mats) + 1L]] <<- red$vals
    infos[[length(infos) + 1L]] <<- data.frame(parameter = .param_label(model, p), name = ext(p),
                                               .cell = colnames(red$vals), criterion = crit,
                                               .active = paste(red$active, collapse = ","))
  }
  for (p in names(setup$tp)) {
    if (!is.null(internal) && !p %in% internal) next
    if (all(setup$tp[[p]] == 0)) next          # fixed at zero (a cumulative mu with no terms): not an estimate
    rows <- if (p %in% .lure_param_set && !is.null(roles)) roles$noise
            else if (p %in% .signal_param_set && !is.null(roles) && fam != "source_mixture") roles$signal
            else seq_len(setup$N)
    push(p, reduce(cell_mean(setup$tp[[p]], rows)))
  }
  crit_sets <- list(criterion = setup$th, criterion2 = if (length(dim(setup$th2)) == 3L) setup$th2)
  for (cn in names(crit_sets)) {
    th <- crit_sets[[cn]]
    if (is.null(th) || (!is.null(internal) && !cn %in% internal)) next
    J <- dim(th)[3]
    for (k in seq_len(J)) {
      push(cn, reduce(cell_mean(matrix(th[, , k], nrow = setup$S), seq_len(setup$N))), paste0("c", k))
    }
  }
  # source criteria that vary with the detection response: one set per set of the model (the
  # "new" responses share one under source_criteria = "varying_shared_new")
  th2 <- setup$th2
  if (length(dim(th2)) == 4L && (is.null(internal) || "criterion2" %in% internal)) {
    vbin <- .vbin_of(model$model_data)
    labs <- .vbin_labels(model$model_data)
    for (v in seq_along(labs)) {
      b <- match(v, vbin)
      if (is.na(b)) next
      lab <- if (labs[v] == "new") "new" else paste0("bin ", labs[v])
      for (k in seq_len(dim(th2)[4])) {
        push("criterion2", reduce(cell_mean(matrix(th2[, , b, k], nrow = setup$S), seq_len(setup$N))),
             paste0(lab, ", c", k))
      }
    }
  }
  if (!length(mats)) return(list(draws = matrix(numeric(0), nrow = setup$S, ncol = 0), info = data.frame()))
  info <- do.call(rbind, infos)
  info$.cell <- factor(info$.cell, levels = levels(cf$.cell))
  if (!is.null(by)) {
    bycols <- cf[match(as.character(info$.cell), as.character(cf$.cell)), by, drop = FALSE]
    rownames(bycols) <- NULL
    for (v in by) bycols[[v]][!vapply(strsplit(info$.active, ","), function(a) v %in% a, logical(1))] <- NA
    info <- cbind(info[c("parameter", "name")], bycols, info[c("criterion", ".cell")])
  } else {
    info <- info[c("parameter", "name", "criterion", ".cell")]
  }
  ord <- order(match(info$name, unique(info$name)), match(info$criterion, unique(info$criterion)),
               as.integer(info$.cell), na.last = FALSE)
  draws <- do.call(cbind, mats)[, ord, drop = FALSE]
  info <- info[ord, , drop = FALSE]
  rownames(info) <- NULL
  chain <- if (is.null(ndraws)) .draw_chains(setup$draws) else NULL
  out_info <- info
  out_info$criterion <- if (all(is.na(info$criterion))) NULL else info$criterion
  list(draws = draws, info = out_info[setdiff(names(out_info), ".cell")], chain = chain, cells = info$.cell)
}

.draw_chains <- function(draws) {
  ch <- tryCatch(posterior::as_draws_df(draws)$.chain, error = function(e) NULL)
  if (is.null(ch) || length(ch) != nrow(draws)) NULL else as.integer(ch)
}
