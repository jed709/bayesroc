# =============================================================================
# K-fold cross-validation
# =============================================================================

#' K-Fold Cross-Validation for Fitted SDT Models
#'
#' Exact K-fold cross-validation, following `brms::kfold()`: the model is refit
#' once per fold without the held-out data, and the held-out data are scored by
#' the expected log pointwise predictive density of the refit.
#'
#' @details The unit of cross-validation is a trial, matching
#' `loo(type = "trial")`: for aggregated fits, each row stands for `counts[n]`
#' trials, numbered in row order (`rep(seq_len(N), counts)`), and folds are
#' assigned to trials. With `type = "cell"`, the units are the rows and each
#' row's score is the joint density of its `counts[n]` trials, as in
#' `loo(type = "cell")`. The two are not comparable via [loo::loo_compare()].
#'
#' Each refit keeps every row and sets the held-out trials' counts to zero, so
#' the refit is the same model on the training data, every fold runs the same
#' compiled program, and the held-out trials are scored from the refit's own
#' log-likelihood. A random-effect level whose trials are all held out keeps its
#' effects in the refit; with no data, their posterior given the
#' hyperparameters is the estimated population distribution, so they are draws
#' of the new level as in brms' `sample_new_levels = "gaussian"`. Smooth and
#' per-bin random effects are handled the same way.
#'
#' Folds are fitted one after another. Each refit uses the fit's
#' sampler, number of chains and draws per chain; arguments in `...` are passed
#' to [fit_broc()] and override them (e.g. `iter_warmup`, `control`, `seed`).
#' A `seed` is offset by the fold number for each refit. Refits
#' that would leave out the same trials (as with `folds = "loo"` on aggregated
#' data) are run once.
#'
#' Several fits of the same data are cross-validated on the same folds, so
#' their pointwise differences are paired, as [loo::loo_compare()] assumes.
#'
#' @param x A `broc_fit` object from [fit_broc()].
#' @param ... Further `broc_fit` objects of the same data, which are
#'   cross-validated on the same folds and compared, and arguments passed to
#'   [fit_broc()] for the refits.
#' @param K Number of folds (default 10). Ignored when `folds` or `group`
#'   determines it.
#' @param Ksub Optional subset of folds to run: a single number draws that many
#'   folds at random; a vector of fold numbers (or a one-dimensional array, for
#'   a single fold) names them.
#' @param folds How the folds are formed. `NULL` (default) assigns units to `K`
#'   folds at random ([loo::kfold_split_random()]), or, when `group` is given,
#'   leaves out one level of `group` at a time with `K` set to its number of
#'   levels. `"stratified"` and `"grouped"` use [loo::kfold_split_stratified()]
#'   and [loo::kfold_split_grouped()] on `group`. `"loo"` gives exact
#'   leave-one-out (`K` = number of units); with `group`, each unit is predicted
#'   from a refit without its whole level. Otherwise a vector of fold ids, one
#'   per unit; for `type = "trial"` on an aggregated fit, one per row is also
#'   accepted and applies to all trials of the row.
#' @param group Name of a column of the model data (usually a random-effect
#'   grouping factor) used by `folds`.
#' @param joint Scoring unit. `FALSE` or `"obs"` (default) scores each unit;
#'   `TRUE` or `"fold"` scores the held-out data of each fold jointly; `"group"`
#'   scores each level of `group` within a fold jointly.
#' @param type `"trial"` (default) or `"cell"`; see Details. Ignored for
#'   trial-level fits.
#' @param save_fits If `TRUE`, keep the refits (with their log-likelihood files)
#'   in the result.
#' @param model_names Names of the models when several fits are given;
#'   defaults to the deparsed arguments.
#' @return An object of class `c("kfold", "loo")` with `estimates`
#'   (`elpd_kfold`, `p_kfold`, `kfoldic`) and `pointwise`, usable with
#'   [loo::loo_compare()]. `p_kfold` is the full-data log pointwise predictive
#'   density minus `elpd_kfold`, and is `NA` when `x` has no log-likelihood.
#'   With several fits, a list of class `broc_kfold_list` with the `kfolds` and
#'   their [loo::loo_compare()] table `diffs`.
#' @examples
#' \dontrun{
#' kf <- kfold(fit, K = 10)
#' kf_sub <- kfold(fit, group = "subject")
#' kfold(fit1, fit2, group = "subject")
#' }
#' @importFrom loo kfold
#' @method kfold broc_fit
#' @export
kfold.broc_fit <- function(x, ..., K = 10, Ksub = NULL, folds = NULL, group = NULL,
                           joint = FALSE, type = c("trial", "cell"),
                           save_fits = FALSE, model_names = NULL) {
  type <- match.arg(type)
  joint <- .kfold_joint(joint)
  dots <- list(...)
  is_fit <- vapply(dots, inherits, logical(1), "broc_fit")
  fits <- c(list(x), dots[is_fit])
  dots <- dots[!is_fit]
  if (is.null(model_names)) {
    cl <- as.list(match.call(expand.dots = FALSE)$...)
    model_names <- c(deparse1(substitute(x)), vapply(cl[is_fit], deparse1, ""))
  }
  if (length(model_names) != length(fits)) stop("`model_names` must have one name per fit.", call. = FALSE)

  lay <- .kfold_layout(x, type, group)
  if (joint == "group" && is.null(lay$gvar)) stop("`joint = \"group\"` needs `group`.", call. = FALSE)
  U <- length(lay$unit_row)

  if (is.null(folds)) {
    if (is.null(lay$gvar)) {
      fold_type <- "random"
      folds <- loo::kfold_split_random(K, U)
    } else {
      fold_type <- "group"
      folds <- as.integer(lay$gvar)
      K <- nlevels(lay$gvar)
      message("Setting 'K' to the number of levels of '", group, "' (", K, ")")
    }
  } else if (is.character(folds) && length(folds) == 1L) {
    fold_type <- match.arg(folds, c("loo", "stratified", "grouped"))
    if (fold_type %in% c("stratified", "grouped") && is.null(lay$gvar)) {
      stop("Argument 'group' is required for fold type '", fold_type, "'.", call. = FALSE)
    }
    if (fold_type == "loo") {
      folds <- seq_len(U)
      K <- U
      message("Setting 'K' to the number of ", if (lay$trial) "trials" else "rows", " (", K, ")")
    } else if (fold_type == "stratified") {
      folds <- loo::kfold_split_stratified(K, lay$gvar)
    } else {
      folds <- loo::kfold_split_grouped(K, lay$gvar)
    }
  } else {
    fold_type <- "custom"
    N <- length(lay$counts)
    if (lay$trial && lay$aggregated && length(folds) == N && N != U) folds <- folds[lay$unit_row]
    if (length(folds) != U) {
      stop("If 'folds' is a vector, it must have one entry per ", if (lay$trial) "trial" else "row",
           " (", U, ")", if (lay$trial && lay$aggregated) paste0(" or per row (", N, ")") else "", ".",
           call. = FALSE)
    }
    folds <- as.integer(factor(folds))
    K <- max(folds)
    message("Setting 'K' to the number of folds (", K, ")")
  }

  if (is.null(Ksub)) {
    Ksub <- seq_len(K)
  } else {
    one <- length(Ksub) == 1L && !is.array(Ksub)
    Ksub <- as.integer(Ksub)
    if (any(is.na(Ksub) | Ksub <= 0L | Ksub > K)) {
      stop("'Ksub' must contain positive integers not larger than 'K'.", call. = FALSE)
    }
    Ksub <- sort(if (one) sample(seq_len(K), Ksub) else unique(Ksub))
  }

  spec <- list(K = K, Ksub = Ksub, folds = folds, fold_type = fold_type, group = group,
               joint = joint, save_fits = save_fits)
  res <- vector("list", length(fits))
  for (m in seq_along(fits)) {
    lay_m <- if (m == 1L) lay else .kfold_layout(fits[[m]], type, group)
    if (!identical(lay_m$unit_row, lay$unit_row) || !identical(lay_m$gvar, lay$gvar)) {
      stop("All fits must be of the same data: '", model_names[m], "' has different rows, counts or ",
           "groups from '", model_names[1], "'.", call. = FALSE)
    }
    if (length(fits) > 1L) message("Model '", model_names[m], "'")
    res[[m]] <- .kfold_run(fits[[m]], lay_m, spec, dots)
    attr(res[[m]], "model_name") <- model_names[m]
  }
  if (length(res) == 1L) return(res[[1]])
  names(res) <- model_names
  structure(list(kfolds = res, diffs = loo::loo_compare(res)), class = "broc_kfold_list")
}

#' @export
print.broc_kfold_list <- function(x, ...) {
  for (nm in names(x$kfolds)) {
    cat("Output of model '", nm, "':\n", sep = "")
    print(x$kfolds[[nm]], ...)
    cat("\n")
  }
  cat("Model comparisons:\n")
  print(x$diffs, ...)
  invisible(x)
}

.kfold_layout <- function(fit, type, group) {
  model <- attr(fit, "broc_model")
  if (is.null(model)) stop("Fit object must have 'broc_model' attribute. Use fit_broc() to fit the model.", call. = FALSE)
  N <- as.integer(model$stan_data$N)
  counts <- model$stan_data$counts
  aggregated <- !is.null(counts)
  counts <- if (aggregated) as.integer(counts) else rep(1L, N)
  trial <- !aggregated || identical(type, "trial")
  unit_row <- if (trial) rep.int(seq_len(N), counts) else seq_len(N)
  gvar <- NULL
  if (!is.null(group)) {
    if (!is.character(group) || length(group) != 1L || !group %in% names(model$data)) {
      stop("`group` must name a column of the model data. Columns: ",
           paste(setdiff(names(model$data), ""), collapse = ", "), call. = FALSE)
    }
    if (nrow(model$data) != N) stop("internal: model data and likelihood rows differ", call. = FALSE)
    gvar <- factor(model$data[[group]])[unit_row]
  }
  list(model = model, counts = counts, aggregated = aggregated, trial = trial, unit_row = unit_row, gvar = gvar)
}

.kfold_run <- function(x, lay, spec, dots) {
  model <- lay$model
  counts <- lay$counts
  unit_row <- lay$unit_row
  gvar <- lay$gvar
  N <- length(counts)
  K <- spec$K
  Ksub <- spec$Ksub
  folds <- spec$folds
  joint <- spec$joint
  omit_of <- function(k) if (spec$fold_type == "loo" && !is.null(gvar)) which(gvar == gvar[k]) else which(folds == k)
  sets <- lapply(Ksub, function(k) {
    omitted <- omit_of(k)
    predicted <- if (spec$fold_type == "loo") k else omitted
    om <- tabulate(unit_row[omitted], N)
    list(k = k, omitted = om, predicted = predicted,
         key = paste(c(which(om > 0L), om[om > 0L]), collapse = ","))
  })
  keys <- unique(vapply(sets, `[[`, "", "key"))
  units <- .kfold_units(sets, unit_row, counts, lay$trial, joint, gvar)

  fit_args <- list(sampler = x$.sampler, chains = x$.num_chains,
                   iter_sampling = x$.iter_sampling)
  fit_args[names(dots)] <- dots
  fit_args$save_loglik <- TRUE
  fit_args$loo <- FALSE
  fit_args$file <- NULL
  seed <- fit_args$seed

  elpd <- rep(NA_real_, length(units$rows))
  fits <- if (spec$save_fits) array(list(), dim = c(length(keys), 3L),
                                    dimnames = list(NULL, c("fit", "omitted", "predicted"))) else NULL
  for (i in seq_along(keys)) {
    sk <- Filter(function(s) identical(s$key, keys[i]), sets)
    ks <- vapply(sk, `[[`, 0L, "k")
    message("Fitting model ", i, " out of ", length(keys),
            if (length(ks) == 1L) paste0(" (fold ", ks, " of ", K, ")") else paste0(" (", length(ks), " folds)"))
    if (!is.null(seed)) fit_args$seed <- seed + i
    fit_k <- do.call(fit_broc, c(list(.with_counts(model, counts - sk[[1]]$omitted)), fit_args))
    j <- which(units$set %in% match(ks, Ksub))
    M <- .kfold_unit_ll(fit_k, units, j)
    elpd[j] <- apply(M, 2L, .log_mean_exp)
    if (spec$save_fits) {
      fits[i, ] <- list(fit_k, omit_of(ks[1]), sort(unique(unlist(lapply(sk, `[[`, "predicted")))))
    } else {
      suppressMessages(drop_loglik(fit_k))
    }
    rm(fit_k)
  }

  Mf <- tryCatch(.kfold_unit_ll(x, units, seq_along(units$rows)), error = function(e) NULL)
  if (is.null(Mf)) message("The fit has no log-likelihood, so p_kfold is NA.")
  lpd <- if (is.null(Mf)) rep(NA_real_, length(elpd)) else apply(Mf, 2L, .log_mean_exp)

  idx <- rep.int(seq_along(units$rows), units$mult)
  if (joint == "obs") idx <- idx[order(unlist(units$trials))]
  pointwise <- cbind(elpd_kfold = elpd[idx], p_kfold = lpd[idx] - elpd[idx], kfoldic = -2 * elpd[idx])
  n <- nrow(pointwise)
  estimates <- cbind(Estimate = colSums(pointwise),
                     SE = sqrt(n * apply(pointwise, 2L, stats::var)))
  rownames(estimates) <- colnames(pointwise)
  out <- list(estimates = estimates, pointwise = pointwise)
  atts <- list(K = K, Ksub = Ksub, group = spec$group, folds = folds, fold_type = spec$fold_type,
               joint = joint, broc_type = if (lay$trial) "trial" else "cell")
  attributes(out)[names(atts)] <- atts
  if (spec$save_fits) {
    out$fits <- fits
    out$data <- model$data
    out$counts <- counts
  }
  structure(out, class = c("kfold", "loo"), dims = c(NA_integer_, n))
}

#' Predictions of the Held-Out Data from K-Fold Refits
#'
#' Simulates each held-out trial from the refit that left it out, following
#' `brms::kfold_predict()`. Every trial of an aggregated row is simulated
#' separately.
#'
#' @param x A `kfold` object from [kfold()][kfold.broc_fit] run with
#'   `save_fits = TRUE`.
#' @param response Passed to [predict()][predict.broc_fit]: `NULL` (default)
#'   simulates every response of the family; `1` or `2` one response of a
#'   family with two responses.
#' @param ... Further arguments passed to [predict()][predict.broc_fit], such as
#'   `ndraws` or `seed`.
#' @return A list with `y`, the observed responses of the held-out trials in
#'   trial order, and `yrep`, an S x n matrix of simulated responses. When the
#'   family has two responses and `response = NULL`, `y` and `yrep` are lists
#'   with one element per response, named as in
#'   `predict(type = "prediction", summary = FALSE)`.
#' @examples
#' \dontrun{
#' kf <- kfold(fit, K = 5, save_fits = TRUE)
#' pp <- kfold_predict(kf)
#' }
#' @export
kfold_predict <- function(x, response = NULL, ...) {
  if (!inherits(x, "kfold")) stop("'x' must be a 'kfold' object.", call. = FALSE)
  if (is.null(x$fits) || is.null(x$counts)) {
    stop("Slots 'fits' and 'counts' are required. Please run kfold with save_fits = TRUE.", call. = FALSE)
  }
  trial_row <- rep.int(seq_along(x$counts), x$counts)
  cell <- identical(attr(x, "broc_type"), "cell")
  parts <- lapply(seq_len(nrow(x$fits)), function(i) {
    fit_k <- x$fits[[i, "fit"]]
    pred <- x$fits[[i, "predicted"]]
    tr <- if (cell) which(trial_row %in% pred) else pred
    rows <- trial_row[tr]
    yrep <- predict(fit_k, newdata = x$data[rows, , drop = FALSE], type = "prediction", summary = FALSE,
                    response = response, ...)
    sdz <- .source_design(attr(fit_k, "broc_model"))
    if (!is.null(sdz) && is.list(yrep) && !is.null(yrep$y2)) {
      yrep$y2[if (sdz$rule) !(yrep$y1 %in% sdz$levels) else matrix(sdz$obs[rows][col(yrep$y2)] == 0L, nrow(yrep$y2))] <-
        NA_integer_
    }
    list(trials = tr, y = .kfold_observed(attr(fit_k, "broc_model"), rows, yrep, response), yrep = yrep)
  })
  o <- order(unlist(lapply(parts, `[[`, "trials")))
  if (is.list(parts[[1]]$yrep)) {
    nm <- names(parts[[1]]$yrep)
    y <- stats::setNames(lapply(nm, function(k) unlist(lapply(parts, function(p) p$y[[k]]))[o]), nm)
    yrep <- stats::setNames(lapply(nm, function(k) {
      do.call(cbind, lapply(parts, function(p) p$yrep[[k]]))[, o, drop = FALSE]
    }), nm)
  } else {
    y <- unlist(lapply(parts, `[[`, "y"))[o]
    yrep <- do.call(cbind, lapply(parts, `[[`, "yrep"))[, o, drop = FALSE]
  }
  list(y = y, yrep = yrep)
}

.kfold_observed <- function(model, rows, yrep, response) {
  sd <- model$stan_data
  rk <- if (!is.null(sd$rk)) {
    v <- as.integer(sd$rk)
    v[v <= 0L] <- NA_integer_
    v
  }
  y2 <- sd$y2
  if (!is.null(sd$src_obs)) y2[sd$src_obs == 0L] <- NA_integer_
  src <- list(y = sd$y, y1 = sd$y, y2 = y2, rk = rk)
  if (is.list(yrep)) {
    return(stats::setNames(lapply(names(yrep), function(k) as.integer(src[[k]][rows])), names(yrep)))
  }
  second <- identical(as.integer(response), 2L)
  v <- if (!second) sd$y else if (identical(model$family$family, "cdp")) rk else y2
  as.integer(v[rows])
}

.kfold_joint <- function(joint) {
  if (isFALSE(joint)) return("obs")
  if (isTRUE(joint)) return("fold")
  if (!is.character(joint) || length(joint) != 1L || !joint %in% c("obs", "fold", "group")) {
    stop("`joint` must be FALSE, TRUE, \"obs\", \"fold\" or \"group\".", call. = FALSE)
  }
  joint
}

.log_mean_exp <- function(v) {
  m <- max(v)
  if (!is.finite(m)) return(m)
  m + log(mean(exp(v - m)))
}

.kfold_units <- function(sets, unit_row, counts, trial, joint, gvar) {
  rows <- list(); w <- list(); mult <- integer(0); set <- integer(0); trials <- list()
  add <- function(r, wt, m, s, tr) {
    rows[[length(rows) + 1L]] <<- r
    w[[length(w) + 1L]] <<- wt
    mult <<- c(mult, m)
    set <<- c(set, s)
    trials[[length(trials) + 1L]] <<- tr
  }
  for (s in seq_along(sets)) {
    pred <- sets[[s]]$predicted
    pr <- unit_row[pred]
    if (joint == "obs") {
      by_row <- split(pred, pr)
      for (r in names(by_row)) {
        rr <- as.integer(r)
        add(rr, if (trial) 1 else counts[rr], length(by_row[[r]]), s, by_row[[r]])
      }
    } else {
      grp <- if (joint == "group") split(pred, droplevels(gvar[pred])) else list(pred)
      for (g in grp) {
        tab <- tabulate(unit_row[g], length(counts))
        r <- which(tab > 0L)
        add(r, if (trial) tab[r] else counts[r], 1L, s, g)
      }
    }
  }
  list(rows = rows, w = w, mult = mult, set = set, trials = trials)
}

.kfold_unit_ll <- function(fit, units, j, chunk = 2048L) {
  r_all <- unlist(units$rows[j])
  w_all <- unlist(units$w[j])
  u_all <- rep.int(seq_along(j), lengths(units$rows[j]))
  path <- .loglik_path(fit)
  if (is.null(path) && !is.null(fit$.loglik_path)) stop(.missing_sidecar_msg(fit$.loglik_path), call. = FALSE)
  read <- if (!is.null(path)) {
    function(a, b) .read_loglik_SN_cols(path, a, b)
  } else {
    ll <- as.matrix(fit$draws(variables = "log_lik", format = "matrix"))
    function(a, b) ll[, a:b, drop = FALSE]
  }
  M <- NULL
  for (a in seq(min(r_all), max(r_all), by = chunk)) {
    b <- min(a + chunk - 1L, max(r_all))
    sel <- which(r_all >= a & r_all <= b)
    if (!length(sel)) next
    llc <- read(a, b)
    if (is.null(M)) M <- matrix(0, nrow(llc), length(j))
    part <- rowsum(t(llc[, r_all[sel] - a + 1L, drop = FALSE]) * w_all[sel], u_all[sel])
    cols <- as.integer(rownames(part))
    M[, cols] <- M[, cols] + t(part)
  }
  M
}

.with_counts <- function(model, w) {
  md <- model$model_data
  regen <- !isTRUE(md$has_counts)
  md$has_counts <- TRUE
  md$rows_live <- TRUE
  md$stan_data$counts <- as.integer(w)
  if (regen) {
    model$stan_data <- .model_stan_data(md, .model_kind(model))
  } else {
    model$stan_data$counts <- as.integer(w)
  }
  model$model_data <- md
  model
}
