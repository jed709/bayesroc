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
#' Folds are fitted one after another. Each refit uses the fit's backend,
#' sampler, number of chains and draws per chain; arguments in `...` are passed
#' to [fit_broc()] and override them (e.g. `iter_warmup`, `control`, `seed`,
#' `refresh`). A `seed` is offset by the fold number for each refit.
#'
#' @param x A `broc_fit` object from [fit_broc()].
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
#' @param ... Arguments passed to [fit_broc()] for the refits.
#' @return An object of class `c("kfold", "loo")` with `estimates`
#'   (`elpd_kfold`, `p_kfold`, `kfoldic`) and `pointwise`, usable with
#'   [loo::loo_compare()]. `p_kfold` is the full-data log pointwise predictive
#'   density minus `elpd_kfold`, and is `NA` when `x` has no log-likelihood.
#' @examples
#' \dontrun{
#' kf <- kfold(fit, K = 10)
#' kf_sub <- kfold(fit, group = "subject")
#' loo::loo_compare(kfold(fit1, group = "subject"), kfold(fit2, group = "subject"))
#' }
#' @importFrom loo kfold
#' @method kfold broc_fit
#' @export
kfold.broc_fit <- function(x, K = 10, Ksub = NULL, folds = NULL, group = NULL,
                           joint = FALSE, type = c("trial", "cell"),
                           save_fits = FALSE, ...) {
  type <- match.arg(type)
  joint <- .kfold_joint(joint)
  model <- attr(x, "broc_model")
  if (is.null(model)) stop("Fit object must have 'broc_model' attribute. Use fit_broc() to fit the model.", call. = FALSE)
  N <- as.integer(model$stan_data$N)
  counts <- model$stan_data$counts
  aggregated <- !is.null(counts)
  counts <- if (aggregated) as.integer(counts) else rep(1L, N)
  if (!aggregated) type <- "trial"
  trial <- identical(type, "trial")
  unit_row <- if (trial) rep.int(seq_len(N), counts) else seq_len(N)
  U <- length(unit_row)

  gvar <- NULL
  if (!is.null(group)) {
    if (!is.character(group) || length(group) != 1L || !group %in% names(model$data)) {
      stop("`group` must name a column of the model data. Columns: ",
           paste(setdiff(names(model$data), ""), collapse = ", "), call. = FALSE)
    }
    if (nrow(model$data) != N) stop("internal: model data and likelihood rows differ", call. = FALSE)
    gvar <- factor(model$data[[group]])[unit_row]
  }
  if (joint == "group" && is.null(gvar)) stop("`joint = \"group\"` needs `group`.", call. = FALSE)

  if (is.null(folds)) {
    if (is.null(gvar)) {
      fold_type <- "random"
      folds <- loo::kfold_split_random(K, U)
    } else {
      fold_type <- "group"
      folds <- as.integer(gvar)
      K <- nlevels(gvar)
      message("Setting 'K' to the number of levels of '", group, "' (", K, ")")
    }
  } else if (is.character(folds) && length(folds) == 1L) {
    fold_type <- match.arg(folds, c("loo", "stratified", "grouped"))
    if (fold_type %in% c("stratified", "grouped") && is.null(gvar)) {
      stop("Argument 'group' is required for fold type '", fold_type, "'.", call. = FALSE)
    }
    if (fold_type == "loo") {
      folds <- seq_len(U)
      K <- U
      message("Setting 'K' to the number of ", if (trial) "trials" else "rows", " (", K, ")")
    } else if (fold_type == "stratified") {
      folds <- loo::kfold_split_stratified(K, gvar)
    } else {
      folds <- loo::kfold_split_grouped(K, gvar)
    }
  } else {
    fold_type <- "custom"
    if (trial && aggregated && length(folds) == N && N != U) folds <- folds[unit_row]
    if (length(folds) != U) {
      stop("If 'folds' is a vector, it must have one entry per ", if (trial) "trial" else "row",
           " (", U, ")", if (trial && aggregated) paste0(" or per row (", N, ")") else "", ".", call. = FALSE)
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

  sets <- lapply(Ksub, function(k) {
    omitted <- if (fold_type == "loo" && !is.null(gvar)) which(gvar == gvar[k]) else which(folds == k)
    predicted <- if (fold_type == "loo") k else omitted
    om <- tabulate(unit_row[omitted], N)
    list(k = k, omitted = om, predicted = predicted,
         key = paste(c(which(om > 0L), om[om > 0L]), collapse = ","))
  })
  keys <- unique(vapply(sets, `[[`, "", "key"))

  units <- .kfold_units(sets, unit_row, counts, trial, joint, gvar)

  dots <- list(...)
  fit_args <- list(backend = x$.backend, sampler = x$.sampler, chains = x$.num_chains,
                   iter_sampling = x$.iter_sampling)
  fit_args[names(dots)] <- dots
  fit_args$save_loglik <- TRUE
  fit_args$loo <- FALSE
  fit_args$file <- NULL
  seed <- fit_args$seed

  elpd <- rep(NA_real_, length(units$rows))
  fits <- if (save_fits) array(list(), dim = c(length(keys), 3L),
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
    if (save_fits) {
      fits[i, ] <- list(fit_k, list(which(counts - sk[[1]]$omitted < counts)),
                        list(sort(unique(unlist(lapply(sk, `[[`, "predicted"))))))
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
  atts <- list(K = K, Ksub = Ksub, group = group, folds = folds, fold_type = fold_type,
               joint = joint, broc_type = type)
  attributes(out)[names(atts)] <- atts
  if (save_fits) {
    out$fits <- fits
    out$data <- model$data
  }
  structure(out, class = c("kfold", "loo"), dims = c(NA_integer_, n))
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
    code <- .generate_model_code(md, model$family, .model_kind(model))
    model$stan_code <- code$stan_code
    model$stan_data <- code$stan_data
    model$batch_cpp <- code$batch_cpp
  } else {
    model$stan_data$counts <- as.integer(w)
  }
  model$model_data <- md
  model
}
