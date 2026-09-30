# =============================================================================
# Sidecar Resolution
# =============================================================================

#' Per-row trial counts, or NULL if the fit is trial-level
#' @noRd
.broc_counts <- function(fit) {
  model <- attr(fit, "broc_model")
  cnt <- model$stan_data$counts
  if (is.null(cnt)) NULL else as.integer(cnt)
}

#' Resolve a usable log_lik sidecar path, or NULL
#' @noRd
.loglik_path <- function(fit) {
  p <- fit$.loglik_path
  if (!is.null(p) && file.exists(p)) p else NULL
}

.missing_sidecar_msg <- function(p) {
  paste0("The log-likelihood file '", basename(p), "' no longer exists. Files written ",
         "during a session are temporary; keep log_lik across sessions with ",
         "save_broc_fit() or fit_broc(file = ), not saveRDS(). Otherwise refit.")
}

.owned_sidecar <- function(fit) {
  p <- fit$.loglik_path
  if (!isTRUE(fit$.loglik_owned) || is.null(p) || !nzchar(p) || !file.exists(p)) return(FALSE)
  root <- paste0(normalizePath(tempdir(), winslash = "/", mustWork = FALSE), "/")
  startsWith(normalizePath(p, winslash = "/", mustWork = FALSE), root)
}

#' Delete a fit's temporary log_lik sidecar to reclaim disk
#'
#' Only a log-likelihood file that bayesroc itself wrote to this session's
#' temporary directory is deleted; a file saved with [save_broc_fit()] is left
#' in place. After dropping, [loo()] is unavailable for the fit on either
#' backend (a stored `fit$.loo` is kept).
#'
#' @param fit A `broc_fit`.
#' @return The slimmed fit, invisibly.
#' @export
drop_loglik <- function(fit) {
  stopifnot(inherits(fit, "broc_fit"))
  p <- fit$.loglik_path
  freed <- 0
  if (.owned_sidecar(fit)) {
    freed <- as.double(file.size(p))
    unlink(p)
    fit$.loglik_path <- NULL
    fit$.loglik_owned <- FALSE
    fit$.loglik_dropped <- TRUE
  } else if (!is.null(p) && !file.exists(p)) {
    fit$.loglik_path <- NULL
    fit$.loglik_dropped <- TRUE
  }
  kept <- !is.null(fit$.loglik_path)
  message(sprintf("drop_loglik: freed %.2f GB%s. ", freed / 1e9,
                  if (freed == 0) " (nothing to remove)" else ""),
          if (kept) "A saved log-likelihood file remains; loo() still works."
          else "loo() is unavailable for this fit until it is refit.")
  invisible(fit)
}

# =============================================================================
# LOO-CV
# =============================================================================

#' LOO-CV for Fitted SDT Models
#'
#' Approximate leave-one-out cross-validation (PSIS-LOO) for model comparison via
#' [loo::loo_compare()].
#'
#' @details `log_lik` is streamed from an on-disk sidecar in column chunks sized
#' to `mem_budget_gb`. For aggregated fits, `type = "trial"` replicates each
#' row's pointwise result `counts[n]` times, giving `sum(counts)` observations;
#' `type = "cell"` folds the count into `log_lik`, giving `N` observations. The
#' two are not comparable via [loo::loo_compare()].
#'
#' @param x A `broc_fit` object from [fit_broc()].
#' @param cores PSOCK workers for PSIS (default 4).
#' @param type CV partition for aggregated fits: `"trial"` (default) or
#'   `"cell"`. Ignored for trial-level fits.
#' @param chunk_mb Per-worker column-chunk size in MiB. `NULL` auto-sizes.
#' @param mem_budget_gb Total RAM budget for the pool; `NULL` uses ~60% of free.
#' @param ... Additional arguments passed to [loo::loo()].
#' @return A `loo` object.
#' @importFrom loo loo
#' @method loo broc_fit
#' @export
loo.broc_fit <- function(x, cores = 4, type = c("trial", "cell"),
                         chunk_mb = NULL, mem_budget_gb = NULL, ...) {
  if (!requireNamespace("loo", quietly = TRUE)) {
    stop("Package 'loo' is required for LOO-CV.")
  }
  fit <- x
  type <- match.arg(type)

  cached <- fit$.loo
  if (!is.null(cached) && identical(attr(cached, "broc_loo_type"), type)) {
    return(cached)
  }

  counts <- .broc_counts(fit)
  path <- .loglik_path(fit)

  if (is.null(path) && !is.null(fit$.loglik_path)) {
    stop(.missing_sidecar_msg(fit$.loglik_path), call. = FALSE)
  }
  if (is.null(path) && (isTRUE(fit$.loglik_dropped) || !is.null(cached))) {
    stop("No log_lik is available for this fit",
         if (!is.null(cached)) paste0(" (only a stored type = '",
                                      attr(cached, "broc_loo_type") %||% "trial",
                                      "' loo result is kept)"),
         ". Refit with save_loglik = TRUE to run loo(type = '", type, "').",
         call. = FALSE)
  }

  if (is.null(path)) {
    ll <- tryCatch(
      as.matrix(fit$draws(variables = "log_lik", format = "matrix")),
      error = function(e) NULL)
    if (is.null(ll)) {
      stop("No log_lik is available for this fit. It was either dropped ",
           "(fit_broc(loo = TRUE) or drop_loglik()) or never computed ",
           "(save_loglik = FALSE). Refit with save_loglik = TRUE to run ",
           "loo() interactively.", call. = FALSE)
    }
    if (is.null(counts) || length(counts) != ncol(ll)) counts <- NULL
    return(.loo_in_memory(ll, fit, counts, type, cores, ...))
  }

  meta <- .loglik_meta(path)
  if (!is.null(counts) && length(counts) != meta$N) {
    warning("counts has length ", length(counts), " but log_lik has ", meta$N,
            " columns; ignoring counts.", call. = FALSE)
    counts <- NULL
  } else if (!is.null(counts) && all(counts == 1L)) {
    counts <- NULL
  }
  .loo_stream(path, counts = counts, type = type, cores = cores,
              chunk_mb = chunk_mb, mem_budget_gb = mem_budget_gb, ...)
}

.loo_in_memory <- function(ll, fit, counts, type, cores, ...) {
  S <- nrow(ll)
  chain_id <- rep(seq_len(fit$num_chains()), each = S %/% fit$num_chains())
  if (!is.null(counts) && identical(type, "cell")) {
    ll <- ll * rep(counts, each = S)
    counts <- NULL
  }
  if (!all(is.finite(ll))) {
    warning(sum(!is.finite(ll)), " non-finite log_lik replaced with -1e6", call. = FALSE)
    ll[!is.finite(ll)] <- -1e6
  }
  r_eff <- .stable_r_eff(ll, chain_id, cores)
  res <- loo::loo(ll, r_eff = r_eff, cores = cores, ...)
  if (is.null(counts)) return(res)
  .expand_loo_by_counts(res$pointwise, res$diagnostics$pareto_k,
                        res$diagnostics$n_eff, counts, S)
}

.stable_r_eff <- function(ll, chain_id, cores) {
  lik <- exp(sweep(ll, 2L, apply(ll, 2L, max)))
  r_eff <- loo::relative_eff(lik, chain_id = chain_id, cores = cores)
  r_eff[!is.finite(r_eff)] <- 1
  r_eff
}

.LOO_CHUNK_COPIES <- 9

.loo_stream <- function(path, counts, type, cores = 4, chunk_mb = NULL,
                        mem_budget_gb = NULL, ...) {
  loo_args <- list(...)
  meta <- .loglik_meta(path)
  C <- meta$C; D <- meta$D; N <- meta$N; S <- as.integer(meta$S)
  chain_id <- rep(seq_len(C), each = D)
  n_workers <- max(1L, min(as.integer(cores), N))

  budget_gb <- mem_budget_gb
  if (is.null(budget_gb)) {
    fr <- .free_ram_gb()
    budget_gb <- if (is.na(fr)) 8 else max(2, 0.6 * fr)
  }
  budget_cols <- max(1L, floor(budget_gb * 2^30 /
                                 (as.double(n_workers) * .LOO_CHUNK_COPIES * S * 8)))
  user_cols <- if (is.null(chunk_mb)) budget_cols else
    max(1L, floor(chunk_mb * 2^20 / (as.double(S) * 8)))
  cols_per_chunk <- max(1L, min(budget_cols, user_cols))
  n_chunks <- min(N, max(n_workers, ceiling(N / cols_per_chunk)))

  bnd <- floor(seq(0, N, length.out = n_chunks + 1L))
  ranges <- Map(function(a, b) c(a + 1L, b), bnd[-length(bnd)], bnd[-1L])
  ranges <- Filter(function(r) r[2] >= r[1], ranges)

  cell <- identical(type, "cell") && !is.null(counts)
  wcounts <- if (cell) counts else NULL

  worker <- function(rg) {
    j0 <- rg[1]; j1 <- rg[2]
    captured <- character(0)
    out <- tryCatch(withCallingHandlers({
      ll <- .read_loglik_SN_cols(path, j0, j1)
      if (!is.null(wcounts)) {
        ll <- ll * rep(wcounts[j0:j1], each = nrow(ll))
      }
      if (!all(is.finite(ll))) {
        captured <- c(captured, sprintf("%d non-finite log_lik replaced with -1e6",
                                        sum(!is.finite(ll))))
        ll[!is.finite(ll)] <- -1e6
      }
      r_eff <- .stable_r_eff(ll, chain_id, 1)
      do.call(loo::loo, c(list(ll, r_eff = r_eff, cores = 1), loo_args))
    }, warning = function(w) {
      captured[[length(captured) + 1L]] <<- conditionMessage(w)
      invokeRestart("muffleWarning")
    }), error = function(e) list(.err = conditionMessage(e), cols = c(j0, j1)))
    list(loo = out, warnings = captured, cols = c(j0, j1))
  }

  if (n_workers == 1L) {
    res <- lapply(ranges, worker)
  } else {
    cl <- parallel::makeCluster(n_workers)
    on.exit(parallel::stopCluster(cl), add = TRUE)
    parallel::clusterEvalQ(cl, library(loo))
    fn_env <- environment(.read_loglik_SN_cols)
    parallel::clusterExport(cl, c(".read_loglik_SN_cols", ".loglik_meta",
                                  ".npy_header", ".stable_r_eff"), envir = fn_env)
    parallel::clusterExport(cl, c("path", "chain_id", "wcounts", "loo_args"),
                            envir = environment())
    environment(worker) <- globalenv()
    res <- parallel::clusterApplyLB(cl, ranges, worker)
  }

  .combine_loo(res, counts = if (cell) NULL else counts, S = S)
}

.combine_loo <- function(res, counts, S) {
  errs <- Filter(function(r) is.list(r$loo) && !is.null(r$loo$.err), res)
  if (length(errs)) {
    msgs <- vapply(errs, function(r) sprintf("  cols %d-%d: %s",
                                             r$loo$cols[1], r$loo$cols[2],
                                             r$loo$.err), "")
    stop("loo() failed in ", length(errs), " of ", length(res),
         " worker chunk(s):\n", paste(msgs, collapse = "\n"),
         "\n\nRerun with cores = 1 for a clearer trace.", call. = FALSE)
  }
  for (r in res) {
    for (w in r$warnings) {
      if (!grepl("Pareto", w)) message("[loo chunk ", r$cols[1], "-", r$cols[2],
                                       "] ", w)
    }
  }
  sub <- lapply(res, `[[`, "loo")
  pointwise <- do.call(rbind, lapply(sub, function(z) z$pointwise))
  pareto_k <- unlist(lapply(sub, function(z) z$diagnostics$pareto_k))
  n_eff <- unlist(lapply(sub, function(z) z$diagnostics$n_eff))

  out <- .expand_loo_by_counts(pointwise, pareto_k, n_eff, counts, S)

  unit <- if (is.null(counts)) "observation" else "row"
  bad <- sum(pareto_k > 0.7, na.rm = TRUE)
  vbad <- sum(pareto_k > 1.0, na.rm = TRUE)
  if (vbad > 0) {
    warning("Found ", vbad, " ", unit, "(s) with Pareto k > 1.0.", call. = FALSE)
  } else if (bad > 0) {
    warning("Found ", bad, " ", unit, "(s) with Pareto k > 0.7.", call. = FALSE)
  }
  out
}

#' Assemble a loo object, replicating pointwise rows by trial counts
#' @noRd
.expand_loo_by_counts <- function(pointwise, pareto_k, n_eff, counts, S) {
  if (is.null(rownames(pointwise))) {
    nm <- paste0("log_lik[", seq_len(nrow(pointwise)), "]")
    rownames(pointwise) <- nm
    names(pareto_k) <- nm
    names(n_eff) <- nm
  }
  if (!is.null(counts)) {
    idx <- rep.int(seq_len(nrow(pointwise)), counts)
    pointwise <- pointwise[idx, , drop = FALSE]
    pareto_k  <- pareto_k[idx]
    n_eff     <- n_eff[idx]
  }
  N <- nrow(pointwise)

  cols <- c("elpd_loo", "p_loo", "looic")
  estimates <- cbind(
    Estimate = vapply(cols, function(cn) sum(pointwise[, cn]), numeric(1)),
    SE       = vapply(cols, function(cn) sqrt(N * stats::var(pointwise[, cn])),
                      numeric(1))
  )
  rownames(estimates) <- cols

  structure(
    list(
      pointwise   = pointwise,
      estimates   = estimates,
      diagnostics = list(pareto_k = pareto_k, n_eff = n_eff)
    ),
    class = c("psis_loo", "loo"),
    dims  = c(S, N)
  )
}
