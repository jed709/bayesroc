# =============================================================================
# Chunk Pooling
# =============================================================================

.pool_chunks <- function(fits) {
  if (length(fits) == 1L) return(fits[[1]])
  pooled <- do.call(posterior::bind_draws,
                    c(lapply(fits, function(z) z$.draws_array), along = "iteration"))
  first <- fits[[1]]
  out <- new_broc_fit(
    backend_fit = list(extra_fields = .pool_extra_fields(fits)),
    model = attr(first, "broc_model"),
    backend = first$.backend,
    num_chains = first$.num_chains,
    iter_sampling = sum(vapply(fits, function(z) z$.iter_sampling, integer(1))),
    elapsed = sum(vapply(fits, function(z) as.numeric(z$.elapsed %||% 0), numeric(1))),
    draws_array = pooled,
    max_treedepth = first$.max_treedepth,
    sampler = first$.sampler,
    save_loglik = FALSE
  )
  out$.state_path <- fits[[length(fits)]]$.state_path
  out
}

.pool_extra_fields <- function(fits) {
  efs <- lapply(fits, function(z) z$.backend_fit$extra_fields)
  keys <- Reduce(intersect, lapply(efs, names))
  out <- lapply(keys, function(k) do.call(cbind, lapply(efs, function(e) as.matrix(e[[k]]))))
  names(out) <- keys
  out
}

.chunk_divergences <- function(fit) {
  d <- tryCatch(fit$diagnostic_summary(), error = function(e) NULL)
  if (is.null(d) || is.null(d$num_divergent)) return(0L)
  as.integer(sum(d$num_divergent))
}

# =============================================================================
# Stopping Diagnostics
# =============================================================================

.ess_class <- function(v) {
  ifelse(grepl("^sd_|^sds_|^sigma_", v), "sd",
         ifelse(grepl("^cor_|^corr_", v), "cor", "pop"))
}

.monitored <- function(vars) {
  vars[!grepl("^log_lik|^lp__|^z_|^u_|^zs_|^s_|^L_corr_", vars)]
}

.ess_targets <- function(ess_target) {
  if (!is.numeric(ess_target) || !length(ess_target) || any(!is.finite(ess_target)) ||
      any(ess_target <= 0)) {
    stop("`ess_target` must be positive: one number, or named values for pop, sd and cor.",
         call. = FALSE)
  }
  if (is.null(names(ess_target))) {
    if (length(ess_target) != 1L) {
      stop("An unnamed `ess_target` must be a single number.", call. = FALSE)
    }
    return(c(pop = ess_target, sd = ess_target, cor = ess_target))
  }
  bad <- setdiff(names(ess_target), c("pop", "sd", "cor"))
  if (length(bad)) {
    stop("`ess_target` names must be pop, sd or cor; got ", paste(bad, collapse = ", "), ".",
         call. = FALSE)
  }
  out <- c(pop = 400, sd = 400, cor = 400)
  out[names(ess_target)] <- ess_target
  out
}

.stop_diagnostics <- function(draws) {
  sm <- posterior::summarise_draws(draws, ess_bulk = posterior::ess_bulk,
                                   ess_tail = posterior::ess_tail, rhat = posterior::rhat)
  cls <- .ess_class(sm$variable)
  ess <- pmin(sm$ess_bulk, sm$ess_tail)
  floor_of <- function(k) {
    x <- ess[cls == k]
    x <- x[is.finite(x)]
    if (length(x)) min(x) else Inf
  }
  rh <- sm$rhat[is.finite(sm$rhat)]
  list(ess = c(pop = floor_of("pop"), sd = floor_of("sd"), cor = floor_of("cor")),
       rhat = if (length(rh)) max(rh) else 1,
       worst = sm$variable[which.min(replace(ess, !is.finite(ess), Inf))])
}

.model_fingerprint <- function(model) {
  sd <- model$stan_data
  tf <- tempfile()
  on.exit(unlink(tf))
  con <- file(tf, "wb")
  writeLines(c(model$family$family, model$stan_code, model$batch_cpp$cpp_code %||% ""), con)
  for (k in sort(names(sd))) {
    writeLines(k, con)
    v <- sd[[k]]
    writeBin(as.character(c(dim(v) %||% length(v))), con)
    if (is.numeric(v) || is.logical(v)) writeBin(as.double(v), con) else writeBin(as.character(v), con)
  }
  close(con)
  unname(tools::md5sum(tf))
}

.next_chunk <- function(check_every, drawn, need, t_check, t_draw) {
  opt <- sqrt(2 * max(drawn, need) * t_check / t_draw)
  as.integer(check_every * 2^max(0, floor(log2(max(opt, 1) / check_every))))
}

# =============================================================================
# Adaptive Sampling
# =============================================================================

#' Sample until convergence and precision targets are met
#'
#' Runs one set of JAX chains and extends them in chunks until every monitored
#' parameter has rank-normalised bulk and tail effective sample size of at least
#' `ess_target` and split R-hat at most `rhat_target` (Vehtari et al., 2021), or a
#' cap is reached. Chains continue exactly where the previous chunk stopped, so
#' the draws are identical to one run of the final length. The stopping rule uses
#' the same `posterior` estimators that [summary()] reports.
#'
#' Monitored parameters are classed as population-level (`pop`: fixed effects,
#' thresholds and other family parameters), random-effect and smooth SDs (`sd`),
#' and random-effect correlations (`cor`). Per-level random effects and smooth
#' coefficients are not monitored.
#'
#' @param model A `broc_model` from [broc()].
#' @param ess_target Minimum bulk and tail ESS. One number applies to every class;
#'   a named vector such as `c(pop = 2000, sd = 400, cor = 200)` sets classes
#'   separately (unnamed classes keep 400).
#' @param rhat_target Maximum R-hat over all monitored parameters.
#' @param max_draws Cap on post-warmup draws per chain.
#' @param max_time Optional cap on sampling time in seconds. With both caps, the
#'   run stops at whichever is reached first. A time cap makes the stopping point
#'   machine-dependent; the draws themselves are unaffected.
#' @param check_every Minimum draws per chain between checks. Checks cost time that
#'   grows with the number of draws, so later chunks are lengthened (to
#'   `check_every` times a power of two) to balance checking time against draws
#'   taken past the stopping point; this never changes the draws themselves.
#' @param chains Number of chains.
#' @param iter_warmup Warmup iterations; `NULL` uses the sampler's default.
#' @param seed Seed.
#' @param sampler `"nuts"` or `"drghmc"`.
#' @param resume A fit returned by `autostop_broc()` to continue, for example after
#'   it stopped at a cap. Its chains are extended exactly and its draws kept;
#'   `max_draws` counts them. A fit saved with [save_broc_fit()] (or `file =`) and
#'   reloaded with [load_broc_fit()] can be resumed in a later session. The model
#'   (code and data) must be identical to the one the fit was run on.
#' @param loo Compute PSIS-LOO on the result (stored as `fit$.loo`).
#' @param loo_cores Workers for [loo()].
#' @param progress Print one line per check.
#' @param save_loglik Keep the log-likelihood so [loo()] works on the returned fit.
#'   `NULL` (default) keeps it unless `loo = TRUE`.
#' @param file Optional path: the fit is saved there with [save_broc_fit()]; if it
#'   already exists and `file_refit = FALSE`, it is loaded instead of refitting.
#' @param file_refit Refit even if `file` exists.
#' @param ... Passed to [fit_broc()]. Arguments autostop controls (chains,
#'   iterations, seed, sampler, warm state, log-likelihood handling) are rejected.
#' @return A `broc_fit` with attributes `"stop_reason"` (`"targets"`,
#'   `"max_draws"` or `"max_time"`), `"ess_trace"` (one row per check),
#'   `"targets"` and `"divergences"`. A warning is issued when a cap is reached
#'   before the targets are met.
#' @references Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C.
#'   (2021). Rank-normalization, folding, and localization: An improved R-hat for
#'   assessing convergence of MCMC. *Bayesian Analysis*, 16(2), 667-718.
#' @export
autostop_broc <- function(model, ess_target = 400, rhat_target = 1.01, max_draws = 50000,
                          max_time = NULL, check_every = 100, chains = 4, iter_warmup = NULL,
                          seed = 1L, sampler = c("nuts", "drghmc"), resume = NULL,
                          loo = FALSE, loo_cores = 4, progress = TRUE, save_loglik = NULL,
                          file = NULL, file_refit = FALSE, ...) {
  sampler <- match.arg(sampler)
  targets <- .ess_targets(ess_target)
  whole <- function(x, nm) {
    if (length(x) != 1L || !is.finite(x) || x < 1 || x != round(x)) {
      stop("`", nm, "` must be a positive whole number.", call. = FALSE)
    }
    as.integer(x)
  }
  max_draws <- whole(max_draws, "max_draws")
  check_every <- whole(check_every, "check_every")
  if (!is.null(max_time) && (length(max_time) != 1L || !is.finite(max_time) || max_time <= 0)) {
    stop("`max_time` must be a positive number of seconds.", call. = FALSE)
  }
  if (length(rhat_target) != 1L || !is.finite(rhat_target) || rhat_target < 1) {
    stop("`rhat_target` must be a number of at least 1.", call. = FALSE)
  }
  save_loglik <- if (is.null(save_loglik)) !isTRUE(loo) else isTRUE(save_loglik)
  if (!is.null(file)) {
    file <- if (grepl("\\.rds$", file, ignore.case = TRUE)) file else paste0(file, ".rds")
    if (file.exists(file) && !isTRUE(file_refit)) return(load_broc_fit(file))
  }
  dots <- list(...)
  owned <- c("chains", "parallel_chains", "iter_warmup", "iter_sampling", "seed", "sampler",
             "warm_state", "save_state", "save_loglik", "loo", "file", "file_refit")
  clash <- intersect(names(dots), owned)
  if (length(clash)) {
    stop("autostop_broc() sets ", paste(clash, collapse = ", "), " itself; use its own ",
         "arguments instead of passing them through `...`.", call. = FALSE)
  }
  backend <- dots$backend %||% "jax"
  if (!identical(backend, "jax")) {
    stop("autostop_broc() is JAX-only: continuing chains exactly needs sampler state ",
         "that cmdstanr cannot return. Use backend = \"jax\".", call. = FALSE)
  }
  dots$backend <- NULL

  fits <- list()
  trace <- list()
  total_div <- 0L
  prior_draws <- 0L
  if (!is.null(resume)) {
    if (!inherits(resume, "broc_fit") || is.null(resume$.state_path) ||
        !file.exists(resume$.state_path)) {
      stop("`resume` must be an autostop_broc() fit that still has its sampler state.",
           call. = FALSE)
    }
    fp <- attr(resume, "model_fingerprint")
    if (is.null(fp) || !identical(fp, .model_fingerprint(model))) {
      stop("`resume` was fitted to a different model or data than `model`; it can only be ",
           "continued with the identical model.", call. = FALSE)
    }
    if (!identical(resume$.sampler %||% "nuts", sampler) || resume$.num_chains != chains) {
      stop("`resume` was run with sampler '", resume$.sampler, "' and ", resume$.num_chains,
           " chains; pass the same `sampler` and `chains`.", call. = FALSE)
    }
    if ((isTRUE(loo) || save_loglik) &&
        (is.null(resume$.loglik_path) || !file.exists(resume$.loglik_path))) {
      stop("The fit passed to `resume` no longer has its log-likelihood, so the combined ",
           "fit could not carry one. Resume with save_loglik = FALSE and loo = FALSE, or ",
           "resume a fit that kept its log-likelihood.", call. = FALSE)
    }
    fits[[1]] <- resume
    total_div <- as.integer(attr(resume, "divergences") %||% 0L)
    trace <- list(attr(resume, "ess_trace"))
    prior_draws <- resume$.iter_sampling
  }

  worker <- .jax_worker_start(python = dots$python, chains = chains, sampler = sampler,
                              control = dots$control %||% list())
  on.exit(.jax_worker_stop(worker), add = TRUE)
  token <- basename(tempfile("autostop_"))
  t_start <- proc.time()[["elapsed"]]
  drawn <- prior_draws
  stop_reason <- NULL
  mon <- NULL
  mon_vars <- NULL
  if (!is.null(resume)) {
    mon_vars <- .monitored(posterior::variables(resume$.draws_array))
    mon <- posterior::subset_draws(resume$.draws_array, variable = mon_vars)
  }
  n_next <- check_every
  t_draw <- Inf
  repeat {
    n <- min(n_next, max_draws - drawn)
    first <- length(fits) == 0L || (length(fits) == 1L && !is.null(resume) && drawn == prior_draws)
    args <- c(list(model = model, backend = "jax", chains = chains, parallel_chains = chains,
                   iter_warmup = if (first && is.null(resume)) iter_warmup else 0L,
                   iter_sampling = n, seed = seed, sampler = sampler,
                   save_loglik = isTRUE(loo) || save_loglik, save_state = TRUE,
                   warm_state = if (first && !is.null(resume)) resume$.state_path else NULL,
                   .worker = worker, .worker_token = token,
                   .worker_continue = !first), dots)
    fr <- suppressMessages(do.call(fit_broc, args))
    if (length(fits) && !is.null(fits[[length(fits)]]$.state_path) &&
        !identical(fits[[length(fits)]], resume)) {
      unlink(fits[[length(fits)]]$.state_path)
    }
    fits[[length(fits) + 1L]] <- fr
    drawn <- drawn + n
    div <- .chunk_divergences(fr)
    total_div <- total_div + div

    if (is.null(mon_vars)) mon_vars <- .monitored(posterior::variables(fr$.draws_array))
    chunk_mon <- posterior::subset_draws(fr$.draws_array, variable = mon_vars)
    mon <- if (is.null(mon)) chunk_mon else posterior::bind_draws(mon, chunk_mon, along = "iteration")
    t_diag <- proc.time()[["elapsed"]]
    dg <- .stop_diagnostics(mon)
    t_diag <- proc.time()[["elapsed"]] - t_diag
    elapsed <- proc.time()[["elapsed"]] - t_start
    ess_met <- all(dg$ess >= targets)
    need <- max(drawn * max(targets / pmax(dg$ess, 1)), drawn)
    trace[[length(trace) + 1L]] <- data.frame(
      draws = drawn, divergences = total_div, min_pop_ess = dg$ess[["pop"]],
      min_sd_ess = dg$ess[["sd"]], min_cor_ess = dg$ess[["cor"]], max_rhat = dg$rhat,
      projected_draws = ceiling(need), seconds = elapsed)
    if (isTRUE(progress)) {
      fmt <- function(k) if (is.finite(dg$ess[[k]])) sprintf("%.0f/%g", dg$ess[[k]], targets[[k]]) else "-"
      message(sprintf("[autostop] %d draws/chain | ESS pop %s sd %s cor %s | R-hat %.3f/%g | div %d | %s | %.0fs",
                      drawn, fmt("pop"), fmt("sd"), fmt("cor"), dg$rhat, rhat_target, total_div,
                      if (ess_met) "ESS met, waiting on R-hat" else
                        sprintf("~%d draws needed for ESS", as.integer(ceiling(need))),
                      elapsed))
    }
    if (!first && is.finite(fr$.elapsed %||% NA) && fr$.elapsed > 0) {
      t_draw <- min(t_draw, fr$.elapsed / n)
      n_next <- max(n_next, .next_chunk(check_every, drawn, need, t_diag, t_draw))
    }
    if (ess_met && dg$rhat <= rhat_target) {
      stop_reason <- "targets"
      break
    }
    if (drawn >= max_draws) {
      stop_reason <- "max_draws"
      break
    }
    if (!is.null(max_time) && elapsed >= max_time) {
      stop_reason <- "max_time"
      break
    }
  }
  if (!identical(stop_reason, "targets")) {
    warning(sprintf(paste0("autostop stopped at %s before the targets were met: min ESS pop %.0f/%g, ",
                           "sd %.0f/%g, cor %.0f/%g, max R-hat %.3f/%g (slowest: %s). ",
                           "Continue with autostop_broc(..., resume = fit)."),
                    if (stop_reason == "max_draws") paste(max_draws, "draws per chain") else
                      paste(max_time, "seconds"),
                    dg$ess[["pop"]], targets[["pop"]], dg$ess[["sd"]], targets[["sd"]],
                    dg$ess[["cor"]], targets[["cor"]], dg$rhat, rhat_target, dg$worst),
            call. = FALSE)
  }
  if (total_div > 0) {
    warning(total_div, " divergent transition(s) during sampling.", call. = FALSE)
  }

  pooled <- .pool_chunks(fits)
  paths <- Filter(function(p) !is.null(p) && file.exists(p), lapply(fits, function(z) z$.loglik_path))
  if ((isTRUE(loo) || save_loglik) && length(fits) > 1L && length(paths)) {
    merged <- tempfile("bayesroc_ll_", fileext = ".npy")
    if (length(paths) == 1L) {
      if (!isTRUE(file.copy(paths[[1]], merged))) {
        stop("Could not write the pooled log-likelihood file.", call. = FALSE)
      }
    } else {
      .merge_loglik_npy(unlist(paths), "warm", merged)
    }
    pooled$.loglik_path <- merged
    pooled$.loglik_owned <- TRUE
    pooled$.save_loglik <- save_loglik
  } else if (length(fits) == 1L) {
    pooled$.save_loglik <- save_loglik
  }
  if (length(fits) > 1L) {
    for (z in fits) {
      if (!identical(z, resume) && .owned_sidecar(z)) unlink(z$.loglik_path)
      z$.loglik_path <- NULL
    }
  }
  if (isTRUE(loo)) {
    res <- loo.broc_fit(pooled, cores = loo_cores)
    attr(res, "broc_loo_type") <- "trial"
    pooled$.loo <- res
    if (!save_loglik) pooled <- suppressMessages(drop_loglik(pooled))
  }
  attr(pooled, "stop_reason") <- stop_reason
  attr(pooled, "ess_trace") <- do.call(rbind, trace)
  attr(pooled, "targets") <- c(targets, rhat = rhat_target)
  attr(pooled, "divergences") <- total_div
  attr(pooled, "model_fingerprint") <- .model_fingerprint(model)
  if (!is.null(file)) {
    suppressMessages(save_broc_fit(pooled, file, verbose = FALSE))
  }
  pooled
}
