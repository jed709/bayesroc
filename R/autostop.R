# =============================================================================
# Round Pooling
# =============================================================================

.pool_rounds <- function(fits, restart) {
  if (length(fits) == 1L) return(fits[[1]])
  das <- lapply(fits, function(z) z$.draws_array)
  pooled <- do.call(posterior::bind_draws,
                    c(das, along = if (restart == "warm") "iteration" else "chain"))

  first <- fits[[1]]
  n_ch <- if (restart == "warm") first$.num_chains
          else sum(vapply(fits, function(z) z$.num_chains, integer(1)))
  n_it <- if (restart == "warm")
            sum(vapply(fits, function(z) z$.iter_sampling, integer(1)))
          else first$.iter_sampling

  out <- new_broc_fit(
    backend_fit = list(extra_fields = .pool_extra_fields(fits, restart)),
    model = attr(first, "broc_model"),
    backend = first$.backend,
    num_chains = n_ch,
    iter_sampling = n_it,
    elapsed = sum(vapply(fits, function(z) as.numeric(z$.elapsed %||% 0),
                         numeric(1))),
    draws_array = pooled,
    max_treedepth = first$.max_treedepth,
    sampler = first$.sampler,
    save_loglik = FALSE
  )
  out$.state_path <- fits[[length(fits)]]$.state_path
  out
}

.pool_extra_fields <- function(fits, restart) {
  efs <- lapply(fits, function(z) z$.backend_fit$extra_fields)
  keys <- Reduce(intersect, lapply(efs, names))
  bind <- if (restart == "warm") cbind else rbind
  out <- lapply(keys, function(k) do.call(bind, lapply(efs, function(e) as.matrix(e[[k]]))))
  names(out) <- keys
  out
}

.round_divergences <- function(fit) {
  d <- tryCatch(fit$diagnostic_summary(), error = function(e) NULL)
  if (is.null(d) || is.null(d$num_divergent)) return(0L)
  as.integer(sum(d$num_divergent))
}

# =============================================================================
# ESS Floors
# =============================================================================

.ess_class <- function(v) {
  ifelse(grepl("^sd_|^sds_|^sigma_", v), "sd",
         ifelse(grepl("^cor_|^corr_|^L_corr_", v), "cor", "pop"))
}

.ess_floors <- function(fit) {
  da <- fit$.draws_array
  vars <- posterior::variables(da)
  vars <- vars[!grepl("^log_lik|^lp__|^z_|^u_|^zs_|^s_", vars)]
  if (!length(vars)) return(c(pop = Inf, sd = Inf, cor = Inf))
  sm <- posterior::summarise_draws(
    posterior::subset_draws(da, variable = vars),
    ess_bulk = posterior::ess_bulk, ess_tail = posterior::ess_tail)
  cls <- .ess_class(sm$variable)
  flr <- function(sel) {
    if (!any(sel)) return(Inf)
    min(sm$ess_bulk[sel], sm$ess_tail[sel], na.rm = TRUE)
  }
  c(pop = flr(cls == "pop"), sd = flr(cls == "sd"), cor = flr(cls == "cor"))
}

# =============================================================================
# Adaptive Sampling
# =============================================================================

#' Sample until ESS targets are met
#'
#' @param model An `broc_model` from [broc()].
#' @param pop_target Minimum bulk and tail ESS for population parameters.
#' @param sd_target Minimum bulk and tail ESS for random-effect SDs.
#' @param cor_target Minimum bulk and tail ESS for random-effect correlations.
#' @param chains Chains per round.
#' @param iter_warmup Warmup for the first round (and every round when cold).
#' @param samples_per_round Post-warmup draws per chain per round.
#' @param max_rounds Cap on rounds.
#' @param restart `"warm"` continues the chains, `"cold"` re-warms each round.
#' @param seed Base seed; each round offsets it.
#' @param sampler `"nuts"` or `"drghmc"`.
#' @param loo Compute PSIS-LOO on the pooled fit (stored as `fit$.loo`).
#' @param save_loglik Keep the pooled log-likelihood so [loo()] works on the
#'   returned fit. `NULL` (default) keeps it unless `loo = TRUE`.
#' @param file Optional path: the pooled fit is saved there with [save_broc_fit()];
#'   if it already exists and `file_refit = FALSE`, it is loaded instead of refitting.
#' @param file_refit Refit even if `file` exists.
#' @param loo_cores Workers for [loo()].
#' @param progress Print a per-round ESS line.
#' @param ... Passed to [fit_broc()]. Arguments that autostop controls per round
#'   (chains, iterations, seed, sampler, warm state, log-likelihood handling) are
#'   rejected.
#' @return The pooled `broc_fit`, with attribute `"ess_trace"`. A warning is
#'   issued when `max_rounds` is reached before every ESS target is met.
#' @export
autostop_broc <- function(model, pop_target = 5000, sd_target = 400,
                          cor_target = 400, chains = 4, iter_warmup = 1000,
                          samples_per_round = 1000, max_rounds = 10,
                          restart = c("warm", "cold"), seed = 1L,
                          sampler = c("nuts", "drghmc"), loo = FALSE,
                          loo_cores = 4, progress = TRUE, save_loglik = NULL,
                          file = NULL, file_refit = FALSE, ...) {
  restart <- match.arg(restart)
  sampler <- match.arg(sampler)
  if (length(max_rounds) != 1L || !is.finite(max_rounds) || max_rounds < 1 ||
      max_rounds != round(max_rounds)) {
    stop("`max_rounds` must be a positive whole number.", call. = FALSE)
  }
  save_loglik <- if (is.null(save_loglik)) !isTRUE(loo) else isTRUE(save_loglik)
  if (!is.null(file)) {
    file <- if (grepl("\\.rds$", file, ignore.case = TRUE)) file else paste0(file, ".rds")
    if (file.exists(file) && !isTRUE(file_refit)) return(load_broc_fit(file))
  }
  dots <- list(...)
  owned <- c("chains", "parallel_chains", "iter_warmup", "iter_sampling", "seed",
             "sampler", "warm_state", "save_state", "save_loglik", "loo", "file",
             "file_refit")
  clash <- intersect(names(dots), owned)
  if (length(clash)) {
    stop("autostop_broc() sets ", paste(clash, collapse = ", "), " itself; use its own ",
         "arguments instead of passing them through `...`.", call. = FALSE)
  }
  backend <- dots$backend %||% "jax"
  if (!identical(backend, "jax")) {
    stop("autostop_broc() is JAX-only: warm continuation needs sampler state ",
         "that cmdstanr cannot return. Use backend = \"jax\".", call. = FALSE)
  }
  dots$backend <- NULL

  fits <- list()
  trace <- list()
  total_div <- 0L
  state <- NULL
  worker <- .jax_worker_start(python = dots$python, chains = chains, sampler = sampler,
                              control = dots$control %||% list())
  on.exit(.jax_worker_stop(worker), add = TRUE)
  token <- basename(tempfile("autostop_"))

  for (r in seq_len(max_rounds)) {
    warm <- restart == "warm" && r > 1L
    args <- c(list(model = model, backend = "jax", chains = chains,
                   parallel_chains = chains,
                   iter_warmup = if (warm) 0L else iter_warmup,
                   iter_sampling = samples_per_round,
                   seed = seed + r * 1009L, sampler = sampler,
                   save_loglik = isTRUE(loo) || save_loglik,
                   warm_state = if (warm) state else NULL,
                   save_state = restart == "warm",
                   .worker = worker,
                   .worker_token = if (restart == "warm") token else NULL), dots)
    fr <- do.call(fit_broc, args)
    state <- fr$.state_path
    fits[[length(fits) + 1L]] <- fr

    div <- .round_divergences(fr)
    total_div <- total_div + div
    pooled <- .pool_rounds(fits, restart)
    fl <- .ess_floors(pooled)
    trace[[r]] <- data.frame(
      round = r,
      total_draws = pooled$.num_chains * pooled$.iter_sampling,
      divergences = div, cumulative_divergences = total_div,
      min_pop_ess = fl[["pop"]], min_sd_ess = fl[["sd"]],
      min_cor_ess = fl[["cor"]])

    if (div > 0) {
      warning("Round ", r, " produced ", div, " divergent transition(s) ",
              "(cumulative ", total_div, ").", call. = FALSE)
    }
    if (isTRUE(progress)) {
      message(sprintf(paste0("[autostop:%s] round %d: %d draws | div %d (cum %d)",
                             " | min ESS  POP %.0f (t %g)  SD %.0f (t %g)  ",
                             "COR %.0f (t %g)"),
                      restart, r, pooled$.num_chains * pooled$.iter_sampling,
                      div, total_div, fl[["pop"]], pop_target, fl[["sd"]],
                      sd_target, fl[["cor"]], cor_target))
    }
    met <- fl[["pop"]] >= pop_target && fl[["sd"]] >= sd_target &&
      fl[["cor"]] >= cor_target
    if (met) break
  }
  if (!met) {
    warning("ESS targets not met after ", max_rounds, " round(s): min ESS POP ",
            round(fl[["pop"]]), "/", pop_target, ", SD ", round(fl[["sd"]]), "/", sd_target,
            ", COR ", round(fl[["cor"]]), "/", cor_target,
            ". Increase max_rounds or samples_per_round.", call. = FALSE)
  }

  pooled <- .pool_rounds(fits, restart)
  paths <- Filter(function(p) !is.null(p) && file.exists(p),
                  lapply(fits, function(z) z$.loglik_path))
  single <- length(fits) == 1L
  if ((isTRUE(loo) || save_loglik) && !single) {
    merged <- tempfile("bayesroc_ll_", fileext = ".npy")
    if (length(paths) == 1L) {
      if (!isTRUE(file.copy(paths[[1]], merged))) {
        stop("Could not write the pooled log-likelihood file.", call. = FALSE)
      }
    } else if (length(paths) > 1L) {
      .merge_loglik_npy(unlist(paths), restart, merged)
    }
    if (file.exists(merged)) {
      pooled$.loglik_path <- merged
      pooled$.loglik_owned <- TRUE
      pooled$.save_loglik <- save_loglik
    }
  }
  if (single) {
    pooled$.save_loglik <- save_loglik
  } else {
    for (z in fits) {
      if (.owned_sidecar(z)) unlink(z$.loglik_path)
      z$.loglik_path <- NULL
    }
  }
  states <- unique(unlist(lapply(fits[-length(fits)], function(z) z$.state_path)))
  states <- setdiff(states, pooled$.state_path)
  if (length(states)) unlink(states)
  if (isTRUE(loo)) {
    res <- loo.broc_fit(pooled, cores = loo_cores)
    attr(res, "broc_loo_type") <- "trial"
    pooled$.loo <- res
    if (!save_loglik) pooled <- suppressMessages(drop_loglik(pooled))
  }
  attr(pooled, "ess_trace") <- do.call(rbind, trace)
  attr(pooled, "targets") <- c(pop = pop_target, sd = sd_target, cor = cor_target)
  attr(pooled, "restart") <- restart
  attr(pooled, "divergences") <- total_div
  if (!is.null(file)) {
    suppressMessages(save_broc_fit(pooled, file, verbose = FALSE))
  }
  pooled
}
