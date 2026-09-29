# =============================================================================
# Round Pooling
# =============================================================================

.bind_draw_arrays <- function(arrs) {
  do.call(function(...) posterior::bind_draws(..., along = "iteration"), arrs)
}

.bind_chain_arrays <- function(arrs) {
  do.call(function(...) posterior::bind_draws(..., along = "chain"), arrs)
}

.pool_rounds <- function(fits, restart) {
  if (length(fits) == 1L) return(fits[[1]])
  das <- lapply(fits, function(z) z$.draws_array)
  pooled <- if (restart == "warm") .bind_draw_arrays(das) else .bind_chain_arrays(das)

  first <- fits[[1]]
  n_ch <- if (restart == "warm") first$.num_chains
          else sum(vapply(fits, function(z) z$.num_chains, integer(1)))
  n_it <- if (restart == "warm")
            sum(vapply(fits, function(z) z$.iter_sampling, integer(1)))
          else first$.iter_sampling

  out <- new_broc_fit(
    backend_fit = NULL,
    model = attr(first, "broc_model"),
    backend = first$.backend,
    num_chains = n_ch,
    iter_sampling = n_it,
    elapsed = sum(vapply(fits, function(z) as.numeric(z$.elapsed %||% 0),
                         numeric(1))),
    draws_array = pooled,
    max_treedepth = first$.max_treedepth,
    sampler = first$.sampler,
    save_loglik = FALSE,
    diagnostics = .pool_diagnostics(fits)
  )
  out$.state_path <- fits[[length(fits)]]$.state_path
  out
}

.pool_diagnostics <- function(fits) {
  ds <- lapply(fits, function(z) tryCatch(z$diagnostic_summary(),
                                          error = function(e) NULL))
  ds <- Filter(Negate(is.null), ds)
  if (!length(ds)) return(NULL)
  keys <- unique(unlist(lapply(ds, names)))
  out <- lapply(keys, function(k) {
    vals <- lapply(ds, function(z) z[[k]])
    vals <- Filter(Negate(is.null), vals)
    if (!length(vals)) return(NULL)
    if (k == "num_divergent") Reduce(`+`, vals) else vals[[length(vals)]]
  })
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
  ifelse(grepl("^sd_|^sigma_[a-z]+_", v), "sd",
         ifelse(grepl("^cor_|^corr_|^L_corr_", v), "cor", "pop"))
}

.ess_floors <- function(fit) {
  da <- fit$.draws_array
  vars <- posterior::variables(da)
  vars <- vars[!grepl("^log_lik|^lp__|^z_|^u_", vars)]
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
#' @param loo Compute PSIS-LOO on the pooled fit.
#' @param loo_cores Workers for [loo()].
#' @param progress Print a per-round ESS line.
#' @param ... Passed to [fit_broc()].
#' @return The pooled `broc_fit`, with attribute `"ess_trace"`.
#' @export
autostop_broc <- function(model, pop_target = 5000, sd_target = 400,
                          cor_target = 400, chains = 4, iter_warmup = 1000,
                          samples_per_round = 1000, max_rounds = 10,
                          restart = c("warm", "cold"), seed = 1L,
                          sampler = c("nuts", "drghmc"), loo = FALSE,
                          loo_cores = 4, progress = TRUE, ...) {
  restart <- match.arg(restart)
  sampler <- match.arg(sampler)
  dots <- list(...)
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

  for (r in seq_len(max_rounds)) {
    warm <- restart == "warm" && r > 1L
    args <- c(list(model = model, backend = "jax", chains = chains,
                   parallel_chains = chains,
                   iter_warmup = if (warm) 0L else iter_warmup,
                   iter_sampling = samples_per_round,
                   seed = seed + r * 1009L, sampler = sampler,
                   save_loglik = isTRUE(loo),
                   warm_state = if (warm) state else NULL,
                   save_state = restart == "warm"), dots)
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
    if (fl[["pop"]] >= pop_target && fl[["sd"]] >= sd_target &&
        fl[["cor"]] >= cor_target) break
  }

  pooled <- .pool_rounds(fits, restart)
  if (isTRUE(loo)) {
    paths <- Filter(function(p) !is.null(p) && file.exists(p),
                    lapply(fits, function(z) z$.loglik_path))
    if (length(paths) == 1L) {
      pooled$.loglik_path <- paths[[1]]
    } else if (length(paths) > 1L) {
      merged <- tempfile("bayesroc_ll_", fileext = ".npy")
      .merge_loglik_npy(unlist(paths), restart, merged)
      pooled$.loglik_path <- merged
    }
    res <- loo.broc_fit(pooled, cores = loo_cores)
    attr(res, "broc_loo_type") <- "trial"
    pooled$.loo <- res
    pooled <- suppressMessages(drop_loglik(pooled))
  }
  attr(pooled, "ess_trace") <- do.call(rbind, trace)
  attr(pooled, "targets") <- c(pop = pop_target, sd = sd_target, cor = cor_target)
  attr(pooled, "restart") <- restart
  attr(pooled, "divergences") <- total_div
  pooled
}
