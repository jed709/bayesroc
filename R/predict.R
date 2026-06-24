# Prediction Methods for SDT Model Fits
# Generate posterior predictive category probabilities or sampled responses


# =============================================================================
# Gap Transform Helpers
# =============================================================================

.gap_transform <- function(x, gap_link = "log") {
  if (identical(gap_link, "softplus")) log1p(exp(x)) else exp(x)
}

.gap_param <- function(gap_link = "log", suffix = "") {
  base <- if (identical(gap_link, "softplus")) "beta_raw_gaps" else "beta_log_gaps"
  paste0(base, suffix)
}

# Detect whether gap draws use 1D (gap[k]) or 2D (gap[k,j]) naming.
# JAX backend with P_crit==1 uses 1D; Stan always uses 2D.
.gap_names_1d <- function(draws, gap_prefix) {
  avail <- colnames(draws)
  test_1d <- paste0(gap_prefix, "[1]")
  test_2d <- paste0(gap_prefix, "[1,1]")
  (test_1d %in% avail) && !(test_2d %in% avail)
}

# Get gap column names for gap index k, handling 1D/2D naming
.gap_cols <- function(gap_prefix, k, P, use_1d) {
  if (use_1d) {
    paste0(gap_prefix, "[", k, "]")
  } else {
    paste0(gap_prefix, "[", k, ",", 1:P, "]")
  }
}

# =============================================================================
# Link CDF Helper
# =============================================================================

#' R-side link CDF
#' @noRd
link_cdf_r <- function(x, link_name) {
  switch(link_name,
    probit = pnorm(x),
    logit  = plogis(x),
    stop("Unknown link: ", link_name)
  )
}


# =============================================================================
# Main Prediction Function
# =============================================================================

#' Posterior Predictions from a Fitted Bayesian ROC Model
#'
#' Generate posterior predictive response probabilities or simulated responses
#' from a fitted model.
#'
#' @param object A `broc_fit` object from [fit_broc()].
#' @param newdata Optional data frame for out-of-sample prediction. If `NULL`
#'   (default), uses the training data.
#' @param type `"response"` (default) returns posterior category probabilities
#'   P(Y = k); `"prediction"` returns simulated categorical responses.
#' @param re_formula Controls random effects inclusion. `NULL` (default) includes
#'   all random effects; `NA` excludes all (population-level predictions only).
#' @param ndraws Number of posterior draws to use. `NULL` (default) uses all
#'   available draws.
#' @param summary If `TRUE` (default), returns posterior mean and credible
#'   intervals. If `FALSE`, returns the full S x N x K array of draws.
#' @param prob Width of credible intervals when `summary = TRUE` (default 0.95).
#' @param cores Number of cores for parallel probability computation (default 1).
#' @param response Which response dimension. For bivariate families: `1` (detection)
#'   or `2` (source). For CDP: `1` (confidence, default) or `2` (R/K/G probabilities).
#'   Ignored for other families.
#' @param seed Optional integer for reproducibility. Controls the random
#'   subsampling of `ndraws` and (for `type = "prediction"`) the categorical
#'   draws used to simulate responses. The previous `.Random.seed` is restored
#'   on exit. Default `NULL` leaves the RNG untouched.
#' @param allow_new_levels If `TRUE`, allow `newdata` to contain random-effect
#'   levels not seen during fitting; their effects are drawn from the estimated
#'   population distribution. Default `FALSE` errors if `newdata` introduces new
#'   levels.
#' @param ... Ignored.
#' @return Depends on `type` and `summary`:
#'   \describe{
#'     \item{`type = "response"`, `summary = TRUE`}{A data frame with columns
#'       for each category's posterior mean probability plus lower/upper CI bounds.}
#'     \item{`type = "response"`, `summary = FALSE`}{An S x N x K array where
#'       S = draws, N = observations, K = response categories.}
#'     \item{`type = "prediction"`, `summary = TRUE`}{A vector of modal predicted categories.}
#'     \item{`type = "prediction"`, `summary = FALSE`}{An S x N matrix of
#'       simulated categorical responses.}
#'   }
#' @examples
#' \dontrun{
#' fit <- fit_broc(model)
#' # Category probabilities (summarized)
#' preds <- predict(fit)
#' # Full posterior array
#' preds_full <- predict(fit, summary = FALSE, ndraws = 100)
#' # Population-level only (no random effects)
#' preds_pop <- predict(fit, re_formula = NA)
#' }
#' @seealso [pp_check()] for visual posterior predictive checks.
#' @method predict broc_fit
#' @export
predict.broc_fit <- function(object, newdata = NULL, type = c("response", "prediction"),
                             re_formula = NULL, ndraws = NULL, summary = TRUE,
                             prob = 0.95, cores = 1, response = 1, seed = NULL,
                             allow_new_levels = FALSE, ...) {
  fit <- object
  type <- match.arg(type)

  # Optional reproducibility. Save + restore the caller's RNG state so seed=
  # only affects sampling inside this call, not the global stream.
  if (!is.null(seed)) {
    old_seed <- if (exists(".Random.seed", envir = .GlobalEnv))
      get(".Random.seed", envir = .GlobalEnv) else NULL
    on.exit({
      if (is.null(old_seed)) {
        if (exists(".Random.seed", envir = .GlobalEnv))
          rm(".Random.seed", envir = .GlobalEnv)
      } else {
        assign(".Random.seed", old_seed, envir = .GlobalEnv)
      }
    }, add = TRUE)
    set.seed(seed)
  }

  model <- attr(fit, "broc_model")
  if (is.null(model)) {
    stop("Fit object must have 'broc_model' attribute. Use fit_broc() to fit the model.")
  }

  family <- model$family
  family_name <- family$family
  gap_link <- if (!is.null(model$gap_link)) model$gap_link else "log"

  supported <- c("evsdt", "uvsdt", "dpsdt", "mixture", "cumulative", "source_mixture",
                  "bivariate_sdt", "bivariate_dp", "bivariate_cumulative", "vrdp2d", "cdp")
  if (!family_name %in% supported) {
    stop("predict() does not yet support family '", family_display_name(family_name),
         "'. Supported: ",
         paste(vapply(supported, family_display_name, character(1)), collapse = ", "))
  }

  is_cumulative <- family_name == "cumulative"
  is_bivariate <- family_name %in% c("bivariate_sdt", "bivariate_dp", "vrdp2d", "bivariate_cumulative")
  is_cdp <- family_name == "cdp"
  link_name <- family$link$name
  K <- model$model_data$K
  K2 <- model$model_data$K2  # NULL for non-bivariate

  # ---- Data source ----
  if (!is.null(newdata)) {
    data_src <- rebuild_design_matrices_for_linpred(model, newdata,
                                                    allow_new_levels)
  } else {
    data_src <- build_data_source_from_model(model)
  }
  data_src$.allow_new_levels <- isTRUE(allow_new_levels)

  N <- data_src$N

  # ---- Extract posterior draws ----
  vars_needed <- get_predict_variables(model, include_re = !identical(re_formula, NA))
  if (allow_new_levels) {
    extra <- get_re_prior_variables(model)
    meta_base <- unique(sub("\\[.*$", "", fit$metadata()$variables))
    extra <- intersect(extra, meta_base)
    vars_needed <- unique(c(vars_needed, extra))
  }
  draws <- fit$draws(variables = vars_needed, format = "matrix")
  S <- nrow(draws)

  # Subsample draws
  if (!is.null(ndraws) && ndraws < S) {
    idx <- sample.int(S, ndraws)
    draws <- draws[idx, , drop = FALSE]
    S <- ndraws
  }

  # ---- Compute trial-level parameters ----
  trial_params <- compute_trial_params(draws, model, data_src, re_formula, S, N)

  # ---- Compute thresholds ----
  # For CDP: J thresholds (one per old confidence level), not K-1
  n_thresh_K <- if (is_cdp) family$J + 1 else K
  thresh <- compute_thresholds(draws, model, data_src, re_formula, S, N, n_thresh_K)

  # Compute second-dimension thresholds for bivariate families
  thresh2 <- NULL
  if (is_bivariate) {
    thresh2 <- compute_thresholds_2(draws, model, data_src, re_formula, S, N)
  }

  # ---- Compute category probabilities ----
  probs <- compute_category_probs(trial_params, thresh, data_src, model, link_name, S, N, K,
                                    thresh2 = thresh2, cores = cores)

  # ---- Bivariate output path (4D probs: S x N x K1 x K2) ----
  if (is_bivariate) {
    K1 <- K

    if (type == "prediction") {
      # Sample (y1, y2) pairs from joint distribution
      pred_y1 <- matrix(0L, nrow = S, ncol = N)
      pred_y2 <- matrix(0L, nrow = S, ncol = N)
      for (s in seq_len(S)) {
        for (n in seq_len(N)) {
          p_joint <- pmax(as.vector(probs[s, n, , ]), 0)
          p_joint <- p_joint / sum(p_joint)
          cell <- sample.int(K1 * K2, 1L, prob = p_joint)
          pred_y1[s, n] <- ((cell - 1) %% K1) + 1L
          pred_y2[s, n] <- ((cell - 1) %/% K1) + 1L
        }
      }
      if (summary) {
        pred_summary <- data.frame(
          trial = seq_len(N),
          mode_y1 = apply(pred_y1, 2, function(x) which.max(tabulate(x, nbins = K1))),
          mode_y2 = apply(pred_y2, 2, function(x) which.max(tabulate(x, nbins = K2))),
          mean_y1 = colMeans(pred_y1),
          mean_y2 = colMeans(pred_y2)
        )
        return(pred_summary)
      }
      return(list(y1 = pred_y1, y2 = pred_y2))
    }

    # type = "response"
    if (summary) {
      probs_lower <- (1 - prob) / 2
      probs_upper <- 1 - probs_lower

      result_list <- vector("list", K1 * K2)
      idx <- 0
      for (j in seq_len(K1)) {
        for (k in seq_len(K2)) {
          idx <- idx + 1
          p_jk <- probs[, , j, k]  # S x N
          result_list[[idx]] <- data.frame(
            trial = seq_len(N),
            category1 = j,
            category2 = k,
            mean = colMeans(p_jk),
            lower = apply(p_jk, 2, quantile, probs = probs_lower),
            upper = apply(p_jk, 2, quantile, probs = probs_upper)
          )
        }
      }
      result <- do.call(rbind, result_list)
      rownames(result) <- NULL
      return(result)
    }

    # summary = FALSE: return S x N x K1 x K2 array
    return(probs)
  }

  # ---- CDP response=2 path: R/K(/G) probabilities ----
  if (is_cdp && response == 2) {
    p_remember <- attr(probs, "p_remember")  # S x N x K: P(R | Y=k)
    p_know <- attr(probs, "p_know")          # S x N x K or NULL
    old_levels <- family$old_levels
    n_rkg <- if (!is.null(family$n_rkg)) family$n_rkg else 2L

    # Compute marginal R/K(/G) probs by weighting conditionals by P(Y=k)
    # and summing over old confidence levels.
    # P(R | trial) = sum_k P(R | Y=k) * P(Y=k) for k in old_levels
    # P(old response | trial) = sum_k P(Y=k) for k in old_levels
    rk_probs <- array(0, dim = c(S, N, n_rkg))

    for (s in seq_len(S)) {
      for (n in seq_len(N)) {
        p_old <- sum(probs[s, n, old_levels])
        if (p_old < 1e-10) {
          # No old response probability -- assign uniform R/K
          rk_probs[s, n, ] <- 1 / n_rkg
          next
        }
        p_r <- sum(p_remember[s, n, old_levels] * probs[s, n, old_levels]) / p_old
        if (n_rkg == 3 && !is.null(p_know)) {
          p_k <- sum(p_know[s, n, old_levels] * probs[s, n, old_levels]) / p_old
          p_g <- max(1 - p_r - p_k, 0)
          rk_probs[s, n, ] <- c(p_r, p_k, p_g)
        } else {
          rk_probs[s, n, ] <- c(p_r, max(1 - p_r, 0))
        }
      }
    }

    rk_labels <- if (n_rkg == 3) c("Remember", "Know", "Guess") else c("Remember", "Know")

    if (type == "prediction") {
      # Sample R/K(/G) per draw per observation
      pred_rk <- matrix(0L, nrow = S, ncol = N)
      for (s in seq_len(S)) {
        for (n in seq_len(N)) {
          pred_rk[s, n] <- sample.int(n_rkg, 1L, prob = pmax(rk_probs[s, n, ], 1e-10))
        }
      }
      if (summary) {
        return(data.frame(
          trial = seq_len(N),
          mode = apply(pred_rk, 2, function(x) rk_labels[which.max(tabulate(x, nbins = n_rkg))]),
          p_remember = colMeans(rk_probs[, , 1])
        ))
      }
      return(pred_rk)
    }

    # type = "response": return R/K(/G) probability array
    if (summary) {
      probs_lower <- (1 - prob) / 2
      probs_upper <- 1 - probs_lower
      result_list <- vector("list", n_rkg)
      for (r in seq_len(n_rkg)) {
        result_list[[r]] <- data.frame(
          trial = seq_len(N),
          category = rk_labels[r],
          mean = colMeans(rk_probs[, , r]),
          lower = apply(rk_probs[, , r], 2, quantile, probs = probs_lower),
          upper = apply(rk_probs[, , r], 2, quantile, probs = probs_upper)
        )
      }
      return(do.call(rbind, result_list))
    }
    return(rk_probs)
  }

  # ---- CDP output path ----
  if (is_cdp && type == "prediction") {
    p_remember <- attr(probs, "p_remember")
    p_know <- attr(probs, "p_know")  # NULL for n_rkg==2
    old_levels <- family$old_levels
    n_rkg <- if (!is.null(family$n_rkg)) family$n_rkg else 2L

    pred_y <- matrix(0L, nrow = S, ncol = N)
    pred_rk <- matrix(NA_integer_, nrow = S, ncol = N)

    for (s in seq_len(S)) {
      for (n in seq_len(N)) {
        p <- pmax(probs[s, n, ], 0)
        p <- p / sum(p)
        y <- sample.int(K, 1L, prob = p)
        pred_y[s, n] <- y

        # For old-level responses, sample R/K(/G)
        if (y %in% old_levels) {
          p_r <- p_remember[s, n, y]
          if (n_rkg == 3 && !is.null(p_know)) {
            # R/K/G sampling
            p_k <- p_know[s, n, y]
            p_g <- max(1 - p_r - p_k, 0)
            pred_rk[s, n] <- sample(1:3, 1L, prob = c(p_r, p_k, p_g))
          } else {
            # R/K sampling
            if (!is.na(p_r) && p_r > 0) {
              pred_rk[s, n] <- if (runif(1) < p_r) 1L else 2L
            } else {
              pred_rk[s, n] <- 2L
            }
          }
        }
        # NA for new-level responses
      }
    }

    if (summary) {
      pred_summary <- data.frame(
        trial = seq_len(N),
        mode = apply(pred_y, 2, function(x) which.max(tabulate(x, nbins = K))),
        mean = colMeans(pred_y)
      )
      return(pred_summary)
    }
    return(list(y = pred_y, rk = pred_rk))
  }

  # ---- Standard output path (univariate families + CDP response) ----
  if (type == "prediction") {
    pred_mat <- matrix(0L, nrow = S, ncol = N)
    for (s in seq_len(S)) {
      for (n in seq_len(N)) {
        p <- pmax(probs[s, n, ], 0)
        p <- p / sum(p)
        pred_mat[s, n] <- sample.int(K, 1L, prob = p)
      }
    }
    if (summary) {
      pred_summary <- data.frame(
        trial = seq_len(N),
        mode = apply(pred_mat, 2, function(x) {
          tab <- tabulate(x, nbins = K)
          which.max(tab)
        }),
        mean = colMeans(pred_mat)
      )
      return(pred_summary)
    }
    return(pred_mat)
  }

  # ---- Type = "response": return probabilities ----
  if (summary) {
    probs_lower <- (1 - prob) / 2
    probs_upper <- 1 - probs_lower

    result_list <- vector("list", K)
    for (k in seq_len(K)) {
      p_k <- probs[, , k]  # S x N
      result_list[[k]] <- data.frame(
        trial = seq_len(N),
        category = k,
        mean = colMeans(p_k),
        lower = apply(p_k, 2, quantile, probs = probs_lower),
        upper = apply(p_k, 2, quantile, probs = probs_upper)
      )
    }
    result <- do.call(rbind, result_list)
    rownames(result) <- NULL
    return(result)
  }

  # summary = FALSE: return S x N x K array
  probs
}


# =============================================================================
# LOO-CV Wrapper
# =============================================================================

#' LOO-CV for Fitted SDT Models
#'
#' Approximate leave-one-out cross-validation (PSIS-LOO) for model comparison via
#' [loo::loo_compare()].
#'
#' For trial-level data, passes log_lik directly to loo. For aggregated data with
#' counts, `log_lik[n]` is the per-row log-probability (without count weighting).
#'
#' @details Computed in parallel across observations with a PSOCK cluster.
#' @param x A `broc_fit` object from [fit_broc()].
#' @param cores Number of cores for the PSOCK cluster (default 4)
#' @param ... Additional arguments passed to [loo::loo()]
#' @return A `loo` object.
#' @importFrom loo loo
#' @method loo broc_fit
#' @export
loo.broc_fit <- function(x, cores = 4, ...) {
  fit <- x

  # Extract log_lik as a plain matrix (works for both Stan and JAX backends)
  ll <- as.matrix(fit$draws(variables = "log_lik", format = "matrix"))
  S <- nrow(ll)
  N <- ncol(ll)
  n_chains <- fit$num_chains()
  n_iter <- S %/% n_chains
  chain_id <- rep(seq_len(n_chains), each = n_iter)

  n_workers <- max(1L, min(cores, N))

  # Single-worker fast path: no cluster overhead, but still capture warnings
  if (n_workers == 1L) {
    res <- withCallingHandlers(
      {
        r_eff <- loo::relative_eff(exp(ll), chain_id = chain_id)
        loo::loo(ll, r_eff = r_eff, cores = 1, ...)
      },
      warning = function(w) NULL  # let warnings flow through normally
    )
    return(res)
  }

  cl <- parallel::makeCluster(n_workers)
  on.exit(parallel::stopCluster(cl))
  parallel::clusterEvalQ(cl, library(loo))

  chunks <- split(seq_len(N), rep(seq_len(n_workers), length.out = N))
  chunk_data <- lapply(chunks, function(idx) {
    list(ll = ll[, idx, drop = FALSE], chain_id = chain_id)
  })
  rm(ll)

  # Worker: compute r_eff + loo on its chunk. Wrapped in withCallingHandlers to
  # capture warnings (e.g. high pareto-k) and errors per chunk rather than
  # losing them inside the subprocess.
  .loo_worker <- function(cd) {
    captured <- character(0)
    out <- tryCatch(
      withCallingHandlers(
        {
          ll_chunk <- cd$ll
          # Defensive: replace any non-finite log_lik with very negative
          # finite values so loo doesn't error on NaN/Inf inputs.
          if (!all(is.finite(ll_chunk))) {
            n_bad <- sum(!is.finite(ll_chunk))
            captured <- c(captured,
                          sprintf("%d non-finite log_lik values replaced with -1e6",
                                  n_bad))
            ll_chunk[!is.finite(ll_chunk)] <- -1e6
          }
          r_eff <- loo::relative_eff(exp(ll_chunk), chain_id = cd$chain_id)
          # Defensive: relative_eff can return NA for degenerate columns;
          # replace with 1 (no autocorrelation correction) so loo doesn't
          # propagate NA into its internal `if (...)` checks.
          if (any(is.na(r_eff))) {
            n_na <- sum(is.na(r_eff))
            captured <- c(captured,
                          sprintf("%d NA r_eff values replaced with 1.0",
                                  n_na))
            r_eff[is.na(r_eff)] <- 1
          }
          loo::loo(ll_chunk, r_eff = r_eff, cores = 1)
        },
        warning = function(w) {
          captured[[length(captured) + 1]] <<- conditionMessage(w)
          invokeRestart("muffleWarning")
        }
      ),
      error = function(e) {
        # Surface the error with chunk info instead of letting cluster
        # error out cryptically.
        list(.loo_worker_error = conditionMessage(e),
             chunk_dim = dim(cd$ll))
      }
    )
    list(loo = out, warnings = captured)
  }
  environment(.loo_worker) <- globalenv()

  worker_results <- parallel::clusterApply(cl, chunk_data, .loo_worker)
  # Surface any worker errors to the user
  err_idx <- which(vapply(worker_results, function(r) {
    is.list(r$loo) && !is.null(r$loo$.loo_worker_error)
  }, logical(1)))
  if (length(err_idx) > 0) {
    msgs <- vapply(err_idx, function(i) {
      sprintf("  worker %d (chunk dim %s): %s",
              i,
              paste(worker_results[[i]]$loo$chunk_dim, collapse = "x"),
              worker_results[[i]]$loo$.loo_worker_error)
    }, character(1))
    stop("loo() failed in ", length(err_idx), " of ", length(worker_results),
         " workers:\n", paste(msgs, collapse = "\n"),
         "\n\nTry rerunning with cores = 1 to get a clearer error trace.",
         call. = FALSE)
  }
  sub_loos <- lapply(worker_results, `[[`, "loo")

  # Combine pointwise results in original observation order
  all_idx <- unlist(chunks)
  reorder <- order(all_idx)

  pointwise <- do.call(rbind, lapply(sub_loos, function(x) x$pointwise))
  pointwise <- pointwise[reorder, , drop = FALSE]

  pareto_k <- unlist(lapply(sub_loos, function(x) x$diagnostics$pareto_k))
  pareto_k <- pareto_k[reorder]

  n_eff <- unlist(lapply(sub_loos, function(x) x$diagnostics$n_eff))
  n_eff <- n_eff[reorder]

  # Recompute aggregate estimates from combined pointwise
  estimates <- cbind(
    Estimate = c(
      elpd_loo = sum(pointwise[, "elpd_loo"]),
      p_loo    = sum(pointwise[, "p_loo"]),
      looic    = sum(pointwise[, "looic"])
    ),
    SE = c(
      sqrt(N * var(pointwise[, "elpd_loo"])),
      sqrt(N * var(pointwise[, "p_loo"])),
      sqrt(N * var(pointwise[, "looic"]))
    )
  )

  result <- structure(
    list(
      pointwise   = pointwise,
      estimates   = estimates,
      diagnostics = list(pareto_k = pareto_k, n_eff = n_eff)
    ),
    class = c("psis_loo", "loo"),
    dims  = c(S, N)
  )

  # Re-issue diagnostic warnings from the combined view. Per-chunk loo()
  # warnings are misleading because each chunk only saw a subset of obs;
  # recompute on the combined pareto_k vector (loo's own threshold
  # is 0.7 for "bad" with reff-aware variants).
  bad_k_count <- sum(pareto_k > 0.7, na.rm = TRUE)
  vbad_k_count <- sum(pareto_k > 1.0, na.rm = TRUE)
  if (vbad_k_count > 0) {
    warning("Found ", vbad_k_count, " observation(s) with Pareto k > 1.0.",
            call. = FALSE)
  } else if (bad_k_count > 0) {
    warning("Found ", bad_k_count, " observation(s) with Pareto k > 0.7.",
            call. = FALSE)
  }
  # Also surface any per-worker warning that wasn't a pareto-k message
  # (e.g. "Some Pareto k diagnostic values are slightly high" -- k > 0.5).
  worker_warnings <- unique(unlist(lapply(worker_results, `[[`, "warnings")))
  worker_warnings <- worker_warnings[
    !grepl("Pareto k diagnostic", worker_warnings, fixed = TRUE)
  ]
  for (wm in worker_warnings) warning(wm, call. = FALSE)

  result
}


# =============================================================================
# Data Source Construction
# =============================================================================

#' Build data source from the stored model (training data)
#' @noRd
build_data_source_from_model <- function(model) {
  md <- model$model_data
  is_cumulative <- model$family$family == "cumulative"

  src <- list(
    N = md$N,
    is_old = md$stan_data$is_old,
    source = md$stan_data$source
  )

  # Bivariate/CDP specific data
  src$item_type <- md$stan_data$item_type
  src$y2 <- md$stan_data$y2
  src$rk <- md$stan_data$rk

  # Fixed effects design matrices
  dprime_name <- if (is_cumulative) "mu" else "dprime"
  if (isTRUE(md$has_dprime_fixed)) {
    src$X_dprime <- md$dprime_fixed$X
  } else {
    src$X_dprime <- NULL
  }

  src$X_criterion <- md$criterion$X

  param_names <- c("sigma", "lambda", "dprime2", "sigma2",
                    "dprime_B", "lambda_B",
                    "dprime_L", "sigma_L", "lambda_L",
                    "discrim", "discrim_B", "sigma_B", "sigma2_B",
                    "rho", "rho_B", "rho_N", "lambda2", "lambda2_B", "rec_crit", "know_crit")
  for (pname in param_names) {
    if (isTRUE(md[[paste0("has_", pname)]])) {
      src[[paste0("X_", pname)]] <- md[[paste0(pname, "_fixed")]]$X
    }
  }

  # Random effects structures (group indices, term_idx, Z matrices)
  all_re_params <- c("dprime", "sigma", "lambda", "dprime2", "sigma2",
                      "dprime_B", "lambda_B",
                      "dprime_L", "sigma_L", "lambda_L", "rec_crit", "know_crit",
                      "discrim", "discrim_B", "sigma_B", "sigma2_B",
                      "rho", "rho_B", "rho_N", "lambda2", "lambda2_B")
  for (pname in all_re_params) {
    re_list <- md[[paste0(pname, "_random")]]
    if (!is.null(re_list)) {
      src[[paste0(pname, "_random")]] <- re_list
    }
  }

  # Criterion random effects
  src$criterion_random <- md$criterion$random

  # Criterion2 (for bivariate_sdt, vrdp2d)
  if (isTRUE(md$has_criterion2) && !is.null(md$criterion2)) {
    src$X_criterion2 <- md$criterion2$X
    src$criterion2_random <- md$criterion2$random
  }

  # Cross-correlation info
  src$cross_cor <- md$cross_cor

  # Smooth term data (Zs matrices)
  src$smooth_data <- md$smooth_data

  src
}


# =============================================================================
# Variable Extraction
# =============================================================================

#' Get Stan variable names needed for prediction
#' @noRd
get_predict_variables <- function(model, include_re = TRUE) {
  vars <- character(0)
  md <- model$model_data
  is_cumulative <- model$family$family == "cumulative"
  is_bivariate_cumulative <- model$family$family == "bivariate_cumulative"
  dprime_name <- if (is_cumulative) "mu" else if (is_bivariate_cumulative) "mu1" else "dprime"
  criterion_name <- if (is_cumulative) "cutpoints" else if (is_bivariate_cumulative) "cutpoints1" else "criterion"
  K <- md$K
  is_cdp <- model$family$family == "cdp"
  gap_link <- if (!is.null(model$gap_link)) model$gap_link else "log"

  # Fixed effects: dprime/mu
  if (isTRUE(md$has_dprime_fixed)) {
    P <- md$dprime_fixed$n_coef
    if (P > 0) vars <- c(vars, paste0("beta_", dprime_name, "[", seq_len(P), "]"))
  }

  # Thresholds
  P_crit <- md$criterion$n_coef
  vars <- c(vars, paste0("beta_thresh_mid[", 1:P_crit, "]"))

  n_gaps <- if (is_cdp) model$family$J - 1 else K - 2
  if (n_gaps > 0) {
    # Use prefix-only so both 1D (JAX P_crit==1) and 2D (Stan) naming works
    vars <- c(vars, .gap_param(gap_link))
  }

  # Other parameters
  param_names <- c("sigma", "lambda", "dprime2", "sigma2",
                    "dprime_B", "lambda_B",
                    "dprime_L", "sigma_L", "lambda_L",
                    "discrim", "discrim_B", "sigma_B", "sigma2_B",
                    "rho", "rho_B", "rho_N", "lambda2", "lambda2_B", "rec_crit", "know_crit")
  for (pname in param_names) {
    has_flag <- if (pname %in% c("dprime2")) {
      md$has_dprime2 || isTRUE(md$needs_ordered_dprime)
    } else {
      isTRUE(md[[paste0("has_", pname)]])
    }
    if (has_flag) {
      P <- md[[paste0(pname, "_fixed")]]$n_coef
      if (P > 0) {
        stan_pname <- if (is_bivariate_cumulative && pname == "discrim") "mu2" else pname
        vars <- c(vars, paste0("beta_", stan_pname, "[", seq_len(P), "]"))
      }
    }
  }

  # Criterion2 thresholds (for bivariate_sdt, vrdp2d)
  if (isTRUE(md$has_criterion2)) {
    if (isTRUE(md$varying_source_criteria)) {
      # Varying: beta_thresh_mid_2_varying[1..n_bins], beta_log_gaps_2_varying[k, 1..n_bins]
      K2 <- md$K2
      n_bins <- K  # K is K1
      vars <- c(vars, paste0("beta_thresh_mid_2_varying[", 1:n_bins, "]"))
      if (K2 > 2) {
        vars <- c(vars, .gap_param(gap_link, "_2_varying"))
      }
      # new_source_criteria = "shared": separate thresholds for new items
      if (identical(md$new_source_criteria, "shared")) {
        vars <- c(vars, "beta_thresh_mid_2_new")
        if (K2 > 2) {
          vars <- c(vars, .gap_param(gap_link, "_2_new"))
        }
      }
    } else {
      # Non-varying: beta_thresh_mid_2[1..P_crit2], beta_log_gaps_2[k, 1..P_crit2]
      P_crit2 <- md$criterion2$n_coef
      K2 <- md$K2
      vars <- c(vars, paste0("beta_thresh_mid_2[", 1:P_crit2, "]"))
      if (K2 > 2) {
        vars <- c(vars, .gap_param(gap_link, "_2"))
      }
    }
  }

  # Random effects (transformed parameters u_*)
  if (include_re) {
    vars <- c(vars, get_re_variables(model))
  }

  # Smooth term coefficients (s_ = sds * zs, transformed parameters)
  if (!is.null(md$smooth_data)) {
    for (pname in names(md$smooth_data)) {
      for (sm in md$smooth_data[[pname]]) {
        n_thresh_sm <- if (!is.null(sm$n_thresh)) sm$n_thresh else 0L
        for (comp in sm$components) {
          for (k in seq_along(comp$Zs_list)) {
            nbasis <- ncol(comp$Zs_list[[k]])
            s_name <- paste0("s_", pname, "_", comp$san_label, "_", k)
            if (n_thresh_sm > 0) {
              # Per-threshold: s_[t,b] matrix
              for (t in seq_len(n_thresh_sm)) {
                vars <- c(vars, paste0(s_name, "[", t, ",", 1:nbasis, "]"))
              }
            } else {
              vars <- c(vars, paste0(s_name, "[", 1:nbasis, "]"))
            }
          }
        }
      }
    }
  }

  unique(vars)
}


#' Get RE variable names for prediction
#' @noRd
get_re_variables <- function(model) {
  vars <- character(0)
  md <- model$model_data
  is_cumulative <- model$family$family == "cumulative"
  is_bivariate_cumulative <- model$family$family == "bivariate_cumulative"
  criterion_name <- if (is_cumulative) "cutpoints" else if (is_bivariate_cumulative) "cutpoints1" else "criterion"

  # Regular parameter REs
  all_re_params <- c("dprime", "sigma", "lambda", "dprime2", "sigma2",
                      "dprime_B", "lambda_B",
                      "dprime_L", "sigma_L", "lambda_L", "rec_crit", "know_crit",
                      "discrim", "discrim_B", "sigma_B", "sigma2_B",
                      "rho", "rho_B", "rho_N", "lambda2", "lambda2_B")
  for (pname in all_re_params) {
    re_list <- md[[paste0(pname, "_random")]]
    if (is.null(re_list)) next
    stan_pname <- if (is_cumulative && pname == "dprime") "mu" else
      if (is_bivariate_cumulative && pname == "dprime") "mu1" else
      if (is_bivariate_cumulative && pname == "discrim") "mu2" else pname

    for (group in names(re_list)) {
      re <- re_list[[group]]
      var_base <- if (!is.null(re$cor_id)) {
        paste0("u_", stan_pname, "_", group, "_from_", re$cor_id)
      } else {
        paste0("u_", stan_pname, "_", group)
      }

      # Base name only; both backends expand it to all elements (spans 1D/2D).
      vars <- c(vars, var_base)
    }
  }

  # Criterion REs
  for (group in names(md$criterion$random)) {
    re <- md$criterion$random[[group]]
    var_base <- if (!is.null(re$cor_id)) {
      paste0("u_", criterion_name, "_", group, "_from_", re$cor_id)
    } else {
      paste0("u_", criterion_name, "_", group)
    }
    vars <- c(vars, var_base)
  }

  # Criterion2 REs (for bivariate_sdt, vrdp2d, bivariate_cumulative)
  crit2_stan <- if (is_bivariate_cumulative) "cutpoints2" else "criterion2"
  if (isTRUE(md$has_criterion2) && !is.null(md$criterion2$random)) {
    for (group in names(md$criterion2$random)) {
      re <- md$criterion2$random[[group]]
      var_base <- if (!is.null(re$cor_id)) {
        paste0("u_", crit2_stan, "_", group, "_from_", re$cor_id)
      } else {
        paste0("u_", crit2_stan, "_", group)
      }
      # Base name only; the backend expands it to all elements (see note above).
      vars <- c(vars, var_base)
    }
  }

  vars
}


# =============================================================================
# Trial Parameter Computation
# =============================================================================

#' Compute trial-level parameters for all draws
#' @return Named list of S x N matrices
#' @noRd
compute_trial_params <- function(draws, model, data_src, re_formula, S, N) {
  md <- model$model_data
  is_cumulative <- model$family$family == "cumulative"
  is_bivariate_cumulative <- model$family$family == "bivariate_cumulative"
  dprime_stan <- if (is_cumulative) "mu" else if (is_bivariate_cumulative) "mu1" else "dprime"
  include_re <- !identical(re_formula, NA)
  params <- list()

  # dprime / mu
  if (isTRUE(md$has_dprime_fixed)) {
    P <- md$dprime_fixed$n_coef
    if (P > 0) {
      beta <- draws[, paste0("beta_", dprime_stan, "[", seq_len(P), "]"), drop = FALSE]
      params$dprime <- data_src$X_dprime %*% t(beta)  # N x S
    } else {
      params$dprime <- matrix(0, nrow = nrow(data_src$X_dprime),
                              ncol = nrow(draws))
    }
  } else {
    params$dprime <- matrix(0, nrow = N, ncol = S)
  }

  if (include_re) {
    # Use RE structures from data_src (remapped for newdata) not md (training)
    re_list <- data_src$dprime_random
    if (!is.null(re_list)) {
      params$dprime <- params$dprime + compute_re_contribution(
        draws, re_list, data_src, "dprime", dprime_stan, S, N, md
      )
    }
  }

  # Smooth contribution for dprime
  params$dprime <- params$dprime + compute_smooth_contribution(
    draws, data_src$smooth_data, "dprime", S, N,
    allow_new = isTRUE(data_src$.allow_new_levels)
  )

  # Apply dprime link transform from family definition
  # (log link for bivariate_dp and bounded bivariate_sdt)
  # family$params is internal-keyed across all families, so direct lookup works.
  dprime_link <- model$family$params$dprime$link
  if (identical(dprime_link, "log")) {
    params$dprime <- exp(params$dprime)
  }

  # Helper: get link function from family param definition.
  .plink <- function(pname) {
    p <- model$family$params[[pname]]
    if (!is.null(p)) p$link else "identity"
  }

  # Map family-defined links to inverse link functions for prediction
  .inv_link <- function(pname) {
    lnk <- .plink(pname)
    switch(lnk,
      log = "exp", logit = "logis", logis = "logis",
      fisherz = "tanh", identity = "identity",
      "identity"  # fallback
    )
  }

  # Other parameters with link transforms from family definitions
  param_specs <- list(
    sigma   = list(has = md$has_sigma,   link = "exp"),
    lambda  = list(has = md$has_lambda,  link = "logis"),
    dprime2 = list(has = md$has_dprime2 || isTRUE(md$needs_ordered_dprime), link = "identity"),
    sigma2  = list(has = isTRUE(md$has_sigma2), link = "exp"),
    dprime_B = list(has = isTRUE(md$has_dprime_B), link = .inv_link("dprime_B")),
    lambda_B = list(has = isTRUE(md$has_lambda_B), link = "logis"),
    dprime_L = list(has = isTRUE(md$has_dprime_L), link = "identity"),
    sigma_L  = list(has = isTRUE(md$has_sigma_L), link = "exp"),
    lambda_L = list(has = isTRUE(md$has_lambda_L), link = "logis"),
    discrim    = list(has = isTRUE(md$has_discrim),    link = .inv_link("discrim"),
                     stan_name = if (is_bivariate_cumulative) "mu2" else NULL),
    discrim_B  = list(has = isTRUE(md$has_discrim_B),  link = .inv_link("discrim_B")),
    sigma_B    = list(has = isTRUE(md$has_sigma_B),    link = "exp"),
    sigma2_B   = list(has = isTRUE(md$has_sigma2_B),   link = "exp"),
    rho        = list(has = isTRUE(md$has_rho),        link = .inv_link("rho")),
    rho_B      = list(has = isTRUE(md$has_rho_B),      link = .inv_link("rho_B")),
    rho_N      = list(has = isTRUE(md$has_rho_N),      link = "tanh"),
    lambda2    = list(has = isTRUE(md$has_lambda2),    link = "logis"),
    rec_crit   = list(has = isTRUE(md$has_rec_crit),   link = "identity"),
    know_crit  = list(has = isTRUE(md$has_know_crit),  link = "identity")
  )

  for (pname in names(param_specs)) {
    spec <- param_specs[[pname]]
    if (!isTRUE(spec$has)) next

    stan_pname <- if (!is.null(spec$stan_name)) spec$stan_name else pname
    P <- md[[paste0(pname, "_fixed")]]$n_coef
    X <- data_src[[paste0("X_", pname)]]
    if (P > 0) {
      beta <- draws[, paste0("beta_", stan_pname, "[", seq_len(P), "]"), drop = FALSE]
      param_raw <- X %*% t(beta)  # N x S (linear predictor scale)
    } else {
      # No fixed effects (e.g. bivariate_cumulative mu1/mu2 lose intercept) --
      # everything comes from REs / smooths, start at 0.
      param_raw <- matrix(0, nrow = nrow(X), ncol = nrow(draws))
    }

    if (include_re) {
      # Use RE structures from data_src (remapped for newdata)
      re_list <- data_src[[paste0(pname, "_random")]]
      if (!is.null(re_list)) {
        param_raw <- param_raw + compute_re_contribution(
          draws, re_list, data_src, pname, stan_pname, S, N, md
        )
      }
    }

    # Smooth contribution
    param_raw <- param_raw + compute_smooth_contribution(
      draws, data_src$smooth_data, pname, S, N,
      allow_new = isTRUE(data_src$.allow_new_levels)
    )

    # Apply inverse link
    if (spec$link == "exp") {
      params[[pname]] <- exp(param_raw)
    } else if (spec$link == "logis") {
      params[[pname]] <- 1 / (1 + exp(-param_raw))
    } else if (spec$link == "tanh") {
      params[[pname]] <- tanh(param_raw)
    } else {
      params[[pname]] <- param_raw
    }
  }

  # Transpose everything to S x N for consistency
  for (nm in names(params)) {
    params[[nm]] <- t(params[[nm]])  # S x N
  }

  params
}


# =============================================================================
# Random Effects Contribution
# =============================================================================

#' Compute Smooth Contribution for One Parameter
#'
#' For each smooth component, extracts s_ draws (= sds * zs) and computes Zs %*% t(s).
#'
#' @param draws S x P matrix of posterior draws
#' @param smooth_data smooth_data list from model_data (or NULL)
#' @param pname Parameter name (e.g., "dprime")
#' @param S Number of draws
#' @param N Number of observations
#' @return N x S matrix of smooth contributions (zeros if no smooths)
#' @noRd
compute_smooth_contribution <- function(draws, smooth_data, pname, S, N,
                                         allow_new = FALSE) {
  if (is.null(smooth_data) || is.null(smooth_data[[pname]])) {
    return(matrix(0, nrow = N, ncol = S))
  }

  contrib <- matrix(0, nrow = N, ncol = S)

  for (sm in smooth_data[[pname]]) {
    for (comp in sm$components) {
      for (k in seq_along(comp$Zs_list)) {
        Zs <- comp$Zs_list[[k]]
        nbasis <- ncol(Zs)
        if (isTRUE(comp$new_level)) {
          # New by-level: sample coefficients from the prior (s = sds * z), using
          # the reference component's penalty SD as the wiggliness scale.
          if (!allow_new || nbasis == 0) next
          sds_name <- paste0("sds_", pname, "_", comp$san_label, "_", k)
          sds_col <- intersect(c(sds_name, paste0(sds_name, "[1]")), colnames(draws))
          if (length(sds_col) == 0) next
          sds_draws <- as.numeric(draws[, sds_col[1]])  # length S
          s_draws <- matrix(rnorm(S * nbasis), nrow = S, ncol = nbasis) * sds_draws
          contrib <- contrib + Zs %*% t(s_draws)
        } else {
          # s_ variables are the transformed coefficients (sds * zs)
          s_name <- paste0("s_", pname, "_", comp$san_label, "_", k)
          s_vars <- paste0(s_name, "[", 1:nbasis, "]")
          s_draws <- draws[, s_vars, drop = FALSE]  # S x nbasis
          contrib <- contrib + Zs %*% t(s_draws)    # N x S
        }
      }
    }
  }

  contrib
}


#' Compute Criterion Smooth Contribution to Thresholds
#'
#' For per-threshold criterion smooths, computes the smooth contribution
#' to the mid-anchor (modifies thresh_mid) and each gap (modifies gaps).
#'
#' @param draws S x P matrix of posterior draws
#' @param smooth_data smooth_data list from model_data
#' @param S Number of draws
#' @param N Number of observations
#' @param n_thresh Number of thresholds (K-1)
#' @return List with mid (N x S) and gaps (list of N x S matrices, one per gap)
#' @noRd
compute_criterion_smooth_contribution <- function(draws, smooth_data, S, N, n_thresh) {
  mid_contrib <- matrix(0, nrow = N, ncol = S)
  n_gaps <- n_thresh - 1
  gap_contribs <- lapply(seq_len(n_gaps), function(g) matrix(0, nrow = N, ncol = S))

  if (is.null(smooth_data) || is.null(smooth_data[["criterion"]])) {
    return(list(mid = mid_contrib, gaps = gap_contribs))
  }

  for (sm in smooth_data[["criterion"]]) {
    for (comp in sm$components) {
      if (isTRUE(comp$new_level)) next  # new by-level criterion smooths: zeroed
      for (k in seq_along(comp$Zs_list)) {
        Zs <- comp$Zs_list[[k]]
        nbasis <- ncol(Zs)
        s_name <- paste0("s_criterion_", comp$san_label, "_", k)

        # Row 1 = mid-anchor
        s_mid_vars <- paste0(s_name, "[1,", 1:nbasis, "]")
        s_mid_draws <- draws[, s_mid_vars, drop = FALSE]  # S x nbasis
        mid_contrib <- mid_contrib + Zs %*% t(s_mid_draws)

        # Rows 2..n_thresh = gaps
        for (g in seq_len(n_gaps)) {
          s_gap_vars <- paste0(s_name, "[", 1 + g, ",", 1:nbasis, "]")
          s_gap_draws <- draws[, s_gap_vars, drop = FALSE]
          gap_contribs[[g]] <- gap_contribs[[g]] + Zs %*% t(s_gap_draws)
        }
      }
    }
  }

  list(mid = mid_contrib, gaps = gap_contribs)
}


#' Compute RE contribution for one parameter across all groups
#' @return N x S matrix of RE contributions
#' @noRd
compute_re_contribution <- function(draws, re_list, data_src, pname, stan_pname, S, N, md) {
  contrib <- matrix(0, nrow = N, ncol = S)
  allow_new <- isTRUE(data_src$.allow_new_levels)

  for (group in names(re_list)) {
    re <- re_list[[group]]
    group_idx <- re$group_idx  # length N vector of group indices

    var_base <- if (!is.null(re$cor_id)) {
      paste0("u_", stan_pname, "_", group, "_from_", re$cor_id)
    } else {
      paste0("u_", stan_pname, "_", group)
    }

    n_groups <- re$n_groups
    new_mask <- group_idx == 0L
    n_new <- sum(new_mask)

    if (re$dim == 1 && is.null(re$term_idx)) {
      # Intercept-only: u[group_idx[n]] -- vectorized
      u_names <- paste0(var_base, "[", 1:n_groups, "]")
      if (!all(u_names %in% colnames(draws))) {
        u_names <- paste0(var_base, "[", 1:n_groups, ",1]")
      }
      u_draws <- draws[, u_names, drop = FALSE]  # S x n_groups
      valid <- group_idx > 0 & group_idx <= n_groups
      if (any(valid)) {
        contrib[valid, ] <- contrib[valid, ] + t(u_draws[, group_idx[valid], drop = FALSE])
      }
      if (allow_new && n_new > 0) {
        u_new <- sample_new_u_uncorrelated(draws, stan_pname, group, dim = 1L,
                                           n_new = n_new, S = S)
        # u_new is S x n_new; transpose to n_new x S
        contrib[new_mask, ] <- contrib[new_mask, ] + t(u_new)
      }
    } else if (!isTRUE(re$use_z_matrix) && !is.null(re$term_idx)) {
      # Index-based: u[group_idx[n], term_idx[n]] -- vectorized per term
      u_draws_list <- vector("list", re$dim)
      for (d in seq_len(re$dim)) {
        u_draws_list[[d]] <- draws[, paste0(var_base, "[", 1:n_groups, ",", d, "]"), drop = FALSE]
      }
      term_idx <- re$term_idx
      for (d in seq_len(re$dim)) {
        mask <- group_idx > 0 & group_idx <= n_groups & term_idx == d
        if (any(mask)) {
          contrib[mask, ] <- contrib[mask, ] + t(u_draws_list[[d]][, group_idx[mask], drop = FALSE])
        }
        if (allow_new) {
          new_term_mask <- new_mask & term_idx == d
          if (any(new_term_mask)) {
            u_new <- sample_new_u_uncorrelated(draws, stan_pname, group,
                                               dim = re$dim, term = d,
                                               n_new = sum(new_term_mask), S = S)
            contrib[new_term_mask, ] <- contrib[new_term_mask, ] + t(u_new)
          }
        }
      }
    } else if (isTRUE(re$use_z_matrix)) {
      # Z-matrix: Z[n,] %*% u[group_idx[n],] -- vectorized
      Z <- re$Z
      u_draws_all <- array(NA_real_, dim = c(S, n_groups, re$dim))
      for (d in seq_len(re$dim)) {
        u_draws_all[, , d] <- draws[, paste0(var_base, "[", 1:n_groups, ",", d, "]"), drop = FALSE]
      }
      valid <- group_idx > 0 & group_idx <= n_groups
      for (n in which(valid)) {
        g <- group_idx[n]
        z_n <- Z[n, ]
        u_g <- u_draws_all[, g, , drop = FALSE]
        dim(u_g) <- c(S, re$dim)
        contrib[n, ] <- contrib[n, ] + u_g %*% z_n
      }
      if (allow_new && n_new > 0) {
        # Z-matrix path: sample u_new ~ MVN(0, Sigma) per draw
        u_new_all <- sample_new_u_correlated(draws, stan_pname, group,
                                              dim = re$dim, n_new = n_new, S = S,
                                              correlated = isTRUE(re$correlated))
        # u_new_all is array(S, n_new, dim)
        for (i in seq_len(n_new)) {
          n_idx <- which(new_mask)[i]
          z_n <- Z[n_idx, ]
          u_i <- u_new_all[, i, , drop = FALSE]
          dim(u_i) <- c(S, re$dim)
          contrib[n_idx, ] <- contrib[n_idx, ] + u_i %*% z_n
        }
      }
    }
  }

  contrib
}


# Sample u ~ N(0, sigma_pname_group[term]) for n_new new levels, returning S x n_new.
# Errors clearly if sigma variable can't be found in draws.
#' @noRd
sample_new_u_uncorrelated <- function(draws, stan_pname, group, dim,
                                       term = 1L, n_new, S) {
  base <- paste0("sigma_", stan_pname, "_", group)
  candidate_names <- c(
    paste0(base, "[", term, "]"),
    if (term == 1L && dim == 1L) base else NULL
  )
  hit <- candidate_names[candidate_names %in% colnames(draws)][1]
  if (is.na(hit)) {
    stop("allow_new_levels: cannot find prior SD '", base,
         "' (or '[", term, "]' variant) in fit draws. ",
         "This RE structure may not be supported with allow_new_levels = TRUE.",
         call. = FALSE)
  }
  sigma_draws <- as.numeric(draws[, hit])  # length S
  # u ~ N(0, sigma_draws): S x n_new
  matrix(rnorm(S * n_new) * sigma_draws, nrow = S, ncol = n_new)
}

# Sample u ~ MVN(0, Sigma) for n_new new levels in correlated case.
# Returns array(S, n_new, dim). If !correlated, falls back to independent samples per term.
#' @noRd
sample_new_u_correlated <- function(draws, stan_pname, group, dim, n_new, S,
                                     correlated) {
  out <- array(NA_real_, dim = c(S, n_new, dim))

  if (!correlated) {
    for (d in seq_len(dim)) {
      u_d <- sample_new_u_uncorrelated(draws, stan_pname, group, dim = dim,
                                        term = d, n_new = n_new, S = S)
      out[, , d] <- u_d
    }
    return(out)
  }

  # Correlated: need sigma vector + L_corr (Cholesky) per draw
  sigma_names <- paste0("sigma_", stan_pname, "_", group, "[", seq_len(dim), "]")
  if (!all(sigma_names %in% colnames(draws))) {
    stop("allow_new_levels: cannot find sigma vector for correlated RE '",
         stan_pname, "_", group, "'. ",
         "Per-bin / smooth REs are not yet supported with allow_new_levels = TRUE.",
         call. = FALSE)
  }
  sigma_mat <- as.matrix(draws[, sigma_names, drop = FALSE])  # S x dim (plain)

  L_prefix <- paste0("L_corr_", stan_pname, "_", group)
  L_names <- paste0(L_prefix, "[", rep(seq_len(dim), each = dim), ",",
                    rep(seq_len(dim), times = dim), "]")
  if (!all(L_names %in% colnames(draws))) {
    stop("allow_new_levels: cannot find Cholesky factor '", L_prefix,
         "[i,j]' in fit draws. RE may not be sampled with a Cholesky parameterization.",
         call. = FALSE)
  }
  L_arr <- array(as.numeric(draws[, L_names, drop = FALSE]),
                 dim = c(S, dim, dim))

  for (s in seq_len(S)) {
    sigma_s <- as.numeric(sigma_mat[s, ])
    # L_names are enumerated row-major but the array fills column-major, so
    # L_arr[s,,] is the transpose of the Cholesky factor; transpose back so that
    # L_s L_s' is the correlation matrix.
    L_s <- t(matrix(as.numeric(L_arr[s, , ]), nrow = dim, ncol = dim))
    Sigma_chol <- diag(sigma_s, nrow = dim) %*% L_s
    z <- matrix(rnorm(n_new * dim), nrow = n_new, ncol = dim)
    out[s, , ] <- z %*% t(Sigma_chol)
  }
  out
}


# =============================================================================
# Threshold Computation
# =============================================================================

#' Compute thresholds: S x N x (K-1) array
#' @noRd
compute_thresholds <- function(draws, model, data_src, re_formula, S, N, K) {
  md <- model$model_data
  is_cumulative <- model$family$family == "cumulative"
  is_bivariate_cumulative <- model$family$family == "bivariate_cumulative"
  criterion_name <- if (is_cumulative) "cutpoints" else if (is_bivariate_cumulative) "cutpoints1" else "criterion"
  include_re <- !identical(re_formula, NA)
  gap_link <- if (!is.null(model$gap_link)) model$gap_link else "log"
  n_thresh <- K - 1
  P_crit <- md$criterion$n_coef
  X_crit <- data_src$X_criterion

  mid <- ceiling(n_thresh / 2)
  n_upper <- n_thresh - mid

  # Use criterion RE from data_src (remapped for newdata) not md (training)
  crit_random <- data_src$criterion_random

  # Pre-sample new-level criterion REs once (coherent across thresholds);
  # reused by every compute_criterion_re() call below.
  crit_new_u <- if (include_re && !is.null(crit_random) &&
                    isTRUE(data_src$.allow_new_levels)) {
    sample_criterion_new_u(draws, crit_random, criterion_name, S, md)
  } else NULL

  # Build S x N x (K-1) array
  thresh <- array(NA_real_, dim = c(S, N, n_thresh))

    # Detect 1D vs 2D gap naming
    gap_prefix <- .gap_param(gap_link)
    use_1d <- .gap_names_1d(draws, gap_prefix)

    # === Mid-anchor threshold ===
    beta_mid <- draws[, paste0("beta_thresh_mid[", 1:P_crit, "]"), drop = FALSE]
    thresh_mid_val <- X_crit %*% t(beta_mid)

    if (include_re && !is.null(crit_random)) {
      thresh_mid_val <- thresh_mid_val + compute_criterion_re(
        draws, crit_random, data_src, criterion_name, S, N,
        thresh_k = mid, md = md, new_u_cache = crit_new_u
      )
    }

    # Criterion smooth contribution (per-threshold)
    crit_sm <- compute_criterion_smooth_contribution(
      draws, data_src$smooth_data, S, N, n_thresh
    )
    thresh_mid_val <- thresh_mid_val + crit_sm$mid

    thresh[, , mid] <- t(thresh_mid_val)

    # === Upper gaps: positions mid+1 to n_thresh ===
    if (n_upper > 0) {
      for (k in (mid + 1):n_thresh) {
        gap_row <- k - mid
        beta_gaps_k <- draws[, .gap_cols(gap_prefix, gap_row, P_crit, use_1d), drop = FALSE]
        log_gap <- X_crit %*% t(beta_gaps_k)

        if (include_re && !is.null(crit_random)) {
          log_gap <- log_gap + compute_criterion_re(
            draws, crit_random, data_src, criterion_name, S, N,
            thresh_k = k, md = md, new_u_cache = crit_new_u
          )
        }

        # Criterion smooth per-gap contribution
        if (gap_row <= length(crit_sm$gaps)) {
          log_gap <- log_gap + crit_sm$gaps[[gap_row]]
        }

        thresh[, , k] <- thresh[, , k - 1] + .gap_transform(t(log_gap), gap_link)
      }
    }

    # === Lower gaps: positions mid-1 down to 1 ===
    if (mid > 1) {
      for (k_down in seq_len(mid - 1)) {
        k <- mid - k_down
        gap_row <- n_upper + k_down
        beta_gaps_k <- draws[, .gap_cols(gap_prefix, gap_row, P_crit, use_1d), drop = FALSE]
        log_gap <- X_crit %*% t(beta_gaps_k)

        if (include_re && !is.null(crit_random)) {
          log_gap <- log_gap + compute_criterion_re(
            draws, crit_random, data_src, criterion_name, S, N,
            thresh_k = k, md = md, new_u_cache = crit_new_u
          )
        }

        # Criterion smooth per-gap contribution
        if (gap_row <= length(crit_sm$gaps)) {
          log_gap <- log_gap + crit_sm$gaps[[gap_row]]
        }

        thresh[, , k] <- thresh[, , k + 1] - .gap_transform(t(log_gap), gap_link)
      }
    }

  thresh
}


#' Compute criterion RE contribution for a specific threshold
#' @return N x S matrix
#' @noRd
# `new_u_cache` (built once per call to compute_thresholds) holds, per group, an
# S x n_new x dim array of new-level RE draws so a new observation's threshold
# deviations stay coherent across thresholds.
compute_criterion_re <- function(draws, crit_random, data_src, criterion_name, S, N,
                                  thresh_k, md, new_u_cache = NULL) {
  contrib <- matrix(0, nrow = N, ncol = S)

  for (group in names(crit_random)) {
    re <- crit_random[[group]]
    group_idx <- re$group_idx
    n_groups <- re$n_groups
    new_pos <- integer(N)
    new_pos[group_idx == 0L] <- seq_len(sum(group_idx == 0L))
    u_new <- if (!is.null(new_u_cache)) new_u_cache[[group]] else NULL  # S x n_new x dim

    var_base <- if (!is.null(re$cor_id)) {
      paste0("u_", criterion_name, "_", group, "_from_", re$cor_id)
    } else {
      paste0("u_", criterion_name, "_", group)
    }

    n_re_terms <- if (!is.null(re$n_re_terms)) re$n_re_terms else {
      if (!is.null(re$n_cond_levels)) re$n_cond_levels else 1
    }

    if (isTRUE(re$use_z_matrix)) {
      # Z-matrix based
      Z <- re$Z
      # Column range: ((thresh_k-1)*n_re_terms + 1) to (thresh_k*n_re_terms)
      col_start <- (thresh_k - 1) * n_re_terms + 1
      col_end <- thresh_k * n_re_terms

      u_draws_all <- array(NA_real_, dim = c(S, n_groups, re$dim))
      for (d in seq_len(re$dim)) {
        u_draws_all[, , d] <- draws[, paste0(var_base, "[", 1:n_groups, ",", d, "]"), drop = FALSE]
      }

      for (n in seq_len(N)) {
        g <- group_idx[n]
        if (g > 0 && g <= n_groups) {
          z_n <- Z[n, ]
          u_g <- u_draws_all[, g, col_start:col_end, drop = FALSE]
          dim(u_g) <- c(S, n_re_terms)
          contrib[n, ] <- contrib[n, ] + u_g %*% z_n
        } else if (!is.null(u_new) && new_pos[n] > 0) {
          z_n <- Z[n, ]
          u_g <- u_new[, new_pos[n], col_start:col_end, drop = FALSE]
          dim(u_g) <- c(S, n_re_terms)
          contrib[n, ] <- contrib[n, ] + u_g %*% z_n
        }
      }
    } else if (n_re_terms == 1) {
      # Intercept-only: column = thresh_k
      u_draws <- matrix(NA_real_, nrow = S, ncol = n_groups)
      for (g in seq_len(n_groups)) {
        u_draws[, g] <- draws[, paste0(var_base, "[", g, ",", thresh_k, "]")]
      }
      for (n in seq_len(N)) {
        g <- group_idx[n]
        if (g > 0 && g <= n_groups) {
          contrib[n, ] <- contrib[n, ] + u_draws[, g]
        } else if (!is.null(u_new) && new_pos[n] > 0) {
          contrib[n, ] <- contrib[n, ] + u_new[, new_pos[n], thresh_k]
        }
      }
    } else if (!is.null(re$term_idx)) {
      # Index-based: column = (thresh_k - 1) * n_re_terms + term_idx[n]
      term_idx <- re$term_idx
      u_draws <- array(NA_real_, dim = c(S, n_groups, re$dim))
      for (d in seq_len(re$dim)) {
        u_draws[, , d] <- draws[, paste0(var_base, "[", 1:n_groups, ",", d, "]"), drop = FALSE]
      }
      for (n in seq_len(N)) {
        g <- group_idx[n]
        tidx <- term_idx[n]
        col <- (thresh_k - 1) * n_re_terms + tidx
        if (tidx > 0 && col <= re$dim) {
          if (g > 0 && g <= n_groups) {
            contrib[n, ] <- contrib[n, ] + u_draws[, g, col]
          } else if (!is.null(u_new) && new_pos[n] > 0) {
            contrib[n, ] <- contrib[n, ] + u_new[, new_pos[n], col]
          }
        }
      }
    }
  }

  contrib
}

# Pre-sample new-level (group_idx == 0) RE draws for each criterion RE group,
# once, as an S x n_new x dim array per group. NULL if no new levels / not allowed.
sample_criterion_new_u <- function(draws, crit_random, criterion_name, S, md) {
  cache <- list()
  for (group in names(crit_random)) {
    re <- crit_random[[group]]
    n_new <- sum(re$group_idx == 0L)
    if (n_new == 0) next
    sig0 <- paste0("sigma_", criterion_name, "_", group)
    sig_names <- c(sig0, paste0(sig0, "[", seq_len(max(re$dim, 1L)), "]"))
    if (!any(sig_names %in% colnames(draws))) {
      warning("allow_new_levels: prior SD for criterion RE '", group,
              "' not found in draws; new levels zeroed for this group.",
              call. = FALSE)
      next
    }
    use_corr <- isTRUE(md$cor_threshold) && isTRUE(re$correlated) && re$dim > 1
    cache[[group]] <- sample_new_u_correlated(
      draws, criterion_name, group, dim = re$dim, n_new = n_new, S = S,
      correlated = use_corr)
  }
  if (length(cache)) cache else NULL
}


# =============================================================================
# Criterion2 Threshold Computation (bivariate_sdt, vrdp2d)
# =============================================================================

#' Compute second-dimension thresholds: S x N x (K2-1) array
#'
#' For non-varying models: uses beta_thresh_mid_2, beta_log_gaps_2, X_criterion2.
#' For varying models: returns S x N x K1 x (K2-1) array indexed by item response.
#'
#' @return For non-varying: S x N x (K2-1) array.
#'   For varying: S x N x K1 x (K2-1) array.
#' @noRd
compute_thresholds_2 <- function(draws, model, data_src, re_formula, S, N) {
  md <- model$model_data
  K2 <- md$K2
  K1 <- md$K  # K is K1 for bivariate models
  include_re <- !identical(re_formula, NA)
  varying <- isTRUE(md$varying_source_criteria)
  gap_link <- if (!is.null(model$gap_link)) model$gap_link else "log"
  is_bivariate_cumulative <- model$family$family == "bivariate_cumulative"
  criterion2_stan_name <- if (is_bivariate_cumulative) "cutpoints2" else "criterion2"

  n_thresh2 <- K2 - 1
  mid2 <- ceiling(n_thresh2 / 2)
  n_upper2 <- n_thresh2 - mid2

  # Pre-sample new-level criterion2 REs once (coherent across thresholds/bins).
  crit2_random <- data_src$criterion2_random
  crit2_new_u <- if (include_re && !is.null(crit2_random) &&
                     isTRUE(data_src$.allow_new_levels)) {
    sample_criterion_new_u(draws, crit2_random, criterion2_stan_name, S, md)
  } else NULL

  if (varying) {
    # Varying source criteria: one set of thresholds per item response level
    n_bins <- K1

    # Extract varying mid-anchor threshold parameters
    beta_mid_v <- draws[, paste0("beta_thresh_mid_2_varying[", 1:n_bins, "]"), drop = FALSE]  # S x n_bins

    # Build S x N x K1 x (K2-1) array
    thresh2 <- array(0, dim = c(S, N, K1, n_thresh2))

    for (b in seq_len(n_bins)) {
      bin_idx <- b

      # RE contribution for mid-anchor threshold of this bin
      re_shift <- matrix(0, nrow = S, ncol = N)
      if (include_re && !is.null(data_src$criterion2_random)) {
        for (group in names(data_src$criterion2_random)) {
          re <- data_src$criterion2_random[[group]]
          group_idx <- re$group_idx
          n_groups <- re$n_groups
          new_pos <- integer(N); new_pos[group_idx == 0L] <- seq_len(sum(group_idx == 0L))
          u_new <- if (!is.null(crit2_new_u)) crit2_new_u[[group]] else NULL
          varying_re_mode <- if (!is.null(re$varying_re_mode)) re$varying_re_mode else "shared"
          var_base <- if (!is.null(re$cor_id)) {
            paste0("u_criterion2_", group, "_from_", re$cor_id)
          } else {
            paste0("u_criterion2_", group)
          }

          if (varying_re_mode == "shared") {
            if (re$dim == 1) {
              u_draws <- draws[, paste0(var_base, "[", 1:n_groups, "]"), drop = FALSE]
            } else {
              u_draws <- draws[, paste0(var_base, "[", 1:n_groups, ",1]"), drop = FALSE]
            }
            for (n in seq_len(N)) {
              g <- group_idx[n]
              if (g > 0 && g <= n_groups) {
                re_shift[, n] <- re_shift[, n] + u_draws[, g]
              } else if (!is.null(u_new) && new_pos[n] > 0) {
                re_shift[, n] <- re_shift[, n] + u_new[, new_pos[n], 1]
              }
            }
          } else if (varying_re_mode == "per_bin") {
            if (re$dim == 1) {
              u_draws <- draws[, paste0(var_base, "[", 1:n_groups, "]"), drop = FALSE]
            } else {
              u_draws <- draws[, paste0(var_base, "[", 1:n_groups, ",", b, "]"), drop = FALSE]
            }
            for (n in seq_len(N)) {
              g <- group_idx[n]
              if (g > 0 && g <= n_groups) {
                re_shift[, n] <- re_shift[, n] + u_draws[, g]
              } else if (!is.null(u_new) && new_pos[n] > 0) {
                re_shift[, n] <- re_shift[, n] + u_new[, new_pos[n], b]
              }
            }
          } else if (varying_re_mode == "full") {
            col_mid <- (b - 1) * n_thresh2 + mid2
            u_draws <- draws[, paste0(var_base, "[", 1:n_groups, ",", col_mid, "]"), drop = FALSE]
            for (n in seq_len(N)) {
              g <- group_idx[n]
              if (g > 0 && g <= n_groups) {
                re_shift[, n] <- re_shift[, n] + u_draws[, g]
              } else if (!is.null(u_new) && new_pos[n] > 0) {
                re_shift[, n] <- re_shift[, n] + u_new[, new_pos[n], col_mid]
              }
            }
          }
        }
      }

      # Mid-anchor threshold for this bin
      thresh2[, , bin_idx, mid2] <- matrix(beta_mid_v[, b], nrow = S, ncol = N) + re_shift

      # Upper gaps: mid2+1 to n_thresh2
      if (n_upper2 > 0) {
        for (k2 in (mid2 + 1):n_thresh2) {
          gap_row <- k2 - mid2
          beta_gap <- draws[, paste0(.gap_param(gap_link, "_2_varying"), "[", b, ",", gap_row, "]")]  # S vector

          re_gap_shift <- matrix(0, nrow = S, ncol = N)
          if (include_re && !is.null(data_src$criterion2_random)) {
            for (group in names(data_src$criterion2_random)) {
              re <- data_src$criterion2_random[[group]]
              varying_re_mode <- if (!is.null(re$varying_re_mode)) re$varying_re_mode else "shared"
              if (varying_re_mode == "full") {
                group_idx <- re$group_idx
                n_groups <- re$n_groups
                new_pos <- integer(N); new_pos[group_idx == 0L] <- seq_len(sum(group_idx == 0L))
                u_new <- if (!is.null(crit2_new_u)) crit2_new_u[[group]] else NULL
                var_base <- if (!is.null(re$cor_id)) {
                  paste0("u_criterion2_", group, "_from_", re$cor_id)
                } else {
                  paste0("u_criterion2_", group)
                }
                col_idx <- (b - 1) * n_thresh2 + k2
                u_draws <- draws[, paste0(var_base, "[", 1:n_groups, ",", col_idx, "]"), drop = FALSE]
                for (n in seq_len(N)) {
                  g <- group_idx[n]
                  if (g > 0 && g <= n_groups) {
                    re_gap_shift[, n] <- re_gap_shift[, n] + u_draws[, g]
                  } else if (!is.null(u_new) && new_pos[n] > 0) {
                    re_gap_shift[, n] <- re_gap_shift[, n] + u_new[, new_pos[n], col_idx]
                  }
                }
              }
            }
          }

          thresh2[, , bin_idx, k2] <- thresh2[, , bin_idx, k2 - 1] +
            .gap_transform(matrix(beta_gap, nrow = S, ncol = N) + re_gap_shift)
        }
      }

      # Lower gaps: mid2-1 down to 1
      if (mid2 > 1) {
        for (k2_down in seq_len(mid2 - 1)) {
          k2 <- mid2 - k2_down
          gap_row <- n_upper2 + k2_down
          beta_gap <- draws[, paste0(.gap_param(gap_link, "_2_varying"), "[", b, ",", gap_row, "]")]  # S vector

          re_gap_shift <- matrix(0, nrow = S, ncol = N)
          if (include_re && !is.null(data_src$criterion2_random)) {
            for (group in names(data_src$criterion2_random)) {
              re <- data_src$criterion2_random[[group]]
              varying_re_mode <- if (!is.null(re$varying_re_mode)) re$varying_re_mode else "shared"
              if (varying_re_mode == "full") {
                group_idx <- re$group_idx
                n_groups <- re$n_groups
                new_pos <- integer(N); new_pos[group_idx == 0L] <- seq_len(sum(group_idx == 0L))
                u_new <- if (!is.null(crit2_new_u)) crit2_new_u[[group]] else NULL
                var_base <- if (!is.null(re$cor_id)) {
                  paste0("u_criterion2_", group, "_from_", re$cor_id)
                } else {
                  paste0("u_criterion2_", group)
                }
                col_idx <- (b - 1) * n_thresh2 + k2
                u_draws <- draws[, paste0(var_base, "[", 1:n_groups, ",", col_idx, "]"), drop = FALSE]
                for (n in seq_len(N)) {
                  g <- group_idx[n]
                  if (g > 0 && g <= n_groups) {
                    re_gap_shift[, n] <- re_gap_shift[, n] + u_draws[, g]
                  } else if (!is.null(u_new) && new_pos[n] > 0) {
                    re_gap_shift[, n] <- re_gap_shift[, n] + u_new[, new_pos[n], col_idx]
                  }
                }
              }
            }
          }

          thresh2[, , bin_idx, k2] <- thresh2[, , bin_idx, k2 + 1] -
            .gap_transform(matrix(beta_gap, nrow = S, ncol = N) + re_gap_shift)
        }
      }
    }

    # Compute shared new-item thresholds if new_source_criteria = "shared" (mid-anchor)
    if (identical(md$new_source_criteria, "shared")) {
      beta_mid_new <- draws[, "beta_thresh_mid_2_new"]  # S vector
      thresh2_new <- array(0, dim = c(S, n_thresh2))
      thresh2_new[, mid2] <- beta_mid_new
      if (n_upper2 > 0) {
        for (k2 in (mid2 + 1):n_thresh2) {
          gap_row <- k2 - mid2
          beta_gap_new <- draws[, paste0(.gap_param(gap_link, "_2_new"), "[", gap_row, "]")]
          thresh2_new[, k2] <- thresh2_new[, k2 - 1] + .gap_transform(beta_gap_new, gap_link)
        }
      }
      if (mid2 > 1) {
        for (k2_down in seq_len(mid2 - 1)) {
          k2 <- mid2 - k2_down
          gap_row <- n_upper2 + k2_down
          beta_gap_new <- draws[, paste0(.gap_param(gap_link, "_2_new"), "[", gap_row, "]")]
          thresh2_new[, k2] <- thresh2_new[, k2 + 1] - .gap_transform(beta_gap_new, gap_link)
        }
      }
      attr(thresh2, "thresh2_new") <- thresh2_new
    }

    return(thresh2)

  } else {
    # Non-varying: standard criterion2 thresholds using X_criterion2 (mid-anchor)
    P_crit2 <- md$criterion2$n_coef
    X_crit2 <- data_src$X_criterion2

    crit2_random <- data_src$criterion2_random

    thresh2 <- array(NA_real_, dim = c(S, N, n_thresh2))

    # Detect 1D vs 2D gap naming for dimension 2
    gap_prefix_2 <- .gap_param(gap_link, "_2")
    use_1d_2 <- .gap_names_1d(draws, gap_prefix_2)

    # Mid-anchor threshold
    beta_mid_2 <- draws[, paste0("beta_thresh_mid_2[", 1:P_crit2, "]"), drop = FALSE]  # S x P
    thresh_mid_val <- X_crit2 %*% t(beta_mid_2)  # N x S

    if (include_re && !is.null(crit2_random)) {
      thresh_mid_val <- thresh_mid_val + compute_criterion_re(
        draws, crit2_random, data_src, criterion2_stan_name, S, N,
        thresh_k = mid2, md = md, new_u_cache = crit2_new_u
      )
    }
    thresh2[, , mid2] <- t(thresh_mid_val)

    # Upper gaps: mid2+1 to n_thresh2
    if (n_upper2 > 0) {
      for (k in (mid2 + 1):n_thresh2) {
        gap_row <- k - mid2
        beta_gaps_k <- draws[, .gap_cols(gap_prefix_2, gap_row, P_crit2, use_1d_2), drop = FALSE]
        log_gap <- X_crit2 %*% t(beta_gaps_k)  # N x S

        if (include_re && !is.null(crit2_random)) {
          log_gap <- log_gap + compute_criterion_re(
            draws, crit2_random, data_src, criterion2_stan_name, S, N,
            thresh_k = k, md = md, new_u_cache = crit2_new_u
          )
        }

        thresh2[, , k] <- thresh2[, , k - 1] + .gap_transform(t(log_gap), gap_link)
      }
    }

    # Lower gaps: mid2-1 down to 1
    if (mid2 > 1) {
      for (k_down in seq_len(mid2 - 1)) {
        k <- mid2 - k_down
        gap_row <- n_upper2 + k_down
        beta_gaps_k <- draws[, .gap_cols(gap_prefix_2, gap_row, P_crit2, use_1d_2), drop = FALSE]
        log_gap <- X_crit2 %*% t(beta_gaps_k)  # N x S

        if (include_re && !is.null(crit2_random)) {
          log_gap <- log_gap + compute_criterion_re(
            draws, crit2_random, data_src, criterion2_stan_name, S, N,
            thresh_k = k, md = md, new_u_cache = crit2_new_u
          )
        }

        thresh2[, , k] <- thresh2[, , k + 1] - .gap_transform(t(log_gap), gap_link)
      }
    }

    return(thresh2)
  }
}


# =============================================================================
# Family-Specific Probability Functions
# =============================================================================

#' Compute category probabilities (dispatcher)
#' @return S x N x K array (or S x N x K1 x K2 for bivariate families)
#' @noRd
compute_category_probs <- function(trial_params, thresh, data_src, model, link_name, S, N, K,
                                    thresh2 = NULL, cores = 1) {
  family_name <- model$family$family
  K2 <- model$model_data$K2
  is_bivariate <- family_name %in% c("bivariate_sdt", "bivariate_dp", "vrdp2d", "bivariate_cumulative")

  # Helper: dispatch to family-specific function for observation indices
  compute_chunk <- function(idx) {
    n_chunk <- length(idx)
    # Slice trial_params (each is S x N matrix -- slice columns for observations)
    tp <- lapply(trial_params, function(m) m[, idx, drop = FALSE])
    # Slice thresh (S x N x (K-1)) -> S x n_chunk x (K-1)
    th <- thresh[, idx, , drop = FALSE]
    th2 <- if (!is.null(thresh2)) {
      if (length(dim(thresh2)) == 4) {
        thresh2[, idx, , , drop = FALSE]
      } else {
        thresh2[, idx, , drop = FALSE]
      }
    }
    # Copy thresh2 attributes
    if (!is.null(th2) && !is.null(attr(thresh2, "thresh2_new"))) {
      attr(th2, "thresh2_new") <- attr(thresh2, "thresh2_new")
    }
    # Slice data_src vectors
    ds <- list(
      is_old = if (!is.null(data_src$is_old)) data_src$is_old[idx] else NULL,
      source = if (!is.null(data_src$source)) data_src$source[idx] else NULL,
      item_type = if (!is.null(data_src$item_type)) data_src$item_type[idx] else NULL
    )

    switch(family_name,
      evsdt          = probs_evsdt(tp, th, ds$is_old, link_name, S, n_chunk, K),
      uvsdt          = probs_uvsdt(tp, th, ds$is_old, link_name, S, n_chunk, K),
      dpsdt          = probs_dpsdt(tp, th, ds$is_old, link_name, S, n_chunk, K,
                                    has_sigma = isTRUE(model$model_data$has_sigma)),
      mixture        = probs_mixture(tp, th, ds$is_old, link_name, S, n_chunk, K, model = model),
      cumulative     = probs_cumulative(tp, th, link_name, S, n_chunk, K),
      source_mixture = probs_source_mixture(tp, th, ds$source, link_name, S, n_chunk, K, model = model),
      bivariate_sdt  = probs_bivariate_sdt(tp, th, th2, ds$item_type, S, n_chunk, K, K2, model = model),
      bivariate_dp   = probs_bivariate_dp(tp, th, th2, ds$item_type, S, n_chunk, K, K2, model = model),
      bivariate_cumulative = probs_bivariate_cumulative(tp, th, th2, S, n_chunk, K, K2, model = model),
      vrdp2d         = probs_vrdp2d(tp, th, th2, ds$item_type, S, n_chunk, K, K2, model = model),
      cdp            = probs_cdp(tp, th, ds$is_old, S, n_chunk, K, model = model)
    )
  }

  # Single-core: compute directly. cores > 1 is honored even for small N
  # (PSOCK startup may outweigh the speedup, but cores is never silently ignored).
  if (cores <= 1 || N < 2) {
    return(compute_chunk(seq_len(N)))
  }

  # PSOCK parallel: split observations into chunks
  n_workers <- min(cores, N)
  chunks <- split(seq_len(N), rep(seq_len(n_workers), length.out = N))

  # Build chunk data list -- slice all data by observation index
  # trial_params: named list of S x N matrices (S=draws, N=observations)
  # thresh: S x N x (K-1) array
  chunk_data <- lapply(chunks, function(idx) {
    list(
      idx = idx,
      trial_params = lapply(trial_params, function(m) m[, idx, drop = FALSE]),
      thresh = thresh[, idx, , drop = FALSE],
      thresh2 = if (!is.null(thresh2)) {
        t2 <- if (length(dim(thresh2)) == 4) thresh2[, idx, , , drop = FALSE] else thresh2[, idx, , drop = FALSE]
        if (!is.null(attr(thresh2, "thresh2_new"))) attr(t2, "thresh2_new") <- attr(thresh2, "thresh2_new")
        t2
      },
      is_old = if (!is.null(data_src$is_old)) data_src$is_old[idx] else NULL,
      source = if (!is.null(data_src$source)) data_src$source[idx] else NULL,
      item_type = if (!is.null(data_src$item_type)) data_src$item_type[idx] else NULL
    )
  })

  # Pack everything workers need into each chunk (self-contained, no closure capture)
  for (i in seq_along(chunk_data)) {
    chunk_data[[i]]$family_name <- family_name
    chunk_data[[i]]$link_name <- link_name
    chunk_data[[i]]$S <- S
    chunk_data[[i]]$K <- K
    chunk_data[[i]]$K2 <- K2
    chunk_data[[i]]$model <- model
  }

  # Remove large objects from this scope so they're not serialized via closure
  rm(trial_params, thresh, thresh2, data_src)

  cl <- parallel::makeCluster(n_workers)
  on.exit(parallel::stopCluster(cl))

  # Only export the specific functions workers need (not entire package env)
  pkg_env <- environment(compute_category_probs)
  worker_fns <- c("probs_evsdt", "probs_uvsdt", "probs_dpsdt", "probs_mixture",
                   "probs_cumulative", "probs_source_mixture", "probs_bivariate_sdt",
                   "probs_bivariate_dp", "probs_bivariate_cumulative", "probs_vrdp2d",
                   "probs_cdp", "link_cdf_r", "binormal_cdf_r", "owens_t_r",
                   "owens_t_r_core_vec", "compute_bivariate_prob_r",
                   "compute_bounded_prob_r", "compute_bounded_marginal_source_r",
                   "univariate_cell_prob_r", "univariate_cell_prob_from_z",
                   "compute_rkg_probs_r", "binormal_strip_upper_r",
                   "binormal_strip_lower_r", "binormal_cdf_r_scalar",
                   "G_cdp_r", "owens_t_r",
                   ".gl20_nodes", ".gl20_weights")
  # Filter to only functions that exist (some may not be defined)
  worker_fns <- worker_fns[worker_fns %in% ls(pkg_env, all.names = TRUE)]
  parallel::clusterExport(cl, worker_fns, envir = pkg_env)

  # Self-contained worker -- uses only cd contents + exported functions
  .predict_worker <- function(cd) {
    n_chunk <- length(cd$idx)
    ds <- list(is_old = cd$is_old, source = cd$source, item_type = cd$item_type)
    switch(cd$family_name,
      evsdt          = probs_evsdt(cd$trial_params, cd$thresh, ds$is_old, cd$link_name, cd$S, n_chunk, cd$K),
      uvsdt          = probs_uvsdt(cd$trial_params, cd$thresh, ds$is_old, cd$link_name, cd$S, n_chunk, cd$K),
      dpsdt          = probs_dpsdt(cd$trial_params, cd$thresh, ds$is_old, cd$link_name, cd$S, n_chunk, cd$K,
                                    has_sigma = isTRUE(cd$model$model_data$has_sigma)),
      mixture        = probs_mixture(cd$trial_params, cd$thresh, ds$is_old, cd$link_name, cd$S, n_chunk, cd$K,
                                      model = cd$model),
      cumulative     = probs_cumulative(cd$trial_params, cd$thresh, cd$link_name, cd$S, n_chunk, cd$K),
      source_mixture = probs_source_mixture(cd$trial_params, cd$thresh, ds$source, cd$link_name, cd$S, n_chunk, cd$K,
                                             model = cd$model),
      bivariate_sdt  = probs_bivariate_sdt(cd$trial_params, cd$thresh, cd$thresh2, ds$item_type,
                                            cd$S, n_chunk, cd$K, cd$K2, model = cd$model),
      bivariate_dp   = probs_bivariate_dp(cd$trial_params, cd$thresh, cd$thresh2, ds$item_type,
                                            cd$S, n_chunk, cd$K, cd$K2, model = cd$model),
      bivariate_cumulative = probs_bivariate_cumulative(cd$trial_params, cd$thresh, cd$thresh2,
                                            cd$S, n_chunk, cd$K, cd$K2, model = cd$model),
      vrdp2d         = probs_vrdp2d(cd$trial_params, cd$thresh, cd$thresh2, ds$item_type,
                                     cd$S, n_chunk, cd$K, cd$K2, model = cd$model),
      cdp            = probs_cdp(cd$trial_params, cd$thresh, ds$is_old, cd$S, n_chunk, cd$K, model = cd$model)
    )
  }
  environment(.predict_worker) <- globalenv()

  chunk_results <- parallel::clusterApply(cl, chunk_data, .predict_worker)

  # Reassemble: concatenate along the N (observation) dimension
  if (is_bivariate) {
    probs <- array(0, dim = c(S, N, K, K2))
    for (i in seq_along(chunks)) {
      probs[, chunks[[i]], , ] <- chunk_results[[i]]
    }
  } else {
    probs <- array(0, dim = c(S, N, K))
    for (i in seq_along(chunks)) {
      probs[, chunks[[i]], ] <- chunk_results[[i]]
    }
  }

  # Reassemble CDP attributes (p_remember, p_know) from chunk results
  if (family_name == "cdp" && !is.null(attr(chunk_results[[1]], "p_remember"))) {
    p_rem <- array(0, dim = c(S, N, K))
    for (i in seq_along(chunks)) {
      p_rem[, chunks[[i]], ] <- attr(chunk_results[[i]], "p_remember")
    }
    attr(probs, "p_remember") <- p_rem

    if (!is.null(attr(chunk_results[[1]], "p_know"))) {
      p_kn <- array(0, dim = c(S, N, K))
      for (i in seq_along(chunks)) {
        p_kn[, chunks[[i]], ] <- attr(chunk_results[[i]], "p_know")
      }
      attr(probs, "p_know") <- p_kn
    }
  }

  probs
}


#' EVSDT probabilities
#' @noRd
probs_evsdt <- function(params, thresh, is_old, link_name, S, N, K) {
  dprime <- params$dprime  # S x N
  probs <- array(0, dim = c(S, N, K))

  # mu = is_old * dprime (works for both treatment 0/1 and centered -0.5/0.5)
  mu <- sweep(dprime, 2, is_old, "*")

  # P(Y=1) = F(thresh[1] - mu)
  probs[, , 1] <- link_cdf_r(thresh[, , 1] - mu, link_name)

  # P(Y=k) = F(thresh[k] - mu) - F(thresh[k-1] - mu)
  if (K > 2) {
    for (k in 2:(K - 1)) {
      probs[, , k] <- link_cdf_r(thresh[, , k] - mu, link_name) -
                       link_cdf_r(thresh[, , k - 1] - mu, link_name)
    }
  }

  # P(Y=K) = 1 - F(thresh[K-1] - mu)
  probs[, , K] <- 1 - link_cdf_r(thresh[, , K - 1] - mu, link_name)

  # Clamp to avoid numerical issues
  probs <- pmax(probs, 1e-10)
  probs
}


#' UVSDT probabilities
#' @noRd
probs_uvsdt <- function(params, thresh, is_old, link_name, S, N, K) {
  dprime <- params$dprime  # S x N
  sigma <- params$sigma    # S x N
  probs <- array(0, dim = c(S, N, K))

  # mu = is_old * dprime; s = sigma for old, 1 for new/noise
  mu <- sweep(dprime, 2, is_old, "*")
  s <- sigma
  for (n in seq_len(N)) {
    if (is_old[n] <= 0) {
      s[, n] <- 1
    }
  }

  probs[, , 1] <- link_cdf_r((thresh[, , 1] - mu) / s, link_name)

  if (K > 2) {
    for (k in 2:(K - 1)) {
      probs[, , k] <- link_cdf_r((thresh[, , k] - mu) / s, link_name) -
                       link_cdf_r((thresh[, , k - 1] - mu) / s, link_name)
    }
  }

  probs[, , K] <- 1 - link_cdf_r((thresh[, , K - 1] - mu) / s, link_name)
  probs <- pmax(probs, 1e-10)
  probs
}


#' DPSDT probabilities
#' @noRd
probs_dpsdt <- function(params, thresh, is_old, link_name, S, N, K, has_sigma) {
  dprime <- params$dprime
  lambda <- params$lambda
  sigma <- if (has_sigma) params$sigma else matrix(1, nrow = nrow(dprime), ncol = ncol(dprime))
  probs <- array(0, dim = c(S, N, K))

  for (n in seq_len(N)) {
    if (is_old[n] > 0) {
      # Old items: recollection + familiarity mixture
      d <- is_old[n] * dprime[, n]
      lam <- lambda[, n]
      s <- sigma[, n]

      # Familiarity-based probabilities (UVSDT)
      p_fam <- numeric(K)
      for (k in seq_len(K)) {
        if (k == 1) {
          p_fam_k <- link_cdf_r((thresh[, n, 1] - d) / s, link_name)
        } else if (k == K) {
          p_fam_k <- 1 - link_cdf_r((thresh[, n, K - 1] - d) / s, link_name)
        } else {
          p_fam_k <- link_cdf_r((thresh[, n, k] - d) / s, link_name) -
                     link_cdf_r((thresh[, n, k - 1] - d) / s, link_name)
        }

        if (k == K) {
          # P(Y=K|old) = lambda + (1-lambda)*P_fam(Y=K)
          probs[, n, k] <- lam + (1 - lam) * p_fam_k
        } else {
          # P(Y=k|old) = (1-lambda)*P_fam(Y=k)
          probs[, n, k] <- (1 - lam) * p_fam_k
        }
      }
    } else {
      # New items: EVSDT with mu = is_old * dprime (0 for treatment, negative for centered)
      mu_noise <- is_old[n] * dprime[, n]
      probs[, n, 1] <- link_cdf_r(thresh[, n, 1] - mu_noise, link_name)
      if (K > 2) {
        for (k in 2:(K - 1)) {
          probs[, n, k] <- link_cdf_r(thresh[, n, k] - mu_noise, link_name) -
                           link_cdf_r(thresh[, n, k - 1] - mu_noise, link_name)
        }
      }
      probs[, n, K] <- 1 - link_cdf_r(thresh[, n, K - 1] - mu_noise, link_name)
    }
  }

  probs <- pmax(probs, 1e-10)
  probs
}


#' Mixture SDT probabilities
#' @noRd
probs_mixture <- function(params, thresh, is_old, link_name, S, N, K, model) {
  md <- model$model_data
  dprime <- params$dprime
  lambda <- params$lambda
  sigma <- if (!is.null(params$sigma)) params$sigma else matrix(1, S, N)
  dprime2 <- if (!is.null(params$dprime2)) params$dprime2 else matrix(0, S, N)
  sigma2 <- if (!is.null(params$sigma2)) params$sigma2 else matrix(1, S, N)

  has_lure_mixture <- isTRUE(md$has_lure_mixture)
  dprime_L <- if (has_lure_mixture && !is.null(params$dprime_L)) params$dprime_L else NULL
  sigma_L  <- if (has_lure_mixture && !is.null(params$sigma_L)) params$sigma_L else NULL
  lambda_L <- if (has_lure_mixture && !is.null(params$lambda_L)) params$lambda_L else NULL

  probs <- array(0, dim = c(S, N, K))

  for (n in seq_len(N)) {
    if (is_old[n] > 0) {
      # Old: lambda * P(mu1,s1) + (1-lambda) * P(mu2,s2)
      mu1 <- is_old[n] * dprime[, n]
      mu2 <- dprime2[, n] - (1 - is_old[n]) * dprime[, n]
      s1 <- sigma[, n]; lam <- lambda[, n]
      s2 <- sigma2[, n]

      for (k in seq_len(K)) {
        if (k == 1) {
          p1 <- link_cdf_r((thresh[, n, 1] - mu1) / s1, link_name)
          p2 <- link_cdf_r((thresh[, n, 1] - mu2) / s2, link_name)
        } else if (k == K) {
          p1 <- 1 - link_cdf_r((thresh[, n, K - 1] - mu1) / s1, link_name)
          p2 <- 1 - link_cdf_r((thresh[, n, K - 1] - mu2) / s2, link_name)
        } else {
          p1 <- link_cdf_r((thresh[, n, k] - mu1) / s1, link_name) -
                link_cdf_r((thresh[, n, k - 1] - mu1) / s1, link_name)
          p2 <- link_cdf_r((thresh[, n, k] - mu2) / s2, link_name) -
                link_cdf_r((thresh[, n, k - 1] - mu2) / s2, link_name)
        }
        probs[, n, k] <- lam * p1 + (1 - lam) * p2
      }
    } else {
      # New items: EVSDT (or lure mixture)
      if (has_lure_mixture && !is.null(dprime_L) && !is.null(lambda_L)) {
        # Lure mixture: lambda_L * P(-dprime_L, sigma_L) + (1-lambda_L) * P(0,1)
        dL <- dprime_L[, n]
        sL <- if (!is.null(sigma_L)) sigma_L[, n] else rep(1, S)
        lamL <- lambda_L[, n]

        for (k in seq_len(K)) {
          if (k == 1) {
            p_lure <- link_cdf_r((thresh[, n, 1] + dL) / sL, link_name)
            p_ref  <- link_cdf_r(thresh[, n, 1], link_name)
          } else if (k == K) {
            p_lure <- 1 - link_cdf_r((thresh[, n, K - 1] + dL) / sL, link_name)
            p_ref  <- 1 - link_cdf_r(thresh[, n, K - 1], link_name)
          } else {
            p_lure <- link_cdf_r((thresh[, n, k] + dL) / sL, link_name) -
                      link_cdf_r((thresh[, n, k - 1] + dL) / sL, link_name)
            p_ref  <- link_cdf_r(thresh[, n, k], link_name) -
                      link_cdf_r(thresh[, n, k - 1], link_name)
          }
          probs[, n, k] <- lamL * p_lure + (1 - lamL) * p_ref
        }
      } else {
        # Standard EVSDT for new items with noise mean shift
        mu_noise <- is_old[n] * dprime[, n]
        probs[, n, 1] <- link_cdf_r(thresh[, n, 1] - mu_noise, link_name)
        if (K > 2) {
          for (k in 2:(K - 1)) {
            probs[, n, k] <- link_cdf_r(thresh[, n, k] - mu_noise, link_name) -
                             link_cdf_r(thresh[, n, k - 1] - mu_noise, link_name)
          }
        }
        probs[, n, K] <- 1 - link_cdf_r(thresh[, n, K - 1] - mu_noise, link_name)
      }
    }
  }

  probs <- pmax(probs, 1e-10)
  probs
}


#' Cumulative ordinal probabilities
#' @noRd
probs_cumulative <- function(params, thresh, link_name, S, N, K) {
  mu <- params$dprime  # S x N (stored as dprime internally)
  probs <- array(0, dim = c(S, N, K))

  probs[, , 1] <- link_cdf_r(thresh[, , 1] - mu, link_name)

  if (K > 2) {
    for (k in 2:(K - 1)) {
      probs[, , k] <- link_cdf_r(thresh[, , k] - mu, link_name) -
                       link_cdf_r(thresh[, , k - 1] - mu, link_name)
    }
  }

  probs[, , K] <- 1 - link_cdf_r(thresh[, , K - 1] - mu, link_name)
  probs <- pmax(probs, 1e-10)
  probs
}


#' Source mixture SDT probabilities
#' @noRd
probs_source_mixture <- function(params, thresh, source, link_name, S, N, K, model) {
  md <- model$model_data
  dprime <- params$dprime
  lambda <- params$lambda
  has_dprime_B <- isTRUE(md$has_dprime_B)
  has_lambda_B <- isTRUE(md$has_lambda_B)

  probs <- array(0, dim = c(S, N, K))

  for (n in seq_len(N)) {
    src <- source[n]  # 0 = A, 1 = B

    # Select d and lambda for this source
    if (src == 0) {
      d <- dprime[, n]
      lam <- lambda[, n]
    } else {
      d <- if (has_dprime_B) params$dprime_B[, n] else -dprime[, n]
      lam <- if (has_lambda_B) params$lambda_B[, n] else lambda[, n]
    }

    for (k in seq_len(K)) {
      if (k == 1) {
        p_att <- link_cdf_r(thresh[, n, 1] - d, link_name)
        p_non <- link_cdf_r(thresh[, n, 1], link_name)
      } else if (k == K) {
        p_att <- 1 - link_cdf_r(thresh[, n, K - 1] - d, link_name)
        p_non <- 1 - link_cdf_r(thresh[, n, K - 1], link_name)
      } else {
        p_att <- link_cdf_r(thresh[, n, k] - d, link_name) -
                 link_cdf_r(thresh[, n, k - 1] - d, link_name)
        p_non <- link_cdf_r(thresh[, n, k], link_name) -
                 link_cdf_r(thresh[, n, k - 1], link_name)
      }
      probs[, n, k] <- lam * p_att + (1 - lam) * p_non
    }
  }

  probs <- pmax(probs, 1e-10)
  probs
}


#' VRDP2D probabilities (2D Variable Recollection Dual-Process)
#'
#' Returns S x N x K1 x K2 array of joint response probabilities.
#' Since rho=0 for all VRDP2D distributions, bivariate normal factors into
#' products of univariates, making this the simplest bivariate family.
#' @noRd
probs_vrdp2d <- function(params, thresh, thresh2, item_type, S, N, K1, K2, model) {
  md <- model$model_data
  dprime <- params$dprime       # S x N (d'_F: familiarity strength)
  dprime2 <- params$dprime2     # S x N (d'_R: recollection boost)
  discrim <- params$discrim     # S x N (d'_S: source discriminability for A)
  lambda <- params$lambda       # S x N (R: recollection probability)
  sigma <- if (!is.null(params$sigma)) params$sigma else matrix(1, S, N)   # sigma_item
  sigma2 <- if (!is.null(params$sigma2)) params$sigma2 else matrix(1, S, N) # sigma_S

  has_discrim_B <- isTRUE(md$has_discrim_B)
  discrim_B <- if (has_discrim_B) params$discrim_B else NULL

  varying <- isTRUE(md$varying_source_criteria)

  probs <- array(0, dim = c(S, N, K1, K2))

  # Vectorized helper: univariate cell prob for S-length vectors
  univar_cell_vec <- function(j, K, mu_s, sig_s, thresh_sn) {
    # thresh_sn is S x (K-1)
    if (j == 1) {
      pnorm((thresh_sn[, 1] - mu_s) / sig_s)
    } else if (j == K) {
      1 - pnorm((thresh_sn[, K - 1] - mu_s) / sig_s)
    } else {
      pnorm((thresh_sn[, j] - mu_s) / sig_s) - pnorm((thresh_sn[, j - 1] - mu_s) / sig_s)
    }
  }

  for (n in seq_len(N)) {
    it <- item_type[n]

    # thresh for this obs: S x (K1-1)
    c1_sn <- thresh[, n, , drop = FALSE]
    dim(c1_sn) <- c(S, K1 - 1)

    if (it == 1) {
      mu1_s <- rep(0, S); sig1_s <- rep(1, S)
      mu2_s <- rep(0, S); sig2_s <- rep(1, S)

      for (j in seq_len(K1)) {
        p1_s <- univar_cell_vec(j, K1, mu1_s, sig1_s, c1_sn)
        c2_sn <- if (varying) {
          m <- thresh2[, n, j, , drop = FALSE]; dim(m) <- c(S, K2 - 1); m
        } else {
          m <- thresh2[, n, , drop = FALSE]; dim(m) <- c(S, K2 - 1); m
        }
        for (k in seq_len(K2)) {
          p2_s <- univar_cell_vec(k, K2, mu2_s, sig2_s, c2_sn)
          probs[, n, j, k] <- p1_s * p2_s
        }
      }
    } else {
      d_F_s <- dprime[, n]
      d_R_s <- dprime2[, n]
      R_s <- lambda[, n]
      s_item_s <- sigma[, n]
      s_S_s <- sigma2[, n]
      source_d_s <- if (it == 2) {
        discrim[, n]
      } else {
        if (has_discrim_B) discrim_B[, n] else -discrim[, n]
      }

      for (j in seq_len(K1)) {
        p1_fam_s <- univar_cell_vec(j, K1, d_F_s, s_item_s, c1_sn)
        p1_rec_s <- univar_cell_vec(j, K1, d_F_s + d_R_s, s_item_s, c1_sn)

        c2_sn <- if (varying) {
          m <- thresh2[, n, j, , drop = FALSE]; dim(m) <- c(S, K2 - 1); m
        } else {
          m <- thresh2[, n, , drop = FALSE]; dim(m) <- c(S, K2 - 1); m
        }

        for (k in seq_len(K2)) {
          p2_fam_s <- univar_cell_vec(k, K2, rep(0, S), rep(1, S), c2_sn)
          p2_rec_s <- univar_cell_vec(k, K2, source_d_s, s_S_s, c2_sn)

          probs[, n, j, k] <- (1 - R_s) * p1_fam_s * p2_fam_s +
                                R_s * p1_rec_s * p2_rec_s
        }
      }
    }
  }

  probs <- pmax(probs, 1e-10)
  probs
}


#' Bivariate Cumulative probabilities
#' @noRd
probs_bivariate_cumulative <- function(params, thresh, thresh2, S, N, K1, K2, model) {
  dprime <- params$dprime
  discrim <- params$discrim
  rho_vals <- params$rho

  probs <- array(0, dim = c(S, N, K1, K2))

  for (n in seq_len(N)) {
    mu1_s <- if (!is.null(dprime)) dprime[, n] else rep(0, S)
    mu2_s <- if (!is.null(discrim)) discrim[, n] else rep(0, S)
    rho_s <- if (!is.null(rho_vals)) rho_vals[, n] else rep(0, S)

    for (j in seq_len(K1)) {
      c1_lo_z <- if (j > 1) thresh[, n, j - 1] - mu1_s else rep(-Inf, S)
      c1_hi_z <- if (j < K1) thresh[, n, j] - mu1_s else rep(Inf, S)

      for (k in seq_len(K2)) {
        c2_lo_z <- if (k > 1) thresh2[, n, k - 1] - mu2_s else rep(-Inf, S)
        c2_hi_z <- if (k < K2) thresh2[, n, k] - mu2_s else rep(Inf, S)

        probs[, n, j, k] <- binormal_cdf_r(c1_hi_z, c2_hi_z, rho_s) -
                             binormal_cdf_r(c1_hi_z, c2_lo_z, rho_s) -
                             binormal_cdf_r(c1_lo_z, c2_hi_z, rho_s) +
                             binormal_cdf_r(c1_lo_z, c2_lo_z, rho_s)
      }
    }
  }

  pmax(probs, 1e-10)
}


#' Bivariate SDT probabilities
#'
#' Returns S x N x K1 x K2 array of joint response probabilities.
#' Uses bivariate normal CDF for correlated dimensions.
#' @noRd
probs_bivariate_sdt <- function(params, thresh, thresh2, item_type, S, N, K1, K2, model) {
  md <- model$model_data
  dprime <- params$dprime           # S x N
  discrim <- params$discrim         # S x N
  rho_A <- params$rho              # S x N

  has_dprime_B <- isTRUE(md$has_dprime_B)
  has_discrim_B <- isTRUE(md$has_discrim_B)
  has_sigma <- !is.null(params$sigma)
  has_sigma_B <- !is.null(params$sigma_B)
  has_sigma2 <- !is.null(params$sigma2)
  has_sigma2_B <- !is.null(params$sigma2_B)
  has_rho_B <- !is.null(params$rho_B)
  has_rho_N <- !is.null(params$rho_N)

  varying <- isTRUE(md$varying_source_criteria)
  bounded <- isTRUE(md$bounded)
  new_shared <- identical(md$new_source_criteria, "shared")
  thresh2_new <- attr(thresh2, "thresh2_new")  # S x (K2-1) or NULL
  is_new_response <- if (new_shared) {
    !(seq_len(K1) %in% md$old_levels)
  } else {
    rep(FALSE, K1)
  }

  prob_fn <- if (bounded) compute_bounded_prob_r else compute_bivariate_prob_r
  probs <- array(0, dim = c(S, N, K1, K2))

  for (n in seq_len(N)) {
    it <- item_type[n]

    # Get S-length vectors of parameters for this observation
    if (it == 1) {
      mu1_s <- rep(0, S); mu2_s <- rep(0, S)
      sig1_s <- rep(1, S); sig2_s <- rep(1, S)
      rho_s <- if (has_rho_N) params$rho_N[, n] else rep(0, S)
    } else if (it == 2) {
      mu1_s <- dprime[, n]
      mu2_s <- discrim[, n]
      sig1_s <- if (has_sigma) params$sigma[, n] else rep(1, S)
      sig2_s <- if (has_sigma2) params$sigma2[, n] else rep(1, S)
      rho_s <- rho_A[, n]
    } else {
      mu1_s <- if (has_dprime_B) params$dprime_B[, n] else dprime[, n]
      mu2_s <- if (has_discrim_B) -params$discrim_B[, n] else -discrim[, n]
      sig1_s <- if (has_sigma_B) params$sigma_B[, n] else {
        if (has_sigma) params$sigma[, n] else rep(1, S)
      }
      sig2_s <- if (has_sigma2_B) params$sigma2_B[, n] else {
        if (has_sigma2) params$sigma2[, n] else rep(1, S)
      }
      rho_s <- if (has_rho_B) -params$rho_B[, n] else -rho_A[, n]
    }

    use_bounded <- bounded && it != 1

    # Bounded model: the likelihood places Source A on the negative source axis
    # (mu2 = -discrim, rho = -rho_A) and Source B on the positive axis, i.e. the
    # sign-flip of the unbounded convention for both sources. Mirror it here.
    if (use_bounded) {
      mu2_s <- -mu2_s
      rho_s <- -rho_s
    }

    for (j in seq_len(K1)) {
      # Standardized dim1 thresholds: S-length vectors
      c1_lo_z <- if (j > 1) (thresh[, n, j - 1] - mu1_s) / sig1_s else rep(-Inf, S)
      c1_hi_z <- if (j < K1) (thresh[, n, j] - mu1_s) / sig1_s else rep(Inf, S)

      # Source thresholds for this detection level
      for (k in seq_len(K2)) {
        if (new_shared && is_new_response[j] && !is.null(thresh2_new)) {
          c2_lo_z <- if (k > 1) (thresh2_new[, k - 1] - mu2_s) / sig2_s else rep(-Inf, S)
          c2_hi_z <- if (k < K2) (thresh2_new[, k] - mu2_s) / sig2_s else rep(Inf, S)
        } else if (varying) {
          c2_lo_z <- if (k > 1) (thresh2[, n, j, k - 1] - mu2_s) / sig2_s else rep(-Inf, S)
          c2_hi_z <- if (k < K2) (thresh2[, n, j, k] - mu2_s) / sig2_s else rep(Inf, S)
        } else {
          c2_lo_z <- if (k > 1) (thresh2[, n, k - 1] - mu2_s) / sig2_s else rep(-Inf, S)
          c2_hi_z <- if (k < K2) (thresh2[, n, k] - mu2_s) / sig2_s else rep(Inf, S)
        }

        if (use_bounded) {
          # Bounded model: per-draw scalar computation (can't vectorize due to conditional clamping)
          for (s in seq_len(S)) {
            c1_raw <- thresh[s, n, ]
            if (new_shared && is_new_response[j] && !is.null(thresh2_new)) {
              c2_raw <- thresh2_new[s, ]
            } else if (varying) {
              c2_raw <- thresh2[s, n, j, ]
            } else {
              c2_raw <- thresh2[s, n, ]
            }
            probs[s, n, j, k] <- compute_bounded_prob_r(
              j, k, K1, K2, mu1_s[s], mu2_s[s], sig1_s[s], sig2_s[s],
              c1_raw, c2_raw, rho_s[s]
            )
          }
        } else {
          # Standard: vectorized bivariate rectangle probability
          p <- binormal_cdf_r(c1_hi_z, c2_hi_z, rho_s) -
               binormal_cdf_r(c1_hi_z, c2_lo_z, rho_s) -
               binormal_cdf_r(c1_lo_z, c2_hi_z, rho_s) +
               binormal_cdf_r(c1_lo_z, c2_lo_z, rho_s)
          probs[, n, j, k] <- p
        }
      }
    }
  }

  probs <- pmax(probs, 1e-10)
  probs
}


#' Bivariate Dual-Process probabilities
#'
#' Implements p1 (familiarity) + p2 (item recollection) + p3 (both recollected)
#' with optional bounding.
#' @noRd
probs_bivariate_dp <- function(params, thresh, thresh2, item_type, S, N, K1, K2, model) {
  md <- model$model_data
  dprime <- params$dprime
  discrim <- params$discrim
  rho_A <- params$rho
  lambda <- params$lambda    # R_I (already logistic-transformed)
  lambda2 <- params$lambda2  # R_S (already logistic-transformed)

  has_dprime_B <- isTRUE(md$has_dprime_B)
  has_discrim_B <- isTRUE(md$has_discrim_B)
  has_sigma <- !is.null(params$sigma)
  has_sigma_B <- !is.null(params$sigma_B)
  has_sigma2 <- !is.null(params$sigma2)
  has_sigma2_B <- !is.null(params$sigma2_B)
  has_rho_B <- !is.null(params$rho_B)
  has_rho_N <- !is.null(params$rho_N)

  varying <- isTRUE(md$varying_source_criteria)
  bounded <- isTRUE(md$bounded)
  new_shared <- identical(md$new_source_criteria, "shared")
  thresh2_new <- attr(thresh2, "thresh2_new")
  # Build is_new_response lookup: TRUE for detection responses judged as "new"
  is_new_response <- if (new_shared) {
    !(seq_len(K1) %in% md$old_levels)
  } else {
    rep(FALSE, K1)
  }

  probs <- array(0, dim = c(S, N, K1, K2))

  for (n in seq_len(N)) {
    it <- item_type[n]

    if (it == 1) {
      # New items: standard bivariate, no recollection
      rho_s <- if (has_rho_N) params$rho_N[, n] else rep(0, S)
      mu1_s <- rep(0, S); mu2_s <- rep(0, S); sig1_s <- rep(1, S); sig2_s <- rep(1, S)
    } else if (it == 2) {
      mu1_s <- dprime[, n]
      mu2_s <- discrim[, n]
      sig1_s <- if (has_sigma) params$sigma[, n] else rep(1, S)
      sig2_s <- if (has_sigma2) params$sigma2[, n] else rep(1, S)
      rho_s <- rho_A[, n]
    } else {
      mu1_s <- if (has_dprime_B) params$dprime_B[, n] else dprime[, n]
      mu2_s <- if (has_discrim_B) -params$discrim_B[, n] else -discrim[, n]
      sig1_s <- if (has_sigma_B) params$sigma_B[, n] else {
        if (has_sigma) params$sigma[, n] else rep(1, S)
      }
      sig2_s <- if (has_sigma2_B) params$sigma2_B[, n] else {
        if (has_sigma2) params$sigma2[, n] else rep(1, S)
      }
      rho_s <- if (has_rho_B) -params$rho_B[, n] else -rho_A[, n]
    }

    # Bounded model: Source A on the negative source axis (mu2 = -discrim,
    # rho = -rho_A), i.e. the sign-flip of the unbounded convention for both
    # sources (matches the bounded bivariate likelihood cell).
    if (bounded && it != 1) {
      mu2_s <- -mu2_s
      rho_s <- -rho_s
    }

    R_I_s <- if (it != 1) lambda[, n] else NULL
    R_S_s <- if (it != 1) lambda2[, n] else NULL

    for (j in seq_len(K1)) {
      c1_lo_z <- if (j > 1) (thresh[, n, j - 1] - mu1_s) / sig1_s else rep(-Inf, S)
      c1_hi_z <- if (j < K1) (thresh[, n, j] - mu1_s) / sig1_s else rep(Inf, S)

      # Source thresholds for this detection level
      if (new_shared && is_new_response[j] && !is.null(thresh2_new)) {
        c2_raw <- thresh2_new  # S x (K2-1)
      } else if (varying) {
        c2_raw <- thresh2[, n, j, , drop = FALSE]
        dim(c2_raw) <- c(S, K2 - 1)
      } else {
        c2_raw <- thresh2[, n, , drop = FALSE]
        dim(c2_raw) <- c(S, K2 - 1)
      }

      for (k in seq_len(K2)) {
        c2_lo_z <- if (k > 1) (c2_raw[, k - 1] - mu2_s) / sig2_s else rep(-Inf, S)
        c2_hi_z <- if (k < K2) (c2_raw[, k] - mu2_s) / sig2_s else rep(Inf, S)

        if (it == 1) {
          # New items: standard bivariate familiarity (rho_N), never bounded
          probs[, n, j, k] <-
            binormal_cdf_r(c1_hi_z, c2_hi_z, rho_s) -
            binormal_cdf_r(c1_hi_z, c2_lo_z, rho_s) -
            binormal_cdf_r(c1_lo_z, c2_hi_z, rho_s) +
            binormal_cdf_r(c1_lo_z, c2_lo_z, rho_s)
          next
        }

        # Recollection corner: Source A items recollect to the low source end
        # (y2 = 1, "sure A"), Source B to the high end (y2 = K2), matching the
        # Stan/JAX/batch likelihood kernels.
        is_corner <- (it == 2 && k == 1) || (it == 3 && k == K2)

        if (bounded) {
          # Bounded (BBDP): conditional source-mean clamping can't be vectorized,
          # so compute p1 (bounded bivariate) and p2 (bounded marginal source)
          # per draw, mirroring bounded_bivariate_dp_cell in the likelihood.
          for (s in seq_len(S)) {
            c1_raw_s <- thresh[s, n, ]
            c2_raw_s <- c2_raw[s, ]
            p1s <- (1 - R_I_s[s]) * compute_bounded_prob_r(
              j, k, K1, K2, mu1_s[s], mu2_s[s], sig1_s[s], sig2_s[s],
              c1_raw_s, c2_raw_s, rho_s[s])
            p2s <- if (j == K1) R_I_s[s] * (1 - R_S_s[s]) *
              compute_bounded_marginal_source_r(k, K2, mu1_s[s], mu2_s[s],
                sig1_s[s], sig2_s[s], rho_s[s], c2_raw_s) else 0
            p3s <- if (j == K1 && is_corner) R_I_s[s] * R_S_s[s] else 0
            probs[s, n, j, k] <- p1s + p2s + p3s
          }
        } else {
          # Unbounded: vectorized over draws.
          p1 <- (1 - R_I_s) * (
            binormal_cdf_r(c1_hi_z, c2_hi_z, rho_s) -
            binormal_cdf_r(c1_hi_z, c2_lo_z, rho_s) -
            binormal_cdf_r(c1_lo_z, c2_hi_z, rho_s) +
            binormal_cdf_r(c1_lo_z, c2_lo_z, rho_s))
          # p2: item recollected (j==K1), source from the familiarity marginal
          p2 <- if (j == K1)
            R_I_s * (1 - R_S_s) * (pnorm(c2_hi_z) - pnorm(c2_lo_z)) else rep(0, S)
          p3 <- if (j == K1 && is_corner) R_I_s * R_S_s else rep(0, S)
          probs[, n, j, k] <- p1 + p2 + p3
        }
      }
    }
  }

  probs <- pmax(probs, 1e-10)
  probs
}


#' CDP (Continuous Dual-Process) probabilities
#'
#' Returns S x N x K array of marginal confidence probabilities (marginalizing over R/K or R/K/G).
#' Also stores R/K(/G) conditional probabilities as attributes for prediction sampling.
#' @noRd
probs_cdp <- function(params, thresh, is_old, S, N, K, model) {
  md <- model$model_data
  family <- model$family

  old_levels <- family$old_levels
  J <- family$J
  old_level_map <- old_levels  # Maps 1:J to actual confidence levels
  new_levels <- setdiff(1:K, old_levels)
  n_rkg <- if (!is.null(family$n_rkg)) family$n_rkg else 2L

  probs <- array(0, dim = c(S, N, K))
  p_remember <- array(0, dim = c(S, N, K))  # P(Remember | Y=k)
  p_know <- if (n_rkg == 3) array(0, dim = c(S, N, K)) else NULL  # P(Know | Y=k) for RKG

  for (n in seq_len(N)) {
    for (s in seq_len(S)) {
      # Get CDP parameters
      mu_R <- params$dprime[s, n]  # rec -> dprime mapping
      mu_F <- if (!is.null(params$dprime2)) params$dprime2[s, n] else 0
      sigma_R <- if (!is.null(params$sigma)) params$sigma[s, n] else 1
      sigma_F <- if (!is.null(params$sigma2)) params$sigma2[s, n] else 1
      c_R <- if (!is.null(params$rec_crit)) params$rec_crit[s, n] else 1.5
      c_K <- if (!is.null(params$know_crit)) params$know_crit[s, n] else 1.0

      # Derived CDP parameters
      mu_M <- mu_R + mu_F
      sigma_M <- sqrt(sigma_R^2 + sigma_F^2)
      rho <- sigma_R / sigma_M

      # Lure parameters (fixed reference)
      sigma_M_l <- sqrt(2)
      rho_l <- 1 / sqrt(2)

      # Get tau thresholds (J thresholds for old levels)
      tau <- thresh[s, n, ]  # length J (from CDP threshold construction)

      if (is_old[n] == 1) {
        # Target trial
        z_cR <- (c_R - mu_R) / sigma_R

        for (k in seq_len(K)) {
          old_idx <- match(k, old_level_map)

          if (!is.na(old_idx)) {
            tau_lo <- tau[old_idx]
            tau_hi <- if (old_idx == J) 20.0 else tau[old_idx + 1]

            if (n_rkg == 3) {
              # R/K/G mode: use compute_rkg_probs_r
              rkg <- compute_rkg_probs_r(mu_R, sigma_R, mu_F, sigma_F, c_R, c_K, tau_lo, tau_hi)
              probs[s, n, k] <- sum(rkg)
              total <- sum(rkg)
              p_remember[s, n, k] <- rkg[1] / total
              p_know[s, n, k] <- rkg[2] / total
            } else {
              # R/K mode: existing binormal strip logic
              z_tau_lo <- (tau_lo - mu_M) / sigma_M
              z_tau_hi <- (tau_hi - mu_M) / sigma_M
              p_R <- binormal_strip_upper_r(z_cR, z_tau_lo, z_tau_hi, rho)
              p_Kn <- binormal_strip_lower_r(z_cR, z_tau_lo, z_tau_hi, rho)
              probs[s, n, k] <- p_R + p_Kn
              p_remember[s, n, k] <- p_R / (p_R + p_Kn)
            }
          } else {
            new_idx <- match(k, new_levels)
            if (new_idx == 1) {
              tau_hi <- tau[1]
              z_tau_hi <- (tau_hi - mu_M) / sigma_M
              probs[s, n, k] <- pnorm(z_tau_hi)
            } else {
              tau_hi <- tau[1]
              z_tau_hi <- (tau_hi - mu_M) / sigma_M
              if (k < old_level_map[1]) {
                # new_idx > 1 here (new_idx == 1 is handled by the branch above)
                probs[s, n, k] <- pnorm(z_tau_hi) / length(new_levels[new_levels < old_level_map[1]])
              }
            }
          }
        }

      } else {
        # Lure trial: reference distribution mu_R=0, sigma_R=1, mu_F=0, sigma_F=1
        z_cR_l <- c_R

        for (k in seq_len(K)) {
          old_idx <- match(k, old_level_map)

          if (!is.na(old_idx)) {
            tau_lo <- tau[old_idx]
            tau_hi <- if (old_idx == J) 20.0 else tau[old_idx + 1]

            if (n_rkg == 3) {
              # R/K/G mode with lure params
              rkg <- compute_rkg_probs_r(0, 1, 0, 1, c_R, c_K, tau_lo, tau_hi)
              probs[s, n, k] <- sum(rkg)
              total <- sum(rkg)
              p_remember[s, n, k] <- rkg[1] / total
              p_know[s, n, k] <- rkg[2] / total
            } else {
              z_tau_lo_l <- tau_lo / sigma_M_l
              z_tau_hi_l <- tau_hi / sigma_M_l
              p_R <- binormal_strip_upper_r(z_cR_l, z_tau_lo_l, z_tau_hi_l, rho_l)
              p_Kn <- binormal_strip_lower_r(z_cR_l, z_tau_lo_l, z_tau_hi_l, rho_l)
              probs[s, n, k] <- p_R + p_Kn
              p_remember[s, n, k] <- p_R / (p_R + p_Kn)
            }
          } else {
            z_tau1_l <- tau[1] / sigma_M_l
            if (k < old_level_map[1]) {
              probs[s, n, k] <- pnorm(z_tau1_l)
            }
          }
        }
      }

      # Handle multiple new levels below tau[1]
      new_below <- new_levels[new_levels < old_level_map[1]]
      if (length(new_below) > 1) {
        if (is_old[n] == 1) {
          total_new <- pnorm((tau[1] - mu_M) / sigma_M)
        } else {
          total_new <- pnorm(tau[1] / sigma_M_l)
        }
        for (idx in seq_along(new_below)) {
          probs[s, n, new_below[idx]] <- total_new / length(new_below)
        }
      }
    }
  }

  probs <- pmax(probs, 1e-10)
  for (s in seq_len(S)) {
    for (n in seq_len(N)) {
      total <- sum(probs[s, n, ])
      if (total > 0) probs[s, n, ] <- probs[s, n, ] / total
    }
  }
  attr(probs, "p_remember") <- p_remember
  if (n_rkg == 3) attr(probs, "p_know") <- p_know
  probs
}


# =============================================================================
# Newdata Support
# =============================================================================

#' Rebuild design matrices for new data
#' @noRd
rebuild_design_matrices <- function(model, newdata) {
  md <- model$model_data
  parsed <- model$parsed
  family <- model$family
  family_name <- family$family
  is_cumulative <- family_name == "cumulative"
  is_source_mixture <- family_name == "source_mixture"
  is_bivariate <- family_name %in% c("bivariate_sdt", "bivariate_dp", "bivariate_cumulative")
  is_vrdp2d <- family_name == "vrdp2d"
  is_cdp <- family_name == "cdp"
  train_data <- model$data

  # Validate required columns
  required_cols <- character(0)
  if (!is_cumulative && !is_bivariate && !is_vrdp2d) {
    is_old_var <- parsed$response$is_old
    if (!is.null(is_old_var)) required_cols <- c(required_cols, is_old_var)
  }
  if (is_source_mixture) {
    source_var <- parsed$response$source
    if (!is.null(source_var)) required_cols <- c(required_cols, source_var)
  }
  if (is_bivariate || is_vrdp2d) {
    item_type_var <- parsed$response$item_type
    if (!is.null(item_type_var)) required_cols <- c(required_cols, item_type_var)
    resp2_var <- parsed$response2
    if (!is.null(resp2_var)) required_cols <- c(required_cols, resp2_var)
  }
  if (is_cdp) {
    rk_var <- parsed$response$rk
    if (!is.null(rk_var)) required_cols <- c(required_cols, rk_var)
  }

  missing_cols <- setdiff(required_cols, names(newdata))
  if (length(missing_cols) > 0) {
    stop("newdata is missing required columns: ", paste(missing_cols, collapse = ", "))
  }

  N <- nrow(newdata)
  src <- list(N = N)

  # is_old and source vectors
  if (!is_cumulative && !is_bivariate && !is_vrdp2d && !is.null(parsed$response$is_old)) {
    src$is_old <- .as_is_old(newdata[[parsed$response$is_old]])
  } else if (is_cumulative) {
    src$is_old <- rep(1L, N)
  }

  if (is_source_mixture && !is.null(parsed$response$source)) {
    src_vals <- newdata[[parsed$response$source]]
    if (is.factor(src_vals)) {
      src$source <- as.integer(src_vals) - 1L  # 0-indexed
    } else {
      src$source <- as.integer(src_vals)
    }
  }

  # Bivariate/vrdp2d: item_type and y2
  if ((is_bivariate || is_vrdp2d) && !is.null(parsed$response$item_type)) {
    item_type_vals <- newdata[[parsed$response$item_type]]
    if (is.factor(item_type_vals)) {
      src$item_type <- as.integer(item_type_vals)  # 1=new, 2=A, 3=B
    } else {
      src$item_type <- as.integer(item_type_vals)
    }
    if (!is.null(parsed$response2)) {
      src$y2 <- as.integer(newdata[[parsed$response2]])
    }
  }

  # CDP: rk
  if (is_cdp && !is.null(parsed$response$rk)) {
    rk_values <- newdata[[parsed$response$rk]]
    if (is.character(rk_values) || is.factor(rk_values)) {
      rk_values <- as.character(rk_values)
      rk_numeric <- ifelse(toupper(rk_values) %in% c("R", "REMEMBER", "1"), 1L,
                            ifelse(toupper(rk_values) %in% c("K", "KNOW", "2"), 2L,
                                   ifelse(toupper(rk_values) %in% c("G", "GUESS", "3"), 3L, NA_integer_)))
    } else {
      rk_numeric <- as.integer(rk_values)
    }
    # Replace NAs with dummy value (1) -- Stan only reads rk for old-level responses
    rk_numeric[is.na(rk_numeric)] <- 1L
    src$rk <- rk_numeric
  }

  # Rebuild fixed effects design matrices
  encoding_vars <- parsed$encoding_vars
  is_old_var <- if (!is_cumulative) parsed$response$is_old else ".cumulative_is_old"
  if (is_cumulative) newdata$.cumulative_is_old <- 1L

  # Helper: strip smooth Xs column names (prefixed with "sXs") from ref_colnames
  # since those are appended separately from smooth basis reconstruction
  strip_smooth_cols <- function(colnames) {
    colnames[!grepl("^sXs", colnames)]
  }

  # dprime
  if (isTRUE(md$has_dprime_fixed)) {
    src$X_dprime <- rebuild_one_design_matrix(
      md$dprime_fixed$formula, newdata, encoding_vars, is_old_var,
      ref_colnames = strip_smooth_cols(md$dprime_fixed$coef_names),
      conditional_info = md$dprime_fixed$conditional_info,
      train_data = train_data
    )
  }

  # criterion
  if (md$criterion$is_intercept_only) {
    src$X_criterion <- matrix(1, nrow = N, ncol = 1)
    colnames(src$X_criterion) <- "(Intercept)"
  } else if (!is.null(md$criterion$conditional_info)) {
    # Criterion has conditional terms -- rebuild via conditional path
    src$X_criterion <- rebuild_one_design_matrix(
      md$criterion$fixed_formula, newdata, NULL, NULL,
      ref_colnames = md$criterion$coef_names,
      conditional_info = md$criterion$conditional_info,
      train_data = train_data
    )
  } else if (!is.null(md$criterion$fixed_formula)) {
    # Use stored formula (from models built after this fix)
    src$X_criterion <- rebuild_one_design_matrix(
      md$criterion$fixed_formula, newdata, NULL, NULL,
      ref_colnames = md$criterion$coef_names,
      train_data = train_data
    )
  } else {
    # Fallback for models built before criterion formula was stored
    src$X_criterion <- model.matrix(
      as.formula(paste("~", paste(md$criterion$coef_names, collapse = " + "))),
      data = newdata
    )
    src$X_criterion <- align_matrix_columns(src$X_criterion, md$criterion$coef_names)
  }

  # Other parameter design matrices
  param_names <- c("sigma", "lambda", "dprime2", "sigma2",
                    "dprime_B", "lambda_B",
                    "dprime_L", "sigma_L", "lambda_L",
                    "discrim", "discrim_B", "sigma_B", "sigma2_B",
                    "rho", "rho_B", "rho_N", "lambda2", "lambda2_B", "rec_crit", "know_crit")
  for (pname in param_names) {
    if (isTRUE(md[[paste0("has_", pname)]])) {
      fixed_info <- md[[paste0(pname, "_fixed")]]
      src[[paste0("X_", pname)]] <- rebuild_one_design_matrix(
        fixed_info$formula, newdata, encoding_vars, is_old_var,
        ref_colnames = strip_smooth_cols(fixed_info$coef_names),
        conditional_info = fixed_info$conditional_info,
        train_data = train_data
      )
    }
  }

  # Random effects: reuse original group indices mapped to newdata
  all_re_params <- c("dprime", "sigma", "lambda", "dprime2", "sigma2",
                      "dprime_B", "lambda_B",
                      "dprime_L", "sigma_L", "lambda_L", "rec_crit", "know_crit",
                      "discrim", "discrim_B", "sigma_B", "sigma2_B",
                      "rho", "rho_B", "rho_N", "lambda2", "lambda2_B")
  for (pname in all_re_params) {
    re_list <- md[[paste0(pname, "_random")]]
    if (is.null(re_list)) next

    new_re_list <- list()
    for (group in names(re_list)) {
      re <- re_list[[group]]
      new_re <- re  # copy structure

      # Remap group indices
      remap <- remap_group_indices(newdata, group, model$data, re)
      new_re$group_idx <- remap$group_idx
      if (!is.null(remap$warning)) warning(remap$warning)

      # Rebuild term_idx if needed
      if (!is.null(re$conditional_info)) {
        # Conditional RE: rebuild via conditional columns
        cond_result <- build_conditional_columns(
          term = re$conditional_info$term,
          where_str = re$conditional_info$where_str,
          data = newdata,
          interaction_partners = re$conditional_info$interaction_partners
        )
        Z_new <- align_matrix_columns(cond_result$X, re$conditional_info$col_names)
        if (check_simple_categorical(Z_new)) {
          new_re$term_idx <- build_term_index(Z_new)
          new_re$Z <- NULL
          new_re$use_z_matrix <- FALSE
        } else {
          new_re$term_idx <- NULL
          new_re$Z <- Z_new
          new_re$use_z_matrix <- TRUE
        }
      } else if (!is.null(re$term_idx) && !isTRUE(re$use_z_matrix)) {
        new_re$term_idx <- rebuild_term_idx(re, newdata, encoding_vars, is_old_var)
      } else if (isTRUE(re$use_z_matrix)) {
        # Rebuild Z matrix if needed
        new_re$Z <- rebuild_z_matrix(re, newdata, encoding_vars, is_old_var)
      }

      new_re_list[[group]] <- new_re
    }
    src[[paste0(pname, "_random")]] <- new_re_list
  }

  # Criterion random effects
  if (!is.null(md$criterion$random)) {
    new_crit_re <- list()
    for (group in names(md$criterion$random)) {
      re <- md$criterion$random[[group]]
      new_re <- re

      remap <- remap_group_indices(newdata, group, model$data, re)
      new_re$group_idx <- remap$group_idx
      if (!is.null(remap$warning)) warning(remap$warning)

      if (!is.null(re$conditional_info)) {
        cond_result <- build_conditional_columns(
          term = re$conditional_info$term,
          where_str = re$conditional_info$where_str,
          data = newdata,
          interaction_partners = re$conditional_info$interaction_partners
        )
        Z_new <- align_matrix_columns(cond_result$X, re$conditional_info$col_names)
        if (check_simple_categorical(Z_new)) {
          new_re$term_idx <- build_term_index(Z_new)
          new_re$Z <- NULL
          new_re$use_z_matrix <- FALSE
        } else {
          new_re$term_idx <- NULL
          new_re$Z <- Z_new
          new_re$use_z_matrix <- TRUE
        }
      } else if (!is.null(re$term_idx) && !isTRUE(re$use_z_matrix)) {
        new_re$term_idx <- rebuild_term_idx(re, newdata, NULL, NULL)
      } else if (isTRUE(re$use_z_matrix)) {
        new_re$Z <- rebuild_z_matrix(re, newdata, NULL, NULL)
      }

      new_crit_re[[group]] <- new_re
    }
    src$criterion_random <- new_crit_re
  }

  # Criterion2 design matrix and random effects (for bivariate_sdt, vrdp2d)
  if (isTRUE(md$has_criterion2) && !is.null(md$criterion2)) {
    if (md$criterion2$is_intercept_only) {
      src$X_criterion2 <- matrix(1, nrow = N, ncol = 1)
      colnames(src$X_criterion2) <- "(Intercept)"
    } else if (!is.null(md$criterion2$conditional_info)) {
      src$X_criterion2 <- rebuild_one_design_matrix(
        md$criterion2$fixed_formula, newdata, NULL, NULL,
        ref_colnames = md$criterion2$coef_names,
        conditional_info = md$criterion2$conditional_info,
        train_data = train_data
      )
    } else if (!is.null(md$criterion2$fixed_formula)) {
      src$X_criterion2 <- rebuild_one_design_matrix(
        md$criterion2$fixed_formula, newdata, NULL, NULL,
        ref_colnames = md$criterion2$coef_names,
        train_data = train_data
      )
    } else {
      src$X_criterion2 <- matrix(1, nrow = N, ncol = 1)
      colnames(src$X_criterion2) <- "(Intercept)"
    }

    if (!is.null(md$criterion2$random)) {
      new_crit2_re <- list()
      for (group in names(md$criterion2$random)) {
        re <- md$criterion2$random[[group]]
        new_re <- re
        remap <- remap_group_indices(newdata, group, model$data, re)
        new_re$group_idx <- remap$group_idx
        if (!is.null(remap$warning)) warning(remap$warning)

        if (!is.null(re$conditional_info)) {
          cond_result <- build_conditional_columns(
            term = re$conditional_info$term,
            where_str = re$conditional_info$where_str,
            data = newdata,
            interaction_partners = re$conditional_info$interaction_partners
          )
          Z_new <- align_matrix_columns(cond_result$X, re$conditional_info$col_names)
          if (check_simple_categorical(Z_new)) {
            new_re$term_idx <- build_term_index(Z_new)
            new_re$Z <- NULL
            new_re$use_z_matrix <- FALSE
          } else {
            new_re$term_idx <- NULL
            new_re$Z <- Z_new
            new_re$use_z_matrix <- TRUE
          }
        } else if (!is.null(re$term_idx) && !isTRUE(re$use_z_matrix)) {
          new_re$term_idx <- rebuild_term_idx(re, newdata, NULL, NULL)
        } else if (isTRUE(re$use_z_matrix)) {
          new_re$Z <- rebuild_z_matrix(re, newdata, NULL, NULL)
        }
        new_crit2_re[[group]] <- new_re
      }
      src$criterion2_random <- new_crit2_re
    }
  }

  src$cross_cor <- md$cross_cor

  # Rebuild smooth basis matrices for new data
  if (!is.null(md$smooth_data)) {
    new_smooth_data <- list()
    for (pname in names(md$smooth_data)) {
      new_smooth_data[[pname]] <- list()
      for (i in seq_along(md$smooth_data[[pname]])) {
        sm <- md$smooth_data[[pname]][[i]]
        new_sm <- sm  # copy metadata
        new_sm$components <- list()
        for (j in seq_along(sm$components)) {
          comp <- sm$components[[j]]
          pred <- build_smooth_prediction_basis(comp, newdata)
          new_comp <- comp
          new_comp$Xs <- pred$Xs
          new_comp$Zs_list <- pred$Zs_list
          new_sm$components[[j]] <- new_comp
        }
        # By-factor smooths: a new by-level (unseen in training) gets a
        # prior-sampled penalized deviation (under allow_new_levels). The
        # reference component supplies the basis and penalty SD scale.
        if (!is.null(sm$by) && !is.na(sm$by) && sm$by %in% names(newdata) &&
            length(sm$components) > 0) {
          nd_by <- as.character(newdata[[sm$by]])
          for (nl in setdiff(unique(nd_by), sm$bylevels)) {
            ref <- sm$components[[1]]
            syn <- ref
            syn$new_level <- TRUE
            syn$Zs_list <- build_new_bylevel_basis(ref, newdata, sm$by, nl)
            syn$Xs <- matrix(0, nrow = nrow(newdata), ncol = 0)
            syn$Xs_colnames <- character(0)
            new_sm$components[[length(new_sm$components) + 1L]] <- syn
          }
        }
        new_smooth_data[[pname]][[i]] <- new_sm
      }

      # Append new Xs columns to the X matrix for this parameter
      x_name <- paste0("X_", pname)
      if (!is.null(src[[x_name]])) {
        for (sm in new_smooth_data[[pname]]) {
          for (comp in sm$components) {
            if (NCOL(comp$Xs) > 0) {
              src[[x_name]] <- cbind(src[[x_name]], comp$Xs)
            }
          }
        }
      }
    }
    src$smooth_data <- new_smooth_data
  }

  src
}


#' Align matrix columns to match a reference set of column names
#'
#' Ensures the output matrix has exactly the columns in ref_colnames, in order.
#' Missing columns are filled with zeros; extra columns are dropped.
#' @noRd
align_matrix_columns <- function(X, ref_colnames) {
  current_cols <- colnames(X)
  if (identical(current_cols, ref_colnames)) return(X)

  N <- nrow(X)
  X_aligned <- matrix(0, nrow = N, ncol = length(ref_colnames))
  colnames(X_aligned) <- ref_colnames

  shared <- intersect(ref_colnames, current_cols)
  if (length(shared) > 0) {
    X_aligned[, shared] <- X[, shared, drop = FALSE]
  }

  X_aligned
}


#' Safely drop unused factor levels for encoding variables in old-item data.
#' Only drops levels if the result still has 2+ levels, avoiding the
#' "contrasts can be applied only to factors with 2 or more levels" error
#' that occurs when non-focal variables are held at a single value
#' (e.g., in prediction grids).
#' @noRd
safe_droplevels_old <- function(old_data, formula_vars, encoding_vars = NULL) {
  for (v in formula_vars) {
    if (v %in% names(old_data) && is.factor(old_data[[v]])) {
      # Only drop levels for encoding vars, and only if 2+ remain
      if (!is.null(encoding_vars) && v %in% encoding_vars) {
        dropped <- droplevels(old_data[[v]])
        if (nlevels(dropped) >= 2) {
          old_data[[v]] <- dropped
        }
      }
    }
  }
  old_data
}


#' Rebuild a single fixed effects design matrix for newdata
#' @param ref_colnames Column names from the training design matrix. If provided,
#'   output is aligned to match (missing cols zeroed, extra cols dropped).
#' @noRd
rebuild_one_design_matrix <- function(formula, newdata, encoding_vars, is_old_var,
                                       ref_colnames = NULL, conditional_info = NULL,
                                       train_data = NULL) {
  # If conditional info present, rebuild via conditional path
  if (!is.null(conditional_info)) {
    # conditional_info is a list of conditional_info objects (one per conditional term)
    # Rebuild the full matrix
    N <- nrow(newdata)
    blocks <- list()

    for (ci in conditional_info) {
      X_cond <- rebuild_conditional_columns(ci, newdata)
      blocks[[length(blocks) + 1]] <- X_cond
    }

    # Regular columns (intercept, non-conditional terms) come first, then
    # conditional columns. Build the conditional blocks and align to ref_colnames.
    if (length(blocks) == 1) {
      X <- blocks[[1]]
    } else {
      X <- do.call(cbind, blocks)
    }

    if (!is.null(ref_colnames)) {
      # The ref_colnames includes ALL columns (regular + conditional)
      # Only conditional columns were built here. If there are regular columns,
      # detect and add them.
      cond_cols <- colnames(X)
      missing_cols <- setdiff(ref_colnames, cond_cols)
      if (length(missing_cols) > 0) {
        # These are regular (non-conditional) columns -- build via model.matrix
        # Use the formula but only for non-conditional terms
        if (!is.null(formula) && inherits(formula, "formula")) {
          X_reg <- tryCatch({
            f <- as.formula(paste(deparse(formula), collapse = " "))
            model.matrix(f, data = newdata)
          }, error = function(e) NULL)
          if (!is.null(X_reg)) {
            X <- cbind(X_reg[, intersect(colnames(X_reg), missing_cols), drop = FALSE], X)
          }
        }
      }
      X <- align_matrix_columns(X, ref_colnames)
    }
    return(X)
  }

  if (is.null(formula) || !inherits(formula, "formula")) return(NULL)

  # Strip cached terms/predvars from the formula to avoid
  # "object 'conditiondeep' not found" errors when reusing formulas
  formula <- as.formula(paste(deparse(formula), collapse = " "))

  formula_vars <- all.vars(formula)
  has_encoding <- !is.null(encoding_vars) && any(encoding_vars %in% formula_vars)

  if (has_encoding && !is.null(is_old_var)) {
    old_idx <- which(newdata[[is_old_var]] == 1)
    old_data <- newdata[old_idx, , drop = FALSE]
    old_data <- safe_droplevels_old(old_data, formula_vars, encoding_vars)
    # Ensure all categorical formula variables have 2+ factor levels.
    # When newdata is a sparse grid (e.g., from a prediction grid), a variable
    # may have only 1 unique value. Use train_data to recover proper levels.
    if (!is.null(train_data)) {
      for (v in formula_vars) {
        if (v %in% names(old_data) && v %in% names(train_data)) {
          col <- old_data[[v]]
          tcol <- train_data[[v]]
          if (is.character(col) || (is.factor(col) && nlevels(col) < 2)) {
            if (is.factor(tcol)) {
              old_data[[v]] <- factor(old_data[[v]], levels = levels(tcol))
            } else if (is.character(tcol)) {
              old_data[[v]] <- factor(old_data[[v]], levels = sort(unique(tcol)))
            }
          }
        }
      }
    }
    X_old <- model.matrix(formula, data = old_data)
    N <- nrow(newdata)
    X <- matrix(0, nrow = N, ncol = ncol(X_old))
    colnames(X) <- colnames(X_old)
    X[old_idx, ] <- X_old
  } else {
    # Ensure categorical variables have 2+ levels for model.matrix contrasts
    if (!is.null(train_data)) {
      for (v in formula_vars) {
        if (v %in% names(newdata) && v %in% names(train_data)) {
          col <- newdata[[v]]
          tcol <- train_data[[v]]
          needs_fix <- is.character(col) ||
                       (is.factor(col) && nlevels(col) < 2)
          if (needs_fix) {
            if (is.factor(tcol)) {
              newdata[[v]] <- factor(newdata[[v]], levels = levels(tcol))
            } else if (is.character(tcol)) {
              newdata[[v]] <- factor(newdata[[v]], levels = sort(unique(tcol)))
            }
          }
        }
      }
    }
    X <- model.matrix(formula, data = newdata)
  }

  # Align columns to match training structure
  if (!is.null(ref_colnames)) {
    X <- align_matrix_columns(X, ref_colnames)
  }

  X
}


#' Remap group indices from training data levels to newdata
#' @noRd
remap_group_indices <- function(newdata, group, train_data, re) {
  if (!group %in% names(newdata)) {
    stop("newdata is missing grouping variable: ", group)
  }

  # Get training levels
  if (is.factor(train_data[[group]])) {
    train_levels <- levels(train_data[[group]])
  } else {
    train_levels <- levels(as.factor(train_data[[group]]))
  }

  new_vals <- as.character(newdata[[group]])
  new_in_train <- new_vals %in% train_levels

  group_idx <- integer(length(new_vals))
  for (i in seq_along(new_vals)) {
    if (new_in_train[i]) {
      group_idx[i] <- match(new_vals[i], train_levels)
    } else {
      group_idx[i] <- 0L  # sentinel for new levels
    }
  }

  warn_msg <- NULL
  new_levels <- unique(new_vals[!new_in_train])
  if (length(new_levels) > 0) {
    warn_msg <- paste0("New levels in grouping variable '", group,
                        "' not in training data: ",
                        paste(new_levels, collapse = ", "),
                        ". Random effects set to 0 for these levels.")
  }

  list(group_idx = group_idx, warning = warn_msg)
}


#' Rebuild term index for newdata
#' @noRd
rebuild_term_idx <- function(re, newdata, encoding_vars, is_old_var) {
  if (is.null(re$level_names) || length(re$level_names) == 0) {
    return(rep(0L, nrow(newdata)))
  }

  if (re$dim == 1) {
    return(rep(1L, nrow(newdata)))
  }

  # Use the stored RE formula (with original variable names, not dummy-coded)
  re_formula <- re$re_formula
  if (is.null(re_formula)) {
    # Fallback: intercept only
    return(rep(1L, nrow(newdata)))
  }

  is_encoding <- isTRUE(re$is_encoding)
  if (is_encoding && !is.null(is_old_var)) {
    old_idx <- which(newdata[[is_old_var]] == 1)
    old_data <- newdata[old_idx, , drop = FALSE]
    # For encoding REs, all vars in the formula are encoding vars
    old_data <- safe_droplevels_old(old_data, all.vars(re_formula), all.vars(re_formula))
    Z_old <- tryCatch(model.matrix(re_formula, data = old_data), error = function(e) NULL)
    if (is.null(Z_old)) return(rep(0L, nrow(newdata)))

    # Align columns to training level names before building full Z
    Z_old <- align_matrix_columns(Z_old, re$level_names)

    Z <- matrix(0, nrow = nrow(newdata), ncol = ncol(Z_old))
    Z[old_idx, ] <- Z_old
  } else {
    Z <- tryCatch(model.matrix(re_formula, data = newdata), error = function(e) NULL)
    if (is.null(Z)) return(rep(0L, nrow(newdata)))

    # Align columns to training level names
    Z <- align_matrix_columns(Z, re$level_names)
  }

  build_term_index(Z)
}


#' Rebuild Z matrix for newdata
#' @noRd
rebuild_z_matrix <- function(re, newdata, encoding_vars, is_old_var) {
  # Use the stored RE formula (with original variable names, not dummy-coded)
  re_formula <- re$re_formula
  if (is.null(re_formula)) return(NULL)

  is_encoding <- isTRUE(re$is_encoding)
  if (is_encoding && !is.null(is_old_var)) {
    old_idx <- which(newdata[[is_old_var]] == 1)
    old_data <- newdata[old_idx, , drop = FALSE]
    old_data <- safe_droplevels_old(old_data, all.vars(re_formula), all.vars(re_formula))
    Z_old <- model.matrix(re_formula, data = old_data)
    # Align columns to training level names
    Z_old <- align_matrix_columns(Z_old, re$level_names)
    Z <- matrix(0, nrow = nrow(newdata), ncol = ncol(Z_old))
    Z[old_idx, ] <- Z_old
  } else {
    Z <- model.matrix(re_formula, data = newdata)
    # Align columns to training level names
    Z <- align_matrix_columns(Z, re$level_names)
  }

  Z
}
