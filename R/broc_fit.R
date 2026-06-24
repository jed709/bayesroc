#' SDT Fit Wrapper Class
#'
#' Backend-agnostic S3 wrapper around CmdStanMCMC (Stan) or NumPyro fit objects.
#' Uses environment-based S3 class so $draws(), $summary(), etc. work like
#' CmdStanMCMC's R6 interface.


# =============================================================================
# Constructor
# =============================================================================

#' Create an broc_fit object
#'
#' @param backend_fit Raw backend object (CmdStanMCMC for Stan, list for JAX)
#' @param model broc_model object
#' @param backend "stan" or "jax"
#' @param num_chains Number of chains used
#' @param iter_sampling Number of post-warmup iterations per chain
#' @param elapsed Elapsed time in seconds (numeric)
#' @param draws_array For JAX: pre-converted posterior::draws_array (NULL for Stan)
#' @param max_treedepth Maximum NUTS tree depth used in the fit (default 10).
#' @return An broc_fit object (environment with S3 class)
#' @keywords internal
#' @export
new_broc_fit <- function(backend_fit, model, backend = "stan",
                        num_chains = 4L, iter_sampling = 1000L,
                        elapsed = NULL, draws_array = NULL,
                        max_treedepth = 10L, save_loglik = TRUE,
                        diagnostics = NULL) {

  self <- new.env(parent = emptyenv())

  # Store internals
  self$.backend_fit <- backend_fit
  self$.backend <- backend
  self$.num_chains <- as.integer(num_chains)
  self$.iter_sampling <- as.integer(iter_sampling)
  self$.elapsed <- elapsed
  self$.draws_array <- draws_array
  self$.max_treedepth <- as.integer(max_treedepth)
  self$.save_loglik <- isTRUE(save_loglik)
  self$.diagnostics <- diagnostics   # precomputed; used when backend_fit absent

  # No live CmdStanMCMC (JAX, or a draws-only reload) -> use the draws_array path.
  use_cmdstan <- backend == "stan" && !is.null(backend_fit)

  # ---- draws() ----
  self$draws <- function(variables = NULL, format = "array") {
    if (use_cmdstan) {
      if (is.null(variables)) {
        backend_fit$draws(format = format)
      } else {
        backend_fit$draws(variables = variables, format = format)
      }
    } else {
      # JAX / draws-only: subset from cached draws_array
      da <- self$.draws_array
      if (!is.null(variables)) {
        all_vars <- posterior::variables(da)
        # Expand variable prefixes (e.g. "beta_dprime" -> all beta_dprime[...])
        matched <- character(0)
        for (v in variables) {
          exact <- v %in% all_vars
          if (exact) {
            matched <- c(matched, v)
          } else {
            # Try prefix match (Stan convention: "beta_dprime" matches "beta_dprime[1]", etc.)
            pat <- paste0("^", gsub("\\[", "\\\\[", v), "(\\[|$)")
            hits <- grep(pat, all_vars, value = TRUE)
            matched <- c(matched, hits)
          }
        }
        if (length(matched) == 0) {
          stop("No matching variables found for: ", paste(variables, collapse = ", "))
        }
        da <- posterior::subset_draws(da, variable = matched)
      }
      if (format == "matrix") {
        posterior::as_draws_matrix(da)
      } else if (format == "array") {
        da
      } else if (format == "df" || format == "data.frame") {
        posterior::as_draws_df(da)
      } else {
        da
      }
    }
  }

  # ---- summary() ----
  self$summary <- function(variables = NULL, ...) {
    if (use_cmdstan) {
      if (is.null(variables)) {
        backend_fit$summary(...)
      } else {
        backend_fit$summary(variables = variables, ...)
      }
    } else {
      da <- self$.draws_array
      if (!is.null(variables)) {
        all_vars <- posterior::variables(da)
        matched <- character(0)
        for (v in variables) {
          if (v %in% all_vars) {
            matched <- c(matched, v)
          } else {
            pat <- paste0("^", gsub("\\[", "\\\\[", v), "(\\[|$)")
            hits <- grep(pat, all_vars, value = TRUE)
            matched <- c(matched, hits)
          }
        }
        da <- posterior::subset_draws(da, variable = matched)
      }
      posterior::summarise_draws(da, ...)
    }
  }

  # ---- diagnostics() ----
  self$diagnostic_summary <- function(...) {
    if (use_cmdstan) {
      backend_fit$diagnostic_summary(...)
    } else if (!is.null(self$.diagnostics)) {
      self$.diagnostics            # precomputed (draws-only reload)
    } else {
      # JAX: compute from extra_fields
      # Extra field arrays are [chains, draws]. Process per-chain directly
      # to avoid column-major flattening issues with as.vector().
      ef <- backend_fit$extra_fields
      nc <- self$.num_chains
      list(
        num_divergent = if (!is.null(ef$diverging)) {
          vapply(seq_len(nc), function(ch) sum(as.integer(ef$diverging[ch, ])), integer(1))
        } else rep(0L, nc),
        ebfmi = if (!is.null(ef$energy)) {
          vapply(seq_len(nc), function(ch) {
            e <- ef$energy[ch, ]
            de <- diff(e)
            sum(de^2) / sum((e - mean(e))^2)
          }, numeric(1))
        } else rep(NA_real_, nc),
        num_max_treedepth = if (!is.null(ef$num_steps)) {
          max_steps <- 2^(if (!is.null(self$.max_treedepth)) self$.max_treedepth else 10L)
          vapply(seq_len(nc), function(ch) sum(as.integer(ef$num_steps[ch, ] >= max_steps)), integer(1))
        } else rep(0L, nc)
      )
    }
  }

  # ---- loo() ----
  self$loo <- function(variables = "log_lik", cores = 4, ...) {
    if (use_cmdstan) {
      backend_fit$loo(variables = variables, cores = cores, ...)
    } else {
      if (!requireNamespace("loo", quietly = TRUE)) {
        stop("Package 'loo' is required for LOO-CV.")
      }
      if (identical(variables, "log_lik") && !isTRUE(self$.save_loglik) &&
          !any(grepl("^log_lik(\\[|$)", posterior::variables(self$.draws_array)))) {
        stop("This fit was saved with save_loglik = FALSE, so the ",
             "log-likelihood needed for loo() is not available. Refit with ",
             "save_loglik = TRUE (the default).", call. = FALSE)
      }
      ll <- self$draws(variables = variables, format = "matrix")
      r_eff <- loo::relative_eff(exp(ll), chain_id = rep(seq_len(self$.num_chains),
        each = self$.iter_sampling))
      loo::loo(ll, r_eff = r_eff, cores = cores, ...)
    }
  }

  # ---- num_chains() ----
  self$num_chains <- function() {
    self$.num_chains
  }

  # ---- metadata() ----
  self$metadata <- function() {
    if (use_cmdstan) {
      backend_fit$metadata()
    } else {
      list(iter_sampling = self$.iter_sampling,
           variables = if (!is.null(self$.draws_array))
             posterior::variables(self$.draws_array) else NULL)
    }
  }

  # ---- save_object() ----
  # Delegates to the top-level save_broc_fit() helper, which extracts a
  # plain-data bundle rather than serializing the whole environment-based
  # broc_fit (which captures large closures and can be slow / unstable for
  # real-scale fits).
  self$save_object <- function(file, ...) {
    save_broc_fit(self, file)
  }

  # Set broc_model attribute for backwards compatibility
  attr(self, "broc_model") <- model
  class(self) <- c("broc_fit", "environment")

  self
}


# =============================================================================
# Save / load
# =============================================================================

#' Save a fitted broc_fit to disk
#'
#' Writes a portable .rds containing the draws, the broc_model, and the metadata
#' needed to reconstruct the fit. Pass `file` to [fit_broc()] to save
#' automatically, or call this on an existing fit. Always save with this (or
#' `file =` in [fit_broc()]), not `saveRDS()`, which can lose Stan draws.
#'
#' @param fit A `broc_fit` object from [fit_broc()].
#' @param file Path to write to (use `.rds` extension by convention).
#' @param compress Compression for `saveRDS`. Default `FALSE` (fast; posterior
#'   draws barely compress). Set `"gzip"`/`"xz"` for a smaller, slower file.
#' @param verbose If `TRUE` (default), print a brief progress message.
#' @param include_loglik Whether to write `log_lik` (needed for [loo()]). `NULL`
#'   (default) follows the fit's `fit_broc(save_loglik = )` setting; `TRUE`/`FALSE`
#'   overrides it for this save.
#' @return Invisibly returns `file`.
#' @seealso [load_broc_fit()].
#' @examples
#' \dontrun{
#' fit <- fit_broc(model)
#' save_broc_fit(fit, "my_fit.rds")
#' fit2 <- load_broc_fit("my_fit.rds")
#' summary(fit2)
#'
#' # Smaller file (slower) when disk space matters:
#' save_broc_fit(fit, "tmp.rds", compress = "gzip")
#' }
#' @export
save_broc_fit <- function(fit, file, compress = FALSE, verbose = TRUE,
                          include_loglik = NULL) {
  if (!inherits(fit, "broc_fit")) {
    stop("`fit` must be a broc_fit object.", call. = FALSE)
  }
  backend <- fit$.backend %||% "stan"
  keep_ll <- if (!is.null(include_loglik)) isTRUE(include_loglik)
             else isTRUE(fit$.save_loglik %||% TRUE)

  if (verbose) {
    message("Saving broc_fit to ", file, "...")
  }
  t0 <- proc.time()["elapsed"]

  if (!keep_ll) {
    # Drop log_lik (only used by loo()); store the draws + precomputed diagnostics.
    da <- if (backend == "stan")
            posterior::as_draws_array(fit$.backend_fit$draws())
          else fit$.draws_array
    vars  <- posterior::variables(da)
    da    <- posterior::subset_draws(da, variable = vars[!grepl("^log_lik(\\[|$)", vars)])
    bundle <- list(
      .bayesroc_save_format = 1L,
      backend          = backend,
      backend_fit      = NULL,
      model            = attr(fit, "broc_model"),
      num_chains       = fit$.num_chains,
      iter_sampling    = fit$.iter_sampling,
      elapsed          = fit$.elapsed,
      max_treedepth    = fit$.max_treedepth %||% 10L,
      draws_array      = da,
      save_loglik      = FALSE,
      diagnostics      = fit$diagnostic_summary()
    )
  } else {
    # For Stan: ask cmdstanr to inline the draws into a portable CmdStanMCMC
    # via its own save_object, then read it back. For JAX: the backend_fit is
    # already a plain in-memory list with the samples/extra_fields.
    bundle_backend_fit <- if (backend == "stan") {
      tmp <- tempfile(fileext = ".rds")
      on.exit(unlink(tmp), add = TRUE)
      fit$.backend_fit$save_object(file = tmp)
      readRDS(tmp)
    } else {
      # draws_array (saved below) already holds these draws; drop the duplicate.
      bf <- fit$.backend_fit
      bf$samples <- NULL
      bf
    }

    bundle <- list(
      .bayesroc_save_format = 1L,
      backend          = backend,
      backend_fit      = bundle_backend_fit,
      model            = attr(fit, "broc_model"),
      num_chains       = fit$.num_chains,
      iter_sampling    = fit$.iter_sampling,
      elapsed          = fit$.elapsed,
      max_treedepth    = fit$.max_treedepth %||% 10L,
      draws_array      = fit$.draws_array,     # NULL for Stan, populated for JAX
      save_loglik      = TRUE
    )
  }
  class(bundle) <- "broc_fit_saved"

  saveRDS(bundle, file = file, compress = compress)

  if (verbose) {
    elapsed <- as.numeric(proc.time()["elapsed"] - t0)
    sz_mb <- file.info(file)$size / 1e6
    message(sprintf("Saved %.1f MB in %.1fs.", sz_mb, elapsed))
  }
  invisible(file)
}

#' Load a previously-saved broc_fit
#'
#' Counterpart to [save_broc_fit()]. Reconstructs the broc_fit wrapper from
#' the saved data bundle, restoring all methods (`$draws()`, `$summary()`,
#' `$loo()`, etc.) and the `broc_model` attribute that downstream functions
#' need.
#'
#' @param file Path to a .rds file written by [save_broc_fit()].
#' @return A `broc_fit` object.
#' @seealso [save_broc_fit()].
#' @export
load_broc_fit <- function(file) {
  if (!file.exists(file)) {
    stop("File not found: ", file, call. = FALSE)
  }
  obj <- readRDS(file)
  if (inherits(obj, "broc_fit")) {
    # Backwards-compat: very early saves were the env directly.
    return(obj)
  }
  # Accept the current marker and the legacy `.ord_save_format` one.
  if (!inherits(obj, "broc_fit_saved") ||
      (is.null(obj$.bayesroc_save_format) && is.null(obj$.ord_save_format))) {
    stop("File '", file, "' does not contain a broc_fit_saved bundle ",
         "(got class: ", paste(class(obj), collapse = "/"), ").",
         call. = FALSE)
  }
  new_broc_fit(
    backend_fit  = obj$backend_fit,
    model        = obj$model,
    backend      = obj$backend,
    num_chains   = obj$num_chains,
    iter_sampling = obj$iter_sampling,
    elapsed      = obj$elapsed,
    draws_array  = obj$draws_array,
    max_treedepth = obj$max_treedepth %||% 10L,
    save_loglik  = obj$save_loglik %||% TRUE,
    diagnostics  = obj$diagnostics
  )
}


# =============================================================================
# Print Method
# =============================================================================

#' @export
print.broc_fit <- function(x, ...) {
  backend <- x$.backend
  cat("bayesroc model fit\n")
  cat("  Backend:", backend, "\n")
  cat("  Chains:", x$.num_chains, "\n")
  cat("  Iterations:", x$.iter_sampling, "(post-warmup per chain)\n")
  if (!is.null(x$.elapsed)) {
    cat("  Elapsed:", round(x$.elapsed, 1), "seconds\n")
  }
  model <- attr(x, "broc_model")
  if (!is.null(model)) {
    cat("  Family:", family_display_name(model$family$family), "\n")
    cat("  N:", model$model_data$N, "\n")
  }
  invisible(x)
}


# =============================================================================
# NumPyro Samples Conversion
# =============================================================================

#' Convert NumPyro samples to posterior::draws_array
#'
#' Takes named list of arrays (each shape [chains, draws, ...]) and converts
#' to a posterior::draws_array with Stan-style bracket naming.
#'
#' @param samples Named list of arrays from NumPyro MCMC
#' @param n_chains Number of chains
#' @param n_draws Number of draws per chain
#' @return posterior::draws_array
#' @noRd
numpyro_samples_to_draws_array <- function(samples, n_chains, n_draws) {
  if (!requireNamespace("posterior", quietly = TRUE)) {
    stop("Package 'posterior' is required for JAX backend.")
  }

  # First pass: count total parameters and collect names
  var_names <- character(0)
  for (name in names(samples)) {
    dims <- dim(samples[[name]])
    if (is.null(dims) || length(dims) == 2) {
      var_names <- c(var_names, name)
    } else if (length(dims) == 3) {
      var_names <- c(var_names, paste0(name, "[", seq_len(dims[3]), "]"))
    } else if (length(dims) == 4) {
      for (k1 in seq_len(dims[3])) {
        var_names <- c(var_names, paste0(name, "[", k1, ",", seq_len(dims[4]), "]"))
      }
    }
  }

  n_params <- length(var_names)

  # Build 3D array directly: [iteration, chain, variable]
  draws_3d <- array(NA_real_, dim = c(n_draws, n_chains, n_params))
  col_idx <- 1L

  for (name in names(samples)) {
    arr <- samples[[name]]
    dims <- dim(arr)

    if (is.null(dims) || length(dims) == 2) {
      if (is.null(dims)) {
        arr <- matrix(arr, nrow = n_chains, ncol = n_draws, byrow = TRUE)
      }
      for (ch in seq_len(n_chains)) {
        draws_3d[, ch, col_idx] <- arr[ch, ]
      }
      col_idx <- col_idx + 1L

    } else if (length(dims) == 3) {
      for (k in seq_len(dims[3])) {
        for (ch in seq_len(n_chains)) {
          draws_3d[, ch, col_idx] <- arr[ch, , k]
        }
        col_idx <- col_idx + 1L
      }

    } else if (length(dims) == 4) {
      for (k1 in seq_len(dims[3])) {
        for (k2 in seq_len(dims[4])) {
          for (ch in seq_len(n_chains)) {
            draws_3d[, ch, col_idx] <- arr[ch, , k1, k2]
          }
          col_idx <- col_idx + 1L
        }
      }
    }
  }

  dimnames(draws_3d) <- list(
    iteration = seq_len(n_draws),
    chain = seq_len(n_chains),
    variable = var_names
  )

  # Set class directly to avoid the copy that as_draws_array() would make
  class(draws_3d) <- c("draws_array", "draws", "array")
  draws_3d
}
