#' SDT Fit Wrapper Class
#'
#' S3 wrapper around a fit's draws and sampler output. Uses an environment-based S3
#' class so $draws(), $summary(), etc. work like CmdStanMCMC's R6 interface.

# =============================================================================
# Constructor
# =============================================================================

#' Create an broc_fit object
#'
#' @param backend_fit Raw backend object (the JAX result list)
#' @param model broc_model object
#' @param backend The backend's name ("jax")
#' @param num_chains Number of chains used
#' @param iter_sampling Number of post-warmup iterations per chain
#' @param elapsed Elapsed time in seconds (numeric)
#' @param draws_array The draws as a posterior::draws_array
#' @param max_treedepth Maximum NUTS tree depth used in the fit (default 10).
#' @param loglik_path Path to the `.npy` log_lik sidecar, or `NULL`
#' @return An broc_fit object (environment with S3 class)
#' @noRd
new_broc_fit <- function(backend_fit, model, backend = "jax",
                        num_chains = 4L, iter_sampling = 1000L,
                        elapsed = NULL, draws_array = NULL,
                        max_treedepth = 10L, sampler = "nuts",
                        save_loglik = TRUE,
                        diagnostics = NULL, loglik_path = NULL,
                        loglik_owned = !is.null(loglik_path), loo = NULL,
                        keep_re = TRUE) {

  self <- new.env(parent = emptyenv())

  self$.backend_fit <- backend_fit
  self$.backend <- backend
  self$.num_chains <- as.integer(num_chains)
  self$.iter_sampling <- as.integer(iter_sampling)
  self$.elapsed <- elapsed
  self$.draws_array <- draws_array
  self$.max_treedepth <- as.integer(max_treedepth)
  self$.sampler <- sampler
  self$.save_loglik <- isTRUE(save_loglik)
  self$.diagnostics <- diagnostics
  self$.loglik_path <- loglik_path
  self$.loglik_owned <- isTRUE(loglik_owned) && !is.null(loglik_path)
  self$.loo <- loo
  self$.keep_re <- isTRUE(keep_re)
  reg.finalizer(self, function(e) {
    if (.owned_sidecar(e)) unlink(e$.loglik_path)
  })

  # ---- draws() ----
  self$draws <- function(variables = NULL, format = "array") {
    if (identical(variables, "log_lik") && !is.null(self$.loglik_path)) {
      if (!file.exists(self$.loglik_path)) {
        stop(.missing_sidecar_msg(self$.loglik_path), call. = FALSE)
      }
      ll <- .read_loglik_CDN(self$.loglik_path)
      arr <- aperm(ll, c(2, 1, 3))
      dimnames(arr) <- list(iteration = NULL, chain = NULL,
                            variable = paste0("log_lik[",
                                              seq_len(dim(arr)[3]), "]"))
      da <- posterior::as_draws_array(arr)
      return(switch(format,
                    matrix = posterior::as_draws_matrix(da),
                    df = posterior::as_draws_df(da),
                    data.frame = posterior::as_draws_df(da),
                    da))
    }
    {
      da <- self$.draws_array
      if (!is.null(variables)) {
        all_vars <- posterior::variables(da)
        matched <- character(0)
        for (v in variables) {
          exact <- v %in% all_vars
          if (exact) {
            matched <- c(matched, v)
          } else {
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
    {
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
      do.call(posterior::summarise_draws, c(list(da), .summary_measures(..., .sampler = self$.sampler)))
    }
  }

  # ---- diagnostics() ----
  self$diagnostic_summary <- function(...) {
    if (!is.null(self$.diagnostics)) {
      self$.diagnostics
    } else {
      ef <- backend_fit$extra_fields
      nc <- self$.num_chains
      if (identical(self$.sampler, "drghmc")) {
        return(list(
          accept = if (!is.null(ef$accepted)) mean(ef$accepted) else NA_real_,
          dr_stage = if (!is.null(ef$dr_stage)) {
            s <- ef$dr_stage
            c(s1 = mean(s == 1), s2 = mean(s == 2),
              s3 = mean(s == 3), rej = mean(s < 0))
          } else NULL
        ))
      }
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
        num_max_treedepth = if (identical(self$.sampler, "hmc")) NULL else if (!is.null(ef$num_steps)) {
          max_steps <- 2^(if (!is.null(self$.max_treedepth)) self$.max_treedepth else 10L)
          vapply(seq_len(nc), function(ch) sum(as.integer(ef$num_steps[ch, ] >= max_steps)), integer(1))
        } else rep(0L, nc)
      )
    }
  }

  # ---- loo() ----
  self$loo <- function(variables = "log_lik", cores = 4, ...) {
    if (!requireNamespace("loo", quietly = TRUE)) {
      stop("Package 'loo' is required for LOO-CV.")
    }
    if (identical(variables, "log_lik") &&
        !isTRUE(self$.save_loglik) &&
        !any(grepl("^log_lik(\\[|$)", posterior::variables(self$.draws_array)))) {
      stop("This fit was saved with save_loglik = FALSE, so the ",
           "log-likelihood needed for loo() is not available. Refit with ",
           "save_loglik = TRUE (the default).", call. = FALSE)
    }
    if (identical(variables, "log_lik")) {
      return(loo.broc_fit(self, cores = cores, ...))
    }
    ll <- as.matrix(self$draws(variables = variables, format = "matrix"))
    r_eff <- loo::relative_eff(exp(ll), chain_id = rep(seq_len(self$.num_chains),
      each = self$.iter_sampling))
    loo::loo(ll, r_eff = r_eff, cores = cores, ...)
  }

  # ---- num_chains() ----
  self$num_chains <- function() {
    self$.num_chains
  }

  # ---- metadata() ----
  self$metadata <- function() {
    {
      list(iter_sampling = self$.iter_sampling,
           variables = if (!is.null(self$.draws_array))
             posterior::variables(self$.draws_array) else NULL)
    }
  }

  # ---- save_object() ----
  self$save_object <- function(file, ...) {
    save_broc_fit(self, file)
  }

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
#' `file =` in [fit_broc()]), not `saveRDS()`, which does not keep the fit's side files.
#'
#' @param fit A `broc_fit` object from [fit_broc()].
#' @param file Path to write to (use `.rds` extension by convention).
#' @param compress Compression for `saveRDS`. Default `FALSE` (fast; posterior
#'   draws barely compress). Set `"gzip"`/`"xz"` for a smaller, slower file.
#' @param verbose If `TRUE` (default), print a brief progress message.
#' @param include_loglik Whether to keep `log_lik` (needed for [loo()]). `NULL`
#'   (default) follows the fit's `fit_broc(save_loglik = )` setting; `TRUE`/`FALSE`
#'   overrides it for this save. When kept, the on-disk log-likelihood is written
#'   next to `file` as `<file>_loglik.npy`. A JAX fit also has its sampler state
#'   written as `<file>_state.pkl`, so [load_broc_fit()] in a later session returns a
#'   fit that `fit_broc(resume = )` accepts.
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
                          include_loglik = NULL,
                          re_store = c("f64", "f32", "int16", "int8")) {
  if (!inherits(fit, "broc_fit")) {
    stop("`fit` must be a broc_fit object.", call. = FALSE)
  }
  re_store <- if (missing(re_store) && !is.null(fit$.re_store)) fit$.re_store else match.arg(re_store)
  backend <- fit$.backend %||% "jax"
  keep_ll <- if (!is.null(include_loglik)) isTRUE(include_loglik)
             else isTRUE(fit$.save_loglik %||% TRUE)
  persist_side <- keep_ll

  if (verbose) {
    message("Saving broc_fit to ", file, "...")
  }
  t0 <- proc.time()["elapsed"]

  side <- NULL
  if (persist_side && !is.null(fit$.loglik_path) && file.exists(fit$.loglik_path)) {
    side <- paste0(tools::file_path_sans_ext(file), "_loglik.npy")
    if (!identical(normalizePath(fit$.loglik_path, winslash = "/", mustWork = FALSE),
                   normalizePath(side, winslash = "/", mustWork = FALSE))) {
      if (file.exists(side)) unlink(side)
      if (!isTRUE(suppressWarnings(file.link(fit$.loglik_path, side))) &&
          !isTRUE(file.copy(fit$.loglik_path, side, overwrite = TRUE))) {
        stop("Could not write the log-likelihood file '", side, "'.", call. = FALSE)
      }
    }
  }

  state_side <- NULL
  if (!is.null(fit$.state_path) && file.exists(fit$.state_path)) {
    state_side <- paste0(tools::file_path_sans_ext(file), "_state.pkl")
    if (!identical(normalizePath(fit$.state_path, winslash = "/", mustWork = FALSE),
                   normalizePath(state_side, winslash = "/", mustWork = FALSE))) {
      if (!isTRUE(file.copy(fit$.state_path, state_side, overwrite = TRUE))) {
        stop("Could not write the sampler-state file '", state_side, "'.", call. = FALSE)
      }
    }
  }
  fit_attrs <- attributes(fit)
  fit_attrs <- fit_attrs[setdiff(names(fit_attrs), c("class", "names", "broc_model"))]

  if (!keep_ll) {
    da <- fit$.draws_array
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
      sampler          = fit$.sampler %||% "nuts",
      draws_array      = da,
      save_loglik      = FALSE,
      loglik_sidecar   = NULL,
      diagnostics      = fit$diagnostic_summary(),
      loo              = fit$.loo
    )
  } else {
    bundle_backend_fit <- fit$.backend_fit
    bundle_backend_fit$samples <- NULL

    bundle <- list(
      .bayesroc_save_format = 1L,
      backend          = backend,
      backend_fit      = bundle_backend_fit,
      model            = attr(fit, "broc_model"),
      num_chains       = fit$.num_chains,
      iter_sampling    = fit$.iter_sampling,
      elapsed          = fit$.elapsed,
      max_treedepth    = fit$.max_treedepth %||% 10L,
      sampler          = fit$.sampler %||% "nuts",
      draws_array      = fit$.draws_array,
      save_loglik      = TRUE,
      loglik_sidecar   = if (is.null(side)) NULL else basename(side),
      loo              = fit$.loo
    )
  }
  if (!is.null(bundle$draws_array)) {
    sp <- .pack_re_draws(bundle$draws_array, re_store)
    bundle$draws_array <- sp$draws
    bundle$re_packed <- sp$packed
  }
  bundle$keep_re <- isTRUE(fit$.keep_re %||% TRUE)
  bundle$state_sidecar <- if (is.null(state_side)) NULL else basename(state_side)
  bundle$fit_attributes <- fit_attrs
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
    return(obj)
  }
  if (!inherits(obj, "broc_fit_saved") ||
      is.null(obj$.bayesroc_save_format)) {
    stop("File '", file, "' does not contain a broc_fit_saved bundle ",
         "(got class: ", paste(class(obj), collapse = "/"), ").",
         call. = FALSE)
  }
  side <- NULL
  if (!is.null(obj$loglik_sidecar)) {
    cand <- file.path(dirname(file), obj$loglik_sidecar)
    if (file.exists(cand)) {
      side <- cand
    } else {
      warning("The log-likelihood file '", obj$loglik_sidecar, "' saved with this fit ",
              "was not found next to '", basename(file), "'; loo() is unavailable ",
              "unless it is restored there.", call. = FALSE)
    }
  }
  fit <- new_broc_fit(
    backend_fit  = obj$backend_fit,
    model        = obj$model,
    backend      = obj$backend,
    num_chains   = obj$num_chains,
    iter_sampling = obj$iter_sampling,
    elapsed      = obj$elapsed,
    draws_array  = .unpack_re_draws(obj$draws_array, obj$re_packed),
    max_treedepth = obj$max_treedepth %||% 10L,
    sampler      = obj$sampler %||% "nuts",
    save_loglik  = obj$save_loglik %||% TRUE,
    diagnostics  = obj$diagnostics,
    loglik_path  = side,
    loglik_owned = FALSE,
    loo          = obj$loo,
    keep_re      = obj$keep_re %||% TRUE
  )
  if (is.null(side) && (!is.null(obj$loglik_sidecar) || !isTRUE(obj$save_loglik))) {
    fit$.loglik_dropped <- TRUE
  }
  if (!is.null(obj$state_sidecar)) {
    cand <- file.path(dirname(file), obj$state_sidecar)
    if (file.exists(cand)) {
      fit$.state_path <- normalizePath(cand, winslash = "/")
    } else {
      warning("The sampler-state file '", obj$state_sidecar, "' saved with this fit was not ",
              "found next to '", basename(file), "'; the fit cannot be resumed unless it is ",
              "restored there.", call. = FALSE)
    }
  }
  for (a in names(obj$fit_attributes)) attr(fit, a) <- obj$fit_attributes[[a]]
  fit
}

# =============================================================================
# Print Method
# =============================================================================

#' @export
print.broc_fit <- function(x, ...) {
  model <- attr(x, "broc_model")
  cat("bayesroc model fit\n")
  if (!is.null(model)) {
    cat("  Family:  ", family_display_name(model$family$family), " (", model$family$link$name, " link)\n", sep = "")
    fl <- .formula_lines(model)
    if (length(fl)) {
      cat("  Formula: ", fl[1], "\n", sep = "")
      for (f in fl[-1]) cat("           ", f, "\n", sep = "")
    }
    cat("  Data:    ", model$model_data$N, if (!is.null(model$stan_data$counts)) " rows of counts" else " observations",
        "\n", sep = "")
  }
  stride <- attr(x, "thin_stride") %||% 1L
  cat("  Draws:   ", x$.num_chains, " chains x ", x$.iter_sampling, " post-warmup",
      if (stride > 1L) paste0(", every ", .ordinal(stride), " of ", attr(x, "draws_per_chain")),
      " (", x$.backend, if (!is.null(x$.elapsed)) paste0(", ", round(x$.elapsed, 1), " s"), ")\n", sep = "")
  if (!is.null(attr(x, "stop_reason"))) {
    cat("  Stopped: ", switch(attr(x, "stop_reason"), targets = "ESS and R-hat targets met",
                              max_draws = "max_draws reached before the targets",
                              max_time = "max_time reached before the targets", draws = "requested draws taken"),
        "\n", sep = "")
  }
  d <- tryCatch(x$diagnostic_summary(), error = function(e) NULL)
  if (!is.null(d)) {
    cat("  ")
    .print_sampler_banner(d, x$.sampler %||% "nuts")
  }
  cat("Use summary(), plot(), plot_roc() or pp_check() to inspect the results.\n")
  invisible(x)
}

# =============================================================================
# NumPyro Samples Conversion
# =============================================================================

#' Convert NumPyro samples to posterior::draws_array
#'
#' Takes named list of arrays (each shape `[chains, draws, ...]`) and converts
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
  src <- if (is.environment(samples)) samples else NULL
  if (!is.null(src)) samples <- src$samples

  site_names <- names(samples)
  names_of <- function(name, dims) {
    if (is.null(dims) || length(dims) == 2) return(name)
    if (length(dims) == 3) return(paste0(name, "[", seq_len(dims[3]), "]"))
    paste0(name, "[", rep(seq_len(dims[3]), each = dims[4]), ",", rep(seq_len(dims[4]), times = dims[3]), "]")
  }
  var_list <- lapply(site_names, function(name) names_of(name, dim(samples[[name]])))
  var_names <- unlist(var_list, use.names = FALSE)
  n_params <- length(var_names)
  if (!is.null(src)) rm(samples)

  draws_3d <- array(NA_real_, dim = c(n_draws, n_chains, n_params))
  col_idx <- 0L

  for (i in seq_along(site_names)) {
    name <- site_names[i]
    arr <- if (is.null(src)) samples[[name]] else src$samples[[name]]
    if (!is.null(src)) src$samples[[name]] <- NULL
    dims <- dim(arr)
    if (is.null(dims)) arr <- matrix(arr, nrow = n_chains, ncol = n_draws, byrow = TRUE)
    nv <- length(var_list[[i]])
    block <- if (is.null(dims) || length(dims) == 2) t(arr)
             else if (length(dims) == 3) aperm(arr, c(2, 1, 3))
             else aperm(arr, c(2, 1, 4, 3))
    dim(block) <- c(n_draws, n_chains, nv)
    draws_3d[, , col_idx + seq_len(nv)] <- block
    col_idx <- col_idx + nv
  }

  dimnames(draws_3d) <- list(
    iteration = seq_len(n_draws),
    chain = seq_len(n_chains),
    variable = var_names
  )

  class(draws_3d) <- c("draws_array", "draws", "array")
  draws_3d
}

.jax_samples_as_stan <- function(samples, model) {
  nm <- names(samples)
  drop <- grepl("^sigma_batch_|^z_batch_|^beta_batch_", nm) |
    (grepl("^sigma_cross_", nm) & (grepl("\\[[0-9]+\\]$", nm) | grepl("_block_", nm)))
  samples <- samples[!drop]
  nm <- names(samples)
  out <- list()
  done <- character(0)
  for (k in nm) {
    if (k %in% done) next
    arr <- samples[[k]]
    m <- regmatches(k, regexec("^(sigma|z)_(.+)\\[([0-9]+)\\]$", k))[[1]]
    if (length(m)) {
      kind <- m[2]; base <- m[3]
      prefix <- paste0(kind, "_", base, "[")
      sibs <- nm[startsWith(nm, prefix) & grepl("^[0-9]+\\]$", substring(nm, nchar(prefix) + 1L))]
      sibs <- sibs[order(as.integer(sub("\\]$", "", substring(sibs, nchar(prefix) + 1L))))]
      done <- c(done, sibs)
      vector_stan <- length(sibs) > 1 || startsWith(base, "criterion_") || startsWith(base, "bias_")
      if (kind == "sigma") {
        if (vector_stan) {
          for (s in sibs) out[[s]] <- samples[[s]]
        } else {
          out[[paste0("sigma_", base)]] <- arr
        }
      } else {
        if (vector_stan) {
          st <- simplify2array(lapply(sibs, function(s) samples[[s]]))
          out[[paste0("u_", base, "_raw")]] <- st
        } else {
          out[[paste0("u_", base, "_raw")]] <- arr
        }
      }
      next
    }
    if (startsWith(k, "z_") && length(dim(arr)) == 4L) {
      out[[paste0("u_", substring(k, 3), "_raw")]] <- aperm(arr, c(1, 2, 4, 3))
      next
    }
    if (grepl("^beta_(log|raw)_gaps(_2)?$", k) && length(dim(arr)) == 3L) {
      dim(arr) <- c(dim(arr), 1L)
    }
    out[[k]] <- arr
  }
  if (!is.null(out$dprime_order_loggap)) {
    out$dprime_order_gap <- exp(out$dprime_order_loggap)
    out$dprime_order_loggap <- NULL
  }
  if (inherits(model, "bivariate_cumulative_model")) {
    names(out) <- gsub("criterion", "cutpoints", gsub("criterion2", "cutpoints2",
                  gsub("discrim", "mu2", gsub("dprime", "mu", names(out)))))
  } else if (inherits(model, "cumulative_model")) {
    names(out) <- gsub("criterion", "cutpoints", gsub("dprime", "mu", names(out)))
  }
  out
}
