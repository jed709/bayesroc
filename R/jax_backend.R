#' JAX/NumPyro Backend Interface
#'
#' R-side interface to fit SDT models via NumPyro.
#' Uses subprocess (system2) rather than reticulate to avoid DLL conflicts
#' in RStudio and other embedded R environments.

# =============================================================================
# Python Discovery
# =============================================================================

#' Enumerate candidate Python executables across common install locations
#'
#' Searches PATH, the Windows `Programs\\Python` dirs, and (on unix) conda envs,
#' pyenv/named venvs, and Homebrew. Returns existing executables, with Windows
#' Store / PsychoPy shims filtered out. Shared by the install-time and fit-time
#' Python resolvers so both look in the same places.
#' @return Character vector of candidate Python paths (may be empty).
#' @noRd
.python_candidate_paths <- function() {
  candidates <- character(0)

  if (.Platform$OS.type == "windows") {
    user_local <- Sys.getenv("LOCALAPPDATA",
                             unset = file.path(Sys.getenv("USERPROFILE"), "AppData", "Local"))
    py_base <- file.path(user_local, "Programs", "Python")
    if (dir.exists(py_base)) {
      py_dirs <- list.dirs(py_base, recursive = FALSE)
      py_dirs <- py_dirs[grepl("Python3\\d+$", basename(py_dirs))]
      py_dirs <- sort(py_dirs, decreasing = TRUE)
      candidates <- c(candidates, file.path(py_dirs, "python.exe"))
    }
  }

  if (.Platform$OS.type == "unix") {
    home <- Sys.getenv("HOME", unset = path.expand("~"))
    venv_patterns <- c("numpyro*", "jax*", ".venv", "venv", "pyenv")
    for (pat in venv_patterns) {
      venv_dirs <- Sys.glob(file.path(home, pat))
      for (vd in venv_dirs) {
        py_bin <- file.path(vd, "bin", "python3")
        if (file.exists(py_bin)) candidates <- c(candidates, py_bin)
        py_bin2 <- file.path(vd, "bin", "python")
        if (file.exists(py_bin2)) candidates <- c(candidates, py_bin2)
      }
    }
    conda_dirs <- c(
      file.path(home, "miniconda3", "envs"),
      file.path(home, "anaconda3", "envs"),
      file.path(home, "miniforge3", "envs"),
      file.path(home, "mambaforge", "envs"),
      file.path(home, ".conda", "envs")
    )
    for (cd in conda_dirs) {
      if (dir.exists(cd)) {
        env_dirs <- list.dirs(cd, recursive = FALSE)
        for (ed in env_dirs) {
          py_bin <- file.path(ed, "bin", "python3")
          if (file.exists(py_bin)) candidates <- c(candidates, py_bin)
        }
      }
    }
    brew_pythons <- Sys.glob("/opt/homebrew/opt/python@3.*/bin/python3.*")
    candidates <- c(candidates, brew_pythons)
    brew_pythons2 <- Sys.glob("/usr/local/opt/python@3.*/bin/python3.*")
    candidates <- c(candidates, brew_pythons2)
  }

  for (name in c("python3", "python")) {
    p <- Sys.which(name)
    if (nzchar(p)) {
      p <- normalizePath(p, mustWork = FALSE)
      candidates <- c(candidates, p)
    }
  }

  candidates <- unique(candidates)
  candidates <- candidates[file.exists(candidates)]
  candidates[!grepl("WindowsApps|PsychoPy", candidates, ignore.case = TRUE)]
}

#' Find a Python installation that has JAX
#'
#' Searches common locations and PATH. Caches result in an environment variable.
#' @return Path to Python executable
#' @noRd
find_python_with_jax <- function() {
  cached <- Sys.getenv("BAYESROC_PYTHON", unset = "")
  if (nzchar(cached) && file.exists(cached)) return(cached)

  managed <- managed_venv_python()
  if (file.exists(managed)) {
    ok <- tryCatch({
      res <- system2(managed, args = c("-c", shQuote("import jax, numpyro")),
                     stdout = FALSE, stderr = FALSE, timeout = 15)
      identical(res, 0L)
    }, error = function(e) FALSE, warning = function(w) FALSE)
    if (ok) {
      Sys.setenv(BAYESROC_PYTHON = managed)
      return(managed)
    }
  }

  retic <- Sys.getenv("RETICULATE_PYTHON", unset = "")
  if (nzchar(retic) && file.exists(retic)) {
    ok <- tryCatch({
      res <- system2(retic, args = c("-c", shQuote("import jax, numpyro")),
                     stdout = FALSE, stderr = FALSE, timeout = 15)
      identical(res, 0L)
    }, error = function(e) FALSE, warning = function(w) FALSE)
    if (ok) {
      Sys.setenv(BAYESROC_PYTHON = retic)
      return(retic)
    }
  }

  candidates <- .python_candidate_paths()

  for (py in candidates) {
    ok <- tryCatch({
      res <- system2(py, args = c("-c", shQuote("import jax, numpyro")),
                     stdout = FALSE, stderr = FALSE, timeout = 15)
      identical(res, 0L)
    }, error = function(e) FALSE, warning = function(w) FALSE)
    if (ok) {
      Sys.setenv(BAYESROC_PYTHON = py)
      return(py)
    }
  }

  stop("No Python with JAX + NumPyro found. Either:\n",
       "  1. Run `bayesroc::install_jax_backend()` for an isolated, pinned setup, or\n",
       "  2. Set BAYESROC_PYTHON / RETICULATE_PYTHON to a Python with both packages, or\n",
       "  3. Run `bayesroc::check_jax_backend()` for a diagnosis.",
       call. = FALSE)
}

.bayesroc_source_root <- local({
  root <- NULL
  for (i in seq_len(sys.nframe())) {
    ofile <- tryCatch(get("ofile", envir = sys.frame(i)), error = function(e) NULL)
    if (!is.null(ofile)) {
      candidate <- dirname(dirname(normalizePath(ofile, mustWork = FALSE)))
      if (file.exists(file.path(candidate, "inst", "python", "numpyro_backend.py"))) {
        root <- candidate
        break
      }
    }
  }
  root
})

#' Find the numpyro_backend.py script
#' @noRd
find_backend_script <- function() {
  py_script <- system.file("python", "numpyro_backend.py", package = "bayesroc")
  if (nzchar(py_script) && file.exists(py_script)) return(py_script)

  search_roots <- c(
    Sys.getenv("BAYESROC_PACKAGE_ROOT", unset = ""),
    .bayesroc_source_root,
    getwd(),
    dirname(getwd())
  )
  for (root in search_roots) {
    if (!is.null(root) && nzchar(root)) {
      candidate <- file.path(root, "inst", "python", "numpyro_backend.py")
      if (file.exists(candidate)) {
        return(normalizePath(candidate, mustWork = FALSE))
      }
    }
  }
  stop("Cannot find numpyro_backend.py. Set BAYESROC_PACKAGE_ROOT to the package directory.")
}

# =============================================================================
# Config Builder
# =============================================================================

.cross_member_order <- function(re_configs, cross_cor_groups) {
  for (cid in names(cross_cor_groups)) {
    slots <- which(vapply(re_configs, function(r) identical(r$cor_id, cid), logical(1)))
    if (length(slots) < 2L) next
    key <- vapply(re_configs[slots], function(r) paste(r$param, r$group), "")
    want <- vapply(cross_cor_groups[[cid]]$members, function(m) paste(m$param, m$group), "")
    ord <- match(want, key)
    if (anyNA(ord) || length(ord) != length(slots)) {
      stop("internal: cross-parameter group '", cid, "' members do not match the random-effect configs", call. = FALSE)
    }
    re_configs[slots] <- re_configs[slots][ord]
  }
  re_configs
}

#' Build NumPyro config dict from broc_model
#'
#' Translates the R-side broc_model into a flat config dictionary that
#' the Python numpyro_backend.py can consume.
#'
#' @param model An broc_model object
#' @return A list suitable for JSON serialization
#' @noRd
build_numpyro_config <- function(model) {
  md <- model$model_data
  family <- model$family
  family_name <- family$family
  link_name <- family$link$name
  parsed <- model$parsed

  config <- list(
    family = family_name,
    link = link_name,
    N = md$N,
    K = md$K,
    y = as.integer(md$stan_data$y),
    is_old = if (!is.null(md$stan_data$is_old)) as.numeric(md$stan_data$is_old) else NULL
  )

  if (!is.null(md$K2)) config$K2 <- md$K2
  if (!is.null(md$stan_data$item_type)) config$item_type <- as.integer(md$stan_data$item_type)
  if (!is.null(md$stan_data$y2)) config$y2 <- as.integer(md$stan_data$y2)
  if (!is.null(md$stan_data$rk)) config$rk <- as.integer(md$stan_data$rk)
  if (!is.null(md$stan_data$J)) config$J <- md$stan_data$J
  if (!is.null(md$stan_data$n_rkg)) config$n_rkg <- md$stan_data$n_rkg
  if (!is.null(md$stan_data$n_new)) config$n_new <- md$stan_data$n_new
  if (!is.null(md$stan_data$old_level_map)) config$old_level_map <- as.integer(md$stan_data$old_level_map)
  if (!is.null(md$stan_data$new_level_map)) config$new_level_map <- as.integer(md$stan_data$new_level_map)

  if (!is.null(md$stan_data$source)) config$source <- as.integer(md$stan_data$source)

  config$has_lure_mixture <- isTRUE(md$has_lure_mixture)

  if (!is.null(md$stan_data$counts)) {
    config$counts <- as.numeric(md$stan_data$counts)
  }

  # ---- Fixed effects for each parameter ----
  config$params <- list()
  prior_lookup <- md$prior_lookup

  all_params <- c("dprime", "criterion", "sigma", "lambda", "dprime2", "sigma2",
                  "dprime_B", "lambda_B", "discrim", "discrim_B",
                  "sigma_B", "sigma2_B", "rho", "rho_B", "rho_N",
                  "criterion2", "lambda2", "lambda2_B",
                  "rec_crit", "know_crit",
                  "dprime_L", "sigma_L", "lambda_L", "criterion_scale", "criterion2_scale")

  # the sampler centres each design's non-intercept columns (the intercept absorbs them; see
  # .centre_info); not for an ordered d' pair or a know criterion under the slack transform,
  # whose intercepts carry constraints of their own
  skip_centre <- c(if (isTRUE(md$needs_ordered_dprime)) c("dprime", "dprime2"),
                   if (isTRUE(md$stan_data$kc_knob == 1L)) "know_crit", "criterion_scale", "criterion2_scale")
  for (param in all_params) {
    fe_key <- paste0(param, "_fixed")
    fe <- md[[fe_key]]
    if (!is.null(fe)) {
      param_link <- if (!is.null(family$params[[param]]$link)) family$params[[param]]$link else "identity"

      col_names <- fe$coef_names %||% fe$col_names
      if (length(col_names) == 0 || all(!nzchar(col_names))) {
        col_names <- paste0("V", seq_len(fe$n_coef))
      }

      pconf <- list(
        name = param,
        link = param_link,
        n_coef = fe$n_coef,
        X = as.matrix(fe$X),
        col_names = col_names
      )

      pconf$priors <- list()
      for (j in seq_len(fe$n_coef)) {
        coef_name <- col_names[j]
        pconf$priors[[coef_name]] <- get_fixed_prior(prior_lookup, param, coef_name)
      }

      pconf$centre <- .centre_info(fe$X, col_names, skip = param %in% skip_centre)
      config$params[[param]] <- pconf
    }
  }

  # ---- Threshold parameterization ----
  if (family_name == "cdp") {
    n_thresh <- config$J
  } else {
    n_thresh <- config$K - 1L
  }
  config$n_thresh <- n_thresh
  config$thresh_mid_idx <- ceiling(n_thresh / 2)

  gaps_key <- "log_gaps"
  gaps2_key <- "log_gaps2"
  config$prior_thresh_mid <- get_fixed_prior(prior_lookup, "thresh_mid")
  config$prior_log_gaps <- get_fixed_prior(prior_lookup, gaps_key)
  n_gaps_jax <- max(0L, n_thresh - 1L)
  if (n_gaps_jax > 0) {
    config$prior_log_gaps_per_gap <- vapply(seq_len(n_gaps_jax), function(k)
      get_fixed_prior(prior_lookup, gaps_key, coef = sprintf("[%d]", k)),
      character(1))
  }
  tg <- .thresh_prior_grid(prior_lookup, "thresh_mid", "log_gaps", md$criterion$coef_names, md$K - 2L)
  config$prior_thresh_mid_grid <- as.list(tg$mid)
  config$prior_log_gaps_grid <- lapply(seq_len(nrow(tg$gaps)), function(k) as.list(tg$gaps[k, ]))
  if (isTRUE(md$has_criterion2)) {
    tg2 <- .thresh_prior_grid(prior_lookup, "thresh_mid2", "log_gaps2", md$criterion2$coef_names, md$K2 - 2L)
    config$prior_thresh_mid2_grid <- as.list(tg2$mid)
    config$prior_log_gaps2_grid <- lapply(seq_len(nrow(tg2$gaps)), function(k) as.list(tg2$gaps[k, ]))
    if (isTRUE(md$varying_source_criteria)) {
      vg <- .crit2_vary_prior_grid(prior_lookup, md)
      config$prior_vary2 <- list(mid = as.list(vg$mid),
                                 gaps = lapply(seq_len(ncol(vg$gaps)), function(s) as.list(vg$gaps[, s])),
                                 cov_mid = as.list(vg$cov_mid),
                                 cov_gaps = lapply(seq_len(nrow(vg$cov_gaps)), function(k) as.list(vg$cov_gaps[k, ])))
    }
  }
  config$prior_thresh_mid2 <- get_fixed_prior(prior_lookup, "thresh_mid2")
  config$prior_log_gaps2 <- get_fixed_prior(prior_lookup, gaps2_key)
  n_gaps2_jax <- if (!is.null(md$K2)) max(0L, md$K2 - 2L) else 0L
  if (n_gaps2_jax > 0) {
    config$prior_log_gaps2_per_gap <- vapply(seq_len(n_gaps2_jax), function(k)
      get_fixed_prior(prior_lookup, gaps2_key, coef = sprintf("[%d]", k)),
      character(1))
  }

  crit <- md$criterion
  if (!is.null(crit)) {
    config$P_crit <- crit$n_coef
    config$X_crit <- as.matrix(crit$X)
    config$crit_col_names <- crit$col_names
    config$crit_centre <- .centre_info(crit$X, crit$coef_names %||% colnames(crit$X))
  } else {
    config$P_crit <- 1L
    config$X_crit <- matrix(1, nrow = config$N, ncol = 1)
    config$crit_col_names <- "Intercept"
  }

  crit2 <- md$criterion2
  if (isTRUE(md$varying_source_criteria) && isTRUE(md$stan_data$P_criterion2_cov > 0)) {
    config$crit2_cov <- list(X = as.matrix(md$stan_data$X_criterion2_cov),
                             col_names = as.list(crit2$cov_names))
  }
  if (!isTRUE(md$varying_source_criteria) &&
      !is.null(crit2) && !is.null(crit2$X) && isTRUE(crit2$n_coef > 1L)) {
    c2_cols <- crit2$col_names
    if (length(c2_cols) == 0 || all(!nzchar(c2_cols))) {
      c2_cols <- colnames(crit2$X)
    }
    if (length(c2_cols) == 0) c2_cols <- paste0("V", seq_len(crit2$n_coef))
    c2_conf <- list(
      name = "criterion2",
      link = "identity",
      n_coef = crit2$n_coef,
      X = as.matrix(crit2$X),
      col_names = c2_cols
    )
    c2_conf$priors <- list()
    for (j in seq_len(crit2$n_coef)) {
      c2_conf$priors[[c2_cols[j]]] <- config$prior_thresh_mid2
    }
    c2_conf$centre <- .centre_info(crit2$X, c2_cols)
    config$params[["criterion2"]] <- c2_conf
  }

  # ---- Random effects ----
  config$random_effects <- list()

  re_params <- c("dprime", "criterion", "sigma", "lambda", "dprime2", "sigma2",
                 "dprime_B", "lambda_B", "discrim", "discrim_B",
                 "sigma_B", "sigma2_B", "rho", "rho_B", "rho_N",
                 "criterion2", "lambda2", "lambda2_B", "rec_crit", "know_crit",
                 "dprime_L", "sigma_L", "lambda_L", "criterion_scale", "criterion2_scale")

  for (param in re_params) {
    re_key <- paste0(param, "_random")
    re_list <- md[[re_key]]
    if (is.null(re_list) && param == "criterion" && !is.null(md$criterion$random)) {
      re_list <- md$criterion$random
    }
    if (is.null(re_list) && param == "criterion2" && !is.null(md$criterion2$random)) {
      re_list <- md$criterion2$random
    }
    if (!is.null(re_list) && length(re_list) > 0) {
      for (re in re_list) {
        re_conf <- list(
          param = param,
          group = re$group,
          n_groups = re$n_groups,
          n_terms = if (!is.null(re$dim)) re$dim else if (!is.null(re$n_re_terms)) re$n_re_terms else 1L,
          correlated = isTRUE(re$correlated),
          term_names = re$level_names
        )

        re_conf$index <- as.integer(re$group_idx)
        if (isTRUE(re$use_z_matrix) && !is.null(re$Z)) {
          re_conf$type <- "matrix"
          re_conf$Z <- as.matrix(re$Z)
        } else {
          re_conf$type <- "index"
          if (!is.null(re$term_idx)) {
            re_conf$term_idx <- as.integer(re$term_idx)
          }
        }

        if (!is.null(re$cor_id)) {
          re_conf$cor_id <- re$cor_id
        }
        if (isTRUE(re$correlated) && !.re_full_block(re)) {
          re_conf$cor_blocks <- lapply(re$cor_blocks, function(b) as.integer(b - 1L))
        }

        re_conf$prior_sd <- get_sd_prior(prior_lookup, param, re$group)
        re_conf$prior_cor <- get_cor_prior(prior_lookup, param, re$group)
        if (!is.null(re$cor_id)) {
          re_conf$prior_cor <- .cross_cor_prior(prior_lookup, re$cor_id, re$group)
          per <- .re_sd_priors(prior_lookup, param, re$group, re)
          if (length(per) == re_conf$n_terms) {
            re_conf$prior_sd_per_coef <- as.list(per)
            if (length(unique(per)) == 1L) re_conf$prior_sd <- per[1]
          }
        } else if (!param %in% c("criterion", "criterion2")) {
          per <- .re_sd_priors(prior_lookup, param, re$group, re)
          if (length(per) == re_conf$n_terms) {
            re_conf$prior_sd_per_coef <- per
            if (length(unique(per)) == 1L) re_conf$prior_sd <- per[1]
          }
        }

        if (param %in% c("criterion", "criterion2") && is.null(re$cor_id)) {
          per <- .crit_re_sd_priors(prior_lookup, param, re$group, re)
          if (length(per) > 1L) re_conf$prior_sd_per_coef <- per
          if (length(unique(per)) == 1L) re_conf$prior_sd <- per[1]
        }

        if (param == "criterion" && isTRUE(md$has_criterion_scale)) re_conf$location_only <- TRUE
        if (param == "criterion2" && isTRUE(md$has_criterion2_scale) && !isTRUE(md$varying_source_criteria))
          re_conf$location_only <- TRUE
        config$random_effects[[length(config$random_effects) + 1]] <- re_conf
      }
    }
  }

  if (!is.null(md$smooth_data)) {
    config$smooths <- list()
    for (pname in names(md$smooth_data)) {
      for (sm in md$smooth_data[[pname]]) {
        for (j in seq_along(sm$components)) {
          comp <- sm$components[[j]]
          comp_suffix <- comp$san_label
          n_thresh <- if (!is.null(sm$n_thresh)) sm$n_thresh else 0L
          for (k in seq_along(comp$Zs_list)) {
            sm_conf <- list(
              param = pname,
              label = comp_suffix,
              component = k,
              Zs = as.matrix(comp$Zs_list[[k]]),
              dim = ncol(comp$Zs_list[[k]]),
              n_thresh = n_thresh,
              prior_sds = get_sds_prior(prior_lookup, pname, paste0(comp_suffix, "_", k))
            )
            base <- paste0(pname, "_", comp_suffix, "_", k)
            if (base %in% md$smooth_info_bases) {
              I <- md$stan_data[[paste0("Ism_", base)]]
              sm_conf$info <- if (n_thresh > 0) lapply(seq_len(n_thresh), function(t) I[t, , , drop = TRUE]) else I
            }
            config$smooths[[length(config$smooths) + 1]] <- sm_conf
          }
        }
      }
    }
  }

  if (!is.null(md$cross_cor_groups)) {
    config$cross_cor_groups <- md$cross_cor_groups
  }
  if (length(md$cross_cor)) {
    config$random_effects <- .cross_member_order(config$random_effects, md$cross_cor)
  }

  if (!is.null(md$stan_data$is_old)) {
    config$is_old <- as.numeric(md$stan_data$is_old)
  }

  config$varying_source_criteria <- isTRUE(family$varying_source_criteria)
  if (isTRUE(md$needs_ordered_dprime)) {
    config$needs_ordered_dprime <- TRUE
    config$dprime_pivot <- as.integer(md$stan_data$dprime_pivot)
    config$dprime_xbar <- as.list(as.numeric(md$stan_data$dprime_xbar))
    config$dprime2_xbar <- as.list(as.numeric(md$stan_data$dprime2_xbar))
  }
  if (!is.null(md$bounded)) config$bounded <- md$bounded
  if (!is.null(md$stan_data$vbin)) {
    config$vbin <- as.list(as.integer(md$stan_data$vbin))
    config$n_vbin <- as.integer(md$stan_data$n_vbin)
    config$n_vcell <- .n_vcell(md)
    config$vcell <- as.list(.vcell_of(md))
  }
  if (!is.null(md$gap_link)) config$gap_link <- md$gap_link
  if (isTRUE(md$has_criterion_scale)) config$crit_ls <- TRUE
  if (isTRUE(md$has_criterion2_scale)) {
    config$crit2_ls <- TRUE
    if (isTRUE(md$varying_source_criteria)) config$crit2_re_mode <- "per_bin"
  }
  cs <- getOption("bayesroc.crit_scalar")
  if (isTRUE(cs)) config$crit_scalar <- TRUE
  if (is.character(cs)) config$crit_scalar_groups <- as.list(cs)
  wh <- getOption("bayesroc.whiten")
  if (is.list(wh) && length(config$random_effects)) {
    for (i in seq_along(config$random_effects)) {
      re <- config$random_effects[[i]]
      key <- paste0(re$param, "_", re$group)
      if (!is.null(wh[[key]])) {
        W <- as.matrix(wh[[key]])
        config$random_effects[[i]]$whiten <- lapply(seq_len(nrow(W)), function(r) as.numeric(W[r, ]))
      }
    }
  }

  .check_jax_priors(config)
  config
}

.check_jax_priors <- function(config) {
  s <- unlist(rapply(config, function(x) x, classes = "character", how = "unlist"), use.names = FALSE)
  nm <- sub("[[:space:]]*[(].*$", "", s[grepl("^[[:space:]]*[a-z_]+[[:space:]]*[(]", s)])
  bad <- unique(nm[nm %in% names(.stan_only_dists)])
  if (length(bad)) {
    stop("The prior distribution", if (length(bad) > 1) "s", " ", paste0(bad, "()", collapse = ", "),
         if (length(bad) > 1) " are" else " is", " only available with backend = \"stan\".", call. = FALSE)
  }
  invisible(NULL)
}

# =============================================================================
# Main Fitting Function (subprocess-based)
# =============================================================================

#' Fit an SDT model using NumPyro via subprocess
#'
#' Runs Python as a separate process to avoid DLL conflicts in RStudio.
#' Config is passed via JSON temp file; results returned via numpy .npz.
#'
#' @param model An broc_model object
#' @param chains Number of MCMC chains
#' @param warmup Number of warmup iterations
#' @param samples Number of post-warmup samples per chain
#' @param init Initial values (0 for zeros, numeric for uniform range)
#' @param seed Random seed (NULL for random)
#' @param loglik_precision "f32" or "f64" for the log_lik sidecar
#' @param keep_re Retain per-level random-effect draws
#' @param save_state Return the end-of-run sampler state (for `resume =`)
#' @return A list with samples, extra_fields, elapsed, loglik_path, state_path
#' @noRd
fit_jax <- function(model, chains = 4L, warmup = 2000L, samples = 2000L,
                    thin = 1L, init = 0, init_strategy = NULL, seed = NULL,
                    adapt_delta = 0.8, max_treedepth = 10L, step_size = NULL,
                    sampler = "nuts", control = list(), save_loglik = TRUE,
                    chain_method = "parallel", python = NULL,
                    progress_bar = TRUE, loglik_precision = "f32",
                    keep_re = FALSE, save_state = TRUE, init_values = NULL, total_cores = NULL) {

  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("Package 'jsonlite' is required for the JAX backend.")
  }

  if (!is.null(python)) {
    if (!file.exists(python)) {
      stop("python = '", python, "' does not exist.", call. = FALSE)
    }
  } else {
    python <- find_python_with_jax()
  }
  py_script <- find_backend_script()
  config <- build_numpyro_config(model)

  if (is.null(seed)) seed <- sample.int(.Machine$integer.max, 1)

  tmp_dir <- tempfile("numpyro_")
  dir.create(tmp_dir)
  on.exit(unlink(tmp_dir, recursive = TRUE), add = TRUE)
  config_file <- file.path(tmp_dir, "config.json")
  result_file <- file.path(tmp_dir, "results.json")

  if (is.null(init_strategy)) init_strategy <- "numeric"
  fit_args <- list(
    config = config,
    fit_kwargs = list(
      chains = as.integer(chains),
      warmup = as.integer(warmup),
      samples = as.integer(samples),
      seed = as.integer(seed),
      target_accept_prob = as.numeric(adapt_delta),
      max_tree_depth = as.integer(max_treedepth),
      step_size = if (is.null(step_size)) 1.0 else as.numeric(step_size),
      thinning = as.integer(thin),
      chain_method = chain_method,
      progress_bar = isTRUE(progress_bar),
      init_strategy = init_strategy,
      init_radius = if (is.numeric(init)) as.numeric(init) else 2.0,
      sampler = sampler,
      control = if (length(control)) control else NULL,
      save_loglik = isTRUE(save_loglik),
      keep_re = isTRUE(keep_re),
      save_state = isTRUE(save_state),
      init_values = init_values,
      total_cores = total_cores
    )
  )

  # matrices go through jsonlite's own row-major encoding, which is much faster than a list of rows
  jsonlite::write_json(fit_args, config_file, auto_unbox = TRUE,
                       digits = 17, null = "null")

  chain_note <- if (identical(chain_method, "sequential")) ", sequential" else ""
  if (identical(sampler, "drghmc")) {
    message("Fitting with DR-G-HMC via JAX (", chains, " chains", chain_note, ", ",
            warmup, " adapt, ", samples, " samples)...")
  } else {
    message("Fitting with NumPyro NUTS via JAX (", chains, " chains", chain_note, ", ",
            warmup, " warmup, ", samples, " samples)...")
  }

  if (!requireNamespace("processx", quietly = TRUE)) {
    stop("Package 'processx' is required for the JAX backend.\n",
         "Install it with: install.packages('processx')")
  }
  env <- .jax_env(python, chains, chain_method, sampler, control)
  proc <- processx::process$new(
    python, args = c(file.path(dirname(py_script), "worker.py"), config_file, tmp_dir),
    env = env, wd = dirname(py_script),
    stdout = "|", stderr = "|",
    cleanup = TRUE, cleanup_tree = TRUE
  )
  on.exit({
    if (proc$is_alive()) {
      message("\nKilling NumPyro subprocess...")
      proc$kill_tree()
    }
  }, add = TRUE)
  .stream_process(proc)
  exit_code <- proc$get_exit_status()
  if (exit_code != 0) {
    stop("NumPyro fitting failed (exit code ", exit_code, ").\n",
         "Check the Python output above for details.")
  }

  if (!file.exists(result_file)) {
    stop("NumPyro fitting did not produce results. Check Python output.")
  }

  .read_jax_results(tmp_dir, loglik_precision)
}

.read_jax_results <- function(dir, loglik_precision = "f32") {
  result_meta <- jsonlite::read_json(file.path(dir, "results.json"))

  loglik_path <- NULL
  samples_list <- list()
  for (name in names(result_meta$samples)) {
    npy_file <- file.path(dir, result_meta$samples[[name]])
    if (identical(name, "log_lik")) {
      loglik_path <- tempfile("bayesroc_ll_", fileext = ".npy")
      .emit_loglik_side(npy_file, loglik_path, loglik_precision)
      next
    }
    samples_list[[name]] <- read_npy(npy_file)
  }

  extra_fields <- list()
  for (name in names(result_meta$extra_fields)) {
    extra_fields[[name]] <- read_npy(file.path(dir, result_meta$extra_fields[[name]]))
  }

  state_path <- NULL
  if (!is.null(result_meta$state) && nzchar(result_meta$state)) {
    src <- file.path(dir, result_meta$state)
    if (file.exists(src)) {
      state_path <- tempfile("bayesroc_state_", fileext = ".pkl")
      if (!isTRUE(file.copy(src, state_path, overwrite = TRUE))) {
        stop("Could not copy the sampler state to '", state_path, "'.", call. = FALSE)
      }
    }
  }

  list(
    samples = samples_list,
    extra_fields = extra_fields,
    elapsed = result_meta$elapsed,
    loglik_path = loglik_path,
    state_path = state_path,
    meta = result_meta
  )
}

# =============================================================================
# Minimal .npy Reader
# =============================================================================

#' Read a numpy .npy file into an R array
#'
#' Supports float32, float64, int32, int64, bool arrays.
#' @param path Path to .npy file
#' @return R array
#' @noRd
read_npy <- function(path) {
  con <- file(path, "rb")
  on.exit(close(con))

  magic <- readBin(con, "raw", 6)
  version <- readBin(con, "integer", 1, size = 2, endian = "little")

  if (version == 1L) {
    header_len <- readBin(con, "integer", 1, size = 2, endian = "little")
  } else {
    header_len <- readBin(con, "integer", 1, size = 4, endian = "little")
  }

  header_str <- rawToChar(readBin(con, "raw", header_len))

  dtype_match <- regmatches(header_str, regexec("'descr':\\s*'([^']+)'", header_str))[[1]]
  dtype <- dtype_match[2]

  shape_match <- regmatches(header_str, regexec("'shape':\\s*\\(([^)]+)\\)", header_str))[[1]]
  shape_str <- shape_match[2]
  shape <- as.integer(strsplit(trimws(gsub(",\\s*$", "", shape_str)), ",\\s*")[[1]])
  if (length(shape) == 0 || any(is.na(shape))) shape <- 1L

  fortran <- grepl("'fortran_order':\\s*True", header_str)

  n_elem <- prod(shape)

  if (grepl("f8|float64", dtype)) {
    data <- readBin(con, "double", n_elem, size = 8, endian = "little")
  } else if (grepl("f4|float32", dtype)) {
    data <- readBin(con, "double", n_elem, size = 4, endian = "little")
  } else if (grepl("i8|int64", dtype)) {
    raw_data <- readBin(con, "raw", n_elem * 8)
    data <- readBin(raw_data, "double", n_elem, size = 8, endian = "little")
    data <- as.numeric(readBin(raw_data, "integer", n_elem * 2, size = 4, endian = "little"))
    data <- data[seq(1, length(data), 2)] + data[seq(2, length(data), 2)] * 2^32
  } else if (grepl("i4|int32", dtype)) {
    data <- readBin(con, "integer", n_elem, size = 4, endian = "little")
  } else if (grepl("b1|bool", dtype)) {
    data <- as.logical(readBin(con, "integer", n_elem, size = 1, endian = "little"))
  } else {
    data <- readBin(con, "double", n_elem, size = 8, endian = "little")
  }

  if (length(shape) == 1) {
    return(data)
  } else {
    arr <- array(data, dim = rev(shape))
    return(aperm(arr, rev(seq_along(shape))))
  }
}

# =============================================================================
# Helper
# =============================================================================

.xla_cache_dir <- function() {
  opt <- getOption("bayesroc.xla_cache", TRUE)
  if (isFALSE(opt)) return("0")
  if (is.character(opt) && nzchar(opt)) return(normalizePath(opt, winslash = "/", mustWork = FALSE))
  normalizePath(file.path(tools::R_user_dir("bayesroc", "cache"), "xla"), winslash = "/", mustWork = FALSE)
}

.jax_init_values <- function(init, chains) {
  if (is.function(init)) {
    init <- lapply(seq_len(chains), function(ch) {
      if ("chain_id" %in% names(formals(init))) init(chain_id = ch) else init()
    })
  }
  if (length(init) != chains || !all(vapply(init, is.list, logical(1)))) {
    stop("`init` must be a function or a list of ", chains,
         " per-chain named lists.", call. = FALSE)
  }
  lapply(init, function(ch) {
    if (length(ch) && is.null(names(ch))) {
      stop("Each per-chain `init` list must be named by parameter.", call. = FALSE)
    }
    lapply(ch, function(v) if (is.array(v) && length(dim(v)) > 2L)
      stop("`init` arrays with more than 2 dimensions are not supported.", call. = FALSE)
      else v)
  })
}

# =============================================================================
# Subprocess Environment and Persistent Worker
# =============================================================================

.jax_env <- function(python, chains, chain_method = "parallel", sampler = "nuts",
                     control = list(), precision = NULL) {
  device_count <- if (identical(chain_method, "sequential")) 1L else as.integer(chains)
  f32 <- identical(precision %||% control$precision, "f32") && identical(sampler, "nuts")
  bvn_lib <- .bvn_ffi_lib(python)
  c("current",
    XLA_FLAGS = paste0("--xla_force_host_platform_device_count=", device_count),
    TF_CPP_MIN_LOG_LEVEL = "2",   # XLA's C++ log lines (cache loads, slow-compile alarms) are noise to the user
    BAYESROC_XLA_CACHE = .xla_cache_dir(),
    BAYESROC_BVN_FFI = if (is.null(bvn_lib)) "" else bvn_lib,
    BAYESROC_ENABLE_X64 = if (f32) "0" else "1",
    BAYESROC_PROGRESS = "events")
}

.stream_process <- function(proc) {
  .prog_reset()
  on.exit(.prog_done(), add = TRUE)
  while (proc$is_alive()) {
    proc$poll_io(200)
    .prog_out(proc$read_output())
    .prog_err(proc$read_error())
  }
  .prog_out(proc$read_output())
  .prog_err(proc$read_error())
}

.jax_worker_start <- function(python = NULL, chains = 4L, chain_method = "parallel",
                              sampler = "nuts", control = list()) {
  python <- python %||% find_python_with_jax()
  py_dir <- dirname(find_backend_script())
  ctrl <- tempfile("bayesroc_worker_")
  dir.create(ctrl)
  proc <- processx::process$new(
    python, args = c(file.path(py_dir, "worker.py"), "--serve", ctrl),
    env = .jax_env(python, chains, chain_method, sampler, control), wd = py_dir,
    stdout = "|", stderr = "|", cleanup = TRUE, cleanup_tree = TRUE)
  structure(list(proc = proc, ctrl = ctrl, python = python), class = "bayesroc_worker")
}

.jax_worker_stop <- function(worker) {
  if (!inherits(worker, "bayesroc_worker")) return(invisible(NULL))
  try(file.create(file.path(worker$ctrl, "stop")), silent = TRUE)
  try(worker$proc$wait(timeout = 5000), silent = TRUE)
  if (worker$proc$is_alive()) worker$proc$kill_tree()
  unlink(worker$ctrl, recursive = TRUE)
  invisible(NULL)
}

.jax_worker_run <- function(worker, config_file, out_dir, token = NULL, op = "fit") {
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  req <- file.path(worker$ctrl, paste0("req_", basename(out_dir), ".json"))
  jsonlite::write_json(list(config = normalizePath(config_file, winslash = "/"),
                            out = normalizePath(out_dir, winslash = "/"),
                            token = token, op = op),
                       paste0(req, ".tmp"), auto_unbox = TRUE, null = "null")
  file.rename(paste0(req, ".tmp"), req)
  done <- file.path(out_dir, "DONE")
  errf <- file.path(out_dir, "ERROR")
  # an interrupt ends the fit: tell the worker (it leaves at once, mid-computation) and
  # kill its tree before unwinding, so no later cleanup has to wait on it
  withCallingHandlers(
    repeat {
      worker$proc$poll_io(100)
      .prog_out(worker$proc$read_output())
      .prog_err(worker$proc$read_error())
      if (file.exists(done)) break
      if (file.exists(errf)) {
        .prog_done()
        stop("The JAX worker failed:\n", paste(utils::tail(readLines(errf), 12), collapse = "\n"),
             call. = FALSE)
      }
      if (!worker$proc$is_alive()) {
        .prog_done()
        stop("The JAX worker process exited before returning a result.", call. = FALSE)
      }
    },
    interrupt = function(e) {
      .prog_done()
      try(file.create(file.path(worker$ctrl, "stop")), silent = TRUE)
      try(worker$proc$kill_tree(), silent = TRUE)
    })
  invisible(out_dir)
}


# The sampler's coordinates for a design with an intercept and other columns: the intercept it
# samples is the value at the columns' means, b0_c = b0 + sum_j m_j b_j, an exact linear map that
# removes the intercept-slope ridge (and, through the map's couplings, the one between a
# between-group coefficient and the nested groups' effects). Means are taken over the rows the
# intercept column switches on, so encoded designs (lure rows all zero) centre their old rows only.
# Stored as list(col, means = list(m)): the nested list keeps `_trim_config` from slicing it.
.centre_info <- function(X, col_names, skip = FALSE) {
  X <- as.matrix(X)
  if (skip || ncol(X) < 2L) return(NULL)
  nm <- colnames(X) %||% col_names
  j <- match("(Intercept)", nm)
  if (is.na(j)) {
    # cell-means designs: the 0/1 columns that partition the rows act as the intercept jointly
    bin <- which(apply(X, 2, function(v) all(v %in% c(0, 1))))
    if (!length(bin) || length(bin) == ncol(X)) return(NULL)
    S <- rowSums(X[, bin, drop = FALSE])
    if (any(S > 1)) return(NULL)
    on <- S == 1
    j <- bin
  } else {
    on <- X[, j] == 1
  }
  if (!any(on)) return(NULL)
  m <- colMeans(X[on, , drop = FALSE])
  m[j] <- 0
  if (all(abs(m) < 1e-12)) return(NULL)
  list(col = as.list(unname(j - 1L)), means = list(unname(m)))
}

.centre_all <- function(model) {
  cfg <- tryCatch(build_numpyro_config(model), error = function(e) NULL)
  if (is.null(cfg)) return(NULL)
  list(params = lapply(cfg$params, `[[`, "centre"), crit = cfg$crit_centre)
}
