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

  # Windows common locations (newest first)
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

  # macOS / Linux: check common virtualenv and conda locations in home dir
  if (.Platform$OS.type == "unix") {
    home <- Sys.getenv("HOME", unset = path.expand("~"))
    # Named venvs in home directory (e.g., ~/numpyro-env, ~/jax-env, ~/.venv)
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
    # Conda envs
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
    # Homebrew Python (macOS)
    brew_pythons <- Sys.glob("/opt/homebrew/opt/python@3.*/bin/python3.*")
    candidates <- c(candidates, brew_pythons)
    brew_pythons2 <- Sys.glob("/usr/local/opt/python@3.*/bin/python3.*")
    candidates <- c(candidates, brew_pythons2)
  }

  # PATH-based lookup
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
  # Check cache first
  cached <- Sys.getenv("BAYESROC_PYTHON", unset = "")
  if (nzchar(cached) && file.exists(cached)) return(cached)

  # Then prefer the bayesroc-managed venv (created by install_jax_backend()).
  # cmdstanr does the analogous thing with ~/.cmdstan/.
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

  # Then RETICULATE_PYTHON
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
      # Require BOTH jax and numpyro (the previous "import jax" check could
      # match a Python that had jax but no numpyro, or had a numpyro version
      # incompatible with jax -- the import would succeed and the fit would
      # crash later with a confusing error).
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


# Cache the package root at source-time (works when files are source()'d)
.bayesroc_source_root <- local({
  # Walk the call stack to find the source() call for this file
  root <- NULL
  for (i in seq_len(sys.nframe())) {
    ofile <- tryCatch(get("ofile", envir = sys.frame(i)), error = function(e) NULL)
    if (!is.null(ofile)) {
      # ofile is the path passed to source(); R/ is one level below the package root
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

  # Bivariate / CDP specifics
  if (!is.null(md$K2)) config$K2 <- md$K2
  if (!is.null(md$stan_data$item_type)) config$item_type <- as.integer(md$stan_data$item_type)
  if (!is.null(md$stan_data$y2)) config$y2 <- as.integer(md$stan_data$y2)
  if (!is.null(md$stan_data$rk)) config$rk <- as.integer(md$stan_data$rk)
  if (!is.null(md$stan_data$J)) config$J <- md$stan_data$J
  if (!is.null(md$stan_data$n_rkg)) config$n_rkg <- md$stan_data$n_rkg
  if (!is.null(md$stan_data$n_new)) config$n_new <- md$stan_data$n_new
  if (!is.null(md$stan_data$old_level_map)) config$old_level_map <- as.integer(md$stan_data$old_level_map)
  if (!is.null(md$stan_data$new_level_map)) config$new_level_map <- as.integer(md$stan_data$new_level_map)

  # Source variable (for source_mixture)
  if (!is.null(md$stan_data$source)) config$source <- as.integer(md$stan_data$source)

  # Lure mixture flag
  config$has_lure_mixture <- isTRUE(md$has_lure_mixture)

  # Count weights
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
                  "dprime_L", "sigma_L", "lambda_L")

  for (param in all_params) {
    fe_key <- paste0(param, "_fixed")
    fe <- md[[fe_key]]
    if (!is.null(fe)) {
      # family$params is now keyed by internal name across all families
      param_link <- if (!is.null(family$params[[param]]$link)) family$params[[param]]$link else "identity"

      # Handle empty col_names (intercept-only params)
      col_names <- fe$col_names
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

      config$params[[param]] <- pconf
    }
  }

  # ---- Threshold parameterization ----
  # CDP uses J tau thresholds (not K-1)
  if (family_name == "cdp") {
    n_thresh <- config$J
  } else {
    n_thresh <- config$K - 1L
  }
  config$n_thresh <- n_thresh
  config$thresh_mid_idx <- ceiling(n_thresh / 2)

  config$prior_thresh_mid <- get_fixed_prior(prior_lookup, "thresh_mid")
  config$prior_log_gaps <- get_fixed_prior(prior_lookup, "log_gaps")
  # Per-gap fixed-effect priors: resolved at gap index `[k]` (1-based), falling
  # back to the dpar-level `prior_log_gaps` when no coef-specific prior given.
  # JAX reads this vector when present; all-equal entries indicate "no override".
  n_gaps_jax <- max(0L, n_thresh - 1L)
  if (n_gaps_jax > 0) {
    config$prior_log_gaps_per_gap <- vapply(seq_len(n_gaps_jax), function(k)
      get_fixed_prior(prior_lookup, "log_gaps", coef = sprintf("[%d]", k)),
      character(1))
  }
  # Criterion2 (source/dim2) priors -- default to same as criterion, independently settable
  config$prior_thresh_mid2 <- get_fixed_prior(prior_lookup, "thresh_mid2")
  config$prior_log_gaps2 <- get_fixed_prior(prior_lookup, "log_gaps2")

  # Criterion uses build_criterion_structure, stored as md$criterion (not md$criterion_fixed)
  crit <- md$criterion
  if (!is.null(crit)) {
    config$P_crit <- crit$n_coef
    config$X_crit <- as.matrix(crit$X)
    config$crit_col_names <- crit$col_names
  } else {
    config$P_crit <- 1L
    config$X_crit <- matrix(1, nrow = config$N, ncol = 1)
    config$crit_col_names <- "Intercept"
  }

  # ---- Random effects ----
  config$random_effects <- list()

  re_params <- c("dprime", "criterion", "sigma", "lambda", "dprime2", "sigma2",
                 "dprime_B", "lambda_B", "discrim", "discrim_B",
                 "sigma_B", "sigma2_B", "rho", "rho_B", "rho_N",
                 "criterion2", "lambda2", "lambda2_B", "rec_crit", "know_crit",
                 "dprime_L", "sigma_L", "lambda_L")

  for (param in re_params) {
    re_key <- paste0(param, "_random")
    re_list <- md[[re_key]]
    # Criterion and criterion2 store REs inside the criterion structure
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

        # Always include group_idx: needed by matrix path too (Z is (N, n_terms),
        # not pre-expanded to (N, n_terms * n_groups))
        re_conf$index <- as.integer(re$group_idx)
        if (isTRUE(re$use_z_matrix) && !is.null(re$Z)) {
          re_conf$type <- "matrix"
          re_conf$Z <- as.matrix(re$Z)
        } else {
          re_conf$type <- "index"
          # For varying slopes with term_idx, pass that too
          if (!is.null(re$term_idx)) {
            re_conf$term_idx <- as.integer(re$term_idx)
          }
        }

        if (!is.null(re$cor_id)) {
          re_conf$cor_id <- re$cor_id
        }

        re_conf$prior_sd <- get_sd_prior(prior_lookup, param, re$group)
        re_conf$prior_cor <- get_cor_prior(prior_lookup, param, re$group)

        # Per-coef RE SD priors: for criterion REs the coefs are `thresh1..K-1`
        # (one per threshold position); for other params they're per fixed-effect
        # column name. Resolve each coef against the prior lookup; JAX reads the
        # per-coef vector and applies position-wise.
        if (param == "criterion" && !is.null(re$dim) && re$dim > 1L) {
          level_names <- if (!is.null(re$level_names)) re$level_names
                         else paste0("thresh", seq_len(re$dim))
          re_conf$prior_sd_per_coef <- vapply(level_names,
            function(coef) get_sd_prior(prior_lookup, "criterion", re$group, coef),
            character(1), USE.NAMES = FALSE)
        }

        config$random_effects[[length(config$random_effects) + 1]] <- re_conf
      }
    }
  }

  # Smooth terms
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
            config$smooths[[length(config$smooths) + 1]] <- sm_conf
          }
        }
      }
    }
  }

  if (!is.null(md$cross_cor_groups)) {
    config$cross_cor_groups <- md$cross_cor_groups
  }

  if (!is.null(md$stan_data$is_old)) {
    config$is_old <- as.numeric(md$stan_data$is_old)
  }

  config$varying_source_criteria <- isTRUE(family$varying_source_criteria)
  if (!is.null(family$varying_re)) config$varying_re <- family$varying_re
  if (isTRUE(md$needs_ordered_dprime)) config$needs_ordered_dprime <- TRUE
  if (!is.null(md$bounded)) config$bounded <- md$bounded
  if (!is.null(md$new_source_criteria)) config$new_source_criteria <- md$new_source_criteria
  if (!is.null(md$stan_data$is_new_response)) config$is_new_response <- as.integer(md$stan_data$is_new_response)
  if (!is.null(md$gap_link)) config$gap_link <- md$gap_link

  config
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
#' @return A list with samples, extra_fields, elapsed
#' @noRd
fit_jax <- function(model, chains = 4L, warmup = 2000L, samples = 2000L,
                    thin = 1L, init = 0, init_strategy = NULL, seed = NULL,
                    adapt_delta = 0.8, max_treedepth = 10L,
                    chain_method = "parallel", python = NULL,
                    progress_bar = TRUE) {

  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("Package 'jsonlite' is required for the JAX backend.")
  }

  # Per-call override beats discovery
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

  # Write config + fitting args to temp JSON
  tmp_dir <- tempfile("numpyro_")
  dir.create(tmp_dir)
  config_file <- file.path(tmp_dir, "config.json")
  result_file <- file.path(tmp_dir, "results.json")

  # init_strategy: "prior" | "random" | "numeric" (treated as init_to_uniform).
  # Default = "prior" to match fit_broc's default.
  if (is.null(init_strategy)) init_strategy <- "prior"
  fit_args <- list(
    config = config,
    chains = as.integer(chains),
    warmup = as.integer(warmup),
    samples = as.integer(samples),
    seed = as.integer(seed),
    result_file = result_file,
    init_strategy = init_strategy,
    init_radius  = if (is.numeric(init)) as.numeric(init) else 2.0
  )

  # Convert matrices to nested lists for JSON serialization
  fit_args_json <- rapply(fit_args, function(x) {
    if (is.matrix(x)) {
      # Convert matrix to list of rows (each row is a list)
      lapply(seq_len(nrow(x)), function(i) as.numeric(x[i, ]))
    } else {
      x
    }
  }, how = "replace")

  jsonlite::write_json(fit_args_json, config_file, auto_unbox = TRUE,
                       digits = 17, null = "null")

  # Parallel chains need one virtual device each (pmap); sequential runs them
  # all on a single device.
  device_count <- if (identical(chain_method, "sequential")) 1L else as.integer(chains)

  # Build the Python runner script
  runner_script <- file.path(tmp_dir, "run_fit.py")
  writeLines(c(
    "import sys, os, json",
    paste0("os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=", device_count, "'"),
    paste0("sys.path.insert(0, ", deparse(dirname(py_script)), ")"),
    paste0("os.chdir(", deparse(tmp_dir), ")"),
    "",
    "# Read config",
    "with open('config.json', 'r') as f:",
    "    args = json.load(f)",
    "",
    "config = args['config']",
    "result_file = args['result_file']",
    "",
    "# Convert list-of-lists back to list-of-lists (numpy will handle)",
    "import numpy as np",
    "def convert_matrices(obj):",
    "    if isinstance(obj, dict):",
    "        for k, v in obj.items():",
    "            if k in ('X', 'Z', 'X_crit') and isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):",
    "                obj[k] = [list(map(float, row)) for row in v]",
    "            else:",
    "                convert_matrices(v)",
    "    elif isinstance(obj, list):",
    "        for item in obj:",
    "            if isinstance(item, dict):",
    "                convert_matrices(item)",
    "convert_matrices(config)",
    "",
    "# Import and run",
    "from numpyro_backend import fit_model",
    "",
    "result = fit_model(",
    "    config=config,",
    paste0("    chains=", as.integer(chains), ","),
    paste0("    warmup=", as.integer(warmup), ","),
    paste0("    samples=", as.integer(samples), ","),
    paste0("    seed=", as.integer(seed), ","),
    paste0("    target_accept_prob=", adapt_delta, ","),
    paste0("    max_tree_depth=", as.integer(max_treedepth), ","),
    paste0("    thinning=", as.integer(thin), ","),
    paste0("    chain_method=", shQuote(chain_method), ","),
    paste0("    progress_bar=", if (isTRUE(progress_bar)) "True" else "False", ","),
    paste0("    init_strategy=", shQuote(init_strategy), ","),
    paste0("    init_radius=", as.numeric(fit_args$init_radius)),
    ")",
    "",
    "# Save results as JSON-compatible format",
    "output = {",
    "    'elapsed': result['elapsed'],",
    "    'samples': {},",
    "    'extra_fields': {}",
    "}",
    "",
    "# Save samples as .npy files (fast binary), record filenames in JSON",
    "for k, v in result['samples'].items():",
    "    fname = f'sample_{k}.npy'",
    "    np.save(fname, v)",
    "    output['samples'][k] = fname",
    "",
    "for k, v in result['extra_fields'].items():",
    "    fname = f'extra_{k}.npy'",
    "    np.save(fname, v)",
    "    output['extra_fields'][k] = fname",
    "",
    "with open(result_file, 'w') as f:",
    "    json.dump(output, f)",
    "",
    "# Print diagnostic warnings (only if problems detected)",
    "ef = result['extra_fields']",
    "warnings = []",
    "if 'diverging' in ef:",
    "    n_div = int(ef['diverging'].sum())",
    "    n_total = ef['diverging'].size",
    "    if n_div > 0:",
    "        warnings.append(f'WARNING: {n_div} of {n_total} ({100*n_div/n_total:.1f}%) transitions ended with a divergence.')",
    "if 'energy' in ef:",
    "    n_chains = ef['energy'].shape[0]",
    "    low_bfmi = []",
    "    for c in range(n_chains):",
    "        e = ef['energy'][c]",
    "        de = np.diff(e)",
    "        bfmi = np.sum(de**2) / np.sum((e - np.mean(e))**2)",
    "        if bfmi < 0.3:",
    "            low_bfmi.append((c+1, bfmi))",
    "    if low_bfmi:",
    "        chains_str = ', '.join(f'chain {c} (E-BFMI={b:.3f})' for c, b in low_bfmi)",
    "        warnings.append(f'WARNING: Low E-BFMI for {chains_str}.')",
    paste0("if 'num_steps' in ef:"),
    paste0("    max_steps = 2**", as.integer(max_treedepth)),
    "    n_hit = int((ef['num_steps'] >= max_steps).sum())",
    "    n_total = ef['num_steps'].size",
    "    if n_hit > 0:",
    paste0("        warnings.append(f'WARNING: {n_hit} of {n_total} ({100*n_hit/n_total:.1f}%) transitions hit max treedepth (", max_treedepth, ").')"),
    "for w in warnings:",
    "    print(w)"
  ), runner_script)

  # Run Python as interruptible subprocess with live output
  chain_note <- if (identical(chain_method, "sequential")) ", sequential" else ""
  message("Fitting with NumPyro (", chains, " chains", chain_note, ", ", warmup,
          " warmup, ", samples, " samples)...")

  if (!requireNamespace("processx", quietly = TRUE)) {
    stop("Package 'processx' is required for the JAX backend.\n",
         "Install it with: install.packages('processx')")
  }

  proc <- processx::process$new(
    python, args = runner_script,
    stdout = "|", stderr = "|",
    cleanup = TRUE, cleanup_tree = TRUE
  )

  # Kill the process tree on interrupt, error, or early exit
  on.exit({
    if (proc$is_alive()) {
      message("\nKilling NumPyro subprocess...")
      proc$kill_tree()
    }
  }, add = TRUE)

  # Stream output to console while polling -- R checks for interrupts during poll_io
  while (proc$is_alive()) {
    proc$poll_io(200)  # Wait up to 200ms for output; R can be interrupted here
    out <- proc$read_output()
    err <- proc$read_error()
    if (nzchar(out)) cat(out)
    if (nzchar(err)) cat(err)
  }
  # Flush remaining output
  out <- proc$read_output()
  err <- proc$read_error()
  if (nzchar(out)) cat(out)
  if (nzchar(err)) cat(err)

  exit_code <- proc$get_exit_status()

  if (exit_code != 0) {
    stop("NumPyro fitting failed (exit code ", exit_code, ").\n",
         "Check the Python output above for details.")
  }

  if (!file.exists(result_file)) {
    stop("NumPyro fitting did not produce results. Check Python output.")
  }

  # Read results
  result_meta <- jsonlite::read_json(result_file)

  # Read numpy arrays using a minimal .npy reader
  samples_list <- list()
  for (name in names(result_meta$samples)) {
    npy_file <- file.path(tmp_dir, result_meta$samples[[name]])
    samples_list[[name]] <- read_npy(npy_file)
  }

  extra_fields <- list()
  for (name in names(result_meta$extra_fields)) {
    npy_file <- file.path(tmp_dir, result_meta$extra_fields[[name]])
    extra_fields[[name]] <- read_npy(npy_file)
  }

  # Cleanup
  unlink(tmp_dir, recursive = TRUE)

  list(
    samples = samples_list,
    extra_fields = extra_fields,
    elapsed = result_meta$elapsed
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

  # Magic number and version
  magic <- readBin(con, "raw", 6)
  version <- readBin(con, "integer", 1, size = 2, endian = "little")

  # Header length
  if (version == 1L) {
    header_len <- readBin(con, "integer", 1, size = 2, endian = "little")
  } else {
    header_len <- readBin(con, "integer", 1, size = 4, endian = "little")
  }

  # Parse header (Python dict as string)
  header_str <- rawToChar(readBin(con, "raw", header_len))

  # Extract dtype
  dtype_match <- regmatches(header_str, regexec("'descr':\\s*'([^']+)'", header_str))[[1]]
  dtype <- dtype_match[2]

  # Extract shape
  shape_match <- regmatches(header_str, regexec("'shape':\\s*\\(([^)]+)\\)", header_str))[[1]]
  shape_str <- shape_match[2]
  shape <- as.integer(strsplit(trimws(gsub(",\\s*$", "", shape_str)), ",\\s*")[[1]])
  if (length(shape) == 0 || any(is.na(shape))) shape <- 1L

  # Extract fortran_order
  fortran <- grepl("'fortran_order':\\s*True", header_str)

  # Total elements
  n_elem <- prod(shape)

  # Read data based on dtype
  if (grepl("f8|float64", dtype)) {
    data <- readBin(con, "double", n_elem, size = 8, endian = "little")
  } else if (grepl("f4|float32", dtype)) {
    data <- readBin(con, "double", n_elem, size = 4, endian = "little")
  } else if (grepl("i8|int64", dtype)) {
    # Read as raw bytes, convert to double (R doesn't have int64)
    raw_data <- readBin(con, "raw", n_elem * 8)
    data <- readBin(raw_data, "double", n_elem, size = 8, endian = "little")
    # Actually for int64, read as integer pairs
    data <- as.numeric(readBin(raw_data, "integer", n_elem * 2, size = 4, endian = "little"))
    data <- data[seq(1, length(data), 2)] + data[seq(2, length(data), 2)] * 2^32
  } else if (grepl("i4|int32", dtype)) {
    data <- readBin(con, "integer", n_elem, size = 4, endian = "little")
  } else if (grepl("b1|bool", dtype)) {
    data <- as.logical(readBin(con, "integer", n_elem, size = 1, endian = "little"))
  } else {
    # Fallback: try as double
    data <- readBin(con, "double", n_elem, size = 8, endian = "little")
  }

  # Reshape
  if (length(shape) == 1) {
    return(data)
  } else {
    # numpy is row-major (C order), R is column-major (Fortran order)
    # Need to fill array in C order then transpose
    arr <- array(data, dim = rev(shape))
    return(aperm(arr, rev(seq_along(shape))))
  }
}


# =============================================================================
# Helper
# =============================================================================

