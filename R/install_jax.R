# =============================================================================
# JAX backend management -- managed venv at tools::R_user_dir("bayesroc", "data")
# =============================================================================
# These functions give bayesroc the same UX shape as cmdstanr for setting up
# the JAX/NumPyro backend:
#
#   bayesroc::check_jax_backend()   -> diagnose what's there, what's needed
#   bayesroc::install_jax_backend() -> create an isolated, version-pinned venv
#   bayesroc::jax_python()          -> query the resolved Python path
#   bayesroc::set_jax_python()      -> override the path explicitly
#
# The managed venv lives in tools::R_user_dir("bayesroc", "data")/venv. The
# cmdstanr analogue is ~/.cmdstan/. Pinning is essential: jax 0.10.x and
# numpyro 0.20.x are mutually broken (numpyro imports a JAX symbol jax 0.10
# removed). The latest known-good combination is baked into JAX_VERSION_PIN /
# NUMPYRO_VERSION_PIN below, which install_jax_backend() defaults to and
# check_jax_backend() validates against.

# Pinned versions known to work together. Bump these as numpyro releases
# catch up to jax.
JAX_VERSION_PIN     <- "0.9.2"
NUMPYRO_VERSION_PIN <- "0.20.1"

# Whitelist of compatible (jax, numpyro) pairs that check_jax_backend treats
# as "good". Any pair not in this list with both packages installed gets a
# warning recommending install_jax_backend().
.JAX_COMPAT_WHITELIST <- list(
  c(jax = "0.9.2", numpyro = "0.20.1"),
  c(jax = "0.9.2", numpyro = "0.20.0")
)


# Per-user R location (tools::R_user_dir) where install_jax_backend() writes its
# managed venv.
#' @noRd
jax_backend_dir <- function() {
  tools::R_user_dir("bayesroc", "data")
}


#' Path to the Python executable inside the managed venv
#'
#' Returns the path the managed venv would have at the given `dir`, regardless
#' of whether it currently exists. Use [jax_python()] to query the *resolved*
#' Python (which may be a non-managed install).
#'
#' @param dir Backend dir (defaults to `jax_backend_dir()`).
#' @return Character path.
#' @noRd
managed_venv_python <- function(dir = jax_backend_dir()) {
  if (.Platform$OS.type == "windows") {
    file.path(dir, "venv", "Scripts", "python.exe")
  } else {
    file.path(dir, "venv", "bin", "python")
  }
}


#' Absolutize a path while preserving a final symlink
#'
#' A venv's `bin/python` is a symlink to the base interpreter. `normalizePath()`
#' resolves it on Unix, which would point the backend at the base install whose
#' site-packages has no jax. We normalize only the (real) parent directory and
#' re-attach the basename, keeping the venv interpreter itself.
#' @noRd
abs_keep_symlink <- function(p) {
  file.path(normalizePath(dirname(p), winslash = "/", mustWork = FALSE),
            basename(p))
}


#' Install the JAX/NumPyro backend into a managed venv
#'
#' Creates an isolated Python virtual environment under `jax_backend_dir()`
#' and installs `jax` and `numpyro` at known-compatible pinned versions. This
#' never touches the user's system or global Python install. The cmdstanr
#' analogue is [cmdstanr::install_cmdstan()].
#'
#' Subsequent JAX-backend fits via `fit_broc(backend = "jax")` will prefer
#' this managed venv automatically; no further configuration is needed.
#'
#' @param dir Directory to install into. Default: `jax_backend_dir()`.
#' @param python Path to the Python executable used to *create* the venv.
#'   Default: auto-detect from PATH (must be Python 3.10+). The venv itself
#'   uses its own bundled interpreter regardless of this choice.
#' @param jax_version,numpyro_version Versions to pin. Defaults to
#'   `JAX_VERSION_PIN` / `NUMPYRO_VERSION_PIN` constants in the package
#'   (the latest mutually-compatible pair as of release).
#' @param overwrite If `TRUE`, recreate the venv even if it already exists.
#'   Default `FALSE` errors if a managed venv is already present.
#' @param quiet If `TRUE`, suppress pip output.
#' @return Invisibly returns the path to the installed venv's Python.
#' @seealso [check_jax_backend()], [jax_python()], [set_jax_python()].
#' @export
install_jax_backend <- function(dir = jax_backend_dir(),
                                python = NULL,
                                jax_version = JAX_VERSION_PIN,
                                numpyro_version = NUMPYRO_VERSION_PIN,
                                overwrite = FALSE,
                                quiet = FALSE) {
  venv_dir <- file.path(dir, "venv")
  venv_py  <- managed_venv_python(dir)

  if (file.exists(venv_py) && !overwrite) {
    stop("Managed JAX venv already exists at ", venv_dir,
         "\nUse `install_jax_backend(overwrite = TRUE)` to recreate it.",
         call. = FALSE)
  }
  if (file.exists(venv_py) && overwrite) {
    if (!quiet) message("Removing existing venv at ", venv_dir)
    unlink(venv_dir, recursive = TRUE, force = TRUE)
  }

  # Find a system Python to bootstrap the venv
  if (is.null(python)) {
    python <- find_system_python()
  }
  if (!file.exists(python) && !nzchar(Sys.which(python))) {
    stop("Python executable not found: ", python, call. = FALSE)
  }
  ver <- python_version(python)
  if (is.null(ver) || ver < "3.10") {
    stop("install_jax_backend() requires Python >= 3.10 to bootstrap the venv. ",
         "Found: ", ver %||% "unknown", " at ", python, call. = FALSE)
  }

  if (!quiet) message("Creating venv at ", venv_dir, " (Python ", ver, ")")
  dir.create(dir, showWarnings = FALSE, recursive = TRUE)
  res <- system2(python, c("-m", "venv", shQuote(venv_dir)))
  if (res != 0L || !file.exists(venv_py)) {
    stop("Failed to create venv (exit code ", res, ")", call. = FALSE)
  }

  if (!quiet) message("Upgrading pip")
  pip_args <- if (quiet) "--quiet" else character(0)
  system2(venv_py, c("-m", "pip", "install", "--upgrade", "pip", pip_args))

  if (!quiet) message(sprintf("Installing jax==%s + numpyro==%s",
                              jax_version, numpyro_version))
  # Capture stderr to a temp file (stdout still streams live) so a failed
  # install reports *why*, not just an exit code.
  err_file <- tempfile("pip_err_")
  res <- system2(venv_py, c(
    "-m", "pip", "install",
    pip_args,
    paste0("jax[cpu]==", jax_version),
    paste0("numpyro==", numpyro_version)
  ), stderr = err_file)
  if (res != 0L) {
    err <- tryCatch(readLines(err_file, warn = FALSE), error = function(e) character(0))
    unlink(err_file)
    stop("pip install failed (exit code ", res, ").\n",
         if (length(err)) {
           paste0("pip stderr (last lines):\n",
                  paste(utils::tail(err, 20), collapse = "\n"), "\n")
         } else "",
         "If this is a wheel/version error, jax==", jax_version,
         " may have no wheel for your Python (", ver, "); ",
         "jax==", jax_version, " supports roughly Python 3.10-3.13.",
         call. = FALSE)
  }
  unlink(err_file)

  # Verify
  ok <- tryCatch({
    out <- system2(venv_py, c("-c",
      shQuote("import jax, numpyro; print(jax.__version__, numpyro.__version__)")),
      stdout = TRUE, stderr = TRUE)
    if (!quiet) message("verified: ", paste(out, collapse = " "))
    TRUE
  }, error = function(e) FALSE)
  if (!ok) stop("Install completed but jax/numpyro import failed.", call. = FALSE)

  # Cache the path so future fit_broc() calls find it first. Keep the venv
  # symlink intact -- resolving it points at the base interpreter (no jax).
  Sys.setenv(BAYESROC_PYTHON = abs_keep_symlink(venv_py))

  if (!quiet) message("\nDone. The JAX backend will use this venv from now on.")
  invisible(abs_keep_symlink(venv_py))
}


#' Resolve the Python executable used by the JAX backend
#'
#' Returns the Python path that `fit_broc(backend = "jax")` will currently use.
#' Resolution order:
#'   1. `Sys.getenv("BAYESROC_PYTHON")` if set and the file exists (set by
#'      [set_jax_python()] or auto-cached during the last fit / install).
#'   2. The managed venv at `jax_backend_dir()` if it exists.
#'   3. `Sys.getenv("RETICULATE_PYTHON")` if set and the file exists.
#'   4. Auto-discovery: scans common Python locations and PATH for an install
#'      that can `import jax`. (See `find_python_with_jax`.)
#' Returns `NULL` if nothing is found -- call [install_jax_backend()] then.
#'
#' @return Character path to a Python executable, or `NULL`.
#' @seealso [set_jax_python()], [check_jax_backend()], [install_jax_backend()].
#' @export
jax_python <- function() {
  cached <- Sys.getenv("BAYESROC_PYTHON", unset = "")
  if (nzchar(cached) && file.exists(cached)) return(cached)
  managed <- managed_venv_python()
  if (file.exists(managed)) return(managed)
  retic <- Sys.getenv("RETICULATE_PYTHON", unset = "")
  if (nzchar(retic) && file.exists(retic)) return(retic)
  found <- tryCatch(find_python_with_jax(), error = function(e) NULL)
  found
}


#' Override the Python executable used by the JAX backend
#'
#' Sets `BAYESROC_PYTHON` for the current R session. Future
#' `fit_broc(backend = "jax")` calls will use this Python without going
#' through auto-discovery.
#'
#' @param python Path to a Python executable. Must already have `jax` and
#'   `numpyro` installed (use [check_jax_backend()] to verify after).
#' @return Invisibly returns the new path.
#' @export
set_jax_python <- function(python) {
  if (!file.exists(python)) {
    stop("Python executable not found: ", python, call. = FALSE)
  }
  python <- abs_keep_symlink(python)
  Sys.setenv(BAYESROC_PYTHON = python)
  invisible(python)
}


#' Diagnose the JAX backend setup
#'
#' Reports the resolved Python, whether `jax` and `numpyro` import, and whether
#' their installed versions are compatible.
#'
#' @return Invisibly, a list with `status` (one of `"ok"`, `"compat_warning"`,
#'   `"missing_numpyro"`, `"no_jax"`, `"no_python"`), `python`, `jax_version`,
#'   `numpyro_version`, and `message`.
#' @seealso [install_jax_backend()].
#' @export
check_jax_backend <- function() {
  py <- jax_python()
  result <- list(status = NA_character_, python = py,
                 jax_version = NA_character_,
                 numpyro_version = NA_character_,
                 message = "")

  if (is.null(py)) {
    result$status  <- "no_python"
    result$message <- paste(
      "No Python with JAX found.",
      "Run `bayesroc::install_jax_backend()` to create an isolated venv,",
      "or install Python 3.10+ first if you don't have one.",
      sep = "\n  ")
    cat("[bayesroc] check_jax_backend: no usable Python\n",
                     result$message, "\n", sep = "")
    return(invisible(result))
  }

  # Try to read versions
  out <- tryCatch(
    system2(py, c("-c",
      shQuote("import sys\ntry:\n  import jax;     j = jax.__version__\nexcept Exception: j = ''\ntry:\n  import numpyro; n = numpyro.__version__\nexcept Exception: n = ''\nprint(j); print(n)")),
      stdout = TRUE, stderr = TRUE),
    error = function(e) character(0))
  jax_v     <- if (length(out) >= 1) out[1] else ""
  numpyro_v <- if (length(out) >= 2) out[2] else ""
  result$jax_version     <- jax_v
  result$numpyro_version <- numpyro_v

  if (!nzchar(jax_v)) {
    result$status  <- "no_jax"
    result$message <- paste(
      sprintf("Python at '%s' has no JAX installed.", py),
      "Run `bayesroc::install_jax_backend()` for an isolated install,",
      "or run `<that python> -m pip install jax==", JAX_VERSION_PIN,
      " numpyro==", NUMPYRO_VERSION_PIN, "`.",
      sep = "\n  ")
    cat("[bayesroc] check_jax_backend: no JAX\n",
                     result$message, "\n", sep = "")
    return(invisible(result))
  }

  if (!nzchar(numpyro_v)) {
    result$status  <- "missing_numpyro"
    result$message <- paste(
      sprintf("Python at '%s' has JAX %s but no NumPyro.", py, jax_v),
      "Run `bayesroc::install_jax_backend()` for an isolated install,",
      sprintf("or `<that python> -m pip install numpyro==%s`.",
              NUMPYRO_VERSION_PIN),
      sep = "\n  ")
    cat("[bayesroc] check_jax_backend: missing NumPyro\n",
                     result$message, "\n", sep = "")
    return(invisible(result))
  }

  # Both present -- check compatibility
  pair <- c(jax = jax_v, numpyro = numpyro_v)
  whitelisted <- any(vapply(.JAX_COMPAT_WHITELIST,
                            function(p) identical(p[c("jax", "numpyro")], pair),
                            logical(1)))
  if (whitelisted) {
    result$status  <- "ok"
    result$message <- sprintf("OK. JAX %s + NumPyro %s at %s.",
                              jax_v, numpyro_v, py)
    cat("[bayesroc] check_jax_backend: ", result$message, "\n",
                     sep = "")
  } else {
    result$status  <- "compat_warning"
    pin_pair <- sprintf("jax %s + numpyro %s", JAX_VERSION_PIN,
                        NUMPYRO_VERSION_PIN)
    result$message <- paste(
      sprintf("Found JAX %s + NumPyro %s at %s.", jax_v, numpyro_v, py),
      sprintf("This pair is not on the known-compatible whitelist (latest: %s).",
              pin_pair),
      "If JAX-backend fits crash with a numpyro/jax import error, run",
      "  `bayesroc::install_jax_backend(overwrite = TRUE)` for the pinned versions.",
      sep = "\n  ")
    cat("[bayesroc] check_jax_backend: ", result$message, "\n",
                     sep = "")
  }
  invisible(result)
}


# =============================================================================
# Internal helpers
# =============================================================================

#' @noRd
find_system_python <- function() {
  candidates <- if (.Platform$OS.type == "windows") {
    c("python", "py", "python3")
  } else {
    c("python3", "python")
  }
  for (cmd in candidates) {
    p <- Sys.which(cmd)
    if (nzchar(p)) {
      ver <- python_version(p)
      if (!is.null(ver) && ver >= "3.10") return(unname(p))
    }
  }
  # PATH found nothing usable -- fall back to the same common-location search
  # the fitting backend uses (conda / pyenv / Homebrew / named venvs). This runs
  # ONLY when the PATH loop above came up empty, so the normal PATH-based install
  # is completely unaffected.
  for (p in .python_candidate_paths()) {
    ver <- python_version(p)
    if (!is.null(ver) && ver >= "3.10") return(unname(p))
  }
  stop("No Python >= 3.10 found on PATH or in common install locations ",
       "(conda / pyenv / Homebrew / named venvs). Install Python 3.10+ (on ",
       "Windows, from python.org -- not the Microsoft Store), or pass ",
       "`python = \"/path/to/python\"`.", call. = FALSE)
}

#' @noRd
python_version <- function(python) {
  out <- tryCatch(
    system2(python, c("-c",
      shQuote("import sys; print('%d.%d.%d' % sys.version_info[:3])")),
      stdout = TRUE, stderr = TRUE),
    error = function(e) character(0))
  if (length(out) == 0 || !grepl("^\\d+\\.\\d+", out[1])) return(NULL)
  package_version(out[1])
}

#' @noRd
`%||%` <- function(a, b) if (is.null(a)) b else a
