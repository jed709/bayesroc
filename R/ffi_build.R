.bvn_ffi_state <- new.env(parent = emptyenv())

.bvn_ffi_source_dir <- function() {
  backend <- find_backend_script()
  inst <- dirname(dirname(backend))
  cc <- file.path(inst, "ffi", "bvn_ffi.cc")
  inc <- file.path(inst, "include")
  if (!file.exists(cc) || !file.exists(file.path(inc, "bayesroc", "bvn_simd.h"))) return(NULL)
  list(cc = cc, include = inc)
}

.cxx17_command <- function() {
  r <- file.path(R.home("bin"), "R")
  cxx <- tryCatch(system2(r, c("CMD", "config", "CXX17"), stdout = TRUE, stderr = FALSE),
                  error = function(e) character(0), warning = function(w) character(0))
  cxx <- trimws(paste(cxx, collapse = " "))
  if (!nzchar(cxx)) return(NULL)
  parts <- strsplit(cxx, "\\s+")[[1]]
  prog <- Sys.which(parts[1])
  if (!nzchar(prog)) return(NULL)
  list(prog = unname(prog), args = parts[-1])
}

.bvn_ffi_flags <- function() {
  flags <- c("-O3", "-std=c++17", "-shared", "-fno-math-errno", "-w")
  arch <- Sys.info()[["machine"]]
  flags <- c(flags, if (grepl("arm|aarch", arch, ignore.case = TRUE)) "-mcpu=native" else "-march=native")
  if (.Platform$OS.type == "windows") {
    flags <- c(flags, "-Wa,-muse-unaligned-vector-move", "-static-libgcc", "-static-libstdc++")
  } else {
    flags <- c(flags, "-fPIC")
  }
  flags
}

.jaxlib_ffi_include <- function(python) {
  out <- tryCatch(
    system2(python, c("-c", shQuote("import jax, jax.ffi; print(jax.__version__); print(jax.ffi.include_dir())")),
            stdout = TRUE, stderr = FALSE),
    error = function(e) character(0), warning = function(w) character(0))
  if (length(out) < 2 || !dir.exists(out[2])) return(NULL)
  list(jax_version = out[1], include = out[2])
}

.bvn_ffi_lib <- function(python = jax_python(), rebuild = FALSE, quiet = TRUE) {
  if (is.null(python) || !nzchar(python)) return(NULL)
  memo <- .bvn_ffi_state[[python]]
  if (!rebuild && !is.null(memo)) return(if (nzchar(memo)) memo else NULL)
  fail <- function(msg) {
    if (!isTRUE(.bvn_ffi_state$.warned)) {
      message("bayesroc: native bivariate-normal kernel unavailable (", msg,
              "); the JAX backend uses its pure-JAX equivalent, which gives the same ",
              "results more slowly.")
      .bvn_ffi_state$.warned <- TRUE
    }
    .bvn_ffi_state[[python]] <- ""
    NULL
  }
  src <- .bvn_ffi_source_dir()
  if (is.null(src)) return(fail("sources not found"))
  jl <- .jaxlib_ffi_include(python)
  if (is.null(jl)) return(fail("jax.ffi headers not found"))
  cxx <- .cxx17_command()
  if (is.null(cxx)) return(fail("no C++17 compiler"))
  flags <- .bvn_ffi_flags()
  hdr <- file.path(src$include, "bayesroc", "bvn_simd.h")
  key <- paste(c(unname(tools::md5sum(c(src$cc, hdr))), jl$jax_version, jl$include,
                 cxx$prog, cxx$args, flags, R.version$platform), collapse = "|")
  tag <- substr(digest_string(key), 1, 16)
  ext <- if (.Platform$OS.type == "windows") ".dll" else ".so"
  cache <- file.path(tools::R_user_dir("bayesroc", "cache"), "ffi")
  lib <- file.path(cache, paste0("bvn_ffi_", tag, ext))
  if (file.exists(lib) && !rebuild) {
    .bvn_ffi_state[[python]] <- normalizePath(lib, winslash = "/")
    return(.bvn_ffi_state[[python]])
  }
  dir.create(cache, recursive = TRUE, showWarnings = FALSE)
  tmp <- tempfile("bvn_ffi_", tmpdir = cache, fileext = ext)
  args <- c(cxx$args, flags, paste0("-I", shQuote(jl$include)), paste0("-I", shQuote(src$include)),
            "-o", shQuote(tmp), shQuote(src$cc))
  if (!quiet) message("bayesroc: building native bivariate-normal kernel")
  out <- suppressWarnings(system2(cxx$prog, args, stdout = TRUE, stderr = TRUE))
  status <- attr(out, "status")
  if (!is.null(status) && status != 0L || !file.exists(tmp)) {
    unlink(tmp)
    return(fail(paste0("compilation failed: ", paste(utils::tail(out, 5), collapse = " "))))
  }
  if (!file.rename(tmp, lib)) {
    unlink(tmp)
    if (!file.exists(lib)) return(fail("could not write to the cache directory"))
  }
  .bvn_ffi_state[[python]] <- normalizePath(lib, winslash = "/")
  .bvn_ffi_state[[python]]
}

digest_string <- function(x) {
  f <- tempfile()
  on.exit(unlink(f))
  writeLines(x, f, useBytes = TRUE)
  unname(tools::md5sum(f))
}
