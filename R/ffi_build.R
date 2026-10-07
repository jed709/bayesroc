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
  if (!rebuild && !is.null(memo)) return(memo)
  fail <- function(msg, fix) {
    stop("bayesroc: the JAX backend needs its native kernel, which could not be built: ", msg,
         ".
", fix, call. = FALSE)
  }
  toolchain <- paste0("Install a C++17 toolchain (the one cmdstanr uses): Rtools on Windows, ",
                      "the Xcode command-line tools on macOS, g++ on Linux.")
  src <- .bvn_ffi_source_dir()
  if (is.null(src)) return(fail("sources not found", "Reinstall bayesroc."))
  jl <- .jaxlib_ffi_include(python)
  if (is.null(jl)) return(fail("jax.ffi headers not found", "Update JAX with install_jax()."))
  cxx <- .cxx17_command()
  if (is.null(cxx)) return(fail("no C++17 compiler", toolchain))
  flags <- .bvn_ffi_flags()
  hdr <- file.path(src$include, "bayesroc", c("bvn_simd.h", "bvn_tail.h", "tmap_kernel.h"))
  key <- paste(c(unname(tools::md5sum(c(src$cc, hdr))), jl$jax_version, jl$include,
                 cxx$prog, cxx$args, flags, R.version$platform), collapse = "|")
  tag <- substr(digest_string(key), 1, 16)
  ext <- if (.Platform$OS.type == "windows") ".dll" else ".so"
  cache <- file.path(tools::R_user_dir("bayesroc", "cache"), "ffi")
  lib <- file.path(cache, paste0("bvn_ffi_", tag, ext))
  if (file.exists(lib) && !rebuild) {
    .bvn_ffi_state[[python]] <- normalizePath(lib, winslash = "/")
    .prune_ffi_cache(cache, python, lib, ext)
    return(.bvn_ffi_state[[python]])
  }
  dir.create(cache, recursive = TRUE, showWarnings = FALSE)
  tmp <- tempfile("bvn_ffi_", tmpdir = cache, fileext = ext)
  args <- c(cxx$args, flags, paste0("-I", shQuote(jl$include)), paste0("-I", shQuote(src$include)),
            "-o", shQuote(tmp), shQuote(src$cc))
  if (!quiet) message("bayesroc: building the native likelihood kernels")
  out <- suppressWarnings(system2(cxx$prog, args, stdout = TRUE, stderr = TRUE))
  status <- attr(out, "status")
  if (!is.null(status) && status != 0L || !file.exists(tmp)) {
    unlink(tmp)
    return(fail(paste0("compilation failed: ", paste(utils::tail(out, 5), collapse = " ")), toolchain))
  }
  if (!file.rename(tmp, lib)) {
    unlink(tmp)
    if (!file.exists(lib)) return(fail(paste0("could not write to ", cache), "Make that directory writable."))
  }
  .bvn_ffi_state[[python]] <- normalizePath(lib, winslash = "/")
  .prune_ffi_cache(cache, python, lib, ext)
  .bvn_ffi_state[[python]]
}

.prune_ffi_cache <- function(cache, python, lib, ext) {
  idx_file <- file.path(cache, "index.rds")
  idx <- if (file.exists(idx_file)) tryCatch(readRDS(idx_file), error = function(e) list()) else list()
  idx[[normalizePath(python, winslash = "/", mustWork = FALSE)]] <- basename(lib)
  idx <- idx[vapply(idx, function(f) file.exists(file.path(cache, f)), logical(1))]
  try(saveRDS(idx, idx_file), silent = TRUE)
  stale <- setdiff(list.files(cache, pattern = paste0("^bvn_ffi_.*\\", ext, "$")), unlist(idx))
  if (length(stale)) suppressWarnings(unlink(file.path(cache, stale)))
  invisible(NULL)
}

digest_string <- function(x) {
  f <- tempfile()
  on.exit(unlink(f))
  writeLines(x, f, useBytes = TRUE)
  unname(tools::md5sum(f))
}
