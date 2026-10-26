.cmdstan_log_prob <- function(mod, data, draws_csv) {
  js <- tempfile(fileext = ".json")
  cmdstanr::write_stan_json(data, js)
  out <- tempfile(fileext = ".csv")
  old <- Sys.getenv("PATH")
  on.exit(Sys.setenv(PATH = old), add = TRUE)
  tool <- if (.Platform$OS.type == "windows") cmdstanr:::toolchain_PATH_env_var() else NULL
  Sys.setenv(PATH = paste(c(file.path(cmdstanr::cmdstan_path(), "stan/lib/stan_math/lib/tbb"), tool, old),
                          collapse = .Platform$path.sep))
  system2(mod$exe_file(), c("log_prob", paste0("constrained_params=", draws_csv), "data",
                            paste0("file=", js), "output", paste0("file=", out), "sig_figs=18"),
          stdout = FALSE, stderr = FALSE)
  as.matrix(utils::read.csv(out, comment.char = "#"))
}

.stan_parity <- function(mp, mb, warmup = 20) {
  modp <- .compile_broc_stan(mp)
  modb <- .compile_broc_stan(mb)
  fp <- modb$sample(data = mb$stan_data, chains = 1, iter_warmup = warmup, iter_sampling = 5,
                    init = 0, sig_figs = 18, refresh = 50, seed = 1, max_treedepth = 6, show_messages = FALSE)
  a <- .cmdstan_log_prob(modp, mp$stan_data, fp$output_files()[1])
  b <- .cmdstan_log_prob(modb, mb$stan_data, fp$output_files()[1])
  ok <- is.finite(a[, 1])
  if (sum(ok) < 3) stop("fewer than 3 parity points where the plain model is finite")
  list(lp = max(abs(a[ok, 1] - b[ok, 1])),
       grad = max(abs(a[ok, -1] - b[ok, -1]) / pmax(1, abs(a[ok, -1]))),
       dim = c(ncol(a), ncol(b)))
}

.stan_parity_at <- function(mp, mb, edits, warmup = 20) {
  modp <- .compile_broc_stan(mp)
  modb <- .compile_broc_stan(mb)
  fp <- modb$sample(data = mb$stan_data, chains = 1, iter_warmup = warmup, iter_sampling = 5,
                    init = 0, sig_figs = 18, refresh = 50, seed = 1, max_treedepth = 6, show_messages = FALSE)
  src <- readLines(fp$output_files()[1])
  hdr <- grep("^lp__", src)
  cols <- strsplit(src[hdr], ",")[[1]]
  rows <- which(seq_along(src) > hdr & !startsWith(src, "#") & nzchar(src))
  for (r in rows) {
    v <- strsplit(src[r], ",")[[1]]
    for (nm in names(edits)) v[match(nm, cols)] <- format(edits[[nm]], digits = 17)
    src[r] <- paste(v, collapse = ",")
  }
  csv <- tempfile(fileext = ".csv")
  writeLines(src, csv)
  a <- .cmdstan_log_prob(modp, mp$stan_data, csv)
  b <- .cmdstan_log_prob(modb, mb$stan_data, csv)
  ok <- is.finite(a[, 1])
  if (sum(ok) < 3) stop("fewer than 3 parity points where the plain model is finite")
  list(lp = max(abs(a[ok, 1] - b[ok, 1])),
       grad = max(abs(a[ok, -1] - b[ok, -1]) / pmax(1, abs(a[ok, -1]))),
       lp_values = a[, 1])
}

.plain_stan_dev_env <- function() {
  src <- testthat::test_path("..", "..", "dev", "plain_stan.R")
  if (!file.exists(src)) testthat::skip("dev/plain_stan.R not available (development-only path)")
  env <- new.env(parent = asNamespace("bayesroc"))
  sys.source(src, envir = env)
  env
}

.broc_plain <- function(...) {
  old <- options(bayesroc.dev_plain_stan = .plain_stan_dev_env())
  on.exit(options(old), add = TRUE)
  broc(...)
}

.broc_with <- function(batch, ...) if (isTRUE(batch)) broc(...) else .broc_plain(...)
