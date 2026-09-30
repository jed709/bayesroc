.cmdstan_log_prob <- function(mod, data, draws_csv) {
  js <- tempfile(fileext = ".json")
  cmdstanr::write_stan_json(data, js)
  out <- tempfile(fileext = ".csv")
  old <- Sys.getenv("PATH")
  on.exit(Sys.setenv(PATH = old), add = TRUE)
  Sys.setenv(PATH = paste(file.path(cmdstanr::cmdstan_path(), "stan/lib/stan_math/lib/tbb"),
                          old, sep = .Platform$path.sep))
  system2(mod$exe_file(), c("log_prob", paste0("constrained_params=", draws_csv), "data",
                            paste0("file=", js), "output", paste0("file=", out), "sig_figs=18"),
          stdout = FALSE, stderr = FALSE)
  as.matrix(utils::read.csv(out, comment.char = "#"))
}

.stan_parity <- function(mp, mb, warmup = 100) {
  modp <- .compile_broc_stan(mp)
  modb <- .compile_broc_stan(mb)
  fp <- modp$sample(data = mp$stan_data, chains = 1, iter_warmup = warmup, iter_sampling = 5,
                    init = 0, sig_figs = 18, refresh = 50, seed = 1, show_messages = FALSE)
  a <- .cmdstan_log_prob(modp, mp$stan_data, fp$output_files()[1])
  b <- .cmdstan_log_prob(modb, mb$stan_data, fp$output_files()[1])
  list(lp = max(abs(a[, 1] - b[, 1])),
       grad = max(abs(a[, -1] - b[, -1]) / pmax(1, abs(a[, -1]))),
       dim = c(ncol(a), ncol(b)))
}
