# Helpers shared with tests/testthat/
# test-dpsd-new-recollection.R (copied).

.dpsd_rr_data <- function(seed = 4, n_per = 150, lam = 0.3, lam_new = 0.2, d = 1, n_sub = 6) {
  set.seed(seed)
  K <- 6L
  old <- rep(c(1L, 0L), each = n_per)
  y <- as.integer(cut(rnorm(2 * n_per, d * old), c(-Inf, -1.2, -0.4, 0.3, 1.0, 1.8, Inf)))
  rec <- stats::runif(2 * n_per) < ifelse(old == 1L, lam, lam_new)
  y[rec] <- ifelse(old[rec] == 1L, K, 1L)
  data.frame(conf = y, old = old, sub = factor(sample(seq_len(n_sub), 2 * n_per, TRUE)))
}
.dpsd_rr_cases <- function(d) {
  f <- function(link, ...) brf(conf | old ~ 1, criterion ~ 1, lambda ~ 1, ..., family = dpsd(link = link))
  list(
    probit_sigma = list(f(link_probit(), lambda_new ~ 1 + (1 | sub), sigma ~ 1), d),
    logit = list(f(link_logit(), lambda_new ~ 1), d),
    cloglog = list(f(link_cloglog(), lambda_new ~ 1 + (1 | sub)), d)
  )
}

test_that("new-item recollection: batch Stan matches R, and its gradient matches finite differences", {
  skip_if_no_cmdstan()
  lp_at <- function(mod, data, u) {
    js <- tempfile(fileext = ".json"); cmdstanr::write_stan_json(data, js)
    up <- tempfile(fileext = ".json")
    writeLines(paste0('{"params_r": [', paste(sprintf("%.17g", u), collapse = ","), "]}"), up)
    out <- tempfile(fileext = ".csv")
    old <- Sys.getenv("PATH"); on.exit(Sys.setenv(PATH = old), add = TRUE)
    tool <- if (.Platform$OS.type == "windows") cmdstanr:::toolchain_PATH_env_var() else NULL
    Sys.setenv(PATH = paste(c(file.path(cmdstanr::cmdstan_path(), "stan/lib/stan_math/lib/tbb"), tool, old),
                            collapse = .Platform$path.sep))
    system2(mod$exe_file(), c("log_prob", paste0("unconstrained_params=", up), "data", paste0("file=", js),
                              "output", paste0("file=", out), "sig_figs=18"), stdout = FALSE, stderr = FALSE)
    as.numeric(as.matrix(utils::read.csv(out, comment.char = "#"))[1, ])
  }
  for (nm in names(cs <- .dpsd_rr_cases(.dpsd_rr_data()))) {
    m <- suppressMessages(broc(cs[[nm]][[1]], cs[[nm]][[2]]))
    f <- suppressMessages(stan_fit_broc(m, sampler = "nuts", chains = 1, iter_warmup = 100,
                                   iter_sampling = 10, seed = 1, refresh = 100))
    ll_r <- .obs_loglik_r(m, build_data_source_from_model(m), f$draws(format = "matrix"), re_formula = NULL)
    ll_s <- .read_loglik_SN_cols(.loglik_path(f), 1L, m$model_data$N)
    expect_lt(max(abs(ll_s - ll_r)), 1e-5, label = paste(nm, "Stan vs R log_lik"))
    mod <- .compile_broc_stan(m)
    D <- ncol(.cmdstan_log_prob(mod, m$stan_data, f$.backend_fit$output_files()[1])) - 1L
    set.seed(7)
    u <- stats::rnorm(D, 0, 0.3)
    h <- 1e-5
    g <- lp_at(mod, m$stan_data, u)[-1]
    fd <- vapply(seq_len(D), function(i) {
      e <- replace(numeric(D), i, h)
      (lp_at(mod, m$stan_data, u + e)[1] - lp_at(mod, m$stan_data, u - e)[1]) / (2 * h)
    }, numeric(1))
    expect_lt(max(abs(g - fd) / pmax(1, abs(g))), 1e-5, label = paste(nm, "gradient vs finite differences"))
  }
})
