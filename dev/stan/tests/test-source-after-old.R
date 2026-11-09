# Helpers shared with tests/testthat/
# test-source-after-old.R (copied).

.sao_data <- function(partial = TRUE) {
  set.seed(3)
  bv <- .fixture_bivariate_sdt(n_per = 80)
  bv$sub <- factor(sample(1:6, nrow(bv), TRUE))
  bv$y2[bv$y1 <= 2] <- NA
  if (partial) bv$y2[sample(which(bv$y1 == 3), 5)] <- NA
  bv
}
.sao_cases <- function(bv) {
  f2 <- function(fam, ...) brf(resp(y1, y2) | item_type ~ 1 + (1 | sub), discrim ~ 1, rho ~ 1, criterion ~ 1,
                               criterion2 ~ 1, ..., family = fam)
  bc <- .fixture_bivariate_cumulative(n = 200)
  bc$y2[bc$y1 <= 2] <- NA
  list(
    bg = list(f2(bivariate_gaussian()), bv),
    bg_bounded = list(f2(bivariate_gaussian(bounded = TRUE), sigma ~ 1), bv),
    bg_varying = list(f2(bivariate_gaussian(source_criteria = "varying")), bv),
    bdp = list(f2(bivariate_dp(), lambda ~ 1, lambda2 ~ 1, sigma ~ 1), bv),
    vrdp2d_varying = list(brf(resp(y1, y2) | item_type ~ 1 + (1 | sub), dprime2 ~ 1, discrim ~ 1, lambda ~ 1, sigma ~ 1,
                              criterion ~ 1, criterion2 ~ 1, family = vrdp2d(source_criteria = "varying")), bv),
    bcum = list(brf(resp(y1, y2) ~ 1, cutpoints ~ 1, cutpoints2 ~ 1, rho ~ 1, family = bivariate_cumulative()), bc)
  )
}

test_that("rows without a source rating: batch Stan matches R, and its gradient matches finite differences", {
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
  for (nm in names(cs <- .sao_cases(.sao_data()))) {
    m <- suppressMessages(broc(cs[[nm]][[1]], cs[[nm]][[2]]))
    f <- suppressMessages(stan_fit_broc(m, sampler = "nuts", chains = 1, iter_warmup = 100,
                                   iter_sampling = 10, seed = 1, refresh = 100))
    ll_r <- .obs_loglik_r(m, build_data_source_from_model(m), f$draws(format = "matrix"), re_formula = NULL)
    ll_s <- .read_loglik_SN_cols(.loglik_path(f), 1L, m$model_data$N)
    expect_lt(max(abs(ll_s - ll_r)), 1e-5, label = paste(nm, "Stan vs R log_lik"))
    if (!nm %in% c("bg", "bg_bounded", "bdp", "vrdp2d_varying")) next
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
