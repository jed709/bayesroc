# Helpers shared with tests/testthat/
# test-re-duplicate-group.R (copied).

re_dup_data <- function() {
  d <- .fixture_crossed()
  set.seed(11)
  d$x <- round(rnorm(nrow(d)), 2)
  d$z <- round(rnorm(nrow(d)), 2)
  d
}
.blocks_of <- function(f, d, pname = "criterion") {
  m <- suppressMessages(broc(f, data = d))
  re <- if (pname == "criterion") m$model_data$criterion$random$sub else m$model_data[[paste0(pname, "_random")]]$sub
  lapply(re$cor_blocks, as.integer)
}

test_that("block-diagonal threshold correlations: Stan and JAX agree and cross-block correlations are zero", {
  skip_if_not_slow()
  skip_if_no_cmdstan(); skip_if_no_jax()
  d <- .fixture_crossed(n_sub = 12, n_item = 24)
  set.seed(3)
  d$x <- round(rnorm(nrow(d)), 2)
  m <- suppressMessages(broc(brf(item_resp | is_old ~ 1,
                                 criterion ~ 1 + x + (1 + x | gr(sub, cor_thresholds = FALSE)),
                                 family = evsd()), data = d))
  a <- list(chains = 4, parallel_chains = 4, iter_warmup = 1000, iter_sampling = 1000, seed = 2)
  fj <- do.call(fit_broc, c(list(m, backend = "jax", progress_bar = FALSE), a))
  fs <- do.call(stan_fit_broc, c(list(m, refresh = 50), a))
  params <- function(f) grep("^(beta_|sigma_|corr_|L_corr_)",
                             unique(sub("\\[.*$", "", posterior::variables(f$draws()))), value = TRUE)
  expect_setequal(params(fj), params(fs))
  expect_true(all(c("L_corr_criterion_sub", "L_corr_criterion_sub_b1", "L_corr_criterion_sub_b5") %in%
                    params(fj)))
  expect_equal(max(abs(posterior::extract_variable(fj$draws(variables = "corr_criterion_sub"),
                                                   "corr_criterion_sub[1,3]"))), 0)
  v <- c("beta_thresh_mid[1]", "beta_thresh_mid[2]", "sigma_criterion_sub[1]", "sigma_criterion_sub[2]",
         "corr_criterion_sub[1,2]", "corr_criterion_sub[3,4]")
  sj <- posterior::summarise_draws(fj$draws(variables = v), "mean", "sd", "ess_bulk")
  ss <- posterior::summarise_draws(fs$draws(variables = v), "mean", "sd", "ess_bulk")
  z <- abs(sj$mean - ss$mean) / sqrt(sj$sd^2 / sj$ess_bulk + ss$sd^2 / ss$ess_bulk)
  expect_lt(max(z), 4)
})

test_that("block-diagonal threshold correlations: Stan and JAX name the same parameters, cross-block correlations are zero, and R reproduces both log-likelihoods", {
  skip_if_no_cmdstan(); skip_if_no_jax()
  d <- .fixture_crossed(n_sub = 12, n_item = 24)
  set.seed(3)
  d$x <- round(rnorm(nrow(d)), 2)
  m <- suppressMessages(broc(brf(item_resp | is_old ~ 1,
                                 criterion ~ 1 + x + (1 + x | gr(sub, cor_thresholds = FALSE)),
                                 family = evsd()), data = d))
  fj <- suppressMessages(fit_broc(m, backend = "jax", chains = 1, iter_warmup = 50, iter_sampling = 10, seed = 2,
                                  progress_bar = FALSE))
  fs <- suppressMessages(stan_fit_broc(m, chains = 1, iter_warmup = 20, iter_sampling = 5, seed = 2,
                                  refresh = 50, show_messages = FALSE))
  params <- function(f) grep("^(beta_|sigma_|corr_|L_corr_)",
                             unique(sub("\\[.*$", "", posterior::variables(f$draws()))), value = TRUE)
  expect_setequal(params(fj), params(fs))
  expect_true(all(c("L_corr_criterion_sub", "L_corr_criterion_sub_b1", "L_corr_criterion_sub_b5") %in%
                    params(fj)))
  for (f in list(fj, fs)) {
    expect_equal(max(abs(posterior::extract_variable(f$draws(variables = "corr_criterion_sub"),
                                                     "corr_criterion_sub[1,3]"))), 0)
  }
  src <- build_data_source_from_model(m)
  ll_j <- .read_loglik_SN_cols(.loglik_path(fj), 1L, m$model_data$N)
  expect_lt(max(abs(ll_j - .obs_loglik_r(m, src, fj$draws(format = "matrix"), re_formula = NULL))), 1e-5)
  ll_s <- unname(as.matrix(posterior::as_draws_matrix(fs$draws(variables = "log_lik"))))
  expect_lt(max(abs(ll_s - .obs_loglik_r(m, src, fs$draws(format = "matrix"), re_formula = NULL))), 1e-5)
})
