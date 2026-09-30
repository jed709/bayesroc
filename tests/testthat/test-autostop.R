.warnings_of <- function(expr) {
  msgs <- character(0)
  value <- withCallingHandlers(expr, warning = function(w) {
    msgs <<- c(msgs, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
  list(value = value, warnings = msgs)
}

.autostop_model <- function() {
  d <- .fixture_crossed(n_sub = 6, n_item = 10)
  broc(brf(item_resp | is_old ~ 1 + (1 | sub), family = evsd()), d)
}

test_that("autostop_broc rejects the Stan backend", {
  m <- .autostop_model()
  expect_error(autostop_broc(m, backend = "stan"), "JAX-only")
})

test_that("warm continuation reproduces a single long run exactly", {
  skip_if_no_jax()
  m <- .autostop_model()
  f1 <- fit_broc(m, backend = "jax", chains = 2, parallel_chains = 2,
                 iter_warmup = 300, iter_sampling = 300, seed = 7,
                 save_state = TRUE, progress_bar = FALSE)
  expect_true(!is.null(f1$.state_path))
  expect_true(file.exists(f1$.state_path))

  f2 <- fit_broc(m, backend = "jax", chains = 2, parallel_chains = 2,
                 iter_warmup = 0, iter_sampling = 300, seed = 7,
                 warm_state = f1$.state_path, progress_bar = FALSE)
  fref <- fit_broc(m, backend = "jax", chains = 2, parallel_chains = 2,
                   iter_warmup = 300, iter_sampling = 600, seed = 7,
                   progress_bar = FALSE)

  pooled <- posterior::bind_draws(f1$.draws_array, f2$.draws_array,
                                  along = "iteration")
  vars <- grep("^log_lik", posterior::variables(pooled), value = TRUE,
               invert = TRUE)
  a <- posterior::summarise_draws(
    posterior::subset_draws(pooled, variable = vars), mean)
  b <- posterior::summarise_draws(
    posterior::subset_draws(fref$.draws_array, variable = vars), mean)
  mg <- merge(a, b, by = "variable", suffixes = c("_warm", "_ref"))
  expect_gt(nrow(mg), 0)
  expect_equal(mg$mean_warm, mg$mean_ref, tolerance = 1e-10)
})

test_that("warm rounds pool on iteration, cold rounds pool on chain", {
  skip_if_no_jax()
  m <- .autostop_model()
  fw <- autostop_broc(m, pop_target = 1, sd_target = 1, cor_target = 1,
                      chains = 2, iter_warmup = 200, samples_per_round = 200,
                      max_rounds = 2, restart = "warm", seed = 3,
                      progress = FALSE, progress_bar = FALSE)
  expect_equal(fw$.num_chains, 2L)
  expect_equal(fw$.iter_sampling, 200L)
  expect_equal(nrow(attr(fw, "ess_trace")), 1L)

  run <- .warnings_of(autostop_broc(m, pop_target = 1e9, sd_target = 1, cor_target = 1,
                                    chains = 2, iter_warmup = 200, samples_per_round = 200,
                                    max_rounds = 2, restart = "cold", seed = 3,
                                    progress = FALSE, progress_bar = FALSE))
  fc <- run$value
  expect_true(any(grepl("ESS targets not met after 2 round", run$warnings)))
  expect_equal(fc$.num_chains, 4L)
  expect_length(fc$diagnostic_summary()$num_divergent, 4L)
  expect_length(fw$diagnostic_summary()$num_divergent, 2L)
  expect_equal(fc$.iter_sampling, 200L)
  expect_equal(nrow(attr(fc, "ess_trace")), 2L)
})

test_that("ESS floors gate the loop and the trace records every round", {
  skip_if_no_jax()
  m <- .autostop_model()
  run <- .warnings_of(autostop_broc(m, pop_target = 1e9, sd_target = 1, cor_target = 1,
                                    chains = 2, iter_warmup = 200, samples_per_round = 200,
                                    max_rounds = 3, restart = "warm", seed = 5,
                                    progress = FALSE, progress_bar = FALSE))
  f <- run$value
  expect_true(any(grepl("ESS targets not met after 3 round", run$warnings)))
  tr <- attr(f, "ess_trace")
  expect_equal(nrow(tr), 3L)
  expect_true(all(diff(tr$total_draws) > 0))
  expect_equal(f$.iter_sampling, 600L)
  expect_equal(attr(f, "targets")[["pop"]], 1e9)
  expect_true(all(c("divergences", "cumulative_divergences") %in% names(tr)))
  expect_equal(tr$cumulative_divergences, cumsum(tr$divergences))
})

test_that("drghmc warm continuation reproduces a single long run exactly", {
  skip_if_no_jax()
  m <- .autostop_model()
  f1 <- fit_broc(m, backend = "jax", sampler = "drghmc", chains = 2,
                 parallel_chains = 2, iter_warmup = 300, iter_sampling = 300,
                 seed = 11, save_state = TRUE, progress_bar = FALSE)
  expect_true(!is.null(f1$.state_path))
  expect_true(file.exists(f1$.state_path))

  f2 <- fit_broc(m, backend = "jax", sampler = "drghmc", chains = 2,
                 parallel_chains = 2, iter_warmup = 0, iter_sampling = 300,
                 seed = 11, warm_state = f1$.state_path, progress_bar = FALSE)
  expect_equal(posterior::niterations(f2$.draws_array), 300L)

  fref <- fit_broc(m, backend = "jax", sampler = "drghmc", chains = 2,
                   parallel_chains = 2, iter_warmup = 300, iter_sampling = 600,
                   seed = 11, progress_bar = FALSE)
  pooled <- posterior::bind_draws(f1$.draws_array, f2$.draws_array, along = "iteration")
  vars <- grep("^log_lik", posterior::variables(pooled), value = TRUE, invert = TRUE)
  x <- posterior::subset_draws(pooled, variable = vars)
  y <- posterior::subset_draws(fref$.draws_array, variable = vars)
  expect_equal(unclass(x), unclass(y), tolerance = 1e-10, ignore_attr = TRUE)
})

test_that("ESS classes follow the parameter naming", {
  v <- c("beta_dprime[1]", "sigma_dprime_sub[1]", "sigma_criterion2_sub[2]",
         "sds_criterion_sw_1[1]", "sd_x", "corr_dprime_sub[1,2]", "L_corr_x[1,1]")
  expect_equal(.ess_class(v), c("pop", "sd", "sd", "sd", "sd", "cor", "cor"))
})

test_that("autostop_broc rejects arguments it controls and bad max_rounds", {
  m <- .autostop_model()
  expect_error(autostop_broc(m, iter_sampling = 5), "sets iter_sampling itself")
  expect_error(autostop_broc(m, loo = TRUE, save_loglik = TRUE, file_refit = TRUE,
                             seed = 1, max_rounds = 0), "positive whole number")
})

test_that("warm rounds through the worker reproduce one long run exactly", {
  skip_if_no_jax()
  m <- .autostop_model()
  run <- .warnings_of(autostop_broc(m, pop_target = 1e9, sd_target = 1, cor_target = 1,
                                    chains = 2, iter_warmup = 200, samples_per_round = 200,
                                    max_rounds = 2, restart = "warm", seed = 9,
                                    progress = FALSE, progress_bar = FALSE))
  fa <- run$value
  fref <- fit_broc(m, backend = "jax", chains = 2, parallel_chains = 2, iter_warmup = 200,
                   iter_sampling = 400, seed = 9 + 1009L, progress_bar = FALSE)
  vars <- grep("^log_lik", posterior::variables(fa$.draws_array), value = TRUE, invert = TRUE)
  x <- posterior::subset_draws(fa$.draws_array, variable = vars)
  y <- posterior::subset_draws(fref$.draws_array, variable = vars)
  expect_equal(unclass(x), unclass(y), tolerance = 1e-10, ignore_attr = TRUE)
})

test_that("autostop keeps the pooled log-likelihood and saves to file", {
  skip_if_no_jax()
  m <- .autostop_model()
  file <- tempfile(fileext = ".rds")
  f <- autostop_broc(m, pop_target = 1, sd_target = 1, cor_target = 1, chains = 2,
                     iter_warmup = 200, samples_per_round = 200, max_rounds = 1, seed = 2,
                     loo = TRUE, save_loglik = TRUE, file = file,
                     progress = FALSE, progress_bar = FALSE)
  expect_s3_class(f$.loo, "loo")
  expect_true(file.exists(f$.loglik_path))
  expect_equal(loo(f, cores = 1)$estimates, f$.loo$estimates)
  expect_true(file.exists(file))
  g <- autostop_broc(m, file = file, progress = FALSE)
  expect_equal(unclass(g$.draws_array), unclass(f$.draws_array), ignore_attr = TRUE)
  expect_equal(g$.loo$estimates, f$.loo$estimates)
})
