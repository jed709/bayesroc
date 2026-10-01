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

.same_draws <- function(a, b) {
  vars <- grep("^log_lik", posterior::variables(a$.draws_array), value = TRUE, invert = TRUE)
  x <- posterior::subset_draws(a$.draws_array, variable = vars)
  y <- posterior::subset_draws(b$.draws_array, variable = vars)
  expect_equal(unclass(x), unclass(y), tolerance = 1e-10, ignore_attr = TRUE)
}

test_that("autostop_broc rejects the Stan backend", {
  m <- .autostop_model()
  expect_error(autostop_broc(m, backend = "stan"), "JAX-only")
})

test_that("autostop_broc rejects arguments it controls and invalid settings", {
  m <- .autostop_model()
  expect_error(autostop_broc(m, iter_sampling = 5), "sets iter_sampling itself")
  expect_error(autostop_broc(m, max_draws = 0), "positive whole number")
  expect_error(autostop_broc(m, check_every = 2.5), "positive whole number")
  expect_error(autostop_broc(m, max_time = -1), "positive number of seconds")
  expect_error(autostop_broc(m, rhat_target = 0.9), "at least 1")
  expect_error(autostop_broc(m, ess_target = c(pop = 100, slope = 50)), "names must be pop")
  expect_error(autostop_broc(m, ess_target = c(100, 200)), "single number")
})

test_that("per-class ESS targets fill unnamed classes with 400", {
  expect_equal(.ess_targets(250), c(pop = 250, sd = 250, cor = 250))
  expect_equal(.ess_targets(c(pop = 2000)), c(pop = 2000, sd = 400, cor = 400))
  expect_equal(.ess_targets(c(sd = 100, cor = 50)), c(pop = 400, sd = 100, cor = 50))
})

test_that("ESS classes follow the parameter naming and skip per-level draws", {
  v <- c("beta_dprime[1]", "sigma_dprime_sub", "sigma_criterion2_sub[2]",
         "sds_criterion_sw_1[1]", "sd_x", "corr_dprime_sub[1,2]")
  expect_equal(.ess_class(v), c("pop", "sd", "sd", "sd", "sd", "cor"))
  expect_equal(.monitored(c(v, "u_dprime_sub[1]", "u_dprime_sub_raw[1]", "L_corr_dprime_sub[2,1]",
                            "zs_criterion_sw_1[1,1]", "s_criterion_sw_1[1,1]", "log_lik[3]", "lp__")), v)
})

test_that("warm continuation reproduces a single long run exactly", {
  skip_if_no_jax()
  m <- .autostop_model()
  f1 <- fit_broc(m, backend = "jax", chains = 2, parallel_chains = 2,
                 iter_warmup = 300, iter_sampling = 300, seed = 7,
                 save_state = TRUE, progress_bar = FALSE)
  expect_true(file.exists(f1$.state_path))
  f2 <- fit_broc(m, backend = "jax", chains = 2, parallel_chains = 2,
                 iter_warmup = 0, iter_sampling = 300, seed = 7,
                 warm_state = f1$.state_path, progress_bar = FALSE)
  fref <- fit_broc(m, backend = "jax", chains = 2, parallel_chains = 2,
                   iter_warmup = 300, iter_sampling = 600, seed = 7, progress_bar = FALSE)
  pooled <- list(.draws_array = posterior::bind_draws(f1$.draws_array, f2$.draws_array,
                                                      along = "iteration"))
  .same_draws(pooled, fref)
})

test_that("drghmc warm continuation reproduces a single long run exactly", {
  skip_if_no_jax()
  m <- .autostop_model()
  f1 <- fit_broc(m, backend = "jax", sampler = "drghmc", chains = 2,
                 parallel_chains = 2, iter_warmup = 300, iter_sampling = 300,
                 seed = 11, save_state = TRUE, progress_bar = FALSE)
  expect_true(file.exists(f1$.state_path))
  f2 <- fit_broc(m, backend = "jax", sampler = "drghmc", chains = 2,
                 parallel_chains = 2, iter_warmup = 0, iter_sampling = 300,
                 seed = 11, warm_state = f1$.state_path, progress_bar = FALSE)
  fref <- fit_broc(m, backend = "jax", sampler = "drghmc", chains = 2,
                   parallel_chains = 2, iter_warmup = 300, iter_sampling = 600,
                   seed = 11, progress_bar = FALSE)
  pooled <- list(.draws_array = posterior::bind_draws(f1$.draws_array, f2$.draws_array,
                                                      along = "iteration"))
  .same_draws(pooled, fref)
})

test_that("autostop chunks reproduce one run of the final length exactly (NUTS)", {
  skip_if_no_jax()
  m <- .autostop_model()
  run <- .warnings_of(autostop_broc(m, ess_target = 1e9, max_draws = 300, check_every = 100,
                                    chains = 2, iter_warmup = 200, seed = 9,
                                    progress = FALSE, progress_bar = FALSE))
  f <- run$value
  expect_identical(attr(f, "stop_reason"), "max_draws")
  expect_true(any(grepl("stopped at 300 draws per chain", run$warnings)))
  expect_equal(f$.iter_sampling, 300L)
  tr <- attr(f, "ess_trace")
  expect_equal(tr$draws[1], 100)
  expect_equal(tail(tr$draws, 1), 300)
  expect_true(all(diff(tr$draws) >= 100))
  expect_true(all(c("min_pop_ess", "min_sd_ess", "min_cor_ess", "max_rhat", "divergences") %in% names(tr)))
  fref <- fit_broc(m, backend = "jax", chains = 2, parallel_chains = 2, iter_warmup = 200,
                   iter_sampling = 300, seed = 9, progress_bar = FALSE)
  .same_draws(f, fref)
})

test_that("autostop chunks reproduce one run of the final length exactly (DR-G-HMC)", {
  skip_if_no_jax()
  m <- .autostop_model()
  f <- suppressWarnings(autostop_broc(m, ess_target = 1e9, max_draws = 300, check_every = 100,
                                      chains = 2, iter_warmup = 300, seed = 4, sampler = "drghmc",
                                      progress = FALSE, progress_bar = FALSE))
  fref <- fit_broc(m, backend = "jax", sampler = "drghmc", chains = 2, parallel_chains = 2,
                   iter_warmup = 300, iter_sampling = 300, seed = 4, progress_bar = FALSE)
  .same_draws(f, fref)
})

test_that("autostop stops once the targets are met and on the time cap", {
  skip_if_no_jax()
  m <- .autostop_model()
  f <- autostop_broc(m, ess_target = 1, rhat_target = 10, check_every = 100, chains = 2,
                     iter_warmup = 200, seed = 3, progress = FALSE, progress_bar = FALSE)
  expect_identical(attr(f, "stop_reason"), "targets")
  expect_equal(f$.iter_sampling, 100L)
  run <- .warnings_of(autostop_broc(m, ess_target = 1e9, max_time = 1e-6, check_every = 100,
                                    chains = 2, iter_warmup = 200, seed = 3,
                                    progress = FALSE, progress_bar = FALSE))
  expect_identical(attr(run$value, "stop_reason"), "max_time")
  expect_equal(run$value$.iter_sampling, 100L)
})

test_that("resume continues a capped run exactly", {
  skip_if_no_jax()
  m <- .autostop_model()
  f1 <- suppressWarnings(autostop_broc(m, ess_target = 1e9, max_draws = 200, check_every = 100,
                                       chains = 2, iter_warmup = 200, seed = 6,
                                       progress = FALSE, progress_bar = FALSE))
  f2 <- suppressWarnings(autostop_broc(m, ess_target = 1e9, max_draws = 400, check_every = 100,
                                       chains = 2, resume = f1, progress = FALSE,
                                       progress_bar = FALSE))
  expect_equal(f2$.iter_sampling, 400L)
  expect_equal(tail(attr(f2, "ess_trace")$draws, 1), 400)
  expect_equal(attr(f2, "ess_trace")$draws[seq_along(attr(f1, "ess_trace")$draws)],
               attr(f1, "ess_trace")$draws)
  fref <- fit_broc(m, backend = "jax", chains = 2, parallel_chains = 2, iter_warmup = 200,
                   iter_sampling = 400, seed = 6, progress_bar = FALSE)
  .same_draws(f2, fref)
})

test_that("autostop keeps the pooled log-likelihood and saves to file", {
  skip_if_no_jax()
  m <- .autostop_model()
  file <- tempfile(fileext = ".rds")
  f <- suppressWarnings(autostop_broc(m, ess_target = 1e9, max_draws = 200, check_every = 100,
                                      chains = 2, iter_warmup = 200, seed = 2, loo = TRUE,
                                      save_loglik = TRUE, file = file, progress = FALSE,
                                      progress_bar = FALSE))
  expect_s3_class(f$.loo, "loo")
  expect_true(file.exists(f$.loglik_path))
  expect_equal(dim(read_npy(f$.loglik_path))[1:2], c(2L, 200L))
  expect_equal(loo(f, cores = 1)$estimates, f$.loo$estimates)
  expect_true(file.exists(file))
  g <- autostop_broc(m, file = file, progress = FALSE)
  expect_equal(unclass(g$.draws_array), unclass(f$.draws_array), ignore_attr = TRUE)
  expect_equal(g$.loo$estimates, f$.loo$estimates)
})

test_that("chunk length grows in powers of two to balance checking against overshoot", {
  expect_equal(.next_chunk(100, 1000, 1000, t_check = 0.001, t_draw = 1), 100L)
  expect_equal(.next_chunk(100, 4000, 4000, t_check = 0.8, t_draw = 0.02), 400L)
  expect_equal(.next_chunk(100, 4000, 8000, t_check = 0.8, t_draw = 0.02), 800L)
})

test_that("a saved and reloaded fit resumes exactly and keeps its autostop attributes", {
  skip_if_no_jax()
  m <- .autostop_model()
  file <- tempfile(fileext = ".rds")
  f1 <- suppressWarnings(autostop_broc(m, ess_target = 1e9, max_draws = 200, check_every = 100,
                                       chains = 2, iter_warmup = 200, seed = 8, file = file,
                                       progress = FALSE, progress_bar = FALSE))
  unlink(f1$.state_path)
  expect_true(file.exists(sub("[.]rds$", "_state.pkl", file)))
  g <- load_broc_fit(file)
  expect_true(file.exists(g$.state_path))
  expect_identical(attr(g, "stop_reason"), "max_draws")
  expect_equal(attr(g, "ess_trace"), attr(f1, "ess_trace"))
  expect_identical(attr(g, "model_fingerprint"), attr(f1, "model_fingerprint"))
  f2 <- suppressWarnings(autostop_broc(m, ess_target = 1e9, max_draws = 400, check_every = 100,
                                       chains = 2, resume = g, progress = FALSE, progress_bar = FALSE))
  fref <- fit_broc(m, backend = "jax", chains = 2, parallel_chains = 2, iter_warmup = 200,
                   iter_sampling = 400, seed = 8, progress_bar = FALSE)
  .same_draws(f2, fref)
})

test_that("resume refuses a fit made on a different model or data", {
  skip_if_no_jax()
  m <- .autostop_model()
  f1 <- suppressWarnings(autostop_broc(m, ess_target = 1e9, max_draws = 100, check_every = 100,
                                       chains = 2, iter_warmup = 100, seed = 1,
                                       progress = FALSE, progress_bar = FALSE))
  d2 <- .fixture_crossed(n_sub = 6, n_item = 10)
  d2$item_resp <- rev(d2$item_resp)
  m2 <- broc(brf(item_resp | is_old ~ 1 + (1 | sub), family = evsd()), d2)
  expect_error(autostop_broc(m2, chains = 2, resume = f1, progress = FALSE), "different model or data")
  expect_identical(.model_fingerprint(m), attr(f1, "model_fingerprint"))
})
