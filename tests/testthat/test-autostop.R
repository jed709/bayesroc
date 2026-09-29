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

  fc <- autostop_broc(m, pop_target = 1e9, sd_target = 1, cor_target = 1,
                      chains = 2, iter_warmup = 200, samples_per_round = 200,
                      max_rounds = 2, restart = "cold", seed = 3,
                      progress = FALSE, progress_bar = FALSE)
  expect_equal(fc$.num_chains, 4L)
  expect_equal(fc$.iter_sampling, 200L)
  expect_equal(nrow(attr(fc, "ess_trace")), 2L)
})

test_that("ESS floors gate the loop and the trace records every round", {
  skip_if_no_jax()
  m <- .autostop_model()
  f <- autostop_broc(m, pop_target = 1e9, sd_target = 1, cor_target = 1,
                     chains = 2, iter_warmup = 200, samples_per_round = 200,
                     max_rounds = 3, restart = "warm", seed = 5,
                     progress = FALSE, progress_bar = FALSE)
  tr <- attr(f, "ess_trace")
  expect_equal(nrow(tr), 3L)
  expect_true(all(diff(tr$total_draws) > 0))
  expect_true(all(diff(tr$min_pop_ess) > 0))
  expect_equal(f$.iter_sampling, 600L)
  expect_equal(attr(f, "targets")[["pop"]], 1e9)
  expect_true(all(c("divergences", "cumulative_divergences") %in% names(tr)))
  expect_equal(tr$cumulative_divergences, cumsum(tr$divergences))
})

test_that("drghmc returns and consumes a resume state", {
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

  v <- "beta_dprime[1]"
  a <- posterior::summarise_draws(
    posterior::subset_draws(f1$.draws_array, variable = v), mean)$mean
  b <- posterior::summarise_draws(
    posterior::subset_draws(f2$.draws_array, variable = v), mean)$mean
  expect_lt(abs(a - b), 0.5)
})
