.worker_fit <- function(m, seed = 1) {
  fit_broc(m, backend = "jax", sampler = "drghmc", seed = seed, iter_warmup = 50, iter_sampling = 20,
           chains = 2, parallel_chains = 2, progress_bar = FALSE)
}

.worker_draws <- function(fit) unclass(fit$draws(format = "draws_array"))

test_that("JAX fits survive a worker that dies between fits, at start or mid-fit, with the same draws", {
  skip_on_cran()
  skip_if_no_jax()
  m <- .fx_model("uvsd")
  withr::defer({
    Sys.unsetenv(c("BAYESROC_WORKER_CRASH_ONCE", "BAYESROC_WORKER_CRASH_AT_START", "BAYESROC_JAX_WORKER"))
    .jax_worker_discard()
  })
  ref <- .worker_draws(.worker_fit(m))
  w1 <- .jax_state$worker
  expect_true(!is.null(w1) && w1$proc$is_alive())

  w1$proc$kill_tree()
  expect_false(w1$proc$is_alive())
  expect_identical(.worker_draws(.worker_fit(m)), ref)
  w2 <- .jax_state$worker
  expect_true(w2$proc$is_alive())
  expect_false(identical(w2$proc$get_pid(), w1$proc$get_pid()))

  .jax_worker_discard()
  marker <- tempfile("crash_start_")
  Sys.setenv(BAYESROC_WORKER_CRASH_AT_START = marker)
  expect_message(d3 <- .worker_draws(.worker_fit(m)), "stopped; restarting")
  expect_identical(d3, ref)
  expect_true(file.exists(marker))
  Sys.unsetenv("BAYESROC_WORKER_CRASH_AT_START")

  .jax_worker_discard()
  marker <- tempfile("crash_mid_")
  Sys.setenv(BAYESROC_WORKER_CRASH_ONCE = marker)
  expect_message(d4 <- .worker_draws(.worker_fit(m)), "stopped; restarting")
  expect_identical(d4, ref)
  Sys.unsetenv("BAYESROC_WORKER_CRASH_ONCE")
  expect_true(.jax_state$worker$proc$is_alive())

  Sys.setenv(BAYESROC_JAX_WORKER = "0")
  expect_identical(.worker_draws(.worker_fit(m)), ref)
  Sys.unsetenv("BAYESROC_JAX_WORKER")

  .jax_worker_discard()
  Sys.setenv(BAYESROC_WORKER_CRASH_AT_START = "always")
  expect_message(d6 <- .worker_draws(.worker_fit(m)), "could not be restarted; running this fit in its own process")
  expect_identical(d6, ref)
  Sys.unsetenv("BAYESROC_WORKER_CRASH_AT_START")
  expect_null(.jax_state$worker)
})

test_that("a worker that dies during auto sampling restarts the fit and reproduces it", {
  skip_on_cran()
  skip_if_no_jax()
  m <- .fx_model("uvsd")
  withr::defer({
    Sys.unsetenv("BAYESROC_WORKER_CRASH_ONCE")
    .jax_worker_discard()
  })
  auto_fit <- function() fit_broc(m, backend = "jax", sampler = "drghmc", seed = 2, iter_warmup = 50,
                                  iter_sampling = "auto", chains = 2, parallel_chains = 2, progress_bar = FALSE,
                                  control = list(max_draws = 40, check_every = 20))
  ref <- suppressWarnings(.worker_draws(auto_fit()))
  .jax_worker_discard()
  Sys.setenv(BAYESROC_WORKER_CRASH_ONCE = tempfile("crash_auto_"))
  expect_message(d2 <- suppressWarnings(.worker_draws(auto_fit())), "restarting it and the fit")
  expect_identical(d2, ref)
})
