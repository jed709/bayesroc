
.bd1 <- function(f) {
  mean(posterior::as_draws_matrix(f$draws(variables = "beta_dprime[1]"))[, 1])
}

test_that("JAX: thin, parallel_chains = 1 and the warning for unused `...` arguments", {
  skip_if_no_jax()
  m <- .fx_model("uvsd")
  expect_warning(
    fp <- fit_broc(m, sampler = "nuts", chains = 2L, parallel_chains = 2L, iter_warmup = 30L,
                   iter_sampling = 40L, thin = 2L, seed = 1, progress_bar = FALSE, sig_figs = 5),
    "ignored: sig_figs"
  )
  expect_equal(fp$.iter_sampling, 20L)
  fs <- fit_broc(m, sampler = "nuts", chains = 2L, parallel_chains = 1L, iter_warmup = 30L,
                 iter_sampling = 40L, thin = 2L, seed = 1, progress_bar = FALSE)
  expect_identical(dim(fs$draws()), dim(fp$draws()))
})

test_that("one chain, or chains run one at a time, compile and sample on both JAX samplers", {
  skip_if_no_jax()
  m <- suppressMessages(broc(brf(item_resp | is_old ~ 1 + (1 | sub), family = uvsd()),
                             .fixture_crossed(n_sub = 8, n_item = 16)))
  runs <- list(list("drghmc", 30, 1), list("nuts", 30, 1), list("drghmc", 30, 2), list("drghmc", "auto", 1))
  for (r in runs) {
    s <- r[[1]]; it <- r[[2]]; ch <- r[[3]]
    f <- suppressMessages(fit_broc(m, sampler = s, chains = ch, parallel_chains = 1,
                                   iter_warmup = 50, iter_sampling = it, seed = 1, progress_bar = FALSE,
                                   control = if (identical(it, "auto")) list(check_every = 50, max_draws = 100)))
    expect_true(all(is.finite(posterior::as_draws_matrix(f$draws(variables = "beta_dprime")))),
                label = paste(s, it, ch, "chains"))
  }
})
