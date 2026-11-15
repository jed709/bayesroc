test_that("hmc: the JAX default; banner without treedepth, shapes follow thinning", {
  skip_if_no_jax()
  m <- .fx_model("uvsd")
  f <- fit_broc(m, seed = 1, iter_warmup = 150, iter_sampling = 37, thin = 2,
                chains = 2, parallel_chains = 2, progress_bar = FALSE)
  expect_identical(f$.sampler, "hmc")
  expect_equal(dim(f$draws())[1:2], c(19L, 2L))
  dd <- f$diagnostic_summary()
  expect_null(dd$num_max_treedepth)
  expect_equal(sum(dd$num_divergent), 0)
  banner <- capture.output(print(summary(f)))
  expect_true(any(grepl("Sampler \\(hmc\\):", banner)))
  expect_false(any(grepl("treedepth", banner)))
})

test_that("hmc: sequential chains and transport = FALSE run", {
  skip_if_no_jax()
  m <- .fx_model("uvsd")
  f1 <- fit_broc(m, seed = 2, iter_warmup = 150, iter_sampling = 20,
                 chains = 2, parallel_chains = 1, progress_bar = FALSE)
  expect_equal(dim(f1$draws())[1:2], c(20L, 2L))
  f2 <- fit_broc(m, seed = 3, iter_warmup = 150, iter_sampling = 20,
                 chains = 2, parallel_chains = 2, control = list(transport = FALSE), progress_bar = FALSE)
  expect_equal(dim(f2$draws())[1:2], c(20L, 2L))
})

test_that("hmc: control keys", {
  skip_if_no_jax()
  m <- .fx_model("uvsd")
  expect_error(fit_broc(m, sampler = "hmc", control = list(max_treedepth = 8)),
               "Unknown control key")
  expect_error(fit_broc(m, sampler = "hmc", control = list(adapt_gamma = -1)), "positive number")
  f <- fit_broc(m, seed = 4, iter_warmup = 150, iter_sampling = 20, chains = 2, parallel_chains = 2,
                control = list(adapt_gamma = 0.05), progress_bar = FALSE)
  expect_equal(dim(f$draws())[1:2], c(20L, 2L))
})

test_that("a two-site model without a transport map initialises its chains in parallel", {
  skip_if_not_slow()
  skip_if_no_jax()
  m <- suppressMessages(broc(brf(item_resp ~ 1, cutpoints ~ 1, family = cumulative()), data = .fixture_univariate()))
  expect_no_error(fit <- fit_broc(m, chains = 2, iter_warmup = 100, iter_sampling = 100,
                                  seed = 1))
  expect_equal(posterior::ndraws(fit$draws()), 200L)
})
