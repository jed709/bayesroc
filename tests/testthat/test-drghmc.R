
test_that("drghmc: posterior means agree with nuts (jax)", {
  skip_if_no_jax()
  d <- .fixture_univariate(sigma = 1.3)
  m <- broc(brf(item_resp | is_old ~ 1, sigma ~ 1, family = uvsd()), d)
  fn <- fit_broc(m, backend = "jax", sampler = "nuts", seed = 1,
                 iter_warmup = 500, iter_sampling = 500, chains = 4,
                 progress_bar = FALSE)
  fj <- fit_broc(m, backend = "jax", sampler = "drghmc", seed = 1,
                 iter_warmup = 1000, iter_sampling = 3000, chains = 4,
                 progress_bar = FALSE)
  vars <- c("beta_dprime[1]", "beta_sigma[1]", "beta_thresh_mid[1]")
  sn <- posterior::as_draws_matrix(fn$draws(variables = vars))
  sj <- posterior::as_draws_matrix(fj$draws(variables = vars))
  for (v in vars) {
    expect_lt(abs(mean(sn[, v]) - mean(sj[, v])), 0.05,
              label = sprintf("|mean_nuts - mean_drghmc|(%s)", v))
  }
})

test_that("drghmc: diagnostics carry DR stages, no divergences/treedepth", {
  skip_if_no_jax()
  d <- .fixture_univariate(sigma = 1.3)
  m <- broc(brf(item_resp | is_old ~ 1, sigma ~ 1, family = uvsd()), d)
  fj <- fit_broc(m, backend = "jax", sampler = "drghmc", seed = 1,
                 iter_warmup = 800, iter_sampling = 1500, chains = 4,
                 progress_bar = FALSE)
  dd <- fj$diagnostic_summary()
  expect_true(all(c("accept", "dr_stage") %in% names(dd)))
  expect_named(dd$dr_stage, c("s1", "s2", "s3", "rej"))
  expect_equal(sum(dd$dr_stage), 1, tolerance = 1e-6)
  banner <- capture.output(print(summary(fj)))
  expect_true(any(grepl("Sampler \\(drghmc\\):", banner)))
  expect_false(any(grepl("divergent|treedepth", banner)))
})

test_that("drghmc: control knobs plumb through; unknown keys and short warmup error", {
  skip_if_no_jax()
  d <- .fixture_univariate(sigma = 1.3)
  m <- broc(brf(item_resp | is_old ~ 1, sigma ~ 1, family = uvsd()), d)
  expect_error(
    fit_broc(m, backend = "jax", sampler = "drghmc", control = list(bogus = 1),
             iter_warmup = 200, iter_sampling = 200, chains = 4,
             progress_bar = FALSE),
    "Unknown control key")
  expect_error(
    fit_broc(m, backend = "jax", sampler = "drghmc", iter_warmup = 5,
             iter_sampling = 200, chains = 4, progress_bar = FALSE),
    "needs iter_warmup >= control")
  fj <- fit_broc(m, backend = "jax", sampler = "drghmc",
                 control = list(precision = "f32"), seed = 1,
                 iter_warmup = 800, iter_sampling = 1500, chains = 4,
                 progress_bar = FALSE)
  sm <- fj$summary(variables = "beta_dprime[1]")
  expect_lt(sm$rhat[1], 1.05)
})

test_that("drghmc requires the jax backend", {
  d <- .fixture_univariate()
  m <- broc(brf(item_resp | is_old ~ 1, sigma ~ 1, family = uvsd()), d)
  expect_error(fit_broc(m, backend = "stan", sampler = "drghmc"),
               "requires backend = 'jax'")
})

test_that("drghmc accepts sample counts that are not a multiple of the chunk length", {
  skip_if_no_jax()
  m <- broc(brf(item_resp | is_old ~ 1, family = evsd()), .fixture_univariate())
  f <- fit_broc(m, backend = "jax", sampler = "drghmc", chains = 2, parallel_chains = 2,
                iter_warmup = 200, iter_sampling = 1001, seed = 3, progress_bar = FALSE)
  expect_equal(posterior::niterations(f$.draws_array), 1001L)
  f2 <- fit_broc(m, backend = "jax", sampler = "drghmc", chains = 2, parallel_chains = 2,
                 iter_warmup = 200, iter_sampling = 1001, thin = 7, seed = 3, progress_bar = FALSE)
  expect_equal(posterior::niterations(f2$.draws_array), 143L)
})

test_that("drghmc recovers NUTS posterior means and SDs for dr_k 2 and 3", {
  skip_if_no_jax()
  d <- .fixture_univariate(sigma = 1.3, n_per = 400)
  m <- broc(brf(item_resp | is_old ~ 1, sigma ~ 1, family = uvsd()), d)
  vars <- c("beta_dprime[1]", "beta_sigma[1]", "beta_thresh_mid[1]")
  ref <- fit_broc(m, backend = "jax", chains = 4, iter_warmup = 1000, iter_sampling = 4000,
                  seed = 5, progress_bar = FALSE)
  rs <- posterior::summarise_draws(posterior::subset_draws(ref$.draws_array, variable = vars),
                                   "mean", "sd", "mcse_mean")
  for (k in 2:3) {
    f <- fit_broc(m, backend = "jax", sampler = "drghmc", control = list(dr_k = k),
                  chains = 4, iter_warmup = 1000, iter_sampling = 4000, seed = 6,
                  progress_bar = FALSE)
    fs <- posterior::summarise_draws(posterior::subset_draws(f$.draws_array, variable = vars),
                                     "mean", "sd", "mcse_mean")
    se <- sqrt(rs$mcse_mean^2 + fs$mcse_mean^2)
    expect_true(all(abs(fs$mean - rs$mean) < 5 * se), label = paste("means, dr_k", k))
    expect_true(all(abs(fs$sd / rs$sd - 1) < 0.1), label = paste("SDs, dr_k", k))
  }
})

test_that("drghmc seeds do not share random streams", {
  skip_if_no_jax()
  m <- broc(brf(item_resp | is_old ~ 1, family = evsd()), .fixture_univariate())
  a <- fit_broc(m, backend = "jax", sampler = "drghmc", chains = 2, parallel_chains = 2,
                iter_warmup = 100, iter_sampling = 100, seed = 1, progress_bar = FALSE)
  b <- fit_broc(m, backend = "jax", sampler = "drghmc", chains = 2, parallel_chains = 2,
                iter_warmup = 100, iter_sampling = 100, seed = 94, progress_bar = FALSE)
  v <- "beta_dprime[1]"
  expect_false(isTRUE(all.equal(as.numeric(a$.draws_array[, , v]),
                                as.numeric(b$.draws_array[, , v]))))
})

test_that("drghmc runs chains sequentially when parallel_chains < chains", {
  skip_if_no_jax()
  d <- .fixture_univariate(sigma = 1.3)
  m <- broc(brf(item_resp | is_old ~ 1, sigma ~ 1, family = uvsd()), d)
  fit <- function(pc) fit_broc(m, backend = "jax", sampler = "drghmc", seed = 7, chains = 2,
                               parallel_chains = pc, iter_warmup = 200, iter_sampling = 300,
                               progress_bar = FALSE)
  fs <- fit(1)
  fp <- fit(2)
  expect_equal(dim(fs$.draws_array)[1:2], c(300L, 2L))
  v <- c("beta_dprime[1]", "beta_sigma[1]", "beta_thresh_mid[1]")
  ms <- colMeans(posterior::as_draws_matrix(fs$draws(variables = v)))
  mp <- colMeans(posterior::as_draws_matrix(fp$draws(variables = v)))
  expect_lt(max(abs(ms - mp)), 0.05)
})

test_that("f32 fits (NUTS and DR-G-HMC) sum the likelihood relative to a warmed-up offset", {
  skip_if_no_jax()
  d <- .fixture_univariate(sigma = 1.3)
  m <- broc(brf(item_resp | is_old ~ 1, sigma ~ 1, family = uvsd()), d)
  v <- c("beta_dprime[1]", "beta_sigma[1]", "beta_thresh_mid[1]")
  ref <- fit_broc(m, backend = "jax", seed = 3, chains = 2, iter_warmup = 300, iter_sampling = 600,
                  progress_bar = FALSE)
  mr <- colMeans(posterior::as_draws_matrix(ref$draws(variables = v)))
  sr <- apply(posterior::as_draws_matrix(ref$draws(variables = v)), 2, sd)
  for (smp in c("nuts", "drghmc")) {
    f <- fit_broc(m, backend = "jax", sampler = smp, seed = 3, chains = 2, iter_warmup = 300,
                  iter_sampling = 600, control = list(precision = "f32"), progress_bar = FALSE)
    expect_equal(sum(f$diagnostic_summary()$num_divergent), 0, label = smp)
    mf <- colMeans(posterior::as_draws_matrix(f$draws(variables = v)))
    expect_true(all(abs(mf - mr) < 0.5 * sr), label = paste(smp, "f32 means vs f64"))
  }
})

test_that("drghmc samples weakly identified standard deviations from the NUTS posterior", {
  skip_if_no_jax()
  d <- .fixture_crossed()
  m <- broc(brf(item_resp | is_old ~ 1 + (1 | sub) + (1 | item), family = evsd()), d)
  ref <- fit_broc(m, backend = "jax", chains = 4, iter_warmup = 1000, iter_sampling = 6000,
                  seed = 3, progress_bar = FALSE, control = list(adapt_delta = 0.99))
  f <- fit_broc(m, backend = "jax", sampler = "drghmc", chains = 4, iter_warmup = 1500,
                iter_sampling = 12000, seed = 4, progress_bar = FALSE)
  vars <- grep("^sigma_", posterior::variables(ref$.draws_array), value = TRUE)
  expect_gt(length(vars), 1L)
  lg <- function(fit) log(posterior::subset_draws(fit$.draws_array, variable = vars))
  sm <- function(fit) posterior::summarise_draws(lg(fit), "mean", "sd", "mcse_mean",
                                                 q05 = function(x) stats::quantile(x, 0.05, names = FALSE),
                                                 lo_se = function(x) posterior::mcse_quantile(x, 0.05, names = FALSE))
  rs <- sm(ref)
  fs <- sm(f)
  expect_true(all(is.finite(c(rs$lo_se, fs$lo_se))))
  expect_true(all(abs(fs$mean - rs$mean) < 5 * sqrt(rs$mcse_mean^2 + fs$mcse_mean^2)), label = "log-SD means")
  expect_true(all(abs(fs$sd / rs$sd - 1) < 0.15), label = "log-SD spreads")
  expect_true(all(abs(fs$q05 - rs$q05) < 5 * sqrt(rs$lo_se^2 + fs$lo_se^2)), label = "log-SD lower tails")
})
