.warnings_of <- function(expr) {
  msgs <- character(0)
  value <- withCallingHandlers(expr, warning = function(w) {
    msgs <<- c(msgs, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
  list(value = value, warnings = msgs)
}

.auto_model <- function() {
  d <- .fixture_crossed(n_sub = 6, n_item = 10)
  broc(brf(item_resp | is_old ~ 1 + (1 | sub), family = evsd()), d)
}

.vars_no_ll <- function(f) grep("^log_lik", posterior::variables(f$.draws_array), value = TRUE, invert = TRUE)

.same_draws <- function(a, b, rows = NULL) {
  vars <- .vars_no_ll(a)
  x <- unclass(posterior::subset_draws(a$.draws_array, variable = vars))
  y <- unclass(posterior::subset_draws(b$.draws_array, variable = vars))
  if (!is.null(rows)) y <- y[rows, , , drop = FALSE]
  expect_equal(x, y, tolerance = 1e-10, ignore_attr = TRUE)
}

.auto_warmup <- function(sampler) if (sampler == "drghmc") 50 else 30

.fit_fixed <- function(m, n, seed, sampler = "nuts", warmup = .auto_warmup(sampler)) {
  fit_broc(m, backend = "jax", sampler = sampler, chains = 2, parallel_chains = 2,
           iter_warmup = warmup, iter_sampling = n, seed = seed, progress_bar = FALSE)
}

.fit_auto_capped <- function(m, max_draws, seed, sampler = "nuts", warmup = .auto_warmup(sampler), ...) {
  ctrl <- utils::modifyList(list(ess_target = 1e9, max_draws = max_draws, check_every = 25), list(...))
  .warnings_of(fit_broc(m, backend = "jax", sampler = sampler, chains = 2, parallel_chains = 2,
                        iter_warmup = warmup, iter_sampling = "auto", seed = seed,
                        control = ctrl, progress_bar = FALSE))
}

test_that("iter_sampling = 'auto' and resume need the JAX backend", {
  m <- .auto_model()
  expect_error(fit_broc(m, backend = "stan", iter_sampling = "auto"), "needs backend = \"jax\"")
})

test_that("auto settings are validated and only accepted with iter_sampling = 'auto'", {
  m <- .auto_model()
  bad <- function(...) fit_broc(m, backend = "jax", sampler = "nuts", iter_sampling = "auto", control = list(...))
  expect_error(bad(max_draws = 0), "positive whole number")
  expect_error(bad(check_every = 2.5), "positive whole number")
  expect_error(bad(max_kept = -1), "positive whole number")
  expect_error(bad(max_time = -1), "positive number of seconds")
  expect_error(bad(rhat_target = 0.9), "at least 1")
  expect_error(bad(ess_target = c(pop = 100, slope = 50)), "names must be pop")
  expect_error(bad(ess_target = c(100, 200)), "single number")
  expect_error(fit_broc(m, backend = "jax", sampler = "nuts", control = list(ess_target = 100)),
               "only applies with iter_sampling")
  expect_error(fit_broc(m, backend = "jax", sampler = "nuts", iter_sampling = "lots"), "positive whole number")
  expect_error(fit_broc(m, backend = "jax", sampler = "nuts", save_loglik = NULL), "TRUE or FALSE")
})

test_that("auto defaults keep every draw, and thin overrides max_kept", {
  a <- .auto_settings(list(), "nuts", 4)
  expect_equal(a$targets, c(pop = 400, sd = 400, cor = 400))
  expect_equal(c(a$check_every, a$max_draws, a$max_kept, a$rhat), c(100, 50000, 50000, 1.01))
  d <- .auto_settings(list(ess_target = c(pop = 4000)), "drghmc", 4)
  expect_equal(c(d$check_every, d$max_draws, d$max_kept), c(1000, 500000, 500000))
  k <- .auto_settings(list(max_kept = 100), "drghmc", 4, thin = 4L)
  expect_equal(k$max_kept, 125000L)
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

test_that("auto rounds reproduce one run of the final length exactly (NUTS)", {
  skip_if_no_jax()
  m <- .auto_model()
  run <- .fit_auto_capped(m, 75, seed = 9)
  f <- run$value
  expect_identical(attr(f, "stop_reason"), "max_draws")
  expect_true(any(grepl("stopped at 75 draws per chain", run$warnings)))
  expect_equal(f$.iter_sampling, 75L)
  tr <- attr(f, "ess_trace")
  expect_equal(tr$draws, c(25, 50, 75))
  expect_true(all(c("kept", "min_pop_ess", "min_sd_ess", "min_cor_ess", "max_rhat", "divergences") %in% names(tr)))
  fref <- .fit_fixed(m, 75, seed = 9)
  .same_draws(f, fref)
  expect_equal(f$diagnostic_summary()$num_divergent, fref$diagnostic_summary()$num_divergent)
  expect_equal(f$diagnostic_summary()$ebfmi, fref$diagnostic_summary()$ebfmi, tolerance = 1e-8)
})

test_that("auto rounds reproduce one run of the final length exactly (DR-G-HMC)", {
  skip_if_no_jax()
  m <- .auto_model()
  f <- .fit_auto_capped(m, 300, seed = 4, sampler = "drghmc", check_every = 100)$value
  fref <- .fit_fixed(m, 300, seed = 4, sampler = "drghmc")
  .same_draws(f, fref)
  expect_equal(f$diagnostic_summary()$dr_stage, fref$diagnostic_summary()$dr_stage)
})

test_that("long runs are thinned evenly to max_kept and diagnostics still count every draw", {
  skip_if_no_jax()
  m <- .auto_model()
  f <- .fit_auto_capped(m, 100, seed = 5, max_kept = 30)$value
  expect_equal(attr(f, "thin_stride"), 4L)
  expect_equal(attr(f, "draws_per_chain"), 100L)
  expect_equal(f$.iter_sampling, 25L)
  expect_equal(dim(f$.backend_fit$extra_fields$energy), c(2L, 25L))
  expect_equal(unlist(attr(f, "sampler_totals")$n), c(100, 100))
  fref <- .fit_fixed(m, 100, seed = 5)
  .same_draws(f, fref, rows = seq(4, 100, by = 4))
  expect_equal(f$diagnostic_summary()$num_divergent, fref$diagnostic_summary()$num_divergent)
  expect_equal(f$diagnostic_summary()$ebfmi, fref$diagnostic_summary()$ebfmi, tolerance = 1e-8)
  expect_equal(dim(read_npy(f$.loglik_path))[1:2], c(2L, 25L))
  ll_ref <- read_npy(fref$.loglik_path)[, seq(4, 100, by = 4), , drop = FALSE]
  expect_equal(read_npy(f$.loglik_path), ll_ref, tolerance = 1e-6, ignore_attr = TRUE)
})

test_that("the stopping rule's ESS and R-hat equal what summaries report on the fit's own draws", {
  skip_if_no_jax()
  d <- .fixture_crossed(n_sub = 6, n_item = 10)
  m <- broc(brf(item_resp | is_old ~ 1 + (1 + is_old | sub), criterion ~ 1 + (1 | sub), family = evsd()), d)
  for (smp in c("nuts", "drghmc")) {
    f <- .fit_auto_capped(m, if (smp == "nuts") 100 else 300, seed = 12, sampler = smp,
                          max_kept = 30, check_every = if (smp == "nuts") 25 else 100)$value
    da <- posterior::subset_draws(f$.draws_array, variable = .monitored(posterior::variables(f$.draws_array)))
    sm <- posterior::summarise_draws(da, ess_bulk = .ess_bulk, ess_tail = .ess_tail, rhat = posterior::rhat)
    ess <- pmin(sm$ess_bulk, sm$ess_tail)
    cls <- .ess_class(sm$variable)
    last <- utils::tail(attr(f, "ess_trace"), 1)
    for (k in c("pop", "sd", "cor")) {
      x <- ess[cls == k & is.finite(ess)]
      expect_equal(last[[paste0("min_", k, "_ess")]], if (length(x)) min(x) else Inf, tolerance = 1e-8)
    }
    expect_equal(last$max_rhat, max(sm$rhat, na.rm = TRUE), tolerance = 1e-10)
    expect_equal(last$kept, f$.iter_sampling)
  }
})

test_that("auto sampling stops once the targets are met and on the time cap", {
  skip_if_no_jax()
  m <- .auto_model()
  f <- fit_broc(m, backend = "jax", sampler = "nuts", chains = 2, parallel_chains = 2, iter_warmup = 30,
                iter_sampling = "auto", seed = 3, progress_bar = FALSE,
                control = list(ess_target = 1, rhat_target = 10, check_every = 25))
  expect_identical(attr(f, "stop_reason"), "targets")
  expect_equal(f$.iter_sampling, 25L)
  run <- .fit_auto_capped(m, 1e6, seed = 3, max_time = 1e-6)
  expect_identical(attr(run$value, "stop_reason"), "max_time")
  expect_equal(run$value$.iter_sampling, 25L)
  expect_true(any(grepl("stopped at 1e-06 seconds", run$warnings)))
})

test_that("resume continues auto and fixed fits exactly and merges the log-likelihood", {
  skip_if_no_jax()
  m <- .auto_model()
  f1 <- .fit_auto_capped(m, 50, seed = 6)$value
  f2 <- .warnings_of(fit_broc(m, iter_sampling = "auto", resume = f1, progress_bar = FALSE,
                              control = list(ess_target = 1e9, max_draws = 100, check_every = 25)))$value
  expect_equal(f2$.iter_sampling, 100L)
  expect_equal(attr(f2, "ess_trace")$draws, c(25, 50, 75, 100))
  f3 <- fit_broc(m, iter_sampling = 25, resume = f2, progress_bar = FALSE)
  expect_equal(attr(f3, "draws_per_chain"), 125L)
  fref <- .fit_fixed(m, 125, seed = 6)
  .same_draws(f3, fref)
  expect_equal(read_npy(f3$.loglik_path), read_npy(fref$.loglik_path), tolerance = 1e-6, ignore_attr = TRUE)
  g1 <- .fit_fixed(m, 250, seed = 6, sampler = "drghmc")
  g2 <- fit_broc(m, iter_sampling = 150, resume = g1, progress_bar = FALSE)
  .same_draws(g2, .fit_fixed(m, 400, seed = 6, sampler = "drghmc"))
})

test_that("a saved and reloaded fit resumes exactly and keeps its attributes", {
  skip_if_no_jax()
  m <- .auto_model()
  file <- tempfile(fileext = ".rds")
  f1 <- .warnings_of(fit_broc(m, backend = "jax", sampler = "nuts", chains = 2, parallel_chains = 2, iter_warmup = 30,
                              iter_sampling = "auto", seed = 8, file = file, loo = TRUE,
                              progress_bar = FALSE,
                              control = list(ess_target = 1e9, max_draws = 50, check_every = 25)))$value
  expect_s3_class(f1$.loo, "loo")
  unlink(f1$.state_path)
  expect_true(file.exists(sub("[.]rds$", "_state.pkl", file)))
  g <- load_broc_fit(file)
  expect_true(file.exists(g$.state_path))
  expect_identical(attr(g, "stop_reason"), "max_draws")
  expect_equal(attr(g, "ess_trace"), attr(f1, "ess_trace"))
  f2 <- fit_broc(m, iter_sampling = 50, resume = g, progress_bar = FALSE)
  .same_draws(f2, .fit_fixed(m, 100, seed = 8))
})

test_that("resume refuses a fit made on a different model or data", {
  skip_if_no_jax()
  m <- .auto_model()
  f1 <- .fit_fixed(m, 20, seed = 1, warmup = 20)
  d2 <- .fixture_crossed(n_sub = 6, n_item = 10)
  d2$item_resp <- rev(d2$item_resp)
  m2 <- broc(brf(item_resp | is_old ~ 1 + (1 | sub), family = evsd()), d2)
  expect_error(fit_broc(m2, iter_sampling = 20, resume = f1, progress_bar = FALSE), "different model or data")
  expect_identical(.model_fingerprint(m), attr(f1, "model_fingerprint"))
})

test_that("auto monitors the ordered mixture gap through its log-scale sampler column", {
  skip_if_no_jax()
  set.seed(2)
  d <- expand.grid(participant = factor(1:10), words = factor(1:24))
  d$is_old <- as.integer((as.integer(d$words) + as.integer(d$participant)) %% 2 == 0)
  d$conf <- sample(1:6, nrow(d), TRUE)
  m <- broc(brf(conf | is_old ~ 1 + (1 | participant), lambda ~ 1, dprime2 ~ 1, family = mixture()), d)
  f <- fit_broc(m, backend = "jax", sampler = "drghmc", iter_sampling = "auto", chains = 2, parallel_chains = 2,
                iter_warmup = 50, control = list(max_draws = 100, check_every = 100), progress_bar = FALSE)
  expect_true("dprime_order_gap" %in% f$metadata()$variables)
  expect_true(all(f$draws(variables = "dprime_order_gap") > 0))
})
