test_that("lognormal meta-noise needs the probit link and an even number of categories, at least 4", {
  d <- .sim_lnoise()
  expect_error(evsd(meta_noise = "beta"))
  expect_error(broc(brf(conf | old ~ 1, criterion ~ 1, sigma_meta ~ 1, family = evsd(link_logit(), "lognormal")),
                    data = d), "lognormal meta-noise model.*probit")
  odd <- d
  odd$conf <- pmin(odd$conf, 5L)
  expect_error(broc(brf(conf | old ~ 1, criterion ~ 1, sigma_meta ~ 1, family = evsd(meta_noise = "lognormal")),
                    data = odd), "even number of response categories")
  m <- broc(brf(conf | old ~ 1, criterion ~ 1, sigma_meta ~ 1 + (1 | sub), family = evsd(meta_noise = "lognormal")),
            data = d)
  expect_identical(m$model_data$meta_noise, "lognormal")
  expect_identical(broc(brf(conf | old ~ 1, criterion ~ 1, sigma_meta ~ 1, family = evsd()), data = d)$model_data$meta_noise,
                   "gaussian")
  expect_null(broc(brf(conf | old ~ 1, criterion ~ 1, family = evsd(meta_noise = "lognormal")), data = d)$model_data$meta_noise)
  d$enc <- factor(ifelse(as.integer(d$item) %% 2 == 0, "aloud", "silent"))
  expect_error(broc(brf(conf | old ~ enc, criterion ~ 1, sigma_meta ~ enc, family = evsd(meta_noise = "lognormal")),
                    data = d, encoding_vars = "enc"), "lures have no encoding condition")
  expect_s3_class(broc(brf(conf | old ~ enc, criterion ~ 1, sigma_meta ~ 1, family = evsd(meta_noise = "lognormal")),
                       data = d, encoding_vars = "enc"), "broc_model")
})

test_that("lognormal meta-noise cells match the 28-digit reference on the native and R paths", {
  ref <- utils::read.csv(testthat::test_path("fixtures", "lnoise_reference.csv"))
  args <- list(ref$a, log(ref$Dn), log(ref$Df), ref$s)
  r_path <- do.call(.lnoise_cell_r, args)
  expect_lt(max(abs(log(r_path) - log(ref$P))), 1e-9)
  skip_if_no_jax()
  expect_true(.bvn_native_ready())
  expect_false(is.null(.bvn_native$sym_lnoise))
  expect_lt(max(abs(log(do.call(.lnoise_cell_v, args)) - log(ref$P))), 3e-8)
})

test_that("lognormal meta-noise cells sum to 1, match the simulator, and become evsd as sigma_meta -> 0", {
  set.seed(5)
  n <- 4e5
  for (cfg in list(list(K = 6L, d = 1.4, sm = 0.6), list(K = 4L, d = 0.7, sm = 1.8), list(K = 8L, d = 2.2, sm = 0.15))) {
    K <- cfg$K
    tau <- 0.6 + c(-rev(cumsum(rep(0.5, K / 2 - 1))), 0, cumsum(rep(0.5, K / 2 - 1)))
    c0 <- tau[K / 2]
    for (s in 0:1) {
      p <- .lnoise_cells(matrix(cfg$d), matrix(cfg$sm), array(tau, c(1, 1, K - 1)), matrix(s), K)[1, 1, ]
      expect_equal(sum(p), 1, tolerance = 1e-9)
      x <- stats::rnorm(n, s * cfg$d)
      e <- exp(cfg$sm * stats::rnorm(n))
      old <- x > c0
      up <- tau[(K / 2 + 1):(K - 1)] - c0
      dn <- c0 - tau[(K / 2 - 1):1]
      k <- ifelse(old, K / 2 + 1 + rowSums((x - c0) > outer(e, up)), K / 2 - rowSums((c0 - x) > outer(e, dn)))
      f <- tabulate(k, K) / n
      expect_lt(max(abs(f - p) / sqrt(p * (1 - p) / n)), 5)
    }
  }
  tau <- c(-0.5, 0.2, 0.8, 1.3, 2)
  for (s in 0:1) {
    p <- .lnoise_cells(matrix(1.4), matrix(1e-5), array(tau, c(1, 1, 5)), matrix(s), 6)[1, 1, ]
    expect_equal(p, diff(c(0, stats::pnorm(tau - 1.4 * s), 1)), tolerance = 1e-4)
  }
})

test_that("lognormal meta-noise native kernel agrees with the reference, gradients included", {
  skip_if_no_jax()
  py <- jax_python()
  out <- system2(py, c(test_path("python", "lnoise_check.py"), dirname(find_backend_script()), .bvn_ffi_lib(),
                       test_path("fixtures", "lnoise_reference.csv")), stdout = TRUE, stderr = FALSE)
  res <- jsonlite::fromJSON(sub("^RESULT ", "", grep("^RESULT ", out, value = TRUE)))
  expect_true(res$ffi_loaded)
  expect_true(res$finite)
  expect_lt(res$native_vs_ref, 3e-6)
})

test_that("evsd with lognormal meta-noise through the JAX transport map (f64 and f32): R reproduces the JAX log-likelihood", {
  skip_if_no_jax()
  d <- .sim_lnoise(n_sub = 16, n_item = 32, seed = 2)
  m <- suppressMessages(broc(brf(conf | old ~ 1 + (1 | sub), criterion ~ 1 + (1 | sub),
                                 sigma_meta ~ 1 + (1 | sub), family = evsd(meta_noise = "lognormal")), data = d))
  src <- build_data_source_from_model(m)
  for (prec in c("f64", "f32")) {
    fj <- suppressMessages(fit_broc(m, sampler = "hmc", chains = 1, iter_warmup = 50, iter_sampling = 10, seed = 3,
                                    control = list(transport = TRUE, precision = prec)))
    dm <- fj$draws(format = "matrix")
    expect_true(all(is.finite(dm)), label = prec)
    ll_r <- .obs_loglik_r(m, src, dm, re_formula = NULL)
    tol <- if (prec == "f64") 1e-5 else 1e-4
    expect_lt(max(abs(ll_r - .read_loglik_SN_cols(.loglik_path(fj), 1L, m$model_data$N))), tol, label = prec)
  }
})
