.mafc_cell_integrate <- function(k, s, d, sg, b) {
  m <- length(b)
  f <- if (k == s) {
    function(z) stats::dnorm(z) * Reduce(`*`, lapply(setdiff(seq_len(m), s), function(j) stats::pnorm(sg * z + d + b[s] - b[j])), 1)
  } else {
    function(z) stats::dnorm(z) * stats::pnorm((z - d + b[k] - b[s]) / sg) *
      Reduce(`*`, lapply(setdiff(seq_len(m), c(k, s)), function(j) stats::pnorm(z + b[k] - b[j])), 1)
  }
  stats::integrate(f, -Inf, Inf, rel.tol = 1e-12, subdivisions = 1000L)$value
}

test_that("mafc() validates its data and formulas", {
  d <- .sim_mafc()
  expect_error(mafc(alternatives = 3))
  expect_error(broc(brf(choice | target ~ 1, criterion ~ 1, family = mafc()), data = d), "not valid for family 'mafc'")
  bad <- d
  bad$choice[1] <- 0
  expect_error(broc(brf(choice | target ~ 1, family = mafc()), data = bad), "positions 1, 2")
  mixed <- d
  mixed$nalt <- 3L
  mixed$nalt[1] <- 2L
  mixed$choice[1] <- 3L
  expect_error(broc(brf(choice | target ~ 1, family = mafc(alternatives = "nalt")), data = mixed), "exceeds its 2")
  m <- broc(brf(choice | target ~ 1 + (1 | sub), bias ~ 1 + (1 | sub), family = mafc()), data = d)
  expect_identical(m$model_data$K, 3L)
  expect_true(isTRUE(m$model_data$has_bias))
  expect_equal(m$model_data$bias$random$sub$dim, 2L)
  expect_true("bias" %in% get_broc_prior(m)$dpar)
})

test_that("m-AFC cells match direct integration on the native and R paths, rows sum to 1", {
  set.seed(3)
  worst_r <- worst_n <- 0
  for (i in 1:30) {
    M <- sample(c(2L, 3L, 4L, 6L), 1)
    d <- stats::runif(1, -0.5, 3.5)
    sg <- exp(stats::rnorm(1, 0, 0.6))
    b <- c(stats::rnorm(M - 1, 0, 0.7), 0)
    s <- sample.int(M, 1)
    ref <- vapply(seq_len(M), function(k) .mafc_cell_integrate(k, s, d, sg, b), numeric(1))
    r_path <- .mafc_probs_r(d, sg, matrix(b, 1), M, s, M)
    worst_r <- max(worst_r, max(abs(log(r_path) - log(ref))))
    expect_equal(sum(r_path), 1, tolerance = 1e-10)
    if (.bvn_native_ready() && !is.null(.bvn_native$sym_mafc)) {
      nat <- probs_mafc(list(dprime = matrix(d), sigma = matrix(sg)), array(b, c(1, 1, M)),
                        list(m_alt = M, target_pos = s), 1, 1, M)[1, 1, ]
      worst_n <- max(worst_n, max(abs(log(nat) - log(ref))))
    }
  }
  expect_lt(worst_r, 1e-9)
  skip_if_no_jax()
  expect_true(.bvn_native_ready())
  expect_false(is.null(.bvn_native$sym_mafc))
  expect_lt(worst_n, 1e-9)
})

test_that("equal-variance m-AFC cells (the unit-slope path) match direct integration, rows sum to 1", {
  skip_if_no_jax()
  expect_true(.bvn_native_ready())
  set.seed(4)
  worst <- 0
  for (i in 1:40) {
    M <- sample(c(3L, 4L, 6L, 9L), 1)
    d <- stats::runif(1, -0.5, 3.5)
    b <- c(stats::rnorm(M - 1, 0, 0.7), 0)
    s <- sample.int(M, 1)
    ref <- vapply(seq_len(M), function(k) .mafc_cell_integrate(k, s, d, 1, b), numeric(1))
    nat <- probs_mafc(list(dprime = matrix(d), sigma = matrix(1)), array(b, c(1, 1, M)),
                      list(m_alt = M, target_pos = s), 1, 1, M)[1, 1, ]
    expect_equal(sum(nat), 1, tolerance = 1e-8)
    worst <- max(worst, max(abs(log(nat) - log(ref))))
  }
  expect_lt(worst, 1e-8)
})

test_that("the m-AFC categorical likelihood gives the multinomial estimates for DeCarlo's (2012) taste data", {
  tabs <- list(WW = rbind(c(54, 5, 1), c(0, 60, 0), c(5, 3, 52)), SS = rbind(c(40, 12, 8), c(6, 49, 4), c(6, 5, 49)))
  want <- list(WW = c(2.628, 0.400, 0.854), SS = c(1.539, -0.172, 0.124))
  for (nm in names(tabs)) {
    y <- tabs[[nm]]
    nll <- function(p) {
      P <- t(vapply(1:3, function(s) .mafc_probs_r(p[1], 1, matrix(c(p[2], p[3], 0), 1), 3L, s, 3L)[1, ], numeric(3)))
      -sum(y * log(P))
    }
    o <- stats::optim(c(1.5, 0, 0), nll, method = "BFGS", control = list(reltol = 1e-12))
    expect_equal(o$par, want[[nm]], tolerance = 2e-3, label = nm)
  }
})

test_that("mafc through the JAX transport map (f64 and f32): R reproduces the JAX log-likelihood", {
  skip_if_no_jax()
  d <- .sim_mafc(n_sub = 14, n_trial = 60, m = 3, seed = 2)
  m <- suppressMessages(broc(brf(choice | target ~ 1 + (1 | sub), bias ~ 1 + (1 | sub),
                                 family = mafc()), data = d))
  for (prec in c("f64", "f32")) {
    fj <- suppressMessages(fit_broc(m, sampler = "hmc", chains = 1, iter_warmup = 50,
                                    iter_sampling = 10, seed = 3, progress_bar = FALSE,
                                    control = list(transport = TRUE, precision = prec)))
    dm <- fj$draws(format = "matrix")
    expect_true(all(is.finite(dm)), label = prec)
    pr <- predict(fj, type = "response", summary = FALSE)
    y <- m$model_data$stan_data$y
    ll_r <- log(sapply(seq_along(y), function(n) pr[, n, y[n]]))
    tol <- if (prec == "f64") 1e-5 else 1e-4
    expect_lt(max(abs(ll_r - .read_loglik_SN_cols(.loglik_path(fj), 1L, m$model_data$N))), tol, label = prec)
  }
})

test_that("mafc() is equal variance: sigma is not a parameter of the family", {
  expect_error(broc(brf(choice | target ~ 1, sigma ~ 1, family = mafc()), data = .sim_mafc()),
               "Parameter 'sigma' not valid for family 'mafc'")
})
