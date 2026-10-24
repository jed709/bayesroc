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

test_that("mafc batch and plain Stan agree on log density and gradient for every layout", {
  skip_if_no_cmdstan()
  d3 <- .sim_mafc(n_sub = 6, n_trial = 40)
  d3$x <- round(stats::rnorm(nrow(d3)), 2)
  dm <- rbind(transform(.sim_mafc(n_sub = 6, n_trial = 20, m = 2, bias = 0.2, seed = 3), nalt = 2L),
              transform(.sim_mafc(n_sub = 6, n_trial = 20, m = 4, bias = c(0.2, -0.3, 0.1), seed = 4), nalt = 4L))
  agg <- stats::aggregate(list(n = rep(1, nrow(d3))), d3[c("choice", "target", "cond")], sum)
  cfgs <- list(
    re = list(f = brf(choice | target ~ 1 + (1 | sub), sigma ~ 1, bias ~ 1 + (1 | sub), family = mafc()), d = d3),
    m2 = list(f = brf(choice | target ~ cond, bias ~ cond, family = mafc()),
              d = .sim_mafc(n_sub = 6, n_trial = 40, m = 2, bias = 0.3)),
    m4_uv = list(f = brf(choice | target ~ 1 + (1 | sub), sigma ~ 1 + (1 | sub), bias ~ 1, family = mafc()),
                 d = .sim_mafc(n_sub = 6, n_trial = 40, m = 4, sigma = 1.6, bias = c(0.2, -0.3, 0.1))),
    slope = list(f = brf(choice | target ~ x + (1 | sub), bias ~ x + (1 + x | sub), family = mafc()), d = d3),
    factor_slope = list(f = brf(choice | target ~ cond + (1 | sub), bias ~ cond + (0 + cond | sub), family = mafc()), d = d3),
    uncor = list(f = brf(choice | target ~ 1 + (1 | sub), bias ~ 1 + (1 | gr(sub, cor_thresholds = FALSE)),
                         family = mafc()), d = d3),
    cross = list(f = brf(choice | target ~ 1 + (1 | p | sub), sigma ~ 1 + (1 | p | sub), bias ~ 1 + (1 | p | sub),
                         family = mafc()), d = d3),
    mixed_m = list(f = brf(choice | target ~ 1 + (1 | sub), bias ~ 1 + (1 | sub), family = mafc(alternatives = "nalt")),
                   d = dm),
    counts = list(f = brf(choice | target ~ cond, bias ~ cond, family = mafc()), d = agg, counts = "n"))
  for (nm in names(cfgs)) {
    cf <- cfgs[[nm]]
    mb <- suppressMessages(broc(cf$f, data = cf$d, counts = cf$counts))
    mp <- suppressMessages(.broc_plain(cf$f, data = cf$d, counts = cf$counts))
    r <- .stan_parity(mp, mb)
    expect_identical(r$dim[1], r$dim[2], label = nm)
    expect_lt(r$lp, 1e-8, label = paste(nm, "lp"))
    expect_lt(r$grad, 1e-7, label = paste(nm, "gradient"))
  }
})

test_that("mafc through the JAX transport map (f64 and f32) recovers Stan's posterior", {
  skip_if_no_cmdstan(); skip_if_no_jax()
  d <- .sim_mafc(n_sub = 14, n_trial = 60, m = 3, sigma = 1.3, seed = 2)
  m <- suppressMessages(broc(brf(choice | target ~ 1 + (1 | sub), sigma ~ 1 + (1 | sub), bias ~ 1 + (1 | sub),
                                 family = mafc()), data = d))
  sm <- function(f) {
    da <- f$draws(format = "draws_array")
    v <- grep("^(beta|sigma_)", posterior::variables(da), value = TRUE)
    s <- posterior::summarise_draws(posterior::subset_draws(da, variable = v), "mean", "mcse_mean")
    s[order(s$variable), ]
  }
  rs <- sm(fit_broc(m, backend = "stan", chains = 4, parallel_chains = 4, iter_warmup = 1000, iter_sampling = 3000,
                    seed = 1, refresh = 0))
  for (prec in c("f64", "f32")) {
    fj <- fit_broc(m, backend = "jax", chains = 4, iter_warmup = 1000, iter_sampling = 3000, seed = 3,
                   progress_bar = FALSE, control = list(transport = TRUE, precision = prec))
    js <- sm(fj)
    expect_identical(js$variable, rs$variable)
    z <- abs(js$mean - rs$mean) / sqrt(js$mcse_mean^2 + rs$mcse_mean^2)
    expect_lt(max(z), 4, label = paste(prec, "max |z|"))
  }
})
