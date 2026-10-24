test_that("sigma_meta needs the probit link and an even number of categories, at least 4", {
  d <- .sim_gnoise()
  expect_error(broc(brf(conf | old ~ 1, criterion ~ 1, sigma_meta ~ 1, family = evsd(link_logit())), data = d),
               "probit")
  odd <- d
  odd$conf <- pmin(odd$conf, 5L)
  expect_error(broc(brf(conf | old ~ 1, criterion ~ 1, sigma_meta ~ 1, family = evsd()), data = odd),
               "even number of response categories")
  m <- broc(brf(conf | old ~ 1, criterion ~ 1, sigma_meta ~ 1 + (1 | sub), family = evsd()), data = d)
  expect_true(isTRUE(m$model_data$has_sigma_meta))
  expect_true("sigma_meta" %in% get_broc_prior(m)$dpar)
  expect_false(isTRUE(broc(brf(conf | old ~ 1, criterion ~ 1, family = evsd()), data = odd)$model_data$has_sigma_meta))
})

test_that("Gaussian meta-noise cells match direct integration over the decision evidence, and become evsd as sigma_meta -> 0", {
  set.seed(4)
  worst <- 0
  for (i in 1:40) {
    K <- sample(c(4L, 6L, 8L), 1)
    tau <- sort(stats::rnorm(K - 1, 0.7, 1.2))
    d <- stats::runif(1, 0, 3); sm <- exp(stats::rnorm(1, 0, 0.8))
    for (s in 0:1) {
      p <- .gnoise_cells(matrix(d), matrix(sm), array(tau, c(1, 1, K - 1)), matrix(s), K)[1, 1, ]
      q <- .gnoise_quad(d, sm, tau, s, K)
      worst <- max(worst, max(abs(p - q) / pmax(q, 1e-12)))
      expect_equal(sum(p), 1, tolerance = 1e-10)
    }
  }
  expect_lt(worst, 1e-8)
  tau <- c(-0.5, 0.2, 0.8, 1.3, 2)
  for (s in 0:1) {
    p <- .gnoise_cells(matrix(1.4), matrix(1e-4), array(tau, c(1, 1, 5)), matrix(s), 6)[1, 1, ]
    expect_equal(p, diff(c(0, stats::pnorm(tau - 1.4 * s), 1)), tolerance = 1e-3)
  }
})

test_that("evsd with sigma_meta: batch and plain Stan agree on log density and gradient", {
  skip_if_no_cmdstan()
  cfgs <- list(
    re = list(f = brf(conf | old ~ 1 + (1 | sub), criterion ~ 1 + (1 | sub), sigma_meta ~ 1 + (1 | sub),
                      family = evsd()), d = .sim_gnoise()),
    k4_cond = list(f = brf(conf | old ~ cond, criterion ~ cond, sigma_meta ~ cond, family = evsd()),
                   d = .sim_gnoise(K = 4, sm = 0.4)))
  for (nm in names(cfgs)) {
    cf <- cfgs[[nm]]
    mb <- suppressMessages(broc(cf$f, data = cf$d))
    mp <- suppressMessages(.broc_plain(cf$f, data = cf$d))
    r <- .stan_parity(mp, mb)
    expect_identical(r$dim[1], r$dim[2], label = nm)
    expect_lt(r$lp, 1e-8, label = paste(nm, "lp"))
    expect_lt(r$grad, 1e-7, label = paste(nm, "gradient"))
  }
})

test_that("evsd with sigma_meta through the JAX transport map (f64 and f32) recovers Stan's posterior", {
  skip_if_no_cmdstan(); skip_if_no_jax()
  d <- .sim_gnoise(n_sub = 16, n_item = 32, seed = 2)
  m <- suppressMessages(broc(brf(conf | old ~ 1 + (1 | sub), criterion ~ 1 + (1 | sub),
                                 sigma_meta ~ 1 + (1 | sub), family = evsd()), data = d))
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
