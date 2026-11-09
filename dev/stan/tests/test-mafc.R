# Helpers shared with tests/testthat/
# test-mafc.R (copied).

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

test_that("mafc batch and plain Stan agree on log density and gradient for every layout", {
  skip_if_no_cmdstan()
  d3 <- .sim_mafc(n_sub = 6, n_trial = 40)
  d3$x <- round(stats::rnorm(nrow(d3)), 2)
  dm <- rbind(transform(.sim_mafc(n_sub = 6, n_trial = 20, m = 2, bias = 0.2, seed = 3), nalt = 2L),
              transform(.sim_mafc(n_sub = 6, n_trial = 20, m = 4, bias = c(0.2, -0.3, 0.1), seed = 4), nalt = 4L))
  agg <- stats::aggregate(list(n = rep(1, nrow(d3))), d3[c("choice", "target", "cond")], sum)
  cfgs <- list(
    re = list(f = brf(choice | target ~ 1 + (1 | sub), bias ~ 1 + (1 | sub), family = mafc()), d = d3),
    m2 = list(f = brf(choice | target ~ cond, bias ~ cond, family = mafc()),
              d = .sim_mafc(n_sub = 6, n_trial = 40, m = 2, bias = 0.3)),
    m4 = list(f = brf(choice | target ~ 1 + (1 | sub), bias ~ 1, family = mafc()),
              d = .sim_mafc(n_sub = 6, n_trial = 40, m = 4, bias = c(0.2, -0.3, 0.1))),
    slope = list(f = brf(choice | target ~ x + (1 | sub), bias ~ x + (1 + x | sub), family = mafc()), d = d3),
    factor_slope = list(f = brf(choice | target ~ cond + (1 | sub), bias ~ cond + (0 + cond | sub), family = mafc()), d = d3),
    uncor = list(f = brf(choice | target ~ 1 + (1 | sub), bias ~ 1 + (1 | gr(sub, cor_thresholds = FALSE)),
                         family = mafc()), d = d3),
    cross = list(f = brf(choice | target ~ 1 + (1 | p | sub), bias ~ 1 + (1 | p | sub),
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
    expect_lt(r$grad, 1e-6, label = paste(nm, "gradient"))
  }
})

test_that("mafc through the JAX transport map (f64 and f32) recovers Stan's posterior", {
  skip_if_not_slow()
  skip_if_no_cmdstan(); skip_if_no_jax()
  d <- .sim_mafc(n_sub = 14, n_trial = 60, m = 3, seed = 2)
  m <- suppressMessages(broc(brf(choice | target ~ 1 + (1 | sub), bias ~ 1 + (1 | sub),
                                 family = mafc()), data = d))
  sm <- function(f) {
    da <- f$draws(format = "draws_array")
    v <- grep("^(beta|sigma_)", posterior::variables(da), value = TRUE)
    s <- posterior::summarise_draws(posterior::subset_draws(da, variable = v), "mean", "mcse_mean")
    s[order(s$variable), ]
  }
  rs <- sm(stan_fit_broc(m, chains = 4, parallel_chains = 4, iter_warmup = 1000, iter_sampling = 3000,
                    seed = 1, refresh = 0))
  for (prec in c("f64", "f32")) {
    fj <- fit_broc(m, backend = "jax", sampler = "nuts", chains = 4, iter_warmup = 1000, iter_sampling = 3000, seed = 3,
                   progress_bar = FALSE, control = list(transport = TRUE, precision = prec))
    js <- sm(fj)
    expect_identical(js$variable, rs$variable)
    z <- abs(js$mean - rs$mean) / sqrt(js$mcse_mean^2 + rs$mcse_mean^2)
    expect_lt(max(z), 4, label = paste(prec, "max |z|"))
  }
})
