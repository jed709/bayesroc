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
  skip_if_not_slow()
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
