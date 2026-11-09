.fixture_gumbel_min <- function(seed = 11, dprime = 1.0, n_per = 1500) {
  set.seed(seed)
  thresh <- c(-1.6, -0.9, -0.3, 0.3, 0.9)
  rgmin <- function(n, loc) loc + log(-log1p(-runif(n)))
  data.frame(
    item_resp = c(as.integer(cut(rgmin(n_per, dprime), c(-Inf, thresh, Inf))),
                  as.integer(cut(rgmin(n_per, 0), c(-Inf, thresh, Inf)))),
    is_old = c(rep(1L, n_per), rep(0L, n_per))
  )
}

test_that("link_cloglog() builds a broc_link and is accepted by univariate families", {
  l <- link_cloglog()
  expect_s3_class(l, "broc_link")
  expect_identical(l$name, "cloglog")
  for (fam in list(evsd, uvsd, dpsd, mixture, cumulative)) {
    expect_identical(fam(link = link_cloglog())$link$name, "cloglog")
  }
})

test_that("evsd + cloglog recovers a Gumbel-min location shift (JAX)", {
  skip_if_not_slow()
  skip_if_no_jax()
  d <- .fixture_gumbel_min()
  m <- broc(brf(item_resp | is_old ~ 1, family = evsd(link = link_cloglog())), d)
  fj <- fit_broc(m, backend = "jax", sampler = "nuts", iter_warmup = 500, iter_sampling = 500,
                 chains = 4)
  dp <- posterior::as_draws_matrix(fj$draws(variables = "beta_dprime[1]"))[, 1]
  expect_lt(abs(mean(dp) - 1.0), 3 * sd(dp) + 0.02)
  ep <- posterior_epred(fj, ndraws = 50)
  expect_true(all(is.finite(ep)))
  pr <- predict(fj, ndraws = 50)
  expect_equal(nrow(pr), nrow(d) * 6)
  expect_equal(as.numeric(tapply(pr$mean, pr$trial, sum)), rep(1, nrow(d)), tolerance = 1e-6)
  for (old in 0:1) {
    rows <- which(d$is_old == old)
    pred <- tapply(pr$mean[pr$trial %in% rows], pr$category[pr$trial %in% rows], mean)
    obs <- as.numeric(table(factor(d$item_resp[rows], levels = 1:6))) / length(rows)
    expect_lt(max(abs(pred - obs)), 0.03)
  }
})

test_that("evsd + cloglog (JAX): predictions are complete distributions and R reproduces the JAX log-likelihood", {
  skip_if_no_jax()
  d <- .fixture_gumbel_min(n_per = 150)
  m <- broc(brf(item_resp | is_old ~ 1, family = evsd(link = link_cloglog())), d)
  fj <- suppressMessages(fit_broc(m, backend = "jax", sampler = "nuts", iter_warmup = 30, iter_sampling = 20,
                                  chains = 1, seed = 1, progress_bar = FALSE))
  ep <- posterior_epred(fj, ndraws = 20)
  expect_true(all(is.finite(ep)))
  pr <- predict(fj, ndraws = 20)
  expect_equal(nrow(pr), nrow(d) * 6)
  expect_equal(as.numeric(tapply(pr$mean, pr$trial, sum)), rep(1, nrow(d)), tolerance = 1e-6)
  ll_r <- .obs_loglik_r(m, build_data_source_from_model(m), fj$draws(format = "matrix"), re_formula = NULL)
  expect_lt(max(abs(ll_r - .read_loglik_SN_cols(.loglik_path(fj), 1L, m$model_data$N))), 1e-5)
})
