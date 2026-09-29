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

test_that("cloglog Stan link_cdf is the Gumbel-min CDF", {
  code <- generate_link_cdf_wrapper(link_cloglog())
  expect_match(code, "-expm1\\(-exp\\(xc\\)\\)")
})

test_that("cloglog has no batch C++ kernel and falls back to plain Stan", {
  d <- .fixture_gumbel_min(n_per = 50)
  m <- broc(brf(item_resp | is_old ~ 1, family = evsd(link = link_cloglog())), d,
            batch_likelihood = TRUE)
  expect_null(generate_batch_cpp(m$model_data, m$family))
  expect_match(m$stan_code, "expm1\\(-exp")
})

test_that("evsd + cloglog recovers a Gumbel-min location shift (JAX)", {
  skip_if_no_jax()
  d <- .fixture_gumbel_min()
  m <- broc(brf(item_resp | is_old ~ 1, family = evsd(link = link_cloglog())), d)
  fj <- fit_broc(m, backend = "jax", iter_warmup = 500, iter_sampling = 500,
                 chains = 4, refresh = 50)
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

test_that("R-side cloglog CDF matches the Gumbel-min CDF and the JAX clamp", {
  x <- c(-800, -30, -1, 0, 1, 3.5, 10, 800)
  expect_equal(link_cdf_r(x, "cloglog"), 1 - exp(-exp(pmin(x, 700))), tolerance = 1e-15)
  expect_true(all(diff(link_cdf_r(x, "cloglog")) >= 0))
})

test_that("evsd + cloglog: Stan and JAX posterior means agree", {
  skip_if_no_cmdstan(); skip_if_no_jax()
  d <- .fixture_gumbel_min(n_per = 400)
  m <- broc(brf(item_resp | is_old ~ 1, family = evsd(link = link_cloglog())), d,
            batch_likelihood = TRUE)
  fs <- fit_broc(m, backend = "stan", refresh = 50,
                 parallel_chains = 4, iter_warmup = 400, iter_sampling = 400)
  fj <- fit_broc(m, backend = "jax",
                 iter_warmup = 400, iter_sampling = 400, chains = 4, refresh = 50)
  for (v in c("beta_dprime[1]", "beta_thresh_mid[1]")) {
    ms <- mean(posterior::as_draws_matrix(fs$draws(variables = v))[, 1])
    mj <- mean(posterior::as_draws_matrix(fj$draws(variables = v))[, 1])
    expect_lt(abs(ms - mj), 0.05, label = sprintf("|stan - jax| for %s", v))
  }
})
