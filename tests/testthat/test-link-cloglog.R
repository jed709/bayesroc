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
                 chains = 4, refresh = 100)
  dp <- posterior::as_draws_matrix(fj$draws(variables = "beta_dprime[1]"))[, 1]
  expect_lt(abs(mean(dp) - 1.0), 3 * sd(dp) + 0.02)
})

test_that("evsd + cloglog: Stan and JAX posterior means agree", {
  skip_if_no_cmdstan(); skip_if_no_jax()
  d <- .fixture_gumbel_min(n_per = 400)
  m <- broc(brf(item_resp | is_old ~ 1, family = evsd(link = link_cloglog())), d,
            batch_likelihood = TRUE)
  fs <- fit_broc(m, backend = "stan", refresh = 100,
                 parallel_chains = 4, iter_warmup = 400, iter_sampling = 400)
  fj <- fit_broc(m, backend = "jax",
                 iter_warmup = 400, iter_sampling = 400, chains = 4, refresh = 100)
  for (v in c("beta_dprime[1]", "beta_thresh_mid[1]")) {
    ms <- mean(posterior::as_draws_matrix(fs$draws(variables = v))[, 1])
    mj <- mean(posterior::as_draws_matrix(fj$draws(variables = v))[, 1])
    expect_lt(abs(ms - mj), 0.05, label = sprintf("|stan - jax| for %s", v))
  }
})
