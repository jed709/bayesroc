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
  code <- .plain_stan_dev_env()$generate_link_cdf_wrapper(link_cloglog())
  expect_match(code, "-expm1\\(-exp\\(xc\\)\\)")
})

test_that("non-probit links: batch C++ matches plain Stan in value and gradient", {
  skip_if_no_cmdstan()
  d <- .fixture_gumbel_min(n_per = 150)
  ds <- .fixture_source_mixture()
  cases <- list(
    dpsd_cloglog = list(function(l) brf(item_resp | is_old ~ 1, lambda ~ 1, family = dpsd(link = l)),
                        d, link_cloglog()),
    mixture_cloglog = list(function(l) brf(item_resp | is_old ~ 1, dprime2 ~ 1, lambda ~ 1,
                                           family = mixture(link = l)), d, link_cloglog()),
    source_mixture_cloglog = list(function(l) brf(item_resp | source ~ 1, lambda ~ 1,
                                                  family = source_mixture(link = l)), ds, link_cloglog()),
    mixture_logit = list(function(l) brf(item_resp | is_old ~ 1, dprime2 ~ 1, lambda ~ 1,
                                         family = mixture(link = l)), d, link_logit())
  )
  for (nm in names(cases)) {
    cs <- cases[[nm]]
    mb <- suppressMessages(broc(cs[[1]](cs[[3]]), cs[[2]]))
    expect_false(is.null(mb$batch_cpp), label = nm)
    mp <- suppressMessages(.broc_plain(cs[[1]](cs[[3]]), cs[[2]]))
    r <- .stan_parity(mp, mb)
    expect_lt(r$lp, 1e-8, label = paste(nm, "log density"))
    expect_lt(r$grad, 1e-8, label = paste(nm, "gradient"))
  }
})

test_that("evsd + cloglog recovers a Gumbel-min location shift (JAX)", {
  skip_if_not_slow()
  skip_if_no_jax()
  d <- .fixture_gumbel_min()
  m <- broc(brf(item_resp | is_old ~ 1, family = evsd(link = link_cloglog())), d)
  fj <- fit_broc(m, backend = "jax", sampler = "nuts", iter_warmup = 500, iter_sampling = 500,
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

test_that("R-side cloglog CDF matches the Gumbel-min CDF and the JAX clamp", {
  x <- c(-800, -30, -1, 0, 1, 3.5, 10, 800)
  expect_equal(link_cdf_r(x, "cloglog"), 1 - exp(-exp(pmin(x, 700))), tolerance = 1e-15)
  expect_true(all(diff(link_cdf_r(x, "cloglog")) >= 0))
})

test_that("evsd + cloglog: Stan and JAX posterior means agree", {
  skip_if_not_slow()
  skip_if_no_cmdstan(); skip_if_no_jax()
  d <- .fixture_gumbel_min(n_per = 400)
  m <- broc(brf(item_resp | is_old ~ 1, family = evsd(link = link_cloglog())), d)
  fs <- fit_broc(m, backend = "stan", refresh = 50,
                 parallel_chains = 4, iter_warmup = 400, iter_sampling = 400)
  fj <- fit_broc(m, backend = "jax", sampler = "nuts",
                 iter_warmup = 400, iter_sampling = 400, chains = 4, refresh = 50)
  for (v in c("beta_dprime[1]", "beta_thresh_mid[1]")) {
    ms <- mean(posterior::as_draws_matrix(fs$draws(variables = v))[, 1])
    mj <- mean(posterior::as_draws_matrix(fj$draws(variables = v))[, 1])
    expect_lt(abs(ms - mj), 0.05, label = sprintf("|stan - jax| for %s", v))
  }
})

test_that("evsd + cloglog: R reproduces the Stan log-likelihood", {
  skip_if_no_cmdstan()
  d <- .fixture_gumbel_min(n_per = 150)
  m <- broc(brf(item_resp | is_old ~ 1, family = evsd(link = link_cloglog())), d)
  fs <- suppressMessages(fit_broc(m, backend = "stan", chains = 1, iter_warmup = 20, iter_sampling = 5,
                                  seed = 1, refresh = 50, show_messages = FALSE))
  ll_stan <- as.matrix(posterior::as_draws_matrix(fs$draws(variables = "log_lik")))
  ll_r <- .obs_loglik_r(m, build_data_source_from_model(m), fs$draws(format = "matrix"), re_formula = NULL)
  expect_lt(max(abs(unname(ll_stan) - ll_r)), 1e-5)
})
