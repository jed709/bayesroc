# Helpers shared with tests/testthat/
# test-link-cloglog.R (copied).

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

test_that("cloglog Stan link_cdf is the Gumbel-min CDF", {
  code <- .plain_env$generate_link_cdf_wrapper(link_cloglog())
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
    expect_false(is.null(.with_stan_code(mb)$batch_cpp), label = nm)
    mp <- suppressMessages(.broc_plain(cs[[1]](cs[[3]]), cs[[2]]))
    r <- .stan_parity(mp, mb)
    expect_lt(r$lp, 1e-8, label = paste(nm, "log density"))
    expect_lt(r$grad, 1e-8, label = paste(nm, "gradient"))
  }
})

test_that("evsd + cloglog: Stan and JAX posterior means agree", {
  skip_if_not_slow()
  skip_if_no_cmdstan(); skip_if_no_jax()
  d <- .fixture_gumbel_min(n_per = 400)
  m <- broc(brf(item_resp | is_old ~ 1, family = evsd(link = link_cloglog())), d)
  fs <- stan_fit_broc(m, refresh = 50,
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
  fs <- suppressMessages(stan_fit_broc(m, chains = 1, iter_warmup = 20, iter_sampling = 5,
                                  seed = 1, refresh = 50, show_messages = FALSE))
  ll_stan <- as.matrix(posterior::as_draws_matrix(fs$draws(variables = "log_lik")))
  ll_r <- .obs_loglik_r(m, build_data_source_from_model(m), fs$draws(format = "matrix"), re_formula = NULL)
  expect_lt(max(abs(unname(ll_stan) - ll_r)), 1e-5)
})
