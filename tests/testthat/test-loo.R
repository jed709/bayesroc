
test_that("loo() returns a valid loo object and loo_compare() ranks models", {
  skip_if_no_cmdstan()
  d <- .fixture_univariate(sigma = 1.3)
  m_ev <- broc(brf(item_resp | is_old ~ 1, family = evsd()), d)
  m_uv <- broc(brf(item_resp | is_old ~ 1, sigma ~ 1, family = uvsd()), d)
  f_ev <- fit_broc(m_ev, backend = "stan", refresh = 0, seed = 1,
                   parallel_chains = 4, iter_warmup = 300, iter_sampling = 300)
  f_uv <- fit_broc(m_uv, backend = "stan", refresh = 0, seed = 1,
                   parallel_chains = 4, iter_warmup = 300, iter_sampling = 300)

  l_ev <- loo(f_ev, cores = 1)
  l_uv <- loo(f_uv, cores = 1)
  expect_s3_class(l_ev, "loo")
  expect_s3_class(l_uv, "loo")

  cmp <- loo_compare(l_ev, l_uv)
  expect_true(is.matrix(cmp) || is.data.frame(cmp))
  expect_true("elpd_diff" %in% colnames(cmp))
  expect_equal(nrow(cmp), 2L)
})

test_that("loo() on an aggregated fit reproduces the trial-level result", {
  skip_if_no_cmdstan()
  d <- .fixture_univariate(sigma = 1.3, n_per = 200)
  agg <- aggregate(list(n = rep(1L, nrow(d))),
                   by = list(item_resp = d$item_resp, is_old = d$is_old),
                   FUN = sum)
  expect_lt(nrow(agg), nrow(d))

  m <- broc(brf(item_resp | is_old ~ 1, family = evsd()), agg, counts = "n")
  f <- fit_broc(m, backend = "stan", refresh = 0, seed = 1,
                parallel_chains = 4, iter_warmup = 300, iter_sampling = 300)

  l <- loo(f, cores = 1)
  expect_equal(dim(l)[2], nrow(d))

  ll <- as.matrix(f$draws(variables = "log_lik", format = "matrix"))
  cnt <- m$stan_data$counts
  ll_exp <- ll[, rep.int(seq_len(ncol(ll)), cnt), drop = FALSE]
  cid <- rep(seq_len(f$num_chains()), each = nrow(ll) / f$num_chains())
  ref <- loo::loo(ll_exp, r_eff = loo::relative_eff(exp(ll_exp), chain_id = cid),
                  cores = 1)

  expect_equal(l$estimates, ref$estimates, tolerance = 1e-8)
  expect_equal(l$pointwise[, "elpd_loo"], ref$pointwise[, "elpd_loo"],
               tolerance = 1e-8)
  expect_equal(unname(l$diagnostics$pareto_k), unname(ref$diagnostics$pareto_k),
               tolerance = 1e-8)

  l4 <- loo(f, cores = 4)
  expect_equal(l4$estimates, l$estimates, tolerance = 1e-8)

  l_cell <- suppressWarnings(loo(f, cores = 1, type = "cell"))
  expect_equal(dim(l_cell)[2], nrow(agg))
  expect_false(isTRUE(all.equal(l_cell$estimates["elpd_loo", "Estimate"],
                                l$estimates["elpd_loo", "Estimate"])))

  expect_equal(dim(f$loo(cores = 1))[2], nrow(d))
})
