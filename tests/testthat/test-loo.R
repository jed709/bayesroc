test_that("loo() returns a valid loo object and loo_compare() ranks models", {
  skip_if_no_cmdstan()
  d <- .fixture_univariate(sigma = 1.3)
  m_ev <- broc(brf(item_resp | is_old ~ 1, family = evsd()), d)
  f_ev <- fit_broc(m_ev, backend = "stan", refresh = 50, seed = 1, chains = 2,
                   parallel_chains = 2, iter_warmup = 50, iter_sampling = 50)
  f_uv <- .fx_fit("uvsd", "stan")

  l_ev <- loo(f_ev, cores = 1)
  l_uv <- loo(f_uv, cores = 1)
  expect_s3_class(l_ev, "loo")
  expect_s3_class(l_uv, "loo")

  cmp <- loo_compare(l_ev, l_uv)
  expect_true(is.matrix(cmp) || is.data.frame(cmp))
  expect_true("elpd_diff" %in% colnames(cmp))
  expect_equal(nrow(cmp), 2L)
})
