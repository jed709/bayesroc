# LOO model comparison -- the package's headline workflow. loo() must return a
# valid loo object, and the re-exported loo_compare() must rank competing models.

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
