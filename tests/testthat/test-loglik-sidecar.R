.agg_fixture <- function(n_per = 200) {
  d <- .fixture_univariate(sigma = 1.3, n_per = n_per)
  agg <- aggregate(list(n = rep(1L, nrow(d))),
                   by = list(item_resp = d$item_resp, is_old = d$is_old),
                   FUN = sum)
  list(trial = d, agg = agg)
}

test_that("Stan fits write a log_lik sidecar and keep it out of the draws array", {
  skip_if_no_cmdstan()
  fx <- .agg_fixture()
  m <- broc(brf(item_resp | is_old ~ 1, family = evsd()), fx$agg, counts = "n")
  f <- fit_broc(m, backend = "stan", refresh = 0, seed = 1,
                parallel_chains = 4, iter_warmup = 300, iter_sampling = 300)

  expect_true(!is.null(f$.loglik_path))
  expect_true(file.exists(f$.loglik_path))
  meta <- .loglik_meta(f$.loglik_path)
  expect_equal(meta$C, 4L)
  expect_equal(meta$D, 300L)
  expect_equal(meta$N, nrow(fx$agg))
  expect_equal(meta$isz, 4L)

  ll <- as.matrix(f$draws(variables = "log_lik", format = "matrix"))
  expect_equal(dim(ll), c(1200L, nrow(fx$agg)))
})

test_that("streamed loo matches the brute-force trial-level oracle", {
  skip_if_no_cmdstan()
  fx <- .agg_fixture()
  m <- broc(brf(item_resp | is_old ~ 1, family = evsd()), fx$agg, counts = "n")
  f <- fit_broc(m, backend = "stan", refresh = 0, seed = 1,
                parallel_chains = 4, iter_warmup = 300, iter_sampling = 300,
                loglik_precision = "f64")

  l <- loo(f, cores = 1)
  expect_equal(dim(l)[2], nrow(fx$trial))

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
  expect_equal(dim(l_cell)[2], nrow(fx$agg))
})

test_that("chunking does not change the streamed result", {
  skip_if_no_cmdstan()
  fx <- .agg_fixture()
  m <- broc(brf(item_resp | is_old ~ 1, family = evsd()), fx$agg, counts = "n")
  f <- fit_broc(m, backend = "stan", refresh = 0, seed = 1,
                parallel_chains = 4, iter_warmup = 300, iter_sampling = 300,
                loglik_precision = "f64")
  wide <- loo(f, cores = 1)
  narrow <- loo(f, cores = 2, chunk_mb = 0.001)
  expect_equal(narrow$estimates, wide$estimates, tolerance = 1e-8)
  expect_equal(dim(narrow)[2], dim(wide)[2])
})

test_that("loo = TRUE computes at fit time and reclaims the sidecar", {
  skip_if_no_cmdstan()
  fx <- .agg_fixture()
  m <- broc(brf(item_resp | is_old ~ 1, family = evsd()), fx$agg, counts = "n")
  f <- fit_broc(m, backend = "stan", refresh = 0, seed = 1,
                parallel_chains = 4, iter_warmup = 300, iter_sampling = 300,
                loo = TRUE)

  expect_s3_class(f$.loo, "loo")
  expect_equal(dim(f$.loo)[2], nrow(fx$trial))
  expect_null(f$.loglik_path)
  expect_equal(loo(f)$estimates, f$.loo$estimates)
})

test_that("drop_loglik frees a temp sidecar and disables loo", {
  skip_if_no_cmdstan()
  fx <- .agg_fixture()
  m <- broc(brf(item_resp | is_old ~ 1, family = evsd()), fx$agg, counts = "n")
  f <- fit_broc(m, backend = "stan", refresh = 0, seed = 1,
                parallel_chains = 4, iter_warmup = 300, iter_sampling = 300)
  p <- f$.loglik_path
  expect_true(file.exists(p))

  f <- suppressMessages(drop_loglik(f))
  expect_false(file.exists(p))
  expect_null(f$.loglik_path)
})

test_that("loo() errors actionably once log_lik is gone", {
  skip_if_no_jax()
  fx <- .agg_fixture(n_per = 120)
  m <- broc(brf(item_resp | is_old ~ 1, family = evsd()), fx$agg, counts = "n")
  f <- fit_broc(m, backend = "jax", chains = 2, parallel_chains = 2,
                iter_warmup = 200, iter_sampling = 200, seed = 6,
                loo = TRUE, progress_bar = FALSE)
  expect_error(loo(f, type = "cell"), "No log_lik is available")
})

test_that("save_broc_fit persists the sidecar only when asked", {
  skip_if_no_cmdstan()
  fx <- .agg_fixture()
  m <- broc(brf(item_resp | is_old ~ 1, family = evsd()), fx$agg, counts = "n")
  f <- fit_broc(m, backend = "stan", refresh = 0, seed = 1,
                parallel_chains = 4, iter_warmup = 300, iter_sampling = 300,
                loglik_precision = "f64")
  ref <- loo(f, cores = 1)$estimates

  plain <- tempfile(fileext = ".rds")
  suppressMessages(save_broc_fit(f, plain, verbose = FALSE))
  expect_false(file.exists(paste0(tools::file_path_sans_ext(plain), "_loglik.npy")))

  kept <- tempfile(fileext = ".rds")
  suppressMessages(save_broc_fit(f, kept, verbose = FALSE, include_loglik = TRUE))
  side <- paste0(tools::file_path_sans_ext(kept), "_loglik.npy")
  expect_true(file.exists(side))

  f2 <- load_broc_fit(kept)
  expect_equal(normalizePath(f2$.loglik_path), normalizePath(side))
  expect_equal(loo(f2, cores = 1)$estimates, ref, tolerance = 1e-8)
})

test_that("keep_re controls whether JAX returns per-level RE draws", {
  skip_if_no_jax()
  d <- .fixture_crossed(n_sub = 6, n_item = 10)
  m <- broc(brf(item_resp | is_old ~ 1 + (1 | sub), family = evsd()), d)

  f0 <- fit_broc(m, backend = "jax", chains = 2, parallel_chains = 2,
                 iter_warmup = 200, iter_sampling = 200, seed = 4,
                 progress_bar = FALSE)
  expect_false(any(grepl("^z_", posterior::variables(f0$.draws_array))))

  f1 <- fit_broc(m, backend = "jax", chains = 2, parallel_chains = 2,
                 iter_warmup = 200, iter_sampling = 200, seed = 4,
                 keep_re = TRUE, progress_bar = FALSE)
  expect_true(any(grepl("^z_", posterior::variables(f1$.draws_array))))
})

test_that("JAX honours loo = TRUE and reclaims the sidecar", {
  skip_if_no_jax()
  fx <- .agg_fixture(n_per = 120)
  m <- broc(brf(item_resp | is_old ~ 1, family = evsd()), fx$agg, counts = "n")
  f <- fit_broc(m, backend = "jax", chains = 2, parallel_chains = 2,
                iter_warmup = 300, iter_sampling = 300, seed = 4,
                loo = TRUE, progress_bar = FALSE)
  expect_s3_class(f$.loo, "loo")
  expect_equal(dim(f$.loo)[2], nrow(fx$trial))
  expect_null(f$.loglik_path)
})

test_that("Stan and JAX sidecars agree on the same model", {
  skip_if_no_cmdstan()
  skip_if_no_jax()
  fx <- .agg_fixture(n_per = 120)
  m <- broc(brf(item_resp | is_old ~ 1, family = evsd()), fx$agg, counts = "n")
  fs <- fit_broc(m, backend = "stan", refresh = 0, seed = 2,
                 parallel_chains = 2, chains = 2,
                 iter_warmup = 300, iter_sampling = 300)
  fj <- fit_broc(m, backend = "jax", seed = 2, chains = 2, parallel_chains = 2,
                 iter_warmup = 300, iter_sampling = 300, progress_bar = FALSE)

  ms <- .loglik_meta(fs$.loglik_path)
  mj <- .loglik_meta(fj$.loglik_path)
  expect_equal(ms$N, mj$N)
  expect_equal(ms$C, mj$C)
  expect_equal(ms$D, mj$D)

  ls <- loo(fs, cores = 1)
  lj <- loo(fj, cores = 1)
  expect_equal(dim(ls)[2], dim(lj)[2])
  expect_equal(ls$estimates["elpd_loo", "Estimate"],
               lj$estimates["elpd_loo", "Estimate"],
               tolerance = 3 * ls$estimates["elpd_loo", "SE"])
})
