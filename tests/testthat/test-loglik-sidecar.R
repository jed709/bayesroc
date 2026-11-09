.agg_fixture <- function(n_per = 200) {
  d <- .fixture_univariate(sigma = 1.3, n_per = n_per)
  agg <- aggregate(list(n = rep(1L, nrow(d))),
                   by = list(item_resp = d$item_resp, is_old = d$is_old),
                   FUN = sum)
  list(trial = d, agg = agg)
}

.ls_model <- function(n_per = 200) {
  .fx_memo(paste0("ls:model:", n_per),
           broc(brf(item_resp | is_old ~ 1, family = evsd()), .agg_fixture(n_per)$agg, counts = "n"))
}

.ls_stan <- function(m, ...) {
  fit_broc(m, seed = 1, chains = 2,
           parallel_chains = 2, iter_warmup = 50, iter_sampling = 100, ...)
}

.ls_stan_f64 <- function() .fx_memo("ls:stan_f64", .ls_stan(.ls_model(), loglik_precision = "f64"))

.ls_jax_loo <- function() {
  .fx_memo("ls:jax_loo", fit_broc(.ls_model(120), backend = "jax", sampler = "nuts", chains = 2,
                                  parallel_chains = 2, iter_warmup = 30, iter_sampling = 50, seed = 4,
                                  loo = TRUE, save_loglik = FALSE, progress_bar = FALSE))
}

test_that("streamed loo matches the brute-force trial-level oracle", {
  skip_if_no_jax()
  fx <- .agg_fixture()
  expect_lt(nrow(fx$agg), nrow(fx$trial))
  m <- .ls_model()
  f <- .ls_stan_f64()

  l <- loo(f, cores = 1)
  expect_equal(dim(l)[2], nrow(fx$trial))

  pr <- predict(f, type = "response", summary = FALSE)
  y <- m$stan_data$y
  ll <- vapply(seq_along(y), function(n) log(pr[, n, y[n]]), numeric(dim(pr)[1]))
  expect_equal(ll, unname(unclass(as.matrix(f$draws(variables = "log_lik", format = "matrix")))),
               tolerance = 1e-5, ignore_attr = TRUE)
  cnt <- m$stan_data$counts
  ll_exp <- ll[, rep.int(seq_len(ncol(ll)), cnt), drop = FALSE]
  cid <- rep(seq_len(f$num_chains()), each = nrow(ll) / f$num_chains())
  ref <- loo::loo(ll_exp, r_eff = loo::relative_eff(exp(ll_exp), chain_id = cid),
                  cores = 1)

  expect_equal(l$estimates, ref$estimates, tolerance = 1e-5)
  expect_equal(unname(l$pointwise[, "elpd_loo"]), unname(ref$pointwise[, "elpd_loo"]),
               tolerance = 1e-5)
  expect_equal(unname(l$diagnostics$pareto_k), unname(ref$diagnostics$pareto_k),
               tolerance = 1e-4)

  l4 <- loo(f, cores = 4)
  expect_equal(l4$estimates, l$estimates, tolerance = 1e-8)

  l_cell <- suppressWarnings(loo(f, cores = 1, type = "cell"))
  expect_equal(dim(l_cell)[2], nrow(fx$agg))
  expect_false(isTRUE(all.equal(l_cell$estimates["elpd_loo", "Estimate"],
                                l$estimates["elpd_loo", "Estimate"])))

  expect_equal(dim(f$loo(cores = 1))[2], nrow(fx$trial))
})

test_that("chunking does not change the streamed result", {
  skip_if_no_jax()
  f <- .ls_stan_f64()
  wide <- loo(f, cores = 1)
  narrow <- loo(f, cores = 2, chunk_mb = 0.001)
  expect_equal(narrow$estimates, wide$estimates, tolerance = 1e-8)
  expect_equal(dim(narrow)[2], dim(wide)[2])
})

test_that("loo = TRUE computes at fit time and reclaims the sidecar", {
  skip_if_no_jax()
  fx <- .agg_fixture()
  f <- .ls_stan(.ls_model(), loo = TRUE, save_loglik = FALSE)

  expect_s3_class(f$.loo, "loo")
  expect_equal(dim(f$.loo)[2], nrow(fx$trial))
  expect_null(f$.loglik_path)
  expect_equal(loo(f)$estimates, f$.loo$estimates)
})

test_that("drop_loglik frees a temp sidecar and disables loo", {
  skip_if_no_jax()
  f <- .ls_stan(.ls_model())
  p <- f$.loglik_path
  expect_true(file.exists(p))

  f <- suppressMessages(drop_loglik(f))
  expect_false(file.exists(p))
  expect_null(f$.loglik_path)
  expect_error(loo(f, cores = 1), "No log_lik is available")
})

test_that("r_eff survives log_lik far below exp() underflow", {
  set.seed(3)
  ll <- matrix(rnorm(400 * 5, -900, 0.5), 400, 5)
  cid <- rep(1:4, each = 100)
  r <- .stable_r_eff(ll, cid, 1)
  expect_true(all(is.finite(r)))
  ref <- loo::relative_eff(exp(ll + 900), chain_id = cid)
  expect_equal(unname(r), unname(ref), tolerance = 0.05)
})

test_that("loo() errors actionably once log_lik is gone", {
  skip_if_no_jax()
  f <- .ls_jax_loo()
  expect_error(loo(f, type = "cell"), "No log_lik is available")
})

test_that("save_broc_fit persists the sidecar only when asked", {
  skip_if_no_jax()
  f <- .ls_stan(.ls_model(), loglik_precision = "f64")
  ref <- loo(f, cores = 1)$estimates

  plain <- tempfile(fileext = ".rds")
  suppressMessages(save_broc_fit(f, plain, verbose = FALSE, include_loglik = FALSE))
  expect_false(file.exists(paste0(tools::file_path_sans_ext(plain), "_loglik.npy")))

  kept <- tempfile(fileext = ".rds")
  suppressMessages(save_broc_fit(f, kept, verbose = FALSE))
  side <- paste0(tools::file_path_sans_ext(kept), "_loglik.npy")
  expect_true(file.exists(side))

  f2 <- load_broc_fit(kept)
  expect_equal(normalizePath(f2$.loglik_path), normalizePath(side))
  expect_equal(loo(f2, cores = 1)$estimates, ref, tolerance = 1e-8)

  f2 <- suppressMessages(drop_loglik(f2))
  expect_true(file.exists(side))
  expect_equal(loo(f2, cores = 1)$estimates, ref, tolerance = 1e-8)

  unlink(side)
  expect_warning(f3 <- load_broc_fit(kept), "was not found")
  expect_error(loo(f3, cores = 1), "No log_lik is available")
})

test_that("a fit-time loo result survives save and load", {
  skip_if_no_jax()
  fx <- .agg_fixture()
  m <- .ls_model()
  f <- .ls_stan(m, loo = TRUE, save_loglik = FALSE)
  file <- tempfile(fileext = ".rds")
  suppressMessages(save_broc_fit(f, file, verbose = FALSE))
  f2 <- load_broc_fit(file)
  expect_equal(loo(f2)$estimates, f$.loo$estimates)
  expect_error(loo(f2, type = "cell"), "only a stored type = 'trial'")

  k <- .ls_stan(m, loo = TRUE, save_loglik = TRUE)
  expect_true(file.exists(k$.loglik_path))
  expect_equal(dim(suppressWarnings(loo(k, cores = 1, type = "cell")))[2],
               nrow(fx$agg))
})

test_that("keep_re controls whether JAX returns per-level RE draws", {
  skip_if_no_jax()
  d <- .fixture_crossed(n_sub = 6, n_item = 10)
  m <- broc(brf(item_resp | is_old ~ 1 + (1 | sub), family = evsd()), d)

  f0 <- fit_broc(m, backend = "jax", sampler = "nuts", chains = 1,
                 iter_warmup = 30, iter_sampling = 10, seed = 4,
                 progress_bar = FALSE)
  expect_false(any(grepl("^u_dprime_sub_raw", posterior::variables(f0$.draws_array))))

  f1 <- fit_broc(m, backend = "jax", sampler = "nuts", chains = 1,
                 iter_warmup = 30, iter_sampling = 10, seed = 4,
                 keep_re = TRUE, progress_bar = FALSE)
  expect_true(any(grepl("^u_dprime_sub_raw", posterior::variables(f1$.draws_array))))
})

test_that("JAX honours loo = TRUE and reclaims the sidecar", {
  skip_if_no_jax()
  fx <- .agg_fixture(n_per = 120)
  f <- .ls_jax_loo()
  expect_s3_class(f$.loo, "loo")
  expect_equal(dim(f$.loo)[2], nrow(fx$trial))
  expect_null(f$.loglik_path)
})
