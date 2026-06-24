# Stan vs JAX agreement tests. The class of bug we fixed in commits
# 05022dc / d6deac8 / etc. would all have been caught by these.
#
# Strategy: fit the same model under both backends with a fixed seed, then
# check posterior means agree to a coarse tolerance. We use coarse tol (0.05
# absolute) because MC noise dominates at 4 chains x 300 iter — true
# agreement under matching analytic specs is to 0.001 or better.

.tol_means <- 0.05

.assert_means_agree <- function(fit_s, fit_j, vars, tol = .tol_means) {
  ds <- posterior::as_draws_matrix(fit_s$draws(variables = vars))
  dj <- posterior::as_draws_matrix(fit_j$draws(variables = vars))
  # Variables may not be in the same order in the two draws_matrix objects
  for (v in vars) {
    if (v %in% colnames(ds) && v %in% colnames(dj)) {
      diff <- abs(mean(ds[, v]) - mean(dj[, v]))
      expect_lt(diff, tol,
                label = sprintf("|mean_stan - mean_jax|(%s) = %.4f", v, diff))
    }
  }
}


test_that("evsdt: Stan and JAX posterior means agree", {
  skip_if_no_cmdstan(); skip_if_no_jax()
  d <- .fixture_univariate()
  m <- broc(brf(item_resp | is_old ~ 1, family = evsd()), d,
            batch_likelihood = TRUE)
  fs <- fit_broc(m, backend = "stan", refresh = 0,
                 parallel_chains = 4, iter_warmup = 400, iter_sampling = 400)
  fj <- fit_broc(m, backend = "jax",
                 iter_warmup = 400, iter_sampling = 400, chains = 4, refresh = 0)
  .assert_means_agree(fs, fj, c("beta_dprime[1]", "beta_thresh_mid[1]"))
})

test_that("uvsdt: Stan and JAX posterior means agree", {
  skip_if_no_cmdstan(); skip_if_no_jax()
  d <- .fixture_univariate(sigma = 1.3)
  m <- broc(brf(item_resp | is_old ~ 1, sigma ~ 1, family = uvsd()), d,
            batch_likelihood = TRUE)
  fs <- fit_broc(m, backend = "stan", refresh = 0,
                 parallel_chains = 4, iter_warmup = 400, iter_sampling = 400)
  fj <- fit_broc(m, backend = "jax",
                 iter_warmup = 400, iter_sampling = 400, chains = 4, refresh = 0)
  .assert_means_agree(fs, fj, c("beta_dprime[1]", "beta_sigma[1]"))
})

test_that("cdp (rotello_2005): Stan and JAX agree on rec/fam/sigma_R/rec_crit", {
  skip_if_no_cmdstan(); skip_if_no_jax()
  d <- .fixture_cdp()
  m <- broc(brf(resp(rating, rk) | is_old ~ condition - 1,
                fam ~ condition - 1, criterion ~ condition - 1,
                rec_crit ~ condition - 1, sigma_R ~ condition - 1,
                family = cdp(old_levels = 2:6), counts = "count"),
            d, batch_likelihood = TRUE)
  fs <- fit_broc(m, backend = "stan", refresh = 0,
                 parallel_chains = 4, iter_warmup = 400, iter_sampling = 400)
  fj <- fit_broc(m, backend = "jax",
                 iter_warmup = 400, iter_sampling = 400, chains = 4, refresh = 0)
  vars <- c("beta_dprime[1]", "beta_dprime[2]",
            "beta_dprime2[1]", "beta_dprime2[2]",
            "beta_sigma[1]", "beta_sigma[2]",
            "beta_rec_crit[1]", "beta_rec_crit[2]")
  .assert_means_agree(fs, fj, vars, tol = 0.03)  # this fit is well-behaved
})

test_that("cumulative: Stan and JAX agree", {
  skip_if_no_cmdstan(); skip_if_no_jax()
  d <- .fixture_univariate()
  m <- broc(brf(item_resp ~ cond, cutpoints ~ 1, family = cumulative()),
            d, batch_likelihood = TRUE)
  fs <- fit_broc(m, backend = "stan", refresh = 0,
                 parallel_chains = 4, iter_warmup = 400, iter_sampling = 400)
  fj <- fit_broc(m, backend = "jax",
                 iter_warmup = 400, iter_sampling = 400, chains = 4, refresh = 0)
  .assert_means_agree(fs, fj,
                      c("beta_mu[1]", "beta_thresh_mid[1]"))
})

# Regression guard: cross-correlated REs `(1|s|sub) + (1|i|item)` on multiple
# parameters used to break prediction. The single-term cross-correlated RE was
# serialized 2D `[i,1]` by JAX but 1D `[i]` by Stan, and get_re_variables
# requested an exact index form that didn't match -- so the RE columns were
# filtered out of the prediction draws and predict()/plot_roc_curve() failed
# with "subscript out of bounds". Both backends must now predict cleanly, and
# the single-term RE must be named identically (1D) across backends.
test_that("crossed REs: predict + plot_roc_curve work on both backends", {
  skip_if_no_cmdstan(); skip_if_no_jax()
  d <- .fixture_crossed()
  m <- broc(brf(item_resp | is_old ~ 1 + (1 | s | sub) + (1 | i | item),
                criterion ~ 1 + (1 | s | sub) + (1 | i | item),
                sigma ~ 1 + (1 | s | sub) + (1 | i | item),
                family = uvsd()),
            d, batch_likelihood = TRUE)

  for (backend in c("stan", "jax")) {
    fit <- if (backend == "stan") {
      fit_broc(m, backend = "stan", refresh = 0, seed = 1,
               parallel_chains = 4, iter_warmup = 300, iter_sampling = 300)
    } else {
      fit_broc(m, backend = "jax", refresh = 0, seed = 1,
               iter_warmup = 300, iter_sampling = 300, chains = 4)
    }

    # type = "response" returns one row per observation x response category.
    pred <- predict(fit, type = "response", summary = TRUE)
    expect_gt(nrow(pred), 0)
    expect_equal(nrow(pred) %% nrow(d), 0, label = paste(backend, "predict rows / N"))

    p <- plot_roc_curve(fit)
    expect_s3_class(p, "ggplot")

    # Single-term cross-correlated RE is named 1D `[i]` (not 2D `[i,1]`) on both
    # backends; the multi-term criterion RE stays 2D `[i,d]`.
    v <- posterior::variables(fit$draws())
    expect_true("u_dprime_sub_from_s[1]" %in% v,
                label = paste(backend, "dprime RE 1D name"))
    expect_false("u_dprime_sub_from_s[1,1]" %in% v,
                 label = paste(backend, "dprime RE not 2D"))
    expect_true("u_criterion_sub_from_s[1,1]" %in% v,
                label = paste(backend, "criterion RE 2D name"))
  }
})
