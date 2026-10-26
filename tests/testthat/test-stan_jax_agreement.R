
.tol_means <- 0.05

.assert_means_agree <- function(fit_s, fit_j, vars, tol = .tol_means) {
  ds <- posterior::as_draws_matrix(fit_s$draws(variables = vars))
  dj <- posterior::as_draws_matrix(fit_j$draws(variables = vars))
  for (v in vars) {
    if (v %in% colnames(ds) && v %in% colnames(dj)) {
      diff <- abs(mean(ds[, v]) - mean(dj[, v]))
      expect_lt(diff, tol,
                label = sprintf("|mean_stan - mean_jax|(%s) = %.4f", v, diff))
    }
  }
}

test_that("evsdt: Stan and JAX posterior means agree", {
  skip_if_not_slow()
  skip_if_no_cmdstan(); skip_if_no_jax()
  d <- .fixture_univariate()
  m <- broc(brf(item_resp | is_old ~ 1, family = evsd()), d)
  fs <- fit_broc(m, backend = "stan", refresh = 50,
                 parallel_chains = 4, iter_warmup = 400, iter_sampling = 400)
  fj <- fit_broc(m, backend = "jax", sampler = "nuts",
                 iter_warmup = 400, iter_sampling = 400, chains = 4, refresh = 50)
  .assert_means_agree(fs, fj, c("beta_dprime[1]", "beta_thresh_mid[1]"))
})

test_that("uvsdt: Stan and JAX posterior means agree", {
  skip_if_not_slow()
  skip_if_no_cmdstan(); skip_if_no_jax()
  d <- .fixture_univariate(sigma = 1.3)
  m <- broc(brf(item_resp | is_old ~ 1, sigma ~ 1, family = uvsd()), d)
  fs <- fit_broc(m, backend = "stan", refresh = 50,
                 parallel_chains = 4, iter_warmup = 400, iter_sampling = 400)
  fj <- fit_broc(m, backend = "jax", sampler = "nuts",
                 iter_warmup = 400, iter_sampling = 400, chains = 4, refresh = 50)
  .assert_means_agree(fs, fj, c("beta_dprime[1]", "beta_sigma[1]"))
})

test_that("cdp (rotello_2005): Stan and JAX agree on rec/fam/sigma_R/rec_crit", {
  skip_if_not_slow()
  skip_if_no_cmdstan(); skip_if_no_jax()
  d <- .fixture_cdp()
  m <- broc(brf(resp(rating, rk) | is_old ~ condition - 1,
                fam ~ condition - 1, criterion ~ condition - 1,
                rec_crit ~ condition - 1, sigma_R ~ condition - 1,
                family = cdp(old_levels = 2:6), counts = "count"),
            d)
  fs <- fit_broc(m, backend = "stan", refresh = 50,
                 parallel_chains = 4, iter_warmup = 400, iter_sampling = 400)
  fj <- fit_broc(m, backend = "jax", sampler = "nuts",
                 iter_warmup = 400, iter_sampling = 400, chains = 4, refresh = 50)
  vars <- c("beta_dprime[1]", "beta_dprime[2]",
            "beta_dprime2[1]", "beta_dprime2[2]",
            "beta_sigma[1]", "beta_sigma[2]",
            "beta_rec_crit[1]", "beta_rec_crit[2]")
  .assert_means_agree(fs, fj, vars, tol = 0.03)
})

test_that("cumulative: Stan and JAX agree", {
  skip_if_not_slow()
  skip_if_no_cmdstan(); skip_if_no_jax()
  d <- .fixture_univariate()
  m <- broc(brf(item_resp ~ 1, cutpoints ~ cond, cutpoints_scale ~ 1, family = cumulative()),
            d)
  fs <- fit_broc(m, backend = "stan", refresh = 50,
                 parallel_chains = 4, iter_warmup = 400, iter_sampling = 400)
  fj <- fit_broc(m, backend = "jax", sampler = "nuts",
                 iter_warmup = 400, iter_sampling = 400, chains = 4, refresh = 50)
  .assert_means_agree(fs, fj,
                      c("beta_thresh_mid[1]", "beta_thresh_mid[2]"))
})

test_that("bivariate_cumulative: Stan and JAX posterior means agree", {
  skip_if_not_slow()
  skip_if_no_cmdstan(); skip_if_no_jax()
  d <- .fixture_bivariate_cumulative()
  m <- broc(brf(resp(y1, y2) ~ 1,
                discrim ~ 1, criterion ~ 1, criterion2 ~ 1, rho ~ 1,
                family = bivariate_cumulative()), d)
  fs <- fit_broc(m, backend = "stan", refresh = 50,
                 parallel_chains = 4, iter_warmup = 400, iter_sampling = 400)
  fj <- fit_broc(m, backend = "jax", sampler = "nuts",
                 iter_warmup = 400, iter_sampling = 400, chains = 4, refresh = 50)
  .assert_means_agree(fs, fj,
                      c("beta_rho[1]", "beta_thresh_mid[1]", "beta_thresh_mid_2[1]"))
})

test_that("crossed REs: predict + plot_roc work on both backends", {
  skip_if_no_cmdstan(); skip_if_no_jax()
  d <- .fixture_crossed()
  m <- .fx_memo("model:uvsd_crossed_cross", broc(brf(item_resp | is_old ~ 1 + (1 | s | sub) + (1 | i | item),
                                                    criterion ~ 1 + (1 | s | sub) + (1 | i | item),
                                                    sigma ~ 1 + (1 | s | sub) + (1 | i | item),
                                                    family = uvsd()),
                                                d))

  for (backend in c("stan", "jax")) {
    fit <- .fx_memo(paste0("fit:uvsd_crossed_cross:", backend), if (backend == "stan") {
      fit_broc(m, backend = "stan", refresh = 50, seed = 1, chains = 2,
               parallel_chains = 2, iter_warmup = 30, iter_sampling = 20)
    } else {
      fit_broc(m, backend = "jax", sampler = "nuts", seed = 1, progress_bar = FALSE,
               iter_warmup = 30, iter_sampling = 20, chains = 2, parallel_chains = 2)
    })

    pred <- predict(fit, type = "response", summary = TRUE)
    expect_gt(nrow(pred), 0)
    expect_equal(nrow(pred) %% nrow(d), 0, label = paste(backend, "predict rows / N"))

    p <- plot_roc(fit)
    expect_s3_class(p, "ggplot")
    for (ty in c("sd", "cor")) expect_no_error(ggplot2::ggplot_build(plot_random_effects(fit, type = ty)))

    v <- posterior::variables(fit$draws())
    expect_true("u_dprime_sub_from_s[1]" %in% v,
                label = paste(backend, "dprime RE 1D name"))
    expect_false("u_dprime_sub_from_s[1,1]" %in% v,
                 label = paste(backend, "dprime RE not 2D"))
    expect_true("u_criterion_sub_from_s[1,1]" %in% v,
                label = paste(backend, "criterion RE 2D name"))
  }
})

test_that("JAX NUTS through the transport map and on the SD quantile scale recovers Stan's posterior (f64 and f32)", {
  skip_if_not_slow()
  skip_if_no_cmdstan(); skip_if_no_jax()
  d <- .fixture_crossed(n_sub = 12, n_item = 24)
  m <- broc(brf(item_resp | is_old ~ 1 + (1 | sub) + (1 | item), sigma ~ 1 + (1 | sub), family = uvsd()), d)
  sm <- function(f) {
    da <- f$draws(format = "draws_array")
    v <- grep("^(beta|sigma|u_)", posterior::variables(da), value = TRUE)
    s <- posterior::summarise_draws(posterior::subset_draws(da, variable = v), "mean", "sd", "mcse_mean")
    s[order(s$variable), ]
  }
  rs <- sm(fit_broc(m, backend = "stan", chains = 4, parallel_chains = 4, iter_warmup = 1000,
                    iter_sampling = 5000, seed = 1, progress_bar = FALSE))
  for (run in list(list(TRUE, "f64"), list(FALSE, "f64"), list(TRUE, "f32"))) {
    tr <- run[[1]]
    out <- utils::capture.output(
      fj <- fit_broc(m, backend = "jax", sampler = "nuts", chains = 4, iter_warmup = 1000, iter_sampling = 5000, seed = 3,
                     progress_bar = TRUE, control = list(transport = tr, precision = run[[2]])))
    expect_identical(sum(grepl("[NUTS] group transport map", out, fixed = TRUE)), as.integer(tr))
    js <- sm(fj)
    expect_identical(js$variable, rs$variable)
    z <- abs(js$mean - rs$mean) / sqrt(js$mcse_mean^2 + rs$mcse_mean^2)
    expect_lt(max(z), 4)
    expect_true(all(abs(js$sd / rs$sd - 1) < 0.1), label = paste("posterior spreads, transport", tr))
    expect_equal(sum(fj$diagnostic_summary()$num_divergent), 0)
  }
})
