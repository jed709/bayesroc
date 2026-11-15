.jp_iter <- if (identical(Sys.getenv("BAYESROC_TEST_FAST", "true"), "true")) 100 else 200
.jp_chains <- if (identical(Sys.getenv("BAYESROC_TEST_FAST", "true"), "true")) 1 else 2
.jp_renders <- function(g) {
  expect_s3_class(g, "ggplot")
  expect_no_error(ggplot2::ggplot_build(g))
}

test_that(".bvn_density integrates to one and .scan_th2 lays the varying criteria out in order", {
  g <- seq(-7, 7, by = 0.05)
  G <- expand.grid(x = g, y = g)
  z <- .bvn_density(G$x, G$y, 0.4, -0.3, 1.3, 0.8, 0.45)
  expect_equal(sum(z) * 0.05^2, 1, tolerance = 1e-4)
  model <- list(model_data = list(varying_source_criteria = TRUE))
  vals <- c(-1, 0, 2)
  a <- .scan_th2(model, S = 3, n = 2, K1 = 4, vals)
  expect_equal(dim(a), c(3, 2, 4, 3))
  for (j in 1:4) expect_equal(as.vector(a[2, 1, j, ]), vals)
  model$model_data$varying_source_criteria <- FALSE
  b <- .scan_th2(model, S = 3, n = 2, K1 = 4, vals)
  expect_equal(dim(b), c(3, 2, 3))
  expect_equal(as.vector(b[1, 2, ]), vals)
})

test_that("the CDP latent plane draws and other families refuse the joint displays", {
  skip_if_no_jax()
  d <- .fixture_cdp()
  m <- broc(brf(resp(rating, rk) | is_old ~ condition - 1, fam ~ condition - 1, criterion ~ condition - 1,
                rec_crit ~ condition - 1, sigma_R ~ condition - 1, family = cdp(old_levels = 2:6)), d, counts = "count")
  fit <- fit_broc(m, chains = .jp_chains,
                  iter_warmup = .jp_iter, iter_sampling = .jp_iter)
  .jp_renders(plot_distributions(fit, type = "joint", ndraws = 20))
  .jp_renders(plot_distributions(fit, type = "joint", by = "condition", ndraws = 20))
  expect_error(plot_roc(fit, type = "conditional"), "bivariate_sdt, bivariate_dp and vrdp2d")
  expect_error(plot_roc(fit, type = "joint"), "bivariate_sdt, bivariate_dp and vrdp2d")
})

test_that("the joint pp_check draws for a bivariate family and the latent plane for bivariate_cumulative", {
  skip_if_no_jax()
  d <- .fixture_bivariate_sdt()
  m <- broc(brf(resp(y1, y2) | item_type ~ 1, discrim ~ 1, criterion ~ 1, criterion2 ~ 1, rho ~ 1,
                family = bivariate_gaussian()), d)
  fit <- fit_broc(m, chains = .jp_chains, iter_warmup = .jp_iter, iter_sampling = .jp_iter)
  .jp_renders(pp_check(fit, type = "joint", ndraws = 20))
  blk <- .ppc_joint_block(fit, .plot_model(fit), NULL, 20, NULL, 1, "bivariate_sdt")
  expect_equal(sum(blk$df$observed), nrow(d))
  expect_equal(nrow(blk$df), 3L * 4L * 4L)
  d2 <- .fixture_bivariate_cumulative()
  m2 <- broc(brf(resp(y1, y2) ~ 1, mu2 ~ 1, cutpoints ~ 1, cutpoints2 ~ 1, rho ~ 1, family = bivariate_cumulative()), d2)
  fit2 <- fit_broc(m2, chains = .jp_chains, iter_warmup = .jp_iter, iter_sampling = .jp_iter)
  .jp_renders(pp_check(fit2, type = "joint", ndraws = 20))
  .jp_renders(plot_distributions(fit2, type = "joint", ndraws = 20))
  .jp_renders(plot_distributions(fit2, ndraws = 20))            # both ordinal dimensions in one figure
  expect_error(plot_roc(fit2), "plot_distributions")
  expect_error(pp_check(fit, type = "joint", response = 2), NA)
})
