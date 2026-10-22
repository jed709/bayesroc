.jp_iter <- if (identical(Sys.getenv("BAYESROC_TEST_FAST", "true"), "true")) 300 else 1000
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

test_that("joint displays draw for a bivariate family and agree with the family's own margins", {
  skip_if_no_cmdstan()
  # the fixture puts source A above B on the source axis; the package's source ROC and the three-way
  # surface read "called B" as a high source rating, so swap the labels
  d <- .fixture_bivariate_sdt()
  d$item_type <- factor(c(new = "new", A = "B", B = "A")[as.character(d$item_type)], levels = c("new", "A", "B"))
  m <- broc(brf(resp(y1, y2) | item_type ~ 1,
                discrim ~ 1, criterion ~ 1, criterion2 ~ 1, rho ~ 1,
                family = bivariate_gaussian()), d)
  fit <- fit_broc(m, backend = "stan", refresh = 50,
                  parallel_chains = 4,
                  iter_warmup = .jp_iter, iter_sampling = .jp_iter)
  .jp_renders(plot_distributions(fit, type = "joint", ndraws = 20))
  .jp_renders(plot_roc(fit, type = "conditional", ndraws = 20))
  .jp_renders(plot_roc(fit, type = "conditional", scale = "z", observed = FALSE, ndraws = 20))
  .jp_renders(plot_roc(fit, type = "joint", ndraws = 10))
  expect_error(plot_distributions(fit, type = "nonsense"))

  joint <- predict(fit, summary = FALSE, ndraws = 20, seed = 5)
  expect_length(dim(joint), 4L)
  p1 <- predict(fit, summary = FALSE, ndraws = 20, seed = 5, response = 1)
  p2 <- predict(fit, summary = FALSE, ndraws = 20, seed = 5, response = 2, cores = 2)
  expect_equal(p1, rowSums(joint, dims = 3L), tolerance = 1e-8)
  expect_equal(p2, rowSums(aperm(joint, c(1, 2, 4, 3)), dims = 3L), tolerance = 1e-8)
  s2 <- predict(fit, ndraws = 20, seed = 5, response = 2)
  expect_setequal(unique(s2$category), seq_len(dim(p2)[3]))
  expect_equal(nrow(s2), dim(p2)[2] * dim(p2)[3])
  expect_equal(dim(predict(fit, type = "prediction", summary = FALSE, ndraws = 20, response = 1)), dim(p1)[1:2])
  expect_error(predict(fit, response = 3), "NULL, 1 or 2")

  # the surface: P(say new | new) is the new-item detection margin, so x = pnorm(c) exactly, the
  # rates lie in [0, 1], and VUS lies between chance and one
  model <- .plot_model(fit)
  cells <- .cell_of_rows(model, NULL)
  cf <- .cell_frame(model, NULL, cells)
  setup <- .plot_setup(fit, model, 10, NA)
  parts <- .roc_joint(model, setup, cells, cf, .row_weights(model), 0.9, TRUE, nx = 9L, nk = 41L)
  expect_true(all(parts$grid$z >= 0 & parts$grid$z <= 1, na.rm = TRUE))
  expect_true(all(parts$pts$x >= 0 & parts$pts$x <= 1 & parts$pts$y >= 0 & parts$pts$z <= 1))
  expect_true(parts$vus$vus > 1 / 6 && parts$vus$vus < 1)
  # every rating-grid rule has a model rate near the surface drawn through the posterior means
  g <- parts$grid[!is.na(parts$grid$z), ]
  for (r in seq_len(nrow(parts$pts))) {
    dd <- (g$x - parts$pts$x[r])^2 + (g$y - parts$pts$y[r])^2
    if (min(dd) < 0.02) expect_lt(abs(g$z[which.min(dd)] - parts$pts$z[r]), 0.35)
  }

  # the conditional ROC's band masses sum to the detection margin: hit rates at the lowest source
  # criterion are one and at the highest are zero
  cparts <- .roc_conditional(model, setup, cells, cf, .row_weights(model), NULL, "probability", 0.9, TRUE)
  expect_equal(nlevels(cparts$curves$.band), model$model_data$K)
  for (b in levels(cparts$curves$.band)) {
    cv <- cparts$curves[cparts$curves$.band == b, ]
    expect_equal(cv$y[1], 0); expect_equal(cv$y[nrow(cv)], 1)
    expect_true(all(diff(cv$y) >= -1e-8))
  }
})

test_that("the CDP latent plane draws and other families refuse the joint displays", {
  skip_if_no_cmdstan()
  d <- .fixture_cdp()
  m <- broc(brf(resp(rating, rk) | is_old ~ condition - 1, fam ~ condition - 1, criterion ~ condition - 1,
                rec_crit ~ condition - 1, sigma_R ~ condition - 1, family = cdp(old_levels = 2:6)), d, counts = "count")
  fit <- fit_broc(m, backend = "stan", refresh = 50,
                  parallel_chains = 4,
                  iter_warmup = .jp_iter, iter_sampling = .jp_iter)
  .jp_renders(plot_distributions(fit, type = "joint", ndraws = 20))
  .jp_renders(plot_distributions(fit, type = "joint", by = "condition", ndraws = 20))
  expect_error(plot_roc(fit, type = "conditional"), "bivariate_sdt, bivariate_dp and vrdp2d")
  expect_error(plot_roc(fit, type = "joint"), "bivariate_sdt, bivariate_dp and vrdp2d")
})

test_that("the joint pp_check draws for a bivariate family and the latent plane for bivariate_cumulative", {
  skip_if_no_cmdstan()
  d <- .fixture_bivariate_sdt()
  m <- broc(brf(resp(y1, y2) | item_type ~ 1, discrim ~ 1, criterion ~ 1, criterion2 ~ 1, rho ~ 1,
                family = bivariate_gaussian()), d)
  fit <- fit_broc(m, backend = "stan", refresh = 50, parallel_chains = 4, iter_warmup = .jp_iter, iter_sampling = .jp_iter)
  .jp_renders(pp_check(fit, type = "joint", ndraws = 20))
  blk <- .ppc_joint_block(fit, .plot_model(fit), NULL, 20, NULL, 1, "bivariate_sdt")
  expect_equal(sum(blk$df$observed), nrow(d))
  expect_equal(nrow(blk$df), 3L * 4L * 4L)
  d2 <- .fixture_bivariate_cumulative()
  m2 <- broc(brf(resp(y1, y2) ~ 1, mu2 ~ 1, cutpoints ~ 1, cutpoints2 ~ 1, rho ~ 1, family = bivariate_cumulative()), d2)
  fit2 <- fit_broc(m2, backend = "stan", refresh = 50, parallel_chains = 4, iter_warmup = .jp_iter, iter_sampling = .jp_iter)
  .jp_renders(pp_check(fit2, type = "joint", ndraws = 20))
  .jp_renders(plot_distributions(fit2, type = "joint", ndraws = 20))
  .jp_renders(plot_distributions(fit2, ndraws = 20))            # both ordinal dimensions in one figure
  expect_error(plot_roc(fit2), "plot_distributions")
  expect_error(pp_check(fit, type = "joint", response = 2), NA)
})
