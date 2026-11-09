.sao_data <- function(partial = TRUE) {
  set.seed(3)
  bv <- .fixture_bivariate_sdt(n_per = 80)
  bv$sub <- factor(sample(1:6, nrow(bv), TRUE))
  bv$y2[bv$y1 <= 2] <- NA
  if (partial) bv$y2[sample(which(bv$y1 == 3), 5)] <- NA
  bv
}

.sao_cases <- function(bv) {
  f2 <- function(fam, ...) brf(resp(y1, y2) | item_type ~ 1 + (1 | sub), discrim ~ 1, rho ~ 1, criterion ~ 1,
                               criterion2 ~ 1, ..., family = fam)
  bc <- .fixture_bivariate_cumulative(n = 200)
  bc$y2[bc$y1 <= 2] <- NA
  list(
    bg = list(f2(bivariate_gaussian()), bv),
    bg_bounded = list(f2(bivariate_gaussian(bounded = TRUE), sigma ~ 1), bv),
    bg_varying = list(f2(bivariate_gaussian(source_criteria = "varying")), bv),
    bdp = list(f2(bivariate_dp(), lambda ~ 1, lambda2 ~ 1, sigma ~ 1), bv),
    vrdp2d_varying = list(brf(resp(y1, y2) | item_type ~ 1 + (1 | sub), dprime2 ~ 1, discrim ~ 1, lambda ~ 1, sigma ~ 1,
                              criterion ~ 1, criterion2 ~ 1, family = vrdp2d(source_criteria = "varying")), bv),
    bcum = list(brf(resp(y1, y2) ~ 1, cutpoints ~ 1, cutpoints2 ~ 1, rho ~ 1, family = bivariate_cumulative()), bc)
  )
}

test_that("missing source ratings are accepted, and varying sets are built only for responses that carry them", {
  bv <- .sao_data()
  f <- function(fam) brf(resp(y1, y2) | item_type ~ 1, discrim ~ 1, rho ~ 1, criterion ~ 1, criterion2 ~ 1, family = fam)
  expect_message(m <- broc(f(bivariate_gaussian()), bv), "Source rating missing on [0-9]+ of 240 rows; detection responses 1, 2 ")
  expect_equal(sum(m$stan_data$src_obs == 0L), sum(is.na(bv$y2)))
  expect_true(all(m$stan_data$y2 %in% seq_len(m$model_data$K2)))
  expect_identical(m$family$source_levels, 3:4)
  mv <- suppressMessages(broc(f(bivariate_gaussian(source_criteria = "varying")), bv))
  expect_equal(mv$stan_data$n_vbin, 2L)
  expect_identical(.varying_bins(mv$family, mv$model_data$K)$labels, c("3", "4"))
  expect_error(suppressMessages(broc(f(bivariate_gaussian(source_criteria = "varying_shared_new", old_levels = 3:4)), bv)),
               "never carry a source rating")
  bc <- .fixture_bivariate_cumulative(n = 200)
  bc$y2[bc$y1 <= 2] <- NA
  mc <- suppressMessages(broc(brf(resp(y1, y2) ~ 1, cutpoints ~ 1, cutpoints2 ~ 1, rho ~ 1,
                                  family = bivariate_cumulative()), bc))
  expect_equal(mc$stan_data$N, 200L)
  expect_length(mc$dropped_rows, 0L)
  full <- .fixture_bivariate_sdt(n_per = 80)
  m0 <- suppressMessages(broc(f(bivariate_gaussian()), full))
  expect_null(m0$stan_data$src_obs)
  expect_null(m0$family$source_levels)
})

test_that("rows without a source rating are scored on the detection response alone (JAX vs R)", {
  skip_if_no_jax()
  for (nm in names(cs <- .sao_cases(.sao_data()))) {
    m <- suppressMessages(broc(cs[[nm]][[1]], cs[[nm]][[2]]))
    f <- suppressMessages(fit_broc(m, backend = "jax", chains = 1, iter_warmup = 100, iter_sampling = 20,
                                   seed = 1))
    dm <- f$draws(format = "matrix")
    ll_r <- .obs_loglik_r(m, build_data_source_from_model(m), dm, re_formula = NULL)
    ll_j <- .read_loglik_SN_cols(.loglik_path(f), 1L, m$model_data$N)
    miss <- m$stan_data$src_obs == 0L
    expect_lt(max(abs(ll_j[, miss] - ll_r[, miss])), 1e-5, label = paste(nm, "missing rows"))
    expect_lt(max(abs(ll_j[, !miss] - ll_r[, !miss])), 1e-5, label = paste(nm, "observed rows"))
  }
})

test_that("pp_check, the source ROC and replicated data follow the design", {
  skip_if_no_jax()
  bv <- .sao_data(partial = FALSE)
  m <- suppressMessages(broc(brf(resp(y1, y2) | item_type ~ 1, discrim ~ 1, rho ~ 1, criterion ~ 1, criterion2 ~ 1,
                                 family = bivariate_gaussian()), bv))
  sd <- .source_design(m)
  expect_true(sd$rule)
  f <- suppressMessages(fit_broc(m, backend = "jax", chains = 2, iter_warmup = 150, iter_sampling = 100, seed = 2))
  b <- .ppc_block(f, m, NULL, 50, NULL, 2L, 1, "bivariate_sdt", TRUE)
  expect_identical(tail(b$cat_labels, 1), "not asked")
  expect_equal(sum(b$obs[, b$K]), sum(m$stan_data$src_obs == 0L))
  expect_equal(sum(b$obs[, -b$K]), sum(m$stan_data$src_obs == 1L))
  expect_lt(abs(mean(apply(b$pred[, , b$K, drop = FALSE], 1, sum)) - sum(b$obs[, b$K])), 0.25 * sum(b$obs[, b$K]))
  pr <- predict(f, type = "prediction", summary = FALSE, ndraws = 20, seed = 1)
  expect_true(all(is.na(pr$y2[!(pr$y1 %in% 3:4)])))
  expect_false(anyNA(pr$y2[pr$y1 %in% 3:4]))
  expect_s3_class(suppressMessages(plot_roc(f, response = 2, ndraws = 20)), "ggplot")
  expect_s3_class(suppressMessages(plot_roc(f, type = "conditional", ndraws = 20)), "ggplot")
  expect_s3_class(suppressMessages(pp_check(f, type = "joint", ndraws = 20)), "ggplot")
})
