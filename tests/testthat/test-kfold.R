.kfold_agg <- function(seed = 7, n_sub = 6, n_per = 30) {
  set.seed(seed)
  thresh <- c(-0.6, 0.0, 0.5, 1.0, 1.6)
  d_re <- rnorm(n_sub, 0, 0.4)
  tr <- do.call(rbind, lapply(seq_len(n_sub), function(s) {
    data.frame(sub = s, is_old = rep(1:0, each = n_per),
               item_resp = as.integer(cut(c(rnorm(n_per, 1.2 + d_re[s]), rnorm(n_per)), c(-Inf, thresh, Inf))))
  }))
  tr$sub <- factor(tr$sub)
  list(trials = tr, agg = aggregate(list(n = rep(1L, nrow(tr))), tr[c("sub", "is_old", "item_resp")], sum))
}

test_that("kfold units sum held-out trials per row, fold and group", {
  unit_row <- c(1L, 1L, 2L, 3L, 3L, 3L)
  counts <- c(2L, 1L, 3L)
  gvar <- factor(c("a", "a", "a", "b", "b", "b"))
  sets <- list(list(predicted = c(1L, 3L, 4L)), list(predicted = c(2L, 5L, 6L)))
  u <- bayesroc:::.kfold_units(sets, unit_row, counts, TRUE, "obs", gvar)
  expect_equal(u$rows, list(1L, 2L, 3L, 1L, 3L))
  expect_equal(u$mult, c(1L, 1L, 1L, 1L, 2L))
  expect_equal(unlist(u$trials), c(1L, 3L, 4L, 2L, 5L, 6L))
  f <- bayesroc:::.kfold_units(sets, unit_row, counts, TRUE, "fold", gvar)
  expect_equal(f$rows, list(1:3, c(1L, 3L)))
  expect_equal(f$w, list(c(1L, 1L, 1L), c(1L, 2L)))
  g <- bayesroc:::.kfold_units(sets, unit_row, counts, TRUE, "group", gvar)
  expect_equal(g$rows, list(1:2, 3L, 1L, 3L))
  expect_equal(g$set, c(1L, 1L, 2L, 2L))
  cell <- bayesroc:::.kfold_units(list(list(predicted = c(1L, 3L))), 1:3, counts, FALSE, "obs", NULL)
  expect_equal(unlist(cell$w), c(2L, 3L))
  expect_identical(bayesroc:::.kfold_joint(FALSE), "obs")
  expect_identical(bayesroc:::.kfold_joint(TRUE), "fold")
  expect_error(bayesroc:::.kfold_joint("x"), "joint")
})

test_that("zero-weighted refit models keep every row and switch counts on for trial-level data", {
  d <- .kfold_agg()
  f <- brf(item_resp | is_old ~ 1 + (1 | sub), family = evsd())
  m <- broc(f, data = d$trials)
  expect_null(m$stan_data$counts)
  w <- rep(1L, m$stan_data$N)
  w[1:5] <- 0L
  mk <- bayesroc:::.with_counts(m, w)
  expect_identical(mk$stan_data$counts, w)
  expect_identical(mk$model_data$stan_data$counts, w)
  expect_true(mk$model_data$has_counts)
  expect_true(mk$model_data$rows_live)
  expect_identical(mk$stan_data$N, m$stan_data$N)
  ma <- broc(f, data = d$agg, counts = "n")
  wa <- ma$stan_data$counts
  wa[1] <- 0L
  expect_identical(bayesroc:::.with_counts(ma, wa)$stan_code, ma$stan_code)
})

test_that("kfold leaves out a subject by zero weight and scores it as a new level", {
  skip_if_not_slow()
  skip_if_no_jax()
  d <- .kfold_agg()
  m <- broc(brf(item_resp | is_old ~ 1 + (1 | sub), family = evsd()), data = d$agg, counts = "n")
  fit <- fit_broc(m, backend = "jax", chains = 2, iter_warmup = 200, iter_sampling = 200, refresh = 100, seed = 3)
  kf <- kfold(fit, group = "sub", Ksub = array(2L), seed = 3, refresh = 100)
  expect_s3_class(kf, "kfold")
  held <- rep(d$agg$sub, d$agg$n) == "2"
  expect_equal(nrow(kf$pointwise), sum(held))
  expect_true(all(is.finite(kf$pointwise)))
  expect_equal(attr(kf, "K"), 6L)
  kc <- kfold(fit, K = 2, Ksub = array(1L), type = "cell", joint = "fold", seed = 3, refresh = 100)
  expect_equal(nrow(kc$pointwise), 1L)
  expect_true(is.finite(kc$estimates["elpd_kfold", "Estimate"]))
})
