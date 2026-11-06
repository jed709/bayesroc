test_that("fit summaries report posterior's ESS for the reversible samplers and batch means for DR-G-HMC", {
  da <- posterior::example_draws()
  x <- posterior::extract_variable_matrix(da, "mu")
  s <- do.call(posterior::summarise_draws, c(list(da), .summary_measures()))
  expect_named(s, c("variable", "mean", "median", "sd", "mad", "q5", "q95", "rhat", "ess_bulk", "ess_tail"))
  expect_equal(s$ess_bulk[1], posterior::ess_bulk(x))
  expect_equal(s$ess_tail[1], posterior::ess_tail(x))
  expect_equal(.ess_bulk(x), posterior::ess_bulk(x))
  sd <- do.call(posterior::summarise_draws, c(list(da), .summary_measures(.sampler = "drghmc")))
  expect_named(sd, names(s))
  expect_equal(sd$ess_bulk[1], .ess_bulk(x, "drghmc"))
  expect_equal(sd$ess_tail[1], .ess_tail(x, "drghmc"))
  s2 <- do.call(posterior::summarise_draws, c(list(da), .summary_measures("mean", "ess_tail", .sampler = "drghmc")))
  expect_named(s2, c("variable", "mean", "ess_tail"))
  expect_equal(s2$ess_tail[1], .ess_tail(x, "drghmc"))
})

test_that("batch-means ESS is near the draw count for independent draws", {
  set.seed(3)
  x <- matrix(stats::rnorm(4000 * 4), 4000)
  expect_gt(.ess_bulk(x, "drghmc"), 0.8 * length(x))
  expect_lt(.ess_bulk(x, "drghmc"), 1.25 * length(x))
  expect_gt(.ess_tail(x, "drghmc"), 0.8 * length(x))
  expect_true(is.na(.ess_bulk(matrix(1, 50, 4), "drghmc")))
  expect_true(is.finite(.ess_bulk(matrix(stats::rnorm(120), 30), "drghmc")))
})

test_that("an oscillating chain's tail ESS is not overstated", {
  set.seed(4)
  n <- 4000
  x <- sapply(1:4, function(c) {
    a <- cumsum(stats::rnorm(n, sd = 0.15))
    amp <- exp(as.numeric(stats::filter(stats::rnorm(n, sd = 0.2), 0.96, "recursive")))
    amp * cos(2 * pi * seq_len(n) / 5 + a) + 0.3 * stats::rnorm(n)
  })
  q <- stats::quantile(x, 0.05, names = FALSE)
  true_ess <- local({
    ind <- (x <= q) + 0
    r <- rowMeans(sapply(1:4, function(c) stats::acf(ind[, c], lag.max = 200, plot = FALSE)$acf[-1]))
    length(x) / (1 + 2 * sum(r))
  })
  expect_gt(posterior::ess_tail(x) / true_ess, 2)
  expect_lt(abs(log(.ess_tail(x, "drghmc") / true_ess)), log(1.6))
})
