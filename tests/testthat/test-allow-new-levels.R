# New-level prediction (allow_new_levels) for criterion REs, cumulative cutpoints
# REs, and new by-factor levels of smooth terms. allow_new_levels = FALSE must
# error on new levels; TRUE must run, return finite probabilities, and add
# (rather than zero) the new-level variability.

test_that("criterion RE: new subject samples from prior under allow_new_levels", {
  skip_if_no_cmdstan()
  set.seed(1); ns <- 6; nps <- 60; thr <- c(-1, -0.3, 0.4, 1.1)
  sc <- rnorm(ns, 0, 0.6)
  d <- do.call(rbind, lapply(seq_len(ns), function(s) {
    yo <- as.integer(cut(rnorm(nps, 1, 1), c(-Inf, thr + sc[s], Inf)))
    yn <- as.integer(cut(rnorm(nps, 0, 1), c(-Inf, thr + sc[s], Inf)))
    data.frame(item_resp = c(yo, yn), is_old = c(rep(1L, nps), rep(0L, nps)),
               sub = factor(s, levels = seq_len(ns)))
  }))
  m <- broc(brf(item_resp | is_old ~ 1, criterion ~ 1 + (1 | sub), family = evsd()), d)
  f <- fit_broc(m, backend = "stan", chains = 2, parallel_chains = 2,
                iter_warmup = 300, iter_sampling = 300, refresh = 0, seed = 1)
  nd <- data.frame(item_resp = 1L, is_old = c(0L, 1L), sub = "99")  # new subject

  expect_error(predict(f, newdata = nd, summary = FALSE, ndraws = 100),
               "allow_new_levels")
  pT <- predict(f, newdata = nd, allow_new_levels = TRUE, summary = FALSE,
                ndraws = 300, seed = 2)
  expect_false(anyNA(pT))
  expect_equal(dim(pT)[2:3], c(2L, 5L))
})

test_that("cumulative cutpoints RE supports new levels", {
  skip_if_no_cmdstan()
  set.seed(3); ns <- 6; nps <- 60
  d <- do.call(rbind, lapply(seq_len(ns), function(s)
    data.frame(rating = as.integer(cut(rnorm(nps, 0.3 * s / ns, 1),
                                        c(-Inf, -1, -0.3, 0.3, 1, Inf))),
               sub = factor(s, levels = seq_len(ns)))))
  m <- broc(brf(rating ~ 1, cutpoints ~ 1 + (1 | sub), family = cumulative()), d)
  f <- fit_broc(m, backend = "stan", chains = 2, parallel_chains = 2,
                iter_warmup = 300, iter_sampling = 300, refresh = 0, seed = 1)
  nd <- data.frame(rating = 1L, sub = "99")
  p <- predict(f, newdata = nd, allow_new_levels = TRUE, summary = FALSE,
               ndraws = 200, seed = 2)
  expect_false(anyNA(p))
})

test_that("smooth by-factor: new by-level is prior-sampled, adds variance", {
  skip_if_no_cmdstan()
  set.seed(7); n <- 500
  x <- runif(n, -2, 2); g <- factor(sample(c("p", "q"), n, TRUE))
  lat <- 0.8 + ifelse(g == "q", 0.6 * sin(x * 1.5), 0.4 * x)
  d <- data.frame(item_resp = as.integer(cut(rnorm(n) + lat,
                                             c(-Inf, -1, -0.3, 0.3, 1, Inf))),
                  is_old = rep(0:1, length.out = n), x = x, g = g)
  m <- broc(brf(item_resp | is_old ~ s(x, by = g), criterion ~ 1, family = evsd()), d)
  f <- fit_broc(m, backend = "stan", chains = 2, parallel_chains = 2,
                iter_warmup = 250, iter_sampling = 250, refresh = 0, seed = 1)
  # old items at basis extremes, new by-level "r"
  nd <- data.frame(item_resp = 1L, is_old = 1L, x = c(-1.5, 1.5),
                   g = factor("r", levels = c("p", "q", "r")))
  pT <- predict(f, newdata = nd, allow_new_levels = TRUE,  summary = FALSE, ndraws = 250, seed = 2)
  pF <- predict(f, newdata = nd, allow_new_levels = FALSE, summary = FALSE, ndraws = 250, seed = 2)
  expect_false(anyNA(pT))
  # TRUE samples a prior smooth deviation -> more across-draw spread than FALSE (flat)
  sdT <- mean(apply(pT[, 1, ], 2, sd)); sdF <- mean(apply(pF[, 1, ], 2, sd))
  expect_gt(sdT, sdF)
})
