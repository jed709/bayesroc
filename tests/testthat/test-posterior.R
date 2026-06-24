# Tests for posterior_linpred / posterior_epred / allow_new_levels.

test_that("posterior_linpred returns S x N matrix", {
  skip_if_no_cmdstan()
  d <- .fixture_univariate(sigma = 1.3)
  m <- broc(brf(item_resp | is_old ~ 1, sigma ~ 1, family = uvsd()), d,
            batch_likelihood = TRUE)
  fit <- fit_broc(m, backend = "stan", refresh = 0,
                  parallel_chains = 4, iter_warmup = 300, iter_sampling = 300)
  lp <- posterior_linpred(fit, dpar = "dprime", ndraws = 100, seed = 1)
  expect_true(is.matrix(lp))
  expect_equal(nrow(lp), 100)
  expect_equal(ncol(lp), nrow(d))
})

test_that("posterior_linpred applies link transform (sigma post-exp)", {
  skip_if_no_cmdstan()
  d <- .fixture_univariate(sigma = 1.3)
  m <- broc(brf(item_resp | is_old ~ 1, sigma ~ 1, family = uvsd()), d,
            batch_likelihood = TRUE)
  fit <- fit_broc(m, backend = "stan", refresh = 0,
                  parallel_chains = 4, iter_warmup = 300, iter_sampling = 300)
  s <- posterior_linpred(fit, dpar = "sigma", ndraws = 50, seed = 1)
  # sigma should be positive (post-exp), not on the linear (log) scale
  expect_true(all(s > 0))
})

test_that("posterior_linpred summary = TRUE returns tidy data frame", {
  skip_if_no_cmdstan()
  d <- .fixture_univariate()
  m <- broc(brf(item_resp | is_old ~ 1, family = evsd()), d,
            batch_likelihood = TRUE)
  fit <- fit_broc(m, backend = "stan", refresh = 0,
                  parallel_chains = 4, iter_warmup = 300, iter_sampling = 300)
  df <- posterior_linpred(fit, dpar = "dprime", summary = TRUE,
                               ndraws = 100, seed = 1)
  expect_s3_class(df, "data.frame")
  expect_named(df, c("trial", "mean", "lower", "upper"))
  expect_equal(nrow(df), nrow(d))
})

test_that("posterior_epred returns expected category in [1, K]", {
  skip_if_no_cmdstan()
  d <- .fixture_univariate()
  m <- broc(brf(item_resp | is_old ~ 1, family = evsd()), d,
            batch_likelihood = TRUE)
  fit <- fit_broc(m, backend = "stan", refresh = 0,
                  parallel_chains = 4, iter_warmup = 300, iter_sampling = 300)
  ep <- posterior_epred(fit, ndraws = 50, seed = 1)
  expect_true(all(ep >= 1 & ep <= 6))
})

test_that("predict(seed = ...) is reproducible", {
  skip_if_no_cmdstan()
  d <- .fixture_univariate()
  m <- broc(brf(item_resp | is_old ~ 1, family = evsd()), d,
            batch_likelihood = TRUE)
  fit <- fit_broc(m, backend = "stan", refresh = 0,
                  parallel_chains = 4, iter_warmup = 300, iter_sampling = 300)
  p1 <- predict(fit, summary = FALSE, ndraws = 50, seed = 7)
  p2 <- predict(fit, summary = FALSE, ndraws = 50, seed = 7)
  expect_equal(p1, p2, tolerance = 0)
  p3 <- predict(fit, summary = FALSE, ndraws = 50, seed = 8)
  expect_false(isTRUE(all.equal(p1, p3, tolerance = 0)))
})

test_that("allow_new_levels = FALSE errors on new RE level, TRUE works", {
  skip_if_no_cmdstan()
  d <- .fixture_univariate()
  d$subj <- factor(rep(1:6, length.out = nrow(d)))
  m <- broc(brf(item_resp | is_old ~ 1 + (1|subj), family = evsd()), d,
            batch_likelihood = TRUE)
  fit <- fit_broc(m, backend = "stan", refresh = 0,
                  parallel_chains = 4, iter_warmup = 300, iter_sampling = 300)
  new_dat <- data.frame(item_resp = 1L, is_old = 1L,
                        subj = factor("z_new", levels = "z_new"),
                        cond = factor("A", levels = c("A", "B")))
  expect_error(predict(fit, newdata = new_dat, summary = FALSE,
                            allow_new_levels = FALSE),
               "allow_new_levels")
  p <- predict(fit, newdata = new_dat, summary = FALSE,
                    allow_new_levels = TRUE, seed = 1)
  expect_true(is.array(p))
})
