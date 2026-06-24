# Predict/likelihood consistency ("bridge") tests.
#
# The per-observation log-likelihood saved by each backend uses the validated
# likelihood code; predict()'s per-family probs_*() are an independent
# reimplementation of the same cell math. For the OBSERVED response of every
# trial/draw the predicted cell probability must equal exp(log_lik). This is a
# strong, family-agnostic check that anchors predict() to the likelihood and
# guards against the bivariate_dp / bounded predict bugs fixed in this package.
#
# Each test also checks:
#   * sum-to-1 of the per-cell probabilities, and
#   * epred == mean(posterior_predict) (grand mean) -- a cross-check of the
#     sampling path (predict type="prediction") against the cell-prob path.

.bridge_iter   <- if (identical(Sys.getenv("BAYESROC_TEST_FAST", "true"), "true")) 250 else 1000
.bridge_chains <- 2

.fit_bridge <- function(m) {
  fit_broc(m, backend = "stan", refresh = 0, parallel_chains = .bridge_chains,
           chains = .bridge_chains, iter_warmup = .bridge_iter,
           iter_sampling = .bridge_iter)
}

# bridge + sum-to-1 + epred==mean(predict), for univariate or bivariate fits.
.predict_checks <- function(fit) {
  model <- attr(fit, "broc_model")
  sd <- model$stan_data
  ll <- as.matrix(fit$draws(variables = "log_lik", format = "matrix"))  # S x N
  pr <- predict(fit, type = "response", summary = FALSE, cores = 2)
  S <- nrow(ll); N <- ncol(ll)
  bivar <- length(dim(pr)) == 4L
  if (bivar) {
    pred_obs <- vapply(seq_len(N), function(n) pr[, n, sd$y[n], sd$y2[n]], numeric(S))
  } else {
    pred_obs <- vapply(seq_len(N), function(n) pr[, n, sd$y[n]], numeric(S))
  }
  ep <- posterior_epred(fit)
  pp <- predict(fit, type = "prediction", summary = FALSE, cores = 2)
  epd <- if (bivar) {
    max(abs(mean(ep$y1) - mean(pp$y1)), abs(mean(ep$y2) - mean(pp$y2)))
  } else {
    abs(mean(ep) - mean(pp))
  }
  list(bridge = max(abs(pred_obs - exp(ll))),
       sum1   = max(abs(apply(pr, c(1, 2), sum) - 1)),
       epred_pp = epd)
}

.expect_predict_ok <- function(fit) {
  ck <- .predict_checks(fit)
  expect_lt(ck$bridge, 1e-4)
  expect_lt(ck$sum1, 1e-6)
  expect_lt(ck$epred_pp, 0.1)   # grand-mean MC tolerance
}

.biv_f <- function(...) {
  brf(resp(y1, y2) | item_type ~ 1, discrim ~ 1, criterion ~ 1, criterion2 ~ 1,
      rho ~ 1, ...)
}

# Small univariate fixture (recognition ratings with subject intercepts).
.fixture_uvsd <- function(seed = 7, n_subj = 6, n_per = 80) {
  set.seed(seed)
  K <- 6; thr <- c(-1.2, -0.5, 0.1, 0.8, 1.6)
  subj <- rep(seq_len(n_subj), each = n_per)
  u <- rnorm(n_subj, 0, 0.4)[subj]
  is_old <- rbinom(length(subj), 1, 0.5)
  z <- ifelse(is_old == 1, rnorm(length(subj), 1.0 + u, 1.25), rnorm(length(subj), u, 1))
  resp <- as.integer(cut(z, c(-Inf, thr, Inf)))
  data.frame(resp = resp, cond = is_old, sub = subj)
}

test_that("bridge: uvsd + RE, incl. epred and allow_new_levels", {
  skip_if_no_cmdstan()
  d <- .fixture_uvsd()
  m <- broc(brf(resp | cond ~ 1 + (1 | sub), criterion ~ 1, sigma ~ 1,
                family = uvsd()), d, batch_likelihood = TRUE)
  fit <- .fit_bridge(m)
  .expect_predict_ok(fit)
  # allow_new_levels: predict on a novel subject runs and yields valid probs.
  nd <- transform(d[1:20, ], sub = 9999L)
  prn <- predict(fit, newdata = nd, type = "response", summary = FALSE,
                 allow_new_levels = TRUE, seed = 1)
  expect_true(all(is.finite(prn)))
  expect_lt(max(abs(apply(prn, c(1, 2), sum) - 1)), 1e-6)
})

test_that("bridge: bivariate_gaussian (bounded) -- guards bounded sign flip", {
  skip_if_no_cmdstan()
  d <- .fixture_bivariate_sdt()
  m <- broc(.biv_f(family = bivariate_gaussian(bounded = TRUE)), d,
            batch_likelihood = TRUE)
  .expect_predict_ok(.fit_bridge(m))
})

test_that("bridge: bivariate_dp (unbounded) -- guards corner-cell swap", {
  skip_if_no_cmdstan()
  d <- .fixture_bivariate_sdt()
  m <- broc(.biv_f(lambda ~ 1, lambda2 ~ 1, family = bivariate_dp()), d,
            batch_likelihood = TRUE)
  .expect_predict_ok(.fit_bridge(m))
})

test_that("bridge: bivariate_dp (bounded) -- guards bounded p1/p2 + sign flip", {
  skip_if_no_cmdstan()
  d <- .fixture_bivariate_sdt()
  m <- broc(.biv_f(lambda ~ 1, lambda2 ~ 1, family = bivariate_dp(bounded = TRUE)),
            d, batch_likelihood = TRUE)
  .expect_predict_ok(.fit_bridge(m))
})

test_that("bridge: cdp joint (rating x R/K reconstructed from conditionals)", {
  skip_if_no_cmdstan()
  d <- .fixture_cdp()
  m <- broc(brf(resp(rating, rk) | is_old ~ condition - 1,
                fam ~ condition - 1, criterion ~ condition - 1,
                rec_crit ~ condition - 1, sigma_R ~ condition - 1,
                family = cdp(old_levels = 2:6)), d, counts = "count",
            batch_likelihood = TRUE)
  fit <- .fit_bridge(m)
  sd <- m$stan_data
  ll <- as.matrix(fit$draws(variables = "log_lik", format = "matrix"))
  pr <- predict(fit, type = "response", summary = FALSE, cores = 2)
  pR <- attr(pr, "p_remember"); pK <- attr(pr, "p_know")
  old_levels <- m$family$old_levels
  n_rkg <- if (!is.null(m$family$n_rkg)) m$family$n_rkg else 2L
  S <- nrow(ll); N <- ncol(ll); y <- sd$y; rk <- sd$rk
  joint <- matrix(0, S, N)
  for (n in seq_len(N)) {
    yn <- y[n]; rkn <- rk[n]; marg <- pr[, n, yn]
    if (yn %in% old_levels) {
      cond <- if (rkn == 1L) pR[, n, yn]
              else if (rkn == 2L) {
                if (n_rkg == 3 && !is.null(pK)) pK[, n, yn] else 1 - pR[, n, yn]
              } else {
                pmax(1 - pR[, n, yn] - if (!is.null(pK)) pK[, n, yn] else 0, 0)
              }
      joint[, n] <- marg * cond
    } else {
      joint[, n] <- marg
    }
  }
  expect_lt(max(abs(joint - exp(ll))), 1e-4)
})
