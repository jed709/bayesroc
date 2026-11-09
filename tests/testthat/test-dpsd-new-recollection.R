.dpsd_rr_data <- function(seed = 4, n_per = 150, lam = 0.3, lam_new = 0.2, d = 1, n_sub = 6) {
  set.seed(seed)
  K <- 6L
  old <- rep(c(1L, 0L), each = n_per)
  y <- as.integer(cut(rnorm(2 * n_per, d * old), c(-Inf, -1.2, -0.4, 0.3, 1.0, 1.8, Inf)))
  rec <- stats::runif(2 * n_per) < ifelse(old == 1L, lam, lam_new)
  y[rec] <- ifelse(old[rec] == 1L, K, 1L)
  data.frame(conf = y, old = old, sub = factor(sample(seq_len(n_sub), 2 * n_per, TRUE)))
}

.dpsd_rr_cases <- function(d) {
  f <- function(link, ...) brf(conf | old ~ 1, criterion ~ 1, lambda ~ 1, ..., family = dpsd(link = link))
  list(
    probit_sigma = list(f(link_probit(), lambda_new ~ 1 + (1 | sub), sigma ~ 1), d),
    logit = list(f(link_logit(), lambda_new ~ 1), d),
    cloglog = list(f(link_cloglog(), lambda_new ~ 1 + (1 | sub)), d)
  )
}

test_that("lambda_new is a dpsd parameter for new items only, named lambda_new in the user-facing output", {
  d <- .dpsd_rr_data()
  m <- broc(brf(conf | old ~ 1, lambda ~ 1, lambda_new ~ 1, family = dpsd()), d)
  expect_true(m$model_data$has_lambda_L)
  expect_false(isTRUE(m$model_data$has_lure_mixture))
  pr <- get_broc_prior(m)
  expect_true("lambda_new" %in% pr$dpar)
  expect_false("lambda_L" %in% pr$dpar)
  m2 <- broc(brf(conf | old ~ 1, lambda ~ 1, lambda_new ~ 1, family = dpsd()), d,
             priors = broc_prior("normal(-1, 0.7)", class = "b", dpar = "lambda_new"))
  pr2 <- get_broc_prior(m2)
  expect_identical(pr2$prior[pr2$dpar == "lambda_new" & pr2$class == "b"], "normal(-1, 0.7)")
  expect_error(broc(brf(conf | old ~ 1, lambda ~ 1, lambda_L ~ 1, family = dpsd()), d), "lambda_new")
  m0 <- broc(brf(conf | old ~ 1, lambda ~ 1, family = dpsd()), d)
  expect_false(isTRUE(m0$model_data$has_lambda_L))
})

test_that("recollected new items give rating 1, and equal familiarity gives a linear ROC", {
  K <- 6L
  th <- array(rep(c(-1.2, -0.4, 0.3, 1.0, 1.8), each = 2), c(1, 2, K - 1L))
  for (link in c("probit", "logit", "cloglog")) {
    F <- switch(link, probit = stats::pnorm, logit = stats::plogis, cloglog = function(x) 1 - exp(-exp(x)))
    for (lam in list(c(0.3, 0.2), c(0.55, 0.05))) {
      p <- bayesroc:::probs_dpsdt(list(dprime = matrix(0, 1, 2), lambda = matrix(lam[1], 1, 2),
                                       lambda_L = matrix(lam[2], 1, 2)), th, c(1, 0), link, 1, 2, K, FALSE)
      cells <- diff(c(0, F(th[1, 2, ]), 1))
      expect_equal(p[1, 2, ], (1 - lam[2]) * cells + lam[2] * (seq_len(K) == 1L), tolerance = 1e-12)
      H <- rev(cumsum(rev(p[1, 1, ])))[-1]
      FA <- rev(cumsum(rev(p[1, 2, ])))[-1]
      expect_equal(H, lam[1] + (1 - lam[1]) / (1 - lam[2]) * FA, tolerance = 1e-12, label = paste(link, "ROC line"))
    }
  }
})

test_that("new-item recollection: JAX matches R", {
  skip_if_no_jax()
  for (nm in names(cs <- .dpsd_rr_cases(.dpsd_rr_data()))) {
    m <- suppressMessages(broc(cs[[nm]][[1]], cs[[nm]][[2]]))
    f <- suppressMessages(fit_broc(m, backend = "jax", chains = 1, iter_warmup = 100, iter_sampling = 20,
                                   seed = 1))
    ll_r <- .obs_loglik_r(m, build_data_source_from_model(m), f$draws(format = "matrix"), re_formula = NULL)
    ll_j <- .read_loglik_SN_cols(.loglik_path(f), 1L, m$model_data$N)
    new <- m$stan_data$is_old <= 0
    expect_lt(max(abs(ll_j[, new] - ll_r[, new])), 1e-5, label = paste(nm, "new rows"))
    expect_lt(max(abs(ll_j[, !new] - ll_r[, !new])), 1e-5, label = paste(nm, "old rows"))
  }
})

test_that("the ROC survival carries the new items' recollected mass below every criterion", {
  skip_if_no_jax()
  d <- .dpsd_rr_data()
  m <- suppressMessages(broc(brf(conf | old ~ 1, criterion ~ 1, lambda ~ 1, lambda_new ~ 1, sigma ~ 1,
                                 family = dpsd()), d))
  f <- suppressMessages(fit_broc(m, backend = "jax", chains = 1, iter_warmup = 150, iter_sampling = 40,
                                 seed = 3))
  setup <- bayesroc:::.plot_setup(f, m, 20, NULL)
  w <- bayesroc:::.row_weights(m)
  grid <- seq(-3, 3, length.out = 13)
  tp <- setup$tp
  for (rows in list(which(d$old == 0L), which(d$old == 1L))) {
    sv <- bayesroc:::.roc_survival(m, setup, rows, w, 1L, grid, mode = "grid")
    old <- d$old[rows] == 1L
    ref <- vapply(grid, function(t) {
      s <- if (all(old)) {
        tp$lambda[, rows] + (1 - tp$lambda[, rows]) * (1 - stats::pnorm((t - tp$dprime[, rows]) / tp$sigma[, rows]))
      } else {
        (1 - tp$lambda_L[, rows]) * (1 - stats::pnorm(t))
      }
      as.vector(s %*% w[rows]) / sum(w[rows])
    }, numeric(nrow(tp$dprime)))
    expect_equal(unname(sv), unname(ref), tolerance = 1e-10)
  }
  expect_s3_class(suppressMessages(plot_roc(f, ndraws = 20)), "ggplot")
  expect_s3_class(suppressMessages(plot_distributions(f, ndraws = 20)), "ggplot")
  pr <- predict(f, ndraws = 20)
  expect_equal(nrow(pr), nrow(d) * 6L)
})
