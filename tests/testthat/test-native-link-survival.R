test_that("native survival kernels use the model's link", {
  skip_if_not(.bvn_native_ready())
  set.seed(4)
  S <- 3L; R <- 5L; J <- 4L
  th <- array(rep(c(-1, -0.2, 0.5, 1.3), each = S * R) + rnorm(S * R * J, 0, 0.1), c(S, R, J))
  mu <- matrix(rnorm(S * R, 0.8, 0.3), S, R)
  sig <- matrix(exp(rnorm(S * R, 0, 0.2)), S, R)
  lam <- matrix(runif(S * R, 0.1, 0.4), S, R)
  mu2 <- matrix(rnorm(S * R, 1.5, 0.3), S, R)
  sig2 <- matrix(exp(rnorm(S * R, 0, 0.2)), S, R)
  w <- runif(R, 0.5, 2)
  u <- seq(-2, 2, length.out = 7)
  for (lk in c("probit", "logit", "cloglog")) {
    F <- function(z) link_cdf_r(z, lk)
    code <- .native_link_code(lk)
    for (kind in c(0L, 1L, 2L, 4L)) {
      acc <- .C(.bvn_native$sym_surv, as.double(th), as.double(mu), as.double(sig), as.double(lam), as.double(mu2),
                as.double(u), as.double(w), as.integer(c(S, R, J, length(u), kind, 0L, 1L, code)), 8.0,
                acc = double(S * length(u)), NAOK = TRUE)$acc
      hand <- sapply(u, function(c) {
        f <- F((c - mu) / sig)
        f <- if (kind == 1) (1 - lam) * f else if (kind == 2) (1 - lam) * f + lam * F((c - mu2) / sig)
             else if (kind == 4) (1 - lam) * f + lam * (1 - mu2) else f
        as.vector((1 - f) %*% w)
      })
      expect_equal(matrix(acc, S), hand, tolerance = 1e-12, info = paste(lk, "kind", kind))
    }
    acc <- .C(.bvn_native$sym_mixsurv, as.double(th), as.double(mu), as.double(sig), as.double(lam), as.double(mu2),
              as.double(sig2), as.double(u), as.double(w), as.integer(c(S, R, J, length(u), 0L, 1L, code)), 8.0,
              acc = double(S * length(u)), NAOK = TRUE)$acc
    hand <- sapply(u, function(c) as.vector((1 - (1 - lam) * F((c - mu) / sig) - lam * F((c - mu2) / sig2)) %*% w))
    expect_equal(matrix(acc, S), hand, tolerance = 1e-12, info = paste(lk, "mixture"))
  }
  expect_identical(.native_link_code("cloglog", "cdp"), 0L)
  expect_true(is.na(.native_link_code("cauchit")))
})
