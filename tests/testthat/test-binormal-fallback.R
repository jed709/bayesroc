test_that("the R bivariate normal CDF keeps relative accuracy on tiny cells without the native kernel", {
  withr::local_options(bayesroc.native_predict = FALSE)
  .bvn_native$ok <- NULL
  withr::defer(.bvn_native$ok <- NULL)
  ref <- function(z1, z2, rho)
    integrate(function(x) dnorm(x) * pnorm((z2 - rho * x) / sqrt(1 - rho^2)), -Inf, z1,
              rel.tol = 1e-13, abs.tol = 0, subdivisions = 2000L)$value
  z1 <- c(-6, -5, -4.5, -8, 2, -3, 0.5)
  z2 <- c(-5, 3, -4, 1, -7, -3.5, 0.2)
  rho <- c(0.3, -0.9, -0.6, 0.95, 0.5, -0.95, 0.1)
  p <- binormal_cdf_r(z1, z2, rho)
  r <- mapply(ref, z1, z2, rho)
  expect_true(all(p > 0))
  expect_lt(max(abs(p - r) / r), 1e-9)
})
