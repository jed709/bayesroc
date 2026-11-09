.syntax_data <- function() {
  set.seed(1)
  d <- expand.grid(participant = factor(1:6), item = factor(1:10))
  d$old <- as.integer(ave(runif(nrow(d)), d$participant, FUN = function(u) rank(u) <= 5))
  d$conf <- sample(1:6, nrow(d), TRUE)
  d
}

test_that("prior strings are rewritten in one standard form", {
  cases <- c(
    "normal(0, 1)" = "normal(0, 1)",
    "normal(0,1)" = "normal(0, 1)",
    "   normal( 0 , 1 )   " = "normal(0, 1)",
    "normal (0, 1)" = "normal(0, 1)",
    "NORMAL(0, 1)" = "normal(0, 1)",
    "normal(0, 1);" = "normal(0, 1)",
    "normal(0,\n1)" = "normal(0, 1)",
    "normal(0, 1)" = "normal(0, 1)",
    "'normal(0, 1)'" = "normal(0, 1)",
    "\"normal(0, 1)\"" = "normal(0, 1)",
    "normal(-.5, 1e-1)" = "normal(-0.5, 0.1)",
    "normal(sigma = 2, mu = 0)" = "normal(0, 2)",
    "normal(0, sqrt(4))" = "normal(0, 2)",
    "normal(0, 1/2)" = "normal(0, 0.5)",
    "student_t(df = 3, 0, 2.5)" = "student_t(3, 0, 2.5)",
    "gamma(shape = 2, rate = 1)" = "gamma(2, 1)",
    "laplace(0, 1)" = "double_exponential(0, 1)",
    "std_normal" = "std_normal()"
  )
  for (x in names(cases)) expect_identical(broc_prior(x, class = "b")$prior, cases[[x]], label = encodeString(x))
  expect_identical(broc_prior("half_normal(sd = 0.5)", class = "sd")$prior, "half_normal(0.5)")
  expect_identical(broc_prior("LKJ(2)", class = "cor")$prior, "lkj_corr_cholesky(2)")
  expect_identical(broc_prior("lkj_corr(4)", class = "cor")$prior, "lkj_corr_cholesky(4)")
})

test_that("malformed or unsupported priors are errors when the prior is made", {
  expect_error(broc_prior("N(0, 1)", class = "b"), "Did you mean 'normal'")
  expect_error(broc_prior("normal(0, -1)", class = "b"), "must be positive")
  expect_error(broc_prior("gamma(0, 1)", class = "sd"), "must be positive")
  expect_error(broc_prior("normal(0)", class = "b"), "takes 2 arguments")
  expect_error(broc_prior("normal(0, 1, 2)", class = "b"), "takes 2 arguments")
  expect_error(broc_prior("normal(0; 1)", class = "b"), "separate them with commas")
  expect_error(broc_prior("normal(0, 1) + normal(0, 2)", class = "b"), "single distribution")
  expect_error(broc_prior("normal(0, system('echo'))", class = "b"), "constant expression")
  expect_error(broc_prior("normal(m = 0, 1)", class = "b"), "unknown argument name")
  expect_error(broc_prior("uniform(1, 0)", class = "b"), "lower < upper")
  expect_error(broc_prior("", class = "b"), "empty")
  expect_error(broc_prior("half_normal(1)", class = "b"), "standard deviations")
  expect_error(broc_prior("lkj(2)", class = "sd"), "correlation prior")
  expect_error(broc_prior("normal(0, 1)", class = "cor"), "must be lkj")
})

test_that("a standardised prior reaches the JAX config unchanged", {
  d <- .syntax_data()
  f <- brf(conf | old ~ 1 + (1 | p | participant), criterion ~ 1, sigma ~ 1 + (1 | p | participant), family = uvsd())
  pr <- c(broc_prior(" Normal( mu = 1 , sigma = sqrt(4) ); ", class = "b", dpar = "dprime"),
          broc_prior("'half_normal(1/2)'", class = "sd", dpar = "dprime"),
          broc_prior("LKJ(3)", class = "cor"))
  m <- suppressMessages(broc(f, d, priors = pr))
  s <- unlist(build_numpyro_config(m), use.names = FALSE)
  expect_true(all(c("normal(1, 2)", "half_normal(0.5)", "lkj_corr_cholesky(3)") %in% s))
})


test_that("JAX SD priors are their distribution restricted to positive values, as in Stan", {
  skip_if_no_jax()
  ps <- c("normal(0, 1)", "normal(1, 1)", "student_t(3, 1, 2)", "half_student_t(3, 2.5)", "logistic(0, 1)",
          "gumbel(0, 1)", "gumbel(1, 0.5)", "double_exponential(1, 1)", "inv_gamma(2, 1)", "weibull(2, 1)",
          "beta(2, 2)", "uniform(0, 2)", "uniform(-1, 2)", "lognormal(0, 1)", "gamma(2, 2)", "exponential(2)")
  out <- system2(jax_python(), c(test_path("python", "sd_prior_check.py"), dirname(find_backend_script()),
                                 .bvn_ffi_lib(), shQuote(paste(ps, collapse = ";"))), stdout = TRUE, stderr = FALSE)
  res <- jsonlite::fromJSON(sub("^RESULT ", "", grep("^RESULT ", out, value = TRUE)))
  for (p in ps) {
    r <- res[[p]]
    expect_false(r$nan, label = p)
    expect_lt(abs(r$mass - 1), 2e-3, label = paste(p, "mass on (0, Inf)"))
    expect_lt(abs(r$shape), 1e-8, label = paste(p, "same shape as the untruncated density"))
    expect_true(r$positive, label = p)
  }
})

test_that("messages about unused priors use the dpar spelling the user gave", {
  d <- .fixture_bivariate_cumulative()
  expect_message(broc(brf(resp(y1, y2) ~ 1, cutpoints ~ 1, cutpoints2 ~ 1, rho ~ 1, family = bivariate_cumulative()), d,
                      priors = broc_prior("normal(0, 2)", class = "b", dpar = "cutpoints2")),
                 "dpar = \"cutpoints2\"", fixed = TRUE)
})

test_that("distributions only Stan provided are refused when the prior is made", {
  expect_error(broc_prior("skew_normal(0, 1, 2)", class = "b"), "skew_normal() is not supported", fixed = TRUE)
  expect_error(broc_prior("von_mises(0, 1)", class = "b"), "not supported")
})
