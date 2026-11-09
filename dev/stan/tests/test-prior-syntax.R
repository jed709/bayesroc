# Helpers shared with tests/testthat/
# test-prior-syntax.R (copied).

.syntax_data <- function() {
  set.seed(1)
  d <- expand.grid(participant = factor(1:6), item = factor(1:10))
  d$old <- as.integer(ave(runif(nrow(d)), d$participant, FUN = function(u) rank(u) <= 5))
  d$conf <- sample(1:6, nrow(d), TRUE)
  d
}

test_that("Stan-only distributions are refused by the JAX backend and by prior inits", {
  d <- .syntax_data()
  f <- brf(conf | old ~ 1 + (1 | participant), criterion ~ 1, sigma ~ 1, family = uvsd())
  m <- suppressMessages(broc(f, d, priors = broc_prior("frechet(2, 1)", class = "sd", dpar = "dprime")))
  expect_true(any(grepl("~ frechet\\(2, 1\\);$", trimws(strsplit(.with_stan_code(m)$stan_code, "\n")[[1]]))))
  expect_error(build_numpyro_config(m), "frechet\\(\\) is only available with backend = \"stan\"")
  expect_error(prior_init(m, chains = 1, seed = 1), "cannot draw starting values from frechet")
})

test_that("prior inits draw from every supported distribution", {
  set.seed(3)
  n <- 40000
  draw <- function(p, cls = "b") .sample_univariate(to_stan_prior(broc_prior(p, class = cls)$prior), n)
  g <- 0.5772156649015329
  checks <- list(
    list("normal(1, 2)", mean = 1, sd = 2),
    list("std_normal()", mean = 0, sd = 1),
    list("student_t(5, 1, 2)", mean = 1, sd = 2 * sqrt(5 / 3)),
    list("logistic(1, 2)", mean = 1, sd = 2 * pi / sqrt(3)),
    list("gumbel(1, 2)", mean = 1 + 2 * g, sd = 2 * pi / sqrt(6)),
    list("double_exponential(1, 2)", mean = 1, sd = 2 * sqrt(2)),
    list("lognormal(0, 0.5)", mean = exp(0.125), sd = sqrt((exp(0.25) - 1) * exp(0.25))),
    list("exponential(2)", mean = 0.5, sd = 0.5),
    list("gamma(3, 2)", mean = 1.5, sd = sqrt(3) / 2),
    list("inv_gamma(10, 9)", mean = 1, sd = sqrt(1 / 8)),
    list("weibull(2, 1)", mean = gamma(1.5), sd = sqrt(1 - gamma(1.5)^2)),
    list("beta(2, 3)", mean = 0.4, sd = 0.2),
    list("uniform(-1, 3)", mean = 1, sd = 4 / sqrt(12))
  )
  for (ck in checks) {
    z <- draw(ck[[1]])
    expect_lt(abs(mean(z) - ck$mean), 4 * ck$sd / sqrt(n), label = paste(ck[[1]], "mean"))
    expect_lt(abs(sd(z) / ck$sd - 1), 0.03, label = paste(ck[[1]], "sd"))
  }
  expect_lt(abs(median(abs(draw("half_cauchy(2)", "sd"))) - 2), 0.1)
  expect_lt(abs(median(abs(draw("half_student_t(5, 1)", "sd"))) - qt(0.75, 5)), 0.03)
  expect_lt(abs(mean(abs(draw("half_normal(2)", "sd"))) - 2 * sqrt(2 / pi)), 0.03)
})
