test_that("threads_per_chain builds the reduce_sum variant at fit time", {
  d <- .fixture_crossed()
  m <- suppressMessages(broc(brf(item_resp | is_old ~ 1 + (1 | sub), family = evsd()), d))
  expect_false(grepl("reduce_sum", .with_stan_code(m)$stan_code, fixed = TRUE))
  mt <- .with_threads(m)
  expect_true(grepl("reduce_sum", .with_stan_code(mt)$stan_code, fixed = TRUE))
  expect_gte(mt$stan_data$grainsize, 1L)
  dc <- d
  mc <- .with_threads(suppressMessages(broc(brf(item_resp ~ 1, cutpoints ~ 1 + (1 | sub), family = cumulative()), dc)))
  expect_true(grepl("reduce_sum", .with_stan_code(mc)$stan_code, fixed = TRUE))
  expect_false(grepl("criterion", .with_stan_code(mc)$stan_code, fixed = TRUE))
  skip_if_no_cmdstan()
  f <- suppressMessages(stan_fit_broc(m, chains = 1, iter_warmup = 20,
                                 iter_sampling = 10, seed = 1, refresh = 50, threads_per_chain = 2))
  expect_s3_class(f, "broc_fit")
  expect_true(grepl("reduce_sum", attr(f, "broc_model")$stan_code, fixed = TRUE))
})

test_that("keep_re drops raw random-effect draws on the Stan backend, including when saved", {
  skip_if_no_cmdstan()
  d <- .fixture_crossed()
  m <- suppressMessages(broc(brf(item_resp | is_old ~ 1 + (1 | sub), family = evsd()), d))
  for (kr in c(FALSE, TRUE)) {
    f <- suppressMessages(stan_fit_broc(m, chains = 1, iter_warmup = 30,
                                   iter_sampling = 10, seed = 1, refresh = 50, keep_re = kr))
    v <- posterior::variables(f$draws())
    expect_identical(any(grepl("^u_dprime_sub_raw", v)), kr)
    expect_true(any(grepl("^u_dprime_sub\\[", v)))
    tf <- tempfile(fileext = ".rds")
    save_broc_fit(f, tf, verbose = FALSE)
    g <- load_broc_fit(tf)
    expect_identical(any(grepl("^u_dprime_sub_raw", posterior::variables(g$draws()))), kr)
    expect_identical(g$.keep_re, kr)
  }
})

test_that("seeded prior inits are reproducible and leave the caller's RNG stream alone", {
  m <- suppressMessages(broc(brf(item_resp | is_old ~ 1, criterion ~ 1, family = evsd()), .fixture_univariate()))
  set.seed(99)
  a <- prior_init(m, chains = 2, seed = 1)
  x <- runif(1)
  set.seed(99)
  b <- prior_init(m, chains = 2, seed = 1)
  expect_identical(a, b)
  expect_identical(runif(1), x)
  set.seed(99)
  expect_identical(runif(1), x)
})
