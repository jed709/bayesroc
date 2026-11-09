


test_that("class-wide user priors override the per-parameter defaults and nested c() flattens", {
  d <- .fixture_crossed()
  f <- brf(item_resp | is_old ~ 1 + (1 | sub), criterion ~ 1 + (1 | sub), sigma ~ 1 + (1 | sub),
           family = uvsd())
  pr <- c(c(broc_prior("normal(0, 2)", class = "b"), broc_prior("student_t(3, 0, 2.5)", class = "sd")),
          broc_prior("normal(0, 1)", class = "b", dpar = "sigma"),
          broc_prior("normal(0, 0.2)", class = "sd", dpar = "sigma"))
  expect_length(pr, 4L)
  g <- get_broc_prior(f, data = d, priors = pr)
  b <- g[g$class == "b", ]
  expect_true(all(b$prior[b$dpar == "sigma"] == "normal(0, 1)"))
  expect_true(all(b$prior[b$dpar != "sigma"] == "normal(0, 2)"))
  s <- g[g$class == "sd", ]
  expect_true(all(s$prior[s$dpar == "sigma"] == "normal(0, 0.2)"))
  expect_true(all(s$prior[s$dpar != "sigma"] == "student_t(3, 0, 2.5)"))
  expect_error(c(broc_prior("normal(0, 1)", class = "b"), list(1)), "broc_prior")
})

