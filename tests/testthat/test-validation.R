# Tests for the up-front validation we added in broc().

test_that("broc() errors when data is NULL or wrong class", {
  expect_error(broc(item_resp | is_old ~ 1, family = evsd()),
               "data.*required")
  expect_error(broc(item_resp | is_old ~ 1, data = list(a = 1), family = evsd()),
               "data.*data.frame")
  expect_error(broc(item_resp | is_old ~ 1,
                    data = data.frame(item_resp = integer(0), is_old = integer(0)),
                    family = evsd()),
               "0 rows")
})

test_that("broc() errors with actionable message when formula references missing column", {
  d <- .fixture_univariate()
  err <- expect_error(
    broc(item_resp | is_old ~ typo_var, data = d, family = evsd()),
    "not in"
  )
  msg <- conditionMessage(err)
  expect_match(msg, "typo_var")
  expect_match(msg, "primary formula")
})

test_that("broc() does not flag cross-correlation IDs as missing columns", {
  d <- .fixture_univariate()
  d$subj <- factor(rep(1:6, length.out = nrow(d)))
  d$item <- factor(rep(1:10, length.out = nrow(d)))
  # `|p|` and `|w|` are brms-style cross-parameter correlation IDs, not data
  # columns. Should NOT be reported as missing. The subj / item grouping
  # factors *are* real columns and should pass.
  expect_no_error(
    broc(brf(item_resp | is_old ~ 1 + (1 |p| subj) + (1 |w| item),
             sigma ~ 1 + (1 |p| subj) + (1 |w| item),
             family = uvsd()),
         d)
  )
  # But a genuine typo on the grouping factor should still be caught
  err <- expect_error(
    broc(brf(item_resp | is_old ~ 1 + (1 |p| subjj),
             family = uvsd()),
         d),
    "subjj"
  )
})

test_that("broc() catches missing variable in secondary formula", {
  d <- .fixture_univariate()
  err <- expect_error(
    broc(brf(item_resp | is_old ~ 1, sigma ~ does_not_exist, family = uvsd()),
         d),
    "does_not_exist"
  )
})

test_that("broc() errors when counts column missing", {
  d <- .fixture_univariate()
  expect_error(
    broc(brf(item_resp | is_old ~ 1, family = evsd()),
         d, counts = "no_such_col"),
    "counts.*not a column"
  )
})

test_that("broc() accepts 0/1 and -0.5/0.5 is_old coding, errors otherwise", {
  d <- .fixture_univariate()
  expect_s3_class(broc(brf(item_resp | is_old ~ 1, family = evsd()), d), "broc_model")

  dc <- d; dc$is_old <- ifelse(d$is_old == 1, 0.5, -0.5)   # centered
  expect_s3_class(broc(brf(item_resp | is_old ~ 1, family = evsd()), dc), "broc_model")

  d12 <- d; d12$is_old <- d$is_old + 1L                    # 1/2 -- invalid
  expect_error(broc(brf(item_resp | is_old ~ 1, family = evsd()), d12),
               "0/1.*or.*-0\\.5/0\\.5|coded 0/1")

  dm <- d; dm$is_old <- rep(c(0, 1, 0.5, -0.5), length.out = nrow(d))  # mixed
  expect_error(broc(brf(item_resp | is_old ~ 1, family = evsd()), dm),
               "coded 0/1")

  # Factor is_old (e.g. from table()) is read by its labels, not factor codes.
  df <- d; df$is_old <- factor(d$is_old, levels = c(0, 1))
  m <- broc(brf(item_resp | is_old ~ 1, family = evsd()), df)
  expect_setequal(sort(unique(m$model_data$stan_data$is_old)), c(0, 1))
})

test_that("centered is_old + encoding_vars recodes consistently (no all-zero d')", {
  # Regression: centered coding (-0.5/0.5) with encoding factors is recoded to
  # 0/1. The recode must reach the design-matrix builders (which detect target
  # rows for encoding()) AND the stored data frame -- otherwise encoding()
  # silently no-ops, the d' design matrix is all-zero, and the lure-only level
  # survives as an unidentified nuisance.
  d_ctr <- .fixture_encoding(old_code = 0.5, new_code = -0.5)
  d_01  <- .fixture_encoding(old_code = 1,   new_code = 0)
  f <- brf(item_resp | is_old ~ condition, sigma ~ 1)

  m_ctr <- suppressWarnings(
    broc(f, data = d_ctr, family = uvsd(), encoding_vars = "condition",
         batch_likelihood = FALSE))
  m_01 <- broc(f, data = d_01, family = uvsd(), encoding_vars = "condition",
               batch_likelihood = FALSE)

  fx <- m_ctr$model_data$dprime_fixed
  # Lure-only level dropped, design matrix not collapsed to zero.
  expect_false(any(grepl("new", fx$coef_names)))
  expect_false(all(fx$X == 0))
  # Recode reaches both the likelihood input and the stored data frame.
  expect_setequal(unique(m_ctr$stan_data$is_old), c(0, 1))
  expect_setequal(unique(m_ctr$data$is_old), c(0, 1))
  # Centered build is now identical to the treatment-coded build.
  expect_identical(fx$coef_names, m_01$model_data$dprime_fixed$coef_names)
  expect_equal(fx$X, m_01$model_data$dprime_fixed$X)
})

test_that("broc() errors when encoding_vars column missing", {
  d <- .fixture_univariate()
  expect_error(
    broc(brf(item_resp | is_old ~ 1, family = evsd(),
             encoding_vars = "missing_factor"),
         d),
    "missing_factor"
  )
})

test_that("broc() rejects factor/character is_old with non-numeric labels", {
  d <- .fixture_univariate()
  d$is_old <- factor(ifelse(d$is_old == 1, "old", "new"), levels = c("new", "old"))
  expect_error(
    broc(brf(item_resp | is_old ~ 1, criterion ~ 1, family = evsd()), d),
    "coerced to 0/1|coded 0/1"
  )
  # A factor whose LABELS are numeric (0/1) is still accepted (coerced by label).
  d2 <- .fixture_univariate()
  d2$is_old <- factor(d2$is_old, levels = c(0, 1))
  expect_no_error(
    broc(brf(item_resp | is_old ~ 1, criterion ~ 1, family = evsd()), d2)
  )
})
