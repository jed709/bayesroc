
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
  expect_no_error(
    broc(brf(item_resp | is_old ~ 1 + (1 |p| subj) + (1 |w| item),
             sigma ~ 1 + (1 |p| subj) + (1 |w| item),
             family = uvsd()),
         d)
  )
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

  dc <- d; dc$is_old <- ifelse(d$is_old == 1, 0.5, -0.5)
  expect_s3_class(broc(brf(item_resp | is_old ~ 1, family = evsd()), dc), "broc_model")

  d12 <- d; d12$is_old <- d$is_old + 1L
  expect_error(broc(brf(item_resp | is_old ~ 1, family = evsd()), d12),
               "0/1.*or.*-0\\.5/0\\.5|coded 0/1")

  dm <- d; dm$is_old <- rep(c(0, 1, 0.5, -0.5), length.out = nrow(d))
  expect_error(broc(brf(item_resp | is_old ~ 1, family = evsd()), dm),
               "coded 0/1")

  df <- d; df$is_old <- factor(d$is_old, levels = c(0, 1))
  m <- broc(brf(item_resp | is_old ~ 1, family = evsd()), df)
  expect_setequal(sort(unique(m$model_data$stan_data$is_old)), c(0, 1))
})

test_that("centered is_old + encoding_vars recodes consistently (no all-zero d')", {
  d_ctr <- .fixture_encoding(old_code = 0.5, new_code = -0.5)
  d_01  <- .fixture_encoding(old_code = 1,   new_code = 0)
  f <- brf(item_resp | is_old ~ condition, sigma ~ 1)

  m_ctr <- suppressWarnings(
    broc(f, data = d_ctr, family = uvsd(), encoding_vars = "condition"))
  m_01 <- broc(f, data = d_01, family = uvsd(), encoding_vars = "condition")

  fx <- m_ctr$model_data$dprime_fixed
  expect_false(any(grepl("new", fx$coef_names)))
  expect_false(all(fx$X == 0))
  expect_setequal(unique(m_ctr$stan_data$is_old), c(0, 1))
  expect_setequal(unique(m_ctr$data$is_old), c(0, 1))
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
    "text labels|coded 0/1"
  )
  d2 <- .fixture_univariate()
  d2$is_old <- factor(d2$is_old, levels = c(0, 1))
  expect_no_error(
    broc(brf(item_resp | is_old ~ 1, criterion ~ 1, family = evsd()), d2)
  )
})

test_that("broc() drops zero-count rows", {
  d <- .fixture_univariate()
  agg <- aggregate(list(n = rep(1L, nrow(d))),
                   by = list(item_resp = d$item_resp, is_old = d$is_old),
                   FUN = sum)
  agg$n[1] <- 0L
  expect_message(
    m <- broc(brf(item_resp | is_old ~ 1, family = evsd()), agg, counts = "n"),
    "Dropping 1 row"
  )
  expect_equal(m$stan_data$N, nrow(agg) - 1L)
  expect_true(all(m$stan_data$counts > 0))
})

test_that("broc() errors when a formula names a parameter the family lacks", {
  d <- .fixture_univariate()
  expect_error(
    broc(brf(item_resp | is_old ~ 1, lambda ~ 1, family = uvsd()), d),
    "lambda"
  )
})

test_that("encoding factor with one surviving level does not crash", {
  d <- .fixture_encoding()
  d <- d[d$condition %in% c("new", "speak"), , drop = FALSE]
  d$condition <- droplevels(d$condition)
  expect_no_error(
    broc(brf(item_resp | is_old ~ condition, family = evsd(),
             encoding_vars = "condition"), d)
  )
})

test_that("cumulative default slope prior is centred at 0", {
  d <- .fixture_univariate()
  m <- broc(brf(item_resp ~ cond, cutpoints ~ 1, family = cumulative()), d)
  expect_identical(m$model_data$prior_lookup$b[["dprime"]], "normal(0, 1)")

  d2 <- .fixture_univariate()
  m2 <- broc(brf(item_resp | is_old ~ 1, family = evsd()), d2)
  expect_identical(m2$model_data$prior_lookup$b[["dprime"]], "normal(1, 1)")
})

test_that("an encoding variable in the criterion formula warns", {
  d <- .fixture_encoding(n_per = 60)
  d$item <- factor(rep(seq_len(12), length.out = nrow(d)))
  expect_warning(suppressMessages(broc(brf(item_resp | is_old ~ condition - 1, criterion ~ (condition - 1 | item),
                                           family = evsd(), encoding_vars = "condition"), d)),
                 "Encoding variable in criterion formula: condition")
  expect_warning(suppressMessages(broc(brf(item_resp | is_old ~ condition - 1, criterion ~ condition,
                                           family = evsd(), encoding_vars = "condition"), d)),
                 "Encoding variable in criterion formula")
  expect_no_warning(suppressMessages(broc(brf(item_resp | is_old ~ condition - 1, criterion ~ (1 | item),
                                              family = evsd(), encoding_vars = "condition"), d)))
})
