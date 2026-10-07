.coding_data <- function() {
  d <- .fixture_univariate(n_per = 60)
  d$sub <- factor(rep(1:6, length.out = nrow(d)))
  d$x <- seq(-1, 1, length.out = nrow(d))
  d
}
.build <- function(d, f = brf(item_resp | is_old ~ 1, family = evsd()), ...) suppressMessages(broc(f, d, ...))

test_that("rows with missing values are dropped and recorded", {
  d <- .coding_data()
  d$x[4] <- NA; d$sub[7] <- NA; d$item_resp[9] <- NA
  expect_message(broc(brf(item_resp | is_old ~ x + (1 | sub), family = evsd()), d), "Dropping 3 row")
  m <- .build(d, brf(item_resp | is_old ~ x + (1 | sub), family = evsd()))
  expect_identical(m$dropped_rows, c(4L, 7L, 9L))
  expect_equal(m$stan_data$N, nrow(d) - 3L)
  expect_false(anyNA(m$stan_data$y))
})

test_that("the old/new indicator accepts numeric and logical codes and rejects labels", {
  d <- .coding_data()
  ref <- .build(d)$stan_data$is_old
  for (v in list(as.integer(d$is_old), as.logical(d$is_old), factor(d$is_old), as.character(d$is_old))) {
    dd <- d; dd$is_old <- v
    expect_identical(as.numeric(.build(dd)$stan_data$is_old), as.numeric(ref))
  }
  dd <- d; dd$is_old <- ifelse(d$is_old == 1, "old", "new")
  expect_error(.build(dd), "text labels")
  dd$is_old <- d$is_old + 1
  expect_error(.build(dd), "coded 0/1")
})

test_that("responses are coded from their observed range or an ordered factor", {
  d <- .coding_data()
  labs <- c("sure new", "new", "maybe new", "maybe old", "old", "sure old")
  ref <- .build(d)$stan_data$y
  dd <- d; dd$item_resp <- d$item_resp - 1L
  expect_identical(.build(dd)$stan_data$y, ref)
  dd$item_resp <- d$item_resp + 1L
  expect_message(broc(brf(item_resp | is_old ~ 1, family = evsd()), dd), "ranges from 2 to 7")
  expect_identical(.build(dd)$stan_data$y, ref)
  dd$item_resp <- factor(labs[d$item_resp], levels = labs, ordered = TRUE)
  m <- .build(dd)
  expect_identical(m$stan_data$y, ref)
  expect_identical(m$model_data$response_labels, labs)
  dd$item_resp <- ifelse(d$item_resp >= 4, d$item_resp + 1L, d$item_resp)
  m <- .build(dd)
  expect_equal(m$stan_data$K, 7L)
  expect_true(all(m$stan_data$y <= 7L))
  for (bad in list(labs[d$item_resp], factor(labs[d$item_resp]))) {
    dd$item_resp <- bad
    expect_error(.build(dd), "ordered factor")
  }
  dd$item_resp <- d$item_resp * 1.5
  expect_error(.build(dd), "whole numbers")
})

test_that("bivariate item types are codes or a factor in new/A/B order", {
  y <- yonelinas_1999
  f <- brf(resp(detect_rat, source_rat) | item_type ~ 1, discrim ~ 1, criterion ~ 1, criterion2 ~ 1, rho ~ 1,
           family = bivariate_gaussian())
  ref <- .build(y, f, counts = "count")$stan_data$item_type
  d <- y; d$item_type <- as.integer(y$item_type) - 1L
  expect_identical(.build(d, f, counts = "count")$stan_data$item_type, ref)
  d$item_type <- factor(as.character(y$item_type))
  expect_error(.build(d, f, counts = "count"), "level order must be new")
  d$item_type <- as.character(y$item_type)
  expect_error(.build(d, f, counts = "count"), "is text")
})

test_that("source codes and counts are validated", {
  h <- hilford_2002
  f <- brf(item_resp | source ~ 1, lambda ~ 1, family = source_mixture())
  ref <- .build(h, f, counts = "count")$stan_data$source
  d <- h; d$source <- as.integer(h$source)
  expect_identical(.build(d, f, counts = "count")$stan_data$source, ref)
  d$source <- as.character(h$source)
  expect_error(.build(d, f, counts = "count"), "is text")
  d <- h; d$count[1] <- d$count[1] + 0.5
  expect_error(.build(d, f, counts = "count"), "whole numbers")
})

test_that("cdp rejects text old/new labels and out-of-range old_levels", {
  r <- rotello_2005
  f <- brf(resp(rating, rk) | is_old ~ condition - 1, fam ~ condition - 1, criterion ~ condition - 1,
           rec_crit ~ condition - 1, sigma_R ~ condition - 1, family = cdp(old_levels = 2:6))
  d <- r; d$is_old <- ifelse(r$is_old == 1, "old", "new")
  expect_error(.build(d, f, counts = "count"), "text labels")
  d <- r; d$rating <- r$rating - 1L
  expect_error(.build(d, f, counts = "count"), "old_levels")
  f2 <- brf(resp(rating, rk) | is_old ~ condition - 1, fam ~ condition - 1, criterion ~ condition - 1,
            rec_crit ~ condition - 1, sigma_R ~ condition - 1, family = cdp(old_levels = 1:5))
  m0 <- .build(r, f, counts = "count")
  m1 <- .build(d, f2, counts = "count")
  expect_identical(m1$stan_data$y, m0$stan_data$y)
  expect_identical(m1$stan_data$old_level_map, m0$stan_data$old_level_map)
})

test_that("broc() drops factor levels a subset no longer contains", {
  set.seed(4)
  d <- data.frame(sub = factor(rep(1:8, each = 30)), cond = factor(rep(c("a", "b", "c", "d"), each = 60)),
                  old = rep(0:1, 120), conf = sample(1:4, 240, TRUE))
  sub_d <- d[d$cond %in% c("a", "b"), ]
  expect_equal(nlevels(sub_d$cond), 4L)
  m <- suppressMessages(broc(brf(conf | old ~ cond - 1 + (cond - 1 | sub), criterion ~ 1, family = evsd()), sub_d))
  expect_equal(nlevels(m$data$cond), 2L)
  expect_equal(nlevels(m$data$sub), 4L)
  cfg <- build_numpyro_config(m)
  expect_equal(as.integer(cfg$random_effects[[1]]$n_terms), 2L)
  expect_equal(as.integer(cfg$random_effects[[1]]$n_groups), 4L)
})
