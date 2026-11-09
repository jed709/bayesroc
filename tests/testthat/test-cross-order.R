.cross_order_data <- function(conf) {
  set.seed(3)
  d <- expand.grid(participant = factor(1:6), item = factor(1:10))
  d$old <- as.integer(ave(runif(nrow(d)), d$participant, FUN = function(u) rank(u) <= 5))
  d$conf <- conf(nrow(d))
  d
}

test_that("JAX cross-parameter blocks follow the R member order", {
  d <- .cross_order_data(function(n) sample(1:6, n, TRUE))
  m <- broc(brf(conf | old ~ 1 + (1|p|participant) + (1|i|item),
                sigma ~ 1 + (1|p|participant) + (1|i|item),
                criterion ~ 1 + (1|p|participant) + (1|i|item),
                family = uvsd()), d)
  cfg <- build_numpyro_config(m)
  for (cid in names(m$model_data$cross_cor)) {
    want <- vapply(m$model_data$cross_cor[[cid]]$members, `[[`, "", "param")
    got <- vapply(Filter(function(r) identical(r$cor_id, cid), cfg$random_effects), `[[`, "", "param")
    expect_equal(got, want)
  }
  expect_equal(vapply(m$model_data$cross_cor$p$members, `[[`, "", "param"), c("dprime", "sigma", "criterion"))
})

test_that("an ordered response keeps its declared scale when a category is unused", {
  d <- .cross_order_data(function(n) factor(sample(1:5, n, TRUE), levels = 1:6, ordered = TRUE))
  m <- broc(brf(conf | old ~ 1, criterion ~ 1, family = evsd()), d)
  expect_equal(m$stan_data$K, 6L)
  d$conf <- as.integer(as.character(d$conf))
  expect_equal(broc(brf(conf | old ~ 1, criterion ~ 1, family = evsd()), d)$stan_data$K, 5L)
})
