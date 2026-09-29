re_dup_data <- function() {
  d <- .fixture_crossed()
  set.seed(11)
  d$x <- round(rnorm(nrow(d)), 2)
  d
}

test_that("duplicate grouping factors within a parameter are rejected", {
  d <- re_dup_data()
  expect_error(
    broc(brf(item_resp | is_old ~ 1 + (1 | sub) + (0 + x | sub), family = evsd()), data = d),
    "one term per grouping factor"
  )
  expect_error(
    broc(brf(item_resp | is_old ~ 1 + (1 | sub),
             criterion ~ 1 + (1 | sub) + (0 + x || sub), family = evsd()), data = d),
    "one term per grouping factor"
  )
})

test_that("distinct grouping factors and cross-parameter ids still build", {
  d <- re_dup_data()
  expect_s3_class(
    broc(brf(item_resp | is_old ~ 1 + (1 | sub) + (1 | item), family = evsd()), data = d),
    "broc_model"
  )
  expect_s3_class(
    broc(brf(item_resp | is_old ~ 1 + (1 | p | sub) + (1 | i | item),
             sigma ~ 1 + (1 | p | sub) + (1 | i | item),
             criterion ~ 1 + (1 | p | sub) + (1 | i | item),
             family = uvsd()), data = d),
    "broc_model"
  )
  expect_s3_class(
    broc(brf(item_resp | is_old ~ 1 + (1 + x | sub), family = evsd()), data = d),
    "broc_model"
  )
})
