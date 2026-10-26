test_that("a design with as many rows as random-effect blocks fits through the transport map", {
  skip_on_cran()
  skip_if_no_jax()
  data(prm09, package = "bayesroc", envir = environment())
  d <- prm09[match(0:5, prm09$resp), ]
  d$sub <- factor(d$sub); d$item <- factor(d$item); d$conf <- d$resp + 1L
  m <- suppressMessages(broc(brf(conf | cond ~ 1 + (1 |s| sub) + (1 |i| item),
                                 criterion ~ 1 + (1 |s| sub) + (1 |i| item),
                                 sigma ~ 1 + (1 |s| sub) + (1 |i| item),
                                 family = uvsd()), data = d))
  expect_equal(nrow(m$data), length(build_numpyro_config(m)$random_effects))
  for (s in c("drghmc", "nuts")) {
    f <- fit_broc(m, backend = "jax", sampler = s, chains = 2, iter_warmup = 50, iter_sampling = 10,
                  seed = 1, progress_bar = FALSE)
    expect_equal(dim(f$draws(variables = "sigma_cross_s_sub"))[3], 7L)
  }
})
