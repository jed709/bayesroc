test_that("prior_init draws each element from its own prior", {
  set.seed(1)
  d <- expand.grid(participant = factor(1:6), item = factor(1:10))
  d$old <- as.integer(ave(runif(nrow(d)), d$participant, FUN = function(u) rank(u) <= 5))
  d$conf <- sample(1:6, nrow(d), TRUE)
  m <- broc(brf(conf | old ~ 1 + (1|p|participant),
                sigma ~ 1 + (1|p|participant),
                criterion ~ 1 + (1|p|participant),
                family = uvsd()), d,
            priors = broc_prior("normal(0, 0.05)", class = "sd", dpar = "log_gaps", group = "participant"))
  L <- strsplit(m$stan_code, "\n")[[1]]
  st <- regmatches(L, regexec("sigma_cross_p_participant\\[([0-9]+)\\] ~ normal\\(0, ([0-9.]+)\\)", L))
  st <- st[lengths(st) == 3]
  want <- as.numeric(vapply(st, `[`, "", 3))[order(as.integer(vapply(st, `[`, "", 2)))]
  expect_equal(want, c(1, 0.3, 0.05, 0.05, 1, 0.05, 0.05))
  sds <- t(vapply(prior_init(m, chains = 4000, seed = 4), `[[`, numeric(7), "sigma_cross_p_participant"))
  got <- colMeans(sds) / sqrt(2 / pi)
  expect_true(all(abs(got / want - 1) < 0.06), info = paste(round(got, 3), collapse = " "))
})
