# Helpers shared with tests/testthat/
# test-smooth-criterion.R (copied).

fixture_criterion_smooth <- function(seed = 7, n_sub = 12, n_item = 40) {
  set.seed(seed)
  d <- expand.grid(item = seq_len(n_item), sub = seq_len(n_sub))
  d$old <- as.integer((d$item + d$sub) %% 2 == 0)
  d$x <- round(runif(nrow(d), -2, 2), 3)
  tau <- c(-1.0, -0.35, 0.3, 0.95, 1.7)
  n <- nrow(d)
  lat <- ifelse(d$old == 1, rnorm(n, 1.0, 1), rnorm(n))
  thr <- outer(rep(1, n), tau) + 0.45 * sin(1.6 * d$x)
  d$conf <- as.integer(1L + rowSums(lat > thr))
  d$sub <- factor(d$sub)
  d
}
pop_means <- function(f) {
  s <- posterior::summarise_draws(f$draws(format = "array"), "mean")
  s <- s[grep("^beta_|^sds_", s$variable), ]
  nm <- sub("[[]([0-9]+)[]]$", "[\\1,1]", s$variable)
  stats::setNames(s$mean, nm)
}

test_that("criterion smooth builds valid batch C++ that indexes the coefficient matrix by row", {
  d <- fixture_criterion_smooth()
  m <- broc(brf(conf | old ~ 1, criterion ~ s(x), family = evsd()), data = d)
  l <- strsplit(.with_stan_code(m)$batch_cpp$cpp_code, "\n", fixed = TRUE)[[1]]
  expect_length(grep("s_criterion_sx_1_d[.]size[(][)]", l), 0)
  expect_gt(length(grep("thresh_mid_val [+][=] .*s_criterion_sx_1_d[(]0, j[)]", l)), 0)
  expect_gt(length(grep("Eigen::VectorXd obs_exp_gaps", l)), 0)
  expect_length(grep("propagate_thresh_grad[^;]*pop_exp_gaps", paste(l, collapse = " ")), 0)
})

test_that("criterion smooth agrees across batch C++, plain Stan and JAX", {
  skip_if_not_slow()
  skip_on_cran()
  skip_if_no_cmdstan()
  skip_if_no_jax()
  d <- fixture_criterion_smooth()
  m_batch <- broc(brf(conf | old ~ 1, criterion ~ s(x), family = evsd()), data = d)
  m_plain <- .broc_plain(brf(conf | old ~ 1, criterion ~ s(x), family = evsd()), data = d)

  fb <- stan_fit_broc(m_batch, chains = 2, parallel_chains = 2,
                 iter_warmup = 600, iter_sampling = 600, seed = 11, refresh = 50)
  fp <- stan_fit_broc(m_plain, chains = 2, parallel_chains = 2,
                 iter_warmup = 600, iter_sampling = 600, seed = 11, refresh = 50)
  fj <- fit_broc(m_batch, backend = "jax", sampler = "nuts", chains = 2, parallel_chains = 2,
                 iter_warmup = 600, iter_sampling = 600, seed = 11, progress_bar = FALSE)

  a <- pop_means(fb); b <- pop_means(fp); j <- pop_means(fj)
  common <- intersect(intersect(names(a), names(b)), names(j))
  expect_true(any(grepl("^beta_log_gaps", common)))

  fixed <- grep("^beta_", common, value = TRUE)
  expect_equal(unname(a[fixed]), unname(b[fixed]), tolerance = 0.05)
  expect_equal(unname(a[fixed]), unname(j[fixed]), tolerance = 0.05)

  sds <- grep("^sds_", common, value = TRUE)
  expect_equal(unname(a[sds]), unname(b[sds]), tolerance = 0.25)
})
