.si_data <- function(seed = 5, n_sub = 12, n_per = 60) {
  set.seed(seed)
  d <- expand.grid(trial = seq_len(n_per), sub = seq_len(n_sub))
  d$is_old <- as.integer(d$trial %% 2 == 0)
  d$w <- round(runif(nrow(d), -2, 2), 3)
  shift <- 0.5 * sin(1.5 * d$w) + rnorm(n_sub, 0, 0.3)[d$sub]
  lat <- rnorm(nrow(d), 1.1 * d$is_old)
  d$resp <- as.integer(1L + rowSums((lat - shift) > outer(rep(1, nrow(d)), c(-0.9, -0.2, 0.5, 1.2))))
  d$sub <- factor(d$sub)
  d
}

.si_formula <- function(...) {
  brf(resp | is_old ~ 1, criterion ~ 1 + s(w, k = 6) + (1 | sub), ..., family = evsd())
}

test_that("smooth_info builds positive semi-definite blocks for population smooths only", {
  d <- .si_data()
  m <- suppressMessages(broc(brf(resp | is_old ~ 1 + s(w, k = 5),
                                 criterion ~ 1 + s(w, k = 6) + s(w, sub, bs = "fs", k = 4),
                                 family = evsd()), d, smooth_info = TRUE))
  info <- m$smooth_info$info
  expect_setequal(names(info), c("criterion_sw_1", "dprime_sw_1"))
  expect_equal(dim(info$criterion_sw_1)[1], 4L)
  for (b in names(info)) {
    blocks <- if (length(dim(info[[b]])) == 3) lapply(seq_len(dim(info[[b]])[1]), function(r) info[[b]][r, , ]) else list(info[[b]])
    for (I in blocks) {
      expect_equal(I, t(I), tolerance = 1e-12)
      expect_gte(min(eigen(I, symmetric = TRUE, only.values = TRUE)$values), -1e-8)
    }
    expect_true(any(grepl(paste0("Ism_", b), names(m$stan_data), fixed = TRUE)))
  }
  expect_false(any(grepl("Ism_criterion_swsub", names(m$stan_data))))
  m0 <- suppressMessages(broc(.si_formula(), d))
  expect_null(m0$smooth_info)
  expect_false(any(grepl("Ism_", names(m0$stan_data), fixed = TRUE)))
  expect_warning(suppressMessages(broc(brf(resp | is_old ~ 1, family = evsd()), d, smooth_info = TRUE)),
                 "no population-level smooth terms")
})

test_that("smooth_info blocks equal the curvature of the log-likelihood in the smooth coefficients", {
  d <- .si_data()
  m <- suppressMessages(broc(.si_formula(), d, smooth_info = TRUE))
  src <- build_data_source_from_model(m)
  fx <- names(m$smooth_info$pilot)
  svars <- grep("^s_", get_predict_variables(m, include_re = FALSE), value = TRUE)
  base <- matrix(0, 1, length(fx) + length(svars), dimnames = list(NULL, c(fx, svars)))
  base[1, fx] <- m$smooth_info$pilot
  row <- 2L
  cols <- grep(paste0("^s_criterion_sw_1\\[", row, ","), svars, value = TRUE)
  K <- length(cols); h <- 1e-3
  D <- base[rep(1, 1 + 2 * K + 4 * K * (K - 1) / 2), , drop = FALSE]
  i <- 1L
  for (a in seq_len(K)) {
    D[i + 1, cols[a]] <- h; D[i + 2, cols[a]] <- -h; i <- i + 2L
  }
  pairs <- which(upper.tri(diag(K)), arr.ind = TRUE)
  for (p in seq_len(nrow(pairs))) {
    a <- pairs[p, 1]; b <- pairs[p, 2]
    for (sg in list(c(1, 1), c(1, -1), c(-1, 1), c(-1, -1))) {
      i <- i + 1L
      D[i, cols[a]] <- sg[1] * h; D[i, cols[b]] <- sg[2] * h
    }
  }
  L <- rowSums(.obs_loglik_r(m, src, D))
  H <- matrix(0, K, K)
  for (a in seq_len(K)) H[a, a] <- -(L[2 * a] - 2 * L[1] + L[2 * a + 1]) / h^2
  off <- 1 + 2 * K
  for (p in seq_len(nrow(pairs))) {
    q <- L[off + (p - 1) * 4 + 1:4]
    H[pairs[p, 1], pairs[p, 2]] <- H[pairs[p, 2], pairs[p, 1]] <- -(q[1] - q[2] - q[3] + q[4]) / (4 * h^2)
  }
  I <- m$smooth_info$info$criterion_sw_1[row, , ]
  expect_lt(max(abs(I - H)) / max(abs(H)), 1e-3)
})

test_that("smooth_info: JAX log_lik matches the R reference", {
  skip_if_no_jax()
  m <- suppressMessages(broc(.si_formula(), .si_data(), smooth_info = TRUE))
  f <- suppressMessages(fit_broc(m, backend = "jax", sampler = "nuts", chains = 1, iter_warmup = 15, iter_sampling = 5,
                                 seed = 3, progress_bar = FALSE))
  dm <- posterior::as_draws_matrix(f$draws())
  ll_j <- as.matrix(posterior::as_draws_matrix(f$draws(variables = "log_lik")))
  expect_lt(max(abs(ll_j - .obs_loglik_r(m, build_data_source_from_model(m), dm, NULL))), 1e-5)
})
