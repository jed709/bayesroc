.rand_bivariate_case <- function(seed, S = 7, N = 9, K1 = 4, K2 = 5, bounded = FALSE, varying = FALSE,
                                 shared = FALSE, dp = FALSE, extras = TRUE) {
  set.seed(seed)
  mat <- function(f) matrix(f(S * N), S, N)
  params <- list(dprime = mat(function(n) runif(n, 0.2, 2)), discrim = mat(function(n) runif(n, 0.05, 1.5)),
                 rho = mat(function(n) runif(n, 0.05, 0.9)))
  if (extras) {
    params$sigma <- mat(function(n) runif(n, 0.7, 1.6)); params$sigma2 <- mat(function(n) runif(n, 0.7, 1.6))
    params$sigma_B <- mat(function(n) runif(n, 0.7, 1.6)); params$rho_B <- mat(function(n) runif(n, 0.05, 0.9))
    params$dprime_B <- mat(function(n) runif(n, 0.2, 2)); params$discrim_B <- mat(function(n) runif(n, 0.05, 1.5))
    if (!bounded) params$rho_N <- mat(function(n) runif(n, -0.3, 0.3))
  }
  if (dp) { params$lambda <- mat(function(n) runif(n, 0.05, 0.6)); params$lambda2 <- mat(function(n) runif(n, 0.1, 0.9)) }
  ord <- function(K) t(apply(matrix(rnorm(S * N * (K - 1), 0, 1.2), S * N), 1, sort))
  thresh <- array(ord(K1), c(S, N, K1 - 1))
  thresh2 <- if (varying) aperm(array(vapply(seq_len(K1), function(j) ord(K2), matrix(0, S * N, K2 - 1)),
                                        c(S, N, K2 - 1, K1)), c(1, 2, 4, 3))
             else array(ord(K2), c(S, N, K2 - 1))
  if (shared) attr(thresh2, "thresh2_new") <- t(apply(matrix(rnorm(S * (K2 - 1)), S), 1, sort))
  md <- list(has_dprime_B = extras, has_discrim_B = extras, varying_source_criteria = varying, bounded = bounded,
             new_source_criteria = if (shared) "shared" else "full", old_levels = if (shared) 3:4 else NULL)
  list(params = params, thresh = thresh, thresh2 = thresh2, item_type = sample(rep_len(1:3, N)), S = S, N = N,
       K1 = K1, K2 = K2, model = list(model_data = md))
}

test_that("vectorised bivariate cell probabilities equal the per-row, per-draw reference", {
  grid <- expand.grid(bounded = c(FALSE, TRUE), varying = c(FALSE, TRUE), shared = c(FALSE, TRUE),
                      dp = c(FALSE, TRUE), extras = c(FALSE, TRUE))
  grid <- grid[!(grid$shared & !grid$varying), ]
  for (native in c(FALSE, TRUE)) {
    withr::local_options(bayesroc.native_predict = native)
    .bvn_native$ok <- NULL
    if (native && !.bvn_native_ready()) next
    for (r in seq_len(nrow(grid))) {
      g <- grid[r, ]
      cs <- .rand_bivariate_case(100 + r, bounded = g$bounded, varying = g$varying, shared = g$shared,
                                 dp = g$dp, extras = g$extras)
      new <- if (g$dp) probs_bivariate_dp else probs_bivariate_sdt
      old <- if (g$dp) .ref_probs_bivariate_dp else .ref_probs_bivariate_sdt
      a <- do.call(new, cs[c("params", "thresh", "thresh2", "item_type", "S", "N", "K1", "K2", "model")])
      .bvn_native$ok <- FALSE
      b <- do.call(old, cs[c("params", "thresh", "thresh2", "item_type", "S", "N", "K1", "K2", "model")])
      .bvn_native$ok <- NULL
      expect_equal(dim(a), dim(b))
      expect_lt(max(abs(a - b)), if (native) 1e-9 else 1e-12, label = paste(c(native, unlist(g)), collapse = " "))
      expect_lt(max(abs(apply(a, c(1, 2), sum) - 1)), 1e-6)
    }
  }
  .bvn_native$ok <- NULL
})

test_that("the detection margin of every bivariate family equals the joint summed over source", {
  .bvn_native$ok <- FALSE
  on.exit(.bvn_native$ok <- NULL)
  for (r in 1:8) {
    g <- expand.grid(bounded = c(FALSE, TRUE), varying = c(FALSE, TRUE), dp = c(FALSE, TRUE))[r, ]
    cs <- .rand_bivariate_case(200 + r, bounded = g$bounded, varying = g$varying, dp = g$dp)
    fam <- if (g$dp) "bivariate_dp" else "bivariate_sdt"
    joint <- do.call(if (g$dp) probs_bivariate_dp else probs_bivariate_sdt,
                     cs[c("params", "thresh", "thresh2", "item_type", "S", "N", "K1", "K2", "model")])
    m1 <- probs_detection_margin(fam, cs$params, cs$thresh, cs$item_type, cs$S, cs$N, cs$K1, cs$model)
    expect_lt(max(abs(m1 - rowSums(joint, dims = 3L))), 1e-8, label = paste(fam, paste(unlist(g), collapse = " ")))
  }
  cs <- .rand_bivariate_case(301, varying = TRUE)
  cs$params$dprime2 <- matrix(runif(cs$S * cs$N, 0.5, 3), cs$S); cs$params$lambda <- matrix(runif(cs$S * cs$N, 0.1, 0.7), cs$S)
  cs$model$model_data$has_discrim_B <- FALSE
  joint <- probs_vrdp2d(cs$params, cs$thresh, cs$thresh2, cs$item_type, cs$S, cs$N, cs$K1, cs$K2, cs$model)
  m1 <- probs_detection_margin("vrdp2d", cs$params, cs$thresh, cs$item_type, cs$S, cs$N, cs$K1, cs$model)
  expect_lt(max(abs(m1 - rowSums(joint, dims = 3L))), 1e-8)
  cs <- .rand_bivariate_case(302)
  joint <- probs_bivariate_cumulative(cs$params, cs$thresh, cs$thresh2, cs$S, cs$N, cs$K1, cs$K2, cs$model)
  m1 <- probs_detection_margin("bivariate_cumulative", cs$params, cs$thresh, NULL, cs$S, cs$N, cs$K1, cs$model)
  expect_lt(max(abs(m1 - rowSums(joint, dims = 3L))), 1e-8)
})

test_that("native detection survival equals the R path along the criterion path and on a grid", {
  .bvn_native$ok <- NULL
  skip_if_not(.bvn_native_ready())
  on.exit(.bvn_native$ok <- NULL)
  for (fam in c("bivariate_sdt", "bivariate_dp", "vrdp2d")) {
    cs <- .rand_bivariate_case(400, S = 6, N = 11, dp = fam != "bivariate_sdt", varying = TRUE)
    if (fam == "vrdp2d") cs$params$dprime2 <- matrix(runif(66, 0.5, 3), 6)
    cs$model$family <- list(family = fam, link = list(name = "probit"))
    setup <- list(S = cs$S, tp = cs$params, th = cs$thresh, th2 = NULL, data_src = list(item_type = cs$item_type))
    rows <- c(2L, 3L, 5L, 7L, 8L, 11L); w <- runif(length(rows), 0.5, 2)
    u <- .roc_u_grid(cs$K1 - 1L)
    for (mode in c("path", "grid")) {
      g <- if (mode == "path") u else seq(-4, 4, length.out = 25)
      crit <- if (mode == "path") function(th) .criterion_path(th, g)
              else function(th) array(rep(g, each = dim(th)[1] * dim(th)[2]), c(dim(th)[1:2], length(g)))
      .bvn_native$ok <- FALSE
      ref <- .roc_survival_rows(cs$model, setup, rows, w, 1, length(g), crit)
      .bvn_native$ok <- NULL
      attr(crit, "u") <- g; attr(crit, "mode") <- mode
      nat <- .roc_survival_rows(cs$model, setup, rows, w, 1, length(g), crit)
      expect_lt(max(abs(nat - ref)), 1e-9, label = paste(fam, mode))
    }
  }
})
