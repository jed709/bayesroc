fixture_cdp_all <- function(n_rkg = 3L, seed = 4, n_sub = 8, n_item = 40, drop = 0) {
  set.seed(seed)
  mu_R <- 1.1; sigma_R <- 1.4; mu_F <- 0.7; sigma_F <- 0.9
  c_R <- 1.0; c_K <- 0.2
  tau <- c(0.1, 0.9, 1.8)
  rows <- expand.grid(sub = seq_len(n_sub), item = seq_len(n_item))
  rows$old <- rep(c(0L, 1L), length.out = nrow(rows))
  n <- nrow(rows)
  R <- ifelse(rows$old == 1L, rnorm(n, mu_R, sigma_R), rnorm(n))
  F_ <- ifelse(rows$old == 1L, rnorm(n, mu_F, sigma_F), rnorm(n))
  rows$conf <- as.integer(1L + rowSums(outer(R + F_, tau, ">")))
  rows$rk <- if (n_rkg == 3L) ifelse(R > c_R, 1L, ifelse(F_ > c_K, 2L, 3L)) else ifelse(R > c_R, 1L, 2L)
  if (drop > 0) rows$rk[sample.int(n, round(drop * n))] <- NA_integer_
  rows$sub <- factor(rows$sub)
  rows
}

cdp_all_formula <- function(n_rkg, family = cdp()) {
  if (n_rkg == 3L) brf(resp(conf, rk) | old ~ 1, fam ~ 1, criterion ~ 1, rec_crit ~ 1, know_crit ~ 1,
                       sigma_R ~ 1, family = family)
  else brf(resp(conf, rk) | old ~ 1, fam ~ 1, criterion ~ 1, rec_crit ~ 1, sigma_R ~ 1, family = family)
}

test_that("judgements on every response switch cdp to the full confidence scale", {
  d <- fixture_cdp_all(3L)
  expect_message(m <- broc(cdp_all_formula(3L), data = d), "every confidence level")
  expect_identical(m$family$old_levels, 2:4)
  expect_identical(m$family$J, 3L)
  expect_identical(m$family$rk_levels, 1:4)
  expect_true(all(m$stan_data$rk %in% 1:3))

  expect_message(m2 <- broc(cdp_all_formula(3L, cdp(old_levels = 3:4)), data = d), "beyond old_levels")
  expect_identical(m2$family$old_levels, 2:4)

  d_old <- d
  d_old$rk[d_old$conf < 3L] <- NA_integer_
  m3 <- suppressMessages(broc(cdp_all_formula(3L), data = d_old))
  expect_identical(m3$family$old_levels, 3:4)
  expect_identical(m3$family$rk_levels, 3:4)
  expect_true(all(m3$stan_data$rk[d_old$conf < 3L] == 0L))
})

test_that("missing judgements are allowed anywhere and enter as confidence only", {
  d <- fixture_cdp_all(2L, drop = 0.2)
  m <- suppressMessages(broc(cdp_all_formula(2L), data = d))
  expect_identical(m$stan_data$rk == 0L, is.na(d$rk))
  expect_identical(m$family$old_levels, 2:4)
})

test_that("cdp on all responses: batch C++ matches plain Stan (R/K and R/K/G, with gaps)", {
  skip_if_no_cmdstan()
  for (n_rkg in 2:3) for (drop in c(0, 0.25)) {
    d <- fixture_cdp_all(n_rkg, drop = drop)
    mk <- function(b) suppressMessages(.broc_with(b, cdp_all_formula(n_rkg), data = d))
    edits <- list(list("beta_rec_crit[1]" = 1.0), list("beta_rec_crit[1]" = 3.5), list("beta_rec_crit[1]" = -2.5))
    if (n_rkg == 3L) edits <- lapply(edits, function(e) c(e, list("beta_know_crit[1]" = 0.4)))
    for (ed in edits) {
      r <- .stan_parity_at(mk(FALSE), mk(TRUE), ed)
      lab <- sprintf("n_rkg %d drop %.2f %s", n_rkg, drop, paste(names(ed), unlist(ed), collapse = " "))
      expect_true(all(is.finite(r$lp_values)), label = lab)
      expect_lt(r$lp, 1e-7, label = paste(lab, "log density"))
      expect_lt(r$grad, 1e-6, label = paste(lab, "gradient"))
    }
  }
})

test_that("cdp on all responses: Stan and JAX agree and recover the generating values", {
  skip_on_cran()
  skip_if_no_cmdstan()
  skip_if_no_jax()
  d <- fixture_cdp_all(3L, n_sub = 20, n_item = 60)
  m <- suppressMessages(broc(cdp_all_formula(3L), data = d))
  fs <- fit_broc(m, backend = "stan", chains = 2, parallel_chains = 2,
                 iter_warmup = 400, iter_sampling = 400, seed = 3, refresh = 50)
  fj <- fit_broc(m, backend = "jax", chains = 2, parallel_chains = 2,
                 iter_warmup = 400, iter_sampling = 400, seed = 3, progress_bar = FALSE)
  strip_idx <- function(v) sub("\\[[0-9]+(,[0-9]+)?\\]$", "", v)
  tab <- function(f) {
    s <- posterior::summarise_draws(f$draws(format = "array"), mean = mean, sd = stats::sd)
    s <- s[grep("^beta_", s$variable), ]
    list(mean = stats::setNames(s$mean, make.unique(strip_idx(s$variable))),
         sd = stats::setNames(s$sd, make.unique(strip_idx(s$variable))))
  }
  a <- tab(fs); b <- tab(fj)
  common <- intersect(names(a$mean), names(b$mean))
  expect_gte(length(common), 7)
  expect_lt(max(abs(a$mean[common] - b$mean[common]) / a$sd[common]), 0.5)
  truth <- c(beta_dprime = 1.1, beta_dprime2 = 0.7, beta_rec_crit = 1.0, beta_know_crit = 0.2, beta_sigma = log(1.4))
  z <- (b$mean[names(truth)] - truth) / b$sd[names(truth)]
  expect_lt(max(abs(z)), 3.5)
  expect_equal(sum(fj$diagnostic_summary()$num_divergent), 0)

  p <- predict(fj, type = "response", ndraws = 20)
  expect_equal(as.vector(tapply(p$mean, p$trial, sum)), rep(1, nrow(d)), tolerance = 1e-6)
  pr <- predict(fj, type = "response", ndraws = 20, summary = FALSE, response = 2)
  expect_equal(range(apply(pr, c(1, 2), sum)), c(1, 1), tolerance = 1e-8)
  obs <- tapply(seq_len(nrow(d)), d$old, function(i) tabulate(d$rk[i], 3) / length(i))
  fit_p <- sapply(c("0", "1"), function(o) colMeans(apply(pr[, d$old == as.integer(o), , drop = FALSE], c(1, 3), mean)))
  expect_lt(max(abs(fit_p - do.call(cbind, obs))), 0.06)
})

test_that("R-side cdp cells partition an open-below band", {
  for (n_rkg in 2:3) {
    fam <- list(old_levels = 2:4, J = 3L, n_rkg = n_rkg)
    S <- 3L; N <- 2L
    mat <- function(v) matrix(v, S, N)
    params <- list(dprime = mat(1.1), dprime2 = mat(0.7), sigma = mat(1.4), sigma2 = mat(0.9),
                   rec_crit = mat(1.0), know_crit = mat(0.2))
    th <- array(rep(c(0.1, 0.9, 1.8), each = S * N), c(S, N, 3L))
    p <- probs_cdp(params, th, c(0, 1), S, N, 4L, list(family = fam))
    expect_equal(as.vector(apply(p, c(1, 2), sum)), rep(1, S * N), tolerance = 1e-10)
    expect_equal(p[1, 1, 1], pnorm(0.1 / sqrt(2)), tolerance = 1e-8)
    expect_equal(p[1, 2, 1], pnorm((0.1 - 1.8) / sqrt(1.4^2 + 0.9^2)), tolerance = 1e-8)
    pr <- attr(p, "p_remember")
    expect_true(all(pr > 0 & pr < 1))
    set.seed(1)
    R <- rnorm(4e5); F_ <- rnorm(4e5)
    sel <- R + F_ < 0.1
    expect_lt(abs(pr[1, 1, 1] - mean(R[sel] > 1.0)), 0.003)
    if (n_rkg == 3L) expect_lt(abs(attr(p, "p_know")[1, 1, 1] - mean(R[sel] <= 1.0 & F_[sel] > 0.2)), 0.005)
  }
})
