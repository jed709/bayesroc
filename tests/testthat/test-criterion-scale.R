.ls_data <- function(seed = 21, n_sub = 12, n_per = 40) {
  set.seed(seed)
  d <- expand.grid(trial = seq_len(n_per), sub = seq_len(n_sub))
  d$x <- rnorm(nrow(d))
  d$cond <- factor(sample(c("a", "b"), nrow(d), TRUE))
  d$is_old <- rbinom(nrow(d), 1, 0.5)
  t <- c(-1.1, -0.5, 0.1, 0.6, 1.2)
  loc <- 0.2 * (d$cond == "b") + rnorm(n_sub, 0, 0.3)[d$sub]
  s <- 0.3 * d$x + rnorm(n_sub, 0, 0.2)[d$sub]
  z <- rnorm(nrow(d), ifelse(d$is_old == 1, 1.2, 0), ifelse(d$is_old == 1, 1.2, 1))
  d$item_resp <- vapply(seq_len(nrow(d)), function(i) sum(z[i] > loc[i] + exp(s[i]) * t) + 1L, integer(1))
  d$sub <- factor(d$sub)
  d
}

.ls_formula <- function() {
  brf(item_resp | is_old ~ 1, sigma ~ 1, criterion ~ cond + (1 | p | sub), criterion_scale ~ x + (1 | p | sub),
      family = uvsd())
}

test_that("a criterion_scale formula gives the criterion one location component and the scale no intercept", {
  d <- .ls_data()
  m <- suppressMessages(broc(.ls_formula(), d))
  md <- m$model_data
  expect_true(md$has_criterion_scale)
  expect_true(md$criterion$location_scale)
  expect_equal(md$criterion$random$sub$dim, 1)
  expect_identical(unname(md$criterion_scale_fixed$coef_names), "x")
  cfg <- build_numpyro_config(m)
  expect_true(cfg$crit_ls)
  expect_true(isTRUE(cfg$random_effects[[which(vapply(cfg$random_effects, `[[`, "", "param") == "criterion")]]$location_only))
  expect_match(m$stan_code, "u_ls_rows", fixed = TRUE)
  m0 <- suppressMessages(broc(brf(item_resp | is_old ~ 1, criterion_scale ~ 1, family = evsd()), d))
  expect_equal(m0$model_data$criterion_scale_fixed$n_coef, 0)
  d2 <- d; d2$item_resp <- pmin(d2$item_resp, 2L)
  expect_error(broc(brf(item_resp | is_old ~ 1, criterion_scale ~ x, family = evsd()), d2), "at least two criteria")
})

test_that("location-scale thresholds: R reproduces the sampler's log-likelihood on both gap links", {
  skip_if_no_jax()
  d <- .ls_data()
  for (gl in c("log", "softplus")) {
    m <- suppressMessages(broc(.ls_formula(), d, gap_link = if (gl == "softplus") "softplus"))
    f <- suppressMessages(fit_broc(m, backend = "jax", sampler = "drghmc", chains = 1, iter_warmup = 150,
                                   iter_sampling = 20, seed = 1, progress_bar = FALSE))
    dm <- f$draws(format = "matrix")
    ll_r <- .obs_loglik_r(m, build_data_source_from_model(m), dm, re_formula = NULL)
    ll_j <- .read_loglik_SN_cols(.loglik_path(f), 1L, m$model_data$N)
    expect_lt(max(abs(ll_r - ll_j)), 1e-5, label = paste(gl, "JAX vs R log_lik"))
  }
})

test_that("location-scale criteria agree across plain Stan, batch Stan and R", {
  skip_if_no_cmdstan()
  d <- .ls_data()
  cases <- list(
    log = list(.ls_formula(), d, NULL),
    softplus = list(.ls_formula(), d, "softplus"),
    cumulative = list(brf(item_resp ~ 1, cutpoints ~ cond + (1 | sub), cutpoints_scale ~ x, family = cumulative()), d, NULL)
  )
  for (nm in names(cases)) {
    c1 <- cases[[nm]]
    mb <- suppressMessages(broc(c1[[1]], c1[[2]], gap_link = c1[[3]]))
    mp <- suppressMessages(.broc_plain(c1[[1]], c1[[2]], gap_link = c1[[3]]))
    r <- .stan_parity(mp, mb)
    expect_lt(r$lp, 1e-8, label = paste(nm, "plain vs batch log density"))
    expect_lt(r$grad, 1e-8, label = paste(nm, "plain vs batch gradient"))
    f <- suppressMessages(fit_broc(mb, backend = "stan", chains = 1, iter_warmup = 100, iter_sampling = 3,
                                   seed = 1, refresh = 50, show_messages = FALSE))
    dm <- posterior::as_draws_matrix(f$draws())
    ll_stan <- as.matrix(dm[, grep("^log_lik\\[", colnames(dm)), drop = FALSE])
    if (!ncol(ll_stan)) ll_stan <- .read_loglik_SN_cols(.loglik_path(f), 1L, mb$model_data$N)
    ll_r <- .obs_loglik_r(mb, build_data_source_from_model(mb), dm, re_formula = NULL)
    expect_lt(max(abs(ll_stan - ll_r)), 1e-5, label = paste(nm, "Stan vs R log_lik"))
  }
})

.ls2_cases <- function() {
  set.seed(4)
  bv <- .fixture_bivariate_sdt(n_per = 150)
  bv$x <- rnorm(nrow(bv)); bv$sub <- factor(sample(1:10, nrow(bv), TRUE))
  bc <- .fixture_bivariate_cumulative(n = 400); bc$x <- rnorm(nrow(bc)); bc$sub <- factor(sample(1:10, nrow(bc), TRUE))
  list(
    varying = list(brf(resp(y1, y2) | item_type ~ 1, dprime2 ~ 1, discrim ~ 1, lambda ~ 1, criterion ~ 1,
                       criterion2 ~ 1 + x + (1 | sub), criterion2_scale ~ x + (1 | sub),
                       family = vrdp2d(source_criteria = "varying")), bv),
    cumulative = list(brf(resp(y1, y2) ~ 1, cutpoints ~ cond, cutpoints2 ~ cond + (1 | sub), cutpoints2_scale ~ x,
                          rho ~ 1, family = bivariate_cumulative()), bc)
  )
}

test_that("criterion2_scale: one scale per response bin, location-only source criteria", {
  m <- suppressMessages(broc(.ls2_cases()$varying[[1]], .ls2_cases()$varying[[2]]))
  md <- m$model_data
  expect_true(md$has_criterion2_scale && md$criterion2$location_scale)
  expect_length(md$criterion2_scale_fixed$coef_names, 4L)
  expect_identical(md$criterion2$random$sub$varying_re_mode, "per_bin")
})

test_that("criterion2_scale: R reproduces the JAX log-likelihood", {
  skip_if_no_jax()
  for (nm in names(cs <- .ls2_cases())) {
    m <- suppressMessages(broc(cs[[nm]][[1]], cs[[nm]][[2]]))
    f <- suppressMessages(fit_broc(m, backend = "jax", sampler = "drghmc", chains = 1, iter_warmup = 150,
                                   iter_sampling = 10, seed = 1, progress_bar = FALSE))
    ll_r <- .obs_loglik_r(m, build_data_source_from_model(m), f$draws(format = "matrix"), re_formula = NULL)
    expect_lt(max(abs(ll_r - .read_loglik_SN_cols(.loglik_path(f), 1L, m$model_data$N))), 1e-5, label = nm)
  }
})

test_that("criterion2_scale: plain and batch Stan agree and match R", {
  skip_if_no_cmdstan()
  for (nm in names(cs <- .ls2_cases())) {
    mb <- suppressMessages(broc(cs[[nm]][[1]], cs[[nm]][[2]]))
    mp <- suppressMessages(.broc_plain(cs[[nm]][[1]], cs[[nm]][[2]]))
    r <- .stan_parity(mp, mb)
    expect_lt(r$lp, 1e-8, label = paste(nm, "plain vs batch log density"))
    expect_lt(r$grad, 1e-8, label = paste(nm, "plain vs batch gradient"))
    f <- suppressMessages(fit_broc(mb, backend = "stan", chains = 1, iter_warmup = 100, iter_sampling = 5, seed = 1,
                                   refresh = 50, show_messages = FALSE))
    ll_r <- .obs_loglik_r(mb, build_data_source_from_model(mb), f$draws(format = "matrix"), re_formula = NULL)
    expect_lt(max(abs(ll_r - .read_loglik_SN_cols(.loglik_path(f), 1L, mb$model_data$N))), 1e-5, label = nm)
  }
})
