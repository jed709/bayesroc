.ls_data <- function(seed = 21, n_sub = 12, n_per = 40) {
  set.seed(seed)
  d <- expand.grid(trial = seq_len(n_per), sub = seq_len(n_sub))
  d$x <- rnorm(nrow(d))
  d$cond <- factor(sample(c("a", "b"), nrow(d), TRUE))
  d$is_old <- rbinom(nrow(d), 1, 0.5)
  t <- c(-1.1, -0.5, 0.1, 0.6, 1.2)
  loc <- 0.2 * (d$cond == "b") + rnorm(n_sub, 0, 0.3)[d$sub]
  s <- 0.3 * d$x + 0.2 * (d$cond == "b") + rnorm(n_sub, 0, 0.2)[d$sub]
  z <- rnorm(nrow(d), ifelse(d$is_old == 1, 1.2, 0), ifelse(d$is_old == 1, 1.2, 1))
  d$item_resp <- vapply(seq_len(nrow(d)), function(i) sum(z[i] > loc[i] + exp(s[i]) * t) + 1L, integer(1))
  d$sub <- factor(d$sub)
  d
}

.ls_formula <- function() {
  brf(item_resp | is_old ~ 1, sigma ~ 1, criterion_shift ~ cond + (1 | p | sub), criterion_scale ~ x + (1 | p | sub),
      family = uvsd())
}

.ls_cm_formula <- function() {
  brf(item_resp | is_old ~ 1, criterion_shift ~ 1 + (1 | sub), criterion_scale ~ 0 + cond, family = evsd())
}

test_that("criterion_shift and criterion_scale give one location component, a scale without intercept, and shape gaps", {
  d <- .ls_data()
  m <- suppressMessages(broc(.ls_formula(), d))
  md <- m$model_data
  expect_true(md$has_criterion_scale)
  expect_true(md$criterion$location_scale)
  expect_equal(md$criterion$random$sub$dim, 1)
  expect_identical(unname(md$criterion$coef_names), c("(Intercept)", "condb"))
  expect_identical(unname(md$criterion_scale_fixed$coef_names), "x")
  expect_false(md$criterion_scale_fixed$normalised_shape)
  cfg <- build_numpyro_config(m)
  expect_true(cfg$crit_ls)
  expect_null(cfg$crit_ls_norm)
  expect_true(isTRUE(cfg$random_effects[[which(vapply(cfg$random_effects, `[[`, "", "param") == "criterion")]]$location_only))
  tab <- get_broc_prior(m)
  expect_setequal(tab$coef[tab$dpar == "criterion_shift" & tab$class == "b"], c("(Intercept)", "condb"))
  expect_setequal(tab$coef[tab$dpar == "log_gaps" & tab$class == "b"], paste0("gap", 1:4))
  expect_false(any(tab$dpar == "thresh_mid"))
  m0 <- suppressMessages(broc(brf(item_resp | is_old ~ 1, criterion_shift ~ cond, family = evsd()), d))
  expect_true(m0$model_data$has_criterion_scale)
  expect_equal(m0$model_data$criterion_scale_fixed$n_coef, 0)
  m1 <- suppressMessages(broc(brf(item_resp | is_old ~ 1, criterion ~ 1, criterion_scale ~ x, family = evsd()), d))
  expect_identical(unname(m1$model_data$criterion$coef_names), "(Intercept)")
  d2 <- d; d2$item_resp <- pmin(d2$item_resp, 2L)
  expect_error(broc(brf(item_resp | is_old ~ 1, criterion_scale ~ x, family = evsd()), d2), "at least two criteria")
})

test_that("a criterion formula with terms is refused next to criterion_shift or criterion_scale", {
  d <- .ls_data()
  expect_error(broc(brf(item_resp | is_old ~ 1, criterion ~ cond, criterion_scale ~ x, family = evsd()), d),
               "criterion_shift ~")
  expect_error(broc(brf(item_resp | is_old ~ 1, criterion ~ 1 + (1 | sub), criterion_shift ~ cond, family = evsd()), d),
               "baseline set of criteria")
  expect_error(broc(brf(item_resp ~ 1, cutpoints ~ cond, cutpoints_scale ~ x, family = cumulative()), d),
               "cutpoints_shift ~")
  p <- broc_prior("normal(0, 0.3)", class = "b", dpar = "criterion_shift", coef = "condb")
  m <- suppressMessages(broc(brf(item_resp | is_old ~ 1, criterion_shift ~ cond, family = evsd()), d, priors = p))
  tab <- get_broc_prior(m)
  expect_identical(tab$prior[tab$dpar == "criterion_shift" & tab$coef == "condb"], "normal(0, 0.3)")
})

test_that("criterion_scale ~ 0 + cond gives each level its own spread and a shape whose gaps sum to zero", {
  d <- .ls_data()
  m <- suppressMessages(broc(.ls_cm_formula(), d))
  md <- m$model_data
  expect_true(md$criterion_scale_fixed$normalised_shape)
  expect_identical(unname(md$criterion_scale_fixed$coef_names), c("conda", "condb"))
  expect_true(build_numpyro_config(m)$crit_ls_norm)
  tab <- get_broc_prior(m)
  expect_identical(unique(tab$prior[tab$dpar == "criterion_scale" & tab$class == "b"]), "normal(-0.5, 0.5)")
  expect_identical(unique(tab$prior[tab$dpar == "log_gaps" & tab$class == "b"]), "normal(0, 0.5)")
  mb <- suppressMessages(broc(brf(item_resp | is_old ~ 1, criterion_shift ~ 0 + cond, criterion_scale ~ 0 + cond,
                                  family = evsd()), d))
  expect_identical(unname(mb$model_data$criterion$coef_names), c("conda", "condb"))
  expect_identical(unname(mb$model_data$criterion_scale_fixed$coef_names), c("conda", "condb"))
  expect_true(mb$model_data$criterion_scale_fixed$normalised_shape)
  mx <- suppressMessages(broc(brf(item_resp | is_old ~ 1, criterion_scale ~ 0 + x, family = evsd()), d))
  expect_false(mx$model_data$criterion_scale_fixed$normalised_shape)
  d3 <- d; d3$item_resp <- pmin(d3$item_resp, 3L)
  m3 <- suppressMessages(broc(brf(item_resp | is_old ~ 1, criterion_scale ~ 0 + cond, family = evsd()), d3))
})

test_that("location-scale thresholds: R reproduces the sampler's log-likelihood on both gap links", {
  skip_if_no_jax()
  d <- .ls_data()
  both_cm <- brf(item_resp | is_old ~ 1, criterion_shift ~ 0 + cond, criterion_scale ~ 0 + cond, family = evsd())
  for (gl in c("log", "softplus")) {
    for (f in list(.ls_formula(), .ls_cm_formula(), both_cm)) {
      m <- suppressMessages(broc(f, d, gap_link = if (gl == "softplus") "softplus"))
      fit <- suppressMessages(fit_broc(m, sampler = "hmc", chains = 1, iter_warmup = 50,
                                       iter_sampling = 20, seed = 1, progress_bar = FALSE))
      dm <- fit$draws(format = "matrix")
      ll_r <- .obs_loglik_r(m, build_data_source_from_model(m), dm, re_formula = NULL)
      ll_j <- .read_loglik_SN_cols(.loglik_path(fit), 1L, m$model_data$N)
      expect_lt(max(abs(ll_r - ll_j)), 1e-5, label = paste(gl, "JAX vs R log_lik"))
      if (isTRUE(m$model_data$criterion_scale_fixed$normalised_shape)) {
        gp <- if (gl == "softplus") "beta_raw_gaps" else "beta_log_gaps"
        g <- dm[, grep(paste0("^", gp, "\\["), colnames(dm)), drop = FALSE]
        expect_equal(ncol(g), 4L)
        expect_lt(max(abs(rowSums(g))), 1e-10, label = paste(gl, "shape gaps sum to zero"))
      }
    }
  }
})

.ls2_cases <- function() {
  set.seed(4)
  bv <- .fixture_bivariate_sdt(n_per = 150)
  bv$x <- rnorm(nrow(bv)); bv$sub <- factor(sample(1:10, nrow(bv), TRUE))
  bv$g <- factor(ifelse(as.integer(bv$sub) %% 2 == 0, "E1", "E2"))
  bc <- .fixture_bivariate_cumulative(n = 400); bc$x <- rnorm(nrow(bc)); bc$sub <- factor(sample(1:10, nrow(bc), TRUE))
  list(
    varying = list(brf(resp(y1, y2) | item_type ~ 1, dprime2 ~ 1, discrim ~ 1, lambda ~ 1, criterion ~ 1,
                       criterion2_shift ~ 1 + x + (1 | sub), criterion2_scale ~ x + (1 | sub),
                       family = vrdp2d(source_criteria = "varying")), bv),
    cumulative = list(brf(resp(y1, y2) ~ 1, cutpoints ~ cond, cutpoints2_shift ~ cond + (1 | sub), cutpoints2_scale ~ x,
                          rho ~ 1, family = bivariate_cumulative()), bc),
    shared_cellmeans = list(brf(resp(y1, y2) | item_type ~ 1, discrim ~ 1, rho ~ 1, criterion ~ 1,
                                criterion2_scale ~ 0 + g, family = bivariate_gaussian()), bv),
    varying_cellmeans = list(brf(resp(y1, y2) | item_type ~ 1, dprime2 ~ 1, discrim ~ 1, lambda ~ 1, criterion ~ 1,
                                 criterion2_scale ~ 0 + g, family = vrdp2d(source_criteria = "varying")), bv)
  )
}

test_that("criterion2_scale: one scale per response bin, location-only source criteria", {
  cs <- .ls2_cases()
  m <- suppressMessages(broc(cs$varying[[1]], cs$varying[[2]]))
  md <- m$model_data
  expect_true(md$has_criterion2_scale && md$criterion2$location_scale)
  expect_length(md$criterion2_scale_fixed$coef_names, 4L)
  expect_identical(md$criterion2$random$sub$varying_re_mode, "per_bin")
  mv <- suppressMessages(broc(cs$varying_cellmeans[[1]], cs$varying_cellmeans[[2]]))
  expect_true(mv$model_data$criterion2_scale_fixed$normalised_shape)
  expect_length(mv$model_data$criterion2_scale_fixed$coef_names, 8L)
  ms <- suppressMessages(broc(cs$shared_cellmeans[[1]], cs$shared_cellmeans[[2]]))
  expect_true(ms$model_data$criterion2_scale_fixed$normalised_shape)
})

test_that("criterion2_scale: R reproduces the JAX log-likelihood", {
  skip_if_no_jax()
  for (nm in names(cs <- .ls2_cases())) {
    m <- suppressMessages(broc(cs[[nm]][[1]], cs[[nm]][[2]]))
    f <- suppressMessages(fit_broc(m, sampler = "hmc", chains = 1, iter_warmup = 50,
                                   iter_sampling = 10, seed = 1, progress_bar = FALSE))
    ll_r <- .obs_loglik_r(m, build_data_source_from_model(m), f$draws(format = "matrix"), re_formula = NULL)
    expect_lt(max(abs(ll_r - .read_loglik_SN_cols(.loglik_path(f), 1L, m$model_data$N))), 1e-5, label = nm)
  }
})

test_that("criterion2_shift with cell-means varying source criteria: R reproduces the JAX log-likelihood", {
  skip_if_no_jax()
  bv <- .ls2_cases()$varying[[2]]
  m <- suppressMessages(broc(brf(resp(y1, y2) | item_type ~ 1, dprime2 ~ 1, discrim ~ 1, lambda ~ 1, criterion ~ 1,
                                 criterion2_shift ~ 0 + g + (1 | sub), criterion2_scale ~ x,
                                 family = vrdp2d(source_criteria = "varying")), bv))
  expect_equal(m$stan_data$n_vcell, 2L)
  f <- suppressMessages(fit_broc(m, sampler = "hmc", chains = 1, iter_warmup = 50,
                                 iter_sampling = 10, seed = 1, progress_bar = FALSE))
  ll_r <- .obs_loglik_r(m, build_data_source_from_model(m), f$draws(format = "matrix"), re_formula = NULL)
  expect_lt(max(abs(ll_r - .read_loglik_SN_cols(.loglik_path(f), 1L, m$model_data$N))), 1e-5)
})
