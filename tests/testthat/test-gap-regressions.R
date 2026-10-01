.gap_univariate <- function() {
  set.seed(11)
  d <- .fixture_univariate(sigma = 1.3, n_per = 60)
  d$sub <- factor(rep(1:6, length.out = nrow(d)))
  d$x <- round(rnorm(nrow(d)), 2)
  d$ind <- as.numeric(sample(0:1, nrow(d), replace = TRUE))
  d$cond <- factor(ifelse(d$ind == 1, "b", "a"))
  d
}

.gap_bivariate <- function() {
  set.seed(12)
  d <- .fixture_bivariate_sdt(n_per = 50)
  d$sub <- factor(rep(1:6, length.out = nrow(d)))
  d$x <- round(rnorm(nrow(d)), 2)
  d$ind <- as.numeric(sample(0:1, nrow(d), replace = TRUE))
  d
}

.gap_cdp <- function(n_sub = 6, n_item = 40) {
  set.seed(13)
  d <- expand.grid(item = seq_len(n_item), sub = seq_len(n_sub))
  d$old <- as.integer(d$item <= n_item / 2)
  n <- nrow(d)
  R <- ifelse(d$old == 1L, rnorm(n, 1, 1.25), rnorm(n))
  F_ <- ifelse(d$old == 1L, rnorm(n, 0.8, 1), rnorm(n))
  d$conf <- as.integer(1L + rowSums(outer(R + F_, c(0.3, 1.0, 1.7), ">")))
  rk <- ifelse(R > 1.2, 1L, ifelse(F_ > 0.3, 2L, 3L))
  d$rk <- ifelse(d$conf >= 2L, rk, NA_integer_)
  d$sub <- factor(d$sub)
  d$x <- round(rnorm(n), 2)
  d$ind <- as.numeric(d$x > 0)
  d
}

.gap_cases <- function() {
  du <- .gap_univariate(); db <- .gap_bivariate(); dc <- .gap_cdp()
  list(
    cumulative_mu_slope = list(brf(item_resp ~ cond + (1 + cond | sub), cutpoints ~ 1 + (1 | sub),
                                   family = cumulative()), du, "u_mu_sub", du$cond == "a"),
    criterion_indicator = list(brf(item_resp | is_old ~ 1, criterion ~ 1 + ind + (0 + ind | sub),
                                   family = evsd()), du, "u_criterion_sub", du$ind == 0),
    criterion2_indicator_per_bin = list(brf(resp(y1, y2) | item_type ~ 1, discrim ~ 1, criterion ~ 1,
                                            criterion2 ~ 1 + ind + (0 + ind | sub), rho ~ 1,
                                            family = bivariate_gaussian(varying_source_criteria = TRUE,
                                                                        varying_re = "per_bin")),
                                        db, "u_criterion2_sub", db$ind == 0),
    criterion2_slope_full = list(brf(resp(y1, y2) | item_type ~ 1, discrim ~ 1, criterion ~ 1,
                                     criterion2 ~ 1 + x + (1 + x | sub), rho ~ 1,
                                     family = bivariate_gaussian(varying_source_criteria = TRUE,
                                                                 varying_re = "full")),
                                 db, NULL, NULL),
    cdp_criterion_slope = list(brf(resp(conf, rk) | old ~ 1, fam ~ 1, criterion ~ 1 + x + (1 + x | sub),
                                   rec_crit ~ 1, know_crit ~ 1, sigma_R ~ 1, family = cdp(old_levels = 2:4)),
                               dc, NULL, NULL)
  )
}

test_that("slope-only random effects agree across plain Stan, batch Stan and R, and touch only their rows", {
  skip_if_no_cmdstan()
  for (nm in names(cs <- .gap_cases())) {
    c1 <- cs[[nm]]
    mb <- suppressMessages(broc(c1[[1]], c1[[2]]))
    mp <- suppressMessages(.broc_plain(c1[[1]], c1[[2]]))
    r <- .stan_parity(mp, mb)
    expect_lt(r$lp, 1e-8, label = paste(nm, "plain vs batch log density"))
    expect_lt(r$grad, 1e-8, label = paste(nm, "plain vs batch gradient"))

    f <- suppressMessages(fit_broc(mb, backend = "stan", chains = 1, iter_warmup = 100, iter_sampling = 3,
                                   seed = 1, refresh = 50, show_messages = FALSE))
    dm <- posterior::as_draws_matrix(f$draws())
    src <- build_data_source_from_model(mb)
    ll_stan <- as.matrix(dm[, grep("^log_lik\\[", colnames(dm)), drop = FALSE])
    ll_r <- .obs_loglik_r(mb, src, dm, re_formula = NULL)
    expect_lt(max(abs(ll_stan - ll_r)), 1e-5, label = paste(nm, "Stan vs R log_lik"))

    if (!is.null(c1[[3]])) {
      cols <- grep(paste0("^", c1[[3]], "\\["), colnames(dm), value = TRUE)
      p <- dm[1, , drop = FALSE]; p[1, cols] <- p[1, cols] + 0.3
      ch <- abs(.obs_loglik_r(mb, src, p, NULL) - .obs_loglik_r(mb, src, dm[1, , drop = FALSE], NULL))[1, ]
      expect_true(all(ch[c1[[4]]] < 1e-12), label = paste(nm, "reference rows untouched"))
      expect_true(any(ch[!c1[[4]]] > 1e-6), label = paste(nm, "slope rows respond"))
    }
    expect_equal(predict(f, summary = FALSE), predict(f, newdata = c1[[2]], summary = FALSE),
                 label = paste(nm, "newdata rebuild"))
  }
})

test_that("slope-only random effects: JAX log_lik matches R", {
  skip_if_no_jax()
  cs <- .gap_cases()[c("cumulative_mu_slope", "criterion_indicator", "criterion2_indicator_per_bin")]
  for (nm in names(cs)) {
    c1 <- cs[[nm]]
    m <- suppressMessages(broc(c1[[1]], c1[[2]]))
    f <- suppressMessages(fit_broc(m, backend = "jax", chains = 1, iter_warmup = 50, iter_sampling = 2,
                                   seed = 1, progress_bar = FALSE))
    dm <- posterior::as_draws_matrix(f$draws())
    ll_j <- as.matrix(posterior::as_draws_matrix(f$draws(variables = "log_lik")))
    expect_lt(max(abs(ll_j - .obs_loglik_r(m, build_data_source_from_model(m), dm, NULL))), 1e-5, label = nm)
  }
})

test_that("every listed prior can be set, is echoed back, and reaches Stan and the JAX config", {
  du <- .gap_univariate(); db <- .gap_bivariate(); dc <- .gap_cdp()
  specs <- list(
    biv_crit2_slope = list(brf(resp(y1, y2) | item_type ~ 1 + (1 | sub), discrim ~ 1, criterion ~ 1 + (1 | sub),
                               criterion2 ~ 1 + x + (1 + x | sub), rho ~ x, family = bivariate_gaussian()), db),
    varying_full = list(brf(resp(y1, y2) | item_type ~ 1, discrim ~ 1, criterion ~ 1,
                            criterion2 ~ 1 + x + (1 + x | sub), rho ~ 1,
                            family = bivariate_gaussian(varying_source_criteria = TRUE, varying_re = "full",
                                                        new_source_criteria = "shared", old_levels = 3:4)), db),
    cumulative = list(brf(item_resp ~ cond + (1 + cond | sub), cutpoints ~ 1 + x + (1 | sub),
                          family = cumulative()), du),
    cross = list(brf(item_resp | is_old ~ 1 + (1 | p | sub), sigma ~ 1 + (1 | p | sub),
                     criterion ~ 1 + (1 | p | sub), family = uvsd()), du),
    mixture = list(brf(item_resp | is_old ~ cond, dprime2 ~ cond, lambda ~ 1, family = mixture()), du),
    smooth = list(brf(item_resp | is_old ~ 1 + s(x, k = 5), criterion ~ 1 + ind + (0 + ind | sub),
                      family = evsd()), du),
    cdp = list(brf(resp(conf, rk) | old ~ 1 + (1 | sub), fam ~ 1, criterion ~ 1 + x + (1 + x | sub),
                   rec_crit ~ 1, know_crit ~ x, sigma_R ~ 1, family = cdp(old_levels = 2:4)), dc)
  )
  norm <- function(x) gsub("\\s+", "", x)
  strings <- function(x) if (is.character(x)) x else if (is.list(x)) unlist(lapply(x, strings)) else character(0)
  for (nm in names(specs)) {
    f <- specs[[nm]][[1]]; d <- specs[[nm]][[2]]
    pt <- get_broc_prior(suppressMessages(broc(f, d)))
    want <- ifelse(pt$class == "cor", sprintf("lkj_corr_cholesky(1.%03d)", seq_len(nrow(pt))),
                   sprintf("normal(0, 0.%03d)", 100 + seq_len(nrow(pt))))
    pri <- do.call(c, lapply(seq_len(nrow(pt)), function(i)
      broc_prior(want[i], class = pt$class[i], dpar = pt$dpar[i],
                 coef = if (is.na(pt$coef[i])) NULL else pt$coef[i],
                 group = if (is.na(pt$group[i])) NULL else pt$group[i])))
    m <- suppressMessages(broc(f, d, priors = pri))
    pt2 <- get_broc_prior(m)
    expect_identical(norm(pt2$prior), norm(want), label = paste(nm, "listing echo"))
    expect_true(all(pt2$source == "user"), label = paste(nm, "listing source"))
    code <- norm(m$stan_code)
    expect_true(all(vapply(norm(want), grepl, logical(1), x = code, fixed = TRUE)), label = paste(nm, "Stan"))
    cfg <- norm(strings(build_numpyro_config(m)))
    expect_true(all(vapply(norm(want), function(w) any(grepl(w, cfg, fixed = TRUE)), logical(1))),
                label = paste(nm, "JAX config"))
  }
})

test_that("threshold and cross-group dpars from get_broc_prior are accepted by broc_prior", {
  expect_s3_class(broc_prior("normal(0, 1)", class = "b", dpar = "cutpoints"), "broc_prior")
  expect_s3_class(broc_prior("normal(0, 1)", class = "b", dpar = "cutpoints2"), "broc_prior")
  expect_s3_class(broc_prior("lkj_corr_cholesky(2)", class = "cor", dpar = "cross_p", group = "sub"), "broc_prior")
})

test_that("criterion2 slope effects under varying source criteria keep their slope structure", {
  db <- .gap_bivariate()
  for (mode in c("shared", "per_bin", "full")) {
    m <- suppressMessages(broc(brf(resp(y1, y2) | item_type ~ 1, discrim ~ 1, criterion ~ 1,
                                   criterion2 ~ 1 + x + (1 + x | sub), rho ~ 1,
                                   family = bivariate_gaussian(varying_source_criteria = TRUE, varying_re = mode)), db))
    re <- m$model_data$criterion2$random$sub
    n_base <- switch(mode, shared = 1L, per_bin = m$model_data$K, full = m$model_data$K * (m$model_data$K2 - 1L))
    expect_identical(as.integer(re$dim), as.integer(n_base * 2L), label = mode)
    expect_true(isTRUE(re$use_z_matrix), label = mode)
  }
})
