# Helpers shared with tests/testthat/
# test-criterion-scale.R (copied).

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

test_that("location-scale criteria agree across plain Stan, batch Stan and R", {
  skip_if_no_cmdstan()
  d <- .ls_data()
  cases <- list(
    log = list(.ls_formula(), d, NULL),
    softplus = list(.ls_formula(), d, "softplus"),
    cellmeans = list(.ls_cm_formula(), d, NULL),
    cellmeans_softplus = list(.ls_cm_formula(), d, "softplus"),
    both_cellmeans = list(brf(item_resp | is_old ~ 1, criterion_shift ~ 0 + cond, criterion_scale ~ 0 + cond,
                              family = evsd()), d, NULL),
    cumulative = list(brf(item_resp ~ 1, cutpoints_shift ~ cond + (1 | sub), cutpoints_scale ~ x, family = cumulative()), d, NULL),
    cumulative_cellmeans = list(brf(item_resp ~ 1, cutpoints_scale ~ 0 + cond, family = cumulative()), d, NULL)
  )
  for (nm in names(cases)) {
    c1 <- cases[[nm]]
    mb <- suppressMessages(broc(c1[[1]], c1[[2]], gap_link = c1[[3]]))
    mp <- suppressMessages(.broc_plain(c1[[1]], c1[[2]], gap_link = c1[[3]]))
    r <- .stan_parity(mp, mb)
    expect_lt(r$lp, 1e-8, label = paste(nm, "plain vs batch log density"))
    expect_lt(r$grad, 1e-8, label = paste(nm, "plain vs batch gradient"))
    f <- suppressMessages(stan_fit_broc(mb, chains = 1, iter_warmup = 100, iter_sampling = 3,
                                   seed = 1, refresh = 50, show_messages = FALSE))
    dm <- posterior::as_draws_matrix(f$draws())
    ll_stan <- as.matrix(dm[, grep("^log_lik\\[", colnames(dm)), drop = FALSE])
    if (!ncol(ll_stan)) ll_stan <- .read_loglik_SN_cols(.loglik_path(f), 1L, mb$model_data$N)
    ll_r <- .obs_loglik_r(mb, build_data_source_from_model(mb), dm, re_formula = NULL)
    expect_lt(max(abs(ll_stan - ll_r)), 1e-5, label = paste(nm, "Stan vs R log_lik"))
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
    f <- suppressMessages(stan_fit_broc(mb, chains = 1, iter_warmup = 100, iter_sampling = 5, seed = 1,
                                   refresh = 50, show_messages = FALSE))
    ll_r <- .obs_loglik_r(mb, build_data_source_from_model(mb), f$draws(format = "matrix"), re_formula = NULL)
    expect_lt(max(abs(ll_r - .read_loglik_SN_cols(.loglik_path(f), 1L, mb$model_data$N))), 1e-5, label = nm)
  }
})
