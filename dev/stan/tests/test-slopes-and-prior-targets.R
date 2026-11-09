# Helpers shared with tests/testthat/
# test-slopes-and-prior-targets.R (copied).

.gap_univariate <- function() {
  set.seed(11)
  d <- .fixture_univariate(sigma = 1.3, n_per = 60)
  d$sub <- factor(rep(1:6, length.out = nrow(d)))
  d$x <- round(rnorm(nrow(d)), 2)
  d$ind <- as.numeric(sample(0:1, nrow(d), replace = TRUE))
  d$cond <- factor(ifelse(d$ind == 1, "b", "a"))
  d$grp <- factor(rep(1:5, length.out = nrow(d)))
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
    cumulative_mu_slope = list(brf(item_resp ~ 1, cutpoints_shift ~ cond + (0 + ind | sub) + (1 | grp),
                                   family = cumulative()), du, "u_cutpoints_sub", du$cond == "a"),
    criterion_indicator = list(brf(item_resp | is_old ~ 1, criterion ~ 1 + ind + (0 + ind | sub),
                                   family = evsd()), du, "u_criterion_sub", du$ind == 0),
    criterion2_indicator_per_bin = list(brf(resp(y1, y2) | item_type ~ 1, discrim ~ 1, criterion ~ 1,
                                            criterion2 ~ 1 + ind + (0 + ind | sub), rho ~ 1,
                                            family = bivariate_gaussian(source_criteria = "varying")),
                                        db, "u_criterion2_sub", db$ind == 0),
    criterion2_slope_full = list(brf(resp(y1, y2) | item_type ~ 1, discrim ~ 1, criterion ~ 1,
                                     criterion2 ~ 1 + x + (1 + x | sub), rho ~ 1,
                                     family = bivariate_gaussian(source_criteria = "varying")),
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

    f <- suppressMessages(stan_fit_broc(mb, chains = 1, iter_warmup = 20, iter_sampling = 3,
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
