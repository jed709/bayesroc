fixture_cdp_judged <- function(n_rkg = 3L, seed = 4, n_sub = 8, n_item = 40, drop = 0, new_judged = FALSE) {
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
  if (!new_judged) rows$rk[rows$conf < 3L] <- NA_integer_
  if (drop > 0) {
    old_rows <- which(rows$conf >= 3L)
    rows$rk[sample(old_rows, round(drop * length(old_rows)))] <- NA_integer_
  }
  rows$sub <- factor(rows$sub)
  rows
}

cdp_judged_formula <- function(n_rkg, family = cdp(old_levels = 3:4)) {
  if (n_rkg == 3L) brf(resp(conf, rk) | old ~ 1, fam ~ 1, criterion ~ 1, rec_crit ~ 1, know_crit ~ 1,
                       sigma_R ~ 1, family = family)
  else brf(resp(conf, rk) | old ~ 1, fam ~ 1, criterion ~ 1, rec_crit ~ 1, sigma_R ~ 1, family = family)
}

test_that("judgements on responses outside old_levels are ignored, with a message", {
  d <- fixture_cdp_judged(3L, new_judged = TRUE)
  expect_message(m <- broc(cdp_judged_formula(3L), data = d), "not part of the model")
  expect_identical(m$family$old_levels, 3:4)
  expect_identical(m$family$rk_levels, 3:4)
  expect_true(all(m$stan_data$rk[d$conf < 3L] == 0L))
  d_na <- d
  d_na$rk[d_na$conf < 3L] <- NA_integer_
  m_na <- suppressMessages(broc(cdp_judged_formula(3L), data = d_na))
  expect_identical(m$stan_data, m_na$stan_data)
  expect_message(m_inf <- broc(cdp_judged_formula(3L, cdp()), data = d), "inferred")
  expect_identical(m_inf$family$old_levels, 3:4)
  m_old <- suppressMessages(broc(cdp_judged_formula(3L, cdp()), data = d_na))
  expect_identical(m_old$family$old_levels, 3:4)
})

test_that("missing judgements on old responses enter as confidence only", {
  d <- fixture_cdp_judged(2L, drop = 0.2)
  m <- suppressMessages(broc(cdp_judged_formula(2L), data = d))
  expect_identical(m$stan_data$rk == 0L, is.na(d$rk))
  expect_identical(m$family$old_levels, 3:4)
})
