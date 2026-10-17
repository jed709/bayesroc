.sim_status <- function(seed = 7, n_sub = 10, n_per = 40) {
  set.seed(seed)
  n <- n_sub * n_per
  d <- data.frame(sub = factor(rep(seq_len(n_sub), each = n_per)))
  d$item_type <- factor(sample(c("new", "A", "B"), n, TRUE), levels = c("new", "A", "B"))
  z1 <- rnorm(n, ifelse(d$item_type == "new", 0, 1.2))
  z2 <- rnorm(n, ifelse(d$item_type == "A", -1, ifelse(d$item_type == "B", 1, 0)))
  d$det <- as.integer(1L + rowSums(outer(z1, c(-0.8, 0, 0.8), ">")))
  d$src <- as.integer(1L + rowSums(outer(z2, c(-0.7, 0, 0.7), ">")))
  d
}

test_that("auto-run status labels name parameters as the summary does", {
  m <- suppressMessages(broc(brf(resp(det, src) | item_type ~ 1, discrim ~ 1, rho ~ 1,
                                 criterion ~ 1 + (1 | sub), criterion2 ~ 1 + (1 | sub),
                                 family = bivariate_gaussian(source_criteria = "varying")), .sim_status()))
  v <- c("sigma_criterion2_sub[2]", "beta_log_gaps_2_varying[2,1]", "beta_thresh_mid_2_varying[3]", "lp__")
  lab <- .auto_labels(m, v)
  expect_match(.lab_of(lab, v[1]), "^sub SD of .*bin 1, c2 \\(midpoint\\)")
  expect_match(.lab_of(lab, v[2]), "bin 2, c2 to c3")
  expect_match(.lab_of(lab, v[3]), "bin 3")
  expect_equal(.lab_of(lab, "lp__"), "lp__")
  expect_equal(.lab_of(lab, NULL), "?")
})

test_that("the status line names a parameter only when it alone blocks the stop", {
  skip_if_no_jax()
  out <- system2(jax_python(), c(testthat::test_path("python", "blocker_check.py"), dirname(find_backend_script())),
                 stdout = TRUE, stderr = FALSE)
  expect_equal(tail(out, 1), "OK")
})
