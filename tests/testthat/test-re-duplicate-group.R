re_dup_data <- function() {
  d <- .fixture_crossed()
  set.seed(11)
  d$x <- round(rnorm(nrow(d)), 2)
  d$z <- round(rnorm(nrow(d)), 2)
  d
}

.blocks_of <- function(f, d, pname = "criterion") {
  m <- suppressMessages(broc(f, data = d))
  re <- if (pname == "criterion") m$model_data$criterion$random$sub else m$model_data[[paste0(pname, "_random")]]$sub
  lapply(re$cor_blocks, as.integer)
}

test_that("threshold random effects: | correlates everything, || splits by term, gr() splits by threshold", {
  d <- re_dup_data()
  crit <- function(rhs) brf(item_resp | is_old ~ 1, rhs, family = evsd())
  K1 <- 5L
  expect_equal(.blocks_of(crit(criterion ~ 1 + x + (1 + x | sub)), d), list(1:(2 * K1)))
  by_term <- list(seq(1L, 2 * K1, 2), seq(2L, 2 * K1, 2))
  expect_equal(.blocks_of(crit(criterion ~ 1 + x + (1 + x || sub)), d), by_term)
  expect_equal(.blocks_of(crit(criterion ~ 1 + x + (1 | sub) + (0 + x | sub)), d), by_term)
  expect_equal(.blocks_of(crit(criterion ~ 1 + x + (1 + x | gr(sub, cor_thresholds = FALSE))), d),
               lapply(seq_len(K1), function(k) c(2L * k - 1L, 2L * k)))
  expect_length(.blocks_of(crit(criterion ~ 1 + x + (1 + x || gr(sub, cor_thresholds = FALSE))), d), 0)
  expect_equal(.blocks_of(crit(criterion ~ 1 + (1 | gr(sub, cor_thresholds = TRUE))), d), list(1:K1))
})

test_that("several terms for one grouping factor merge into block-diagonal correlations", {
  d <- re_dup_data()
  expect_equal(.blocks_of(brf(item_resp | is_old ~ 1 + (1 + x | sub) + (0 + z | sub), family = evsd()), d,
                          "dprime"), list(1:2))
  m1 <- suppressMessages(broc(brf(item_resp | is_old ~ 1 + (1 + x || sub), family = evsd()), data = d))
  m2 <- suppressMessages(broc(brf(item_resp | is_old ~ 1 + (1 | sub) + (0 + x | sub), family = evsd()), data = d))
  expect_identical(m1$stan_code, m2$stan_code)
  m3 <- suppressMessages(broc(brf(item_resp | is_old ~ 1 + (1 + x | sub) + (0 + z | sub), family = evsd()),
                              data = d))
  expect_match(m3$stan_code, "cholesky_factor_corr[2] L_corr_dprime_sub_b1;", fixed = TRUE)
  expect_match(m3$stan_code, "L_corr_dprime_sub[{1, 2}, {1, 2}] = L_corr_dprime_sub_b1;", fixed = TRUE)
})

test_that("invalid grouping options and combinations are rejected", {
  d <- re_dup_data()
  expect_error(broc(brf(item_resp | is_old ~ 1 + (1 | gr(sub, cor_thresholds = FALSE)), family = evsd()), d),
               "applies only to threshold parameters")
  expect_error(broc(brf(item_resp | is_old ~ 1, criterion ~ 1 + (1 | gr(sub, foo = 1)), family = evsd()), d),
               "accepts only the named option cor_thresholds")
  expect_error(broc(brf(item_resp | is_old ~ 1 + (1 | c | sub) + (0 + x | sub), sigma ~ 1 + (1 | c | sub),
                        family = uvsd()), d), "correlation group must be the only term")
  expect_error(broc(brf(item_resp | is_old ~ 1 + (1 | c | sub),
                        criterion ~ 1 + (1 | c | gr(sub, cor_thresholds = FALSE)), family = evsd()), d),
               "cannot be combined with a cross-parameter")
})

test_that("distinct grouping factors and cross-parameter ids still build", {
  d <- re_dup_data()
  expect_s3_class(
    broc(brf(item_resp | is_old ~ 1 + (1 | sub) + (1 | item), family = evsd()), data = d),
    "broc_model"
  )
  expect_s3_class(
    broc(brf(item_resp | is_old ~ 1 + (1 | p | sub) + (1 | i | item),
             sigma ~ 1 + (1 | p | sub) + (1 | i | item),
             criterion ~ 1 + (1 | p | sub) + (1 | i | item),
             family = uvsd()), data = d),
    "broc_model"
  )
})

test_that("block-diagonal threshold correlations: Stan and JAX agree and cross-block correlations are zero", {
  skip_if_no_cmdstan(); skip_if_no_jax()
  d <- .fixture_crossed(n_sub = 12, n_item = 24)
  set.seed(3)
  d$x <- round(rnorm(nrow(d)), 2)
  m <- suppressMessages(broc(brf(item_resp | is_old ~ 1,
                                 criterion ~ 1 + x + (1 + x | gr(sub, cor_thresholds = FALSE)),
                                 family = evsd()), data = d))
  a <- list(chains = 4, parallel_chains = 4, iter_warmup = 1000, iter_sampling = 1000, seed = 2)
  fj <- do.call(fit_broc, c(list(m, backend = "jax", progress_bar = FALSE), a))
  fs <- do.call(fit_broc, c(list(m, backend = "stan", refresh = 50), a))
  params <- function(f) grep("^(beta_|sigma_|corr_|L_corr_)",
                             unique(sub("\\[.*$", "", posterior::variables(f$draws()))), value = TRUE)
  expect_setequal(params(fj), params(fs))
  expect_true(all(c("L_corr_criterion_sub", "L_corr_criterion_sub_b1", "L_corr_criterion_sub_b5") %in%
                    params(fj)))
  expect_equal(max(abs(posterior::extract_variable(fj$draws(variables = "corr_criterion_sub"),
                                                   "corr_criterion_sub[1,3]"))), 0)
  v <- c("beta_thresh_mid[1]", "beta_thresh_mid[2]", "sigma_criterion_sub[1]", "sigma_criterion_sub[2]",
         "corr_criterion_sub[1,2]", "corr_criterion_sub[3,4]")
  sj <- posterior::summarise_draws(fj$draws(variables = v), "mean", "sd", "ess_bulk")
  ss <- posterior::summarise_draws(fs$draws(variables = v), "mean", "sd", "ess_bulk")
  z <- abs(sj$mean - ss$mean) / sqrt(sj$sd^2 / sj$ess_bulk + ss$sd^2 / ss$ess_bulk)
  expect_lt(max(z), 4)
})
