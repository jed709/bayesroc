.honour_data <- function(seed = 1) {
  set.seed(seed)
  d <- expand.grid(subj = factor(1:10), item = factor(1:24))
  n <- nrow(d)
  d$a <- factor(sample(c("a1", "a2"), n, TRUE))
  d$b <- factor(sample(c("b1", "b2"), n, TRUE))
  d$cc <- factor(sample(c("c1", "c2"), n, TRUE))
  d$x <- rnorm(n)
  d$old <- sample(0:1, n, TRUE)
  d$conf <- sample(1:6, n, TRUE)
  d$item_type <- sample(1:3, n, TRUE)
  d$conf2 <- sample(1:6, n, TRUE)
  d
}

.dprime_cols <- function(f, d = .honour_data(), ...) colnames(suppressMessages(broc(f, d, ...))$stan_data$X_dprime)

test_that("* and ^ expand to every lower-order term", {
  expect_identical(.dprime_cols(brf(conf | old ~ a * b * cc, criterion ~ 1, family = evsd())),
                   colnames(model.matrix(~ a * b * cc, .honour_data())))
  expect_identical(.dprime_cols(brf(conf | old ~ (a + b + cc)^2, criterion ~ 1, family = evsd())),
                   colnames(model.matrix(~ (a + b + cc)^2, .honour_data())))
})

test_that("only a top-level - 1 or 0 removes the intercept", {
  d <- .honour_data()
  m <- suppressMessages(broc(brf(conf | old ~ I(x - 1), criterion ~ 1, family = evsd()), d))
  expect_identical(colnames(m$stan_data$X_dprime), c("(Intercept)", "I(x - 1)"))
  expect_equal(unname(m$stan_data$X_dprime[, 2]), d$x - 1)
  expect_identical(.dprime_cols(brf(conf | old ~ x - 1, criterion ~ 1, family = evsd())), "x")
  expect_identical(.dprime_cols(brf(conf | old ~ a + 0 + b, criterion ~ 1, family = evsd())),
                   colnames(model.matrix(~ a + 0 + b, d)))
  m <- suppressMessages(broc(brf(conf | old ~ 1 + (a - 1 | subj), criterion ~ 1, family = evsd()), d))
  expect_identical(m$model_data$dprime_random$subj$level_names, c("aa1", "aa2"))
})

test_that("a correlation label shared by different grouping factors is an error", {
  expect_error(suppressMessages(broc(brf(conf | old ~ 1 + (1 | p | subj), criterion ~ 1 + (1 | p | item),
                                         family = evsd()), .honour_data())),
               "different grouping factors")
})

test_that("lure-only parameters cannot depend on an encoding variable", {
  expect_error(suppressMessages(broc(brf(resp(conf, conf2) | item_type ~ 1, discrim ~ 1, rho ~ 1, rho_N ~ a,
                                         criterion ~ 1, criterion2 ~ 1, family = bivariate_gaussian()),
                                     .honour_data(), encoding_vars = "a")),
               "new items only")
})

test_that("an alias formula is used, and two spellings of one parameter are an error", {
  d <- .honour_data()
  expect_error(suppressMessages(broc(brf(conf | old ~ 1, dprime ~ x, criterion ~ 1, family = evsd()), d)),
               "main formula")
  dc <- .fixture_cdp()
  expect_error(suppressMessages(broc(brf(resp(rating, rk) | is_old ~ 1, fam ~ 1, dprime2 ~ 1, criterion ~ 1,
                                         family = cdp(old_levels = 2:6), counts = "count"), dc)),
               "name the same parameter")
})

test_that("~ NULL drops a parameter's formula for every parameter", {
  d <- .honour_data()
  m0 <- suppressMessages(broc(brf(conf | old ~ 1, criterion ~ 1, family = evsd()), d))
  m1 <- suppressMessages(broc(brf(conf | old ~ 1, criterion ~ 1, criterion_scale ~ NULL, family = evsd()), d))
  expect_identical(m1$stan_code, m0$stan_code)
})

test_that("an SD prior with coef but no dpar reaches only that coefficient", {
  m <- suppressMessages(broc(brf(conf | old ~ 1 + (1 + x | subj), sigma ~ 1 + (1 + x | subj), criterion ~ 1,
                                 family = uvsd()), .honour_data(),
                             priors = broc_prior("normal(0, 0.05)", class = "sd", group = "subj", coef = "x")))
  p <- get_broc_prior(m)
  p <- p[p$class == "sd" & p$group == "subj", ]
  expect_identical(p$prior[p$coef == "x"], rep("normal(0, 0.05)", 2))
  expect_false(any(p$prior[p$coef == "(Intercept)"] == "normal(0, 0.05)"))
  expect_match(m$stan_code, "sigma_dprime_subj[2] ~ normal(0, 0.05);", fixed = TRUE)
  default <- p$prior[p$dpar == "dprime" & p$coef == "(Intercept)"]
  expect_identical(p$source[p$dpar == "dprime" & p$coef == "(Intercept)"], "default")
  expect_match(m$stan_code, paste0("sigma_dprime_subj[1] ~ ", default, ";"), fixed = TRUE)
})

test_that("varying criterion2_scale: a factor keeps its reference level in every set, separate terms stay separate", {
  bv <- .fixture_bivariate_sdt()
  set.seed(9)
  bv$a <- factor(sample(c("a1", "a2"), nrow(bv), TRUE)); bv$x <- rnorm(nrow(bv))
  bv$sub <- factor(sample(1:10, nrow(bv), TRUE))
  m <- suppressMessages(broc(brf(resp(y1, y2) | item_type ~ 1, discrim ~ 1, rho ~ 1, criterion ~ 1,
                                 criterion2 ~ 1, criterion2_scale ~ a + (1 | sub) + (0 + x | sub),
                                 family = bivariate_gaussian(source_criteria = "varying")), bv))
  K1 <- m$stan_data$K1
  expect_identical(m$model_data$criterion2_scale_fixed$coef_names, paste0(".c2bin", seq_len(K1), ":aa2"))
  re <- m$model_data$criterion2_scale_random$sub
  expect_length(re$cor_blocks, 2L)
})

test_that("a smooth in lambda2_B is kept", {
  bv <- .fixture_bivariate_sdt()
  set.seed(10); bv$x <- rnorm(nrow(bv))
  m <- suppressMessages(broc(brf(resp(y1, y2) | item_type ~ 1, discrim ~ 1, rho ~ 1, criterion ~ 1, criterion2 ~ 1,
                                 lambda ~ 1, lambda2 ~ 1, lambda2_B ~ 1 + s(x, k = 5),
                                 family = bivariate_dp()), bv))
  expect_false(is.null(m$model_data$smooth_data$lambda2_B))
})

.l2_smooth_model <- function(batch = TRUE) {
  bv <- .fixture_bivariate_sdt()
  set.seed(10); bv$x <- rnorm(nrow(bv))
  .broc_with(batch, brf(resp(y1, y2) | item_type ~ 1, discrim ~ 1, rho ~ 1, criterion ~ 1, criterion2 ~ 1,
                        lambda ~ 1, lambda2 ~ 1 + s(x, k = 5), lambda2_B ~ 1 + s(x, k = 5),
                        family = bivariate_dp()), bv)
}

test_that("lambda2 and lambda2_B smooths: batch and plain Stan agree", {
  skip_if_no_cmdstan()
  r <- .stan_parity(suppressMessages(.l2_smooth_model(FALSE)), suppressMessages(.l2_smooth_model(TRUE)))
  expect_equal(r$dim[1], r$dim[2])
  expect_lt(r$lp, 1e-8)
  expect_lt(r$grad, 1e-8)
})

test_that("lambda2 and lambda2_B smooths: R reproduces the JAX log-likelihood", {
  skip_if_no_jax()
  m <- suppressMessages(.l2_smooth_model())
  f <- suppressMessages(fit_broc(m, backend = "jax", sampler = "drghmc", chains = 1, iter_warmup = 50,
                                 iter_sampling = 5, seed = 1, progress_bar = FALSE))
  ll_r <- .obs_loglik_r(m, build_data_source_from_model(m), f$draws(format = "matrix"), re_formula = NULL)
  expect_lt(max(abs(ll_r - .read_loglik_SN_cols(.loglik_path(f), 1L, m$model_data$N))), 1e-5)
})

test_that("interaction and nested grouping factors are built as their own groups", {
  set.seed(2)
  d <- expand.grid(subj = factor(1:6), item = factor(1:10), rep = 1:2)
  d$old <- sample(0:1, nrow(d), TRUE); d$conf <- sample(1:6, nrow(d), TRUE); d$x <- rnorm(nrow(d))
  m <- suppressMessages(broc(brf(conf | old ~ 1 + (1 | p | subj/item), criterion ~ 1 + (1 | p | subj/item),
                                 family = evsd()), d))
  re <- m$model_data$dprime_random
  expect_identical(names(re), c("subj", "subj_item"))
  expect_equal(c(re$subj$n_groups, re$subj_item$n_groups), c(6L, 60L))
  expect_setequal(names(m$model_data$cross_cor), c("p", "p_subj_item"))
  mi <- suppressMessages(broc(brf(conf | old ~ 1 + (1 | subj:item), criterion ~ 1, family = evsd()), d))
  expect_identical(names(mi$model_data$dprime_random), "subj_item")
  nd <- d[c(5, 17, 33), c("subj", "item", "old", "conf", "x")]
  src <- rebuild_design_matrices(mi, nd)
  expect_equal(src$dprime_random$subj_item$group_idx,
               mi$model_data$dprime_random$subj_item$group_idx[c(5, 17, 33)])
  mg <- suppressMessages(broc(brf(conf | old ~ 1, criterion ~ 1 + (1 | gr(subj/item, cor_thresholds = FALSE)),
                                  family = evsd()), d))
  expect_identical(names(mg$model_data$criterion$random), c("subj", "subj_item"))
})

test_that("bivariate_dp with lambda_B and lambda2_B: R reproduces the JAX log-likelihood", {
  skip_if_no_jax()
  bv <- .fixture_bivariate_sdt()
  set.seed(11); bv$x <- rnorm(nrow(bv))
  m <- suppressMessages(broc(brf(resp(y1, y2) | item_type ~ 1, discrim ~ 1, rho ~ 1, criterion ~ 1, criterion2 ~ 1,
                                 lambda ~ 1, lambda_B ~ 1 + x, lambda2 ~ 1, lambda2_B ~ 1 + x,
                                 family = bivariate_dp()), bv))
  f <- suppressMessages(fit_broc(m, backend = "jax", sampler = "drghmc", chains = 1, iter_warmup = 50,
                                 iter_sampling = 5, seed = 2, progress_bar = FALSE))
  ll_r <- .obs_loglik_r(m, build_data_source_from_model(m), f$draws(format = "matrix"), re_formula = NULL)
  expect_lt(max(abs(ll_r - .read_loglik_SN_cols(.loglik_path(f), 1L, m$model_data$N))), 1e-5)
})
