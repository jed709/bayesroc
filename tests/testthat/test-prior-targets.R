.prior_target_data <- function() {
  set.seed(1)
  d <- expand.grid(participant = factor(1:6), item = factor(1:10))
  d$old <- as.integer(ave(runif(nrow(d)), d$participant, FUN = function(u) rank(u) <= 5))
  d$conf <- sample(1:6, nrow(d), TRUE)
  d$cond <- factor(sample(c("a", "b"), nrow(d), TRUE))
  d$x <- rnorm(nrow(d))
  d
}

test_that("a prior that reaches no parameter is an error naming the valid targets", {
  d <- .prior_target_data()
  f <- brf(conf | old ~ cond + (1|participant), criterion ~ 1, sigma ~ 1, family = uvsd())
  expect_error(broc(f, d, priors = broc_prior("normal(0, 0.2)", class = "b", dpar = "dprime", coef = "condB")),
               'Valid targets: coef = "\\(Intercept\\)"; coef = "condb"')
  expect_error(broc(f, d, priors = broc_prior("normal(0, 0.1)", class = "sd", dpar = "dprime", group = "subj")),
               'group = "participant"')
})

test_that("priors for parameters the model lacks are reported, not fatal, so one prior set serves several models", {
  d <- .prior_target_data()
  f <- brf(conf | old ~ cond + (1|participant), criterion ~ 1, sigma ~ 1, family = uvsd())
  expect_message(m <- broc(f, d, priors = c(broc_prior("normal(0, 0.2)", class = "b", dpar = "dprime", coef = "condb"),
                                            broc_prior("normal(0, 1)", class = "b", dpar = "lambda"),
                                            broc_prior("lkj_corr_cholesky(2)", class = "cor"))),
                 "not used.*dpar = \"lambda\".*class = \"cor\"")
  expect_true("beta_dprime[2] ~ normal(0, 0.2);" %in% trimws(strsplit(m$stan_code, "\n")[[1]]))
  expect_message(broc(f, d, priors = broc_prior("normal(0, 1)", class = "b", dpar = "criterion")), "thresh_mid")
})

test_that("coef = \"gapK\" sets one gap's prior", {
  d <- .prior_target_data()
  m <- broc(brf(conf | old ~ 1, criterion ~ 1, sigma ~ 1, family = uvsd()), d,
            priors = broc_prior("normal(-0.5, 0.4)", class = "b", dpar = "log_gaps", coef = "gap2"))
  L <- trimws(strsplit(m$stan_code, "\n")[[1]])
  expect_true("beta_log_gaps[2, 1] ~ normal(-0.5, 0.4);" %in% L)
  expect_true("beta_log_gaps[1, 1] ~ normal(-0.5, 0.5);" %in% L)
  tab <- get_broc_prior(m)
  expect_equal(tab$source[tab$dpar == "log_gaps"], c("default", "user", "default", "default"))
})

test_that("a global smooth-SD prior is reported as the user's", {
  d <- .prior_target_data()
  m <- broc(brf(conf | old ~ s(x), criterion ~ 1, sigma ~ 1, family = uvsd()), d,
            priors = broc_prior("student_t(3, 0, 1)", class = "sds"))
  tab <- get_broc_prior(m)
  expect_true(all(tab$source[tab$class == "sds"] == "user"))
})

test_that("location priors go to intercepts and cell means, zero-centred priors to contrasts and slopes", {
  d <- .prior_target_data()
  d$g <- factor(sample(c("x", "y"), nrow(d), TRUE))
  pr <- function(f, dd = d, ...) {
    t <- get_broc_prior(broc(f, dd, ...))
    setNames(t$prior[t$class == "b"], paste(t$dpar, t$coef)[t$class == "b"])
  }
  p <- pr(brf(conf | old ~ cond + x, criterion ~ cond, sigma ~ 1, family = uvsd()))
  expect_equal(unname(p[c("dprime (Intercept)", "dprime condb", "dprime x")]),
               c("normal(1.5, 0.75)", "normal(0, 0.75)", "normal(0, 0.75)"))
  expect_equal(unname(p[c("thresh_mid (Intercept)", "thresh_mid condb", "log_gaps gap2_(Intercept)", "log_gaps gap2_condb")]),
               c("normal(0.5, 0.75)", "normal(0, 0.75)", "normal(-0.5, 0.5)", "normal(0, 0.5)"))
  p <- pr(brf(conf | old ~ 0 + cond + cond:x, criterion ~ 0 + cond, family = evsd()))
  expect_equal(unname(p[c("dprime conda", "dprime condb", "dprime conda:x", "thresh_mid condb", "log_gaps gap1_condb")]),
               c("normal(1.5, 0.75)", "normal(1.5, 0.75)", "normal(0, 0.75)", "normal(0.5, 0.75)", "normal(-0.5, 0.5)"))
  p <- pr(brf(conf | old ~ 0 + cond + g, criterion ~ 1, family = evsd()))
  expect_equal(unname(p[c("dprime conda", "dprime gy")]), c("normal(1.5, 0.75)", "normal(0, 0.75)"))
  p <- pr(brf(conf | old ~ 0 + cond:g, criterion ~ 1, family = evsd()))
  expect_true(all(p[grepl("^dprime", names(p))] == "normal(1.5, 0.75)"))
  m <- broc(brf(conf | old ~ cond, criterion ~ 1, family = evsd()), d)
  expect_true(all(c("beta_dprime[1] ~ normal(1.5, 0.75);", "beta_dprime[2] ~ normal(0, 0.75);") %in%
                    trimws(strsplit(m$stan_code, "
")[[1]])))
  cfg <- build_numpyro_config(m)
  expect_equal(unlist(cfg$params$dprime$priors), c(`(Intercept)` = "normal(1.5, 0.75)", condb = "normal(0, 0.75)"))
})

test_that("an encoding factor's levels are cell means only without an intercept", {
  d <- .prior_target_data()
  d$enc <- factor(ifelse(d$old == 1, sample(c("p", "q"), nrow(d), TRUE), NA))
  pr <- function(f) { t <- get_broc_prior(suppressMessages(broc(f, d))); t$prior[t$dpar == "dprime"] }
  expect_equal(pr(brf(conf | old ~ 0 + encoding(enc), criterion ~ 1, family = evsd())), rep("normal(1.5, 0.75)", 2))
  expect_equal(pr(brf(conf | old ~ encoding(enc), criterion ~ 1, family = evsd())), c("normal(1.5, 0.75)", "normal(0, 0.75)"))
})

test_that("the threshold location default follows the old/new coding and the family", {
  d <- .prior_target_data()
  mid <- function(m) { t <- get_broc_prior(m); t$prior[t$dpar == "thresh_mid" & t$coef == "(Intercept)"] }
  expect_equal(mid(broc(brf(conf | old ~ 1, criterion ~ 1, family = evsd()), d)), "normal(0.5, 0.75)")
  dc <- d; dc$old <- dc$old - 0.5
  m <- broc(brf(conf | old ~ 1, criterion ~ 1, family = evsd()), dc)
  expect_equal(mid(m), "normal(0, 0.5)")
  expect_true("beta_thresh_mid ~ normal(0, 0.5);" %in% trimws(strsplit(m$stan_code, "
")[[1]]))
  expect_equal(unlist(build_numpyro_config(m)$prior_thresh_mid_grid), "normal(0, 0.5)")
  expect_equal(mid(broc(brf(conf | old ~ 1, criterion ~ 1, family = evsd()), dc,
                        priors = broc_prior("normal(1, 1)", class = "b", dpar = "thresh_mid"))), "normal(1, 1)")
  expect_equal(mid(broc(brf(conf ~ 1, cutpoints ~ 1, family = cumulative()), d)), "normal(0, 1.5)")
})

test_that("source-mixture defaults put source A on the negative side and B on the positive side", {
  d <- .prior_target_data()
  d$src <- factor(sample(c("A", "B"), nrow(d), TRUE))
  t <- get_broc_prior(suppressMessages(broc(brf(conf | src ~ 1, criterion ~ 1, lambda ~ 1, dprime_B ~ 1,
                                                family = source_mixture()), d)))
  expect_equal(t$prior[t$dpar == "dprime"], "normal(-1, 1)")
  expect_equal(t$prior[t$dpar == "dprime_B"], "normal(1, 1)")
  expect_equal(t$prior[t$dpar == "thresh_mid"], "normal(0, 1.5)")
})
