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
  expect_equal(t$prior[t$dpar == "thresh_mid"], "normal(0, 0.75)")
})

test_that("source-axis defaults centre on the new-item baseline and follow the A = low convention", {
  set.seed(2)
  n <- 240
  d <- data.frame(subj = factor(rep(1:6, each = n / 6)),
                  type = factor(sample(c("new", "A", "B"), n, TRUE), levels = c("new", "A", "B")),
                  det = sample(1:6, n, TRUE), src = sample(1:6, n, TRUE), src3 = sample(1:3, n, TRUE))
  pr <- function(m, dp) { t <- get_broc_prior(m); t$prior[t$dpar == dp & t$class == "b"][1] }
  vr <- suppressMessages(broc(brf(resp(det, src) | type ~ 1, dprime2 ~ 1, discrim ~ 1, lambda ~ 1,
                                  criterion ~ 1, criterion2 ~ 1, family = vrdp2d()), d))
  expect_equal(pr(vr, "discrim"), "normal(-1, 1)")
  expect_equal(pr(vr, "thresh_mid2"), "normal(0, 0.75)")
  expect_equal(pr(vr, "log_gaps2"), "normal(0, 0.75)")
  bg <- suppressMessages(broc(brf(resp(det, src) | type ~ 1, discrim ~ 1, rho ~ 1, criterion ~ 1, criterion2 ~ 1,
                                  family = bivariate_gaussian()), d))
  expect_equal(pr(bg, "discrim"), "normal(-1, 1)")
  bb <- suppressMessages(broc(brf(resp(det, src) | type ~ 1, discrim ~ 1, rho ~ 1, criterion ~ 1, criterion2 ~ 1,
                                  family = bivariate_gaussian(bounded = TRUE)), d))
  expect_equal(pr(bb, "discrim"), "normal(-1.5, 0.75)")
  expect_equal(pr(vr, "dprime2"), "normal(2, 1)")
  bgB <- suppressMessages(broc(brf(resp(det, src) | type ~ 1, dprime_B ~ 1, discrim ~ 1, rho ~ 1, criterion ~ 1,
                                   criterion2 ~ 1, family = bivariate_gaussian()), d))
  expect_equal(pr(bgB, "dprime_B"), pr(bgB, "dprime"))
  bdB <- suppressMessages(broc(brf(resp(det, src) | type ~ 1, dprime_B ~ 1, discrim ~ 1, rho ~ 1, lambda ~ 1, lambda2 ~ 1,
                                   criterion ~ 1, criterion2 ~ 1, family = bivariate_dp()), d))
  expect_equal(pr(bdB, "dprime_B"), pr(bdB, "dprime"))
  odd <- suppressMessages(broc(brf(resp(det, src3) | type ~ 1, discrim ~ 1, rho ~ 1, criterion ~ 1, criterion2 ~ 1,
                                   family = bivariate_gaussian()), d))
  expect_equal(pr(odd, "thresh_mid2"), "normal(-0.5, 0.75)")
  odd_user <- suppressMessages(broc(brf(resp(det, src3) | type ~ 1, discrim ~ 1, rho ~ 1, criterion ~ 1, criterion2 ~ 1,
                                        family = bivariate_gaussian()), d,
                                    priors = broc_prior("normal(0, 1)", class = "b", dpar = "thresh_mid2")))
  expect_equal(pr(odd_user, "thresh_mid2"), "normal(0, 1)")
  bc <- suppressMessages(broc(brf(resp(det, src) ~ 1, mu2 ~ 1, cutpoints ~ 1, cutpoints2 ~ 1,
                                  family = bivariate_cumulative()), d))
  expect_equal(pr(bc, "thresh_mid2"), "normal(0, 1.5)")
  expect_equal(pr(bc, "log_gaps2"), "normal(-0.5, 0.5)")
})

test_that("a user population prior on a parameter leaves its default random-effect SD prior alone", {
  set.seed(3)
  n <- 240
  d <- data.frame(subj = factor(rep(1:6, each = n / 6)),
                  type = factor(sample(c("new", "A", "B"), n, TRUE), levels = c("new", "A", "B")),
                  det = sample(1:6, n, TRUE), src = sample(1:6, n, TRUE))
  f <- brf(resp(det, src) | type ~ 1, discrim ~ 1 + (1 | subj), rho ~ 1 + (1 | subj), criterion ~ 1, criterion2 ~ 1,
           family = bivariate_gaussian(bounded = TRUE))
  sdp <- function(m, dp) { t <- get_broc_prior(m); t$prior[t$dpar == dp & t$class == "sd"][1] }
  base <- suppressMessages(broc(f, d))
  for (pri in list(broc_prior("normal(-1, 0.5)", class = "b", dpar = "discrim"),
                   broc_prior("normal(0, 0.8)", class = "b", dpar = "rho"),
                   broc_prior("normal(0, 2)", class = "b"))) {
    m <- suppressMessages(broc(f, d, priors = pri))
    expect_equal(sdp(m, "discrim"), sdp(base, "discrim"))
    expect_equal(sdp(m, "rho"), sdp(base, "rho"))
  }
})

test_that("rho defaults: directional only for the unbounded bivariate Gaussian, every symmetry and coding", {
  base <- expand.grid(det = 1:6, src = 1:6, type = c("new", "A", "B"))
  d <- rbind(transform(base, cond = "c1"), transform(base, cond = "c2"))
  d$type <- factor(d$type, levels = c("new", "A", "B")); d$cond <- factor(d$cond)
  pri <- function(fam, sym, rhs) {
    args <- list(resp(det, src) | type ~ 1, discrim ~ 1, criterion ~ 1, criterion2 ~ 1, as.formula(paste("rho ~", rhs)))
    if (sym) args <- c(args, list(as.formula(paste("rho_B ~", rhs))))
    if (identical(fam$family, "bivariate_dp")) args <- c(args, list(lambda ~ 1, lambda2 ~ 1))
    t <- get_broc_prior(suppressMessages(broc(do.call(brf, c(args, list(family = fam))), d)))
    t <- t[t$class == "b" & t$dpar %in% c("rho", "rho_B"), ]
    setNames(t$prior, paste(t$dpar, t$coef))
  }
  for (asym in c(FALSE, TRUE)) {
    p <- pri(bivariate_gaussian(), asym, "0 + cond")
    expect_true(all(p[startsWith(names(p), "rho cond")] == "normal(-0.5, 0.5)"))
    if (asym) expect_true(all(p[startsWith(names(p), "rho_B cond")] == "normal(0.5, 0.5)"))
    p <- pri(bivariate_gaussian(), asym, "1 + cond")
    expect_equal(unname(p["rho (Intercept)"]), "normal(-0.5, 0.5)")
    expect_equal(unname(p["rho condc2"]), "normal(0, 0.5)")
    if (asym) expect_equal(unname(p[c("rho_B (Intercept)", "rho_B condc2")]), c("normal(0.5, 0.5)", "normal(0, 0.5)"))
    for (fam in list(bivariate_gaussian(bounded = TRUE), bivariate_dp(bounded = TRUE)))
      expect_true(all(pri(fam, asym, "1 + cond") == "normal(0, 1)"))
    expect_true(all(pri(bivariate_dp(), asym, "1 + cond") == "normal(0, 0.5)"))
  }
})


test_that("default priors follow the link rule: RE SDs by parameter link, latent units by family link, softplus gaps matched", {
  d <- .prior_target_data()
  tab <- function(m) {
    t <- get_broc_prior(m)
    t <- t[t$class %in% c("b", "sd") & (t$coef %in% c("", "(Intercept)", "gap1", "gap1_(Intercept)")), ]
    t <- t[!duplicated(paste(t$class, t$dpar)), ]
    setNames(t$prior, paste(t$class, t$dpar))
  }
  uv <- function(link, gl) tab(broc(brf(conf | old ~ 1 + (1 | participant), criterion ~ 1 + (1 | participant),
                                         sigma ~ 1 + (1 | participant), family = uvsd(link = link)), d, gap_link = gl))
  want <- list(
    probit  = list(log = c("normal(1.5, 0.75)", "normal(0.5, 0.75)", "normal(-0.5, 0.5)", "normal(0, 1)", "normal(0, 0.5)"),
                   softplus = c("normal(1.5, 0.75)", "normal(0.5, 0.75)", "normal(-0.2, 0.7)", "normal(0, 1)", "normal(0, 0.75)")),
    logit   = list(log = c("normal(2.72, 1.36)", "normal(0.91, 1.36)", "normal(0.1, 0.5)", "normal(0, 1.81)", "normal(0, 0.5)"),
                   softplus = c("normal(2.72, 1.36)", "normal(0.91, 1.36)", "normal(0.7, 0.8)", "normal(0, 1.81)", "normal(0, 0.75)")),
    cloglog = list(log = c("normal(1.92, 0.96)", "normal(0.64, 0.96)", "normal(-0.25, 0.5)", "normal(0, 1.28)", "normal(0, 0.5)"),
                   softplus = c("normal(1.92, 0.96)", "normal(0.64, 0.96)", "normal(0.15, 0.7)", "normal(0, 1.28)", "normal(0, 0.75)")))
  for (lk in names(want)) for (gl in c("log", "softplus")) {
    p <- uv(get(paste0("link_", lk))(), gl)
    w <- want[[lk]][[gl]]
    expect_equal(unname(p[c("b dprime", "b thresh_mid", "b log_gaps", "sd dprime", "sd log_gaps")]), w, info = paste(lk, gl))
    expect_equal(unname(p[c("sd thresh_mid", "b sigma", "sd sigma")]), c(w[4], "normal(0, 0.5)", "normal(0, 0.3)"), info = paste(lk, gl))
  }
  mx <- tab(broc(brf(conf | old ~ 1, criterion ~ 1, lambda ~ 1 + (1 | participant), family = mixture(link = link_logit())), d))
  expect_equal(unname(mx[c("b lambda", "sd lambda")]), c("normal(0, 1.5)", "normal(0, 1)"))

  set.seed(3)
  n <- 240
  b <- data.frame(subj = factor(rep(1:6, each = n / 6)),
                  type = factor(sample(c("new", "A", "B"), n, TRUE), levels = c("new", "A", "B")),
                  det = sample(1:6, n, TRUE), src = sample(1:6, n, TRUE))
  bv <- function(fam, gl = "log", ...) tab(suppressMessages(broc(brf(resp(det, src) | type ~ 1 + (1 | subj),
    discrim ~ 1 + (1 | subj), rho ~ 1 + (1 | subj), sigma2 ~ 1 + (1 | subj), criterion ~ 1 + (1 | subj),
    criterion2 ~ 1 + (1 | subj), family = fam), b, gap_link = gl, ...)))
  u <- bv(bivariate_gaussian())
  expect_equal(unname(u[c("sd discrim", "sd rho", "sd sigma2", "sd thresh_mid2", "sd log_gaps2", "b log_gaps2")]),
               c("normal(0, 1)", "normal(0, 0.5)", "normal(0, 0.3)", "normal(0, 1)", "normal(0, 0.5)", "normal(0, 0.75)"))
  s <- bv(bivariate_gaussian(bounded = TRUE), "softplus")
  expect_equal(unname(s[c("sd dprime", "sd discrim", "sd rho", "sd sigma2", "b log_gaps", "b log_gaps2", "sd log_gaps2")]),
               c("normal(0, 0.5)", "normal(0, 0.5)", "normal(0, 1)", "normal(0, 0.3)", "normal(-0.2, 0.7)", "normal(0.5, 1.25)", "normal(0, 0.75)"))
  cs <- bv(bivariate_gaussian(), priors = broc_prior("normal(0, 0.2)", class = "sd", dpar = "criterion2"))
  expect_equal(unname(cs[c("sd thresh_mid2", "sd log_gaps2", "sd log_gaps")]), c("normal(0, 0.2)", "normal(0, 0.2)", "normal(0, 0.5)"))
  gs <- bv(bivariate_gaussian(), priors = broc_prior("normal(0, 0.4)", class = "sd"))
  expect_true(all(gs[startsWith(names(gs), "sd ")] == "normal(0, 0.4)"))
  ub <- bv(bivariate_gaussian(), priors = broc_prior("normal(0, 0.3)", class = "b", dpar = "rho"))
  expect_equal(unname(ub[c("b rho", "sd rho")]), c("normal(0, 0.3)", "normal(0, 0.5)"))
})
