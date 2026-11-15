.sim_crit_sd <- function(seed = 7, n_sub = 10, n_per = 40) {
  set.seed(seed)
  n <- n_sub * n_per
  d <- data.frame(sub = factor(rep(seq_len(n_sub), each = n_per)))
  d$item_type <- factor(sample(c("new", "A", "B"), n, TRUE), levels = c("new", "A", "B"))
  d$x <- round(rnorm(n), 3)
  z1 <- rnorm(n, ifelse(d$item_type == "new", 0, 1.2))
  z2 <- rnorm(n, ifelse(d$item_type == "A", -1, ifelse(d$item_type == "B", 1, 0)))
  d$det <- as.integer(1L + rowSums(outer(z1, c(-0.8, 0, 0.8), ">")))
  d$src <- as.integer(1L + rowSums(outer(z2, c(-0.7, 0, 0.7), ">")))
  d
}

.crit_sd_model <- function(priors, sc = "varying") {
  suppressMessages(broc(brf(resp(det, src) | item_type ~ 1, discrim ~ 1, rho ~ 1,
                            criterion ~ 1 + (1 + x | sub), criterion2 ~ 1 + (1 | sub),
                            family = bivariate_gaussian(source_criteria = sc)),
                        .sim_crit_sd(), priors = priors))
}

test_that("criterion RE components follow the fixed-effect gap numbering", {
  m <- .crit_sd_model(NULL)
  for (p in c("criterion", "criterion2")) {
    re <- .re_list_of(m$model_data, p)[["sub"]]
    comp <- .crit_re_components(re)
    lab <- .re_dim_labels(m, p, re)
    J <- .n_thresh_of(m, p)
    expect_length(comp, re$dim)
    want <- ifelse(comp == 0L, "midpoint", .crit_gap_label(pmax(comp, 1L), J))
    expect_true(all(mapply(grepl, want, lab, fixed = TRUE)))
  }
})

test_that("criterion SD priors resolve per component, term and bin", {
  pr <- c(broc_prior("normal(0, 0.15)", class = "sd", group = "sub"),
          broc_prior("normal(0, 0.4)", class = "sd", dpar = "thresh_mid", group = "sub"),
          broc_prior("normal(0, 0.2)", class = "sd", dpar = "log_gaps", group = "sub", coef = "gap1"),
          broc_prior("normal(0, 0.05)", class = "sd", dpar = "log_gaps", group = "sub", coef = "gap2_x"),
          broc_prior("normal(0, 0.1)", class = "sd", dpar = "log_gaps2", coef = "gap2"))
  m <- .crit_sd_model(pr)
  pl <- m$prior_lookup
  r1 <- .crit_re_sd_resolve(pl, "criterion", "sub", .re_list_of(m$model_data, "criterion")[["sub"]])
  want1 <- ifelse(r1$comp == 0L, "normal(0, 0.4)",
                  ifelse(r1$comp == 1L, "normal(0, 0.2)",
                         ifelse(r1$comp == 2L & r1$term == "x", "normal(0, 0.05)", "normal(0, 0.15)")))
  expect_equal(r1$prior, want1)
  expect_true(all(r1$user))
  re2 <- .re_list_of(m$model_data, "criterion2")[["sub"]]
  r2 <- .crit_re_sd_resolve(pl, "criterion2", "sub", re2)
  expect_equal(nrow(r2), re2$dim)
  expect_equal(r2$prior, ifelse(r2$comp == 2L, "normal(0, 0.1)", "normal(0, 0.15)"))

  m0 <- .crit_sd_model(NULL)
  r0 <- .crit_re_sd_resolve(m0$prior_lookup, "criterion2", "sub", .re_list_of(m0$model_data, "criterion2")[["sub"]])
  expect_equal(r0$prior, rep("normal(0, 1)", nrow(r0)))
  expect_false(any(r0$user))

  mc <- .crit_sd_model(broc_prior("normal(0, 0.3)", class = "sd", dpar = "criterion2", group = "sub"))
  rc <- .crit_re_sd_resolve(mc$prior_lookup, "criterion2", "sub", .re_list_of(mc$model_data, "criterion2")[["sub"]])
  expect_true(all(rc$prior == "normal(0, 0.3)") && all(rc$user))
})

test_that("component SD priors reach get_broc_prior and the JAX config", {
  pr <- c(broc_prior("normal(0, 0.4)", class = "sd", dpar = "thresh_mid2", group = "sub"),
          broc_prior("normal(0, 0.1)", class = "sd", dpar = "log_gaps2", group = "sub", coef = "gap2"))
  m <- .crit_sd_model(pr)
  gp <- get_broc_prior(m)
  g2 <- gp[gp$class == "sd" & gp$dpar %in% c("thresh_mid2", "log_gaps2"), ]
  expect_equal(g2$coef, c("(Intercept)", "gap1_(Intercept)", "gap2_(Intercept)"))
  expect_equal(g2$prior, c("normal(0, 0.4)", "normal(0, 1)", "normal(0, 0.1)"))
  expect_equal(g2$source, c("user", "default", "user"))
  expect_false(any(gp$class == "sd" & gp$dpar == "criterion2"))

  re2 <- .re_list_of(m$model_data, "criterion2")[["sub"]]
  want <- .crit_re_sd_priors(m$prior_lookup, "criterion2", "sub", re2)
  cfg <- build_numpyro_config(m)
  rc <- Filter(function(r) r$param == "criterion2" && r$group == "sub", cfg$random_effects)[[1]]
  expect_equal(unlist(rc$prior_sd_per_coef), want)
})

test_that("a correlated block whose SD priors differ by component fits and reports its SDs", {
  skip_if_no_jax()
  set.seed(2)
  d <- expand.grid(participant = factor(1:8), words = factor(1:24))
  d$is_old <- as.integer((as.integer(d$words) + as.integer(d$participant)) %% 2 == 0)
  d$condition <- factor(c("aloud", "silent")[(as.integer(d$words) %% 2) + 1])
  d$conf <- sample(1:6, nrow(d), TRUE)
  m <- broc(brf(conf | is_old ~ condition - 1 + (1 | p | participant),
                criterion ~ 1 + (1 | p | participant), sigma ~ 1, family = uvsd(),
                encoding_vars = "condition"), d,
            priors = broc_prior("normal(0, 0.25)", class = "sd", dpar = "log_gaps"))
  per <- .crit_re_sd_priors(m$prior_lookup, "criterion", "participant",
                            .re_list_of(m$model_data, "criterion")[["participant"]])
  expect_gt(length(unique(per)), 1L)
  fit <- fit_broc(m, iter_warmup = 50, iter_sampling = 5,
                  chains = 1, parallel_chains = 1)
  expect_true(any(grepl("^sigma_cross_p_participant", fit$metadata()$variables)))
})
