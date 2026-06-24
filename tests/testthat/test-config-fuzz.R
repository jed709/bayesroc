# Randomized cross-backend config fuzzer (gated).
#
# Each run draws a random config from the cross-product of family x RE structure
# x bounded/varying x counts, fits it on plain Stan (autodiff oracle) AND batch
# Stan (hand-coded gradients), and asserts (a) posterior means agree within MC
# noise, (b) batch treedepth ~ plain, (c) the predict<->likelihood bridge holds.
# Gated behind BAYESROC_FUZZ because it fits many models; seed-swept in CI so
# coverage compounds over time. The likelihood/gradient bug class found in this
# package was batch-specific, so batch-vs-plain agreement directly targets it.

.fuzz_seed <- as.integer(Sys.getenv("BAYESROC_FUZZ_SEED", "0"))
.fuzz_n    <- as.integer(Sys.getenv("BAYESROC_FUZZ_N", "4"))

# Random valid config from curated templates + random knobs. Returns
# list(formula, data, counts, label).
.gen_config <- function(seed) {
  set.seed(seed)
  cut_int <- function(z, thr) as.integer(cut(z, c(-Inf, thr, Inf)))
  fam <- sample(c("uvsd", "dpsd", "bivariate_gaussian", "bivariate_dp"), 1)
  if (fam %in% c("uvsd", "dpsd")) {
    nsub <- 8; nper <- 35; K <- 6; thr <- sort(rnorm(K - 1))
    sub <- rep(seq_len(nsub), each = nper)
    u <- rnorm(nsub, 0, 0.4)[sub]
    io <- rbinom(nsub * nper, 1, 0.5)
    z <- ifelse(io == 1, rnorm(length(io), 1.0 + u, 1.1), rnorm(length(io), u, 1))
    d <- data.frame(resp = cut_int(z, thr), cond = io, sub = sub)
    re <- sample(c("none", "(1|sub)", "(1|s|sub)"), 1)
    dp_rhs <- if (re == "none") "1" else paste0("1 + ", re)
    sig_rhs <- if (re == "(1|s|sub)") "1 + (1|s|sub)" else "1"
    parts <- list(stats::as.formula(paste0("resp | cond ~ ", dp_rhs)))
    parts <- c(parts, criterion ~ 1)
    if (fam == "uvsd") parts <- c(parts, stats::as.formula(paste0("sigma ~ ", sig_rhs)))
    if (fam == "dpsd") parts <- c(parts, lambda ~ 1)
    parts$family <- if (fam == "uvsd") uvsd() else dpsd()
    # Aggregate to (sub x cond x resp) counts -- the grouping factor sub is kept
    # in the key, so the RE structure is preserved (this is the package's
    # "quick version"). Valid whenever there's no trial-level covariate.
    d$resp <- factor(d$resp, levels = seq_len(K))
    agg <- as.data.frame(table(d[c("sub", "cond", "resp")]), responseName = "count")
    agg <- agg[agg$count > 0, ]
    agg$resp <- as.integer(as.character(agg$resp))
    agg$cond <- as.integer(as.character(agg$cond))
    agg$sub  <- as.integer(as.character(agg$sub))
    list(formula = do.call(brf, parts), data = agg, counts = "count",
         label = sprintf("%s/%s", fam, re))
  } else {
    np <- 220; K1 <- 4; K2 <- 4; c1 <- sort(rnorm(3)); c2 <- sort(rnorm(3))
    dp <- runif(1, 0.5, 1.2); di <- runif(1, 0.4, 1.0); rho <- runif(1, 0.1, 0.5)
    chol2 <- function(r) chol(matrix(c(1, r, r, 1), 2))
    simb <- function(mu, r, n) sweep(matrix(rnorm(2 * n), n, 2) %*% chol2(r), 2, mu, "+")
    mk <- function(z, it) data.frame(detect_rat = cut_int(z[, 1], c1),
                                     source_rat = cut_int(z[, 2], c2), item_type = it)
    d <- rbind(mk(simb(c(0, 0), 0, np), "new"), mk(simb(c(dp, di), rho, np), "A"),
               mk(simb(c(dp, -di), -rho, np), "B"))
    d$item_type <- factor(d$item_type, levels = c("new", "A", "B"))
    bounded <- sample(c(TRUE, FALSE), 1); varying <- sample(c(TRUE, FALSE), 1)
    parts <- list(resp(detect_rat, source_rat) | item_type ~ 1, discrim ~ 1,
                  rho ~ 1, criterion2 ~ 1)
    if (fam == "bivariate_dp") {
      parts <- c(parts, lambda ~ 1, lambda2 ~ 1)
      parts$family <- bivariate_dp(bounded = bounded, varying_source_criteria = varying)
    } else {
      parts$family <- bivariate_gaussian(bounded = bounded, varying_source_criteria = varying)
    }
    d$detect_rat <- factor(d$detect_rat, levels = seq_len(K1))
    d$source_rat <- factor(d$source_rat, levels = seq_len(K2))
    agg <- as.data.frame(table(d[c("item_type", "detect_rat", "source_rat")]),
                         responseName = "count")
    agg <- agg[agg$count > 0, ]
    agg$detect_rat <- as.integer(as.character(agg$detect_rat))
    agg$source_rat <- as.integer(as.character(agg$source_rat))
    agg$item_type  <- factor(agg$item_type, levels = c("new", "A", "B"))
    list(formula = do.call(brf, parts), data = agg, counts = "count",
         label = sprintf("%s/bnd=%s/vary=%s", fam, bounded, varying))
  }
}

.fuzz_one <- function(cfg) {
  mp <- suppressMessages(broc(cfg$formula, cfg$data, counts = cfg$counts, batch_likelihood = FALSE))
  mb <- suppressMessages(broc(cfg$formula, cfg$data, counts = cfg$counts, batch_likelihood = TRUE))
  A <- list(backend = "stan", iter_warmup = 300, iter_sampling = 300, chains = 2,
            parallel_chains = 2, refresh = 0, show_messages = FALSE)
  fp <- suppressMessages(do.call(fit_broc, c(list(mp), A)))
  fb <- suppressMessages(do.call(fit_broc, c(list(mb), A)))
  sp <- fp$summary(); sb <- fb$summary()
  common <- intersect(sp$variable, sb$variable)
  common <- common[grepl("^beta_", common) & !grepl("log_lik", common)]
  mpv <- sp$mean[match(common, sp$variable)]; sdv <- sp$sd[match(common, sp$variable)]
  mbv <- sb$mean[match(common, sb$variable)]
  std_diff <- max(abs(mbv - mpv) / (sdv + 0.02), na.rm = TRUE)
  td <- sum(fb$diagnostic_summary()$num_max_treedepth) - sum(fp$diagnostic_summary()$num_max_treedepth)
  # bridge on batch fit
  ll <- as.matrix(fb$draws(variables = "log_lik", format = "matrix"))
  pr <- predict(fb, type = "response", summary = FALSE, cores = 2)
  S <- nrow(ll); N <- ncol(ll); sd0 <- mb$stan_data
  if (length(dim(pr)) == 4L) {
    po <- vapply(seq_len(N), function(n) pr[, n, sd0$y[n], sd0$y2[n]], numeric(S))
  } else {
    po <- vapply(seq_len(N), function(n) pr[, n, sd0$y[n]], numeric(S))
  }
  list(std_diff = std_diff, td = td, bridge = max(abs(po - exp(ll))))
}

test_that("randomized config fuzz: batch == plain + bridge", {
  skip_if_no_cmdstan()
  skip_if_not(identical(Sys.getenv("BAYESROC_FUZZ"), "1"),
              "set BAYESROC_FUZZ=1 to run the config fuzzer")
  for (i in seq_len(.fuzz_n)) {
    cfg <- .gen_config(.fuzz_seed * 1000L + i)
    r <- .fuzz_one(cfg)
    info <- sprintf("config %s (seed %d.%d): std_diff=%.3f td=%d bridge=%.1e",
                    cfg$label, .fuzz_seed, i, r$std_diff, r$td, r$bridge)
    expect_lt(r$std_diff, 0.35, label = info)
    expect_lte(r$td, 30, label = info)
    expect_lt(r$bridge, 1e-4, label = info)
  }
})
