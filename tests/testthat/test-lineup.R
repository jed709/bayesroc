.lineup_fam <- function(...) lineup(size = "k", suspect = "designated", ...)

test_that("lineup coding: positions, identities and the identifications a design cannot produce", {
  d <- .sim_lineup(n_set = 6, n_per = 30)
  m <- suppressMessages(broc(brf(resp(id, conf) | present ~ 1, criterion ~ 1, family = .lineup_fam()), data = d))
  lu <- m$model_data$lineup
  expect_identical(c(lu$R, lu$J, m$model_data$K), c(2L, 3L, 5L))
  sd <- m$stan_data
  expect_true(all(sd$lu_slot[d$present == 1] == 1L))
  expect_true(all(sd$lu_slot[d$present == 0 & d$designated] == 2L))
  expect_true(all(sd$lu_slot[d$present == 0 & !d$designated] == 0L))
  expect_true(all(sd$lu_kind[d$id == "reject"] == 2L))
  expect_true(all(sd$lu_kind[d$id == "suspect" & sd$lu_slot > 0L] == 0L))
  expect_true(all(sd$y[d$id == "reject"] == 3L - d$conf[d$id == "reject"]))
  expect_true(all(sd$y[d$id != "reject"] == 2L + d$conf[d$id != "reject"]))
  expect_identical(lu$code_labels, c("rej 2", "rej 1", paste("sus", 1:3), paste("fil", 1:3)))

  bad <- d
  bad$designated <- FALSE
  i <- which(bad$present == 0 & bad$k > 1)[1]
  bad$id[i] <- "suspect"
  expect_error(broc(brf(resp(id, conf) | present ~ 1, criterion ~ 1, family = .lineup_fam()), data = bad),
               "no designated innocent suspect")
  bad <- d
  bad$id[which(bad$k == 1)[1]] <- "filler"
  expect_error(broc(brf(resp(id, conf) | present ~ 1, criterion ~ 1, family = .lineup_fam()), data = bad),
               "showup")
  bad <- d
  bad$conf[which(bad$id == "suspect")[1]] <- NA
  expect_error(broc(brf(resp(id, conf) | present ~ 1, criterion ~ 1, family = .lineup_fam()), data = bad),
               "needs its confidence")
  bad <- d
  bad$conf[which(bad$id == "reject")[1]] <- NA
  expect_error(broc(brf(resp(id, conf) | present ~ 1, criterion ~ 1, family = .lineup_fam()), data = bad),
               "some rejections but not others")
  expect_error(broc(brf(resp(id, conf) | present ~ x, criterion ~ 1, family = .lineup_fam()), data = d,
                    encoding_vars = "x"), "do not apply to lineup")
  one <- .sim_lineup(n_set = 6, n_per = 30, R = 1L)
  m1 <- suppressMessages(broc(brf(resp(id, conf) | present ~ 1, criterion ~ 1, family = .lineup_fam()), data = one))
  expect_identical(c(m1$model_data$lineup$R, m1$model_data$K), c(1L, 5L))
})

test_that("lineup cells: native, pure-JAX and R match the 40-digit reference; custom JVP, vmap and f32", {
  f <- utils::read.csv(test_path("fixtures", "lineup_reference.csv"))
  expect_gt(nrow(f), 300)
  nat <- vapply(seq_len(nrow(f)), function(i) {
    r <- f[i, ]
    .lineup_cells(r$kind, r$k, r$m, r$s, r$lo, r$hi, r$sb, r$lo_open == 1, r$hi_open == 1)
  }, numeric(1))
  expect_lt(max(abs(log(nat) - f$log_prob) / pmax(1, abs(f$log_prob))), 1e-11)
  sub <- f[f$log_prob > -30, ]
  rr <- vapply(seq_len(nrow(sub)), function(i) {
    r <- sub[i, ]
    .lineup_cell_r(r$kind, r$k, r$m, r$s, r$lo, r$hi, r$sb, r$lo_open == 1, r$hi_open == 1)
  }, numeric(1))
  expect_lt(max(abs(log(rr) - sub$log_prob)), 1e-9)

  skip_if_no_jax()
  out <- system2(jax_python(), c(test_path("python", "lineup_check.py"), dirname(find_backend_script()), .bvn_ffi_lib(),
                                 test_path("fixtures", "lineup_reference.csv")), stdout = TRUE, stderr = FALSE)
  res <- jsonlite::fromJSON(sub("^RESULT ", "", grep("^RESULT ", out, value = TRUE)))
  expect_true(res$ffi_loaded)
  expect_lt(res$fixture_rel, 1e-11)
  expect_lt(res$jax_vs_native, 1e-8)
  expect_lt(res$jvp_vs_fd, 1e-7)
  expect_equal(res$vmap_vs_rows, 0)
  expect_lt(res$f32_rel, 1e-6)
})

test_that("lineup category probabilities match a Monte Carlo of the Independent Observations task", {
  set.seed(21)
  tau <- c(-0.3, 0.6, 1.2, 1.9)
  R <- 2L; J <- 3L
  md <- list(model_data = list(lineup = list(R = R, J = J)))
  n <- 200000
  worst <- 0
  for (cfg in list(list(slot = 1L, k = 6L, sb = 0), list(slot = 1L, k = 3L, sb = 0.5), list(slot = 2L, k = 6L, sb = 0.3),
                   list(slot = 0L, k = 6L, sb = 0.4), list(slot = 1L, k = 1L, sb = 0.3), list(slot = 0L, k = 1L, sb = 0))) {
    pr <- list(dprime = matrix(1.3), sigma = matrix(1.2), dprime_L = matrix(0.4), sigma_L = matrix(0.8),
               sigma_meta = matrix(cfg$sb))
    p <- probs_lineup(pr, array(tau, c(1, 1, 4)), list(lu_slot = cfg$slot, lu_size = cfg$k), 1, 1, md)[1, 1, ]
    W <- matrix(stats::rnorm(n * cfg$k), n, cfg$k)
    if (cfg$slot == 1L) W[, 1] <- 1.3 + 1.2 * W[, 1]
    if (cfg$slot == 2L) W[, 1] <- 0.4 + 0.8 * W[, 1]
    X <- W + cfg$sb * stats::rnorm(n)
    win <- max.col(X, ties.method = "first")
    pos <- findInterval(do.call(pmax, as.data.frame(X)), tau) + 1L
    code <- ifelse(pos <= R, pos, ifelse(cfg$k == 1L | (win == 1L & cfg$slot != 0L), pos, pos + J))
    obs <- tabulate(code, R + 2L * J) / n
    z <- (obs - p) / sqrt(pmax(p * (1 - p), 1e-12) / n)
    z[p < 1e-9 & obs == 0] <- 0
    expect_equal(sum(p), 1, tolerance = 1e-8)
    worst <- max(worst, abs(z))
  }
  expect_lt(worst, 4.5)
})

test_that("lineup batch and plain Stan agree on log density and gradient for every layout", {
  skip_if_no_cmdstan()
  cfgs <- list(
    full = list(f = brf(resp(id, conf) | present ~ 1 + (1 | s | set), criterion ~ 1 + (1 | s | set), sigma ~ 1,
                        dprime_innocent ~ 1 + (1 | set), sigma_innocent ~ 1, sigma_b ~ 1, family = .lineup_fam()),
                d = .sim_lineup(n_set = 10, n_per = 20, sigma_b = 0.4)),
    undesignated_slopes = list(f = brf(resp(id, conf) | present ~ x + (1 + x | set), criterion ~ x + (1 | set),
                                       criterion_scale ~ 1 + (1 | set), sigma ~ 1 + (1 | set), sigma_b ~ x,
                                       family = lineup(size = "k")),
                               d = .sim_lineup(n_set = 10, n_per = 20, R = 1L, designated = FALSE, seed = 2)),
    no_noise = list(f = brf(resp(id, conf) | present ~ 1 + (1 | set), criterion ~ 1 + (1 | set),
                            dprime_innocent ~ x, family = .lineup_fam()),
                    d = .sim_lineup(n_set = 10, n_per = 20, seed = 3)))
  for (nm in names(cfgs)) {
    cf <- cfgs[[nm]]
    mb <- suppressMessages(broc(cf$f, data = cf$d))
    mp <- suppressMessages(.broc_plain(cf$f, data = cf$d))
    expect_true(isTRUE(mb$batch_likelihood))
    r <- .stan_parity(mp, mb, warmup = 30)
    expect_identical(r$dim[1], r$dim[2], label = nm)
    expect_lt(r$lp, 1e-9, label = paste(nm, "lp"))
    expect_lt(r$grad, 1e-8, label = paste(nm, "gradient"))
  }
})

test_that("the JAX lineup likelihood equals R's predicted probability of every observed response, draw by draw", {
  skip_if_no_jax()
  d <- .sim_lineup(n_set = 8, n_per = 25, sigma_b = 0.4, seed = 5)
  m <- suppressMessages(broc(brf(resp(id, conf) | present ~ 1 + (1 | set), criterion ~ 1 + (1 | set), sigma ~ 1,
                                 dprime_innocent ~ 1, sigma_innocent ~ 1, sigma_b ~ 1, family = .lineup_fam()), data = d))
  fit <- suppressMessages(fit_broc(m, backend = "jax", chains = 1, iter_warmup = 100, iter_sampling = 20, seed = 1,
                                   progress_bar = FALSE))
  ll <- fit$draws("log_lik", format = "matrix")
  p <- predict(fit, summary = FALSE)
  code <- m$model_data$lineup$code
  idx <- cbind(seq_along(code), code)
  lp <- t(vapply(seq_len(nrow(ll)), function(s) log(p[s, , ][idx]), numeric(length(code))))
  expect_lt(max(abs(lp - ll) / pmax(1, abs(ll))), 1e-6)
})

test_that("lineup through the JAX transport map (f64 and f32) recovers Stan's posterior", {
  skip_if_no_cmdstan(); skip_if_no_jax()
  d <- .sim_lineup(n_set = 16, n_per = 30, seed = 4)
  m <- suppressMessages(broc(brf(resp(id, conf) | present ~ 1 + (1 | set), criterion ~ 1 + (1 | set), sigma ~ 1,
                                 dprime_innocent ~ 1 + (1 | set), family = .lineup_fam()), data = d))
  sm <- function(f) {
    da <- f$draws(format = "draws_array")
    v <- grep("^(beta|sigma_)", posterior::variables(da), value = TRUE)
    s <- posterior::summarise_draws(posterior::subset_draws(da, variable = v), "mean", "mcse_mean")
    s[order(s$variable), ]
  }
  rs <- sm(fit_broc(m, backend = "stan", chains = 4, parallel_chains = 4, iter_warmup = 1000, iter_sampling = 3000,
                    seed = 1, refresh = 0))
  for (prec in c("f64", "f32")) {
    out <- utils::capture.output(
      fj <- fit_broc(m, backend = "jax", chains = 4, iter_warmup = 1000, iter_sampling = 3000, seed = 3,
                     progress_bar = TRUE, control = list(transport = TRUE, precision = prec)))
    expect_true(any(grepl("[NUTS] group transport map", out, fixed = TRUE)), label = prec)
    js <- sm(fj)
    expect_identical(js$variable, rs$variable)
    z <- abs(js$mean - rs$mean) / sqrt(js$mcse_mean^2 + rs$mcse_mean^2)
    expect_lt(max(z), 4, label = paste(prec, "max |z|"))
  }
})
