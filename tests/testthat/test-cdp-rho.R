.cdp_rho_fixture <- function() {
  src <- readLines(testthat::test_path("test-cdp-rkg.R"))
  env <- new.env()
  eval(parse(text = src[1:(grep("^}", src)[1])]), envir = env)
  env$fixture_cdp_rkg()
}

.cdp_rho_model <- function(extra, data = .cdp_rho_fixture(), priors = NULL, rkg = TRUE) {
  base <- list(resp(conf, rk) | old ~ 1, fam ~ 1, criterion ~ 1, rec_crit ~ 1)
  if (rkg) base <- c(base, list(know_crit ~ 1))
  broc(do.call(brf, c(base, extra, list(family = cdp(old_levels = 2:4)))), data = data, priors = priors)
}

.rho_messages <- function(expr) {
  msgs <- character(0)
  val <- withCallingHandlers(expr, message = function(m) {
    msgs <<- c(msgs, conditionMessage(m))
    invokeRestart("muffleMessage")
  })
  list(value = val, messages = grep("rho|correlation", msgs, value = TRUE))
}

.cdp_ref_cell <- function(kind, muR, sR, muF, sF, cR, cK, tlo, thi, corr) {
  sc <- sF * sqrt(1 - corr^2)
  g <- function(x) {
    Rv <- muR + sR * x
    mF <- muF + corr * sF * x
    l <- (tlo - Rv - mF) / sc; h <- (thi - Rv - mF) / sc; k <- (cK - mF) / sc
    band <- function(l, h) {
      h <- pmax(h, l)
      ifelse(l > 0, pnorm(l, lower.tail = FALSE) - pnorm(h, lower.tail = FALSE), pnorm(h) - pnorm(l))
    }
    dnorm(x) * switch(kind, band(l, h), band(pmax(l, k), h), band(l, pmin(h, k)))
  }
  xc <- (cR - muR) / sR
  edges <- if (kind == 1) c(xc, 40) else c(-40, xc)
  kinks <- c((tlo - cK - muR) / sR, (thi - cK - muR) / sR)
  edges <- sort(unique(c(edges, kinks[is.finite(kinks) & kinks > edges[1] & kinks < edges[2]])))
  sum(vapply(seq_len(length(edges) - 1L), function(j)
    integrate(g, edges[j], edges[j + 1L], rel.tol = 1e-13, abs.tol = 0, subdivisions = 5000L)$value, numeric(1)))
}

test_that("cdp correlation formulas give the five layouts, labelled by their spelling", {
  d <- .cdp_rho_fixture()
  cases <- list(
    none   = list(f = list(), has = c(FALSE, FALSE), shared = FALSE, labels = character(0), note = NA),
    shared = list(f = list(rho ~ 1), has = c(TRUE, FALSE), shared = TRUE, labels = "rho", note = NA),
    both   = list(f = list(rho_target ~ 1, rho_lure ~ 1 + (1 | sub)), has = c(TRUE, TRUE), shared = FALSE,
                  labels = c("rho_target", "rho_lure"), note = NA),
    target = list(f = list(rho_target ~ 1), has = c(TRUE, FALSE), shared = FALSE, labels = "rho_target",
                  note = "rho_lure not specified - the lure"),
    lure   = list(f = list(rho_lure ~ 1), has = c(FALSE, TRUE), shared = FALSE, labels = "rho_lure",
                  note = "rho_target not specified - the target"))
  for (nm in names(cases)) {
    cs <- cases[[nm]]
    r <- .rho_messages(.cdp_rho_model(cs$f, d))
    md <- r$value$model_data
    expect_identical(c(isTRUE(md$has_rho), isTRUE(md$has_rho_N)), cs$has, label = nm)
    expect_identical(isTRUE(md$cdp_rho_shared), cs$shared, label = nm)
    pr <- get_broc_prior(r$value)
    expect_setequal(unique(pr$dpar[grepl("^rho", pr$dpar)]), cs$labels)
    if (is.na(cs$note)) expect_length(r$messages, 0) else expect_match(r$messages, cs$note, fixed = TRUE, all = FALSE)
  }
})

test_that("cdp rejects a shared rho mixed with separate correlations, and rho_N", {
  d <- .cdp_rho_fixture()
  expect_error(.cdp_rho_model(list(rho ~ 1, rho_lure ~ 1), d), "cannot be combined with 'rho_lure'")
  expect_error(.cdp_rho_model(list(rho ~ 1, rho_target ~ 1, rho_lure ~ 1), d),
               "cannot be combined with 'rho_target' or 'rho_lure'")
  expect_error(.cdp_rho_model(list(rho_N ~ 1), d), "'rho_lure'")
})

test_that("cdp correlation priors apply only under their own spelling", {
  d <- .cdp_rho_fixture()
  p_t <- broc_prior("normal(0, 0.3)", class = "b", dpar = "rho_target")
  p_s <- broc_prior("normal(0, 0.2)", class = "b", dpar = "rho")
  r <- .rho_messages(.cdp_rho_model(list(rho ~ 1), d, priors = p_t))
  expect_match(r$messages, "not used.*dpar = \"rho_target\"", all = FALSE)
  pr <- get_broc_prior(r$value)
  expect_identical(pr$source[pr$dpar == "rho"], "default")
  r <- .rho_messages(.cdp_rho_model(list(rho_target ~ 1, rho_lure ~ 1), d, priors = p_s))
  expect_match(r$messages, "not used.*dpar = \"rho\"", all = FALSE)
  m <- suppressMessages(.cdp_rho_model(list(rho ~ 1), d, priors = p_s))
  pr <- get_broc_prior(m)
  expect_identical(pr$prior[pr$dpar == "rho"], "normal(0, 0.2)")
  m <- suppressMessages(.cdp_rho_model(list(rho_target ~ 1, rho_lure ~ 1), d, priors = p_t))
  pr <- get_broc_prior(m)
  expect_identical(pr$prior[pr$dpar == "rho_target"], "normal(0, 0.3)")
  expect_identical(pr$source[pr$dpar == "rho_lure"], "default")
})

test_that("R-side CDP cells with a correlation partition the band and reduce to the independent cells", {
  set.seed(4)
  n <- 200
  a <- rnorm(n, 0.8, 0.4); b <- runif(n, 0.6, 1.5); cc <- rnorm(n, 0.5, 0.4); dd <- runif(n, 0.6, 1.5)
  cr <- rnorm(n, 1.2, 0.6); ck <- rnorm(n, 0.4, 0.6); lo <- rnorm(n, 0, 1); hi <- lo + runif(n, 0.2, 2)
  r <- c(runif(n - 40, -0.8, 0.8), runif(20, 0.9, 0.999), runif(20, -0.999, -0.9))
  dev <- vapply(seq_len(n), function(i) {
    s <- sqrt(b[i]^2 + dd[i]^2 + 2 * r[i] * b[i] * dd[i])
    sum(compute_rkg_probs_r(a[i], b[i], cc[i], dd[i], cr[i], ck[i], lo[i], hi[i], r[i])) -
      (pnorm((hi[i] - a[i] - cc[i]) / s) - pnorm((lo[i] - a[i] - cc[i]) / s))
  }, numeric(1))
  expect_lt(max(abs(dev)), 1e-10)
  z0 <- vapply(seq_len(n), function(i) max(abs(
    compute_rkg_probs_r(a[i], b[i], cc[i], dd[i], cr[i], ck[i], lo[i], hi[i], 0) -
      compute_rkg_probs_r(a[i], b[i], cc[i], dd[i], cr[i], ck[i], lo[i], hi[i]))), numeric(1))
  expect_identical(max(z0), 0)
  tl <- c(cr[1:100], cr[101:200] + 3)
  for (kind in 1:3) {
    ref <- vapply(seq_len(n), function(i) .cdp_ref_cell(kind, a[i], b[i], cc[i], dd[i], tl[i], ck[i], lo[i], hi[i], r[i]), numeric(1))
    tail_lp <- vapply(seq_len(n), function(i) cdp_log_cell_r(kind, a[i], b[i], cc[i], dd[i], tl[i], ck[i], lo[i], hi[i], r[i]),
                      numeric(1))
    closed <- vapply(seq_len(n), function(i) compute_rkg_probs_r(a[i], b[i], cc[i], dd[i], tl[i], ck[i], lo[i], hi[i], r[i])[kind],
                     numeric(1))
    ok <- ref > 1e-200
    expect_lt(max(abs(exp(tail_lp[ok]) - ref[ok]) / ref[ok]), 1e-9, label = paste("tail integral, cell kind", kind))
    expect_lt(max(abs(closed[ok] - ref[ok]) / (ref[ok] + 1e-4)), 1e-6, label = paste("closed form, cell kind", kind))
    if (kind == 1) expect_gt(sum(ok & ref < 1e-6), 10)
  }
})

test_that("CDP predictions with correlations: native band cells match the R loop", {
  d <- .cdp_rho_fixture()
  for (f in list(list(rho ~ 1), list(rho_target ~ 1, rho_lure ~ 1), list(rho_lure ~ 1))) {
    m <- suppressMessages(.cdp_rho_model(f, d))
    S <- 3; N <- 12
    set.seed(9)
    params <- list(dprime = matrix(rnorm(S * N, 1, 0.3), S), dprime2 = matrix(rnorm(S * N, 0.5, 0.3), S),
                   rec_crit = matrix(rnorm(S * N, 1.2, 0.3), S), know_crit = matrix(rnorm(S * N, 0.4, 0.3), S))
    if (isTRUE(m$model_data$has_rho)) params$rho <- matrix(runif(S * N, -0.99, 0.99), S)
    if (isTRUE(m$model_data$has_rho_N)) params$rho_N <- matrix(runif(S * N, -0.99, 0.99), S)
    thresh <- array(rep(c(-0.2, 0.6, 1.4), each = S * N), c(S, N, 3))
    is_old <- rep(0:1, length.out = N)
    K <- 4L
    nat <- probs_cdp(params, thresh, is_old, S, N, K, m)
    old <- options(bayesroc.cdp_native = FALSE)
    rl <- probs_cdp(params, thresh, is_old, S, N, K, m)
    options(old)
    expect_lt(max(abs(nat - rl)), 1e-12)
    expect_lt(max(abs(attr(nat, "p_remember") - attr(rl, "p_remember"))), 1e-12)
    if (isTRUE(m$model_data$cdp_rho_shared)) {
      p2 <- params; p2$rho_N <- params$rho
      m2 <- m; m2$model_data$cdp_rho_shared <- FALSE
      expect_identical(probs_cdp(p2, thresh, is_old, S, N, K, m2), nat)
    }
  }
})

test_that("JAX CDP kernel with a correlation matches the R reference cells", {
  skip_if_no_jax()
  lib <- .bvn_ffi_lib()
  for (l in list(lib, "")) {
    out <- system2(jax_python(), c(testthat::test_path("python", "cdp_corr_values.py"), dirname(find_backend_script()),
                                   shQuote(l)),
                   stdout = TRUE, stderr = FALSE)
    out <- paste(out[seq(grep("^RESULT ", out), length(out))], collapse = "")
    res <- jsonlite::fromJSON(sub("^RESULT ", "", out))
    expect_identical(res$ffi_loaded, nzchar(l))
    for (nm in c("rk", "rkg", "rkg_tail")) {
      x <- res[[nm]]
      lo <- as.numeric(x$tau_lo); hi <- as.numeric(x$tau_hi)
      ref <- log(vapply(seq_along(x$lp), function(i) {
        ks <- if (nm == "rk" && x$rk[i] == 2) 2:3 else x$rk[i]
        sum(vapply(ks, function(k) .cdp_ref_cell(k, x$mu_R[i], x$sigma_R[i], x$mu_F[i], x$sigma_F[i], x$c_R[i], x$c_K[i],
                                                  lo[i], hi[i], x$corr[i]), numeric(1)))
      }, numeric(1)))
      fin <- is.finite(ref) & ref > -460
      expect_lt(max(abs(exp(x$lp[fin]) - exp(ref[fin])) / (exp(ref[fin]) + 1e-4)), 1e-6,
                label = paste(nm, if (nzchar(l)) "native" else "pure JAX"))
    }
  }
})

test_that("cdp correlations: batch C++ matches plain Stan in log density and gradient", {
  skip_if_no_cmdstan()
  d3 <- .cdp_rho_fixture()
  d2 <- d3; d2$rk[d2$rk == 3] <- 2
  for (cs in list(list(d3, list(rho_target ~ 1 + (1 | sub), rho_lure ~ 1), TRUE),
                  list(d3, list(rho ~ 1), TRUE),
                  list(d2, list(rho_lure ~ 1 + (1 | sub)), FALSE))) {
    base <- list(resp(conf, rk) | old ~ 1, fam ~ 1, criterion ~ 1, rec_crit ~ 1, sigma_R ~ 1)
    if (cs[[3]]) base <- c(base, list(know_crit ~ 1))
    f <- do.call(brf, c(base, cs[[2]], list(family = cdp(old_levels = 2:4))))
    mp <- .broc_plain(f, data = cs[[1]]); mb <- broc(f, data = cs[[1]])
    r <- suppressMessages(.stan_parity(mp, mb))
    lab <- paste(vapply(cs[[2]], function(x) deparse(x[[2]]), ""), collapse = "+")
    expect_lt(r$lp, 1e-8, label = paste(lab, "log density"))
    expect_lt(r$grad, 1e-7, label = paste(lab, "gradient"))
    if (identical(cs[[2]][[1]][[2]], quote(rho_target))) {
      r2 <- suppressMessages(.stan_parity_at(mp, mb, list(`beta_rho[1]` = 2.6, `beta_rho_N[1]` = -2.4)))
      expect_lt(r2$lp, 1e-8, label = paste(lab, "log density at rho 0.989 / -0.984"))
      expect_lt(r2$grad, 1e-7, label = paste(lab, "gradient at rho 0.989 / -0.984"))
    }
  }
})
