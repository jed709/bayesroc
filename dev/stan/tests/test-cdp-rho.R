# Helpers shared with tests/testthat/
# test-cdp-rho.R (copied).

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
