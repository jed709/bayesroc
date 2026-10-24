.sim_metad <- function(n_sub = 12, n_item = 24, K = 6, d = 1.5, c = 0.8, m_old = 0.8, m_new = 0.8,
                       sd_sub = c(d = 0.3, c = 0.2, m = 0.25), sd_item = c(d = 0.2, c = 0.1, m = 0.15),
                       relative = FALSE, centred = FALSE, gaps = rep(0.45, K / 2 - 1), seed = 1,
                       link = "probit") {
  pl <- switch(link, probit = stats::pnorm, logit = stats::plogis, cloglog = function(x) -expm1(-exp(x)))
  ql <- switch(link, probit = stats::qnorm, logit = stats::qlogis, cloglog = function(p) log(-log1p(-p)))
  set.seed(seed)
  dat <- expand.grid(sub = factor(seq_len(n_sub)), item = factor(seq_len(n_item)))
  dat$old <- as.integer((as.integer(dat$item) + as.integer(dat$sub)) %% 2 == 0)
  dat$cond <- factor(ifelse(as.integer(dat$item) <= n_item / 2, "a", "b"))
  dat$x <- round(stats::rnorm(nrow(dat)), 3)
  us <- vapply(sd_sub, function(s) stats::rnorm(n_sub, 0, s), numeric(n_sub))
  ui <- vapply(sd_item, function(s) stats::rnorm(n_item, 0, s), numeric(n_item))
  is <- as.integer(dat$sub); ii <- as.integer(dat$item)
  dd <- d + us[is, "d"] + ui[ii, "d"]
  cc <- c + us[is, "c"] + ui[ii, "c"]
  lm <- us[is, "m"] + ui[ii, "m"]
  n <- nrow(dat)
  x <- if (link == "probit") stats::rnorm(n, dat$old * dd) else dat$old * dd + ql(stats::runif(n))
  r <- x > cc
  M <- ifelse(r, m_old, m_new) * exp(lm)
  mid <- dd / 2
  mu_m <- mid + M * dd * (dat$old - 0.5)
  mc <- if (relative) mid + M * (cc - mid) else cc
  pc <- pl(mc - mu_m)
  u <- stats::runif(n)
  xm <- mu_m + ql(ifelse(r, pc + u * (1 - pc), u * pc))
  up <- cumsum(gaps)
  dat$conf <- ifelse(r, K / 2 + 1 + vapply(seq_len(n), function(i) sum(xm[i] > mc[i] + up), 0),
                     K / 2 - vapply(seq_len(n), function(i) sum(xm[i] < mc[i] - up), 0))
  if (centred) dat$old <- dat$old - 0.5
  dat
}

.sim_gnoise <- function(n_sub = 12, n_item = 24, K = 6, d = 1.5, c = 0.8, sm = 0.8,
                        sd_sub = c(d = 0.3, c = 0.2, s = 0.25), gaps = rep(0.45, K / 2 - 1), seed = 1) {
  set.seed(seed)
  dat <- expand.grid(sub = factor(seq_len(n_sub)), item = factor(seq_len(n_item)))
  dat$old <- as.integer((as.integer(dat$item) + as.integer(dat$sub)) %% 2 == 0)
  dat$cond <- factor(ifelse(as.integer(dat$item) <= n_item / 2, "a", "b"))
  us <- vapply(sd_sub, function(s) stats::rnorm(n_sub, 0, s), numeric(n_sub))
  is <- as.integer(dat$sub)
  dd <- d + us[is, "d"]; cc <- c + us[is, "c"]; ss <- sm * exp(us[is, "s"])
  n <- nrow(dat)
  x <- stats::rnorm(n, dat$old * dd)
  y <- x + stats::rnorm(n, 0, ss)
  up <- cumsum(gaps)
  dat$conf <- ifelse(x > cc, K / 2 + 1 + vapply(seq_len(n), function(i) sum(y[i] > cc[i] + up), 0),
                     K / 2 - vapply(seq_len(n), function(i) sum(y[i] < cc[i] - up), 0))
  dat
}

.sim_lnoise <- function(n_sub = 12, n_item = 24, K = 6, d = 1.5, c = 0.8, sm = 0.6,
                        sd_sub = c(d = 0.3, c = 0.2, s = 0.25), gaps = rep(0.45, K / 2 - 1), seed = 1) {
  set.seed(seed)
  dat <- expand.grid(sub = factor(seq_len(n_sub)), item = factor(seq_len(n_item)))
  dat$old <- as.integer((as.integer(dat$item) + as.integer(dat$sub)) %% 2 == 0)
  dat$cond <- factor(ifelse(as.integer(dat$item) <= n_item / 2, "a", "b"))
  us <- vapply(sd_sub, function(s) stats::rnorm(n_sub, 0, s), numeric(n_sub))
  is <- as.integer(dat$sub)
  dd <- d + us[is, "d"]; cc <- c + us[is, "c"]; ss <- sm * exp(us[is, "s"])
  n <- nrow(dat)
  x <- stats::rnorm(n, dat$old * dd)
  e <- exp(ss * stats::rnorm(n))
  up <- cumsum(gaps)
  dat$conf <- ifelse(x > cc, K / 2 + 1 + vapply(seq_len(n), function(i) sum(x[i] > cc[i] + up * e[i]), 0),
                     K / 2 - vapply(seq_len(n), function(i) sum(x[i] < cc[i] - up * e[i]), 0))
  dat
}

.gnoise_quad <- function(d, sm, tau, s, K) {
  h <- K %/% 2
  mu <- s * d
  vapply(seq_len(K), function(k) {
    lo <- if (k == 1 || k == h + 1) -Inf else tau[k - 1]
    hi <- if (k == K || k == h) Inf else tau[k]
    f <- function(x) stats::dnorm(x, mu) * (stats::pnorm(hi, x, sm) - stats::pnorm(lo, x, sm))
    if (k > h) stats::integrate(f, tau[h], Inf, rel.tol = 1e-12)$value
    else stats::integrate(f, -Inf, tau[h], rel.tol = 1e-12)$value
  }, numeric(1))
}

.sim_mafc <- function(n_sub = 12, n_trial = 60, m = 3, d = 1.6, sigma = 1, bias = c(0.3, -0.2),
                      sd_sub = c(d = 0.3, s = 0.2, b = 0.25), seed = 1) {
  set.seed(seed)
  dat <- expand.grid(trial = seq_len(n_trial), sub = factor(seq_len(n_sub)))
  dat$cond <- factor(ifelse(dat$trial %% 2 == 0, "a", "b"))
  n <- nrow(dat)
  is <- as.integer(dat$sub)
  dd <- d + stats::rnorm(n_sub, 0, sd_sub[["d"]])[is]
  ss <- sigma * exp(stats::rnorm(n_sub, 0, sd_sub[["s"]]))[is]
  b <- c(bias, 0)
  bu <- matrix(stats::rnorm(n_sub * (m - 1), 0, sd_sub[["b"]]), n_sub, m - 1)
  B <- sweep(cbind(bu, 0)[is, , drop = FALSE], 2, b, "+")
  dat$target <- sample.int(m, n, TRUE)
  X <- matrix(stats::rnorm(n * m), n, m)
  X[cbind(seq_len(n), dat$target)] <- dd + ss * stats::rnorm(n)
  dat$choice <- max.col(X + B, ties.method = "first")
  dat
}
