.sim_lineup <- function(n_set = 24, n_per = 30, sizes = c(1L, 3L, 6L), d = 1.4, sigma = 1.2, d_i = 0.3, sigma_i = 0.9,
                        sigma_b = 0, tau = c(-0.4, 0.5, 1.1, 1.8), R = 2L, designated = TRUE,
                        sd_set = c(d = 0.3, c = 0.2), seed = 1) {
  set.seed(seed)
  n <- n_set * n_per
  dat <- data.frame(set = factor(rep(seq_len(n_set), each = n_per)), witness = factor(seq_len(n)))
  dat$k <- sample(sizes, n, replace = TRUE)
  dat$present <- stats::rbinom(n, 1, 0.5)
  dat$designated <- if (designated) dat$present == 0 & stats::runif(n) < 0.7 else FALSE
  us <- vapply(sd_set, function(s) stats::rnorm(n_set, 0, s), numeric(n_set))
  is <- as.integer(dat$set)
  dd <- d + us[is, "d"]
  sh <- us[is, "c"]
  b <- sigma_b * stats::rnorm(n)
  id <- character(n)
  conf <- integer(n)
  for (i in seq_len(n)) {
    w <- stats::rnorm(dat$k[i])
    if (dat$present[i] == 1) w[1] <- dd[i] + sigma * w[1] else if (dat$designated[i]) w[1] <- d_i + sigma_i * w[1]
    x <- max(w) + b[i]
    pos <- findInterval(x, tau + sh[i]) + 1L
    if (pos <= R) {
      id[i] <- "reject"
      conf[i] <- if (R == 1L) NA_integer_ else R + 1L - pos
    } else {
      id[i] <- if (which.max(w) == 1L && (dat$present[i] == 1 || dat$designated[i] || dat$k[i] == 1L)) "suspect" else "filler"
      conf[i] <- pos - R
    }
  }
  dat$id <- id
  dat$conf <- conf
  dat$x <- round(stats::rnorm(n), 3)
  dat
}
