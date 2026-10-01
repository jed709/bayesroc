library(bayesroc)

.fixture_univariate <- function(seed = 1, dprime = 1.0, sigma = 1.0, n_per = 60) {
  set.seed(seed)
  thresh <- c(-1.2, -0.4, 0.3, 1.0, 1.8)
  y_old <- as.integer(cut(rnorm(n_per, dprime, sigma), c(-Inf, thresh, Inf)))
  y_new <- as.integer(cut(rnorm(n_per, 0,      1),     c(-Inf, thresh, Inf)))
  data.frame(
    item_resp = c(y_old, y_new),
    is_old    = c(rep(1L, n_per), rep(0L, n_per)),
    cond      = factor(c(rep("A", n_per), rep("B", n_per)))
  )
}

.fixture_encoding <- function(seed = 3, old_code = 1, new_code = 0, n_per = 40) {
  set.seed(seed)
  thresh <- c(-1.2, -0.4, 0.3, 1.0, 1.8)
  d_by_cond <- c(speak = 1.3, read = 0.7, sing = 1.1)
  targets <- do.call(rbind, lapply(names(d_by_cond), function(cn) {
    data.frame(item_resp = as.integer(cut(rnorm(n_per, d_by_cond[[cn]], 1),
                                          c(-Inf, thresh, Inf))),
               is_old = old_code, condition = cn)
  }))
  lures <- data.frame(
    item_resp = as.integer(cut(rnorm(n_per, 0, 1), c(-Inf, thresh, Inf))),
    is_old = new_code, condition = "new")
  d <- rbind(targets, lures)
  d$condition <- factor(d$condition, levels = c("new", "speak", "read", "sing"))
  d
}

.fixture_crossed <- function(seed = 5, n_sub = 8, n_item = 16) {
  set.seed(seed)
  thresh <- c(-1.2, -0.4, 0.3, 1.0, 1.8)
  sub_re  <- rnorm(n_sub, 0, 0.4)
  item_re <- rnorm(n_item, 0, 0.4)
  rows <- expand.grid(sub = seq_len(n_sub), item = seq_len(n_item))
  rows$is_old <- as.integer((rows$sub + rows$item) %% 2L == 0L)
  mu <- ifelse(rows$is_old == 1L, 1.0, 0) + sub_re[rows$sub] + item_re[rows$item]
  rows$item_resp <- as.integer(cut(rnorm(nrow(rows), mu, 1), c(-Inf, thresh, Inf)))
  rows$sub  <- factor(rows$sub)
  rows$item <- factor(rows$item)
  rows
}

.fixture_bivariate_sdt <- function(seed = 2, n_per = 60) {
  set.seed(seed)
  K1 <- 4; K2 <- 4
  c1 <- c(-1, 0, 1)
  c2 <- c(-0.5, 0.3, 1.2)
  N <- 3 * n_per
  it <- factor(rep(c("new", "A", "B"), each = n_per),
               levels = c("new", "A", "B"))
  z1 <- ifelse(it == "new", rnorm(N, 0, 1), rnorm(N, 0.8, 1))
  z2 <- ifelse(it == "A",  rnorm(N,  0.6, 1),
        ifelse(it == "B",  rnorm(N, -0.6, 1),
                           rnorm(N,  0,   1)))
  y1 <- as.integer(cut(z1, c(-Inf, c1, Inf)))
  y2 <- as.integer(cut(z2, c(-Inf, c2, Inf)))
  data.frame(y1 = y1, y2 = y2, item_type = it)
}

.fixture_bivariate_cumulative <- function(seed = 3, n = 200) {
  set.seed(seed)
  rho <- 0.5
  z <- matrix(rnorm(2 * n), nrow = n)
  z[, 2] <- rho * z[, 1] + sqrt(1 - rho^2) * z[, 2]
  c1 <- c(-1, 0, 1); c2 <- c(-0.5, 0.3, 1.2)
  y1 <- as.integer(cut(z[, 1], c(-Inf, c1, Inf)))
  y2 <- as.integer(cut(z[, 2], c(-Inf, c2, Inf)))
  data.frame(y1 = y1, y2 = y2,
             cond = factor(rep(c("A", "B"), each = n / 2)))
}

.fixture_cdp <- function() {
  data(rotello_2005, package = "bayesroc", envir = environment())
  get("rotello_2005", envir = environment())
}

.fixture_source_mixture <- function(seed = 4, n_per = 80) {
  set.seed(seed)
  K <- 6
  thresh <- c(-1.2, -0.4, 0.3, 1.0, 1.8)
  z_a <- rnorm(n_per,  0.8, 1)
  z_b <- rnorm(n_per, -0.8, 1)
  y_a <- as.integer(cut(z_a, c(-Inf, thresh, Inf)))
  y_b <- as.integer(cut(z_b, c(-Inf, thresh, Inf)))
  data.frame(item_resp = c(y_a, y_b),
             source    = factor(c(rep("A", n_per), rep("B", n_per))))
}
