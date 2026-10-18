# =============================================================================
# Effective sample size by batch means
# =============================================================================

.ess_batch <- function(N) {
  b <- max(100L, N %/% 10L)
  if (N >= 2L * b) b else N %/% 2L
}

.ess_bm <- function(x) {
  x <- as.matrix(x)
  N <- nrow(x)
  M <- ncol(x)
  b <- .ess_batch(N)
  if (b < 1L || !all(is.finite(x))) return(NA_real_)
  if (abs(max(x) - min(x)) < .Machine$double.eps) return(NA_real_)
  nb <- N %/% b
  bm <- colMeans(matrix(x[seq_len(nb * b), , drop = FALSE], nrow = b))
  vb <- b * stats::var(bm)
  if (!is.finite(vb) || vb <= 0) return(NA_real_)
  min(M * N * stats::var(as.vector(x)) / vb, M * N * log10(M * N))
}

.ess_z <- function(x) {
  x <- as.matrix(x)
  r <- rank(as.vector(x), ties.method = "average")
  matrix(stats::qnorm((r - 0.375) / (length(r) + 0.25)), nrow(x))
}

.ess_bulk <- function(x) {
  x <- as.matrix(x)
  if (!all(is.finite(x))) return(NA_real_)
  .ess_bm(.ess_z(x))
}

.ess_tail <- function(x) {
  x <- as.matrix(x)
  if (!all(is.finite(x))) return(NA_real_)
  q <- stats::quantile(x, c(0.05, 0.95), names = FALSE)
  min(.ess_bm((x <= q[1]) + 0), .ess_bm((x <= q[2]) + 0))
}

.summary_measures <- function(...) {
  m <- list(...)
  if (!length(m)) m <- as.list(c(posterior::default_summary_measures(), posterior::default_convergence_measures()))
  nm <- names(m) %||% rep("", length(m))
  for (k in c("ess_bulk", "ess_tail")) {
    i <- which(!nzchar(nm) & vapply(m, function(v) identical(v, k), TRUE))
    for (j in i) {
      m[[j]] <- if (k == "ess_bulk") .ess_bulk else .ess_tail
      nm[j] <- k
    }
  }
  names(m) <- nm
  m
}
