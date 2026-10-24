#' Bivariate Normal Utility Functions (R-side)
#'
#' Pure R implementations of bivariate normal CDF and cell probability functions,
#' mirroring the Stan functions in generate_stan.R for use in predict_broc() and
#' pp_check_broc().

# =============================================================================
# Owen's T Function
# =============================================================================

#' Owen's T function via Gauss-Legendre quadrature (20 points)
#'
#' Computes T(h, a) = 1/(2*pi) * integral_0^a exp(-h^2*(1+x^2)/2) / (1+x^2) dx
#' Uses the same quadrature approach as Stan's owens_t().
#'
#' @param h Numeric scalar or vector
#' @param a Numeric scalar or vector (same length as h)
#' @return Numeric vector of Owen's T values
#' @noRd
.gl20_nodes <- c(
  -0.9931285991850949, -0.9639719272779138, -0.9122344282513259,
  -0.8391169718222188, -0.7463319064601508, -0.6360536807265150,
  -0.5108670019508271, -0.3737060887154195, -0.2277858511416451,
  -0.0765265211334973,
   0.0765265211334973,  0.2277858511416451,  0.3737060887154195,
   0.5108670019508271,  0.6360536807265150,  0.7463319064601508,
   0.8391169718222188,  0.9122344282513259,  0.9639719272779138,
   0.9931285991850949
)
.gl20_weights <- c(
  0.0176140071391521, 0.0406014298003869, 0.0626720483341091,
  0.0832767415767048, 0.1019301198172404, 0.1181945319615184,
  0.1316886384491766, 0.1420961093183820, 0.1491729864726037,
  0.1527533871307258,
  0.1527533871307258, 0.1491729864726037, 0.1420961093183820,
  0.1316886384491766, 0.1181945319615184, 0.1019301198172404,
  0.0832767415767048, 0.0626720483341091, 0.0406014298003869,
  0.0176140071391521
)

#' Owen's T function -- fully vectorized over h and a
#' @noRd
owens_t_r <- function(h, a) {
  n <- length(h)
  result <- numeric(n)

  zero_a <- a == 0
  inf_h <- is.infinite(h)
  inf_a <- is.infinite(a)

  result[zero_a] <- 0
  result[inf_h & !zero_a] <- 0
  mask_inf_a <- inf_a & !zero_a & !inf_h
  if (any(mask_inf_a)) {
    result[mask_inf_a] <- sign(a[mask_inf_a]) * (1 - pnorm(abs(h[mask_inf_a]))) / 2
  }

  todo <- !zero_a & !inf_h & !inf_a
  if (!any(todo)) return(result)

  h_t <- h[todo]
  a_t <- a[todo]

  big <- abs(a_t) > 1
  small <- !big

  if (any(small)) {
    result[which(todo)[small]] <- owens_t_r_core_vec(h_t[small], a_t[small])
  }

  if (any(big)) {
    sgn <- sign(a_t[big])
    a_pos <- abs(a_t[big])
    h_b <- h_t[big]
    ph <- pnorm(h_b)
    pah <- pnorm(a_pos * h_b)
    rhs <- 0.5 * (ph + pah) - ph * pah
    t_reduced <- owens_t_r_core_vec(a_pos * h_b, 1 / a_pos)
    result[which(todo)[big]] <- sgn * (rhs - t_reduced)
  }

  result
}

#' Core vectorized Owen's T via GL quadrature (for |a| <= 1)
#' @param h Numeric vector
#' @param a Numeric vector (same length, |a| <= 1)
#' @return Numeric vector
#' @noRd
owens_t_r_core_vec <- function(h, a) {
  n <- length(h)
  mid <- a / 2
  half <- a / 2

  x_mat <- outer(mid, .gl20_nodes, "+") + outer(half - mid, .gl20_nodes, "*")
  x_mat <- outer(half, .gl20_nodes, "*") + mid

  w_mat <- outer(half, .gl20_weights, "*")

  h2 <- h^2
  x2 <- x_mat^2
  integrand <- exp(-h2 * (1 + x2) / 2) / (1 + x2)

  rowSums(w_mat * integrand) / (2 * pi)
}

# =============================================================================
# Bivariate Normal CDF
# =============================================================================

#' Bivariate normal CDF: P(Z1 <= z1, Z2 <= z2) with correlation rho
#'
#' Uses Owen's T formula with safe handling of edge cases.
#'
#' @param z1 Numeric (scalar or vector)
#' @param z2 Numeric (scalar or vector, same length as z1)
#' @param rho Numeric (scalar or vector, same length as z1)
#' @return Numeric vector of bivariate normal CDF values
#' @noRd
binormal_cdf_r <- function(z1, z2, rho) {
  n <- length(z1)
  result <- numeric(n)

  inf1_neg <- is.infinite(z1) & z1 < 0
  inf1_pos <- is.infinite(z1) & z1 > 0
  inf2_neg <- is.infinite(z2) & z2 < 0
  inf2_pos <- is.infinite(z2) & z2 > 0

  result[inf1_neg | inf2_neg] <- 0
  mask_inf1p <- inf1_pos & !inf2_neg
  mask_inf2p <- inf2_pos & !inf1_neg & !inf1_pos
  if (any(mask_inf1p)) result[mask_inf1p] <- .pnorm_v(z2[mask_inf1p])
  if (any(mask_inf2p)) result[mask_inf2p] <- .pnorm_v(z1[mask_inf2p])

  finite <- !is.infinite(z1) & !is.infinite(z2)
  if (!any(finite)) return(result)

  z1f <- z1[finite]
  z2f <- z2[finite]
  rhof <- rho[finite]

  if (.bvn_native_ready() && isTRUE(all(abs(rhof) < 1))) {
    result[finite] <- .bvn_native_cdf(z1f, z2f, rhof)
    return(result)
  }

  z1f[z1f == 0] <- 1e-10
  z2f[z2f == 0] <- 1e-10

  denom <- sqrt((1 + rhof) * (1 - rhof))
  a1 <- (z2f / z1f - rhof) / denom
  a2 <- (z1f / z2f - rhof) / denom
  product <- z1f * z2f
  delta <- as.numeric(product < 0 | (product == 0 & (z1f + z2f) < 0))

  p <- 0.5 * (pnorm(z1f) + pnorm(z2f) - delta) - owens_t_r(z1f, a1) - owens_t_r(z2f, a2)
  for (i in which(p < 1e-6 & abs(rhof) < 1)) {
    s <- sqrt((1 - rhof[i]) * (1 + rhof[i]))
    p[i] <- exp(.log_lin_integral_r(-Inf, 0, z2[finite][i] / s, -rhof[i] / s, -Inf, z1[finite][i]))
  }
  result[finite] <- p
  result
}

# =============================================================================
# Native Bivariate Normal CDF
# =============================================================================

.bvn_native <- new.env(parent = emptyenv())

# Whether the native kernel library (the one the JAX backend builds) can serve Phi2 and the prediction
# kernels to R. Without it, or with options(bayesroc.native_predict = FALSE), the R paths are used. A library
# named by BAYESROC_BVN_LIB is taken only when it carries every kernel; otherwise the library of the current
# sources is built (or found in the cache). A session that ends up on the R path says so once.
.bvn_native_ready <- function() {
  st <- .bvn_native
  # a cluster worker inherits this environment with dead symbol pointers: reload in the new process
  if (!is.null(st$ok) && identical(st$pid, Sys.getpid())) return(st$ok)
  st$ok <- FALSE
  st$pid <- Sys.getpid()
  if (isFALSE(getOption("bayesroc.native_predict", TRUE))) return(FALSE)
  load_lib <- function(lib) {
    tryCatch({
      dll <- dyn.load(lib)
      sym <- function(nm) getNativeSymbolInfo(nm, PACKAGE = dll)
      st$sym <- sym("bayesroc_r_bvn_cdf")
      st$sym_pnorm <- sym("bayesroc_r_pnorm")
      st$sym_rect <- sym("bayesroc_r_bvn_rect")
      st$sym_surv <- sym("bayesroc_r_det_survival")
      st$sym_srcsurv <- sym("bayesroc_r_src_survival")
      st$sym_srcsurv_g <- sym("bayesroc_r_src_survival_gauss")
      st$sym_exp <- sym("bayesroc_r_exp")
      st$sym_cdp <- sym("bayesroc_r_cdp_probs")
      st$sym_mixsurv <- sym("bayesroc_r_mix_survival")
      st$sym_lnoise <- sym("bayesroc_r_lnoise_cell")
      TRUE
    }, error = function(e) FALSE)
  }
  ok <- FALSE
  lib <- Sys.getenv("BAYESROC_BVN_LIB", "")
  if (nzchar(lib) && file.exists(lib)) ok <- load_lib(lib)
  if (!ok) {
    why <- NULL
    lib <- tryCatch(.bvn_ffi_lib(), error = function(e) { why <<- conditionMessage(e); NULL })
    if (!is.null(lib) && nzchar(lib) && file.exists(lib)) {
      ok <- load_lib(lib)
      if (!ok) why <- paste("could not load", lib)
    } else if (is.null(why)) {
      why <- "no C++ toolchain or JAX environment to build it"
    }
    if (!ok) message("bayesroc: native prediction kernels unavailable (", why, "); bivariate predictions take the R path, ",
                     "which is many times slower.")
  }
  if (ok) Sys.setenv(BAYESROC_BVN_LIB = lib)
  st$ok <- ok
  ok
}

.bvn_native_cdf <- function(z1, z2, rho) {
  n <- length(z1)
  .C(.bvn_native$sym, as.double(z1), as.double(z2), as.double(rho), as.integer(n),
     .native_threads(), p = double(n), NAOK = TRUE)$p
}

.native_threads <- function() {
  max(1L, as.integer(.bvn_native$threads %||% getOption("bayesroc.predict_threads", 1L)))
}

# exp() for long vectors (the threshold gaps), threaded through the native library when it is there
.exp_v <- function(x) {
  n <- length(x)
  if (n < 50000L || !.bvn_native_ready() || is.null(.bvn_native$sym_exp)) return(exp(x))
  out <- .C(.bvn_native$sym_exp, as.double(x), as.integer(n), .native_threads(), p = double(n), NAOK = TRUE)$p
  dim(out) <- dim(x)
  out
}

# stats::pnorm() for long vectors, threaded through the native library when it is there
.pnorm_v <- function(x) {
  n <- length(x)
  if (n < 50000L || !.bvn_native_ready()) return(stats::pnorm(x))
  .C(.bvn_native$sym_pnorm, as.double(x), as.integer(n), .native_threads(), p = double(n), NAOK = TRUE)$p
}

# =============================================================================
# Vectorised Cell Probabilities
# =============================================================================

# P(z1lo < Z1 <= z1hi, z2lo < Z2 <= z2hi) elementwise for a standard bivariate normal; edges
# may be infinite. Independent where rho is 0.
.bvn_rect_v <- function(z1lo, z1hi, z2lo, z2hi, rho) {
  n <- length(rho)
  if (n && .bvn_native_ready() && isTRUE(all(abs(rho) < 1)) &&
      !anyNA(z1lo) && !anyNA(z1hi) && !anyNA(z2lo) && !anyNA(z2hi)) {
    edge <- function(x) pmin(pmax(x, -1e10), 1e10)
    return(.C(.bvn_native$sym_rect, edge(z1lo), edge(z1hi), edge(z2lo), edge(z2hi), as.double(rho),
              as.integer(n), .native_threads(), p = double(n))$p)
  }
  out <- (.pnorm_v(z1hi) - .pnorm_v(z1lo)) * (.pnorm_v(z2hi) - .pnorm_v(z2lo))
  nz <- which(rho != 0)
  m <- length(nz)
  if (m) {
    a <- z1lo[nz]; b <- z1hi[nz]; cc <- z2lo[nz]; d <- z2hi[nz]
    p <- binormal_cdf_r(c(b, b, a, a), c(d, cc, d, cc), rep(rho[nz], 4L))
    i <- seq_len(m)
    out[nz] <- p[i] - p[m + i] - p[2L * m + i] + p[3L * m + i]
  }
  out
}

# compute_bounded_prob_r() elementwise: the cell [y_lo, y_hi] x [c2_lo, c2_hi] on the raw
# scale (edges may be infinite).
.bounded_cell_v <- function(y_lo, y_hi, c2_lo, c2_hi, mu_I, mu_S, sigma_I, sigma_S, rho) {
  z1lo <- (y_lo - mu_I) / sigma_I
  z1hi <- (y_hi - mu_I) / sigma_I
  z2lo <- (c2_lo - mu_S) / sigma_S
  z2hi <- (c2_hi - mu_S) / sigma_S
  out <- (.pnorm_v(z1hi) - .pnorm_v(z1lo)) * (.pnorm_v(z2hi) - .pnorm_v(z2lo))
  dep <- which(abs(rho) >= 1e-10)
  if (!length(dep)) return(out)
  r <- rho[dep]; mI <- mu_I[dep]; sI <- sigma_I[dep]; sS <- sigma_S[dep]
  cp <- mI - sI * mu_S[dep] / (r * sS)
  s_cond <- sS * sqrt(1 - r * r)
  src_below <- .pnorm_v(c2_hi[dep] / s_cond) - .pnorm_v(c2_lo[dep] / s_cond)
  ylo <- y_lo[dep]; yhi <- y_hi[dep]
  a1 <- z1lo[dep]; b1 <- z1hi[dep]; a2 <- z2lo[dep]; b2 <- z2hi[dep]
  below <- yhi <= cp
  above <- !below & ylo >= cp
  res <- (.pnorm_v(b1) - .pnorm_v(a1)) * src_below
  i <- which(above)
  if (length(i)) res[i] <- .bvn_rect_v(a1[i], b1[i], a2[i], b2[i], r[i])
  i <- which(!below & !above)
  if (length(i)) {
    z_cp <- (cp[i] - mI[i]) / sI[i]
    z_lo_b <- ifelse(is.infinite(ylo[i]), -8.0, a1[i])
    res[i] <- (.pnorm_v(z_cp) - .pnorm_v(z_lo_b)) * src_below[i] + .bvn_rect_v(z_cp, b1[i], a2[i], b2[i], r[i])
  }
  out[dep] <- res
  out
}

# compute_bounded_marginal_source_r() elementwise, for the source cell [c2_lo, c2_hi].
.bounded_marginal_source_v <- function(c2_lo, c2_hi, mu_I, mu_S, sigma_I, sigma_S, rho) {
  z2lo <- (c2_lo - mu_S) / sigma_S
  z2hi <- (c2_hi - mu_S) / sigma_S
  out <- .pnorm_v(z2hi) - .pnorm_v(z2lo)
  dep <- which(abs(rho) >= 1e-10)
  if (!length(dep)) return(out)
  r <- rho[dep]; sS <- sigma_S[dep]
  z_cp <- -mu_S[dep] / (r * sS)
  s_cond <- sS * sqrt(1 - r * r)
  out[dep] <- .pnorm_v(z_cp) * (.pnorm_v(c2_hi[dep] / s_cond) - .pnorm_v(c2_lo[dep] / s_cond)) +
    .bvn_rect_v(z_cp, rep(Inf, length(dep)), z2lo[dep], z2hi[dep], r)
  out
}

# =============================================================================
# Bivariate Cell Probability (9 cases)
# =============================================================================

#' Compute bivariate cell probability for response (resp1, resp2)
#'
#' Standardizes thresholds and handles all 9 cases (4 corners + 4 edges + 1 interior).
#'
#' @param resp1 Integer response on dimension 1 (1..K1)
#' @param resp2 Integer response on dimension 2 (1..K2)
#' @param K1 Number of categories on dimension 1
#' @param K2 Number of categories on dimension 2
#' @param mu1,mu2 Means on each dimension
#' @param sigma1,sigma2 SDs on each dimension
#' @param c1 Numeric vector of K1-1 thresholds for dimension 1
#' @param c2 Numeric vector of K2-1 thresholds for dimension 2
#' @param rho Correlation between dimensions
#' @return Scalar probability
#' @noRd
compute_bivariate_prob_r <- function(resp1, resp2, K1, K2,
                                      mu1, mu2, sigma1, sigma2,
                                      c1, c2, rho) {
  z1 <- (c1 - mu1) / sigma1
  z2 <- (c2 - mu2) / sigma2

  if (rho == 0) {
    p1 <- univariate_cell_prob_from_z(resp1, K1, z1)
    p2 <- univariate_cell_prob_from_z(resp2, K2, z2)
    return(p1 * p2)
  }

  if (resp1 == 1 && resp2 == 1) {
    return(binormal_cdf_r(z1[1], z2[1], rho))
  }
  if (resp1 == 1 && resp2 == K2) {
    return(pnorm(z1[1]) - binormal_cdf_r(z1[1], z2[K2 - 1], rho))
  }
  if (resp1 == K1 && resp2 == 1) {
    return(pnorm(z2[1]) - binormal_cdf_r(z2[1], z1[K1 - 1], rho))
  }
  if (resp1 == K1 && resp2 == K2) {
    return(1 - pnorm(z1[K1 - 1]) - pnorm(z2[K2 - 1]) +
             binormal_cdf_r(z1[K1 - 1], z2[K2 - 1], rho))
  }
  if (resp1 == 1) {
    return(binormal_cdf_r(z1[1], z2[resp2], rho) -
             binormal_cdf_r(z1[1], z2[resp2 - 1], rho))
  }
  if (resp2 == 1) {
    return(binormal_cdf_r(z1[resp1], z2[1], rho) -
             binormal_cdf_r(z1[resp1 - 1], z2[1], rho))
  }
  if (resp1 == K1) {
    return(pnorm(z2[resp2]) - pnorm(z2[resp2 - 1]) -
             binormal_cdf_r(z1[K1 - 1], z2[resp2], rho) +
             binormal_cdf_r(z1[K1 - 1], z2[resp2 - 1], rho))
  }
  if (resp2 == K2) {
    return(pnorm(z1[resp1]) - pnorm(z1[resp1 - 1]) -
             binormal_cdf_r(z2[K2 - 1], z1[resp1], rho) +
             binormal_cdf_r(z1[resp1 - 1], z2[K2 - 1], rho))
  }
  binormal_cdf_r(z1[resp1], z2[resp2], rho) -
    binormal_cdf_r(z1[resp1], z2[resp2 - 1], rho) -
    binormal_cdf_r(z1[resp1 - 1], z2[resp2], rho) +
    binormal_cdf_r(z1[resp1 - 1], z2[resp2 - 1], rho)
}

#' Helper: univariate cell probability from standardized thresholds
#' @noRd
univariate_cell_prob_from_z <- function(resp, K, z) {
  if (resp == 1) return(pnorm(z[1]))
  if (resp == K) return(1 - pnorm(z[K - 1]))
  pnorm(z[resp]) - pnorm(z[resp - 1])
}

# =============================================================================
# Bounded Bivariate Probability
# =============================================================================

#' Bounded bivariate probability: conditional source mean clamped at 0
#'
#' When rho != 0, identifies a crossing point cp where the conditional source
#' mean mu_{S|I}(y) = mu_S + rho*(sigma_S/sigma_I)*(y - mu_I) crosses zero.
#' Below cp: independent bivariate with source ~ N(0, sigma_cond).
#' Above cp: standard bivariate.
#' Straddle case: split integral at crossing point.
#'
#' @param resp1 Integer detection response (1..K1)
#' @param resp2 Integer source response (1..K2)
#' @param K1,K2 Number of categories
#' @param mu_I,mu_S Means
#' @param sigma_I,sigma_S SDs
#' @param c1 Numeric vector of K1-1 detection thresholds
#' @param c2 Numeric vector of K2-1 source thresholds
#' @param rho Correlation
#' @return Scalar probability
#' @noRd
compute_bounded_prob_r <- function(resp1, resp2, K1, K2,
                                    mu_I, mu_S, sigma_I, sigma_S,
                                    c1, c2, rho) {
  if (abs(rho) < 1e-10) {
    return(compute_bivariate_prob_r(resp1, resp2, K1, K2,
                                     mu_I, mu_S, sigma_I, sigma_S,
                                     c1, c2, 0.0))
  }

  cp <- mu_I - sigma_I * mu_S / (rho * sigma_S)
  y_lo <- if (resp1 == 1)  -Inf else c1[resp1 - 1]
  y_hi <- if (resp1 == K1)  Inf else c1[resp1]

  below_only <- y_hi <= cp
  above_only <- y_lo >= cp

  sigma_cond <- sigma_S * sqrt(1 - rho * rho)
  prob <- 0.0

  if (below_only) {
    prob <- compute_bivariate_prob_r(resp1, resp2, K1, K2,
                                      mu_I, 0.0, sigma_I, sigma_cond,
                                      c1, c2, 0.0)
  } else if (above_only) {
    prob <- compute_bivariate_prob_r(resp1, resp2, K1, K2,
                                      mu_I, mu_S, sigma_I, sigma_S,
                                      c1, c2, rho)
  } else {
    z_cp <- (cp - mu_I) / sigma_I

    z_ylo_b <- if (resp1 == 1) -8.0 else (c1[resp1 - 1] - mu_I) / sigma_I
    prob_below <- 0.0
    if (resp2 == 1) {
      zs_hi <- c2[1] / sigma_cond
      prob_below <- (pnorm(z_cp) - pnorm(z_ylo_b)) * pnorm(zs_hi)
    } else if (resp2 == K2) {
      zs_lo <- c2[K2 - 1] / sigma_cond
      prob_below <- (pnorm(z_cp) - pnorm(z_ylo_b)) * (1 - pnorm(zs_lo))
    } else {
      zs_lo <- c2[resp2 - 1] / sigma_cond
      zs_hi <- c2[resp2] / sigma_cond
      prob_below <- (pnorm(z_cp) - pnorm(z_ylo_b)) * (pnorm(zs_hi) - pnorm(zs_lo))
    }
    prob <- prob + prob_below

    z_yhi_a <- if (resp1 == K1) 8.0 else (c1[resp1] - mu_I) / sigma_I
    z2_a <- (c2 - mu_S) / sigma_S
    prob_above <- 0.0

    if (resp2 == 1 && resp1 == K1) {
      prob_above <- pnorm(z2_a[1]) - binormal_cdf_r(z_cp, z2_a[1], rho)
    } else if (resp2 == K2 && resp1 == K1) {
      prob_above <- 1 - pnorm(z2_a[K2 - 1]) - pnorm(z_cp) +
                    binormal_cdf_r(z_cp, z2_a[K2 - 1], rho)
    } else if (resp2 == 1) {
      prob_above <- binormal_cdf_r(z_yhi_a, z2_a[1], rho) -
                    binormal_cdf_r(z_cp, z2_a[1], rho)
    } else if (resp2 == K2) {
      prob_above <- pnorm(z_yhi_a) - pnorm(z_cp) -
                    binormal_cdf_r(z_yhi_a, z2_a[K2 - 1], rho) +
                    binormal_cdf_r(z_cp, z2_a[K2 - 1], rho)
    } else if (resp1 == K1) {
      prob_above <- pnorm(z2_a[resp2]) - pnorm(z2_a[resp2 - 1]) -
                    binormal_cdf_r(z_cp, z2_a[resp2], rho) +
                    binormal_cdf_r(z_cp, z2_a[resp2 - 1], rho)
    } else {
      prob_above <- binormal_cdf_r(z_yhi_a, z2_a[resp2], rho) -
                    binormal_cdf_r(z_yhi_a, z2_a[resp2 - 1], rho) -
                    binormal_cdf_r(z_cp, z2_a[resp2], rho) +
                    binormal_cdf_r(z_cp, z2_a[resp2 - 1], rho)
    }
    prob <- prob + prob_above
  }

  prob
}

# =============================================================================
# Bounded Marginal Source Probability
# =============================================================================

#' Bounded marginal source probability for BDP p2 component
#'
#' When item is recollected (placed in highest detection bin), source response
#' comes from the marginal source distribution integrated over item dimension.
#' Below the crossing point: source ~ N(0, sigma_cond).
#' Above the crossing point: standard bivariate marginal.
#'
#' @param resp_src Integer source response (1..K2)
#' @param K2 Number of source categories
#' @param mu_I,mu_S Means
#' @param sigma_I,sigma_S SDs
#' @param rho Correlation
#' @param c2 Numeric vector of K2-1 source thresholds
#' @return Scalar probability
#' @noRd
compute_bounded_marginal_source_r <- function(resp_src, K2, mu_I, mu_S,
                                                sigma_I, sigma_S, rho, c2) {
  if (abs(rho) < 1e-10) {
    return(univariate_cell_prob_r(resp_src, K2, mu_S, sigma_S, c2))
  }

  cp <- mu_I - sigma_I * mu_S / (rho * sigma_S)
  sigma_cond <- sigma_S * sqrt(1 - rho * rho)
  z_cp <- (cp - mu_I) / sigma_I
  p_below <- pnorm(z_cp)

  prob <- 0

  if (resp_src == 1) {
    prob <- prob + p_below * pnorm(c2[1] / sigma_cond)
  } else if (resp_src == K2) {
    prob <- prob + p_below * (1 - pnorm(c2[K2 - 1] / sigma_cond))
  } else {
    prob <- prob + p_below * (pnorm(c2[resp_src] / sigma_cond) - pnorm(c2[resp_src - 1] / sigma_cond))
  }

  z2 <- (c2 - mu_S) / sigma_S

  if (resp_src == 1) {
    prob <- prob + pnorm(z2[1]) - binormal_cdf_r(z_cp, z2[1], rho)
  } else if (resp_src == K2) {
    prob <- prob + 1 - pnorm(z2[K2 - 1]) - pnorm(z_cp) + binormal_cdf_r(z_cp, z2[K2 - 1], rho)
  } else {
    prob <- prob + pnorm(z2[resp_src]) - pnorm(z2[resp_src - 1]) -
            binormal_cdf_r(z_cp, z2[resp_src], rho) +
            binormal_cdf_r(z_cp, z2[resp_src - 1], rho)
  }

  prob
}

# =============================================================================
# Univariate Cell Probability
# =============================================================================

#' Univariate cell probability: `P(thresh[k-1] < X <= thresh[k])`
#'
#' @param resp Integer response (1..K)
#' @param K Number of categories
#' @param mu Mean
#' @param sigma SD
#' @param thresh Numeric vector of K-1 thresholds
#' @return Scalar probability
#' @noRd
univariate_cell_prob_r <- function(resp, K, mu, sigma, thresh) {
  if (resp == 1) return(pnorm((thresh[1] - mu) / sigma))
  if (resp == K) return(1 - pnorm((thresh[K - 1] - mu) / sigma))
  pnorm((thresh[resp] - mu) / sigma) - pnorm((thresh[resp - 1] - mu) / sigma)
}

# =============================================================================
# CDP Binormal Strip Integrals
# =============================================================================

#' P(R > c_R, tau_lo < M < tau_hi) -- "Remember" in confidence band
#'
#' @param z_cR Standardized recollection criterion
#' @param z_tau_lo Standardized lower confidence threshold
#' @param z_tau_hi Standardized upper confidence threshold
#' @param rho Correlation between R and M
#' @return Scalar probability; cells below 1e-6 come from the log-space integral
#' @noRd
binormal_strip_upper_r <- function(z_cR, z_tau_lo, z_tau_hi, rho) {
  p <- (pnorm(z_tau_hi) - binormal_cdf_r(z_cR, z_tau_hi, rho)) -
       (pnorm(z_tau_lo) - binormal_cdf_r(z_cR, z_tau_lo, rho))
  if (p >= 1e-6) return(p)
  s <- sqrt((1 - rho) * (1 + rho))
  exp(.log_lin_integral_r(z_tau_lo / s, -rho / s, z_tau_hi / s, -rho / s, z_cR, Inf))
}

#' P(R <= c_R, tau_lo < M < tau_hi) -- "Know" in confidence band
#'
#' @inheritParams binormal_strip_upper_r
#' @return Scalar probability; cells below 1e-6 come from the log-space integral
#' @noRd
binormal_strip_lower_r <- function(z_cR, z_tau_lo, z_tau_hi, rho) {
  p <- (binormal_cdf_r(z_cR, z_tau_hi, rho) - binormal_cdf_r(z_cR, z_tau_lo, rho))
  if (p >= 1e-6) return(p)
  s <- sqrt((1 - rho) * (1 + rho))
  exp(.log_lin_integral_r(z_tau_lo / s, -rho / s, z_tau_hi / s, -rho / s, -Inf, z_cR))
}

# --- R/K/G (Remember/Know/Guess) CDP functions ---

#' G(c) = P(R <= c_R, F <= c_K, M < c) -- unified guess probability formula
#'
#' Uses exact min (no smooth approximation needed on R side).
#'
#' @param c_val Confidence threshold
#' @param mu_R Recollection mean
#' @param sigma_R Recollection SD
#' @param mu_F Familiarity mean
#' @param sigma_F Familiarity SD
#' @param c_R Recollection criterion
#' @param c_K Know criterion
#' @param corr Correlation of R and F
#' @return Scalar cumulative guess probability
#' @noRd
G_cdp_r <- function(c_val, mu_R, sigma_R, mu_F, sigma_F, c_R, c_K, corr = 0) {
  s <- min(c_R, c_val - c_K)
  z_s <- (s - mu_R) / sigma_R
  z_cR <- (c_R - mu_R) / sigma_R
  z_cK <- (c_K - mu_F) / sigma_F
  mu_M <- mu_R + mu_F
  sigma_M <- sqrt(sigma_R^2 + sigma_F^2 + 2 * corr * sigma_R * sigma_F)
  z_c <- (c_val - mu_M) / sigma_M
  rho <- (sigma_R + corr * sigma_F) / sigma_M
  rs_part <- if (corr == 0) pnorm(z_s) * pnorm(z_cK) else binormal_cdf_r(z_s, z_cK, corr)
  rs_part + binormal_cdf_r(z_cR, z_c, rho) - binormal_cdf_r(z_s, z_c, rho)
}

#' Compute R/K/G probabilities for a single confidence band (G(c) approach)
#'
#' Guess-first approach: computes Guess directly via G(c), Know as remainder.
#'
#' @param mu_R Recollection mean
#' @param sigma_R Recollection SD
#' @param mu_F Familiarity mean
#' @param sigma_F Familiarity SD
#' @param c_R Recollection criterion
#' @param c_K Know criterion
#' @param tau_lo Lower confidence threshold (raw, not z-scored)
#' @param tau_hi Upper confidence threshold (raw, not z-scored)
#' @return Numeric vector of length 3: c(p_Remember, p_Know, p_Guess)
#' @noRd
.log_diff_pnorm <- function(l, u) {
  if (!(u > l)) return(-Inf)
  if (l >= 0) {
    a <- stats::pnorm(l, lower.tail = FALSE, log.p = TRUE)
    return(a + log1p(-exp(min(0, stats::pnorm(u, lower.tail = FALSE, log.p = TRUE) - a))))
  }
  if (u <= 0) {
    a <- stats::pnorm(u, log.p = TRUE)
    return(a + log1p(-exp(min(0, stats::pnorm(l, log.p = TRUE) - a))))
  }
  log(stats::pnorm(u) - stats::pnorm(l))
}

.log_lin_integral_r <- function(Al, Bl, Au, Bu, a, b) {
  lo <- max(a, -40)
  hi <- min(b, 40)
  if (!(hi > lo)) return(-Inf)
  g <- function(x) {
    l <- if (is.finite(Al)) Al + Bl * x else Al
    u <- if (is.finite(Au)) Au + Bu * x else Au
    stats::dnorm(x, log = TRUE) + .log_diff_pnorm(l, u)
  }
  gs <- function(x) {
    v <- g(x)
    if (is.finite(v)) v else -1e300
  }
  m <- stats::optimize(gs, c(lo, hi), maximum = TRUE, tol = 1e-12)$maximum
  gm <- max(gs(m), gs(lo), gs(hi))
  if (!(gm > -1e299)) return(-Inf)
  f <- function(x) vapply(x, function(xx) exp(gs(xx) - gm), numeric(1))
  v <- 0
  if (m > lo) v <- v + stats::integrate(f, lo, m, rel.tol = 1e-10, subdivisions = 500L)$value
  if (hi > m) v <- v + stats::integrate(f, m, hi, rel.tol = 1e-10, subdivisions = 500L)$value
  if (!(v > 0)) return(-Inf)
  gm + log(v)
}

cdp_log_cell_r <- function(kind, mu_R, sigma_R, mu_F, sigma_F, c_R, c_K, tau_lo, tau_hi, corr = 0) {
  sc <- sigma_F * sqrt((1 - corr) * (1 + corr))
  B <- -(sigma_R + corr * sigma_F) / sc
  xcR <- (c_R - mu_R) / sigma_R
  xkh <- if (is.finite(tau_hi)) (tau_hi - c_K - mu_R) / sigma_R else Inf
  xkl <- (tau_lo - c_K - mu_R) / sigma_R
  Alo <- (tau_lo - mu_R - mu_F) / sc
  Ahi <- if (is.finite(tau_hi)) (tau_hi - mu_R - mu_F) / sc else Inf
  Bhi <- if (is.finite(tau_hi)) B else 0
  AK <- (c_K - mu_F) / sc
  BK <- -corr * sigma_F / sc
  lse <- function(a, b) {
    m <- max(a, b)
    if (!is.finite(m)) return(m)
    m + log(exp(a - m) + exp(b - m))
  }
  if (kind == 1) return(.log_lin_integral_r(Alo, B, Ahi, Bhi, xcR, Inf))
  if (kind == 3) {
    return(lse(.log_lin_integral_r(Alo, B, AK, BK, xkl, min(xkh, xcR)),
               .log_lin_integral_r(Alo, B, Ahi, Bhi, max(xkh, xkl), xcR)))
  }
  ub <- min(xcR, xkh)
  lse(.log_lin_integral_r(Alo, B, Ahi, Bhi, -Inf, min(xkl, ub)),
      .log_lin_integral_r(AK, BK, Ahi, Bhi, xkl, ub))
}

compute_rkg_probs_r <- function(mu_R, sigma_R, mu_F, sigma_F, c_R, c_K, tau_lo, tau_hi, corr = 0) {
  mu_M <- mu_R + mu_F
  sigma_M <- sqrt(sigma_R^2 + sigma_F^2 + 2 * corr * sigma_R * sigma_F)
  rho <- (sigma_R + corr * sigma_F) / sigma_M
  z_cR <- (c_R - mu_R) / sigma_R

  z_lo <- (tau_lo - mu_M) / sigma_M
  z_hi <- (tau_hi - mu_M) / sigma_M

  p_Remember <- (pnorm(z_hi) - binormal_cdf_r(z_cR, z_hi, rho)) -
                (pnorm(z_lo) - binormal_cdf_r(z_cR, z_lo, rho))

  p_Guess <- G_cdp_r(tau_hi, mu_R, sigma_R, mu_F, sigma_F, c_R, c_K, corr) -
             G_cdp_r(tau_lo, mu_R, sigma_R, mu_F, sigma_F, c_R, c_K, corr)

  p_Band <- pnorm(z_hi) - pnorm(z_lo)
  p_Know <- p_Band - p_Remember - p_Guess

  raw <- c(p_Remember, p_Know, p_Guess)
  vapply(1:3, function(k) {
    if (raw[k] >= 1e-6) raw[k]
    else exp(cdp_log_cell_r(k, mu_R, sigma_R, mu_F, sigma_F, c_R, c_K, tau_lo, tau_hi, corr))
  }, numeric(1))
}
