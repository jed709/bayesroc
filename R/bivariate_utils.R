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
# Precomputed 20-point GL quadrature nodes and weights (module-level constant)
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

  # Handle special cases
  zero_a <- a == 0
  inf_h <- is.infinite(h)
  inf_a <- is.infinite(a)

  result[zero_a] <- 0
  result[inf_h & !zero_a] <- 0
  mask_inf_a <- inf_a & !zero_a & !inf_h
  if (any(mask_inf_a)) {
    result[mask_inf_a] <- sign(a[mask_inf_a]) * (1 - pnorm(abs(h[mask_inf_a]))) / 2
  }

  # Remaining: finite h and a, a != 0
  todo <- !zero_a & !inf_h & !inf_a
  if (!any(todo)) return(result)

  h_t <- h[todo]
  a_t <- a[todo]

  # Split: |a| <= 1 (direct GL) vs |a| > 1 (identity reduction)
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
  # GL quadrature nodes/weights transformed to [0, a_i] for each element
  # mid = a/2, half = a/2, x_ij = mid_i + half_i * node_j
  mid <- a / 2   # [n]
  half <- a / 2  # [n]

  # Build [n x 20] matrix of quadrature points: x_ij = mid_i + half_i * node_j
  x_mat <- outer(mid, .gl20_nodes, "+") + outer(half - mid, .gl20_nodes, "*")
  # Simpler: x_mat[i,j] = half[i] * (1 + nodes[j]) = a[i]/2 + a[i]/2 * nodes[j]
  x_mat <- outer(half, .gl20_nodes, "*") + mid  # mid + half * nodes

  # Weights scaled by half: w_ij = half_i * weight_j
  w_mat <- outer(half, .gl20_weights, "*")  # [n x 20]

  # Integrand: exp(-h^2 * (1 + x^2) / 2) / (1 + x^2)
  h2 <- h^2  # [n]
  x2 <- x_mat^2  # [n x 20]
  integrand <- exp(-h2 * (1 + x2) / 2) / (1 + x2)  # [n x 20]

  # Sum: result_i = sum_j(w_ij * integrand_ij) / (2*pi)
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

  # Handle infinities
  inf1_neg <- is.infinite(z1) & z1 < 0
  inf1_pos <- is.infinite(z1) & z1 > 0
  inf2_neg <- is.infinite(z2) & z2 < 0
  inf2_pos <- is.infinite(z2) & z2 > 0

  result[inf1_neg | inf2_neg] <- 0
  mask_inf1p <- inf1_pos & !inf2_neg
  mask_inf2p <- inf2_pos & !inf1_neg & !inf1_pos
  if (any(mask_inf1p)) result[mask_inf1p] <- pnorm(z2[mask_inf1p])
  if (any(mask_inf2p)) result[mask_inf2p] <- pnorm(z1[mask_inf2p])

  # Remaining finite cases
  finite <- !is.infinite(z1) & !is.infinite(z2)
  if (!any(finite)) return(result)

  z1f <- z1[finite]
  z2f <- z2[finite]
  rhof <- rho[finite]

  # Handle zeros: offset to avoid division by zero
  z1f[z1f == 0] <- 1e-10
  z2f[z2f == 0] <- 1e-10

  denom <- sqrt((1 + rhof) * (1 - rhof))
  a1 <- (z2f / z1f - rhof) / denom
  a2 <- (z1f / z2f - rhof) / denom
  product <- z1f * z2f
  delta <- as.numeric(product < 0 | (product == 0 & (z1f + z2f) < 0))

  result[finite] <- 0.5 * (pnorm(z1f) + pnorm(z2f) - delta) -
                    owens_t_r(z1f, a1) - owens_t_r(z2f, a2)
  result
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
  # Standardize thresholds
  z1 <- (c1 - mu1) / sigma1
  z2 <- (c2 - mu2) / sigma2

  # Fast path: when rho = 0, bivariate factors into product of univariates
  if (rho == 0) {
    p1 <- univariate_cell_prob_from_z(resp1, K1, z1)
    p2 <- univariate_cell_prob_from_z(resp2, K2, z2)
    return(p1 * p2)
  }

  # 9 cases: corners, edges, interior
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
  # Interior
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

    # Below-cp: independent, source ~ N(0, sigma_cond)
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

    # Above-cp: standard bivariate
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
    # Independent: marginal source is N(mu_S, sigma_S^2)
    return(univariate_cell_prob_r(resp_src, K2, mu_S, sigma_S, c2))
  }

  cp <- mu_I - sigma_I * mu_S / (rho * sigma_S)
  sigma_cond <- sigma_S * sqrt(1 - rho * rho)
  z_cp <- (cp - mu_I) / sigma_I
  p_below <- pnorm(z_cp)

  prob <- 0

  # Below-cp: source ~ N(0, sigma_cond)
  if (resp_src == 1) {
    prob <- prob + p_below * pnorm(c2[1] / sigma_cond)
  } else if (resp_src == K2) {
    prob <- prob + p_below * (1 - pnorm(c2[K2 - 1] / sigma_cond))
  } else {
    prob <- prob + p_below * (pnorm(c2[resp_src] / sigma_cond) - pnorm(c2[resp_src - 1] / sigma_cond))
  }

  # Above-cp: standard BVN, marginal for source in bin k with item > cp
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

#' Univariate cell probability: P(thresh[k-1] < X <= thresh[k])
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
#' @return Scalar probability (clamped >= 1e-20)
#' @noRd
binormal_strip_upper_r <- function(z_cR, z_tau_lo, z_tau_hi, rho) {
  p_upper <- pnorm(z_tau_hi) - binormal_cdf_r(z_cR, z_tau_hi, rho)
  p_lower <- pnorm(z_tau_lo) - binormal_cdf_r(z_cR, z_tau_lo, rho)
  max(p_upper - p_lower, 1e-20)
}

#' P(R <= c_R, tau_lo < M < tau_hi) -- "Know" in confidence band
#'
#' @param z_cR Standardized recollection criterion
#' @param z_tau_lo Standardized lower confidence threshold
#' @param z_tau_hi Standardized upper confidence threshold
#' @param rho Correlation between R and M
#' @return Scalar probability (clamped >= 1e-20)
#' @noRd
binormal_strip_lower_r <- function(z_cR, z_tau_lo, z_tau_hi, rho) {
  p_band <- pnorm(z_tau_hi) - pnorm(z_tau_lo)
  max(p_band - binormal_strip_upper_r(z_cR, z_tau_lo, z_tau_hi, rho), 1e-20)
}


# --- R/K/G (Remember/Know/Guess) CDP functions ---
# G(c) approach: computes Guess directly, Know as remainder

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
#' @return Scalar cumulative guess probability
#' @noRd
G_cdp_r <- function(c_val, mu_R, sigma_R, mu_F, sigma_F, c_R, c_K) {
  s <- min(c_R, c_val - c_K)
  z_s <- (s - mu_R) / sigma_R
  z_cR <- (c_R - mu_R) / sigma_R
  z_cK <- (c_K - mu_F) / sigma_F
  mu_M <- mu_R + mu_F
  sigma_M <- sqrt(sigma_R^2 + sigma_F^2)
  z_c <- (c_val - mu_M) / sigma_M
  rho <- sigma_R / sigma_M
  pnorm(z_s) * pnorm(z_cK) + binormal_cdf_r(z_cR, z_c, rho) -
    binormal_cdf_r(z_s, z_c, rho)
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
compute_rkg_probs_r <- function(mu_R, sigma_R, mu_F, sigma_F, c_R, c_K, tau_lo, tau_hi) {
  mu_M <- mu_R + mu_F
  sigma_M <- sqrt(sigma_R^2 + sigma_F^2)
  rho <- sigma_R / sigma_M
  z_cR <- (c_R - mu_R) / sigma_R

  # Standardized thresholds
  z_lo <- (tau_lo - mu_M) / sigma_M
  z_hi <- (tau_hi - mu_M) / sigma_M

  # REMEMBER: P(R > c_R, tau_lo < M < tau_hi)
  p_Remember <- (pnorm(z_hi) - binormal_cdf_r(z_cR, z_hi, rho)) -
                (pnorm(z_lo) - binormal_cdf_r(z_cR, z_lo, rho))

  # GUESS: G(tau_hi) - G(tau_lo) -- computed directly
  p_Guess <- G_cdp_r(tau_hi, mu_R, sigma_R, mu_F, sigma_F, c_R, c_K) -
             G_cdp_r(tau_lo, mu_R, sigma_R, mu_F, sigma_F, c_R, c_K)

  # KNOW: remainder (= P(R <= c_R, band) - Guess)
  p_Band <- pnorm(z_hi) - pnorm(z_lo)
  p_Know <- p_Band - p_Remember - p_Guess

  # Soft clamp: softplus with large kk for negligible distortion + smooth gradients
  kk <- 10000
  c(log1p(exp(kk * p_Remember)) / kk,
    log1p(exp(kk * p_Know)) / kk,
    log1p(exp(kk * p_Guess)) / kk)
}
