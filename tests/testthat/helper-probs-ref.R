# Per-row, per-draw reference implementations of the cell probabilities, against which the
# vectorised probability functions are tested.

link_cdf_r <- function(x, link_name) {
  switch(link_name,
    probit = pnorm(x),
    logit  = plogis(x),
    cloglog = -expm1(-exp(pmin(x, 700))),
    stop("Unknown link: ", link_name)
  )
}

.ref_probs_bivariate_sdt <- function(params, thresh, thresh2, item_type, S, N, K1, K2, model) {
  md <- model$model_data
  dprime <- params$dprime
  discrim <- params$discrim
  rho_A <- params$rho

  has_dprime_B <- isTRUE(md$has_dprime_B)
  has_discrim_B <- isTRUE(md$has_discrim_B)
  has_sigma <- !is.null(params$sigma)
  has_sigma_B <- !is.null(params$sigma_B)
  has_sigma2 <- !is.null(params$sigma2)
  has_sigma2_B <- !is.null(params$sigma2_B)
  has_rho_B <- !is.null(params$rho_B)
  has_rho_N <- !is.null(params$rho_N)

  varying <- isTRUE(md$varying_source_criteria)
  bounded <- isTRUE(md$bounded)

  prob_fn <- if (bounded) compute_bounded_prob_r else compute_bivariate_prob_r
  probs <- array(0, dim = c(S, N, K1, K2))

  for (n in seq_len(N)) {
    it <- item_type[n]

    if (it == 1) {
      mu1_s <- rep(0, S); mu2_s <- rep(0, S)
      sig1_s <- rep(1, S); sig2_s <- rep(1, S)
      rho_s <- if (has_rho_N) params$rho_N[, n] else rep(0, S)
    } else if (it == 2) {
      mu1_s <- dprime[, n]
      mu2_s <- discrim[, n]
      sig1_s <- if (has_sigma) params$sigma[, n] else rep(1, S)
      sig2_s <- if (has_sigma2) params$sigma2[, n] else rep(1, S)
      rho_s <- rho_A[, n]
    } else {
      mu1_s <- if (has_dprime_B) params$dprime_B[, n] else dprime[, n]
      mu2_s <- if (has_discrim_B) (if (bounded) -1 else 1) * params$discrim_B[, n] else -discrim[, n]   # given B values are signed; the bounded kernel mirrors inside
      sig1_s <- if (has_sigma_B) params$sigma_B[, n] else {
        if (has_sigma) params$sigma[, n] else rep(1, S)
      }
      sig2_s <- if (has_sigma2_B) params$sigma2_B[, n] else {
        if (has_sigma2) params$sigma2[, n] else rep(1, S)
      }
      rho_s <- if (has_rho_B) (if (bounded) -1 else 1) * params$rho_B[, n] else -rho_A[, n]
    }

    use_bounded <- bounded && it != 1

    if (use_bounded) {
      mu2_s <- -mu2_s
      rho_s <- -rho_s
    }

    for (j in seq_len(K1)) {
      c1_lo_z <- if (j > 1) (thresh[, n, j - 1] - mu1_s) / sig1_s else rep(-Inf, S)
      c1_hi_z <- if (j < K1) (thresh[, n, j] - mu1_s) / sig1_s else rep(Inf, S)

      for (k in seq_len(K2)) {
        if (varying) {
          c2_lo_z <- if (k > 1) (thresh2[, n, j, k - 1] - mu2_s) / sig2_s else rep(-Inf, S)
          c2_hi_z <- if (k < K2) (thresh2[, n, j, k] - mu2_s) / sig2_s else rep(Inf, S)
        } else {
          c2_lo_z <- if (k > 1) (thresh2[, n, k - 1] - mu2_s) / sig2_s else rep(-Inf, S)
          c2_hi_z <- if (k < K2) (thresh2[, n, k] - mu2_s) / sig2_s else rep(Inf, S)
        }

        if (use_bounded) {
          for (s in seq_len(S)) {
            c1_raw <- thresh[s, n, ]
            if (varying) {
              c2_raw <- thresh2[s, n, j, ]
            } else {
              c2_raw <- thresh2[s, n, ]
            }
            probs[s, n, j, k] <- compute_bounded_prob_r(
              j, k, K1, K2, mu1_s[s], mu2_s[s], sig1_s[s], sig2_s[s],
              c1_raw, c2_raw, rho_s[s]
            )
          }
        } else {
          p <- binormal_cdf_r(c1_hi_z, c2_hi_z, rho_s) -
               binormal_cdf_r(c1_hi_z, c2_lo_z, rho_s) -
               binormal_cdf_r(c1_lo_z, c2_hi_z, rho_s) +
               binormal_cdf_r(c1_lo_z, c2_lo_z, rho_s)
          probs[, n, j, k] <- p
        }
      }
    }
  }

  probs <- pmax(probs, 0)
  probs
}

.ref_probs_bivariate_dp <- function(params, thresh, thresh2, item_type, S, N, K1, K2, model) {
  md <- model$model_data
  dprime <- params$dprime
  discrim <- params$discrim
  rho_A <- params$rho
  lambda <- params$lambda
  lambda2 <- params$lambda2

  has_dprime_B <- isTRUE(md$has_dprime_B)
  has_discrim_B <- isTRUE(md$has_discrim_B)
  has_sigma <- !is.null(params$sigma)
  has_sigma_B <- !is.null(params$sigma_B)
  has_sigma2 <- !is.null(params$sigma2)
  has_sigma2_B <- !is.null(params$sigma2_B)
  has_rho_B <- !is.null(params$rho_B)
  has_rho_N <- !is.null(params$rho_N)

  varying <- isTRUE(md$varying_source_criteria)
  bounded <- isTRUE(md$bounded)

  probs <- array(0, dim = c(S, N, K1, K2))

  for (n in seq_len(N)) {
    it <- item_type[n]

    if (it == 1) {
      rho_s <- if (has_rho_N) params$rho_N[, n] else rep(0, S)
      mu1_s <- rep(0, S); mu2_s <- rep(0, S); sig1_s <- rep(1, S); sig2_s <- rep(1, S)
    } else if (it == 2) {
      mu1_s <- dprime[, n]
      mu2_s <- discrim[, n]
      sig1_s <- if (has_sigma) params$sigma[, n] else rep(1, S)
      sig2_s <- if (has_sigma2) params$sigma2[, n] else rep(1, S)
      rho_s <- rho_A[, n]
    } else {
      mu1_s <- if (has_dprime_B) params$dprime_B[, n] else dprime[, n]
      mu2_s <- if (has_discrim_B) (if (bounded) -1 else 1) * params$discrim_B[, n] else -discrim[, n]   # given B values are signed; the bounded kernel mirrors inside
      sig1_s <- if (has_sigma_B) params$sigma_B[, n] else {
        if (has_sigma) params$sigma[, n] else rep(1, S)
      }
      sig2_s <- if (has_sigma2_B) params$sigma2_B[, n] else {
        if (has_sigma2) params$sigma2[, n] else rep(1, S)
      }
      rho_s <- if (has_rho_B) (if (bounded) -1 else 1) * params$rho_B[, n] else -rho_A[, n]
    }

    if (bounded && it != 1) {
      mu2_s <- -mu2_s
      rho_s <- -rho_s
    }

    R_I_s <- if (it != 1) lambda[, n] else NULL
    R_S_s <- if (it != 1) lambda2[, n] else NULL

    for (j in seq_len(K1)) {
      c1_lo_z <- if (j > 1) (thresh[, n, j - 1] - mu1_s) / sig1_s else rep(-Inf, S)
      c1_hi_z <- if (j < K1) (thresh[, n, j] - mu1_s) / sig1_s else rep(Inf, S)

      if (varying) {
        c2_raw <- thresh2[, n, j, , drop = FALSE]
        dim(c2_raw) <- c(S, K2 - 1)
      } else {
        c2_raw <- thresh2[, n, , drop = FALSE]
        dim(c2_raw) <- c(S, K2 - 1)
      }

      for (k in seq_len(K2)) {
        c2_lo_z <- if (k > 1) (c2_raw[, k - 1] - mu2_s) / sig2_s else rep(-Inf, S)
        c2_hi_z <- if (k < K2) (c2_raw[, k] - mu2_s) / sig2_s else rep(Inf, S)

        if (it == 1) {
          probs[, n, j, k] <-
            binormal_cdf_r(c1_hi_z, c2_hi_z, rho_s) -
            binormal_cdf_r(c1_hi_z, c2_lo_z, rho_s) -
            binormal_cdf_r(c1_lo_z, c2_hi_z, rho_s) +
            binormal_cdf_r(c1_lo_z, c2_lo_z, rho_s)
          next
        }

        is_corner <- (it == 2 && k == 1) || (it == 3 && k == K2)

        if (bounded) {
          for (s in seq_len(S)) {
            c1_raw_s <- thresh[s, n, ]
            c2_raw_s <- c2_raw[s, ]
            p1s <- (1 - R_I_s[s]) * compute_bounded_prob_r(
              j, k, K1, K2, mu1_s[s], mu2_s[s], sig1_s[s], sig2_s[s],
              c1_raw_s, c2_raw_s, rho_s[s])
            p2s <- if (j == K1) R_I_s[s] * (1 - R_S_s[s]) *
              compute_bounded_marginal_source_r(k, K2, mu1_s[s], mu2_s[s],
                sig1_s[s], sig2_s[s], rho_s[s], c2_raw_s) else 0
            p3s <- if (j == K1 && is_corner) R_I_s[s] * R_S_s[s] else 0
            probs[s, n, j, k] <- p1s + p2s + p3s
          }
        } else {
          p1 <- (1 - R_I_s) * (
            binormal_cdf_r(c1_hi_z, c2_hi_z, rho_s) -
            binormal_cdf_r(c1_hi_z, c2_lo_z, rho_s) -
            binormal_cdf_r(c1_lo_z, c2_hi_z, rho_s) +
            binormal_cdf_r(c1_lo_z, c2_lo_z, rho_s))
          p2 <- if (j == K1)
            R_I_s * (1 - R_S_s) * (pnorm(c2_hi_z) - pnorm(c2_lo_z)) else rep(0, S)
          p3 <- if (j == K1 && is_corner) R_I_s * R_S_s else rep(0, S)
          probs[, n, j, k] <- p1 + p2 + p3
        }
      }
    }
  }

  probs <- pmax(probs, 0)
  probs
}

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

univariate_cell_prob_r <- function(resp, K, mu, sigma, thresh) {
  if (resp == 1) return(pnorm((thresh[1] - mu) / sigma))
  if (resp == K) return(1 - pnorm((thresh[K - 1] - mu) / sigma))
  pnorm((thresh[resp] - mu) / sigma) - pnorm((thresh[resp - 1] - mu) / sigma)
}

univariate_cell_prob_from_z <- function(resp, K, z) {
  if (resp == 1) return(pnorm(z[1]))
  if (resp == K) return(1 - pnorm(z[K - 1]))
  pnorm(z[resp]) - pnorm(z[resp - 1])
}
