# Per-row, per-draw reference implementations of the cell probabilities, against which the
# vectorised probability functions are tested.

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

  probs <- pmax(probs, 1e-10)
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

  probs <- pmax(probs, 1e-10)
  probs
}

