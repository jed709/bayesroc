#' Generate Stan Code for SDT Models
#'
#' Supports multiple families: EVSDT, UVSDT, DPSDT, Mixture
#' @noRd

generate_stan_code_v2 <- function(model_data, family, batch_info = NULL) {

  code <- paste(
    generate_functions_block_v2(model_data, family, batch_info = batch_info),
    generate_data_block_v2(model_data, family, batch_info = batch_info),
    generate_transformed_data_block_v2(model_data, family),
    generate_parameters_block_v2(model_data, family),
    generate_transformed_parameters_block_v2(model_data, family),
    generate_model_block_v2(model_data, family, batch_info = batch_info),
    generate_generated_quantities_block_v2(model_data, family),
    sep = "\n"
  )

  # Softplus gap link: replace exp() with log1p_exp() (Stan's softplus)
  # and rename parameter/variable names accordingly
  if (identical(model_data$gap_link, "softplus")) {
    code <- gsub("beta_log_gaps", "beta_raw_gaps", code)
    code <- gsub("log_gaps_prior", "raw_gaps_prior", code)
    code <- gsub("Log-gaps", "Raw-gaps (softplus)", code)
    # Replace gap transformation: exp(X) -> log1p_exp(X) for gap variables
    # The pattern is: + exp(log_gap) or - exp(log_gap)
    code <- gsub("exp(log_gap)", "log1p_exp(raw_gap)", code, fixed = TRUE)
    code <- gsub("real log_gap", "real raw_gap", code, fixed = TRUE)
    code <- gsub("log_gap +=", "raw_gap +=", code, fixed = TRUE)
    code <- gsub("log_gap;", "raw_gap;", code, fixed = TRUE)
  }

  class(code) <- c("stan_code", "character")
  code
}


#' Check if a parameter has a simple encoding pattern
#' 
#' A simple encoding pattern means:
#' - Fixed effects use encoding() wrapper
#' - All random effects for this parameter use encoding with term_idx (not Z matrix)
#' - The fixed effect dimension matches the random effect dimension
#' 
#' This allows using beta[idx] instead of X[n,] * beta for faster sampling.
#' @noRd
is_simple_encoding <- function(fixed_info, random_info) {
  # Must have encoding in fixed effects
  if (!isTRUE(fixed_info$has_encoding)) return(FALSE)
  
  # Must have at least one random effect
  if (is.null(random_info) || length(random_info) == 0) return(FALSE)
  
  # All random effects must be encoding with matching dimensions and not using Z matrix
  for (group in names(random_info)) {
    re <- random_info[[group]]
    if (!isTRUE(re$is_encoding)) return(FALSE)
    if (is.null(re$term_idx)) return(FALSE)
    if (isTRUE(re$use_z_matrix)) return(FALSE)
    # Dimensions should match (n_coef for fixed, dim for random)
    if (re$dim != fixed_info$n_coef) return(FALSE)
  }
  
  TRUE
}


generate_functions_block_v2 <- function(model_data, family, batch_info = NULL) {
  family_name <- if (is.list(family)) family$family else family
  link <- if (is.list(family) && !is.null(family$link)) family$link else link_probit()
  link_name <- link$name
  
  lines <- c("functions {")
  
  # Generate the CDF wrapper function based on link
  cdf_code <- generate_link_cdf_wrapper(link)
  lines <- c(lines, cdf_code)
  
  # Cumulative likelihood (simple ordinal regression - no is_old)
  if (family_name == "cumulative") {
    lines <- c(lines, '
  real cumulative_lpmf(int y, real mu, int K, vector thresh) {
    if (y == 1) {
      return log(fmax(link_cdf(thresh[1] - mu), 1e-20));
    } else if (y == K) {
      return log(fmax(1 - link_cdf(thresh[K-1] - mu), 1e-20));
    } else {
      real p_upper = link_cdf(thresh[y] - mu);
      real p_lower = link_cdf(thresh[y-1] - mu);
      return log(fmax(p_upper - p_lower, 1e-20));
    }
  }')
    # Threading: add partial_log_lik before closing functions block
    if (isTRUE(model_data$threads) && is.null(batch_info)) {
      lines <- c(lines, "", generate_partial_log_lik(model_data, family))
    }
    # Batch likelihood: add function declaration (implemented in C++ user header)
    if (!is.null(batch_info)) {
      lines <- c(lines, "", "  // Batch C++ likelihood (implemented in user header)", batch_info$stan_decl)
      if (isTRUE(model_data$threads)) {
        lines <- c(lines, "", batch_info$stan_partial)
      }
    }
    lines <- c(lines, "}")
    return(paste(lines, collapse = "\n"))
  }
  
  # EVSDT likelihood (used by all SDT families for new items)
  lines <- c(lines, '
  real evsdt_lpmf(int y, real dprime, real is_old, int K, vector thresh) {
    real mu = is_old * dprime;
    
    if (y == 1) {
      return log(link_cdf(thresh[1] - mu));
    } else if (y == K) {
      return log1m(link_cdf(thresh[K-1] - mu));
    } else {
      real p_upper = link_cdf(thresh[y] - mu);
      real p_lower = link_cdf(thresh[y-1] - mu);
      return log(fmax(p_upper - p_lower, 1e-20));
    }
  }')
  
  # UVSDT likelihood - only include if sigma is actually varying
  if (family_name %in% c("uvsdt", "dpsdt", "mixture") && model_data$has_sigma) {
    lines <- c(lines, '
  real uvsdt_lpmf(int y, real dprime, real sigma, real is_old, int K, vector thresh) {
    real mu = is_old * dprime;
    real s = is_old > 0 ? sigma : 1.0;
    
    if (y == 1) {
      return log(link_cdf((thresh[1] - mu) / s));
    } else if (y == K) {
      return log1m(link_cdf((thresh[K-1] - mu) / s));
    } else {
      real p_upper = link_cdf((thresh[y] - mu) / s);
      real p_lower = link_cdf((thresh[y-1] - mu) / s);
      return log(fmax(p_upper - p_lower, 1e-20));
    }
  }')
  }
  
  # DPSDT likelihood
  if (family_name == "dpsdt") {
    if (model_data$has_sigma) {
      lines <- c(lines, '
  real dpsdt_lpmf(int y, real dprime, real sigma, real lambda, real is_old, int K, vector thresh) {
    if (is_old > 0) {
      real mu = is_old * dprime;
      if (y == K) {
        // P(Y=K | old) = lambda + (1-lambda) * (1 - link_cdf((thresh[K-1] - mu)/sigma))
        real p_fam = 1 - link_cdf((thresh[K-1] - mu) / sigma);
        return log(fmax(lambda + (1 - lambda) * p_fam, 1e-20));
      } else {
        // P(Y=y | old) = (1-lambda) * [link_cdf((thresh[y]-mu)/sigma) - link_cdf((thresh[y-1]-mu)/sigma)]
        real log_prob;
        if (y == 1) {
          log_prob = log(link_cdf((thresh[1] - mu) / sigma));
        } else {
          real p_upper = link_cdf((thresh[y] - mu) / sigma);
          real p_lower = link_cdf((thresh[y-1] - mu) / sigma);
          log_prob = log(fmax(p_upper - p_lower, 1e-20));
        }
        return log1m(lambda) + log_prob;
      }
    } else {
      return evsdt_lpmf(y | dprime, is_old, K, thresh);
    }
  }')
    } else {
      lines <- c(lines, '
  real dpsdt_lpmf(int y, real dprime, real lambda, real is_old, int K, vector thresh) {
    if (is_old > 0) {
      real mu = is_old * dprime;
      if (y == K) {
        real p_fam = 1 - link_cdf(thresh[K-1] - mu);
        return log(fmax(lambda + (1 - lambda) * p_fam, 1e-20));
      } else {
        real log_prob;
        if (y == 1) {
          log_prob = log(link_cdf(thresh[1] - mu));
        } else {
          real p_upper = link_cdf(thresh[y] - mu);
          real p_lower = link_cdf(thresh[y-1] - mu);
          log_prob = log(fmax(p_upper - p_lower, 1e-20));
        }
        return log1m(lambda) + log_prob;
      }
    } else {
      return evsdt_lpmf(y | dprime, is_old, K, thresh);
    }
  }')
    }
  }
  
  # Mixture SDT likelihood
  if (family_name == "mixture") {
    sigma_fixed <- !model_data$has_sigma
    sigma2_fixed <- !model_data$has_sigma2
    
    if (sigma_fixed && sigma2_fixed) {
      lines <- c(lines, '
  real mixture_sdt_lpmf(int y, real dprime, real dprime2, real lambda, real is_old, int K, vector thresh) {
    if (is_old > 0) {
      real mu1 = is_old * dprime;
      real mu2 = dprime2 - (1.0 - is_old) * dprime;
      real p1, p2;
      if (y == 1) {
        p1 = link_cdf(thresh[1] - mu1);
        p2 = link_cdf(thresh[1] - mu2);
      } else if (y == K) {
        p1 = 1 - link_cdf(thresh[K-1] - mu1);
        p2 = 1 - link_cdf(thresh[K-1] - mu2);
      } else {
        p1 = link_cdf(thresh[y] - mu1) - link_cdf(thresh[y-1] - mu1);
        p2 = link_cdf(thresh[y] - mu2) - link_cdf(thresh[y-1] - mu2);
      }
      return log(fmax(lambda * p1 + (1 - lambda) * p2, 1e-20));
    } else {
      return evsdt_lpmf(y | dprime, is_old, K, thresh);
    }
  }')
    } else {
      lines <- c(lines, '
  real mixture_sdt_lpmf(int y, real dprime, real sigma, real dprime2, real sigma2, real lambda, real is_old, int K, vector thresh) {
    if (is_old > 0) {
      real mu1 = is_old * dprime;
      real mu2 = dprime2 - (1.0 - is_old) * dprime;
      real p1, p2;
      if (y == 1) {
        p1 = link_cdf((thresh[1] - mu1) / sigma);
        p2 = link_cdf((thresh[1] - mu2) / sigma2);
      } else if (y == K) {
        p1 = 1 - link_cdf((thresh[K-1] - mu1) / sigma);
        p2 = 1 - link_cdf((thresh[K-1] - mu2) / sigma2);
      } else {
        p1 = link_cdf((thresh[y] - mu1) / sigma) - link_cdf((thresh[y-1] - mu1) / sigma);
        p2 = link_cdf((thresh[y] - mu2) / sigma2) - link_cdf((thresh[y-1] - mu2) / sigma2);
      }
      return log(fmax(lambda * p1 + (1 - lambda) * p2, 1e-20));
    } else {
      return evsdt_lpmf(y | dprime, is_old, K, thresh);
    }
  }')
    }
    
    # Lure mixture function: new items are also a mixture
    if (isTRUE(model_data$has_lure_mixture)) {
      if (model_data$has_sigma_L) {
        lines <- c(lines, '
  // Lure mixture: P(y|new) = lambda_L * P(y|-dprime_L, sigma_L) + (1-lambda_L) * P(y|0, 1)
  // dprime_L is lure detection strength; negative sign means lures fall below reference
  real lure_mixture_lpmf(int y, real dprime_L, real sigma_L, real lambda_L, int K, vector thresh) {
    real p_lure, p_ref;
    if (y == 1) {
      p_lure = link_cdf((thresh[1] + dprime_L) / sigma_L);
      p_ref = link_cdf(thresh[1]);
    } else if (y == K) {
      p_lure = 1 - link_cdf((thresh[K-1] + dprime_L) / sigma_L);
      p_ref = 1 - link_cdf(thresh[K-1]);
    } else {
      p_lure = link_cdf((thresh[y] + dprime_L) / sigma_L) - link_cdf((thresh[y-1] + dprime_L) / sigma_L);
      p_ref = link_cdf(thresh[y]) - link_cdf(thresh[y-1]);
    }
    return log(fmax(lambda_L * p_lure + (1 - lambda_L) * p_ref, 1e-20));
  }')
      } else {
        lines <- c(lines, '
  // Lure mixture: P(y|new) = lambda_L * P(y|-dprime_L, 1) + (1-lambda_L) * P(y|0, 1)
  // dprime_L is lure detection strength; negative sign means lures fall below reference
  real lure_mixture_lpmf(int y, real dprime_L, real lambda_L, int K, vector thresh) {
    real p_lure, p_ref;
    if (y == 1) {
      p_lure = link_cdf(thresh[1] + dprime_L);
      p_ref = link_cdf(thresh[1]);
    } else if (y == K) {
      p_lure = 1 - link_cdf(thresh[K-1] + dprime_L);
      p_ref = 1 - link_cdf(thresh[K-1]);
    } else {
      p_lure = link_cdf(thresh[y] + dprime_L) - link_cdf(thresh[y-1] + dprime_L);
      p_ref = link_cdf(thresh[y]) - link_cdf(thresh[y-1]);
    }
    return log(fmax(lambda_L * p_lure + (1 - lambda_L) * p_ref, 1e-20));
  }')
      }
    }
  }
  
  # Source mixture SDT likelihood (for source discrimination)
  if (family_name == "source_mixture") {
    has_dprime_B <- isTRUE(model_data$has_dprime_B)
    has_lambda_B <- isTRUE(model_data$has_lambda_B)
    
    if (has_dprime_B && has_lambda_B) {
      # General model: separate dprime and lambda for each source
      lines <- c(lines, '
  // Source mixture SDT - General model (separate d and lambda for each source)
  real source_mixture_lpmf(int y, int source, real dprime_A, real dprime_B, 
                           real lambda_A, real lambda_B, int K, vector thresh) {
    real p_attended;
    real p_nonattended;
    real lambda;
    real d;
    
    // Select parameters based on source (0 = A, 1 = B)
    if (source == 0) {
      lambda = lambda_A;
      d = dprime_A;
    } else {
      lambda = lambda_B;
      d = dprime_B;
    }
    
    // Probability is mixture of attended (at d) and nonattended (at 0) distributions
    if (y == 1) {
      p_attended = link_cdf(thresh[1] - d);
      p_nonattended = link_cdf(thresh[1]);
    } else if (y == K) {
      p_attended = 1 - link_cdf(thresh[K-1] - d);
      p_nonattended = 1 - link_cdf(thresh[K-1]);
    } else {
      p_attended = link_cdf(thresh[y] - d) - link_cdf(thresh[y-1] - d);
      p_nonattended = link_cdf(thresh[y]) - link_cdf(thresh[y-1]);
    }
    
    return log(fmax(lambda * p_attended + (1 - lambda) * p_nonattended, 1e-20));
  }')
    } else if (has_dprime_B) {
      # dprime_B is free but lambda_B = lambda (equal attention)
      lines <- c(lines, '
  // Source mixture SDT - Equal attention model (lambda_B = lambda)
  real source_mixture_lpmf(int y, int source, real dprime_A, real dprime_B, 
                           real lambda, int K, vector thresh) {
    real p_attended;
    real p_nonattended;
    real d;
    
    d = source == 0 ? dprime_A : dprime_B;
    
    if (y == 1) {
      p_attended = link_cdf(thresh[1] - d);
      p_nonattended = link_cdf(thresh[1]);
    } else if (y == K) {
      p_attended = 1 - link_cdf(thresh[K-1] - d);
      p_nonattended = 1 - link_cdf(thresh[K-1]);
    } else {
      p_attended = link_cdf(thresh[y] - d) - link_cdf(thresh[y-1] - d);
      p_nonattended = link_cdf(thresh[y]) - link_cdf(thresh[y-1]);
    }
    
    return log(fmax(lambda * p_attended + (1 - lambda) * p_nonattended, 1e-20));
  }')
    } else if (has_lambda_B) {
      # lambda_B is free but dprime_B = -dprime (symmetric discrimination)
      lines <- c(lines, '
  // Source mixture SDT - Symmetric d model (dprime_B = -dprime)
  real source_mixture_lpmf(int y, int source, real dprime, real lambda_A, 
                           real lambda_B, int K, vector thresh) {
    real p_attended;
    real p_nonattended;
    real lambda;
    real d;
    
    // dprime_B = -dprime (symmetric)
    if (source == 0) {
      lambda = lambda_A;
      d = dprime;  // dprime_A
    } else {
      lambda = lambda_B;
      d = -dprime;  // dprime_B = -dprime_A
    }
    
    if (y == 1) {
      p_attended = link_cdf(thresh[1] - d);
      p_nonattended = link_cdf(thresh[1]);
    } else if (y == K) {
      p_attended = 1 - link_cdf(thresh[K-1] - d);
      p_nonattended = 1 - link_cdf(thresh[K-1]);
    } else {
      p_attended = link_cdf(thresh[y] - d) - link_cdf(thresh[y-1] - d);
      p_nonattended = link_cdf(thresh[y]) - link_cdf(thresh[y-1]);
    }
    
    return log(fmax(lambda * p_attended + (1 - lambda) * p_nonattended, 1e-20));
  }')
    } else {
      # Fully symmetric model: dprime_B = -dprime, lambda_B = lambda
      lines <- c(lines, '
  // Source mixture SDT - Fully symmetric model (dprime_B = -dprime, lambda_B = lambda)
  real source_mixture_lpmf(int y, int source, real dprime, real lambda, int K, vector thresh) {
    real p_attended;
    real p_nonattended;
    real d;
    
    // Symmetric: d_A = dprime, d_B = -dprime
    d = source == 0 ? dprime : -dprime;
    
    if (y == 1) {
      p_attended = link_cdf(thresh[1] - d);
      p_nonattended = link_cdf(thresh[1]);
    } else if (y == K) {
      p_attended = 1 - link_cdf(thresh[K-1] - d);
      p_nonattended = 1 - link_cdf(thresh[K-1]);
    } else {
      p_attended = link_cdf(thresh[y] - d) - link_cdf(thresh[y-1] - d);
      p_nonattended = link_cdf(thresh[y]) - link_cdf(thresh[y-1]);
    }
    
    return log(fmax(lambda * p_attended + (1 - lambda) * p_nonattended, 1e-20));
  }')
    }
  }
  
  # Bivariate SDT/DP functions (for source monitoring with joint detection/discrimination)
  if (family_name %in% c("bivariate_sdt", "bivariate_dp", "bivariate_cumulative")) {
    lines <- c(lines, '
  // Bivariate normal CDF using Owens T function
  // Computes P(Z1 <= z1, Z2 <= z2) for standard bivariate normal with correlation rho
  real binormal_cdf_custom(real z1, real z2, real rho) {
    // Handle the case where both are zero
    if (z1 == 0 && z2 == 0) {
      return 0.25 + asin(rho) / (2 * pi());
    }
    // Handle edge cases where one of z1 or z2 is zero (avoids division by zero)
    // When z1=0: P(Z1<=0, Z2<=z2) = 0.5 * Phi(z2) + owens_t(z2, rho) for rho != 0
    // Using the identity: Phi_2(0, z2, rho) = 0.5 * Phi(z2) when rho=0
    // More generally: Phi_2(0, z2, rho) = Phi(z2) * 0.5 + owens_t(z2, rho/sqrt(1-0)) 
    // but safer to use the general formula with small epsilon
    if (z1 == 0) {
      // Use limit: as z1->0, owens_t(z1, a1) -> 0 and the formula simplifies
      // Phi_2(0, z2, rho) = Phi(z2)/2 + owens_t(z2, -rho/sqrt(1-rho^2)) when z2 > 0
      // But simpler: use tiny offset to avoid exact zero
      real z1_safe = 1e-10;
      real denom = sqrt((1 + rho) * (1 - rho));
      real a1 = (z2 / z1_safe - rho) / denom;
      real a2 = (z1_safe / z2 - rho) / denom;
      real product = z1_safe * z2;
      real delta = product < 0 || (product == 0 && (z1_safe + z2) < 0);
      return 0.5 * (Phi(z1_safe) + Phi(z2) - delta) - owens_t(z1_safe, a1) - owens_t(z2, a2);
    }
    if (z2 == 0) {
      real z2_safe = 1e-10;
      real denom = sqrt((1 + rho) * (1 - rho));
      real a1 = (z2_safe / z1 - rho) / denom;
      real a2 = (z1 / z2_safe - rho) / denom;
      real product = z1 * z2_safe;
      real delta = product < 0 || (product == 0 && (z1 + z2_safe) < 0);
      return 0.5 * (Phi(z1) + Phi(z2_safe) - delta) - owens_t(z1, a1) - owens_t(z2_safe, a2);
    }
    // General case: neither z1 nor z2 is zero
    real denom = abs(rho) < 1.0 ? sqrt((1 + rho) * (1 - rho)) : not_a_number();
    real a1 = (z2 / z1 - rho) / denom;
    real a2 = (z1 / z2 - rho) / denom;
    real product = z1 * z2;
    real delta = product < 0 || (product == 0 && (z1 + z2) < 0);
    return 0.5 * (Phi(z1) + Phi(z2) - delta) - owens_t(z1, a1) - owens_t(z2, a2);
  }

  // Compute bivariate probability for response cell (resp1, resp2)
  // mu1, mu2: means on each dimension
  // sigma1, sigma2: SDs on each dimension
  // c1, c2: threshold vectors for each dimension
  // rho: correlation
  real compute_bivariate_prob(int resp1, int resp2, int K1, int K2,
                              real mu1, real mu2, real sigma1, real sigma2,
                              vector c1, vector c2, real rho) {
    // Standardize thresholds
    vector[K1-1] z1 = (c1 - mu1) / sigma1;
    vector[K2-1] z2 = (c2 - mu2) / sigma2;
    
    // Fast path: when rho = 0, bivariate factors into product of univariates
    if (rho == 0) {
      real p1;
      real p2;
      if (resp1 == 1) p1 = Phi(z1[1]);
      else if (resp1 == K1) p1 = 1 - Phi(z1[K1-1]);
      else p1 = Phi(z1[resp1]) - Phi(z1[resp1-1]);
      if (resp2 == 1) p2 = Phi(z2[1]);
      else if (resp2 == K2) p2 = 1 - Phi(z2[K2-1]);
      else p2 = Phi(z2[resp2]) - Phi(z2[resp2-1]);
      return p1 * p2;
    }
    
    // Handle all 9 possible cases (corners, edges, interior)
    if (resp1 == 1 && resp2 == 1) {
      return binormal_cdf_custom(z1[1], z2[1], rho);
    }
    else if (resp1 == 1 && resp2 == K2) {
      return Phi(z1[1]) - binormal_cdf_custom(z1[1], z2[K2-1], rho);
    }
    else if (resp1 == K1 && resp2 == 1) {
      return Phi(z2[1]) - binormal_cdf_custom(z2[1], z1[K1-1], rho);
    }
    else if (resp1 == K1 && resp2 == K2) {
      return 1 - Phi(z1[K1-1]) - Phi(z2[K2-1]) + binormal_cdf_custom(z1[K1-1], z2[K2-1], rho);
    }
    else if (resp1 == 1) {
      return binormal_cdf_custom(z1[1], z2[resp2], rho) - 
             binormal_cdf_custom(z1[1], z2[resp2-1], rho);
    }
    else if (resp2 == 1) {
      return binormal_cdf_custom(z1[resp1], z2[1], rho) - 
             binormal_cdf_custom(z1[resp1-1], z2[1], rho);
    }
    else if (resp1 == K1) {
      return Phi(z2[resp2]) - Phi(z2[resp2-1]) - 
             binormal_cdf_custom(z1[K1-1], z2[resp2], rho) + 
             binormal_cdf_custom(z1[K1-1], z2[resp2-1], rho);
    }
    else if (resp2 == K2) {
      return Phi(z1[resp1]) - Phi(z1[resp1-1]) - 
             binormal_cdf_custom(z2[K2-1], z1[resp1], rho) + 
             binormal_cdf_custom(z1[resp1-1], z2[K2-1], rho);
    }
    else {
      // Full interior case
      return binormal_cdf_custom(z1[resp1], z2[resp2], rho) - 
             binormal_cdf_custom(z1[resp1], z2[resp2-1], rho) - 
             binormal_cdf_custom(z1[resp1-1], z2[resp2], rho) + 
             binormal_cdf_custom(z1[resp1-1], z2[resp2-1], rho);
    }
  }')

    # Add compute_bounded_prob function when bounded = TRUE
    if (isTRUE(model_data$bounded)) {
      lines <- c(lines, '
  // Bounded bivariate probability: conditional source mean clamped at 0
  real compute_bounded_prob(int resp_det, int resp_src, int K1, int K2,
                           real mu_I, real mu_S, real sigma_I, real sigma_S,
                           vector c1, vector c2, real rho) {

    if (abs(rho) < 1e-10) {
      return compute_bivariate_prob(resp_det, resp_src, K1, K2,
                                    mu_I, mu_S, sigma_I, sigma_S,
                                    c1, c2, 0.0);
    }

    real cp = mu_I - sigma_I * mu_S / (rho * sigma_S);
    real y_lo = (resp_det == 1)  ? negative_infinity() : c1[resp_det - 1];
    real y_hi = (resp_det == K1) ? positive_infinity() : c1[resp_det];

    int below_only = (y_hi <= cp) ? 1 : 0;
    int above_only = (y_lo >= cp) ? 1 : 0;

    real sigma_cond = sigma_S * sqrt(1 - rho * rho);
    real prob = 0.0;

    if (below_only == 1) {
      prob = compute_bivariate_prob(resp_det, resp_src, K1, K2,
                                     mu_I, 0.0, sigma_I, sigma_cond,
                                     c1, c2, 0.0);
    }
    else if (above_only == 1) {
      prob = compute_bivariate_prob(resp_det, resp_src, K1, K2,
                                     mu_I, mu_S, sigma_I, sigma_S,
                                     c1, c2, rho);
    }
    else {
      real z_cp = (cp - mu_I) / sigma_I;

      // Below-cp: independent, source ~ N(0, sigma_cond)
      {
        real z_ylo_b = (resp_det == 1) ? -8.0 : (c1[resp_det-1] - mu_I) / sigma_I;
        real prob_below = 0.0;
        if (resp_src == 1) {
          real zs_hi = c2[1] / sigma_cond;
          prob_below = (Phi(z_cp) - Phi(z_ylo_b)) * Phi(zs_hi);
        }
        else if (resp_src == K2) {
          real zs_lo = c2[K2-1] / sigma_cond;
          prob_below = (Phi(z_cp) - Phi(z_ylo_b)) * (1 - Phi(zs_lo));
        }
        else {
          real zs_lo = c2[resp_src-1] / sigma_cond;
          real zs_hi = c2[resp_src] / sigma_cond;
          prob_below = (Phi(z_cp) - Phi(z_ylo_b)) * (Phi(zs_hi) - Phi(zs_lo));
        }
        prob += prob_below;
      }

      // Above-cp: standard bivariate
      {
        real z_yhi_a = (resp_det == K1) ? 8.0 : (c1[resp_det] - mu_I) / sigma_I;
        vector[K2-1] z2_a = (c2 - mu_S) / sigma_S;
        real prob_above = 0.0;

        if (resp_src == 1 && resp_det == K1) {
          prob_above = Phi(z2_a[1]) - binormal_cdf_custom(z_cp, z2_a[1], rho);
        }
        else if (resp_src == K2 && resp_det == K1) {
          prob_above = 1 - Phi(z2_a[K2-1]) - Phi(z_cp)
                       + binormal_cdf_custom(z_cp, z2_a[K2-1], rho);
        }
        else if (resp_src == 1) {
          prob_above = binormal_cdf_custom(z_yhi_a, z2_a[1], rho)
                     - binormal_cdf_custom(z_cp, z2_a[1], rho);
        }
        else if (resp_src == K2) {
          prob_above = Phi(z_yhi_a) - Phi(z_cp)
                     - binormal_cdf_custom(z_yhi_a, z2_a[K2-1], rho)
                     + binormal_cdf_custom(z_cp, z2_a[K2-1], rho);
        }
        else if (resp_det == K1) {
          prob_above = Phi(z2_a[resp_src]) - Phi(z2_a[resp_src-1])
                     - binormal_cdf_custom(z_cp, z2_a[resp_src], rho)
                     + binormal_cdf_custom(z_cp, z2_a[resp_src-1], rho);
        }
        else {
          prob_above = binormal_cdf_custom(z_yhi_a, z2_a[resp_src], rho)
                     - binormal_cdf_custom(z_yhi_a, z2_a[resp_src-1], rho)
                     - binormal_cdf_custom(z_cp, z2_a[resp_src], rho)
                     + binormal_cdf_custom(z_cp, z2_a[resp_src-1], rho);
        }
        prob += prob_above;
      }
    }

    return prob;
  }')
    }

    lines <- c(lines, '

  // Bivariate SDT log-likelihood for a single observation
  real bivariate_sdt_lpmf_custom(int y1, int y2, int item_type, int K1, int K2,
                          real dprime_A, real dprime_B, real discrim_A, real discrim_B,
                          real sigma1_A, real sigma1_B, real sigma2_A, real sigma2_B,
                          real rho_A, real rho_B, real rho_N,
                          vector c1, vector c2) {
    real mu1;
    real mu2;
    real sigma1;
    real sigma2;
    real rho;
    
    // Select parameters based on item type (1=new, 2=A, 3=B)
    if (item_type == 1) {
      // New items: reference distribution
      mu1 = 0;
      mu2 = 0;
      sigma1 = 1;
      sigma2 = 1;
      rho = rho_N;
    } else if (item_type == 2) {
      // Source A items
      mu1 = dprime_A;
      mu2 = discrim_A;
      sigma1 = sigma1_A;
      sigma2 = sigma2_A;
      rho = rho_A;
    } else {
      // Source B items: parameters describe B items position on the shared
      // source axis (no negation). discrim_B is the signed source-axis
      // position of B items; rho_B is the bivariate-normal correlation for
      // B items, both in the same shared-axis frame as A items. Expect
      // discrim and discrim_B to come out with opposite signs in a fit
      // (matching A and B being on opposite ends of the source axis).
      mu1 = dprime_B;
      mu2 = discrim_B;
      sigma1 = sigma1_B;
      sigma2 = sigma2_B;
      rho = rho_B;
    }

    // Compute probability and return log
    real p = compute_bivariate_prob(y1, y2, K1, K2, mu1, mu2, sigma1, sigma2, c1, c2, rho);
    return log(fmax(p, 1e-20));
  }')

    # Add varying source criteria version if needed
    if (isTRUE(model_data$varying_source_criteria)) {
      lines <- c(lines, '
  // Bivariate SDT with source criteria varying by item response level
  real bivariate_sdt_varying_lpmf_custom(int y1, int y2, int item_type, int K1, int K2,
                          real dprime_A, real dprime_B, real discrim_A, real discrim_B,
                          real sigma1_A, real sigma1_B, real sigma2_A, real sigma2_B,
                          real rho_A, real rho_B, real rho_N,
                          vector c1, matrix c2_varying) {
    real mu1;
    real mu2;
    real sigma1;
    real sigma2;
    real rho;

    if (item_type == 1) {
      mu1 = 0; mu2 = 0; sigma1 = 1; sigma2 = 1; rho = rho_N;
    } else if (item_type == 2) {
      mu1 = dprime_A; mu2 = discrim_A; sigma1 = sigma1_A; sigma2 = sigma2_A; rho = rho_A;
    } else {
      mu1 = dprime_B; mu2 = discrim_B; sigma1 = sigma1_B; sigma2 = sigma2_B; rho = rho_B;
    }

    // Use source thresholds for this item response level
    vector[K2-1] c2 = to_vector(c2_varying[y1, ]);
    real p = compute_bivariate_prob(y1, y2, K1, K2, mu1, mu2, sigma1, sigma2, c1, c2, rho);
    return log(fmax(p, 1e-20));
  }')
    }

    # Add bounded lpmf variants when bounded = TRUE
    if (isTRUE(model_data$bounded)) {
      lines <- c(lines, '
  // Bounded bivariate SDT log-likelihood: new items use standard, old items use bounded
  real bivariate_sdt_bounded_lpmf_custom(int y1, int y2, int item_type, int K1, int K2,
                          real dprime_A, real dprime_B, real discrim_A, real discrim_B,
                          real sigma1_A, real sigma1_B, real sigma2_A, real sigma2_B,
                          real rho_A, real rho_B, real rho_N,
                          vector c1, vector c2) {
    real mu1; real mu2; real sigma1; real sigma2; real rho;
    if (item_type == 1) {
      mu1 = 0; mu2 = 0; sigma1 = 1; sigma2 = 1; rho = rho_N;
    } else if (item_type == 2) {
      // Source A on negative source axis (low source ratings); discrim_A and
      // rho_A are positive magnitudes (log/logistic links), negated here to
      // place A on the negative axis with the desired effective correlation.
      mu1 = dprime_A; mu2 = -discrim_A; sigma1 = sigma1_A; sigma2 = sigma2_A; rho = -rho_A;
    } else {
      // Source B on positive source axis (high source ratings); discrim_B and
      // rho_B used directly (positive magnitudes from log/logistic links).
      mu1 = dprime_B; mu2 = discrim_B; sigma1 = sigma1_B; sigma2 = sigma2_B; rho = rho_B;
    }
    real p;
    if (item_type == 1) {
      p = compute_bivariate_prob(y1, y2, K1, K2, mu1, mu2, sigma1, sigma2, c1, c2, rho);
    } else {
      p = compute_bounded_prob(y1, y2, K1, K2, mu1, mu2, sigma1, sigma2, c1, c2, rho);
    }
    return log(fmax(p, 1e-20));
  }')

      if (isTRUE(model_data$varying_source_criteria)) {
        lines <- c(lines, '
  // Bounded bivariate SDT with varying source criteria
  real bivariate_sdt_varying_bounded_lpmf_custom(int y1, int y2, int item_type, int K1, int K2,
                          real dprime_A, real dprime_B, real discrim_A, real discrim_B,
                          real sigma1_A, real sigma1_B, real sigma2_A, real sigma2_B,
                          real rho_A, real rho_B, real rho_N,
                          vector c1, matrix c2_varying) {
    real mu1; real mu2; real sigma1; real sigma2; real rho;
    if (item_type == 1) {
      mu1 = 0; mu2 = 0; sigma1 = 1; sigma2 = 1; rho = rho_N;
    } else if (item_type == 2) {
      // Source A on negative source axis; bounded convention: negate to place
      // A on the negative end of the shared source axis.
      mu1 = dprime_A; mu2 = -discrim_A; sigma1 = sigma1_A; sigma2 = sigma2_A; rho = -rho_A;
    } else {
      // Source B on positive source axis; used directly.
      mu1 = dprime_B; mu2 = discrim_B; sigma1 = sigma1_B; sigma2 = sigma2_B; rho = rho_B;
    }
    vector[K2-1] c2 = to_vector(c2_varying[y1, ]);
    real p;
    if (item_type == 1) {
      p = compute_bivariate_prob(y1, y2, K1, K2, mu1, mu2, sigma1, sigma2, c1, c2, rho);
    } else {
      p = compute_bounded_prob(y1, y2, K1, K2, mu1, mu2, sigma1, sigma2, c1, c2, rho);
    }
    return log(fmax(p, 1e-20));
  }')
      }
    }

    # Add new_source_criteria = "shared" lpmf variants
    if (identical(model_data$new_source_criteria, "shared")) {
      if (!isTRUE(model_data$bounded)) {
        lines <- c(lines, '
  // Varying source criteria with shared new-response thresholds
  real bivariate_sdt_varying_new_shared_lpmf_custom(int y1, int y2, int item_type, int K1, int K2,
                          real dprime_A, real dprime_B, real discrim_A, real discrim_B,
                          real sigma1_A, real sigma1_B, real sigma2_A, real sigma2_B,
                          real rho_A, real rho_B, real rho_N,
                          vector c1, matrix c2_varying, vector c2_new,
                          array[] int is_new_response) {
    real mu1; real mu2; real sigma1; real sigma2; real rho;
    if (item_type == 1) {
      mu1 = 0; mu2 = 0; sigma1 = 1; sigma2 = 1; rho = rho_N;
    } else if (item_type == 2) {
      mu1 = dprime_A; mu2 = discrim_A; sigma1 = sigma1_A; sigma2 = sigma2_A; rho = rho_A;
    } else {
      mu1 = dprime_B; mu2 = discrim_B; sigma1 = sigma1_B; sigma2 = sigma2_B; rho = rho_B;
    }
    vector[K2-1] c2;
    if (is_new_response[y1] == 1) {
      c2 = c2_new;
    } else {
      c2 = to_vector(c2_varying[y1, ]);
    }
    real p = compute_bivariate_prob(y1, y2, K1, K2, mu1, mu2, sigma1, sigma2, c1, c2, rho);
    return log(fmax(p, 1e-20));
  }')
      } else {
        lines <- c(lines, '
  // Bounded varying source criteria with shared new-response thresholds
  real bivariate_sdt_varying_bounded_new_shared_lpmf_custom(int y1, int y2, int item_type, int K1, int K2,
                          real dprime_A, real dprime_B, real discrim_A, real discrim_B,
                          real sigma1_A, real sigma1_B, real sigma2_A, real sigma2_B,
                          real rho_A, real rho_B, real rho_N,
                          vector c1, matrix c2_varying, vector c2_new,
                          array[] int is_new_response) {
    real mu1; real mu2; real sigma1; real sigma2; real rho;
    if (item_type == 1) {
      mu1 = 0; mu2 = 0; sigma1 = 1; sigma2 = 1; rho = rho_N;
    } else if (item_type == 2) {
      // Source A on negative source axis; bounded convention.
      mu1 = dprime_A; mu2 = -discrim_A; sigma1 = sigma1_A; sigma2 = sigma2_A; rho = -rho_A;
    } else {
      // Source B on positive source axis.
      mu1 = dprime_B; mu2 = discrim_B; sigma1 = sigma1_B; sigma2 = sigma2_B; rho = rho_B;
    }
    vector[K2-1] c2;
    if (is_new_response[y1] == 1) {
      c2 = c2_new;
    } else {
      c2 = to_vector(c2_varying[y1, ]);
    }
    real p;
    if (item_type == 1) {
      p = compute_bivariate_prob(y1, y2, K1, K2, mu1, mu2, sigma1, sigma2, c1, c2, rho);
    } else {
      p = compute_bounded_prob(y1, y2, K1, K2, mu1, mu2, sigma1, sigma2, c1, c2, rho);
    }
    return log(fmax(p, 1e-20));
  }')
      }
    }
  }

  # Bivariate DP-specific functions
  if (family_name == "bivariate_dp") {
    # univariate_cell_prob for marginal source probability in p2
    lines <- c(lines, '
  real univariate_cell_prob(int resp, int K, real mu, real sigma, vector thresh) {
    if (resp == 1) return Phi((thresh[1] - mu) / sigma);
    else if (resp == K) return 1 - Phi((thresh[K-1] - mu) / sigma);
    else return Phi((thresh[resp] - mu) / sigma) - Phi((thresh[resp-1] - mu) / sigma);
  }')

    # Bounded marginal source probability (for BBDP)
    if (isTRUE(model_data$bounded)) {
      lines <- c(lines, '
  // Bounded marginal source probability for p2 component (item recollected, source from familiarity)
  real compute_bounded_marginal_source(int resp_src, int K2,
                                       real mu_I, real mu_S,
                                       real sigma_I, real sigma_S,
                                       real rho, vector c2) {

    if (abs(rho) < 1e-10) {
      // Independent: marginal source is N(mu_S, sigma_S^2)
      if (resp_src == 1) return Phi((c2[1] - mu_S) / sigma_S);
      else if (resp_src == K2) return 1 - Phi((c2[K2-1] - mu_S) / sigma_S);
      else return Phi((c2[resp_src] - mu_S) / sigma_S) - Phi((c2[resp_src-1] - mu_S) / sigma_S);
    }

    real cp = mu_I - sigma_I * mu_S / (rho * sigma_S);
    real sigma_cond = sigma_S * sqrt(1 - rho * rho);
    real z_cp = (cp - mu_I) / sigma_I;
    real p_below = Phi(z_cp);

    real prob = 0.0;

    // Below-cp: source ~ N(0, sigma_cond)
    if (resp_src == 1)
      prob += p_below * Phi(c2[1] / sigma_cond);
    else if (resp_src == K2)
      prob += p_below * (1 - Phi(c2[K2-1] / sigma_cond));
    else
      prob += p_below * (Phi(c2[resp_src] / sigma_cond) - Phi(c2[resp_src-1] / sigma_cond));

    // Above-cp: standard BVN, marginal for source in bin k with item > cp
    {
      vector[K2-1] z2 = (c2 - mu_S) / sigma_S;

      if (resp_src == 1)
        prob += Phi(z2[1]) - binormal_cdf_custom(z_cp, z2[1], rho);
      else if (resp_src == K2)
        prob += 1 - Phi(z2[K2-1]) - Phi(z_cp) + binormal_cdf_custom(z_cp, z2[K2-1], rho);
      else
        prob += Phi(z2[resp_src]) - Phi(z2[resp_src-1])
              - binormal_cdf_custom(z_cp, z2[resp_src], rho)
              + binormal_cdf_custom(z_cp, z2[resp_src-1], rho);
    }

    return prob;
  }')
    }

    # BDP / BBDP lpmf: non-varying source criteria
    # Per Starns, Rotello & Hautus (2014), the unbounded BDP and bounded BBDP
    # differ in (a) whether familiarity uses standard vs bounded bivariate
    # densities (compute_bivariate_prob vs compute_bounded_prob), and
    # (b) the per-source mu2/rho assignment convention. In the bounded
    # convention, source A is anchored at the negative end of the source
    # axis (mu2_A = -discrim_A, rho = -rho_A) with discrim_A and rho_A as
    # positive magnitudes from log/logistic links; source B is at the
    # positive end with no negation. In the unbounded convention, discrim
    # and rho are signed (identity / Fisher-z links) and used directly.
    bounded <- isTRUE(model_data$bounded)
    biv_fn <- if (bounded) "compute_bounded_prob" else "compute_bivariate_prob"
    marg_fn <- if (bounded) "compute_bounded_marginal_source" else "univariate_cell_prob"
    dp_src_A <- if (bounded) {
      "mu1 = dprime_A; mu2 = -discrim_A; sigma1 = sigma1_A; sigma2 = sigma2_A; rho = -rho_A;"
    } else {
      "mu1 = dprime_A; mu2 = discrim_A; sigma1 = sigma1_A; sigma2 = sigma2_A; rho = rho_A;"
    }
    dp_src_B <- "mu1 = dprime_B; mu2 = discrim_B; sigma1 = sigma1_B; sigma2 = sigma2_B; rho = rho_B;"

    lines <- c(lines, sprintf('
  // Bivariate Dual-Process log-likelihood
  real bivariate_dp_lpmf_custom(int y1, int y2, int item_type, int K1, int K2,
                          real dprime_A, real dprime_B, real discrim_A, real discrim_B,
                          real sigma1_A, real sigma1_B, real sigma2_A, real sigma2_B,
                          real rho_A, real rho_B, real rho_N,
                          real R_I, real R_S, real R_I_B, real R_S_B,
                          vector c1, vector c2) {
    // New items: standard bivariate, no recollection
    if (item_type == 1) {
      real p = compute_bivariate_prob(y1, y2, K1, K2, 0.0, 0.0, 1.0, 1.0, c1, c2, rho_N);
      return log(fmax(p, 1e-20));
    }

    real mu1; real mu2; real sigma1; real sigma2; real rho;
    real R_I_eff; real R_S_eff;
    if (item_type == 2) {
      %s
      R_I_eff = R_I; R_S_eff = R_S;
    } else {
      %s
      R_I_eff = R_I_B; R_S_eff = R_S_B;
    }

    // p1: both from familiarity
    real p1 = (1.0 - R_I_eff) * %s(y1, y2, K1, K2, mu1, mu2, sigma1, sigma2, c1, c2, rho);

    // p2: item recollected (highest conf), source from familiarity
    real p2 = 0.0;
    if (y1 == K1) {%s
    }

    // p3: both recollected -> corner cell
    real p3 = 0.0;
    if (y1 == K1) {
      // Recollection success cells (new convention): Source A items at the
      // low end of the source axis (y2 = 1, "sure A"); Source B items at the
      // high end (y2 = K2, "sure B").
      if (item_type == 2 && y2 == 1) p3 = R_I_eff * R_S_eff;
      else if (item_type == 3 && y2 == K2) p3 = R_I_eff * R_S_eff;
    }

    return log(fmax(p1 + p2 + p3, 1e-20));
  }', dp_src_A, dp_src_B, biv_fn,
    if (bounded) {
      sprintf('\n      p2 = R_I_eff * (1.0 - R_S_eff) * compute_bounded_marginal_source(y2, K2, mu1, mu2, sigma1, sigma2, rho, c2);')
    } else {
      '\n      p2 = R_I_eff * (1.0 - R_S_eff) * univariate_cell_prob(y2, K2, mu2, sigma2, c2);'
    }))

    # Varying source criteria version
    if (isTRUE(model_data$varying_source_criteria)) {
      lines <- c(lines, sprintf('
  // Bivariate DP with varying source criteria
  real bivariate_dp_varying_lpmf_custom(int y1, int y2, int item_type, int K1, int K2,
                          real dprime_A, real dprime_B, real discrim_A, real discrim_B,
                          real sigma1_A, real sigma1_B, real sigma2_A, real sigma2_B,
                          real rho_A, real rho_B, real rho_N,
                          real R_I, real R_S, real R_I_B, real R_S_B,
                          vector c1, matrix c2_varying) {
    if (item_type == 1) {
      vector[K2-1] c2 = to_vector(c2_varying[y1, ]);
      real p = compute_bivariate_prob(y1, y2, K1, K2, 0.0, 0.0, 1.0, 1.0, c1, c2, rho_N);
      return log(fmax(p, 1e-20));
    }

    real mu1; real mu2; real sigma1; real sigma2; real rho;
    real R_I_eff; real R_S_eff;
    if (item_type == 2) {
      %s
      R_I_eff = R_I; R_S_eff = R_S;
    } else {
      %s
      R_I_eff = R_I_B; R_S_eff = R_S_B;
    }

    vector[K2-1] c2 = to_vector(c2_varying[y1, ]);
    real p1 = (1.0 - R_I_eff) * %s(y1, y2, K1, K2, mu1, mu2, sigma1, sigma2, c1, c2, rho);

    real p2 = 0.0;
    if (y1 == K1) {%s
    }

    real p3 = 0.0;
    if (y1 == K1) {
      // Recollection success cells (new convention): Source A items at the
      // low end of the source axis (y2 = 1, "sure A"); Source B items at the
      // high end (y2 = K2, "sure B").
      if (item_type == 2 && y2 == 1) p3 = R_I_eff * R_S_eff;
      else if (item_type == 3 && y2 == K2) p3 = R_I_eff * R_S_eff;
    }

    return log(fmax(p1 + p2 + p3, 1e-20));
  }', dp_src_A, dp_src_B, biv_fn,
      if (bounded) {
        '\n      p2 = R_I_eff * (1.0 - R_S_eff) * compute_bounded_marginal_source(y2, K2, mu1, mu2, sigma1, sigma2, rho, c2);'
      } else {
        '\n      p2 = R_I_eff * (1.0 - R_S_eff) * univariate_cell_prob(y2, K2, mu2, sigma2, c2);'
      }))
    }

    # new_source_criteria = "shared" variant
    if (identical(model_data$new_source_criteria, "shared")) {
      lines <- c(lines, sprintf('
  // Bivariate DP with varying source criteria + shared new-response thresholds
  real bivariate_dp_varying_new_shared_lpmf_custom(int y1, int y2, int item_type, int K1, int K2,
                          real dprime_A, real dprime_B, real discrim_A, real discrim_B,
                          real sigma1_A, real sigma1_B, real sigma2_A, real sigma2_B,
                          real rho_A, real rho_B, real rho_N,
                          real R_I, real R_S, real R_I_B, real R_S_B,
                          vector c1, matrix c2_varying, vector c2_new,
                          array[] int is_new_response) {
    // Select source thresholds based on detection RESPONSE (not item type)
    vector[K2-1] c2;
    if (is_new_response[y1] == 1) {
      c2 = c2_new;
    } else {
      c2 = to_vector(c2_varying[y1, ]);
    }

    if (item_type == 1) {
      real p = compute_bivariate_prob(y1, y2, K1, K2, 0.0, 0.0, 1.0, 1.0, c1, c2, rho_N);
      return log(fmax(p, 1e-20));
    }

    real mu1; real mu2; real sigma1; real sigma2; real rho;
    real R_I_eff; real R_S_eff;
    if (item_type == 2) {
      %s
      R_I_eff = R_I; R_S_eff = R_S;
    } else {
      %s
      R_I_eff = R_I_B; R_S_eff = R_S_B;
    }

    real p1 = (1.0 - R_I_eff) * %s(y1, y2, K1, K2, mu1, mu2, sigma1, sigma2, c1, c2, rho);

    real p2 = 0.0;
    if (y1 == K1) {%s
    }

    real p3 = 0.0;
    if (y1 == K1) {
      // Recollection success cells (new convention): Source A items at the
      // low end of the source axis (y2 = 1, "sure A"); Source B items at the
      // high end (y2 = K2, "sure B").
      if (item_type == 2 && y2 == 1) p3 = R_I_eff * R_S_eff;
      else if (item_type == 3 && y2 == K2) p3 = R_I_eff * R_S_eff;
    }

    return log(fmax(p1 + p2 + p3, 1e-20));
  }', dp_src_A, dp_src_B, biv_fn,
      if (bounded) {
        '\n      p2 = R_I_eff * (1.0 - R_S_eff) * compute_bounded_marginal_source(y2, K2, mu1, mu2, sigma1, sigma2, rho, c2);'
      } else {
        '\n      p2 = R_I_eff * (1.0 - R_S_eff) * univariate_cell_prob(y2, K2, mu2, sigma2, c2);'
      }))
    }
  }

  # Bivariate cumulative lpmf (simple: no item_type branching, sigma=1)
  if (family_name == "bivariate_cumulative") {
    lines <- c(lines, '
  real bivariate_cumulative_lpmf_custom(int y1, int y2, int K1, int K2,
                                         real mu1, real mu2,
                                         vector c1, vector c2, real rho) {
    real p = compute_bivariate_prob(y1, y2, K1, K2, mu1, mu2, 1.0, 1.0, c1, c2, rho);
    return log(fmax(p, 1e-20));
  }')
  }

  # CDP functions (for Remember/Know paradigms)
  if (family_name == "cdp") {
    lines <- c(lines, '
  // Bivariate normal CDF using Owens T function (shared with bivariate_sdt)
  real binormal_cdf_cdp(real z1, real z2, real rho) {
    if (z1 == 0 && z2 == 0) {
      return 0.25 + asin(rho) / (2 * pi());
    }
    if (z1 == 0) {
      real z1_safe = 1e-10;
      real denom = sqrt((1 + rho) * (1 - rho));
      real a1 = (z2 / z1_safe - rho) / denom;
      real a2 = (z1_safe / z2 - rho) / denom;
      real product = z1_safe * z2;
      real delta = product < 0 || (product == 0 && (z1_safe + z2) < 0);
      return 0.5 * (Phi(z1_safe) + Phi(z2) - delta) - owens_t(z1_safe, a1) - owens_t(z2, a2);
    }
    if (z2 == 0) {
      real z2_safe = 1e-10;
      real denom = sqrt((1 + rho) * (1 - rho));
      real a1 = (z2_safe / z1 - rho) / denom;
      real a2 = (z1 / z2_safe - rho) / denom;
      real product = z1 * z2_safe;
      real delta = product < 0 || (product == 0 && (z1 + z2_safe) < 0);
      return 0.5 * (Phi(z1) + Phi(z2_safe) - delta) - owens_t(z1, a1) - owens_t(z2_safe, a2);
    }
    real denom = abs(rho) < 1.0 ? sqrt((1 + rho) * (1 - rho)) : not_a_number();
    real a1 = (z2 / z1 - rho) / denom;
    real a2 = (z1 / z2 - rho) / denom;
    real product = z1 * z2;
    real delta = product < 0 || (product == 0 && (z1 + z2) < 0);
    return 0.5 * (Phi(z1) + Phi(z2) - delta) - owens_t(z1, a1) - owens_t(z2, a2);
  }
  
  // P(R > c_R, tau_lo < M < tau_hi) - "Remember" response in confidence band
  real binormal_strip_upper(real z_cR, real z_tau_lo, real z_tau_hi, real rho) {
    real p_upper = Phi(z_tau_hi) - binormal_cdf_cdp(z_cR, z_tau_hi, rho);
    real p_lower = Phi(z_tau_lo) - binormal_cdf_cdp(z_cR, z_tau_lo, rho);
    return fmax(p_upper - p_lower, 1e-20);
  }
  
  // P(R <= c_R, tau_lo < M < tau_hi) - "Know" response in confidence band
  real binormal_strip_lower(real z_cR, real z_tau_lo, real z_tau_hi, real rho) {
    real p_band = Phi(z_tau_hi) - Phi(z_tau_lo);
    return fmax(p_band - binormal_strip_upper(z_cR, z_tau_lo, z_tau_hi, rho), 1e-20);
  }')

    # R/K/G functions (only emitted when n_rkg == 3)
    if (isTRUE(model_data$stan_data$n_rkg == 3)) {
      lines <- c(lines, '
  // G(c) = P(R <= c_R, F <= c_K, M < c) -- unified formula (Guess-first approach)
  // Uses fmin for exact min (no smooth approximation needed)
  real G_cdp(real c, real mu_R, real sigma_R, real mu_F, real sigma_F,
             real c_R, real c_K) {
    real s = fmin(c_R, c - c_K);
    real z_s = (s - mu_R) / sigma_R;
    real z_cR = (c_R - mu_R) / sigma_R;
    real z_cK = (c_K - mu_F) / sigma_F;
    real mu_M = mu_R + mu_F;
    real sigma_M = sqrt(square(sigma_R) + square(sigma_F));
    real z_c = (c - mu_M) / sigma_M;
    real rho = sigma_R / sigma_M;
    return Phi(z_s) * Phi(z_cK) + binormal_cdf_cdp(z_cR, z_c, rho)
           - binormal_cdf_cdp(z_s, z_c, rho);
  }

  // Compute R/K/G probabilities using G(c) (Guess-first approach)
  // Shares binormal CDF values between Remember and Guess to avoid
  // redundant Owen T evaluations.
  vector compute_rkg_probs(real mu_R, real sigma_R, real mu_F, real sigma_F,
                           real c_R, real c_K, real tau_lo, real tau_hi) {
    vector[3] probs;
    real mu_M = mu_R + mu_F;
    real sigma_M = sqrt(square(sigma_R) + square(sigma_F));
    real rho = sigma_R / sigma_M;
    real z_cR = (c_R - mu_R) / sigma_R;
    real z_cK_F = (c_K - mu_F) / sigma_F;

    // Standardized thresholds
    real z_lo = (tau_lo - mu_M) / sigma_M;
    real z_hi = (tau_hi - mu_M) / sigma_M;

    // SHARED: Phi2(z_cR, z_hi/z_lo, rho) -- used by both Remember and Guess
    real bv_cR_hi = binormal_cdf_cdp(z_cR, z_hi, rho);
    real bv_cR_lo = binormal_cdf_cdp(z_cR, z_lo, rho);

    // REMEMBER: P(R > c_R, tau_lo < M < tau_hi)
    real p_Remember = (Phi(z_hi) - bv_cR_hi) - (Phi(z_lo) - bv_cR_lo);

    // GUESS: G(tau_hi) - G(tau_lo) -- inlined to reuse bv_cR
    // G(c) = Phi(z_s)*Phi(z_cK_F) + Phi2(z_cR, z_c, rho) - Phi2(z_s, z_c, rho)
    // where s = fmin(c_R, c-c_K), z_c = z_hi or z_lo (same as thresholds!)
    real s_hi = fmin(c_R, tau_hi - c_K);
    real z_s_hi = (s_hi - mu_R) / sigma_R;
    real G_hi;
    if (c_R <= tau_hi - c_K) {
      G_hi = Phi(z_cR) * Phi(z_cK_F);
    } else {
      G_hi = Phi(z_s_hi) * Phi(z_cK_F) + bv_cR_hi - binormal_cdf_cdp(z_s_hi, z_hi, rho);
    }

    real s_lo = fmin(c_R, tau_lo - c_K);
    real z_s_lo = (s_lo - mu_R) / sigma_R;
    real G_lo;
    if (c_R <= tau_lo - c_K) {
      G_lo = Phi(z_cR) * Phi(z_cK_F);
    } else {
      G_lo = Phi(z_s_lo) * Phi(z_cK_F) + bv_cR_lo - binormal_cdf_cdp(z_s_lo, z_lo, rho);
    }

    real p_Guess = G_hi - G_lo;

    // KNOW: remainder
    real p_Band = Phi(z_hi) - Phi(z_lo);
    real p_Know = p_Band - p_Remember - p_Guess;

    // Soft clamp
    real kk = 10000.0;
    probs[1] = log1p_exp(kk * p_Remember) / kk;
    probs[2] = log1p_exp(kk * p_Know) / kk;
    probs[3] = log1p_exp(kk * p_Guess) / kk;
    return probs;
  }')
    }
  }

  # 2D-VRDP functions (for conjoint item-source recognition)
  if (family_name == "vrdp2d") {
    lines <- c(lines, '
  // Bivariate normal CDF using Owens T function (same as bivariate_sdt)
  real binormal_cdf_vrdp(real z1, real z2, real rho) {
    if (z1 == 0 && z2 == 0) {
      return 0.25 + asin(rho) / (2 * pi());
    }
    if (z1 == 0) {
      real z1_safe = 1e-10;
      real denom = sqrt((1 + rho) * (1 - rho));
      real a1 = (z2 / z1_safe - rho) / denom;
      real a2 = (z1_safe / z2 - rho) / denom;
      real product = z1_safe * z2;
      real delta = product < 0 || (product == 0 && (z1_safe + z2) < 0);
      return 0.5 * (Phi(z1_safe) + Phi(z2) - delta) - owens_t(z1_safe, a1) - owens_t(z2, a2);
    }
    if (z2 == 0) {
      real z2_safe = 1e-10;
      real denom = sqrt((1 + rho) * (1 - rho));
      real a1 = (z2_safe / z1 - rho) / denom;
      real a2 = (z1 / z2_safe - rho) / denom;
      real product = z1 * z2_safe;
      real delta = product < 0 || (product == 0 && (z1 + z2_safe) < 0);
      return 0.5 * (Phi(z1) + Phi(z2_safe) - delta) - owens_t(z1, a1) - owens_t(z2_safe, a2);
    }
    real denom = abs(rho) < 1.0 ? sqrt((1 + rho) * (1 - rho)) : not_a_number();
    real a1 = (z2 / z1 - rho) / denom;
    real a2 = (z1 / z2 - rho) / denom;
    real product = z1 * z2;
    real delta = product < 0 || (product == 0 && (z1 + z2) < 0);
    return 0.5 * (Phi(z1) + Phi(z2) - delta) - owens_t(z1, a1) - owens_t(z2, a2);
  }

  // Compute bivariate probability for response cell (resp1, resp2) 
  // rho = 0 means circular (uncorrelated) distribution
  real compute_bivariate_prob_vrdp(int resp1, int resp2, int K1, int K2,
                                   real mu1, real mu2, real sigma1, real sigma2,
                                   vector c1, vector c2, real rho) {
    // Standardize thresholds
    vector[K1-1] z1 = (c1 - mu1) / sigma1;
    vector[K2-1] z2 = (c2 - mu2) / sigma2;
    
    // Handle all 9 possible cases (corners, edges, interior)
    if (resp1 == 1 && resp2 == 1) {
      return binormal_cdf_vrdp(z1[1], z2[1], rho);
    }
    else if (resp1 == 1 && resp2 == K2) {
      return Phi(z1[1]) - binormal_cdf_vrdp(z1[1], z2[K2-1], rho);
    }
    else if (resp1 == K1 && resp2 == 1) {
      return Phi(z2[1]) - binormal_cdf_vrdp(z2[1], z1[K1-1], rho);
    }
    else if (resp1 == K1 && resp2 == K2) {
      return 1 - Phi(z1[K1-1]) - Phi(z2[K2-1]) + binormal_cdf_vrdp(z1[K1-1], z2[K2-1], rho);
    }
    else if (resp1 == 1) {
      return binormal_cdf_vrdp(z1[1], z2[resp2], rho) - 
             binormal_cdf_vrdp(z1[1], z2[resp2-1], rho);
    }
    else if (resp2 == 1) {
      return binormal_cdf_vrdp(z1[resp1], z2[1], rho) - 
             binormal_cdf_vrdp(z1[resp1-1], z2[1], rho);
    }
    else if (resp1 == K1) {
      return Phi(z2[resp2]) - Phi(z2[resp2-1]) - 
             binormal_cdf_vrdp(z1[K1-1], z2[resp2], rho) + 
             binormal_cdf_vrdp(z1[K1-1], z2[resp2-1], rho);
    }
    else if (resp2 == K2) {
      return Phi(z1[resp1]) - Phi(z1[resp1-1]) - 
             binormal_cdf_vrdp(z2[K2-1], z1[resp1], rho) + 
             binormal_cdf_vrdp(z1[resp1-1], z2[K2-1], rho);
    }
    else {
      // Full interior case
      return binormal_cdf_vrdp(z1[resp1], z2[resp2], rho) - 
             binormal_cdf_vrdp(z1[resp1], z2[resp2-1], rho) - 
             binormal_cdf_vrdp(z1[resp1-1], z2[resp2], rho) + 
             binormal_cdf_vrdp(z1[resp1-1], z2[resp2-1], rho);
    }
  }

  // 2D-VRDP log-likelihood for a single observation
  // Since rho=0 for all distributions, bivariate normal factors into product of univariates:
  // P(X <= x, Y <= y | rho=0) = Phi(x) * Phi(y)
  //
  // This is MUCH more efficient than computing bivariate CDFs!
  //
  // item_type: 1=new, 2=source A, 3=source B
  // dprime_F: familiarity strength 
  // dprime_R: recollection boost
  // dprime_S: source discriminability (only for recollected items)
  // R: recollection probability
  // sigma_S: SD on source dimension for recollected items
  
  // Helper: compute univariate cell probability
  real univariate_cell_prob(int resp, int K, real mu, real sigma, vector thresh) {
    if (resp == 1) {
      return Phi((thresh[1] - mu) / sigma);
    } else if (resp == K) {
      return 1 - Phi((thresh[K-1] - mu) / sigma);
    } else {
      return Phi((thresh[resp] - mu) / sigma) - Phi((thresh[resp-1] - mu) / sigma);
    }
  }
  
  real vrdp2d_lpmf_custom(int y1, int y2, int item_type, int K1, int K2,
                          real dprime_F, real dprime_R, real dprime_S, real dprime_S_B,
                          real R, real sigma_item, real sigma_S,
                          vector c1, vector c2) {
    real p;
    
    if (item_type == 1) {
      // New items: at (0, 0) with SD=1 on both dimensions
      real p1 = univariate_cell_prob(y1, K1, 0.0, 1.0, c1);
      real p2 = univariate_cell_prob(y2, K2, 0.0, 1.0, c2);
      p = p1 * p2;
    } else {
      // Old items: mixture of non-recollected and recollected
      // Source discriminability: Source A gets dprime_S, Source B gets dprime_S_B
      real source_d = (item_type == 2) ? dprime_S : dprime_S_B;
      
      // Non-recollected: at (dprime_F, 0) - item memory but no source info
      real p1_fam = univariate_cell_prob(y1, K1, dprime_F, sigma_item, c1);
      real p2_fam = univariate_cell_prob(y2, K2, 0.0, 1.0, c2);
      real p_fam = p1_fam * p2_fam;
      
      // Recollected: at (dprime_F + dprime_R, source_d)
      real p1_rec = univariate_cell_prob(y1, K1, dprime_F + dprime_R, sigma_item, c1);
      real p2_rec = univariate_cell_prob(y2, K2, source_d, sigma_S, c2);
      real p_rec = p1_rec * p2_rec;
      
      // Mixture
      p = (1.0 - R) * p_fam + R * p_rec;
    }
    
    return log(fmax(p, 1e-20));
  }
  
  // Version with source criteria varying by item response
  // thresh2_varying is a K1 x (K2-1) matrix where each row is thresholds for that item response
  real vrdp2d_varying_lpmf_custom(int y1, int y2, int item_type, int K1, int K2,
                                   real dprime_F, real dprime_R, real dprime_S, real dprime_S_B,
                                   real R, real sigma_item, real sigma_S,
                                   vector c1, matrix c2_varying) {
    real p;
    
    // Get the source thresholds for this item response level
    vector[K2-1] c2 = to_vector(c2_varying[y1, ]);
    
    if (item_type == 1) {
      // New items: at (0, 0) with SD=1 on both dimensions
      real p1 = univariate_cell_prob(y1, K1, 0.0, 1.0, c1);
      real p2 = univariate_cell_prob(y2, K2, 0.0, 1.0, c2);
      p = p1 * p2;
    } else {
      // Old items: mixture of non-recollected and recollected
      real source_d = (item_type == 2) ? dprime_S : dprime_S_B;
      
      // Non-recollected: at (dprime_F, 0)
      real p1_fam = univariate_cell_prob(y1, K1, dprime_F, sigma_item, c1);
      real p2_fam = univariate_cell_prob(y2, K2, 0.0, 1.0, c2);
      real p_fam = p1_fam * p2_fam;
      
      // Recollected: at (dprime_F + dprime_R, source_d)
      real p1_rec = univariate_cell_prob(y1, K1, dprime_F + dprime_R, sigma_item, c1);
      real p2_rec = univariate_cell_prob(y2, K2, source_d, sigma_S, c2);
      real p_rec = p1_rec * p2_rec;
      
      // Mixture
      p = (1.0 - R) * p_fam + R * p_rec;
    }
    
    return log(fmax(p, 1e-20));
  }')
  }
  
  # Threading: add partial_log_lik before closing functions block
  # When batch + threading: use the batch partial wrapper instead of the standard one
  if (isTRUE(model_data$threads) && is.null(batch_info)) {
    lines <- c(lines, "", generate_partial_log_lik(model_data, family))
  }

  # Batch likelihood: add function declaration (implemented in C++ user header)
  if (!is.null(batch_info)) {
    lines <- c(lines, "", "  // Batch C++ likelihood (implemented in user header)", batch_info$stan_decl)
    # If threading, add the batch partial_log_lik wrapper (replaces the standard one)
    if (isTRUE(model_data$threads)) {
      lines <- c(lines, "", batch_info$stan_partial)
    }
  }

  lines <- c(lines, "}")
  paste(lines, collapse = "\n")
}


#' Generate link CDF wrapper function
#' @noRd
generate_link_cdf_wrapper <- function(link) {
  link_name <- link$name
  
  if (link_name == "probit") {
    return('
  // Probit link (standard normal CDF)
  real link_cdf(real x) {
    return Phi(x);
  }')
  } else if (link_name == "logit") {
    return('
  // Logit link (logistic CDF)
  real link_cdf(real x) {
    return inv_logit(x);
  }')
  } else {
    stop("Unknown link function: ", link_name)
  }
}


generate_data_block_v2 <- function(model_data, family, batch_info = NULL) {
  family_name <- if (is.list(family)) family$family else family
  is_cumulative <- family_name == "cumulative"
  is_source_mixture <- family_name == "source_mixture"
  is_bivariate <- family_name %in% c("bivariate_sdt", "bivariate_dp", "bivariate_cumulative")
  is_cdp <- family_name == "cdp"
  is_vrdp2d <- family_name == "vrdp2d"

  if (is_bivariate || is_vrdp2d) {
    # Bivariate SDT and 2D-VRDP have same data structure
    lines <- c(
      "data {",
      "  int<lower=1> N;",
      "  int<lower=2> K1;  // Item response categories",
      "  int<lower=2> K2;  // Source response categories",
      "  array[N] int<lower=1, upper=K1> y;   // Item responses",
      "  array[N] int<lower=1, upper=K2> y2;  // Source responses",
      "  array[N] int<lower=1, upper=3> item_type;  // 1=new, 2=A, 3=B",
      if (identical(model_data$new_source_criteria, "shared"))
        "  array[K1] int<lower=0, upper=1> is_new_response;  // 1=new response, 0=old response"
    )
  } else if (is_cdp) {
    # CDP model structure
    J <- model_data$stan_data$J
    n_new <- model_data$stan_data$n_new
    lines <- c(
      "data {",
      "  int<lower=1> N;",
      "  int<lower=2> K;  // Total confidence levels",
      sprintf("  int<lower=1> J;  // Number of old confidence levels (=%d)", J),
      sprintf("  int<lower=1> n_new;  // Number of new confidence levels (=%d)", n_new),
      "  array[N] int<lower=1, upper=K> y;  // Confidence response",
      sprintf("  int<lower=2, upper=3> n_rkg;  // 2=R/K, 3=R/K/G (=%d)", model_data$stan_data$n_rkg),
      "  array[N] int<lower=1, upper=n_rkg> rk;  // 1=R, 2=K, (3=G)",
      "  array[N] real is_old;  // 0=lure, 1=target (or centered: -0.5/0.5)",
      "  array[J] int<lower=1> old_level_map;  // Maps 1:J to actual confidence levels",
      "  array[n_new] int<lower=1> new_level_map;  // Maps 1:n_new to actual confidence levels"
    )
  } else {
    lines <- c(
      "data {",
      "  int<lower=1> N;",
      "  int<lower=2> K;",
      "  array[N] int<lower=1, upper=K> y;"
    )
    
    # For source_mixture, use source instead of is_old
    if (is_source_mixture) {
      lines <- c(lines, "  array[N] int<lower=0, upper=1> source;  // 0 = A, 1 = B")
    } else if (!is_cumulative) {
      # Only include is_old for SDT families (not cumulative)
      lines <- c(lines, "  array[N] real is_old;")
    }
  }
  
  # Counts for aggregated data
  if (isTRUE(model_data$has_counts)) {
    lines <- c(lines, "  array[N] int<lower=1> counts;  // Aggregation counts")
  }
  
  # Dprime fixed effects (skip if no fixed effects, e.g., cumulative with only random effects)
  has_dprime_fixed <- isTRUE(model_data$has_dprime_fixed) || is.null(model_data$has_dprime_fixed)
  if (has_dprime_fixed) {
    lines <- c(lines, "",
               "  // d' fixed effects design matrix",
               "  int<lower=1> P_dprime;",
               "  matrix[N, P_dprime] X_dprime;"
    )
  }
  
  lines <- c(lines, "",
             "  // Criterion fixed effects design matrix", 
             "  int<lower=1> P_criterion;",
             "  matrix[N, P_criterion] X_criterion;"
  )
  
  if (model_data$has_sigma) {
    lines <- c(lines, "", "  int<lower=1> P_sigma;", "  matrix[N, P_sigma] X_sigma;")
  }
  if (model_data$has_lambda) {
    lines <- c(lines, "", "  int<lower=1> P_lambda;", "  matrix[N, P_lambda] X_lambda;")
  }
  if (model_data$has_dprime2) {
    lines <- c(lines, "", "  int<lower=1> P_dprime2;", "  matrix[N, P_dprime2] X_dprime2;")
  }
  if (model_data$has_sigma2) {
    lines <- c(lines, "", "  int<lower=1> P_sigma2;", "  matrix[N, P_sigma2] X_sigma2;")
  }
  if (isTRUE(model_data$has_dprime_B)) {
    lines <- c(lines, "", "  int<lower=1> P_dprime_B;", "  matrix[N, P_dprime_B] X_dprime_B;")
  }
  if (isTRUE(model_data$has_lambda_B)) {
    lines <- c(lines, "", "  int<lower=1> P_lambda_B;", "  matrix[N, P_lambda_B] X_lambda_B;")
  }
  # Bivariate SDT parameters (skip discrim data if no fixed effects, e.g., bivariate_cumulative)
  has_discrim_fixed <- isTRUE(model_data$has_discrim_fixed) || (isTRUE(model_data$has_discrim) && is.null(model_data$has_discrim_fixed))
  if (isTRUE(model_data$has_discrim) && has_discrim_fixed) {
    lines <- c(lines, "", "  int<lower=1> P_discrim;", "  matrix[N, P_discrim] X_discrim;")
  }
  if (isTRUE(model_data$has_discrim_B)) {
    lines <- c(lines, "", "  int<lower=1> P_discrim_B;", "  matrix[N, P_discrim_B] X_discrim_B;")
  }
  if (isTRUE(model_data$has_sigma_B)) {
    lines <- c(lines, "", "  int<lower=1> P_sigma_B;", "  matrix[N, P_sigma_B] X_sigma_B;")
  }
  if (isTRUE(model_data$has_sigma2_B)) {
    lines <- c(lines, "", "  int<lower=1> P_sigma2_B;", "  matrix[N, P_sigma2_B] X_sigma2_B;")
  }
  if (isTRUE(model_data$has_rho)) {
    lines <- c(lines, "", "  int<lower=1> P_rho;", "  matrix[N, P_rho] X_rho;")
  }
  if (isTRUE(model_data$has_rho_B)) {
    lines <- c(lines, "", "  int<lower=1> P_rho_B;", "  matrix[N, P_rho_B] X_rho_B;")
  }
  if (isTRUE(model_data$has_rho_N)) {
    lines <- c(lines, "", "  int<lower=1> P_rho_N;", "  matrix[N, P_rho_N] X_rho_N;")
  }
  if (isTRUE(model_data$has_rec_crit)) {
    lines <- c(lines, "", "  // rec_crit (recollection criterion) fixed effects",
               "  int<lower=1> P_rec_crit;", "  matrix[N, P_rec_crit] X_rec_crit;")
  }
  if (isTRUE(model_data$has_know_crit)) {
    lines <- c(lines, "", "  // know_crit (know criterion) fixed effects",
               "  int<lower=1> P_know_crit;", "  matrix[N, P_know_crit] X_know_crit;")
  }
  if (isTRUE(model_data$has_dprime_L)) {
    lines <- c(lines, "", "  int<lower=1> P_dprime_L;", "  matrix[N, P_dprime_L] X_dprime_L;")
  }
  if (isTRUE(model_data$has_sigma_L)) {
    lines <- c(lines, "", "  int<lower=1> P_sigma_L;", "  matrix[N, P_sigma_L] X_sigma_L;")
  }
  if (isTRUE(model_data$has_lambda_L)) {
    lines <- c(lines, "", "  int<lower=1> P_lambda_L;", "  matrix[N, P_lambda_L] X_lambda_L;")
  }
  if (isTRUE(model_data$has_lambda2)) {
    lines <- c(lines, "", "  int<lower=1> P_lambda2;", "  matrix[N, P_lambda2] X_lambda2;")
  }
  if (isTRUE(model_data$has_lambda2_B)) {
    lines <- c(lines, "", "  int<lower=1> P_lambda2_B;", "  matrix[N, P_lambda2_B] X_lambda2_B;")
  }
  if (isTRUE(model_data$has_criterion2)) {
    lines <- c(lines, "", "  // Criterion2 (discrimination) fixed effects design matrix",
               "  int<lower=1> P_criterion2;", "  matrix[N, P_criterion2] X_criterion2;")
  }
  
  # Grouping factors
  all_groups <- get_all_groups(model_data)
  if (length(all_groups) > 0) {
    lines <- c(lines, "", "  // Random effect grouping factors")
    for (group in all_groups) {
      lines <- c(lines, 
                 sprintf("  int<lower=1> N_%s;", group),
                 sprintf("  array[N] int<lower=1, upper=N_%s> %s;", group, group))
    }
  }
  
  # Term indices (for fast index-based RE lookup)
  term_indices <- get_all_term_indices(model_data)
  if (length(term_indices) > 0) {
    lines <- c(lines, "", "  // Term indices for varying slopes")
    for (ti in term_indices) {
      lines <- c(lines,
                 sprintf("  int<lower=1> %s;", ti$dim_name),
                 sprintf("  array[N] int<lower=0, upper=%s> %s;", ti$dim_name, ti$idx_name))
    }
  }
  
  # Z matrices (for general RE structures)
  z_matrices <- get_all_z_matrices(model_data)
  if (length(z_matrices) > 0) {
    lines <- c(lines, "", "  // Random effect design matrices")
    for (zm in z_matrices) {
      lines <- c(lines, sprintf("  matrix[N, %d] %s;", zm$dim, zm$name))
    }
  }
  
  # Threading support: grainsize for reduce_sum
  if (isTRUE(model_data$threads)) {
    lines <- c(lines, "", "  // Threading (reduce_sum)",
               "  int grainsize;")
  }

  # Smooth term data declarations (Zs matrices and dimension integers)
  lines <- c(lines, generate_smooth_data_declarations(model_data))

  lines <- c(lines, "}")
  paste(lines, collapse = "\n")
}


generate_transformed_data_block_v2 <- function(model_data, family) {
  lines <- c("transformed data {")
  
  # Include bivariate parameters and CDP parameters in the list
  param_names <- c("dprime", "sigma", "lambda", "dprime2", "sigma2", "dprime_B", "lambda_B",
                   "discrim", "discrim_B", "sigma_B", "sigma2_B", "rho", "rho_B", "rho_N", "rec_crit", "know_crit",
                   "dprime_L", "sigma_L", "lambda_L", "lambda2", "lambda2_B")
  for (pname in param_names) {
    re_list <- model_data[[paste0(pname, "_random")]]
    if (is.null(re_list)) next
    for (group in names(re_list)) {
      re <- re_list[[group]]
      if (is.null(re$cor_id) && re$dim > 1) {
        lines <- c(lines, sprintf("  int D_%s_%s = %d;", pname, group, re$dim))
      }
    }
  }
  
  for (group in names(model_data$criterion$random)) {
    re <- model_data$criterion$random[[group]]
    if (is.null(re$cor_id)) {
      lines <- c(lines, sprintf("  int D_criterion_%s = %d;", group, re$dim))
    }
  }
  
  # Criterion2 RE dimensions (for bivariate_sdt)
  if (isTRUE(model_data$has_criterion2) && !is.null(model_data$criterion2$random)) {
    for (group in names(model_data$criterion2$random)) {
      re <- model_data$criterion2$random[[group]]
      if (is.null(re$cor_id)) {
        lines <- c(lines, sprintf("  int D_criterion2_%s = %d;", group, re$dim))
      }
    }
  }
  
  for (cor_id in names(model_data$cross_cor)) {
    cc <- model_data$cross_cor[[cor_id]]
    lines <- c(lines, sprintf("  int D_cross_%s_%s = %d;", cor_id, cc$group, cc$total_dim))
  }

  # Mid-anchor threshold indices
  family_name_td <- if (is.list(family)) family$family else family
  is_bivariate_td <- family_name_td %in% c("bivariate_sdt", "bivariate_dp", "bivariate_cumulative")
  is_vrdp2d_td <- family_name_td == "vrdp2d"
  is_cdp_td <- family_name_td == "cdp"

  if (is_bivariate_td || is_vrdp2d_td) {
    # Compute mid-anchor indices from R-side data to avoid Stan int-division warnings
    K1_val <- model_data$K
    lines <- c(lines, sprintf("  int mid_thresh1 = %d;  // mid-anchor for detection thresholds", K1_val %/% 2))
    if (isTRUE(model_data$has_criterion2)) {
      K2_val <- model_data$K2
      lines <- c(lines, sprintf("  int mid_thresh2 = %d;  // mid-anchor for source thresholds", K2_val %/% 2))
    }
  } else if (is_cdp_td) {
    J_val <- model_data$stan_data$J
    lines <- c(lines, sprintf("  int mid_thresh_cdp = %d;  // mid-anchor for tau thresholds", (J_val + 1) %/% 2))
  } else {
    K_val <- model_data$K
    lines <- c(lines, sprintf("  int mid_thresh = %d;  // mid-anchor for thresholds", K_val %/% 2))
  }



  # Threading support: seq_n index array for reduce_sum slicing
  if (isTRUE(model_data$threads)) {
    lines <- c(lines, "",
               "  // Observation index array for reduce_sum",
               "  array[N] int seq_n;",
               "  for (i in 1:N) seq_n[i] = i;")
  }

  lines <- c(lines, "}")
  paste(lines, collapse = "\n")
}


generate_parameters_block_v2 <- function(model_data, family) {
  priors <- model_data$priors
  has_dprime_fixed <- isTRUE(model_data$has_dprime_fixed) || is.null(model_data$has_dprime_fixed)
  has_discrim_fixed <- isTRUE(model_data$has_discrim_fixed) || (isTRUE(model_data$has_discrim) && is.null(model_data$has_discrim_fixed))
  family_name <- if (is.list(family)) family$family else family
  is_bivariate <- family_name %in% c("bivariate_sdt", "bivariate_dp", "bivariate_cumulative")
  is_vrdp2d <- family_name == "vrdp2d"

  lines <- c("parameters {")
  
  # For mixture with both dprime and dprime2 free, use ordered constraint
  # The ordered constraint applies to each column of the design matrix
  if (model_data$needs_ordered_dprime) {
    lines <- c(lines,
               "  // Ordered constraint for identifiability: dprime > dprime2 for each predictor",
               "  array[P_dprime] ordered[2] beta_dprime_ordered;  // [p][1] = dprime2, [p][2] = dprime")
  } else if (has_dprime_fixed) {
    lines <- c(lines, "  vector[P_dprime] beta_dprime;")
    if (model_data$has_dprime2) {
      lines <- c(lines, "  vector[P_dprime2] beta_dprime2;")
    }
  }
  
  # Criterion thresholds - use K1 for bivariate/vrdp2d, J for CDP, K otherwise
  family_name <- if (is.list(family)) family$family else family
  is_cdp <- family_name == "cdp"
  
  if (is_bivariate || is_vrdp2d) {
    lines <- c(lines,
               "  vector[P_criterion] beta_thresh_mid;",
               "  matrix[K1-2, P_criterion] beta_log_gaps;")
  } else if (is_cdp) {
    lines <- c(lines,
               "  vector[P_criterion] beta_thresh_mid;  // Mid-anchor tau threshold",
               "  matrix[J-1, P_criterion] beta_log_gaps;  // Log-gaps for tau")
  } else {
    lines <- c(lines, "  vector[P_criterion] beta_thresh_mid;")
    if (model_data$K > 2) {
      lines <- c(lines, "  matrix[K-2, P_criterion] beta_log_gaps;")
    }
  }
  
  if (model_data$has_sigma) lines <- c(lines, "  vector[P_sigma] beta_sigma;")
  if (model_data$has_lambda) lines <- c(lines, "  vector[P_lambda] beta_lambda;")
  if (model_data$has_sigma2) lines <- c(lines, "  vector[P_sigma2] beta_sigma2;")
  if (isTRUE(model_data$has_dprime_B)) lines <- c(lines, "  vector[P_dprime_B] beta_dprime_B;")
  if (isTRUE(model_data$has_lambda_B)) lines <- c(lines, "  vector[P_lambda_B] beta_lambda_B;")
  
  # Bivariate SDT parameters
  if (isTRUE(model_data$has_discrim) && has_discrim_fixed) lines <- c(lines, "  vector[P_discrim] beta_discrim;")
  if (isTRUE(model_data$has_discrim_B)) lines <- c(lines, "  vector[P_discrim_B] beta_discrim_B;")
  if (isTRUE(model_data$has_sigma_B)) lines <- c(lines, "  vector[P_sigma_B] beta_sigma_B;")
  if (isTRUE(model_data$has_sigma2_B)) lines <- c(lines, "  vector[P_sigma2_B] beta_sigma2_B;")
  if (isTRUE(model_data$has_rho)) lines <- c(lines, "  vector[P_rho] beta_rho;  // Fisher z scale")
  if (isTRUE(model_data$has_rho_B)) lines <- c(lines, "  vector[P_rho_B] beta_rho_B;  // Fisher z scale")
  if (isTRUE(model_data$has_rho_N)) lines <- c(lines, "  vector[P_rho_N] beta_rho_N;  // Fisher z scale")
  if (isTRUE(model_data$has_lambda2)) lines <- c(lines, "  vector[P_lambda2] beta_lambda2;  // R_S source recollection (logit scale)")
  if (isTRUE(model_data$has_lambda2_B)) lines <- c(lines, "  vector[P_lambda2_B] beta_lambda2_B;  // R_S_B source recollection for Source B (logit scale)")
  if (isTRUE(model_data$has_rec_crit)) {
    lines <- c(lines, "  vector[P_rec_crit] beta_rec_crit;  // Recollection criterion")
  }
  if (isTRUE(model_data$has_know_crit)) {
    lines <- c(lines, "  vector[P_know_crit] beta_know_crit;  // Know criterion (R/K/G)")
  }
  if (isTRUE(model_data$has_dprime_L)) lines <- c(lines, "  vector[P_dprime_L] beta_dprime_L;  // Lure d'")
  if (isTRUE(model_data$has_sigma_L)) lines <- c(lines, "  vector[P_sigma_L] beta_sigma_L;  // Lure sigma (log scale)")
  if (isTRUE(model_data$has_lambda_L)) lines <- c(lines, "  vector[P_lambda_L] beta_lambda_L;  // Lure mixture prob (logit scale)")
  if (isTRUE(model_data$has_criterion2)) {
    if (isTRUE(model_data$varying_source_criteria)) {
      # K1 sets of source thresholds (one per item response level)
      # Each set has K2-1 thresholds parameterized as mid-anchor + log-gaps
      lines <- c(lines,
                 "  vector[K1] beta_thresh_mid_2_varying;  // Mid-anchor source threshold for each item level",
                 "  matrix[K1, K2-2] beta_log_gaps_2_varying;  // Log-gaps for each item level")
      # If new_source_criteria = "shared", add separate threshold params for new items
      if (identical(model_data$new_source_criteria, "shared")) {
        lines <- c(lines,
                   "  real beta_thresh_mid_2_new;  // Mid-anchor source threshold for new items (shared)",
                   "  vector[K2-2] beta_log_gaps_2_new;  // Log-gaps for new items (shared)")
      }
    } else {
      lines <- c(lines,
                 "  vector[P_criterion2] beta_thresh_mid_2;  // Mid-anchor threshold for dimension 2",
                 "  matrix[K2-2, P_criterion2] beta_log_gaps_2;  // Log-gaps for dimension 2")
    }
  }
  
  # RE params - include bivariate parameters and CDP parameters
  param_names <- c("dprime", "sigma", "lambda", "dprime2", "sigma2", "dprime_B", "lambda_B",
                   "discrim", "discrim_B", "sigma_B", "sigma2_B", "rho", "rho_B", "rho_N", "rec_crit", "know_crit",
                   "dprime_L", "sigma_L", "lambda_L", "lambda2", "lambda2_B")
  for (pname in param_names) {
    re_list <- model_data[[paste0(pname, "_random")]]
    if (is.null(re_list)) next
    for (group in names(re_list)) {
      re <- re_list[[group]]
      if (is.null(re$cor_id)) {
        lines <- c(lines, generate_re_params_v2(pname, group, re))
      }
    }
  }
  
  # Criterion RE
  cor_threshold <- isTRUE(model_data$cor_threshold)
  for (group in names(model_data$criterion$random)) {
    re <- model_data$criterion$random[[group]]
    if (is.null(re$cor_id)) {
      dim_var <- sprintf("D_criterion_%s", group)
      lines <- c(lines,
                 sprintf("  matrix[N_%s, %s] u_criterion_%s_raw;", group, dim_var, group),
                 sprintf("  vector<lower=0>[%s] sigma_criterion_%s;", dim_var, group))
      if (cor_threshold && isTRUE(re$correlated) && re$dim > 1) {
        lines <- c(lines,
                   sprintf("  cholesky_factor_corr[%s] L_corr_criterion_%s;", dim_var, group))
      }
    }
  }
  
  # Criterion2 RE (for bivariate_sdt)
  if (isTRUE(model_data$has_criterion2) && !is.null(model_data$criterion2$random)) {
    for (group in names(model_data$criterion2$random)) {
      re <- model_data$criterion2$random[[group]]
      if (is.null(re$cor_id)) {
        dim_var <- sprintf("D_criterion2_%s", group)
        if (re$dim == 1) {
          lines <- c(lines,
                     sprintf("  vector[N_%s] u_criterion2_%s_raw;", group, group),
                     sprintf("  real<lower=0> sigma_criterion2_%s;", group))
        } else {
          lines <- c(lines,
                     sprintf("  matrix[N_%s, %s] u_criterion2_%s_raw;", group, dim_var, group),
                     sprintf("  vector<lower=0>[%s] sigma_criterion2_%s;", dim_var, group))
          if (cor_threshold && isTRUE(re$correlated)) {
            lines <- c(lines,
                       sprintf("  cholesky_factor_corr[%s] L_corr_criterion2_%s;", dim_var, group))
          }
        }
      }
    }
  }
  
  # Cross-parameter RE
  for (cor_id in names(model_data$cross_cor)) {
    cc <- model_data$cross_cor[[cor_id]]
    dim_var <- sprintf("D_cross_%s_%s", cor_id, cc$group)
    lines <- c(lines,
               sprintf("  matrix[N_%s, %s] u_cross_%s_%s_raw;", cc$group, dim_var, cor_id, cc$group),
               sprintf("  vector<lower=0>[%s] sigma_cross_%s_%s;", dim_var, cor_id, cc$group),
               sprintf("  cholesky_factor_corr[%s] L_corr_cross_%s_%s;", dim_var, cor_id, cc$group))
  }

  # Smooth term parameters
  lines <- c(lines, generate_smooth_param_declarations(model_data))

  lines <- c(lines, "}")
  paste(lines, collapse = "\n")
}


generate_re_params_v2 <- function(param_name, group, re) {
  if (re$dim == 1) {
    c(sprintf("  vector[N_%s] u_%s_%s_raw;", group, param_name, group),
      sprintf("  real<lower=0> sigma_%s_%s;", param_name, group))
  } else {
    dim_var <- sprintf("D_%s_%s", param_name, group)
    lines <- c(sprintf("  matrix[N_%s, %s] u_%s_%s_raw;", group, dim_var, param_name, group),
               sprintf("  vector<lower=0>[%s] sigma_%s_%s;", dim_var, param_name, group))
    if (isTRUE(re$correlated)) {
      lines <- c(lines,
                 sprintf("  cholesky_factor_corr[%s] L_corr_%s_%s;", dim_var, param_name, group))
    }
    lines
  }
}


generate_transformed_parameters_block_v2 <- function(model_data, family) {
  lines <- c("transformed parameters {")
  
  # If using ordered constraint, extract beta_dprime and beta_dprime2
  if (model_data$needs_ordered_dprime) {
    lines <- c(lines,
               "  // Extract dprime and dprime2 from ordered constraint",
               "  vector[P_dprime] beta_dprime;",
               "  vector[P_dprime] beta_dprime2;",
               "  for (p in 1:P_dprime) {",
               "    beta_dprime[p] = beta_dprime_ordered[p][2];   // Higher value",
               "    beta_dprime2[p] = beta_dprime_ordered[p][1];  // Lower value",
               "  }",
               "")
  }
  
  # NOTE: fixed effects are intentionally NOT precomputed here anymore.
  # Putting vector[N] in transformed parameters creates O(N) autodiff variables
  # that dramatically slow down gradient computation.
  # Instead, X[n,] * beta is computed inside the loop, which is slower per-iteration
  # but doesn't create persistent autodiff overhead.
  
  # Random effects transformations only (include bivariate and CDP parameters)
  param_names <- c("dprime", "sigma", "lambda", "dprime2", "sigma2", "dprime_B", "lambda_B",
                   "discrim", "discrim_B", "sigma_B", "sigma2_B", "rho", "rho_B", "rho_N", "rec_crit", "know_crit",
                   "dprime_L", "sigma_L", "lambda_L", "lambda2", "lambda2_B")
  for (pname in param_names) {
    re_list <- model_data[[paste0(pname, "_random")]]
    if (is.null(re_list)) next
    for (group in names(re_list)) {
      re <- re_list[[group]]
      if (is.null(re$cor_id)) {
        lines <- c(lines, generate_re_transform_v2(pname, group, re))
      }
    }
  }
  
  cor_threshold <- isTRUE(model_data$cor_threshold)
  for (group in names(model_data$criterion$random)) {
    re <- model_data$criterion$random[[group]]
    if (is.null(re$cor_id)) {
      dim_var <- sprintf("D_criterion_%s", group)
      if (cor_threshold && isTRUE(re$correlated) && re$dim > 1) {
        lines <- c(lines, sprintf(
          "  matrix[N_%s, %s] u_criterion_%s = u_criterion_%s_raw * diag_pre_multiply(sigma_criterion_%s, L_corr_criterion_%s)';",
          group, dim_var, group, group, group, group))
      } else {
        # Diagonal: each column scaled independently (no Cholesky)
        lines <- c(lines, sprintf(
          "  matrix[N_%s, %s] u_criterion_%s;", group, dim_var, group))
        lines <- c(lines, sprintf(
          "  for (d in 1:%s) u_criterion_%s[, d] = u_criterion_%s_raw[, d] * sigma_criterion_%s[d];",
          dim_var, group, group, group))
      }
    }
  }
  
  # Criterion2 RE transforms (for bivariate_sdt)
  if (isTRUE(model_data$has_criterion2) && !is.null(model_data$criterion2$random)) {
    for (group in names(model_data$criterion2$random)) {
      re <- model_data$criterion2$random[[group]]
      if (is.null(re$cor_id)) {
        if (re$dim == 1) {
          lines <- c(lines, sprintf(
            "  vector[N_%s] u_criterion2_%s = u_criterion2_%s_raw * sigma_criterion2_%s;",
            group, group, group, group))
        } else {
          dim_var <- sprintf("D_criterion2_%s", group)
          if (cor_threshold && isTRUE(re$correlated)) {
            lines <- c(lines, sprintf(
              "  matrix[N_%s, %s] u_criterion2_%s = u_criterion2_%s_raw * diag_pre_multiply(sigma_criterion2_%s, L_corr_criterion2_%s)';",
              group, dim_var, group, group, group, group))
          } else {
            lines <- c(lines, sprintf(
              "  matrix[N_%s, %s] u_criterion2_%s;", group, dim_var, group))
            lines <- c(lines, sprintf(
              "  for (d in 1:%s) u_criterion2_%s[, d] = u_criterion2_%s_raw[, d] * sigma_criterion2_%s[d];",
              dim_var, group, group, group))
          }
        }
      }
    }
  }
  
  for (cor_id in names(model_data$cross_cor)) {
    cc <- model_data$cross_cor[[cor_id]]
    group <- cc$group
    dim_var <- sprintf("D_cross_%s_%s", cor_id, group)
    lines <- c(lines, sprintf(
      "  matrix[N_%s, %s] u_cross_%s_%s = u_cross_%s_%s_raw * diag_pre_multiply(sigma_cross_%s_%s, L_corr_cross_%s_%s)';",
      group, dim_var, cor_id, group, cor_id, group, cor_id, group, cor_id, group))
    
    col_start <- 1
    for (member in cc$members) {
      col_end <- col_start + member$dim - 1
      if (member$dim == 1) {
        lines <- c(lines, sprintf("  vector[N_%s] u_%s_%s_from_%s = u_cross_%s_%s[, %d];",
                                  group, member$param, group, cor_id, cor_id, group, col_start))
      } else {
        lines <- c(lines, sprintf("  matrix[N_%s, %d] u_%s_%s_from_%s = u_cross_%s_%s[, %d:%d];",
                                  group, member$dim, member$param, group, cor_id, cor_id, group, col_start, col_end))
      }
      col_start <- col_end + 1
    }
  }

  # Smooth transformed parameters: zs = sds * zs_raw
  lines <- c(lines, generate_smooth_transformed_params(model_data))

  lines <- c(lines, "}")
  paste(lines, collapse = "\n")
}


generate_re_transform_v2 <- function(param_name, group, re) {
  if (re$dim == 1) {
    sprintf("  vector[N_%s] u_%s_%s = u_%s_%s_raw * sigma_%s_%s;",
            group, param_name, group, param_name, group, param_name, group)
  } else if (isTRUE(re$correlated)) {
    dim_var <- sprintf("D_%s_%s", param_name, group)
    sprintf("  matrix[N_%s, %s] u_%s_%s = u_%s_%s_raw * diag_pre_multiply(sigma_%s_%s, L_corr_%s_%s)';",
            group, dim_var, param_name, group, param_name, group, param_name, group, param_name, group)
  } else {
    # Uncorrelated (||): diagonal scaling
    dim_var <- sprintf("D_%s_%s", param_name, group)
    c(sprintf("  matrix[N_%s, %s] u_%s_%s;", group, dim_var, param_name, group),
      sprintf("  for (d in 1:%s) u_%s_%s[, d] = u_%s_%s_raw[, d] * sigma_%s_%s[d];",
              dim_var, param_name, group, param_name, group, param_name, group))
  }
}


#' Build shared argument lists for reduce_sum partial_log_lik
#'
#' Inspects model_data and family to enumerate all data arrays and parameters
#' that the per-observation likelihood loop accesses. Returns declarations for
#' the function signature and names for the reduce_sum call.
#' @noRd
build_reduce_sum_args <- function(model_data, family) {
  family_name <- if (is.list(family)) family$family else family
  is_cumulative <- family_name == "cumulative"
  is_source_mixture <- family_name == "source_mixture"
  is_bivariate <- family_name %in% c("bivariate_sdt", "bivariate_dp", "bivariate_cumulative")
  is_vrdp2d <- family_name == "vrdp2d"
  is_cdp <- family_name == "cdp"

  decls <- character(0)
  nms <- character(0)

  add <- function(decl, nm) {
    decls <<- c(decls, decl)
    nms <<- c(nms, nm)
  }

  # --- Response data (data-qualified to avoid autodiff copying) ---
  add("data array[] int y", "y")

  if (is_bivariate || is_vrdp2d) {
    add("data array[] int y2", "y2")
    add("data array[] int item_type", "item_type")
  } else if (is_cdp) {
    add("data array[] int rk", "rk")
    add("data array[] real is_old", "is_old")
  } else if (is_source_mixture) {
    add("data array[] int source", "source")
  } else if (!is_cumulative) {
    add("data array[] real is_old", "is_old")
  }

  # Counts
  if (isTRUE(model_data$has_counts)) {
    add("data array[] int counts", "counts")
  }

  # --- Scalars (data) ---
  if (is_bivariate || is_vrdp2d) {
    add("data int K1", "K1")
    add("data int K2", "K2")
    add("data int mid_thresh1", "mid_thresh1")
    if (isTRUE(model_data$has_criterion2)) {
      add("data int mid_thresh2", "mid_thresh2")
    }
  } else if (is_cdp) {
    add("data int K", "K")
    add("data int J", "J")
    add("data int n_rkg", "n_rkg")
    add("data int n_new", "n_new")
    add("data array[] int old_level_map", "old_level_map")
    add("data array[] int new_level_map", "new_level_map")
    add("data int mid_thresh_cdp", "mid_thresh_cdp")
  } else {
    add("data int K", "K")
    add("data int mid_thresh", "mid_thresh")
  }

  # --- Design matrices (data) & beta vectors (parameters) ---
  has_dprime_fixed <- isTRUE(model_data$has_dprime_fixed) || is.null(model_data$has_dprime_fixed)
  if (has_dprime_fixed) {
    add("data matrix X_dprime", "X_dprime")
    if (model_data$needs_ordered_dprime) {
      add("vector beta_dprime", "beta_dprime")
      add("vector beta_dprime2", "beta_dprime2")
    } else {
      add("vector beta_dprime", "beta_dprime")
    }
  }
  if (model_data$has_dprime2 && !model_data$needs_ordered_dprime) {
    add("data matrix X_dprime2", "X_dprime2")
    add("vector beta_dprime2", "beta_dprime2")
  } else if (model_data$needs_ordered_dprime && model_data$has_dprime2) {
    # Ordered dprime: X_dprime2 exists as data, beta_dprime2 already added above
    add("data matrix X_dprime2", "X_dprime2")
  }

  # Criterion always present
  add("data matrix X_criterion", "X_criterion")

  # Threshold parameters (not data -- these are parameters)
  add("vector beta_thresh_mid", "beta_thresh_mid")
  if (model_data$K > 2) {
    add("matrix beta_log_gaps", "beta_log_gaps")
  }

  param_list <- c("sigma", "lambda", "sigma2", "dprime_B", "lambda_B",
                   "discrim", "discrim_B", "sigma_B", "sigma2_B",
                   "rho", "rho_B", "rho_N", "lambda2", "lambda2_B",
                   "rec_crit", "know_crit",
                   "dprime_L", "sigma_L", "lambda_L")
  for (pname in param_list) {
    if (isTRUE(model_data[[paste0("has_", pname)]])) {
      add(sprintf("data matrix X_%s", pname), sprintf("X_%s", pname))
      # Per-bin params are matrices, not vectors
      if (isTRUE(model_data[[paste0("per_bin_", pname)]])) {
        add(sprintf("matrix beta_%s", pname), sprintf("beta_%s", pname))
      } else {
        add(sprintf("vector beta_%s", pname), sprintf("beta_%s", pname))
      }
    }
  }

  # Criterion2 threshold parameters
  if (isTRUE(model_data$has_criterion2)) {
    if (isTRUE(model_data$varying_source_criteria)) {
      add("vector beta_thresh_mid_2_varying", "beta_thresh_mid_2_varying")
      add("matrix beta_log_gaps_2_varying", "beta_log_gaps_2_varying")
      if (identical(model_data$new_source_criteria, "shared")) {
        add("real beta_thresh_mid_2_new", "beta_thresh_mid_2_new")
        add("vector beta_log_gaps_2_new", "beta_log_gaps_2_new")
        # is_new_response (data) is read at the call site; reduce_sum functions
        # cannot see global data, so it must be threaded through explicitly.
        add("data array[] int is_new_response", "is_new_response")
      }
    } else {
      add("data matrix X_criterion2", "X_criterion2")
      add("vector beta_thresh_mid_2", "beta_thresh_mid_2")
      add("matrix beta_log_gaps_2", "beta_log_gaps_2")
    }
  }

  # --- Random effects ---
  # Grouping factor arrays
  all_groups <- get_all_groups(model_data)
  for (group in all_groups) {
    add(sprintf("data array[] int %s", group), group)
  }

  # RE vectors/matrices (the transformed u_* variables)
  all_re_params <- c("dprime", "sigma", "lambda", "dprime2", "sigma2", "dprime_B", "lambda_B",
                      "discrim", "discrim_B", "sigma_B", "sigma2_B", "rho", "rho_B", "rho_N",
                      "lambda2", "lambda2_B",
                      "rec_crit", "know_crit", "dprime_L", "sigma_L", "lambda_L")
  for (pname in all_re_params) {
    re_list <- model_data[[paste0(pname, "_random")]]
    if (is.null(re_list)) next
    for (group in names(re_list)) {
      re <- re_list[[group]]
      if (!is.null(re$cor_id)) {
        var_name <- sprintf("u_%s_%s_from_%s", pname, group, re$cor_id)
      } else {
        var_name <- sprintf("u_%s_%s", pname, group)
      }
      if (re$dim == 1) {
        add(sprintf("vector %s", var_name), var_name)
      } else {
        add(sprintf("matrix %s", var_name), var_name)
      }
    }
  }

  # Criterion RE
  for (group in names(model_data$criterion$random)) {
    re <- model_data$criterion$random[[group]]
    if (!is.null(re$cor_id)) {
      var_name <- sprintf("u_criterion_%s_from_%s", group, re$cor_id)
    } else {
      var_name <- sprintf("u_criterion_%s", group)
    }
    add(sprintf("matrix %s", var_name), var_name)
  }

  # Criterion2 RE
  if (isTRUE(model_data$has_criterion2) && !is.null(model_data$criterion2$random)) {
    for (group in names(model_data$criterion2$random)) {
      re <- model_data$criterion2$random[[group]]
      if (!is.null(re$cor_id)) {
        var_name <- sprintf("u_criterion2_%s_from_%s", group, re$cor_id)
      } else {
        var_name <- sprintf("u_criterion2_%s", group)
      }
      if (re$dim == 1) {
        add(sprintf("vector %s", var_name), var_name)
      } else {
        add(sprintf("matrix %s", var_name), var_name)
      }
    }
  }

  # Term indices
  term_indices <- get_all_term_indices(model_data)
  for (ti in term_indices) {
    add(sprintf("data array[] int %s", ti$idx_name), ti$idx_name)
  }

  # Z matrices
  z_matrices <- get_all_z_matrices(model_data)
  for (zm in z_matrices) {
    add(sprintf("data matrix %s", zm$name), zm$name)
  }

  # Smooth term data (Zs matrices) and parameters (s_ = sds * zs)
  if (!is.null(model_data$smooth_data)) {
    for (pname in names(model_data$smooth_data)) {
      for (sm in model_data$smooth_data[[pname]]) {
        for (comp in sm$components) {
          for (k in seq_along(comp$Zs_list)) {
            zs_name <- paste0("Zs_", pname, "_", comp$san_label, "_", k)
            s_name <- paste0("s_", pname, "_", comp$san_label, "_", k)
            add(sprintf("data matrix %s", zs_name), zs_name)
            add(sprintf("vector %s", s_name), s_name)
          }
        }
      }
    }
  }

  list(declarations = decls, names = nms)
}


#' Generate the partial_log_lik function for reduce_sum threading
#'
#' Wraps the existing per-observation likelihood code in a Stan function
#' that can be called by reduce_sum. Slices y directly (the response array)
#' and uses start/end to index into shared observation-level data.
#' @noRd
generate_partial_log_lik <- function(model_data, family) {
  family_name <- if (is.list(family)) family$family else family

  # Build argument list
  args <- build_reduce_sum_args(model_data, family)

  # Function header
  sig_args <- paste(args$declarations, collapse = ",\n      ")
  lines <- c(
    "  // Partial log-likelihood for reduce_sum threading",
    "  real partial_log_lik_lpmf(array[] int seq_n_slice, int start, int end,",
    paste0("      ", sig_args, ") {"),
    "    real lp = 0;",
    "    for (i in 1:size(seq_n_slice)) {",
    "      int n = seq_n_slice[i];"
  )

  # Generate the per-observation likelihood code (same as the for loop body)
  ll_lines <- generate_likelihood_code(model_data, family_name)

  # Replace "target +=" with "lp +=" and adjust indentation (add 4 more spaces)
  ll_lines <- gsub("target \\+=", "lp +=", ll_lines)
  ll_lines <- paste0("  ", ll_lines)  # extra indent for being inside the function + loop

  lines <- c(lines, ll_lines)
  lines <- c(lines, "    }",
             "    return lp;",
             "  }")

  lines
}


# Translate a JAX-style prior string to Stan syntax. Stan has no half_* density
# (the <lower=0> constraint makes a symmetric prior half) and names the Laplace
# `double_exponential`. Identity for everything else. Stan path only.
to_stan_prior <- function(prior) {
  if (length(prior) != 1 || is.na(prior) || !nzchar(prior)) return(prior)
  m <- regmatches(prior, regexec("^\\s*([A-Za-z_]+)\\s*\\((.*)\\)\\s*$", prior))[[1]]
  if (length(m) != 3) return(prior)
  name <- m[2]
  args <- trimws(strsplit(m[3], ",")[[1]])
  switch(name,
    laplace        = sprintf("double_exponential(%s)", paste(args, collapse = ", ")),
    half_normal    = if (length(args) == 1) sprintf("normal(0, %s)", args[1]) else prior,
    half_cauchy    = if (length(args) == 1) sprintf("cauchy(0, %s)", args[1]) else prior,
    half_student_t = if (length(args) == 2) sprintf("student_t(%s, 0, %s)", args[1], args[2]) else prior,
    prior
  )
}

generate_model_block_v2 <- function(model_data, family, batch_info = NULL) {
  family_name <- if (is.list(family)) family$family else family
  pl <- model_data$prior_lookup  # prior_lookup object
  pl[] <- rapply(pl, to_stan_prior, classes = "character", how = "replace")

  lines <- c("model {")
  
  # Helper to generate fixed effect priors (handles coef-specific priors)
  generate_fixed_priors <- function(dpar, coef_names, beta_name) {
    # Check if there are any coefficient-specific priors for this dpar
    coef_priors <- pl$b_coef[grepl(paste0("^", dpar, "_"), names(pl$b_coef))]
    
    if (length(coef_priors) == 0) {
      # No coef-specific priors - use vectorized prior
      prior <- get_fixed_prior(pl, dpar)
      return(sprintf("  %s ~ %s;", beta_name, prior))
    }
    
    # Has coef-specific priors - need element-wise
    prior_lines <- character(0)
    default_prior <- get_fixed_prior(pl, dpar)
    
    for (i in seq_along(coef_names)) {
      coef <- coef_names[i]
      key <- paste0(dpar, "_", coef)
      
      if (!is.null(pl$b_coef[[key]])) {
        prior <- pl$b_coef[[key]]
      } else {
        prior <- default_prior
      }
      
      prior_lines <- c(prior_lines, 
                       sprintf("  %s[%d] ~ %s;  // %s", beta_name, i, prior, coef))
    }
    
    prior_lines
  }
  
  # Fixed effects priors
  has_dprime_fixed <- isTRUE(model_data$has_dprime_fixed) || is.null(model_data$has_dprime_fixed)
  
  if (model_data$needs_ordered_dprime) {
    dprime_prior <- get_fixed_prior(pl, "dprime")
    dprime2_prior <- get_fixed_prior(pl, "dprime2")
    lines <- c(lines, 
               "  // Priors on ordered dprime parameters",
               "  for (p in 1:P_dprime) {",
               sprintf("    beta_dprime_ordered[p][2] ~ %s;  // dprime (higher)", dprime_prior),
               sprintf("    beta_dprime_ordered[p][1] ~ %s;  // dprime2 (lower)", dprime2_prior),
               "  }")
  } else if (has_dprime_fixed) {
    dprime_coefs <- model_data$dprime_fixed$coef_names
    lines <- c(lines, generate_fixed_priors("dprime", dprime_coefs, "beta_dprime"))
    
    if (model_data$has_dprime2) {
      dprime2_coefs <- model_data$dprime2_fixed$coef_names
      lines <- c(lines, generate_fixed_priors("dprime2", dprime2_coefs, "beta_dprime2"))
    }
  }
  # If no dprime fixed effects (cumulative with only random effects), skip beta_dprime priors
  
  gaps_prior_key <- if (identical(model_data$gap_link, "softplus")) "raw_gaps" else "log_gaps"
  thresh_mid_prior <- get_fixed_prior(pl, "thresh_mid")
  lines <- c(lines, sprintf("  beta_thresh_mid ~ %s;", thresh_mid_prior))
  if (model_data$K > 2) {
    # Per-gap fixed-effect priors. Resolution: user-specified
    # `coef = "[k]"` overrides the dpar-level default per row of beta_log_gaps.
    n_gaps <- model_data$K - 2L
    base_prior <- get_fixed_prior(pl, gaps_prior_key)
    per_gap <- vapply(seq_len(n_gaps), function(k)
      get_fixed_prior(pl, gaps_prior_key, coef = sprintf("[%d]", k)),
      character(1))
    if (length(unique(per_gap)) == 1L) {
      lines <- c(lines, sprintf("  to_vector(beta_log_gaps) ~ %s;", base_prior))
    } else {
      for (k in seq_len(n_gaps)) {
        lines <- c(lines, sprintf("  beta_log_gaps[%d, ] ~ %s;", k, per_gap[k]))
      }
    }
  }
  
  if (model_data$has_sigma) {
    sigma_coefs <- model_data$sigma_fixed$coef_names
    lines <- c(lines, generate_fixed_priors("sigma", sigma_coefs, "beta_sigma"))
  }
  if (model_data$has_lambda) {
    lambda_coefs <- model_data$lambda_fixed$coef_names
    lines <- c(lines, generate_fixed_priors("lambda", lambda_coefs, "beta_lambda"))
  }
  if (model_data$has_sigma2) {
    sigma2_coefs <- model_data$sigma2_fixed$coef_names
    lines <- c(lines, generate_fixed_priors("sigma2", sigma2_coefs, "beta_sigma2"))
  }
  if (isTRUE(model_data$has_dprime_B)) {
    dprime_B_coefs <- model_data$dprime_B_fixed$coef_names
    lines <- c(lines, generate_fixed_priors("dprime_B", dprime_B_coefs, "beta_dprime_B"))
  }
  if (isTRUE(model_data$has_lambda_B)) {
    lambda_B_coefs <- model_data$lambda_B_fixed$coef_names
    lines <- c(lines, generate_fixed_priors("lambda_B", lambda_B_coefs, "beta_lambda_B"))
  }
  
  # Bivariate SDT fixed effects priors
  has_discrim_fixed_model <- isTRUE(model_data$has_discrim_fixed) || (isTRUE(model_data$has_discrim) && is.null(model_data$has_discrim_fixed))
  if (isTRUE(model_data$has_discrim) && has_discrim_fixed_model) {
    discrim_coefs <- model_data$discrim_fixed$coef_names
    lines <- c(lines, generate_fixed_priors("discrim", discrim_coefs, "beta_discrim"))
  }
  if (isTRUE(model_data$has_discrim_B)) {
    discrim_B_coefs <- model_data$discrim_B_fixed$coef_names
    lines <- c(lines, generate_fixed_priors("discrim_B", discrim_B_coefs, "beta_discrim_B"))
  }
  if (isTRUE(model_data$has_sigma_B)) {
    sigma_B_coefs <- model_data$sigma_B_fixed$coef_names
    lines <- c(lines, generate_fixed_priors("sigma_B", sigma_B_coefs, "beta_sigma_B"))
  }
  if (isTRUE(model_data$has_sigma2_B)) {
    sigma2_B_coefs <- model_data$sigma2_B_fixed$coef_names
    lines <- c(lines, generate_fixed_priors("sigma2_B", sigma2_B_coefs, "beta_sigma2_B"))
  }
  if (isTRUE(model_data$has_rho)) {
    rho_coefs <- model_data$rho_fixed$coef_names
    lines <- c(lines, generate_fixed_priors("rho", rho_coefs, "beta_rho"))
  }
  if (isTRUE(model_data$has_rho_B)) {
    rho_B_coefs <- model_data$rho_B_fixed$coef_names
    lines <- c(lines, generate_fixed_priors("rho_B", rho_B_coefs, "beta_rho_B"))
  }
  if (isTRUE(model_data$has_rho_N)) {
    rho_N_coefs <- model_data$rho_N_fixed$coef_names
    lines <- c(lines, generate_fixed_priors("rho_N", rho_N_coefs, "beta_rho_N"))
  }
  if (isTRUE(model_data$has_lambda2)) {
    lambda2_coefs <- model_data$lambda2_fixed$coef_names
    lines <- c(lines, generate_fixed_priors("lambda2", lambda2_coefs, "beta_lambda2"))
  }
  if (isTRUE(model_data$has_lambda2_B)) {
    lambda2_B_coefs <- model_data$lambda2_B_fixed$coef_names
    lines <- c(lines, generate_fixed_priors("lambda2_B", lambda2_B_coefs, "beta_lambda2_B"))
  }
  if (isTRUE(model_data$has_rec_crit)) {
    rec_crit_coefs <- model_data$rec_crit_fixed$coef_names
    lines <- c(lines, generate_fixed_priors("rec_crit", rec_crit_coefs, "beta_rec_crit"))
  }
  if (isTRUE(model_data$has_know_crit)) {
    know_crit_coefs <- model_data$know_crit_fixed$coef_names
    lines <- c(lines, generate_fixed_priors("know_crit", know_crit_coefs, "beta_know_crit"))
  }
  if (isTRUE(model_data$has_dprime_L)) {
    dprime_L_coefs <- model_data$dprime_L_fixed$coef_names
    lines <- c(lines, generate_fixed_priors("dprime_L", dprime_L_coefs, "beta_dprime_L"))
  }
  if (isTRUE(model_data$has_sigma_L)) {
    sigma_L_coefs <- model_data$sigma_L_fixed$coef_names
    lines <- c(lines, generate_fixed_priors("sigma_L", sigma_L_coefs, "beta_sigma_L"))
  }
  if (isTRUE(model_data$has_lambda_L)) {
    lambda_L_coefs <- model_data$lambda_L_fixed$coef_names
    lines <- c(lines, generate_fixed_priors("lambda_L", lambda_L_coefs, "beta_lambda_L"))
  }
  if (isTRUE(model_data$has_criterion2)) {
    thresh_mid_2_prior <- get_fixed_prior(pl, "thresh_mid2")
    gaps2_key <- if (identical(model_data$gap_link, "softplus")) "raw_gaps" else "log_gaps2"
    log_gaps_2_prior <- get_fixed_prior(pl, gaps2_key)
    if (isTRUE(model_data$varying_source_criteria)) {
      lines <- c(lines,
                 sprintf("  beta_thresh_mid_2_varying ~ %s;", thresh_mid_2_prior),
                 sprintf("  to_vector(beta_log_gaps_2_varying) ~ %s;", log_gaps_2_prior))
      if (identical(model_data$new_source_criteria, "shared")) {
        lines <- c(lines,
                   sprintf("  beta_thresh_mid_2_new ~ %s;", thresh_mid_2_prior),
                   sprintf("  beta_log_gaps_2_new ~ %s;", log_gaps_2_prior))
      }
    } else {
      lines <- c(lines, sprintf("  beta_thresh_mid_2 ~ %s;", thresh_mid_2_prior))
      n_gaps2 <- model_data$K2 - 2L
      per_gap2 <- vapply(seq_len(n_gaps2), function(k)
        get_fixed_prior(pl, gaps2_key, coef = sprintf("[%d]", k)),
        character(1))
      if (length(unique(per_gap2)) == 1L) {
        lines <- c(lines, sprintf("  to_vector(beta_log_gaps_2) ~ %s;", log_gaps_2_prior))
      } else {
        for (k in seq_len(n_gaps2)) {
          lines <- c(lines, sprintf("  beta_log_gaps_2[%d, ] ~ %s;", k, per_gap2[k]))
        }
      }
    }
  }
  
  # RE priors - using prior_lookup (include bivariate and CDP parameters)
  param_names <- c("dprime", "sigma", "lambda", "dprime2", "sigma2", "dprime_B", "lambda_B",
                   "discrim", "discrim_B", "sigma_B", "sigma2_B", "rho", "rho_B", "rho_N", "rec_crit", "know_crit",
                   "dprime_L", "sigma_L", "lambda_L", "lambda2", "lambda2_B")
  for (pname in param_names) {
    re_list <- model_data[[paste0(pname, "_random")]]
    if (is.null(re_list)) next
    
    for (group in names(re_list)) {
      re <- re_list[[group]]
      if (is.null(re$cor_id)) {
        cor_prior <- get_cor_prior(pl, pname, group)
        
        # Check whether coefficient-level SD priors are needed
        level_names <- re$level_names
        if (is.null(level_names) || length(level_names) == 0) {
          level_names <- "(Intercept)"
        }
        
        # Get prior for each coefficient
        coef_priors <- sapply(level_names, function(coef) get_sd_prior(pl, pname, group, coef))
        
        if (length(unique(coef_priors)) == 1) {
          # All same prior - use vector syntax
          lines <- c(lines, sprintf("  sigma_%s_%s ~ %s;", pname, group, coef_priors[1]))
        } else {
          # Different priors - apply element-wise
          for (i in seq_along(coef_priors)) {
            lines <- c(lines, sprintf("  sigma_%s_%s[%d] ~ %s;", pname, group, i, coef_priors[i]))
          }
        }
        
        if (re$dim == 1) {
          lines <- c(lines, sprintf("  u_%s_%s_raw ~ std_normal();", pname, group))
        } else {
          lines <- c(lines, 
                     sprintf("  to_vector(u_%s_%s_raw) ~ std_normal();", pname, group))
          if (isTRUE(re$correlated)) {
            lines <- c(lines,
                       sprintf("  L_corr_%s_%s ~ %s;", pname, group, cor_prior))
          }
        }
      }
    }
  }
  
  # Criterion RE priors
  cor_threshold <- isTRUE(model_data$cor_threshold)
  for (group in names(model_data$criterion$random)) {
    re <- model_data$criterion$random[[group]]
    if (is.null(re$cor_id)) {
      cor_prior <- get_cor_prior(pl, "criterion", group)
      
      # Check whether coefficient-level SD priors are needed
      level_names <- re$level_names
      if (is.null(level_names)) {
        n_thresh <- re$dim
        level_names <- paste0("thresh", 1:n_thresh)
      }
      
      coef_priors <- sapply(level_names, function(coef) get_sd_prior(pl, "criterion", group, coef))
      
      if (length(unique(coef_priors)) == 1) {
        lines <- c(lines, sprintf("  sigma_criterion_%s ~ %s;", group, coef_priors[1]))
      } else {
        for (i in seq_along(coef_priors)) {
          lines <- c(lines, sprintf("  sigma_criterion_%s[%d] ~ %s;", group, i, coef_priors[i]))
        }
      }
      
      lines <- c(lines,
                 sprintf("  to_vector(u_criterion_%s_raw) ~ std_normal();", group))
      if (cor_threshold && isTRUE(re$correlated) && re$dim > 1) {
        lines <- c(lines, sprintf("  L_corr_criterion_%s ~ %s;", group, cor_prior))
      }
    }
  }
  
  # Criterion2 RE priors (for bivariate_sdt)
  if (isTRUE(model_data$has_criterion2) && !is.null(model_data$criterion2$random)) {
    for (group in names(model_data$criterion2$random)) {
      re <- model_data$criterion2$random[[group]]
      if (is.null(re$cor_id)) {
        sd_prior <- get_sd_prior(pl, "criterion", group)  # Use criterion priors
        cor_prior <- get_cor_prior(pl, "criterion", group)
        
        lines <- c(lines,
                   sprintf("  sigma_criterion2_%s ~ %s;", group, sd_prior))
        if (re$dim == 1) {
          lines <- c(lines, sprintf("  u_criterion2_%s_raw ~ std_normal();", group))
        } else {
          lines <- c(lines, sprintf("  to_vector(u_criterion2_%s_raw) ~ std_normal();", group))
          if (cor_threshold && isTRUE(re$correlated)) {
            lines <- c(lines, sprintf("  L_corr_criterion2_%s ~ %s;", group, cor_prior))
          }
        }
      }
    }
  }
  
  # Cross RE priors - need element-wise SD priors for each parameter in the block
  for (cor_id in names(model_data$cross_cor)) {
    cc <- model_data$cross_cor[[cor_id]]
    cor_prior <- get_cor_prior(pl, "cross", cc$group)
    
    # Generate element-wise SD priors based on which parameters are in the block
    sd_lines <- character(0)
    idx <- 1
    for (member in cc$members) {
      param <- member$param
      dim <- member$dim
      sd_prior <- get_sd_prior(pl, param, cc$group)
      
      if (dim == 1) {
        sd_lines <- c(sd_lines, 
                      sprintf("  sigma_cross_%s_%s[%d] ~ %s;  // %s", 
                              cor_id, cc$group, idx, sd_prior, param))
        idx <- idx + 1
      } else {
        for (j in 1:dim) {
          sd_lines <- c(sd_lines, 
                        sprintf("  sigma_cross_%s_%s[%d] ~ %s;  // %s[%d]", 
                                cor_id, cc$group, idx, sd_prior, param, j))
          idx <- idx + 1
        }
      }
    }
    
    lines <- c(lines, sd_lines,
               sprintf("  to_vector(u_cross_%s_%s_raw) ~ std_normal();", cor_id, cc$group),
               sprintf("  L_corr_cross_%s_%s ~ %s;", cor_id, cc$group, cor_prior))
  }
  
  # Smooth term priors
  lines <- c(lines, generate_smooth_priors(model_data))

  # Likelihood
  if (!is.null(batch_info) && isTRUE(model_data$threads)) {
    # Batch + threading: reduce_sum calls batch function on observation slices
    lines <- c(lines, "", batch_info$stan_call_threaded)
  } else if (!is.null(batch_info)) {
    # Batch without threading: single function call
    lines <- c(lines, "", "  // Batch C++ likelihood (single autodiff node)",
               batch_info$stan_call)
  } else if (isTRUE(model_data$threads)) {
    # Threading: emit reduce_sum call (y is sliced, rest are shared)
    args <- build_reduce_sum_args(model_data, family)
    arg_names <- paste(args$names, collapse = ",\n      ")
    lines <- c(lines, "",
               "  // Threaded likelihood via reduce_sum",
               "  target += reduce_sum(partial_log_lik_lpmf, seq_n, grainsize,",
               paste0("      ", arg_names, ");"))
  } else {
    lines <- c(lines, "", "  for (n in 1:N) {")
    lines <- c(lines, generate_likelihood_code(model_data, family_name))
    lines <- c(lines, "  }")
  }
  lines <- c(lines, "}")

  paste(lines, collapse = "\n")
}




generate_likelihood_code <- function(model_data, family_name) {
  lines <- character(0)

  # Check for any encoding variables
  has_encoding_params <- isTRUE(model_data$dprime_fixed$has_encoding) ||
    isTRUE(model_data$lambda_fixed$has_encoding) ||
    isTRUE(model_data$sigma_fixed$has_encoding)

  # Bivariate cumulative has a simple structure (no item_type, no sigma)
  if (family_name == "bivariate_cumulative") {
    lines <- c(lines, generate_bivariate_cumulative_likelihood_code(model_data))
    return(lines)
  }

  # Bivariate SDT has a completely different structure
  if (family_name == "bivariate_sdt") {
    lines <- c(lines, generate_bivariate_likelihood_code(model_data))
    return(lines)
  }

  # Bivariate DP has its own structure (extends bivariate_sdt with lambda/lambda2)
  if (family_name == "bivariate_dp") {
    lines <- c(lines, generate_bivariate_dp_likelihood_code(model_data))
    return(lines)
  }

  # VRDP2D has its own structure
  if (family_name == "vrdp2d") {
    lines <- c(lines, generate_vrdp2d_likelihood_code(model_data))
    return(lines)
  }
  
  # CDP has its own structure
  if (family_name == "cdp") {
    lines <- c(lines, generate_cdp_likelihood_code(model_data))
    return(lines)
  }

  # For models with encoding variables or lure mixture, use the optimized if/else structure
  if ((has_encoding_params || isTRUE(model_data$has_lure_mixture)) && family_name %in% c("mixture", "dpsdt")) {
    lines <- c(lines, generate_likelihood_with_encoding_branch(model_data, family_name))
  } else {
    # Standard structure for models without encoding
    lines <- c(lines, generate_param_lp("dprime", model_data$dprime_fixed, model_data$dprime_random, model_data$smooth_data[["dprime"]]))
    
    if (model_data$has_sigma) {
      lines <- c(lines, generate_param_lp("sigma", model_data$sigma_fixed, model_data$sigma_random, model_data$smooth_data[["sigma"]]))
      lines <- c(lines, "    real sigma_val = exp(sigma_n);")
    }
    
    if (model_data$has_lambda) {
      lines <- c(lines, generate_param_lp("lambda", model_data$lambda_fixed, model_data$lambda_random, model_data$smooth_data[["lambda"]]))
      lines <- c(lines, "    real lambda_val = inv_logit(lambda_n);")
    }
    
    if (model_data$has_dprime2 || model_data$needs_ordered_dprime) {
      lines <- c(lines, generate_param_lp("dprime2", model_data$dprime2_fixed, model_data$dprime2_random, model_data$smooth_data[["dprime2"]]))
      lines <- c(lines, "    real dprime2_val = dprime2_n;")
    } else if (family_name == "mixture") {
      if (model_data$has_sigma || model_data$has_sigma2) {
        lines <- c(lines, "    real dprime2_val = 0;")
      }
    }
    
    if (model_data$has_sigma2) {
      lines <- c(lines, generate_param_lp("sigma2", model_data$sigma2_fixed, model_data$sigma2_random, model_data$smooth_data[["sigma2"]]))
      lines <- c(lines, "    real sigma2_val = exp(sigma2_n);")
    }
    
    # Source mixture specific parameters
    if (isTRUE(model_data$has_dprime_B)) {
      lines <- c(lines, generate_param_lp("dprime_B", model_data$dprime_B_fixed, model_data$dprime_B_random, model_data$smooth_data[["dprime_B"]]))
      lines <- c(lines, "    real dprime_B_val = dprime_B_n;")
    }
    
    if (isTRUE(model_data$has_lambda_B)) {
      lines <- c(lines, generate_param_lp("lambda_B", model_data$lambda_B_fixed, model_data$lambda_B_random, model_data$smooth_data[["lambda_B"]]))
      lines <- c(lines, "    real lambda_B_val = inv_logit(lambda_B_n);")
    }
    
    lines <- c(lines, generate_thresh_code(model_data))
    lines <- c(lines, generate_likelihood_call(model_data, family_name))
  }
  
  lines
}


#' Generate likelihood code for bivariate cumulative ordinal model
#' @noRd
generate_bivariate_cumulative_likelihood_code <- function(model_data) {
  lines <- character(0)

  # mu1 (dprime) linear predictor
  lines <- c(lines, generate_param_lp_bivariate("dprime", model_data$dprime_fixed, model_data$dprime_random))

  # mu2 (discrim) linear predictor
  lines <- c(lines, generate_param_lp_bivariate("discrim", model_data$discrim_fixed, model_data$discrim_random))

  # rho linear predictor + Fisher z transform
  lines <- c(lines, generate_param_lp_bivariate("rho", model_data$rho_fixed, model_data$rho_random))
  lines <- c(lines, "    real rho_val = tanh(rho_n);  // Fisher z to correlation")

  # Threshold vectors for both dimensions
  lines <- c(lines, generate_thresh_code_bivariate(model_data, 1))  # Dim 1 thresholds
  lines <- c(lines, generate_thresh_code_bivariate(model_data, 2))  # Dim 2 thresholds

  # Likelihood call
  has_counts <- isTRUE(model_data$has_counts)
  ll_expr <- "bivariate_cumulative_lpmf_custom(y[n], y2[n], K1, K2, dprime_n, discrim_n, thresh1, thresh2, rho_val)"

  if (has_counts) {
    lines <- c(lines, "", sprintf("    target += counts[n] * %s;", ll_expr))
  } else {
    lines <- c(lines, "", sprintf("    target += %s;", ll_expr))
  }

  lines
}


#' Generate likelihood code for bivariate SDT
#' @noRd
generate_bivariate_likelihood_code <- function(model_data) {
  lines <- character(0)
  
  # Detection dimension (dprime)
  lines <- c(lines, generate_param_lp_bivariate("dprime", model_data$dprime_fixed, model_data$dprime_random))
  
  # dprime_B (for Source B detection)
  if (isTRUE(model_data$has_dprime_B)) {
    lines <- c(lines, generate_param_lp_bivariate("dprime_B", model_data$dprime_B_fixed, model_data$dprime_B_random))
    lines <- c(lines, "    real dprime_B_val = dprime_B_n;")
  } else {
    lines <- c(lines, "    real dprime_B_val = dprime_n;  // constrained equal to dprime")
  }
  
  # Discrimination dimension (discrim)
  lines <- c(lines, generate_param_lp_bivariate("discrim", model_data$discrim_fixed, model_data$discrim_random))
  
  # discrim_B (for Source B discrimination)
  # Symmetric default differs by bounded mode:
  #   bounded:   discrim_B_val = discrim_n (same positive magnitude; lpmf
  #              negates A and uses B directly to place A on negative axis,
  #              B on positive)
  #   unbounded: discrim_B_val = -discrim_n (opposite sign; lpmf uses both
  #              directly so symmetric mirror requires opposite signs)
  if (isTRUE(model_data$has_discrim_B)) {
    lines <- c(lines, generate_param_lp_bivariate("discrim_B", model_data$discrim_B_fixed, model_data$discrim_B_random))
    lines <- c(lines, "    real discrim_B_val = discrim_B_n;")
  } else if (isTRUE(model_data$bounded)) {
    lines <- c(lines, "    real discrim_B_val = discrim_n;  // bounded symmetric: same magnitude")
  } else {
    lines <- c(lines, "    real discrim_B_val = -discrim_n;  // unbounded symmetric: mirror across origin")
  }
  
  # Sigma parameters (detection dimension)
  if (isTRUE(model_data$has_sigma)) {
    lines <- c(lines, generate_param_lp_bivariate("sigma", model_data$sigma_fixed, model_data$sigma_random))
    lines <- c(lines, "    real sigma1_A_val = exp(sigma_n);")
  } else {
    lines <- c(lines, "    real sigma1_A_val = 1;  // fixed")
  }
  
  if (isTRUE(model_data$has_sigma_B)) {
    lines <- c(lines, generate_param_lp_bivariate("sigma_B", model_data$sigma_B_fixed, model_data$sigma_B_random))
    lines <- c(lines, "    real sigma1_B_val = exp(sigma_B_n);")
  } else if (isTRUE(model_data$has_sigma)) {
    lines <- c(lines, "    real sigma1_B_val = sigma1_A_val;  // constrained equal to sigma")
  } else {
    lines <- c(lines, "    real sigma1_B_val = 1;  // fixed")
  }
  
  # Sigma2 parameters (discrimination dimension)
  if (isTRUE(model_data$has_sigma2)) {
    lines <- c(lines, generate_param_lp_bivariate("sigma2", model_data$sigma2_fixed, model_data$sigma2_random))
    lines <- c(lines, "    real sigma2_A_val = exp(sigma2_n);")
  } else {
    lines <- c(lines, "    real sigma2_A_val = 1;  // fixed")
  }
  
  if (isTRUE(model_data$has_sigma2_B)) {
    lines <- c(lines, generate_param_lp_bivariate("sigma2_B", model_data$sigma2_B_fixed, model_data$sigma2_B_random))
    lines <- c(lines, "    real sigma2_B_val = exp(sigma2_B_n);")
  } else if (isTRUE(model_data$has_sigma2)) {
    lines <- c(lines, "    real sigma2_B_val = sigma2_A_val;  // constrained equal to sigma2")
  } else {
    lines <- c(lines, "    real sigma2_B_val = 1;  // fixed")
  }
  
  # Rho parameters (correlations on Fisher z scale)
  lines <- c(lines, generate_param_lp_bivariate("rho", model_data$rho_fixed, model_data$rho_random))
  lines <- c(lines, "    real rho_A_val = tanh(rho_n);  // Fisher z to correlation")
  
  # rho_B symmetric default differs by bounded:
  #   bounded:   rho_B_val = rho_A_val (same positive magnitude, lpmf places signs)
  #   unbounded: rho_B_val = -rho_A_val (mirror correlation across origin)
  if (isTRUE(model_data$has_rho_B)) {
    lines <- c(lines, generate_param_lp_bivariate("rho_B", model_data$rho_B_fixed, model_data$rho_B_random))
    lines <- c(lines, "    real rho_B_val = tanh(rho_B_n);")
  } else if (isTRUE(model_data$bounded)) {
    lines <- c(lines, "    real rho_B_val = rho_A_val;  // bounded symmetric: same magnitude")
  } else {
    lines <- c(lines, "    real rho_B_val = -rho_A_val;  // unbounded symmetric: opposite sign")
  }
  
  if (isTRUE(model_data$has_rho_N)) {
    lines <- c(lines, generate_param_lp_bivariate("rho_N", model_data$rho_N_fixed, model_data$rho_N_random))
    lines <- c(lines, "    real rho_N_val = tanh(rho_N_n);")
  } else {
    lines <- c(lines, "    real rho_N_val = 0;  // fixed at 0 for new items")
  }
  
  # Threshold vectors for both dimensions
  lines <- c(lines, generate_thresh_code_bivariate(model_data, 1))  # Detection thresholds
  
  varying_source_criteria <- isTRUE(model_data$varying_source_criteria)

  if (varying_source_criteria) {
    # K1 sets of source thresholds (only for "old" levels if guessing, all levels otherwise)
    lines <- c(lines, generate_thresh_code_vrdp2d_varying(model_data))
  } else {
    lines <- c(lines, generate_thresh_code_bivariate(model_data, 2))  # Discrimination thresholds
  }

  new_shared <- identical(model_data$new_source_criteria, "shared")

  # Build shared new-item source thresholds if needed (mid-anchor)
  if (new_shared) {
    lines <- c(lines,
      "    // Shared source thresholds for new items (mid-anchor)",
      "    vector[K2-1] thresh2_new;",
      "    thresh2_new[mid_thresh2] = beta_thresh_mid_2_new;",
      "    for (k2 in (mid_thresh2+1):(K2-1)) {",
      "      thresh2_new[k2] = thresh2_new[k2-1] + exp(beta_log_gaps_2_new[k2 - mid_thresh2]);",
      "    }",
      "    for (k2_down in 1:(mid_thresh2-1)) {",
      "      int k2 = mid_thresh2 - k2_down;",
      "      thresh2_new[k2] = thresh2_new[k2+1] - exp(beta_log_gaps_2_new[K2 - 1 - mid_thresh2 + k2_down]);",
      "    }")
  }

  # Likelihood call
  has_counts <- isTRUE(model_data$has_counts)
  bounded <- isTRUE(model_data$bounded)

  # Bounded models require positive dprime, discrim and rho in (0,1)
  if (bounded) {
    # Apply exp() to dprime for positivity: mu_I > 0
    dprime_idx <- grep("real dprime_n = ", lines, fixed = TRUE)
    if (length(dprime_idx) > 0) {
      lines[dprime_idx] <- gsub(
        "real dprime_n = (.+);",
        "real dprime_n = exp(\\1);  // log link: mu_I > 0 (bounded)",
        lines[dprime_idx]
      )
    }
    dprime_B_idx <- grep("real dprime_B_val = dprime_B_n;", lines, fixed = TRUE)
    if (length(dprime_B_idx) > 0) {
      lines[dprime_B_idx] <- "    real dprime_B_val = exp(dprime_B_n);  // log link: mu_I_B > 0 (bounded)"
    }

    # Apply exp() to discrim for positivity: mu_S > 0
    discrim_idx <- grep("real discrim_n = ", lines, fixed = TRUE)
    if (length(discrim_idx) > 0) {
      lines[discrim_idx] <- gsub(
        "real discrim_n = (.+);",
        "real discrim_n = exp(\\1);  // log link: mu_S > 0 (bounded)",
        lines[discrim_idx]
      )
    }
    discrim_B_idx <- grep("real discrim_B_val = discrim_B_n;", lines, fixed = TRUE)
    if (length(discrim_B_idx) > 0) {
      lines[discrim_B_idx] <- "    real discrim_B_val = exp(discrim_B_n);  // log link: mu_S_B > 0 (bounded)"
    }

    # Apply inv_logit() to rho for (0,1) constraint
    rho_idx <- grep("real rho_A_val = tanh(rho_n)", lines, fixed = TRUE)
    if (length(rho_idx) > 0) {
      lines[rho_idx] <- "    real rho_A_val = inv_logit(rho_n);  // logistic link: rho_A in (0,1) (bounded)"
    }
    rho_B_idx <- grep("real rho_B_val = tanh(rho_B_n);", lines, fixed = TRUE)
    if (length(rho_B_idx) > 0) {
      lines[rho_B_idx] <- "    real rho_B_val = inv_logit(rho_B_n);  // logistic link: rho_B in (0,1) (bounded)"
    }
  }

  # Select lpmf function name based on bounded x varying x new_shared
  if (new_shared && bounded) {
    lpmf_name <- "bivariate_sdt_varying_bounded_new_shared_lpmf_custom"
    thresh_arg <- "thresh1, thresh2_varying, thresh2_new, is_new_response"
  } else if (new_shared) {
    lpmf_name <- "bivariate_sdt_varying_new_shared_lpmf_custom"
    thresh_arg <- "thresh1, thresh2_varying, thresh2_new, is_new_response"
  } else if (varying_source_criteria && bounded) {
    lpmf_name <- "bivariate_sdt_varying_bounded_lpmf_custom"
    thresh_arg <- "thresh1, thresh2_varying"
  } else if (varying_source_criteria) {
    lpmf_name <- "bivariate_sdt_varying_lpmf_custom"
    thresh_arg <- "thresh1, thresh2_varying"
  } else if (bounded) {
    lpmf_name <- "bivariate_sdt_bounded_lpmf_custom"
    thresh_arg <- "thresh1, thresh2"
  } else {
    lpmf_name <- "bivariate_sdt_lpmf_custom"
    thresh_arg <- "thresh1, thresh2"
  }

  ll_expr <- paste0(
    lpmf_name, "(y[n], y2[n], item_type[n], K1, K2,\n",
    "                                  dprime_n, dprime_B_val, discrim_n, discrim_B_val,\n",
    "                                  sigma1_A_val, sigma1_B_val, sigma2_A_val, sigma2_B_val,\n",
    "                                  rho_A_val, rho_B_val, rho_N_val,\n",
    "                                  ", thresh_arg, ")"
  )
  
  if (has_counts) {
    lines <- c(lines, "", sprintf("    target += counts[n] * %s;", ll_expr))
  } else {
    lines <- c(lines, "", sprintf("    target += %s;", ll_expr))
  }
  
  lines
}


#' Generate likelihood code for bivariate DP model
#' @noRd
generate_bivariate_dp_likelihood_code <- function(model_data) {
  # Reuse the bivariate_sdt likelihood code for all shared params
  lines <- generate_bivariate_likelihood_code(model_data)

  # Remove the last lines (the target += ... statement and blank line before it)
  # Replace the lpmf call with the bivariate_dp version
  # Find and remove the target line
  target_idx <- grep("target \\+=", lines)
  if (length(target_idx) > 0) {
    # Remove from the blank line before target to the target line
    remove_start <- max(1, target_idx[length(target_idx)] - 1)
    lines <- lines[1:(remove_start - 1)]
  }

  # NOTE: link transforms for dprime/discrim/rho are handled inside
  # `generate_bivariate_likelihood_code` based on `model_data$bounded`.
  # Bounded BDP gets exp/inv_logit (positivity / [0,1] constraints); unbounded
  # BDP keeps the identity / Fisher-z links from the shared emission. No
  # additional transforms are applied here.

  # Add lambda and lambda2 parameter extraction
  lines <- c(lines, generate_param_lp_bivariate("lambda", model_data$lambda_fixed, model_data$lambda_random))
  lines <- c(lines, "    real R_I = inv_logit(lambda_n);")

  # lambda_B (R_I for source B): only emit when model_data$has_lambda_B is TRUE
  # (set in broc_model.R for bivariate_dp). source_mixture also uses has_lambda_B
  # but has its own likelihood path; only bivariate_dp is handled here.
  if (isTRUE(model_data$has_lambda_B)) {
    lines <- c(lines, generate_param_lp_bivariate("lambda_B", model_data$lambda_B_fixed, model_data$lambda_B_random))
    lines <- c(lines, "    real R_I_B = inv_logit(lambda_B_n);")
  } else {
    lines <- c(lines, "    real R_I_B = R_I;  // constrained equal to lambda")
  }

  lines <- c(lines, generate_param_lp_bivariate("lambda2", model_data$lambda2_fixed, model_data$lambda2_random))
  lines <- c(lines, "    real R_S = inv_logit(lambda2_n);")

  if (isTRUE(model_data$has_lambda2_B)) {
    lines <- c(lines, generate_param_lp_bivariate("lambda2_B", model_data$lambda2_B_fixed, model_data$lambda2_B_random))
    lines <- c(lines, "    real R_S_B = inv_logit(lambda2_B_n);")
  } else {
    lines <- c(lines, "    real R_S_B = R_S;  // constrained equal to lambda2")
  }

  # Select BDP lpmf name
  has_counts <- isTRUE(model_data$has_counts)
  varying <- isTRUE(model_data$varying_source_criteria)
  new_shared <- identical(model_data$new_source_criteria, "shared")

  if (new_shared) {
    lpmf_name <- "bivariate_dp_varying_new_shared_lpmf_custom"
    thresh_arg <- "thresh1, thresh2_varying, thresh2_new, is_new_response"
  } else if (varying) {
    lpmf_name <- "bivariate_dp_varying_lpmf_custom"
    thresh_arg <- "thresh1, thresh2_varying"
  } else {
    lpmf_name <- "bivariate_dp_lpmf_custom"
    thresh_arg <- "thresh1, thresh2"
  }

  ll_expr <- paste0(
    lpmf_name, "(y[n], y2[n], item_type[n], K1, K2,\n",
    "                                  dprime_n, dprime_B_val, discrim_n, discrim_B_val,\n",
    "                                  sigma1_A_val, sigma1_B_val, sigma2_A_val, sigma2_B_val,\n",
    "                                  rho_A_val, rho_B_val, rho_N_val,\n",
    "                                  R_I, R_S, R_I_B, R_S_B,\n",
    "                                  ", thresh_arg, ")"
  )

  if (has_counts) {
    lines <- c(lines, "", sprintf("    target += counts[n] * %s;", ll_expr))
  } else {
    lines <- c(lines, "", sprintf("    target += %s;", ll_expr))
  }

  lines
}


#' Generate likelihood code for CDP model
#' @noRd
generate_cdp_likelihood_code <- function(model_data) {
  lines <- character(0)
  
  # Recollection mean (mu_R) - mapped from dprime
  lines <- c(lines, generate_param_lp("dprime", model_data$dprime_fixed, model_data$dprime_random, model_data$smooth_data[["dprime"]]))
  lines <- c(lines, "    real mu_R = dprime_n;")
  
  # Familiarity mean (mu_F) - mapped from dprime2
  if (model_data$has_dprime2) {
    lines <- c(lines, generate_param_lp("dprime2", model_data$dprime2_fixed, model_data$dprime2_random, model_data$smooth_data[["dprime2"]]))
    lines <- c(lines, "    real mu_F = dprime2_n;")
  } else {
    lines <- c(lines, "    real mu_F = 0;")  # Fixed at 0 if not estimated
  }
  
  # sigma_R - mapped from sigma
  if (model_data$has_sigma) {
    lines <- c(lines, generate_param_lp("sigma", model_data$sigma_fixed, model_data$sigma_random, model_data$smooth_data[["sigma"]]))
    lines <- c(lines, "    real sigma_R = exp(sigma_n);")
  } else {
    lines <- c(lines, "    real sigma_R = 1;")
  }
  
  # sigma_F - mapped from sigma2
  if (model_data$has_sigma2) {
    lines <- c(lines, generate_param_lp("sigma2", model_data$sigma2_fixed, model_data$sigma2_random, model_data$smooth_data[["sigma2"]]))
    lines <- c(lines, "    real sigma_F = exp(sigma2_n);")
  } else {
    lines <- c(lines, "    real sigma_F = 1;")
  }
  
  # rec_crit (recollection criterion)
  if (isTRUE(model_data$has_rec_crit)) {
    lines <- c(lines, generate_param_lp("rec_crit", model_data$rec_crit_fixed, model_data$rec_crit_random, model_data$smooth_data[["rec_crit"]]))
    lines <- c(lines, "    real c_R = rec_crit_n;")
  } else {
    lines <- c(lines, "    real c_R = 1.5;")  # Default if not estimated
  }

  # know_crit (know criterion for R/K/G)
  n_rkg <- model_data$stan_data$n_rkg
  if (isTRUE(model_data$has_know_crit)) {
    lines <- c(lines, generate_param_lp("know_crit", model_data$know_crit_fixed, model_data$know_crit_random, model_data$smooth_data[["know_crit"]]))
    lines <- c(lines, "    real c_K = know_crit_n;")
  }

  # Compute derived CDP parameters
  lines <- c(lines, "",
             "    // Derived CDP parameters",
             "    real mu_M = mu_R + mu_F;  // Memory strength mean for targets",
             "    real sigma_M = sqrt(square(sigma_R) + square(sigma_F));  // Memory strength SD",
             "    real rho = sigma_R / sigma_M;  // Correlation between R and M",
             "",
             "    // Lure parameters (fixed reference)",
             "    real sigma_M_l = sqrt(2.0);",
             "    real rho_l = 1.0 / sqrt(2.0);")

  # Build thresholds using log-gaps parameterization
  lines <- c(lines, "", generate_thresh_code_cdp(model_data))

  # CDP likelihood - use local ll variable for counts support
  has_counts <- isTRUE(model_data$has_counts)

  # Common preamble: find old_idx
  lines <- c(lines, "",
             "    // CDP likelihood",
             "    real ll;",
             "    int is_old_resp = 0;",
             "    int old_idx = 0;",
             "    for (j in 1:J) {",
             "      if (y[n] == old_level_map[j]) {",
             "        is_old_resp = 1;",
             "        old_idx = j;",
             "      }",
             "    }")

  if (isTRUE(n_rkg == 3)) {
    # ---- R/K/G likelihood (n_rkg == 3) ----
    lines <- c(lines, "",
               "    if (is_old[n] == 1) {",
               "      // Target trial",
               "      if (is_old_resp == 1) {",
               "        // 'Old' response - use R/K/G",
               "        real tau_lo = tau[old_idx];",
               "        real tau_hi = (old_idx == J) ? 20.0 : tau[old_idx + 1];",
               "        vector[3] rkg = compute_rkg_probs(mu_R, sigma_R, mu_F, sigma_F, c_R, c_K, tau_lo, tau_hi);",
               "        ll = log(rkg[rk[n]]);",
               "      } else {",
               "        // 'New' response (below tau[1])",
               "        real z_tau1 = (tau[1] - mu_M) / sigma_M;",
               "        ll = normal_lcdf(z_tau1 | 0, 1);",
               "      }",
               "    } else {",
               "      // Lure trial",
               "      if (is_old_resp == 1) {",
               "        // 'Old' response (false alarm) - use R/K/G with lure params",
               "        real tau_lo = tau[old_idx];",
               "        real tau_hi = (old_idx == J) ? 20.0 : tau[old_idx + 1];",
               "        vector[3] rkg_l = compute_rkg_probs(0, 1, 0, 1, c_R, c_K, tau_lo, tau_hi);",
               "        ll = log(rkg_l[rk[n]]);",
               "      } else {",
               "        // Correct rejection",
               "        real z_tau1_l = tau[1] / sigma_M_l;",
               "        ll = normal_lcdf(z_tau1_l | 0, 1);",
               "      }",
               "    }")
  } else {
    # ---- R/K likelihood (n_rkg == 2, existing behavior) ----
    lines <- c(lines, "",
               "    if (is_old[n] == 1) {",
               "      // Target trial",
               "      real z_cR = (c_R - mu_R) / sigma_R;",
               "      ",
               "      if (is_old_resp == 1) {",
               "        // 'Old' response - use R/K",
               "        real tau_lo = tau[old_idx];",
               "        real tau_hi = (old_idx == J) ? 20.0 : tau[old_idx + 1];",
               "        real z_tau_lo = (tau_lo - mu_M) / sigma_M;",
               "        real z_tau_hi = (tau_hi - mu_M) / sigma_M;",
               "        ",
               "        if (rk[n] == 1) {",
               "          // Remember",
               "          ll = log(binormal_strip_upper(z_cR, z_tau_lo, z_tau_hi, rho));",
               "        } else {",
               "          // Know",
               "          ll = log(binormal_strip_lower(z_cR, z_tau_lo, z_tau_hi, rho));",
               "        }",
               "      } else {",
               "        // 'New' response (below tau[1])",
               "        real z_tau1 = (tau[1] - mu_M) / sigma_M;",
               "        ll = normal_lcdf(z_tau1 | 0, 1);",
               "      }",
               "    } else {",
               "      // Lure trial",
               "      real z_cR_l = c_R;  // Reference: mu_R=0, sigma_R=1",
               "      ",
               "      if (is_old_resp == 1) {",
               "        // 'Old' response (false alarm) - use R/K",
               "        real tau_lo = tau[old_idx];",
               "        real tau_hi = (old_idx == J) ? 20.0 : tau[old_idx + 1];",
               "        real z_tau_lo_l = tau_lo / sigma_M_l;",
               "        real z_tau_hi_l = tau_hi / sigma_M_l;",
               "        ",
               "        if (rk[n] == 1) {",
               "          ll = log(binormal_strip_upper(z_cR_l, z_tau_lo_l, z_tau_hi_l, rho_l));",
               "        } else {",
               "          ll = log(binormal_strip_lower(z_cR_l, z_tau_lo_l, z_tau_hi_l, rho_l));",
               "        }",
               "      } else {",
               "        // Correct rejection",
               "        real z_tau1_l = tau[1] / sigma_M_l;",
               "        ll = normal_lcdf(z_tau1_l | 0, 1);",
               "      }",
               "    }")
  }

  lines <- c(lines,
             if (has_counts) "    target += counts[n] * ll;" else "    target += ll;")

  lines
}



#' Generate likelihood code for 2D-VRDP model
#' @noRd
generate_vrdp2d_likelihood_code <- function(model_data) {
  lines <- character(0)
  
  varying_source_criteria <- isTRUE(model_data$varying_source_criteria)
  
  # dprime (d'_F - familiarity strength)
  lines <- c(lines, generate_param_lp_bivariate("dprime", model_data$dprime_fixed, model_data$dprime_random))
  lines <- c(lines, "    real dprime_F = dprime_n;")
  
  # dprime2 (d'_R - recollection boost)
  lines <- c(lines, generate_param_lp_bivariate("dprime2", model_data$dprime2_fixed, model_data$dprime2_random))
  lines <- c(lines, "    real dprime_R = dprime2_n;")
  
  # discrim (d'_S - source discriminability for Source A)
  lines <- c(lines, generate_param_lp_bivariate("discrim", model_data$discrim_fixed, model_data$discrim_random))
  lines <- c(lines, "    real dprime_S = discrim_n;")
  
  # discrim_B (d'_S_B - source discriminability for Source B)
  if (isTRUE(model_data$has_discrim_B)) {
    lines <- c(lines, generate_param_lp_bivariate("discrim_B", model_data$discrim_B_fixed, model_data$discrim_B_random))
    lines <- c(lines, "    real dprime_S_B = discrim_B_n;")
  } else {
    lines <- c(lines, "    real dprime_S_B = -dprime_S;  // symmetric: d'_S_B = -d'_S")
  }
  
  # lambda (R - recollection probability)
  lines <- c(lines, generate_param_lp_bivariate("lambda", model_data$lambda_fixed, model_data$lambda_random))
  lines <- c(lines, "    real R = inv_logit(lambda_n);")
  
  # sigma (sigma_item - SD on item dimension for old items)
  if (isTRUE(model_data$has_sigma)) {
    lines <- c(lines, generate_param_lp_bivariate("sigma", model_data$sigma_fixed, model_data$sigma_random))
    lines <- c(lines, "    real sigma_item = exp(sigma_n);")
  } else {
    lines <- c(lines, "    real sigma_item = 1;  // fixed")
  }
  
  # sigma2 (sigma_S - source SD for recollected items)
  if (isTRUE(model_data$has_sigma2)) {
    lines <- c(lines, generate_param_lp_bivariate("sigma2", model_data$sigma2_fixed, model_data$sigma2_random))
    lines <- c(lines, "    real sigma_S = exp(sigma2_n);")
  } else {
    lines <- c(lines, "    real sigma_S = 1;  // fixed")
  }
  
  # Item thresholds (always K1-1)
  lines <- c(lines, generate_thresh_code_bivariate(model_data, 1))
  
  # Source thresholds - either constant or varying by item response
  if (varying_source_criteria) {
    # Generate K1 sets of source thresholds, indexed by item response
    lines <- c(lines, generate_thresh_code_vrdp2d_varying(model_data))
  } else {
    # Single set of source thresholds
    lines <- c(lines, generate_thresh_code_bivariate(model_data, 2))
  }
  
  # Likelihood call
  has_counts <- isTRUE(model_data$has_counts)
  
  if (varying_source_criteria) {
    # Use thresh2_varying which is indexed by item response
    ll_expr <- paste0(
      "vrdp2d_varying_lpmf_custom(y[n], y2[n], item_type[n], K1, K2,\n",
      "                                  dprime_F, dprime_R, dprime_S, dprime_S_B,\n",
      "                                  R, sigma_item, sigma_S,\n",
      "                                  thresh1, thresh2_varying)"
    )
  } else {
    ll_expr <- paste0(
      "vrdp2d_lpmf_custom(y[n], y2[n], item_type[n], K1, K2,\n",
      "                                  dprime_F, dprime_R, dprime_S, dprime_S_B,\n",
      "                                  R, sigma_item, sigma_S,\n",
      "                                  thresh1, thresh2)"
    )
  }
  
  if (has_counts) {
    lines <- c(lines, "", sprintf("    target += counts[n] * %s;", ll_expr))
  } else {
    lines <- c(lines, "", sprintf("    target += %s;", ll_expr))
  }
  
  lines
}


#' Generate threshold code for vrdp2d with varying source criteria
#' @noRd
generate_thresh_code_vrdp2d_varying <- function(model_data) {
  # For varying source criteria, source thresholds vary by item response level
  # All K1 levels get source thresholds, using mid-anchor parameterization

  varying_re_mode <- if (!is.null(model_data$varying_re)) model_data$varying_re else "shared"

  # Check for RE on criterion2
  has_re <- !is.null(model_data$criterion2) && !is.null(model_data$criterion2$random) &&
    length(model_data$criterion2$random) > 0

  # Get RE variable names for later use
  re_info <- NULL
  if (has_re) {
    # Use first group (most common case: single grouping factor)
    group <- names(model_data$criterion2$random)[1]
    re <- model_data$criterion2$random[[group]]
    re_var_name <- if (!is.null(re$cor_id)) {
      sprintf("u_criterion2_%s_from_%s", group, re$cor_id)
    } else {
      sprintf("u_criterion2_%s", group)
    }
    re_info <- list(group = group, var_name = re_var_name, re = re)
  }

  lines <- c(
    "    // Source thresholds varying by item response (mid-anchor)",
    "    matrix[K1, K2-1] thresh2_varying;",
    "    for (k1 in 1:K1) {",
    "      // Build source thresholds for item response level k1"
  )

  # === Mid-anchor threshold for this bin ===
  mid_thresh_line <- sprintf("      thresh2_varying[k1, mid_thresh2] = beta_thresh_mid_2_varying[k1]")

  if (has_re) {
    if (varying_re_mode == "shared") {
      mid_thresh_line <- paste0(mid_thresh_line,
                                sprintf(" + %s[%s[n]];", re_info$var_name, re_info$group))
    } else if (varying_re_mode == "per_bin") {
      if (re_info$re$dim == 1) {
        mid_thresh_line <- paste0(mid_thresh_line,
                                  sprintf(" + %s[%s[n]];", re_info$var_name, re_info$group))
      } else {
        mid_thresh_line <- paste0(mid_thresh_line,
                                  sprintf(" + %s[%s[n], k1];", re_info$var_name, re_info$group))
      }
    } else if (varying_re_mode == "full") {
      mid_thresh_line <- paste0(mid_thresh_line,
                                sprintf(" + %s[%s[n], (k1-1)*(K2-1)+mid_thresh2];", re_info$var_name, re_info$group))
    }
  } else {
    mid_thresh_line <- paste0(mid_thresh_line, ";")
  }
  lines <- c(lines, mid_thresh_line)

  # === Upper gaps: mid+1 to K2-1 ===
  if (has_re && varying_re_mode == "full") {
    lines <- c(lines,
               "      for (k2 in (mid_thresh2+1):(K2-1)) {",
               sprintf("        thresh2_varying[k1, k2] = thresh2_varying[k1, k2-1] + exp(beta_log_gaps_2_varying[k1, k2-mid_thresh2] + %s[%s[n], (k1-1)*(K2-1)+k2]);",
                       re_info$var_name, re_info$group),
               "      }")
  } else {
    lines <- c(lines,
               "      for (k2 in (mid_thresh2+1):(K2-1)) {",
               "        thresh2_varying[k1, k2] = thresh2_varying[k1, k2-1] + exp(beta_log_gaps_2_varying[k1, k2-mid_thresh2]);",
               "      }")
  }

  # === Lower gaps: mid-1 down to 1 ===
  if (has_re && varying_re_mode == "full") {
    lines <- c(lines,
               "      for (k2_down in 1:(mid_thresh2-1)) {",
               "        int k2 = mid_thresh2 - k2_down;",
               sprintf("        thresh2_varying[k1, k2] = thresh2_varying[k1, k2+1] - exp(beta_log_gaps_2_varying[k1, K2-1-mid_thresh2+k2_down] + %s[%s[n], (k1-1)*(K2-1)+k2]);",
                       re_info$var_name, re_info$group),
               "      }")
  } else {
    lines <- c(lines,
               "      for (k2_down in 1:(mid_thresh2-1)) {",
               "        int k2 = mid_thresh2 - k2_down;",
               "        thresh2_varying[k1, k2] = thresh2_varying[k1, k2+1] - exp(beta_log_gaps_2_varying[k1, K2-1-mid_thresh2+k2_down]);",
               "      }")
  }

  lines <- c(lines, "    }")

  lines
}


#' Generate threshold code for CDP model
#' @noRd
generate_thresh_code_cdp <- function(model_data) {
  crit <- model_data$criterion
  J <- model_data$stan_data$J

  is_intercept_only <- isTRUE(crit$is_intercept_only) || crit$n_coef == 1

  # Helper: generate RE contribution for a given threshold position
  gen_re_cdp <- function(target, k_expr) {
    re_lines <- character(0)
    if (!is.null(crit$random)) {
      for (group in names(crit$random)) {
        re <- crit$random[[group]]
        var_name <- if (!is.null(re$cor_id)) {
          sprintf("u_cross_%s_%s", re$cor_id, group)
        } else {
          sprintf("u_criterion_%s", group)
        }

        if (re$dim > 1) {
          re_lines <- c(re_lines, sprintf("    %s += %s[%s[n], %s];", target, var_name, group, k_expr))
        } else {
          re_lines <- c(re_lines, sprintf("    %s += %s[%s[n], 1];", target, var_name, group))
        }
      }
    }
    re_lines
  }

  lines <- c("    // Build thresholds (tau) using mid-anchor + log-gaps parameterization",
             sprintf("    vector[J] tau;"))

  # === Mid-anchor threshold ===
  if (is_intercept_only) {
    lines <- c(lines, "    tau[mid_thresh_cdp] = beta_thresh_mid[1];")
  } else {
    lines <- c(lines, "    tau[mid_thresh_cdp] = X_criterion[n, ] * beta_thresh_mid;")
  }
  lines <- c(lines, gen_re_cdp("tau[mid_thresh_cdp]", "mid_thresh_cdp"))

  # Criterion smooth contribution for CDP
  if (!is.null(model_data$smooth_data[["criterion"]])) {
    for (sm in model_data$smooth_data[["criterion"]]) {
      for (comp in sm$components) {
        for (k in seq_along(comp$Zs_list)) {
          zs_name <- paste0("Zs_criterion_", comp$san_label, "_", k)
          coef_name <- paste0("s_criterion_", comp$san_label, "_", k)
          lines <- c(lines, sprintf("    tau[mid_thresh_cdp] += %s[n, ] * %s[1, ]';", zs_name, coef_name))
        }
      }
    }
  }

  # === Upper gaps: mid+1 to J ===
  lines <- c(lines,
             "    for (j in (mid_thresh_cdp+1):J) {")

  if (is_intercept_only) {
    lines <- c(lines, "      real log_gap = beta_log_gaps[j - mid_thresh_cdp, 1];")
  } else {
    lines <- c(lines, "      real log_gap = X_criterion[n, ] * beta_log_gaps[j - mid_thresh_cdp, ]';")
  }
  if (!is.null(crit$random)) {
    for (group in names(crit$random)) {
      re <- crit$random[[group]]
      var_name <- if (!is.null(re$cor_id)) {
        sprintf("u_cross_%s_%s", re$cor_id, group)
      } else {
        sprintf("u_criterion_%s", group)
      }
      if (re$dim > 1) {
        lines <- c(lines, sprintf("      log_gap += %s[%s[n], j];", var_name, group))
      }
    }
  }
  # CDP criterion smooth per-gap (upper)
  if (!is.null(model_data$smooth_data[["criterion"]])) {
    for (sm in model_data$smooth_data[["criterion"]]) {
      for (comp in sm$components) {
        for (ki in seq_along(comp$Zs_list)) {
          zs_name <- paste0("Zs_criterion_", comp$san_label, "_", ki)
          coef_name <- paste0("s_criterion_", comp$san_label, "_", ki)
          lines <- c(lines, sprintf("      log_gap += %s[n, ] * %s[1 + j - mid_thresh_cdp, ]';", zs_name, coef_name))
        }
      }
    }
  }
  lines <- c(lines, "      tau[j] = tau[j-1] + exp(log_gap);", "    }")

  # === Lower gaps: mid-1 down to 1 ===
  lines <- c(lines,
             "    for (j_down in 1:(mid_thresh_cdp-1)) {",
             "      int j = mid_thresh_cdp - j_down;")

  if (is_intercept_only) {
    lines <- c(lines, "      real log_gap = beta_log_gaps[J - mid_thresh_cdp + j_down, 1];")
  } else {
    lines <- c(lines, "      real log_gap = X_criterion[n, ] * beta_log_gaps[J - mid_thresh_cdp + j_down, ]';")
  }
  if (!is.null(crit$random)) {
    for (group in names(crit$random)) {
      re <- crit$random[[group]]
      var_name <- if (!is.null(re$cor_id)) {
        sprintf("u_cross_%s_%s", re$cor_id, group)
      } else {
        sprintf("u_criterion_%s", group)
      }
      if (re$dim > 1) {
        lines <- c(lines, sprintf("      log_gap += %s[%s[n], j];", var_name, group))
      }
    }
  }
  # CDP criterion smooth per-gap (lower)
  if (!is.null(model_data$smooth_data[["criterion"]])) {
    for (sm in model_data$smooth_data[["criterion"]]) {
      for (comp in sm$components) {
        for (ki in seq_along(comp$Zs_list)) {
          zs_name <- paste0("Zs_criterion_", comp$san_label, "_", ki)
          coef_name <- paste0("s_criterion_", comp$san_label, "_", ki)
          lines <- c(lines, sprintf("      log_gap += %s[n, ] * %s[1 + J - mid_thresh_cdp + j_down, ]';", zs_name, coef_name))
        }
      }
    }
  }
  lines <- c(lines, "      tau[j] = tau[j+1] - exp(log_gap);", "    }")

  lines
}


#' Generate threshold code for bivariate SDT (two sets of thresholds)
#' @noRd
generate_thresh_code_bivariate <- function(model_data, dimension) {
  if (dimension == 1) {
    crit <- model_data$criterion
    K <- model_data$K
    K_var <- "K1"
    thresh_name <- "thresh1"
    beta_thresh <- "beta_thresh_mid"
    beta_log_gaps <- "beta_log_gaps"
    mid_var <- "mid_thresh1"
    crit_name <- "criterion"
  } else {
    crit <- model_data$criterion2
    K <- model_data$K2
    K_var <- "K2"
    thresh_name <- "thresh2"
    beta_thresh <- "beta_thresh_mid_2"
    beta_log_gaps <- "beta_log_gaps_2"
    mid_var <- "mid_thresh2"
    crit_name <- "criterion2"
  }

  if (is.null(crit)) {
    stop("Criterion structure not found for dimension ", dimension)
  }

  is_intercept_only <- isTRUE(crit$is_intercept_only) || crit$n_coef == 1

  # Helper: generate RE contribution for a given threshold position
  gen_re_biv <- function(target, k_expr) {
    re_lines <- character(0)
    if (!is.null(crit$random)) {
      for (group in names(crit$random)) {
        re <- crit$random[[group]]
        var_name <- if (!is.null(re$cor_id)) {
          sprintf("u_%s_%s_from_%s", crit_name, group, re$cor_id)
        } else {
          sprintf("u_%s_%s", crit_name, group)
        }
        n_re_terms <- if (!is.null(re$n_re_terms)) re$n_re_terms else re$n_cond_levels

        if (isTRUE(re$use_z_matrix)) {
          z_name <- paste0("Z_", crit_name, "_", group)
          re_lines <- c(re_lines, sprintf("    %s += %s[n, ] * %s[%s[n], ((%s-1)*%d+1):(%s*%d)]';",
                                          target, z_name, var_name, group,
                                          k_expr, n_re_terms, k_expr, n_re_terms))
        } else if (re$dim == 1) {
          re_lines <- c(re_lines, sprintf("    %s += %s[%s[n]];", target, var_name, group))
        } else {
          re_lines <- c(re_lines, sprintf("    %s += %s[%s[n], %s];", target, var_name, group, k_expr))
        }
      }
    }
    re_lines
  }

  lines <- c(sprintf("    vector[%s-1] %s;", K_var, thresh_name))

  # === Mid-anchor threshold ===
  if (is_intercept_only) {
    lines <- c(lines, sprintf("    %s[%s] = %s[1];", thresh_name, mid_var, beta_thresh))
  } else {
    lines <- c(lines, sprintf("    %s[%s] = X_%s[n, ] * %s;", thresh_name, mid_var, crit_name, beta_thresh))
  }
  lines <- c(lines, gen_re_biv(sprintf("%s[%s]", thresh_name, mid_var), mid_var))

  # Criterion/criterion2 smooth contribution for bivariate (mid-anchor: row 1)
  smooth_key <- if (dimension == 1) "criterion" else "criterion2"
  if (!is.null(model_data$smooth_data[[smooth_key]])) {
    for (sm in model_data$smooth_data[[smooth_key]]) {
      for (comp in sm$components) {
        for (ki in seq_along(comp$Zs_list)) {
          zs_name <- paste0("Zs_", smooth_key, "_", comp$san_label, "_", ki)
          coef_name <- paste0("s_", smooth_key, "_", comp$san_label, "_", ki)
          lines <- c(lines, sprintf("    %s[%s] += %s[n, ] * %s[1, ]';", thresh_name, mid_var, zs_name, coef_name))
        }
      }
    }
  }

  # === Upper gaps: mid+1 to K-1 ===
  lines <- c(lines, sprintf("    for (k in (%s+1):(%s-1)) {", mid_var, K_var))

  if (is_intercept_only) {
    lines <- c(lines, sprintf("      real log_gap = %s[k - %s, 1];", beta_log_gaps, mid_var))
  } else {
    lines <- c(lines, sprintf("      real log_gap = X_%s[n, ] * %s[k - %s, ]';", crit_name, beta_log_gaps, mid_var))
  }
  lines <- c(lines, gen_re_biv("log_gap", "k"))
  # Bivariate criterion smooth per-gap (upper)
  if (!is.null(model_data$smooth_data[[smooth_key]])) {
    for (sm in model_data$smooth_data[[smooth_key]]) {
      for (comp in sm$components) {
        for (ki in seq_along(comp$Zs_list)) {
          zs_name <- paste0("Zs_", smooth_key, "_", comp$san_label, "_", ki)
          coef_name <- paste0("s_", smooth_key, "_", comp$san_label, "_", ki)
          lines <- c(lines, sprintf("      log_gap += %s[n, ] * %s[1 + k - %s, ]';", zs_name, coef_name, mid_var))
        }
      }
    }
  }
  lines <- c(lines, sprintf("      %s[k] = %s[k-1] + exp(log_gap);", thresh_name, thresh_name), "    }")

  # === Lower gaps: mid-1 down to 1 ===
  lines <- c(lines, sprintf("    for (k_down in 1:(%s-1)) {", mid_var),
             sprintf("      int k = %s - k_down;", mid_var))

  if (is_intercept_only) {
    lines <- c(lines, sprintf("      real log_gap = %s[%s - 1 - %s + k_down, 1];", beta_log_gaps, K_var, mid_var))
  } else {
    lines <- c(lines, sprintf("      real log_gap = X_%s[n, ] * %s[%s - 1 - %s + k_down, ]';", crit_name, beta_log_gaps, K_var, mid_var))
  }
  lines <- c(lines, gen_re_biv("log_gap", "k"))
  # Bivariate criterion smooth per-gap (lower)
  if (!is.null(model_data$smooth_data[[smooth_key]])) {
    for (sm in model_data$smooth_data[[smooth_key]]) {
      for (comp in sm$components) {
        for (ki in seq_along(comp$Zs_list)) {
          zs_name <- paste0("Zs_", smooth_key, "_", comp$san_label, "_", ki)
          coef_name <- paste0("s_", smooth_key, "_", comp$san_label, "_", ki)
          lines <- c(lines, sprintf("      log_gap += %s[n, ] * %s[1 + %s - 1 - %s + k_down, ]';", zs_name, coef_name, K_var, mid_var))
        }
      }
    }
  }
  lines <- c(lines, sprintf("      %s[k] = %s[k+1] - exp(log_gap);", thresh_name, thresh_name), "    }")

  lines
}


#' Generate parameter linear predictor for bivariate SDT (no is_old branching)
#' @noRd
generate_param_lp_bivariate <- function(param_name, fixed_info, random_info) {
  lines <- character(0)
  
  has_fixed <- !is.null(fixed_info) && fixed_info$n_coef > 0
  
  if (!has_fixed) {
    lines <- c(lines, sprintf("    real %s_n = 0;", param_name))
  } else {
    lines <- c(lines, sprintf("    real %s_n = X_%s[n, ] * beta_%s;", 
                              param_name, param_name, param_name))
  }
  
  if (!is.null(random_info)) {
    for (group in names(random_info)) {
      re <- random_info[[group]]
      var_name <- if (!is.null(re$cor_id)) {
        sprintf("u_%s_%s_from_%s", param_name, group, re$cor_id)
      } else {
        sprintf("u_%s_%s", param_name, group)
      }
      
      if (isTRUE(re$use_z_matrix)) {
        z_name <- paste0("Z_", param_name, "_", group)
        lines <- c(lines, sprintf("    %s_n += %s[n, ] * %s[%s[n], ]';", param_name, z_name, var_name, group))
      } else if (re$dim == 1) {
        # Scalar random effect: u is a vector
        if (!is.null(re$term_idx)) {
          idx_var <- paste0("idx_", param_name, "_", group)
          lines <- c(lines, sprintf("    if (%s[n] > 0) %s_n += %s[%s[n]];", idx_var, param_name, var_name, group))
        } else {
          lines <- c(lines, sprintf("    %s_n += %s[%s[n]];", param_name, var_name, group))
        }
      } else if (!is.null(re$term_idx)) {
        # Multi-dimensional index-based lookup: u is a matrix
        idx_var <- paste0("idx_", param_name, "_", group)
        lines <- c(lines, sprintf("    if (%s[n] > 0) %s_n += %s[%s[n], %s[n]];", idx_var, param_name, var_name, group, idx_var))
      }
    }
  }

  lines
}


#' Generate likelihood with branching for encoding variables
#' This avoids computing dprime/lambda for new items
#' @noRd
generate_likelihood_with_encoding_branch <- function(model_data, family_name) {
  lines <- character(0)
  has_counts <- isTRUE(model_data$has_counts)
  
  # Thresholds are computed for ALL items (same as handwritten)
  lines <- c(lines, generate_thresh_code(model_data))
  
  # Branch on is_old
  lines <- c(lines, "", "    if (is_old[n] == 1) {")
  
  # Compute encoding parameters ONLY for old items
  lines <- c(lines, generate_param_lp_old_only("dprime", model_data$dprime_fixed, model_data$dprime_random, model_data$smooth_data[["dprime"]]))
  
  if (model_data$has_sigma) {
    lines <- c(lines, generate_param_lp_old_only("sigma", model_data$sigma_fixed, model_data$sigma_random, model_data$smooth_data[["sigma"]]))
    lines <- c(lines, "      real sigma_val = exp(sigma_n);")
  }
  
  if (model_data$has_lambda) {
    lines <- c(lines, generate_param_lp_old_only("lambda", model_data$lambda_fixed, model_data$lambda_random, model_data$smooth_data[["lambda"]]))
    lines <- c(lines, "      real lambda_val = inv_logit(lambda_n);")
  }
  
  if (model_data$has_dprime2 || model_data$needs_ordered_dprime) {
    lines <- c(lines, generate_param_lp_old_only("dprime2", model_data$dprime2_fixed, model_data$dprime2_random, model_data$smooth_data[["dprime2"]]))
  }
  
  if (model_data$has_sigma2) {
    lines <- c(lines, generate_param_lp_old_only("sigma2", model_data$sigma2_fixed, model_data$sigma2_random, model_data$smooth_data[["sigma2"]]))
    lines <- c(lines, "      real sigma2_val = exp(sigma2_n);")
  }
  
  # Likelihood for old items
  count_mult <- if (has_counts) "counts[n] * " else ""
  
  if (family_name == "dpsdt") {
    if (model_data$has_sigma) {
      lines <- c(lines, sprintf("      target += %sdpsdt_lpmf(y[n] | dprime_n, sigma_val, lambda_val, 1, K, thresh);", count_mult))
    } else {
      lines <- c(lines, sprintf("      target += %sdpsdt_lpmf(y[n] | dprime_n, lambda_val, 1, K, thresh);", count_mult))
    }
  } else if (family_name == "mixture") {
    if (model_data$has_lambda) {
      # Signal mixture: old items are a mixture of two states
      sigma_fixed <- !model_data$has_sigma
      sigma2_fixed <- !model_data$has_sigma2

      if (sigma_fixed && sigma2_fixed) {
        dprime2_arg <- if (model_data$has_dprime2 || model_data$needs_ordered_dprime) "dprime2_n" else "0"
        lines <- c(lines, sprintf("      target += %smixture_sdt_lpmf(y[n] | dprime_n, %s, lambda_val, 1, K, thresh);", count_mult, dprime2_arg))
      } else {
        sigma_val <- if (model_data$has_sigma) "sigma_val" else "1"
        sigma2_val <- if (model_data$has_sigma2) "sigma2_val" else "1"
        dprime2_arg <- if (model_data$has_dprime2 || model_data$needs_ordered_dprime) "dprime2_n" else "0"
        lines <- c(lines, sprintf("      target += %smixture_sdt_lpmf(y[n] | dprime_n, %s, %s, %s, lambda_val, 1, K, thresh);",
                                  count_mult, sigma_val, dprime2_arg, sigma2_val))
      }
    } else {
      # No signal mixture (lure-only): old items use UVSDT or EVSDT
      if (model_data$has_sigma) {
        lines <- c(lines, sprintf("      target += %suvsdt_lpmf(y[n] | dprime_n, sigma_val, 1, K, thresh);", count_mult))
      } else {
        lines <- c(lines, sprintf("      target += %sevsdt_lpmf(y[n] | dprime_n, 1, K, thresh);", count_mult))
      }
    }
  }
  
  # Else branch for new items
  if (isTRUE(model_data$has_lure_mixture)) {
    lines <- c(lines, "    } else {")
    # Compute lure mixture parameters for new items
    lines <- c(lines, generate_param_lp_old_only("dprime_L", model_data$dprime_L_fixed, model_data$dprime_L_random, model_data$smooth_data[["dprime_L"]]))
    if (isTRUE(model_data$has_sigma_L)) {
      lines <- c(lines, generate_param_lp_old_only("sigma_L", model_data$sigma_L_fixed, model_data$sigma_L_random, model_data$smooth_data[["sigma_L"]]))
      lines <- c(lines, "      real sigma_L_val = exp(sigma_L_n);")
    }
    lines <- c(lines, generate_param_lp_old_only("lambda_L", model_data$lambda_L_fixed, model_data$lambda_L_random, model_data$smooth_data[["lambda_L"]]))
    lines <- c(lines, "      real lambda_L_val = inv_logit(lambda_L_n);")
    
    if (isTRUE(model_data$has_sigma_L)) {
      lines <- c(lines, sprintf("      target += %slure_mixture_lpmf(y[n] | dprime_L_n, sigma_L_val, lambda_L_val, K, thresh);", count_mult))
    } else {
      lines <- c(lines, sprintf("      target += %slure_mixture_lpmf(y[n] | dprime_L_n, lambda_L_val, K, thresh);", count_mult))
    }
    lines <- c(lines, "    }")
  } else {
    lines <- c(lines, "    } else {",
               sprintf("      target += %sevsdt_lpmf(y[n] | 0, 0, K, thresh);", count_mult),
               "    }")
  }
  
  lines
}


#' Generate parameter linear predictor for old items only (no is_old check)
#' Uses direct indexing when possible for faster sampling
#' @noRd
generate_param_lp_old_only <- function(param_name, fixed_info, random_info, smooth_info = NULL) {
  lines <- character(0)

  # Check whether the fast direct indexing pattern applies
  use_direct_indexing <- is_simple_encoding(fixed_info, random_info)

  if (use_direct_indexing) {
    # Fast path: use beta[c] + u[s, c] + u[i, c] pattern
    # Get condition index variable from first random effect
    first_group <- names(random_info)[1]
    idx_var <- paste0("idx_", param_name, "_", first_group)

    # Start with fixed effect using direct index
    lines <- c(lines, sprintf("      int c_%s = %s[n];", param_name, idx_var))
    lines <- c(lines, sprintf("      real %s_n = beta_%s[c_%s];", param_name, param_name, param_name))

    # Add random effects using the same condition index
    for (group in names(random_info)) {
      re <- random_info[[group]]
      var_name <- if (!is.null(re$cor_id)) sprintf("u_%s_%s_from_%s", param_name, group, re$cor_id) else sprintf("u_%s_%s", param_name, group)
      lines <- c(lines, sprintf("      %s_n += %s[%s[n], c_%s];", param_name, var_name, group, param_name))
    }
  } else {
    # General path: use design matrix multiplication for fixed effects
    lines <- c(lines, sprintf("      real %s_n = X_%s[n, ] * beta_%s;", param_name, param_name, param_name))

    if (!is.null(random_info)) {
      for (group in names(random_info)) {
        re <- random_info[[group]]
        var_name <- if (!is.null(re$cor_id)) sprintf("u_%s_%s_from_%s", param_name, group, re$cor_id) else sprintf("u_%s_%s", param_name, group)

        if (isTRUE(re$use_z_matrix)) {
          # Use Z matrix multiplication
          z_name <- paste0("Z_", param_name, "_", group)
          lines <- c(lines, sprintf("      %s_n += %s[n, ] * %s[%s[n], ]';", param_name, z_name, var_name, group))
        } else if (re$dim == 1) {
          # Scalar random effect: u is a vector
          if (!is.null(re$term_idx)) {
            idx_var <- paste0("idx_", param_name, "_", group)
            lines <- c(lines, sprintf("      if (%s[n] > 0) %s_n += %s[%s[n]];", idx_var, param_name, var_name, group))
          } else {
            lines <- c(lines, sprintf("      %s_n += %s[%s[n]];", param_name, var_name, group))
          }
        } else if (!is.null(re$term_idx)) {
          # Multi-dimensional index-based lookup: u is a matrix
          idx_var <- paste0("idx_", param_name, "_", group)
          lines <- c(lines, sprintf("      if (%s[n] > 0) %s_n += %s[%s[n], %s[n]];", idx_var, param_name, var_name, group, idx_var))
        }
      }
    }
  }

  # Add smooth penalized contributions: Zs * s (where s = sds * zs)
  if (!is.null(smooth_info)) {
    for (sm in smooth_info) {
      for (j in seq_along(sm$components)) {
        comp <- sm$components[[j]]
        comp_suffix <- comp$san_label
        for (k in seq_along(comp$Zs_list)) {
          zs_name <- paste0("Zs_", param_name, "_", comp_suffix, "_", k)
          coef_name <- paste0("s_", param_name, "_", comp_suffix, "_", k)
          lines <- c(lines, sprintf("      %s_n += %s[n, ] * %s;", param_name, zs_name, coef_name))
        }
      }
    }
  }

  lines
}


#' Generate parameter linear predictor (standard version for non-encoding params)
#' @noRd
generate_param_lp <- function(param_name, fixed_info, random_info, smooth_info = NULL) {
  lines <- character(0)

  # Check if there are fixed effects
  has_fixed <- !is.null(fixed_info) && fixed_info$n_coef > 0

  # Compute fixed effects inline (or initialize to 0 if no fixed effects)
  if (!has_fixed) {
    # No fixed effects - start at 0 (for cumulative with only random effects)
    lines <- c(lines, sprintf("    real %s_n = 0;", param_name))
  } else if (isTRUE(fixed_info$has_encoding)) {
    lines <- c(lines, sprintf("    real %s_n = 0;", param_name))
    lines <- c(lines, sprintf("    if (is_old[n] == 1) %s_n = X_%s[n, ] * beta_%s;",
                              param_name, param_name, param_name))
  } else {
    lines <- c(lines, sprintf("    real %s_n = X_%s[n, ] * beta_%s;",
                              param_name, param_name, param_name))
  }

  if (!is.null(random_info)) {
    for (group in names(random_info)) {
      re <- random_info[[group]]
      var_name <- if (!is.null(re$cor_id)) sprintf("u_%s_%s_from_%s", param_name, group, re$cor_id) else sprintf("u_%s_%s", param_name, group)

      if (isTRUE(re$use_z_matrix)) {
        # Use Z matrix multiplication
        z_name <- paste0("Z_", param_name, "_", group)
        if (re$dim == 1) {
          # Scalar RE with Z matrix: u is a vector, Z is N x 1
          # Z[n, 1] * u[group[n]] (scalar * scalar)
          lines <- c(lines, sprintf("    %s_n += %s[n, 1] * %s[%s[n]];", param_name, z_name, var_name, group))
        } else {
          # Multi-dimensional RE: u is a matrix, standard row-vector multiply
          lines <- c(lines, sprintf("    %s_n += %s[n, ] * %s[%s[n], ]';", param_name, z_name, var_name, group))
        }
      } else if (re$dim == 1) {
        # Scalar random effect: u is a vector
        if (!is.null(re$term_idx)) {
          # Has term index -- guard for observations where slope doesn't apply
          idx_var <- paste0("idx_", param_name, "_", group)
          lines <- c(lines, sprintf("    if (%s[n] > 0) %s_n += %s[%s[n]];", idx_var, param_name, var_name, group))
        } else {
          # Intercept-only random effect (no slopes, no indexing needed)
          lines <- c(lines, sprintf("    %s_n += %s[%s[n]];", param_name, var_name, group))
        }
      } else if (!is.null(re$term_idx)) {
        # Multi-dimensional index-based lookup: u is a matrix
        idx_var <- paste0("idx_", param_name, "_", group)
        lines <- c(lines, sprintf("    if (%s[n] > 0) %s_n += %s[%s[n], %s[n]];", idx_var, param_name, var_name, group, idx_var))
      }
    }
  }

  # Add smooth penalized contributions: Zs * s (where s = sds * zs)
  if (!is.null(smooth_info)) {
    for (sm in smooth_info) {
      for (j in seq_along(sm$components)) {
        comp <- sm$components[[j]]
        comp_suffix <- comp$san_label
        for (k in seq_along(comp$Zs_list)) {
          zs_name <- paste0("Zs_", param_name, "_", comp_suffix, "_", k)
          coef_name <- paste0("s_", param_name, "_", comp_suffix, "_", k)
          lines <- c(lines, sprintf("    %s_n += %s[n, ] * %s;", param_name, zs_name, coef_name))
        }
      }
    }
  }

  lines
}


#' Generate the likelihood function call
#' @noRd
generate_likelihood_call <- function(model_data, family_name) {
  has_counts <- isTRUE(model_data$has_counts)
  
  # Generate the log-likelihood expression
  ll_expr <- if (family_name == "cumulative") {
    # For cumulative, use mu (which is dprime_n) directly, no is_old
    "cumulative_lpmf(y[n] | dprime_n, K, thresh)"
  } else if (family_name == "source_mixture") {
    # Source discrimination mixture SDT
    has_dprime_B <- isTRUE(model_data$has_dprime_B)
    has_lambda_B <- isTRUE(model_data$has_lambda_B)
    
    if (has_dprime_B && has_lambda_B) {
      "source_mixture_lpmf(y[n] | source[n], dprime_n, dprime_B_val, lambda_val, lambda_B_val, K, thresh)"
    } else if (has_dprime_B) {
      "source_mixture_lpmf(y[n] | source[n], dprime_n, dprime_B_val, lambda_val, K, thresh)"
    } else if (has_lambda_B) {
      "source_mixture_lpmf(y[n] | source[n], dprime_n, lambda_val, lambda_B_val, K, thresh)"
    } else {
      "source_mixture_lpmf(y[n] | source[n], dprime_n, lambda_val, K, thresh)"
    }
  } else if (family_name == "evsdt") {
    "evsdt_lpmf(y[n] | dprime_n, is_old[n], K, thresh)"
  } else if (family_name == "uvsdt") {
    "uvsdt_lpmf(y[n] | dprime_n, sigma_val, is_old[n], K, thresh)"
  } else if (family_name == "dpsdt") {
    if (model_data$has_sigma) {
      "dpsdt_lpmf(y[n] | dprime_n, sigma_val, lambda_val, is_old[n], K, thresh)"
    } else {
      "dpsdt_lpmf(y[n] | dprime_n, lambda_val, is_old[n], K, thresh)"
    }
  } else if (family_name == "mixture") {
    if (model_data$has_lambda) {
      sigma_fixed <- !model_data$has_sigma
      sigma2_fixed <- !model_data$has_sigma2

      if (sigma_fixed && sigma2_fixed) {
        dprime2_arg <- if (model_data$has_dprime2 || model_data$needs_ordered_dprime) "dprime2_val" else "0"
        sprintf("mixture_sdt_lpmf(y[n] | dprime_n, %s, lambda_val, is_old[n], K, thresh)", dprime2_arg)
      } else {
        sigma_val <- if (model_data$has_sigma) "sigma_val" else "1"
        sigma2_val <- if (model_data$has_sigma2) "sigma2_val" else "1"
        dprime2_arg <- if (model_data$has_dprime2 || model_data$needs_ordered_dprime) "dprime2_val" else "0"
        sprintf("mixture_sdt_lpmf(y[n] | dprime_n, %s, %s, %s, lambda_val, is_old[n], K, thresh)",
                sigma_val, dprime2_arg, sigma2_val)
      }
    } else {
      # No signal mixture (lure-only): old items use UVSDT or EVSDT
      if (model_data$has_sigma) {
        "uvsdt_lpmf(y[n] | dprime_n, sigma_val, is_old[n], K, thresh)"
      } else {
        "evsdt_lpmf(y[n] | dprime_n, is_old[n], K, thresh)"
      }
    }
  } else {
    stop("Unknown family: ", family_name)
  }
  
  # Wrap with counts multiplier if needed
  if (has_counts) {
    sprintf("    target += counts[n] * %s;", ll_expr)
  } else {
    sprintf("    target += %s;", ll_expr)
  }
}


generate_thresh_code <- function(model_data) {
  crit <- model_data$criterion
  K <- model_data$K

  # Check if criterion is intercept-only (most common case)
  is_intercept_only <- isTRUE(crit$is_intercept_only) || crit$n_coef == 1

  # Helper: generate RE contribution for a given threshold position
  # thresh_k_expr: Stan expression for the threshold position index (e.g., "mid_thresh", "k")
  # role: "mid", "gap_up", "gap_down" -- determines how RE column is accessed
  gen_re_lines <- function(target, thresh_k_expr, role = "mid") {
    re_lines <- character(0)
    for (group in names(crit$random)) {
      re <- crit$random[[group]]
      var_name <- if (!is.null(re$cor_id)) sprintf("u_criterion_%s_from_%s", group, re$cor_id) else sprintf("u_criterion_%s", group)
      n_re_terms <- if (!is.null(re$n_re_terms)) re$n_re_terms else re$n_cond_levels

      if (isTRUE(re$use_z_matrix)) {
        z_name <- paste0("Z_criterion_", group)
        re_lines <- c(re_lines, sprintf("    %s += %s[n, ] * %s[%s[n], ((%s-1)*%d + 1):(%s*%d)]';",
                                        target, z_name, var_name, group,
                                        thresh_k_expr, n_re_terms, thresh_k_expr, n_re_terms))
      } else if (n_re_terms == 1) {
        re_lines <- c(re_lines, sprintf("    %s += %s[%s[n], %s];", target, var_name, group, thresh_k_expr))
      } else {
        idx_var <- paste0("idx_criterion_", group)
        re_lines <- c(re_lines, sprintf("    if (%s[n] > 0) %s += %s[%s[n], (%s-1) * %d + %s[n]];",
                                        idx_var, target, var_name, group, thresh_k_expr, n_re_terms, idx_var))
      }
    }
    re_lines
  }

  lines <- c("    vector[K-1] thresh;")

  # === Mid-anchor threshold ===
  if (is_intercept_only) {
    lines <- c(lines, "    thresh[mid_thresh] = beta_thresh_mid[1];")
  } else {
    lines <- c(lines, "    thresh[mid_thresh] = X_criterion[n, ] * beta_thresh_mid;")
  }
  lines <- c(lines, gen_re_lines("thresh[mid_thresh]", "mid_thresh"))

  # Criterion smooth contribution to mid-anchor (row 1 of per-threshold smooth)
  if (!is.null(model_data$smooth_data[["criterion"]])) {
    for (sm in model_data$smooth_data[["criterion"]]) {
      for (comp in sm$components) {
        for (k in seq_along(comp$Zs_list)) {
          zs_name <- paste0("Zs_criterion_", comp$san_label, "_", k)
          coef_name <- paste0("s_criterion_", comp$san_label, "_", k)
          lines <- c(lines, sprintf("    thresh[mid_thresh] += %s[n, ] * %s[1, ]';", zs_name, coef_name))
        }
      }
    }
  }

  if (K > 2) {
    # === Upper gaps: mid+1 to K-1 ===
    lines <- c(lines,
               "    for (k in (mid_thresh+1):(K-1)) {")

    if (is_intercept_only) {
      lines <- c(lines, "      real log_gap = beta_log_gaps[k - mid_thresh, 1];")
    } else {
      lines <- c(lines, "      real log_gap = X_criterion[n, ] * beta_log_gaps[k - mid_thresh, ]';")
    }
    lines <- c(lines, gen_re_lines("log_gap", "k"))
    # Criterion smooth contribution to each gap
    if (!is.null(model_data$smooth_data[["criterion"]])) {
      for (sm in model_data$smooth_data[["criterion"]]) {
        for (comp in sm$components) {
          for (ki in seq_along(comp$Zs_list)) {
            zs_name <- paste0("Zs_criterion_", comp$san_label, "_", ki)
            coef_name <- paste0("s_criterion_", comp$san_label, "_", ki)
            lines <- c(lines, sprintf("      log_gap += %s[n, ] * %s[1 + k - mid_thresh, ]';", zs_name, coef_name))
          }
        }
      }
    }
    lines <- c(lines, "      thresh[k] = thresh[k-1] + exp(log_gap);", "    }")

    # === Lower gaps: mid-1 down to 1 ===
    lines <- c(lines,
               "    for (k_down in 1:(mid_thresh-1)) {",
               "      int k = mid_thresh - k_down;")

    if (is_intercept_only) {
      lines <- c(lines, "      real log_gap = beta_log_gaps[K - 1 - mid_thresh + k_down, 1];")
    } else {
      lines <- c(lines, "      real log_gap = X_criterion[n, ] * beta_log_gaps[K - 1 - mid_thresh + k_down, ]';")
    }
    lines <- c(lines, gen_re_lines("log_gap", "k"))
    # Criterion smooth contribution to lower gaps
    if (!is.null(model_data$smooth_data[["criterion"]])) {
      for (sm in model_data$smooth_data[["criterion"]]) {
        for (comp in sm$components) {
          for (ki in seq_along(comp$Zs_list)) {
            zs_name <- paste0("Zs_criterion_", comp$san_label, "_", ki)
            coef_name <- paste0("s_criterion_", comp$san_label, "_", ki)
            # Gap index in the s_ matrix: n_upper_gaps + k_down (rows after mid and upper gaps)
            lines <- c(lines, sprintf("      log_gap += %s[n, ] * %s[1 + K - 1 - mid_thresh + k_down, ]';", zs_name, coef_name))
          }
        }
      }
    }
    lines <- c(lines, "      thresh[k] = thresh[k+1] - exp(log_gap);", "    }")
  }

  lines
}


generate_generated_quantities_block_v2 <- function(model_data, family) {
  family_name <- if (is.list(family)) family$family else family
  
  lines <- c("generated quantities {")
  
  param_names <- c("dprime", "sigma", "lambda", "dprime2", "sigma2",
                    "rec_crit", "know_crit",
                    "dprime_B", "lambda_B", "discrim", "discrim_B",
                    "sigma_B", "sigma2_B", "rho", "rho_B", "rho_N",
                    "lambda2", "lambda2_B", "dprime_L", "sigma_L", "lambda_L")
  for (pname in param_names) {
    re_list <- model_data[[paste0(pname, "_random")]]
    if (is.null(re_list)) next
    for (group in names(re_list)) {
      re <- re_list[[group]]
      if (is.null(re$cor_id) && isTRUE(re$correlated) && re$dim > 1) {
        dim_var <- sprintf("D_%s_%s", pname, group)
        lines <- c(lines, sprintf("  matrix[%s, %s] corr_%s_%s = multiply_lower_tri_self_transpose(L_corr_%s_%s);",
                                  dim_var, dim_var, pname, group, pname, group))
      }
    }
  }
  
  cor_threshold <- isTRUE(model_data$cor_threshold)
  for (group in names(model_data$criterion$random)) {
    re <- model_data$criterion$random[[group]]
    if (is.null(re$cor_id) && cor_threshold && isTRUE(re$correlated) && re$dim > 1) {
      dim_var <- sprintf("D_criterion_%s", group)
      lines <- c(lines, sprintf("  matrix[%s, %s] corr_criterion_%s = multiply_lower_tri_self_transpose(L_corr_criterion_%s);",
                                dim_var, dim_var, group, group))
    }
  }
  
  # Criterion2 correlations (for bivariate_sdt)
  if (isTRUE(model_data$has_criterion2) && !is.null(model_data$criterion2$random)) {
    for (group in names(model_data$criterion2$random)) {
      re <- model_data$criterion2$random[[group]]
      if (is.null(re$cor_id) && cor_threshold && isTRUE(re$correlated) && re$dim > 1) {
        dim_var <- sprintf("D_criterion2_%s", group)
        lines <- c(lines, sprintf("  matrix[%s, %s] corr_criterion2_%s = multiply_lower_tri_self_transpose(L_corr_criterion2_%s);",
                                  dim_var, dim_var, group, group))
      }
    }
  }
  
  for (cor_id in names(model_data$cross_cor)) {
    cc <- model_data$cross_cor[[cor_id]]
    dim_var <- sprintf("D_cross_%s_%s", cor_id, cc$group)
    lines <- c(lines, sprintf("  matrix[%s, %s] corr_cross_%s_%s = multiply_lower_tri_self_transpose(L_corr_cross_%s_%s);",
                              dim_var, dim_var, cor_id, cc$group, cor_id, cc$group))
  }
  
  lines <- c(lines, "", "  vector[N] log_lik;", "  for (n in 1:N) {")
  ll_code <- generate_likelihood_code(model_data, family_name)
  # Collapse to single string, replace target += with log_lik[n] =, then split back
  ll_str <- paste(ll_code, collapse = "\n")
  ll_str <- gsub("target \\+= ([^;]+);", "log_lik[n] = \\1;", ll_str)
  # Strip count multiplier so log_lik[n] is per-trial (not count-weighted).
  # LOO needs per-observation log-likelihoods; loo_broc() handles expansion.
  ll_str <- gsub("log_lik\\[n\\] = counts\\[n\\] \\* ", "log_lik[n] = ", ll_str)
  ll_code <- strsplit(ll_str, "\n")[[1]]
  lines <- c(lines, ll_code)
  lines <- c(lines, "  }", "}")
  
  paste(lines, collapse = "\n")
}


get_all_groups <- function(model_data) {
  groups <- c(names(model_data$dprime_random), names(model_data$criterion$random))
  if (model_data$has_sigma) groups <- c(groups, names(model_data$sigma_random))
  if (model_data$has_lambda) groups <- c(groups, names(model_data$lambda_random))
  if (model_data$has_dprime2) groups <- c(groups, names(model_data$dprime2_random))
  if (model_data$has_sigma2) groups <- c(groups, names(model_data$sigma2_random))
  if (isTRUE(model_data$has_rec_crit)) groups <- c(groups, names(model_data$rec_crit_random))
  if (isTRUE(model_data$has_know_crit)) groups <- c(groups, names(model_data$know_crit_random))
  # Bivariate/source parameters
  if (isTRUE(model_data$has_criterion2) && !is.null(model_data$criterion2$random)) {
    groups <- c(groups, names(model_data$criterion2$random))
  }
  for (pname in c("dprime_B", "lambda_B", "discrim", "discrim_B",
                  "sigma_B", "sigma2_B", "rho", "rho_B", "rho_N",
                  "dprime_L", "sigma_L", "lambda_L", "lambda2", "lambda2_B")) {
    re_list <- model_data[[paste0(pname, "_random")]]
    if (!is.null(re_list)) groups <- c(groups, names(re_list))
  }
  # Cross-parameter correlation groups
  for (cor_id in names(model_data$cross_cor)) {
    groups <- c(groups, model_data$cross_cor[[cor_id]]$group)
  }
  unique(groups)
}


get_all_term_indices <- function(model_data) {
  indices <- list()
  
  param_names <- c("dprime", "sigma", "lambda", "dprime2", "sigma2", "rec_crit", "know_crit",
                   "dprime_B", "lambda_B", "discrim", "discrim_B",
                   "sigma_B", "sigma2_B", "rho", "rho_B", "rho_N",
                   "dprime_L", "sigma_L", "lambda_L", "lambda2", "lambda2_B")
  for (pname in param_names) {
    re_list <- model_data[[paste0(pname, "_random")]]
    if (is.null(re_list)) next
    for (group in names(re_list)) {
      re <- re_list[[group]]
      # Only include if using index approach (not Z matrix)
      if (!is.null(re$term_idx) && !isTRUE(re$use_z_matrix)) {
        idx_name <- paste0("idx_", pname, "_", group)
        dim_name <- paste0("D_idx_", pname, "_", group)
        indices[[idx_name]] <- list(idx_name = idx_name, dim_name = dim_name, dim = re$dim)
      }
    }
  }
  
  for (group in names(model_data$criterion$random)) {
    re <- model_data$criterion$random[[group]]
    # Only include if using index approach (not Z matrix)
    if (!is.null(re$term_idx) && !isTRUE(re$use_z_matrix)) {
      idx_name <- paste0("idx_criterion_", group)
      dim_name <- paste0("D_idx_criterion_", group)
      indices[[idx_name]] <- list(idx_name = idx_name, dim_name = dim_name, dim = re$n_cond_levels)
    }
  }
  
  indices
}


get_all_z_matrices <- function(model_data) {
  z_matrices <- list()

  param_names <- c("dprime", "sigma", "lambda", "dprime2", "sigma2",
                    "rec_crit", "know_crit",
                    "dprime_B", "lambda_B", "discrim", "discrim_B",
                    "sigma_B", "sigma2_B", "rho", "rho_B", "rho_N",
                    "dprime_L", "sigma_L", "lambda_L", "lambda2", "lambda2_B")
  for (pname in param_names) {
    re_list <- model_data[[paste0(pname, "_random")]]
    if (is.null(re_list)) next
    for (group in names(re_list)) {
      re <- re_list[[group]]
      if (isTRUE(re$use_z_matrix)) {
        z_name <- paste0("Z_", pname, "_", group)
        z_matrices[[z_name]] <- list(name = z_name, dim = re$dim, param = pname, group = group)
      }
    }
  }
  
  # Criterion random effects
  for (group in names(model_data$criterion$random)) {
    re <- model_data$criterion$random[[group]]
    if (isTRUE(re$use_z_matrix)) {
      z_name <- paste0("Z_criterion_", group)
      # For criterion, Z matrix has n_re_terms columns (e.g., 2 for intercept + slope)
      z_matrices[[z_name]] <- list(name = z_name, dim = re$n_re_terms, param = "criterion", group = group)
    }
  }
  
  z_matrices
}


# ---- Smooth term Stan code generation helpers ----

#' Iterate over smooth components, calling fn(param_name, comp_suffix, comp, k, n_thresh)
#' for each penalized basis matrix. n_thresh > 0 for criterion smooths (per-threshold).
#' Returns concatenated character vector.
#' @noRd
iter_smooth_components <- function(model_data, fn) {
  lines <- character(0)
  sd <- model_data$smooth_data
  if (is.null(sd)) return(lines)
  for (pname in names(sd)) {
    for (sm in sd[[pname]]) {
      n_thresh <- if (!is.null(sm$n_thresh)) sm$n_thresh else 0L
      for (j in seq_along(sm$components)) {
        comp <- sm$components[[j]]
        comp_suffix <- comp$san_label
        for (k in seq_along(comp$Zs_list)) {
          lines <- c(lines, fn(pname, comp_suffix, comp, k, n_thresh))
        }
      }
    }
  }
  lines
}

#' Generate smooth data block declarations (Zs matrices)
#' @noRd
generate_smooth_data_declarations <- function(model_data) {
  iter_smooth_components(model_data, function(pname, suffix, comp, k, n_thresh) {
    zs_name <- paste0("Zs_", pname, "_", suffix, "_", k)
    c(sprintf("  // Smooth penalized basis: %s component %d", suffix, k),
      sprintf("  int<lower=1> nbasis_%s_%s_%d;", pname, suffix, k),
      sprintf("  matrix[N, nbasis_%s_%s_%d] %s;", pname, suffix, k, zs_name))
  })
}

#' Generate smooth parameter declarations (non-centered)
#' For criterion smooths with n_thresh > 0, generates per-threshold parameters.
#' @noRd
generate_smooth_param_declarations <- function(model_data) {
  iter_smooth_components(model_data, function(pname, suffix, comp, k, n_thresh) {
    nb <- paste0("nbasis_", pname, "_", suffix, "_", k)
    base <- paste0(pname, "_", suffix, "_", k)
    if (n_thresh > 0) {
      # Per-threshold: vector of sds and matrix of zs
      c(sprintf("  vector<lower=0>[%d] sds_%s;  // per-threshold smooth penalty SDs", n_thresh, base),
        sprintf("  matrix[%d, %s] zs_%s;  // per-threshold standardized smooth coefficients", n_thresh, nb, base))
    } else {
      c(sprintf("  real<lower=0> sds_%s;  // smooth penalty SD", base),
        sprintf("  vector[%s] zs_%s;  // standardized smooth coefficients", nb, base))
    }
  })
}

#' Generate smooth transformed parameters (s = sds * zs, non-centered)
#' @noRd
generate_smooth_transformed_params <- function(model_data) {
  iter_smooth_components(model_data, function(pname, suffix, comp, k, n_thresh) {
    base <- paste0(pname, "_", suffix, "_", k)
    nb <- paste0("nbasis_", pname, "_", suffix, "_", k)
    if (n_thresh > 0) {
      # Per-threshold: s is a matrix [n_thresh, nbasis]
      c(sprintf("  matrix[%d, %s] s_%s;", n_thresh, nb, base),
        sprintf("  for (t in 1:%d) s_%s[t, ] = sds_%s[t] * zs_%s[t, ];", n_thresh, base, base, base))
    } else {
      c(sprintf("  vector[%s] s_%s = sds_%s * zs_%s;", nb, base, base, base))
    }
  })
}

#' Generate smooth priors (zs ~ std_normal, sds ~ half-t)
#' @noRd
generate_smooth_priors <- function(model_data) {
  pl <- model_data$prior_lookup
  pl[] <- rapply(pl, to_stan_prior, classes = "character", how = "replace")
  iter_smooth_components(model_data, function(pname, suffix, comp, k, n_thresh) {
    base <- paste0(pname, "_", suffix, "_", k)
    sds_prior <- get_sds_prior(pl, pname, paste0(suffix, "_", k))
    if (n_thresh > 0) {
      c(sprintf("  to_vector(zs_%s) ~ std_normal();", base),
        sprintf("  sds_%s ~ %s;", base, sds_prior))
    } else {
      c(sprintf("  zs_%s ~ std_normal();", base),
        sprintf("  sds_%s ~ %s;", base, sds_prior))
    }
  })
}


#' @export
print.stan_code <- function(x, ...) {
  cat(x)
  invisible(x)
}
