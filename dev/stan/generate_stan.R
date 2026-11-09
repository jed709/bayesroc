#' Generate Stan Code for SDT Models
#'
#' Supports multiple families: EVSDT, UVSDT, DPSDT, Mixture
#' @noRd

generate_stan_code_v2 <- function(model_data, family, batch_info = NULL) {

  blocks <- .ls_sum_zero_blocks(list(
    functions = generate_functions_block_v2(model_data, family, batch_info = batch_info),
    data = generate_data_block_v2(model_data, family, batch_info = batch_info),
    tdata = generate_transformed_data_block_v2(model_data, family),
    params = generate_parameters_block_v2(model_data, family),
    tparams = generate_transformed_parameters_block_v2(model_data, family),
    model = generate_model_block_v2(model_data, family, batch_info = batch_info),
    gq = generate_generated_quantities_block_v2(model_data, family, batch_info = batch_info)
  ), model_data)
  code <- paste(unlist(blocks, use.names = FALSE), collapse = "\n")

  if (identical(model_data$gap_link, "softplus")) {
    code <- gsub("beta_log_gaps", "beta_raw_gaps", code)
    code <- gsub("(?<![A-Za-z0-9_])exp\\(beta_raw_gaps", "log1p_exp(beta_raw_gaps", code, perl = TRUE)
    code <- gsub("log_gaps_prior", "raw_gaps_prior", code)
    code <- gsub("Log-gaps", "Raw-gaps (softplus)", code)
    code <- gsub("exp(log_gap)", "log1p_exp(raw_gap)", code, fixed = TRUE)
    code <- gsub("real log_gap", "real raw_gap", code, fixed = TRUE)
    code <- gsub("log_gap +=", "raw_gap +=", code, fixed = TRUE)
    code <- gsub("log_gap;", "raw_gap;", code, fixed = TRUE)
  }

  class(code) <- c("stan_code", "character")
  code
}

generate_functions_block_v2 <- function(model_data, family, batch_info = NULL) {
  if (!is.null(batch_info)) {
    lines <- c("functions {", "  // Batch C++ likelihood (implemented in user header)",
               batch_info$stan_decl, batch_info$stan_decl_vec)
    if (isTRUE(model_data$threads)) lines <- c(lines, "", batch_info$stan_partial)
    return(paste(c(lines, "}"), collapse = "\n"))
  }
  .plain_stan_dev(".plain_functions_block")(model_data, family)
}

.plain_stan_env <- function() {
  env <- getOption("bayesroc.dev_plain_stan")
  if (is.environment(env)) env else NULL
}

.plain_stan_dev <- function(name) {
  env <- .plain_stan_env()
  f <- if (!is.null(env)) get0(name, envir = env, inherits = FALSE)
  if (is.null(f)) {
    stop("Internal error: no compiled likelihood was generated for this model.", call. = FALSE)
  }
  f
}

generate_data_block_v2 <- function(model_data, family, batch_info = NULL) {
  family_name <- if (is.list(family)) family$family else family
  is_cumulative <- family_name == "cumulative"
  is_source_mixture <- family_name == "source_mixture"
  is_bivariate <- family_name %in% c("bivariate_sdt", "bivariate_dp", "bivariate_cumulative")
  is_cdp <- family_name == "cdp"
  is_vrdp2d <- family_name == "vrdp2d"

  if (is_bivariate || is_vrdp2d) {
    lines <- c(
      "data {",
      "  int<lower=1> N;",
      "  int<lower=0, upper=N> N_ll;",
      "  int<lower=0, upper=1> prior_only;",
      "  int<lower=2> K1;  // Item response categories",
      "  int<lower=2> K2;  // Source response categories",
      "  array[N] int<lower=1, upper=K1> y;   // Item responses",
      "  array[N] int<lower=1, upper=K2> y2;  // Source responses",
      "  array[N] int<lower=1, upper=3> item_type;  // 1=new, 2=A, 3=B",
      if (isTRUE(model_data$varying_source_criteria))
        c("  int<lower=1, upper=K1> n_vbin;  // sets of varying source criteria",
          "  array[K1] int<lower=1, upper=n_vbin> vbin;  // set read by each detection response",
          "  int<lower=1> n_vcell;",
          "  int<lower=1> n_vset;",
          "  array[N] int<lower=1, upper=n_vcell> vcell;")
    )
  } else if (is_cdp) {
    J <- model_data$stan_data$J
    n_new <- model_data$stan_data$n_new
    lines <- c(
      "data {",
      "  int<lower=1> N;",
      "  int<lower=0, upper=N> N_ll;",
      "  int<lower=0, upper=1> prior_only;",
      "  int<lower=2> K;  // Total confidence levels",
      sprintf("  int<lower=1> J;  // Number of old confidence levels (=%d)", J),
      sprintf("  int<lower=1> n_new;  // Number of new confidence levels (=%d)", n_new),
      "  array[N] int<lower=1, upper=K> y;  // Confidence response",
      sprintf("  int<lower=2, upper=3> n_rkg;  // 2=R/K, 3=R/K/G (=%d)", model_data$stan_data$n_rkg),
      "  array[N] int<lower=0, upper=n_rkg> rk;  // 1=R, 2=K, (3=G), 0=no judgement",
      "  array[N] real is_old;  // 0=lure, 1=target (or centered: -0.5/0.5)",
      "  array[J] int<lower=1> old_level_map;  // Maps 1:J to actual confidence levels",
      "  array[n_new] int<lower=1> new_level_map;  // Maps 1:n_new to actual confidence levels"
    )
  } else {
    lines <- c(
      "data {",
      "  int<lower=1> N;",
      "  int<lower=0, upper=N> N_ll;",
      "  int<lower=0, upper=1> prior_only;",
      "  int<lower=2> K;",
      "  array[N] int<lower=1, upper=K> y;"
    )
    
    if (is_source_mixture) {
      lines <- c(lines, "  array[N] int<lower=0, upper=1> source;  // 0 = A, 1 = B")
    } else if (family_name == "mafc") {
      lines <- c(lines, "  array[N] int<lower=1, upper=K> target_pos;  // target position",
                 "  array[N] int<lower=2, upper=K> m_alt;  // alternatives on the trial")
    } else if (!is_cumulative) {
      lines <- c(lines, "  array[N] real is_old;")
    }
  }
  
  if (isTRUE(model_data$has_counts)) {
    lines <- c(lines, "  array[N] int<lower=0> counts;  // Aggregation counts")
  }
  if (!is.null(model_data$stan_data$src_obs)) {
    lines <- c(lines, "  array[N] int<lower=0, upper=1> src_obs;")
  }

  has_dprime_fixed <- isTRUE(model_data$has_dprime_fixed) || is.null(model_data$has_dprime_fixed)
  if (has_dprime_fixed) {
    lines <- c(lines, "",
               "  // d' fixed effects design matrix",
               "  int<lower=1> P_dprime;",
               "  matrix[N, P_dprime] X_dprime;"
    )
  }
  
  if (family_name == "mafc") {
    lines <- c(lines, .bias_stan(model_data, "data"))
  } else {
  lines <- c(lines, "",
             "  // Criterion fixed effects design matrix", 
             "  int<lower=1> P_criterion;",
             "  matrix[N, P_criterion] X_criterion;"
  )
  }
  
  if (model_data$has_sigma) {
    lines <- c(lines, "", "  int<lower=1> P_sigma;", "  matrix[N, P_sigma] X_sigma;")
  }
  if (model_data$has_lambda) {
    lines <- c(lines, "", "  int<lower=1> P_lambda;", "  matrix[N, P_lambda] X_lambda;")
  }
  if (model_data$has_dprime2) {
    lines <- c(lines, "", "  int<lower=1> P_dprime2;", "  matrix[N, P_dprime2] X_dprime2;")
  }
  if (isTRUE(model_data$needs_ordered_dprime)) {
    lines <- c(lines, "  int<lower=1, upper=P_dprime> dprime_pivot;", "  vector[P_dprime] dprime_xbar;",
               "  vector[P_dprime2] dprime2_xbar;")
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
  if (identical(model_data$family, "metadt")) {
    lines <- c(lines, "  real is_old_mid;  // midpoint of the old/new coding")
  }
  if (isTRUE(model_data$has_mratio)) {
    lines <- c(lines, "", "  int<lower=1> P_mratio;", "  matrix[N, P_mratio] X_mratio;")
  }
  if (isTRUE(model_data$has_mratio_N)) {
    lines <- c(lines, "", "  int<lower=1> P_mratio_N;", "  matrix[N, P_mratio_N] X_mratio_N;")
  }
  if (isTRUE(model_data$has_sigma_meta)) {
    lines <- c(lines, "", "  int<lower=1> P_sigma_meta;", "  matrix[N, P_sigma_meta] X_sigma_meta;")
  }
  if (isTRUE(model_data$has_rec_crit)) {
    lines <- c(lines, "", "  // rec_crit (recollection criterion) fixed effects",
               "  int<lower=1> P_rec_crit;", "  matrix[N, P_rec_crit] X_rec_crit;")
  }
  if (isTRUE(model_data$has_know_crit)) {
    lines <- c(lines, "", "  // know_crit (know criterion) fixed effects",
               "  int<lower=1> P_know_crit;", "  matrix[N, P_know_crit] X_know_crit;")
    if (isTRUE(model_data$stan_data$kc_knob == 1)) {
      lines <- c(lines,
                 "  int<lower=0, upper=1> kc_knob;",
                 "  int<lower=1> N_guess;",
                 "  array[N_guess] int<lower=1, upper=N> guess_idx;",
                 "  array[N_guess] int<lower=1, upper=J> guess_band;",
                 "  vector[P_know_crit] kc_v;",
                 "  matrix[P_know_crit, P_know_crit - 1] kc_Bperp;")
    }
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
  if (.ls_scale_coefs(model_data) > 0) {
    lines <- c(lines, "", "  int<lower=1> P_criterion_scale;", "  matrix[N, P_criterion_scale] X_criterion_scale;")
  }
  if (.ls_scale_coefs(model_data, 2L) > 0) {
    lines <- c(lines, "", "  int<lower=1> P_criterion2_scale;", "  matrix[N, P_criterion2_scale] X_criterion2_scale;")
  }
  if (isTRUE(model_data$has_criterion2)) {
    lines <- c(lines, "", "  // Criterion2 (discrimination) fixed effects design matrix",
               "  int<lower=1> P_criterion2;", "  matrix[N, P_criterion2] X_criterion2;")
    if (isTRUE(model_data$varying_source_criteria)) {
      lines <- c(lines, "  int<lower=0> P_criterion2_cov;",
                 "  matrix[N, P_criterion2_cov] X_criterion2_cov;")
    }
  }
  
  all_groups <- get_all_groups(model_data)
  if (length(all_groups) > 0) {
    lines <- c(lines, "", "  // Random effect grouping factors")
    for (group in all_groups) {
      lines <- c(lines, 
                 sprintf("  int<lower=1> N_%s;", group),
                 sprintf("  array[N] int<lower=1, upper=N_%s> %s;", group, group))
    }
  }
  
  term_indices <- get_all_term_indices(model_data)
  if (length(term_indices) > 0) {
    lines <- c(lines, "", "  // Term indices for varying slopes")
    for (ti in term_indices) {
      lines <- c(lines,
                 sprintf("  int<lower=1> %s;", ti$dim_name),
                 sprintf("  array[N] int<lower=0, upper=%s> %s;", ti$dim_name, ti$idx_name))
    }
  }
  
  z_matrices <- get_all_z_matrices(model_data)
  if (length(z_matrices) > 0) {
    lines <- c(lines, "", "  // Random effect design matrices")
    for (zm in z_matrices) {
      lines <- c(lines, sprintf("  matrix[N, %d] %s;", zm$dim, zm$name))
    }
  }
  
  if (isTRUE(model_data$threads)) {
    lines <- c(lines, "", "  // Threading (reduce_sum)",
               "  int grainsize;")
  }

  lines <- c(lines, generate_smooth_data_declarations(model_data))

  lines <- c(lines, "}")
  paste(lines, collapse = "\n")
}

generate_transformed_data_block_v2 <- function(model_data, family) {
  lines <- c("transformed data {")
  
  param_names <- c("dprime", "sigma", "lambda", "dprime2", "sigma2", "dprime_B", "lambda_B",
                   "discrim", "discrim_B", "sigma_B", "sigma2_B", "rho", "rho_B", "rho_N", "mratio", "mratio_N", "sigma_meta", "rec_crit", "know_crit",
                   "dprime_L", "sigma_L", "lambda_L", "lambda2", "lambda2_B", "criterion_scale", "criterion2_scale")
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
  
  if (isTRUE(model_data$has_criterion2) && !is.null(model_data$criterion2$random)) {
    for (group in names(model_data$criterion2$random)) {
      re <- model_data$criterion2$random[[group]]
      if (is.null(re$cor_id)) {
        lines <- c(lines, sprintf("  int D_criterion2_%s = %d;", group, re$dim))
      }
    }
  }
  
  lines <- c(lines, .bias_stan(model_data, "transformed_data"))

  for (cor_id in names(model_data$cross_cor)) {
    cc <- model_data$cross_cor[[cor_id]]
    lines <- c(lines, sprintf("  int D_cross_%s_%s = %d;", cor_id, cc$group, cc$total_dim))
  }

  family_name_td <- if (is.list(family)) family$family else family
  is_bivariate_td <- family_name_td %in% c("bivariate_sdt", "bivariate_dp", "bivariate_cumulative")
  is_vrdp2d_td <- family_name_td == "vrdp2d"
  is_cdp_td <- family_name_td == "cdp"

  if (is_bivariate_td || is_vrdp2d_td) {
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

  if (isTRUE(model_data$has_criterion_scale) || isTRUE(model_data$has_criterion2_scale)) {
    J_ls <- if (is_cdp_td) model_data$stan_data$J else model_data$K - 1L
    lines <- c(lines,
               "  matrix[N, 1] X_ls_one = rep_matrix(1, N, 1);",
               "  vector[1] ls_zero_mid = rep_vector(0, 1);",
               if (isTRUE(model_data$has_criterion_scale))
                 sprintf("  matrix[%d, 1] ls_zero_gaps = rep_matrix(0, %d, 1);", J_ls - 1L, J_ls - 1L),
               if (isTRUE(model_data$has_criterion2_scale))
                 sprintf("  matrix[%d, 1] ls_zero_gaps2 = rep_matrix(0, %d, 1);", model_data$K2 - 2L, model_data$K2 - 2L),
               "  array[N] int ls_row_idx = linspaced_int_array(N, 1, N);")
  }

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
  
  if (model_data$needs_ordered_dprime) {
    lines <- c(lines,
               "  // dprime > dprime2 on average over the old items' design rows",
               "  vector[P_dprime - 1] beta_dprime_free;",
               "  vector[P_dprime2] beta_dprime2;",
               "  real<lower=0> dprime_order_gap;")
  } else if (has_dprime_fixed) {
    lines <- c(lines, "  vector[P_dprime] beta_dprime;")
    if (model_data$has_dprime2) {
      lines <- c(lines, "  vector[P_dprime2] beta_dprime2;")
    }
  }
  
  family_name <- if (is.list(family)) family$family else family
  is_cdp <- family_name == "cdp"
  
  gap_cols <- if (isTRUE(model_data$has_criterion_scale)) "1" else "P_criterion"
  if (is_bivariate || is_vrdp2d) {
    lines <- c(lines,
               "  vector[P_criterion] beta_thresh_mid;",
               sprintf("  matrix[K1-2, %s] beta_log_gaps;", gap_cols))
  } else if (is_cdp) {
    lines <- c(lines,
               "  vector[P_criterion] beta_thresh_mid;  // Mid-anchor tau threshold",
               sprintf("  matrix[J-1, %s] beta_log_gaps;  // Log-gaps for tau", gap_cols))
  } else if (family_name == "mafc") {
    lines <- c(lines, .bias_stan(model_data, "parameters"))
  } else {
    lines <- c(lines, "  vector[P_criterion] beta_thresh_mid;")
    if (model_data$K > 2) {
      lines <- c(lines, sprintf("  matrix[K-2, %s] beta_log_gaps;", gap_cols))
    }
  }
  if (.ls_scale_coefs(model_data) > 0) lines <- c(lines, "  vector[P_criterion_scale] beta_criterion_scale;")
  if (.ls_scale_coefs(model_data, 2L) > 0) lines <- c(lines, "  vector[P_criterion2_scale] beta_criterion2_scale;")
  
  if (model_data$has_sigma) lines <- c(lines, "  vector[P_sigma] beta_sigma;")
  if (model_data$has_lambda) lines <- c(lines, "  vector[P_lambda] beta_lambda;")
  if (model_data$has_sigma2) lines <- c(lines, "  vector[P_sigma2] beta_sigma2;")
  if (isTRUE(model_data$has_dprime_B)) lines <- c(lines, "  vector[P_dprime_B] beta_dprime_B;")
  if (isTRUE(model_data$has_lambda_B)) lines <- c(lines, "  vector[P_lambda_B] beta_lambda_B;")
  
  if (isTRUE(model_data$has_discrim) && has_discrim_fixed) lines <- c(lines, "  vector[P_discrim] beta_discrim;")
  if (isTRUE(model_data$has_discrim_B)) lines <- c(lines, "  vector[P_discrim_B] beta_discrim_B;")
  if (isTRUE(model_data$has_sigma_B)) lines <- c(lines, "  vector[P_sigma_B] beta_sigma_B;")
  if (isTRUE(model_data$has_sigma2_B)) lines <- c(lines, "  vector[P_sigma2_B] beta_sigma2_B;")
  if (isTRUE(model_data$has_rho)) lines <- c(lines, "  vector[P_rho] beta_rho;  // Fisher z scale")
  if (isTRUE(model_data$has_rho_B)) lines <- c(lines, "  vector[P_rho_B] beta_rho_B;  // Fisher z scale")
  if (isTRUE(model_data$has_rho_N)) lines <- c(lines, "  vector[P_rho_N] beta_rho_N;  // Fisher z scale")
  if (isTRUE(model_data$has_mratio)) lines <- c(lines, "  vector[P_mratio] beta_mratio;  // log scale")
  if (isTRUE(model_data$has_mratio_N)) lines <- c(lines, "  vector[P_mratio_N] beta_mratio_N;  // log scale")
  if (isTRUE(model_data$has_sigma_meta)) lines <- c(lines, "  vector[P_sigma_meta] beta_sigma_meta;  // log scale")
  if (isTRUE(model_data$has_lambda2)) lines <- c(lines, "  vector[P_lambda2] beta_lambda2;  // R_S source recollection (logit scale)")
  if (isTRUE(model_data$has_lambda2_B)) lines <- c(lines, "  vector[P_lambda2_B] beta_lambda2_B;  // R_S_B source recollection for Source B (logit scale)")
  if (isTRUE(model_data$has_rec_crit)) {
    lines <- c(lines, "  vector[P_rec_crit] beta_rec_crit;  // Recollection criterion")
  }
  if (isTRUE(model_data$has_know_crit)) {
    if (isTRUE(model_data$stan_data$kc_knob == 1)) {
      lines <- c(lines, "  vector[P_know_crit - 1] know_crit_free;", "  real know_crit_slack;")
    } else {
      lines <- c(lines, "  vector[P_know_crit] beta_know_crit;  // Know criterion (R/K/G)")
    }
  }
  if (isTRUE(model_data$has_dprime_L)) lines <- c(lines, "  vector[P_dprime_L] beta_dprime_L;  // Lure d'")
  if (isTRUE(model_data$has_sigma_L)) lines <- c(lines, "  vector[P_sigma_L] beta_sigma_L;  // Lure sigma (log scale)")
  if (isTRUE(model_data$has_lambda_L)) lines <- c(lines, "  vector[P_lambda_L] beta_lambda_L;  // Lure mixture prob (logit scale)")
  if (isTRUE(model_data$has_criterion2)) {
    if (isTRUE(model_data$varying_source_criteria)) {
      lines <- c(lines,
                 "  vector[n_vset] beta_thresh_mid_2_varying;  // Mid-anchor source threshold of each set",
                 "  matrix[n_vset, K2-2] beta_log_gaps_2_varying;  // Log-gaps of each set")
      if (.has_crit2_cov(model_data)) {
        lines <- c(lines, "  vector[P_criterion2_cov] beta_thresh_mid_2_cov;",
                   if (!isTRUE(model_data$has_criterion2_scale)) "  matrix[K2-2, P_criterion2_cov] beta_log_gaps_2_cov;")
      }
    } else {
      lines <- c(lines,
                 "  vector[P_criterion2] beta_thresh_mid_2;  // Mid-anchor threshold for dimension 2",
                 sprintf("  matrix[K2-2, %s] beta_log_gaps_2;  // Log-gaps for dimension 2",
                         if (isTRUE(model_data$has_criterion2_scale)) "1" else "P_criterion2"))
    }
  }
  
  param_names <- c("dprime", "sigma", "lambda", "dprime2", "sigma2", "dprime_B", "lambda_B",
                   "discrim", "discrim_B", "sigma_B", "sigma2_B", "rho", "rho_B", "rho_N", "mratio", "mratio_N", "sigma_meta", "rec_crit", "know_crit",
                   "dprime_L", "sigma_L", "lambda_L", "lambda2", "lambda2_B", "criterion_scale", "criterion2_scale")
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
  
  for (group in names(model_data$criterion$random)) {
    re <- model_data$criterion$random[[group]]
    if (is.null(re$cor_id)) {
      dim_var <- sprintf("D_criterion_%s", group)
      lines <- c(lines,
                 sprintf("  matrix[N_%s, %s] u_criterion_%s_raw;", group, dim_var, group),
                 sprintf("  vector<lower=0>[%s] sigma_criterion_%s;", dim_var, group),
                 .re_corr_decl(paste0("criterion_", group), re))
    }
  }
  
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
                     sprintf("  vector<lower=0>[%s] sigma_criterion2_%s;", dim_var, group),
                     .re_corr_decl(paste0("criterion2_", group), re))
        }
      }
    }
  }
  
  for (cor_id in names(model_data$cross_cor)) {
    cc <- model_data$cross_cor[[cor_id]]
    dim_var <- sprintf("D_cross_%s_%s", cor_id, cc$group)
    lines <- c(lines,
               sprintf("  matrix[N_%s, %s] u_cross_%s_%s_raw;", cc$group, dim_var, cor_id, cc$group),
               sprintf("  vector<lower=0>[%s] sigma_cross_%s_%s;", dim_var, cor_id, cc$group),
               sprintf("  cholesky_factor_corr[%s] L_corr_cross_%s_%s;", dim_var, cor_id, cc$group))
  }

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
    c(lines, .re_corr_decl(paste0(param_name, "_", group), re))
  }
}

generate_transformed_parameters_block_v2 <- function(model_data, family) {
  lines <- c("transformed parameters {")
  
  if (model_data$needs_ordered_dprime) {
    lines <- c(lines,
               "  vector[P_dprime] beta_dprime;",
               "  {",
               "    int k = 1;",
               "    for (p in 1:P_dprime) {",
               "      if (p == dprime_pivot) {",
               "        beta_dprime[p] = 0;",
               "      } else {",
               "        beta_dprime[p] = beta_dprime_free[k];",
               "        k += 1;",
               "      }",
               "    }",
               "    beta_dprime[dprime_pivot] = (dprime_order_gap + dot_product(dprime2_xbar, beta_dprime2)",
               "                                 - dot_product(dprime_xbar, beta_dprime)) / dprime_xbar[dprime_pivot];",
               "  }",
               "")
  }
  
  
  param_names <- c("dprime", "sigma", "lambda", "dprime2", "sigma2", "dprime_B", "lambda_B",
                   "discrim", "discrim_B", "sigma_B", "sigma2_B", "rho", "rho_B", "rho_N", "mratio", "mratio_N", "sigma_meta", "rec_crit", "know_crit",
                   "dprime_L", "sigma_L", "lambda_L", "lambda2", "lambda2_B", "criterion_scale", "criterion2_scale")
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
  
  lines <- c(lines, .bias_stan(model_data, "transformed_parameters"))

  for (group in names(model_data$criterion$random)) {
    re <- model_data$criterion$random[[group]]
    if (is.null(re$cor_id)) {
      dim_var <- sprintf("D_criterion_%s", group)
      if (isTRUE(re$correlated)) {
        lines <- c(lines, .re_corr_assemble(paste0("criterion_", group), re), sprintf(
          "  matrix[N_%s, %s] u_criterion_%s = u_criterion_%s_raw * diag_pre_multiply(sigma_criterion_%s, L_corr_criterion_%s)';",
          group, dim_var, group, group, group, group))
      } else {
        lines <- c(lines, sprintf(
          "  matrix[N_%s, %s] u_criterion_%s;", group, dim_var, group))
        lines <- c(lines, sprintf(
          "  for (d in 1:%s) u_criterion_%s[, d] = u_criterion_%s_raw[, d] * sigma_criterion_%s[d];",
          dim_var, group, group, group))
      }
    }
  }
  
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
          if (isTRUE(re$correlated)) {
            lines <- c(lines, .re_corr_assemble(paste0("criterion2_", group), re), sprintf(
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
    for (member in cc$members) {
      lines <- c(lines, if (member$dim == 1)
        sprintf("  vector[N_%s] u_%s_%s_from_%s;", group, member$param, group, cor_id)
      else sprintf("  matrix[N_%s, %d] u_%s_%s_from_%s;", group, member$dim, member$param, group, cor_id))
    }
    lines <- c(lines, "  {", sprintf(
      "    matrix[N_%s, %s] u_cross = u_cross_%s_%s_raw * diag_pre_multiply(sigma_cross_%s_%s, L_corr_cross_%s_%s)';",
      group, dim_var, cor_id, group, cor_id, group, cor_id, group))
    col_start <- 1
    for (member in cc$members) {
      col_end <- col_start + member$dim - 1
      lines <- c(lines, if (member$dim == 1)
        sprintf("    u_%s_%s_from_%s = u_cross[, %d];", member$param, group, cor_id, col_start)
      else sprintf("    u_%s_%s_from_%s = u_cross[, %d:%d];", member$param, group, cor_id, col_start, col_end))
      col_start <- col_end + 1
    }
    lines <- c(lines, "  }")
  }

  lines <- c(lines, generate_smooth_transformed_params(model_data))
  if (isTRUE(model_data$stan_data$kc_knob == 1)) lines <- c(lines, generate_know_crit_knob(model_data))

  lines <- c(lines, "}")
  paste(lines, collapse = "\n")
}

generate_know_crit_knob <- function(model_data) {
  kc_lines <- generate_param_lp("know_crit", model_data$know_crit_fixed, model_data$know_crit_random,
                                model_data$smooth_data[["know_crit"]])
  kc_lines <- gsub("beta_know_crit", "beta_kc_perp", kc_lines, fixed = TRUE)
  rc_lines <- if (isTRUE(model_data$has_rec_crit)) {
    generate_param_lp("rec_crit", model_data$rec_crit_fixed, model_data$rec_crit_random,
                      model_data$smooth_data[["rec_crit"]])
  } else {
    "    real rec_crit_n = 1.5;"
  }
  c("  vector[P_know_crit] beta_know_crit;",
    "  {",
    "    vector[P_know_crit] beta_kc_perp = kc_Bperp * know_crit_free;",
    "    real kc_t = negative_infinity();",
    "    for (g in 1:N_guess) {",
    "    int n = guess_idx[g];",
    rc_lines,
    kc_lines,
    generate_thresh_code_cdp(model_data),
    "    kc_t = fmax(kc_t, tau[guess_band[g]] - rec_crit_n - know_crit_n);",
    "    }",
    "    beta_know_crit = beta_kc_perp + (kc_t + exp(know_crit_slack)) * kc_v;",
    "  }")
}

generate_re_transform_v2 <- function(param_name, group, re) {
  if (re$dim == 1) {
    sprintf("  vector[N_%s] u_%s_%s = u_%s_%s_raw * sigma_%s_%s;",
            group, param_name, group, param_name, group, param_name, group)
  } else if (isTRUE(re$correlated)) {
    dim_var <- sprintf("D_%s_%s", param_name, group)
    c(.re_corr_assemble(paste0(param_name, "_", group), re),
      sprintf("  matrix[N_%s, %s] u_%s_%s = u_%s_%s_raw * diag_pre_multiply(sigma_%s_%s, L_corr_%s_%s)';",
              group, dim_var, param_name, group, param_name, group, param_name, group, param_name, group))
  } else {
    dim_var <- sprintf("D_%s_%s", param_name, group)
    c(sprintf("  matrix[N_%s, %s] u_%s_%s;", group, dim_var, param_name, group),
      sprintf("  for (d in 1:%s) u_%s_%s[, d] = u_%s_%s_raw[, d] * sigma_%s_%s[d];",
              dim_var, param_name, group, param_name, group, param_name, group))
  }
}


.re_corr_decl <- function(prefix, re) {
  if (!isTRUE(re$correlated)) return(character(0))
  if (.re_full_block(re)) return(sprintf("  cholesky_factor_corr[D_%s] L_corr_%s;", prefix, prefix))
  vapply(seq_along(re$cor_blocks), function(b)
    sprintf("  cholesky_factor_corr[%d] L_corr_%s_b%d;", length(re$cor_blocks[[b]]), prefix, b),
    character(1))
}

.re_corr_assemble <- function(prefix, re) {
  if (!isTRUE(re$correlated) || .re_full_block(re)) return(character(0))
  c(sprintf("  matrix[D_%s, D_%s] L_corr_%s = diag_matrix(rep_vector(1, D_%s));",
            prefix, prefix, prefix, prefix),
    vapply(seq_along(re$cor_blocks), function(b) {
      ix <- paste(re$cor_blocks[[b]], collapse = ", ")
      sprintf("  L_corr_%s[{%s}, {%s}] = L_corr_%s_b%d;", prefix, ix, ix, prefix, b)
    }, character(1)))
}

.re_corr_prior <- function(prefix, re, cor_prior) {
  if (!isTRUE(re$correlated)) return(character(0))
  if (.re_full_block(re)) return(sprintf("  L_corr_%s ~ %s;", prefix, cor_prior))
  vapply(seq_along(re$cor_blocks), function(b) sprintf("  L_corr_%s_b%d ~ %s;", prefix, b, cor_prior),
         character(1))
}

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
  pl <- model_data$prior_lookup
  pl[] <- rapply(pl, to_stan_prior, classes = "character", how = "replace")

  lines <- c("model {")
  if (isTRUE(model_data$stan_data$kc_knob == 1)) lines <- c(lines, "  target += know_crit_slack;")

  generate_fixed_priors <- function(dpar, coef_names, beta_name) {
    if (length(coef_names) == 0) return(sprintf("  %s ~ %s;", beta_name, get_fixed_prior(pl, dpar)))
    .stan_prior_vec(beta_name, vapply(coef_names, function(cf) get_fixed_prior(pl, dpar, cf),
                                      character(1), USE.NAMES = FALSE))
  }
  
  has_dprime_fixed <- isTRUE(model_data$has_dprime_fixed) || is.null(model_data$has_dprime_fixed)
  
  if (model_data$needs_ordered_dprime) {
    lines <- c(lines, generate_fixed_priors("dprime", model_data$dprime_fixed$coef_names, "beta_dprime"),
               generate_fixed_priors("dprime2", model_data$dprime2_fixed$coef_names, "beta_dprime2"))
  } else if (has_dprime_fixed) {
    dprime_coefs <- model_data$dprime_fixed$coef_names
    lines <- c(lines, generate_fixed_priors("dprime", dprime_coefs, "beta_dprime"))
    
    if (model_data$has_dprime2) {
      dprime2_coefs <- model_data$dprime2_fixed$coef_names
      lines <- c(lines, generate_fixed_priors("dprime2", dprime2_coefs, "beta_dprime2"))
    }
  }
  
  if (family_name == "mafc") {
    lines <- c(lines, .bias_stan(model_data, "model", pl))
  } else {
  tg <- .thresh_prior_grid(pl, "thresh_mid", "log_gaps", model_data$criterion$coef_names, model_data$K - 2L)
  if (isTRUE(model_data$has_criterion_scale)) tg$gaps <- tg$gaps[, 1, drop = FALSE]
  lines <- c(lines, .stan_prior_vec("beta_thresh_mid", tg$mid), .stan_prior_mat("beta_log_gaps", tg$gaps))
  }
  if (.ls_scale_coefs(model_data) > 0) {
    lines <- c(lines, generate_fixed_priors("criterion_scale", model_data$criterion_scale_fixed$coef_names,
                                            "beta_criterion_scale"))
  }
  if (.ls_scale_coefs(model_data, 2L) > 0) {
    lines <- c(lines, generate_fixed_priors("criterion2_scale", model_data$criterion2_scale_fixed$coef_names,
                                            "beta_criterion2_scale"))
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
  if (isTRUE(model_data$has_mratio)) {
    lines <- c(lines, generate_fixed_priors("mratio", model_data$mratio_fixed$coef_names, "beta_mratio"))
  }
  if (isTRUE(model_data$has_mratio_N)) {
    lines <- c(lines, generate_fixed_priors("mratio_N", model_data$mratio_N_fixed$coef_names, "beta_mratio_N"))
  }
  if (isTRUE(model_data$has_sigma_meta)) {
    lines <- c(lines, generate_fixed_priors("sigma_meta", model_data$sigma_meta_fixed$coef_names, "beta_sigma_meta"))
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
    if (isTRUE(model_data$varying_source_criteria)) {
      vg <- .crit2_vary_prior_grid(pl, model_data)
      lines <- c(lines, .stan_prior_vec("beta_thresh_mid_2_varying", vg$mid),
                 .stan_prior_mat("beta_log_gaps_2_varying", t(vg$gaps)))
      if (.has_crit2_cov(model_data)) {
        lines <- c(lines, .stan_prior_vec("beta_thresh_mid_2_cov", vg$cov_mid),
                   if (!isTRUE(model_data$has_criterion2_scale)) .stan_prior_mat("beta_log_gaps_2_cov", vg$cov_gaps))
      }
    } else {
      tg2 <- .thresh_prior_grid(pl, "thresh_mid2", "log_gaps2", model_data$criterion2$coef_names,
                                model_data$K2 - 2L)
      if (isTRUE(model_data$has_criterion2_scale)) tg2$gaps <- tg2$gaps[, 1, drop = FALSE]
      lines <- c(lines, .stan_prior_vec("beta_thresh_mid_2", tg2$mid),
                 .stan_prior_mat("beta_log_gaps_2", tg2$gaps))
    }
  }
  
  param_names <- c("dprime", "sigma", "lambda", "dprime2", "sigma2", "dprime_B", "lambda_B",
                   "discrim", "discrim_B", "sigma_B", "sigma2_B", "rho", "rho_B", "rho_N", "mratio", "mratio_N", "sigma_meta", "rec_crit", "know_crit",
                   "dprime_L", "sigma_L", "lambda_L", "lambda2", "lambda2_B", "criterion_scale", "criterion2_scale")
  for (pname in param_names) {
    re_list <- model_data[[paste0(pname, "_random")]]
    if (is.null(re_list)) next
    
    for (group in names(re_list)) {
      re <- re_list[[group]]
      if (is.null(re$cor_id)) {
        cor_prior <- get_cor_prior(pl, pname, group)
        
        level_names <- re$level_names
        if (is.null(level_names) || length(level_names) == 0) {
          level_names <- "(Intercept)"
        }
        
        coef_priors <- sapply(level_names, function(coef) get_sd_prior(pl, pname, group, coef))
        
        if (length(unique(coef_priors)) == 1) {
          lines <- c(lines, sprintf("  sigma_%s_%s ~ %s;", pname, group, coef_priors[1]))
        } else {
          for (i in seq_along(coef_priors)) {
            lines <- c(lines, sprintf("  sigma_%s_%s[%d] ~ %s;", pname, group, i, coef_priors[i]))
          }
        }
        
        if (re$dim == 1) {
          lines <- c(lines, sprintf("  u_%s_%s_raw ~ std_normal();", pname, group))
        } else {
          lines <- c(lines, 
                     sprintf("  to_vector(u_%s_%s_raw) ~ std_normal();", pname, group))
          lines <- c(lines, .re_corr_prior(paste0(pname, "_", group), re, cor_prior))
        }
      }
    }
  }
  
  for (group in names(model_data$criterion$random)) {
    re <- model_data$criterion$random[[group]]
    if (is.null(re$cor_id)) {
      cor_prior <- get_cor_prior(pl, "criterion", group)
      coef_priors <- .crit_re_sd_priors(pl, "criterion", group, re)

      if (length(unique(coef_priors)) == 1) {
        lines <- c(lines, sprintf("  sigma_criterion_%s ~ %s;", group, coef_priors[1]))
      } else {
        for (i in seq_along(coef_priors)) {
          lines <- c(lines, sprintf("  sigma_criterion_%s[%d] ~ %s;", group, i, coef_priors[i]))
        }
      }
      
      lines <- c(lines,
                 sprintf("  to_vector(u_criterion_%s_raw) ~ std_normal();", group))
      lines <- c(lines, .re_corr_prior(paste0("criterion_", group), re, cor_prior))
    }
  }
  
  if (isTRUE(model_data$has_criterion2) && !is.null(model_data$criterion2$random)) {
    for (group in names(model_data$criterion2$random)) {
      re <- model_data$criterion2$random[[group]]
      if (is.null(re$cor_id)) {
        coef_priors <- .crit_re_sd_priors(pl, "criterion2", group, re)
        cor_prior <- get_cor_prior(pl, "criterion2", group)

        if (length(unique(coef_priors)) == 1) {
          lines <- c(lines, sprintf("  sigma_criterion2_%s ~ %s;", group, coef_priors[1]))
        } else {
          lines <- c(lines, sprintf("  sigma_criterion2_%s[%d] ~ %s;", group, seq_along(coef_priors), coef_priors))
        }
        if (re$dim == 1) {
          lines <- c(lines, sprintf("  u_criterion2_%s_raw ~ std_normal();", group))
        } else {
          lines <- c(lines, sprintf("  to_vector(u_criterion2_%s_raw) ~ std_normal();", group))
          lines <- c(lines, .re_corr_prior(paste0("criterion2_", group), re, cor_prior))
        }
      }
    }
  }
  
  for (cor_id in names(model_data$cross_cor)) {
    cc <- model_data$cross_cor[[cor_id]]
    cor_prior <- .cross_cor_prior(pl, cor_id, cc$group)

    sd_lines <- character(0)
    idx <- 1
    for (member in cc$members) {
      param <- member$param
      dim <- member$dim
      re <- .re_list_of(model_data, param)[[cc$group]]
      sd_priors <- if (is.null(re)) rep(get_sd_prior(pl, param, cc$group), dim)
                   else rep_len(.re_sd_priors(pl, param, cc$group, re), dim)
      for (j in seq_len(dim)) {
        sd_lines <- c(sd_lines,
                      sprintf("  sigma_cross_%s_%s[%d] ~ %s;  // %s[%d]",
                              cor_id, cc$group, idx, sd_priors[j], param, j))
        idx <- idx + 1
      }
    }
    
    lines <- c(lines, sd_lines,
               sprintf("  to_vector(u_cross_%s_%s_raw) ~ std_normal();", cor_id, cc$group),
               sprintf("  L_corr_cross_%s_%s ~ %s;", cor_id, cc$group, cor_prior))
  }
  
  lines <- c(lines, generate_smooth_priors(model_data))

  ls_rows <- .ls_block(model_data, family)
  if (!is.null(batch_info) && isTRUE(model_data$threads)) {
    lik <- c(ls_rows, batch_info$stan_call_threaded)
  } else if (!is.null(batch_info)) {
    lik <- c("  // Batch C++ likelihood (single autodiff node)", ls_rows, batch_info$stan_call)
  } else {
    lik <- .plain_stan_dev(".plain_model_likelihood")(model_data, family, family_name)
  }
  lines <- c(lines, "", "  if (!prior_only) {", lik, "  }", "}")

  paste(lines, collapse = "\n")
}

.has_crit2_cov <- function(model_data) {
  isTRUE(model_data$varying_source_criteria) &&
    isTRUE(model_data$stan_data$P_criterion2_cov > 0)
}

#' Generate threshold code for CDP model
#' @noRd
generate_thresh_code_cdp <- function(model_data) {
  crit <- model_data$criterion
  J <- model_data$stan_data$J
  if (isTRUE(model_data$has_criterion_scale)) return(.ls_thresh_lines(model_data, "tau", J))

  is_intercept_only <- isTRUE(crit$is_intercept_only) || crit$n_coef == 1

  gen_re_cdp <- function(target, k_expr) {
    re_lines <- character(0)
    if (!is.null(crit$random)) {
      for (group in names(crit$random)) {
        re <- crit$random[[group]]
        var_name <- if (!is.null(re$cor_id)) {
          sprintf("u_criterion_%s_from_%s", group, re$cor_id)
        } else {
          sprintf("u_criterion_%s", group)
        }
        R <- re$n_re_terms %||% 1L

        re_lines <- c(re_lines, if (isTRUE(re$use_z_matrix)) {
          sprintf("    %s += Z_criterion_%s[n, ] * %s[%s[n], ((%s-1)*%d+1):((%s)*%d)]';",
                  target, group, var_name, group, k_expr, R, k_expr, R)
        } else if (!is.null(re$term_idx)) {
          sprintf("    if (idx_criterion_%s[n] > 0) %s += %s[%s[n], (%s-1)*%d + idx_criterion_%s[n]];",
                  group, target, var_name, group, k_expr, R, group)
        } else if (re$dim > 1) {
          sprintf("    %s += %s[%s[n], %s];", target, var_name, group, k_expr)
        } else if (!is.null(re$cor_id)) {
          sprintf("    %s += %s[%s[n]];", target, var_name, group)
        } else {
          sprintf("    %s += %s[%s[n], 1];", target, var_name, group)
        })
      }
    }
    re_lines
  }

  lines <- c("    // Build thresholds (tau) using mid-anchor + log-gaps parameterization",
             sprintf("    vector[J] tau;"))

  if (is_intercept_only) {
    lines <- c(lines, "    tau[mid_thresh_cdp] = beta_thresh_mid[1];")
  } else {
    lines <- c(lines, "    tau[mid_thresh_cdp] = X_criterion[n, ] * beta_thresh_mid;")
  }
  lines <- c(lines, gen_re_cdp("tau[mid_thresh_cdp]", "mid_thresh_cdp"))

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

  lines <- c(lines,
             "    for (j in (mid_thresh_cdp+1):J) {")

  if (is_intercept_only) {
    lines <- c(lines, "      real log_gap = beta_log_gaps[j - mid_thresh_cdp, 1];")
  } else {
    lines <- c(lines, "      real log_gap = X_criterion[n, ] * beta_log_gaps[j - mid_thresh_cdp, ]';")
  }
  lines <- c(lines, gen_re_cdp("log_gap", "j"))
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

  lines <- c(lines,
             "    for (j_down in 1:(mid_thresh_cdp-1)) {",
             "      int j = mid_thresh_cdp - j_down;")

  if (is_intercept_only) {
    lines <- c(lines, "      real log_gap = beta_log_gaps[J - mid_thresh_cdp + j_down, 1];")
  } else {
    lines <- c(lines, "      real log_gap = X_criterion[n, ] * beta_log_gaps[J - mid_thresh_cdp + j_down, ]';")
  }
  lines <- c(lines, gen_re_cdp("log_gap", "j"))
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


.ls_sum_zero_blocks <- function(blocks, model_data) {
  add_data <- character(0)
  add_tp <- character(0)
  sz <- function(re, decl, tp, basis_decl) {
    m <- regmatches(blocks$params, regexec(re, blocks$params))[[1]]
    if (!length(m)) stop("internal: no gap declaration to normalise", call. = FALSE)
    fill <- function(x) gsub("%s", m[2], x, fixed = TRUE)
    blocks$params <<- sub(re, fill(decl), blocks$params)
    add_data <<- c(add_data, fill(basis_decl))
    add_tp <<- c(add_tp, fill(tp))
  }
  if (.ls_normalised(model_data, 1L)) {
    sz("  matrix\\[([^],]+), 1\\] beta_log_gaps;[^\n]*", "  vector[%s-1] beta_log_gaps_z;",
       "  matrix[%s, 1] beta_log_gaps = to_matrix(ls_gap_basis * beta_log_gaps_z, %s, 1);",
       "  matrix[%s, %s-1] ls_gap_basis;")
  }
  if (.ls_normalised(model_data, 2L)) {
    if (isTRUE(model_data$varying_source_criteria)) {
      sz("  matrix\\[n_vset, (K2-2)\\] beta_log_gaps_2_varying;[^\n]*", "  matrix[n_vset, %s-1] beta_log_gaps_2_varying_z;",
         "  matrix[n_vset, %s] beta_log_gaps_2_varying = beta_log_gaps_2_varying_z * ls_gap_basis2';",
         "  matrix[%s, %s-1] ls_gap_basis2;")
    } else {
      sz("  matrix\\[(K2-2), 1\\] beta_log_gaps_2;[^\n]*", "  vector[%s-1] beta_log_gaps_2_z;",
         "  matrix[%s, 1] beta_log_gaps_2 = to_matrix(ls_gap_basis2 * beta_log_gaps_2_z, %s, 1);",
         "  matrix[%s, %s-1] ls_gap_basis2;")
    }
  }
  if (!length(add_tp)) return(blocks)
  blocks$data <- sub("\n}\\s*$", paste0("\n", paste(add_data, collapse = "\n"), "\n}"), blocks$data)
  blocks$tparams <- sub("transformed parameters {", paste0("transformed parameters {\n", paste(add_tp, collapse = "\n")),
                        blocks$tparams, fixed = TRUE)
  blocks
}


.ls_scale_coefs <- function(model_data, which = 1L) {
  pn <- if (which == 2L) "criterion2_scale" else "criterion_scale"
  if (!isTRUE(model_data[[paste0("has_", pn)]])) return(0L)
  as.integer(model_data[[paste0(pn, "_fixed")]]$n_coef %||% 0L)
}

.ls_block <- function(model_data, family) {
  c(if (isTRUE(model_data$has_criterion_scale)) .ls_rows_code(model_data, family),
    if (isTRUE(model_data$has_criterion2_scale)) .ls2_rows_code(model_data))
}

.ls_loc_lines <- function(crit, pname, mid, smooth) {
  loc <- sprintf("real %s_n = X_%s[n, ] * %s;", pname, pname, mid)
  for (group in names(crit$random)) {
    re <- crit$random[[group]]
    u <- if (!is.null(re$cor_id)) sprintf("u_%s_%s_from_%s", pname, group, re$cor_id) else sprintf("u_%s_%s", pname, group)
    vec <- re$dim == 1 && (!is.null(re$cor_id) || pname == "criterion2")
    loc <- c(loc, if (isTRUE(re$use_z_matrix)) {
      if (vec) sprintf("%s_n += Z_%s_%s[n, 1] * %s[%s[n]];", pname, pname, group, u, group)
      else sprintf("%s_n += Z_%s_%s[n, ] * %s[%s[n], ]';", pname, pname, group, u, group)
    } else if (!is.null(re$term_idx)) {
      sprintf("if (idx_%s_%s[n] > 0) %s_n += %s[%s[n], idx_%s_%s[n]];", pname, group, pname, u, group, pname, group)
    } else if (vec) {
      sprintf("%s_n += %s[%s[n]];", pname, u, group)
    } else {
      sprintf("%s_n += %s[%s[n], 1];", pname, u, group)
    })
  }
  for (sm in smooth) {
    for (comp in sm$components) {
      for (k in seq_along(comp$Zs_list)) {
        base <- paste0(pname, "_", comp$san_label, "_", k)
        loc <- c(loc, sprintf("%s_n += Zs_%s[n, ] * s_%s[1, ]';", pname, base, base))
      }
    }
  }
  loc
}

.ls2_parts <- function(model_data) {
  J <- model_data$K2 - 1L
  M <- as.integer(ceiling(J / 2))
  n_up <- J - M
  varying <- isTRUE(model_data$varying_source_criteria)
  gap <- if (varying) function(k) sprintf("beta_log_gaps_2_varying[vs, %s]", k)
         else function(k) sprintf("beta_log_gaps_2[%s, 1]", k)
  shape <- c(sprintf("vector[%d] t_ls2 = rep_vector(0, %d);", J, J),
             if (n_up > 0) sprintf("for (k in %d:%d) t_ls2[k] = t_ls2[k - 1] + exp(%s);", M + 1L, J, gap(sprintf("k - %d", M))),
             if (M > 1) sprintf("for (kd in 1:%d) t_ls2[%d - kd] = t_ls2[%d - kd + 1] - exp(%s);", M - 1L, M, M,
                                gap(sprintf("%d + kd", n_up))),
             "real tbar_ls2 = mean(t_ls2);")
  crit <- model_data$criterion2
  if (varying) {
    loc <- "real criterion2_n = beta_thresh_mid_2_varying[vs];"
    if (.has_crit2_cov(model_data)) loc <- c(loc, "criterion2_n += X_criterion2_cov[n, ] * beta_thresh_mid_2_cov;")
    for (group in names(crit$random)) {
      re <- crit$random[[group]]
      u <- if (!is.null(re$cor_id)) sprintf("u_criterion2_%s_from_%s", group, re$cor_id) else sprintf("u_criterion2_%s", group)
      R <- re$n_re_terms %||% 1L
      loc <- c(loc, if (isTRUE(re$use_z_matrix)) {
        sprintf("for (r in 1:%d) criterion2_n += Z_criterion2_%s[n, r] * %s[%s[n], (vb - 1) * %d + r];", R, group, u, group, R)
      } else {
        sprintf("criterion2_n += %s[%s[n], vb];", u, group)
      })
    }
    for (sm in model_data$smooth_data[["criterion2"]]) {
      for (comp in sm$components) {
        for (k in seq_along(comp$Zs_list)) {
          base <- paste0("criterion2_", comp$san_label, "_", k)
          loc <- c(loc, sprintf("criterion2_n += Zs_%s[n, ] * s_%s[1, ]';", base, base))
        }
      }
    }
  } else {
    loc <- .ls_loc_lines(crit, "criterion2", "beta_thresh_mid_2", model_data$smooth_data[["criterion2"]])
  }
  scale <- sub("^ +", "", generate_param_lp("criterion2_scale", model_data$criterion2_scale_fixed,
                                             model_data$criterion2_scale_random,
                                             model_data$smooth_data[["criterion2_scale"]]))
  list(J = J, M = M, n_up = n_up, varying = varying, gap = gap, shape = shape, loc = loc, scale = scale)
}

.ls2_rows_code <- function(model_data) {
  p <- .ls2_parts(model_data)
  J <- p$J; M <- p$M; n_up <- p$n_up
  gap_row <- if (identical(model_data$gap_link, "softplus")) {
    c(sprintf("        real v_ls = sc_ls2 * exp(%s);", p$gap("g")), "        u_ls_rows2[n, col] = v_ls + log(-expm1(-v_ls));")
  } else {
    sprintf("        u_ls_rows2[n, col] = %s + criterion2_scale_n;", p$gap("g"))
  }
  c(sprintf("    matrix[N, %d] u_ls_rows2;", J),
    if (!p$varying) paste0("    ", p$shape),
    "    for (n in 1:N) {",
    if (p$varying) c("      int vb = vbin[y[n]];", "      int vs = vb + n_vbin * (vcell[n] - 1);", paste0("      ", p$shape)),
    paste0("      ", c(p$loc, p$scale)),
    "      real sc_ls2 = exp(criterion2_scale_n);",
    sprintf("      u_ls_rows2[n, %d] = criterion2_n - sc_ls2 * tbar_ls2;", M),
    sprintf("      for (g in 1:%d) {", J - 1L),
    sprintf("        int col = g <= %d ? %d + g : %d - (g - %d);", n_up, M, M, n_up),
    gap_row,
    "      }",
    "    }")
}

.ls2_thresh_lines <- function(model_data) {
  p <- .ls2_parts(model_data)
  if (!p$varying) {
    return(c(sprintf("    vector[%d] thresh2;", p$J), "    {",
             paste0("      ", c(p$loc, p$scale, p$shape)),
             "      thresh2 = criterion2_n + exp(criterion2_scale_n) * (t_ls2 - tbar_ls2);",
             "    }"))
  }
  c("    matrix[K1, K2-1] thresh2_varying;", "    {",
    paste0("      ", p$scale),
    "      for (k1 in 1:K1) {",
    "        int vb = vbin[k1];",
    "        int vs = vb + n_vbin * (vcell[n] - 1);",
    paste0("        ", c(p$loc, p$shape)),
    "        thresh2_varying[k1] = (criterion2_n + exp(criterion2_scale_n) * (t_ls2 - tbar_ls2))';",
    "      }",
    "    }")
}

.ls_rows_code <- function(model_data, family) {
  fam <- if (is.list(family)) family$family else family
  J <- if (fam == "cdp") model_data$stan_data$J else model_data$K - 1L
  M <- as.integer(ceiling(J / 2))
  n_up <- J - M
  gap_row <- if (identical(model_data$gap_link, "softplus")) {
    c("        real v_ls = sc_ls * exp(beta_log_gaps[g, 1]);",
      "        u_ls_rows[n, col] = v_ls + log(-expm1(-v_ls));")
  } else {
    "        u_ls_rows[n, col] = beta_log_gaps[g, 1] + criterion_scale_n;"
  }
  c(sprintf("    matrix[N, %d] u_ls_rows;", J),
    .ls_shape_lines(J, "    "),
    "    real tbar_ls = mean(t_ls);",
    "    for (n in 1:N) {",
    .ls_lp_lines(model_data, "      "),
    "      real sc_ls = exp(criterion_scale_n);",
    sprintf("      u_ls_rows[n, %d] = criterion_n - sc_ls * tbar_ls;", M),
    sprintf("      for (g in 1:%d) {", J - 1L),
    sprintf("        int col = g <= %d ? %d + g : %d - (g - %d);", n_up, M, M, n_up),
    gap_row,
    "      }",
    "    }")
}

.ls_thresh_lines <- function(model_data, target, J) {
  c(sprintf("    vector[%d] %s;", J, target),
    "    {",
    .ls_lp_lines(model_data, "      "),
    .ls_shape_lines(J, "      "),
    sprintf("      %s = criterion_n + exp(criterion_scale_n) * (t_ls - mean(t_ls));", target),
    "    }")
}

.ls_shape_lines <- function(J, indent) {
  M <- as.integer(ceiling(J / 2))
  n_up <- J - M
  paste0(indent, c(
    sprintf("vector[%d] t_ls = rep_vector(0, %d);", J, J),
    if (n_up > 0) sprintf("for (k in %d:%d) t_ls[k] = t_ls[k - 1] + exp(beta_log_gaps[k - %d, 1]);", M + 1L, J, M),
    if (M > 1) sprintf("for (kd in 1:%d) t_ls[%d - kd] = t_ls[%d - kd + 1] - exp(beta_log_gaps[%d + kd, 1]);",
                       M - 1L, M, M, n_up)))
}

.ls_lp_lines <- function(model_data, indent) {
  crit <- model_data$criterion
  loc <- "real criterion_n = X_criterion[n, ] * beta_thresh_mid;"
  for (group in names(crit$random)) {
    re <- crit$random[[group]]
    u <- if (!is.null(re$cor_id)) sprintf("u_criterion_%s_from_%s", group, re$cor_id) else sprintf("u_criterion_%s", group)
    vec <- !is.null(re$cor_id) && re$dim == 1
    loc <- c(loc, if (isTRUE(re$use_z_matrix)) {
      if (vec) sprintf("criterion_n += Z_criterion_%s[n, 1] * %s[%s[n]];", group, u, group)
      else sprintf("criterion_n += Z_criterion_%s[n, ] * %s[%s[n], ]';", group, u, group)
    } else if (!is.null(re$term_idx)) {
      sprintf("if (idx_criterion_%s[n] > 0) criterion_n += %s[%s[n], idx_criterion_%s[n]];", group, u, group, group)
    } else if (vec) {
      sprintf("criterion_n += %s[%s[n]];", u, group)
    } else {
      sprintf("criterion_n += %s[%s[n], 1];", u, group)
    })
  }
  for (sm in model_data$smooth_data[["criterion"]]) {
    for (comp in sm$components) {
      for (k in seq_along(comp$Zs_list)) {
        base <- paste0("criterion_", comp$san_label, "_", k)
        loc <- c(loc, sprintf("criterion_n += Zs_%s[n, ] * s_%s[1, ]';", base, base))
      }
    }
  }
  scale <- sub("^ +", "", generate_param_lp("criterion_scale", model_data$criterion_scale_fixed,
                                             model_data$criterion_scale_random,
                                             model_data$smooth_data[["criterion_scale"]]))
  paste0(indent, c(loc, scale))
}

generate_param_lp <- function(param_name, fixed_info, random_info, smooth_info = NULL) {
  lines <- character(0)

  has_fixed <- !is.null(fixed_info) && fixed_info$n_coef > 0

  if (!has_fixed) {
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
        z_name <- paste0("Z_", param_name, "_", group)
        if (re$dim == 1) {
          lines <- c(lines, sprintf("    %s_n += %s[n, 1] * %s[%s[n]];", param_name, z_name, var_name, group))
        } else {
          lines <- c(lines, sprintf("    %s_n += %s[n, ] * %s[%s[n], ]';", param_name, z_name, var_name, group))
        }
      } else if (re$dim == 1) {
        if (!is.null(re$term_idx)) {
          idx_var <- paste0("idx_", param_name, "_", group)
          lines <- c(lines, sprintf("    if (%s[n] > 0) %s_n += %s[%s[n]];", idx_var, param_name, var_name, group))
        } else {
          lines <- c(lines, sprintf("    %s_n += %s[%s[n]];", param_name, var_name, group))
        }
      } else if (!is.null(re$term_idx)) {
        idx_var <- paste0("idx_", param_name, "_", group)
        lines <- c(lines, sprintf("    if (%s[n] > 0) %s_n += %s[%s[n], %s[n]];", idx_var, param_name, var_name, group, idx_var))
      }
    }
  }

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

generate_generated_quantities_block_v2 <- function(model_data, family, batch_info = NULL) {
  family_name <- if (is.list(family)) family$family else family
  
  lines <- c("generated quantities {")
  
  param_names <- c("dprime", "sigma", "lambda", "dprime2", "sigma2",
                    "rec_crit", "know_crit",
                    "dprime_B", "lambda_B", "discrim", "discrim_B",
                    "sigma_B", "sigma2_B", "rho", "rho_B", "rho_N", "mratio", "mratio_N", "sigma_meta",
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
  
  for (group in names(model_data$criterion$random)) {
    re <- model_data$criterion$random[[group]]
    if (is.null(re$cor_id) && isTRUE(re$correlated) && re$dim > 1) {
      dim_var <- sprintf("D_criterion_%s", group)
      lines <- c(lines, sprintf("  matrix[%s, %s] corr_criterion_%s = multiply_lower_tri_self_transpose(L_corr_criterion_%s);",
                                dim_var, dim_var, group, group))
    }
  }
  
  if (isTRUE(model_data$has_criterion2) && !is.null(model_data$criterion2$random)) {
    for (group in names(model_data$criterion2$random)) {
      re <- model_data$criterion2$random[[group]]
      if (is.null(re$cor_id) && isTRUE(re$correlated) && re$dim > 1) {
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
  
  if (!is.null(batch_info) && length(.ls_block(model_data, family))) {
    lines <- c(lines, "", "  vector[N_ll] log_lik;", "  if (N_ll > 0) {", .ls_block(model_data, family),
               sub("vector[N] log_lik = ", "log_lik = ", batch_info$stan_call_vec, fixed = TRUE), "  }", "}")
    return(paste(lines, collapse = "\n"))
  }
  if (!is.null(batch_info)) {
    lines <- c(lines, "", "  vector[N_ll] log_lik;", "  if (N_ll > 0) {",
               sub("vector[N] log_lik = ", "log_lik = ", batch_info$stan_call_vec, fixed = TRUE), "  }", "}")
    return(paste(lines, collapse = "\n"))
  }
  lines <- c(lines, .plain_stan_dev(".plain_gq_log_lik")(model_data, family_name), "}")
  
  paste(lines, collapse = "\n")
}

get_all_groups <- function(model_data) {
  groups <- c(names(model_data$dprime_random), names(model_data$criterion$random), names(model_data$bias$random))
  if (model_data$has_sigma) groups <- c(groups, names(model_data$sigma_random))
  if (model_data$has_lambda) groups <- c(groups, names(model_data$lambda_random))
  if (model_data$has_dprime2) groups <- c(groups, names(model_data$dprime2_random))
  if (model_data$has_sigma2) groups <- c(groups, names(model_data$sigma2_random))
  if (isTRUE(model_data$has_rec_crit)) groups <- c(groups, names(model_data$rec_crit_random))
  if (isTRUE(model_data$has_know_crit)) groups <- c(groups, names(model_data$know_crit_random))
  if (isTRUE(model_data$has_criterion2) && !is.null(model_data$criterion2$random)) {
    groups <- c(groups, names(model_data$criterion2$random))
  }
  for (pname in c("dprime_B", "lambda_B", "discrim", "discrim_B",
                  "sigma_B", "sigma2_B", "rho", "rho_B", "rho_N", "mratio", "mratio_N", "sigma_meta",
                  "dprime_L", "sigma_L", "lambda_L", "lambda2", "lambda2_B", "criterion_scale", "criterion2_scale")) {
    re_list <- model_data[[paste0(pname, "_random")]]
    if (!is.null(re_list)) groups <- c(groups, names(re_list))
  }
  for (cor_id in names(model_data$cross_cor)) {
    groups <- c(groups, model_data$cross_cor[[cor_id]]$group)
  }
  unique(groups)
}

get_all_term_indices <- function(model_data) {
  indices <- list()
  
  param_names <- c("dprime", "sigma", "lambda", "dprime2", "sigma2", "rec_crit", "know_crit",
                   "dprime_B", "lambda_B", "discrim", "discrim_B",
                   "sigma_B", "sigma2_B", "rho", "rho_B", "rho_N", "mratio", "mratio_N", "sigma_meta",
                   "dprime_L", "sigma_L", "lambda_L", "lambda2", "lambda2_B", "criterion_scale", "criterion2_scale")
  for (pname in param_names) {
    re_list <- model_data[[paste0(pname, "_random")]]
    if (is.null(re_list)) next
    for (group in names(re_list)) {
      re <- re_list[[group]]
      if (!is.null(re$term_idx) && !isTRUE(re$use_z_matrix)) {
        idx_name <- paste0("idx_", pname, "_", group)
        dim_name <- paste0("D_idx_", pname, "_", group)
        indices[[idx_name]] <- list(idx_name = idx_name, dim_name = dim_name, dim = re$dim)
      }
    }
  }
  
  for (group in names(model_data$bias$random)) {
    re <- model_data$bias$random[[group]]
    if (!is.null(re$term_idx) && !isTRUE(re$use_z_matrix)) {
      idx_name <- paste0("idx_bias_", group)
      indices[[idx_name]] <- list(idx_name = idx_name, dim_name = paste0("D_idx_bias_", group), dim = re$n_cond_levels)
    }
  }

  for (group in names(model_data$criterion$random)) {
    re <- model_data$criterion$random[[group]]
    if (!is.null(re$term_idx) && !isTRUE(re$use_z_matrix)) {
      idx_name <- paste0("idx_criterion_", group)
      dim_name <- paste0("D_idx_criterion_", group)
      indices[[idx_name]] <- list(idx_name = idx_name, dim_name = dim_name, dim = re$n_cond_levels)
    }
  }

  if (isTRUE(model_data$has_criterion2)) {
    for (group in names(model_data$criterion2$random)) {
      re <- model_data$criterion2$random[[group]]
      if (!is.null(re$term_idx) && !isTRUE(re$use_z_matrix)) {
        idx_name <- paste0("idx_criterion2_", group)
        dim_name <- paste0("D_idx_criterion2_", group)
        indices[[idx_name]] <- list(idx_name = idx_name, dim_name = dim_name, dim = re$n_cond_levels)
      }
    }
  }

  indices
}

get_all_z_matrices <- function(model_data) {
  z_matrices <- list()

  param_names <- c("dprime", "sigma", "lambda", "dprime2", "sigma2",
                    "rec_crit", "know_crit",
                    "dprime_B", "lambda_B", "discrim", "discrim_B",
                    "sigma_B", "sigma2_B", "rho", "rho_B", "rho_N", "mratio", "mratio_N", "sigma_meta",
                    "dprime_L", "sigma_L", "lambda_L", "lambda2", "lambda2_B", "criterion_scale", "criterion2_scale")
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
  
  for (group in names(model_data$bias$random)) {
    re <- model_data$bias$random[[group]]
    if (isTRUE(re$use_z_matrix)) {
      z_name <- paste0("Z_bias_", group)
      z_matrices[[z_name]] <- list(name = z_name, dim = re$n_re_terms, param = "bias", group = group)
    }
  }

  for (group in names(model_data$criterion$random)) {
    re <- model_data$criterion$random[[group]]
    if (isTRUE(re$use_z_matrix)) {
      z_name <- paste0("Z_criterion_", group)
      z_matrices[[z_name]] <- list(name = z_name, dim = re$n_re_terms, param = "criterion", group = group)
    }
  }

  if (isTRUE(model_data$has_criterion2)) {
    for (group in names(model_data$criterion2$random)) {
      re <- model_data$criterion2$random[[group]]
      if (isTRUE(re$use_z_matrix)) {
        z_name <- paste0("Z_criterion2_", group)
        z_matrices[[z_name]] <- list(name = z_name, dim = re$n_re_terms, param = "criterion2", group = group)
      }
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
    base <- paste0(pname, "_", suffix, "_", k)
    nb <- paste0("nbasis_", base)
    info <- if (base %in% model_data$smooth_info_bases) {
      if (n_thresh > 0) sprintf("  array[%d] matrix[%s, %s] Ism_%s;", n_thresh, nb, nb, base)
      else sprintf("  matrix[%s, %s] Ism_%s;", nb, nb, base)
    }
    c(sprintf("  // Smooth penalized basis: %s component %d", suffix, k),
      sprintf("  int<lower=1> nbasis_%s_%s_%d;", pname, suffix, k),
      sprintf("  matrix[N, nbasis_%s_%s_%d] %s;", pname, suffix, k, zs_name),
      info)
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
    if (base %in% model_data$smooth_info_bases) {
      if (n_thresh > 0) {
        c(sprintf("  matrix[%d, %s] s_%s;", n_thresh, nb, base),
          sprintf("  for (t in 1:%d) {", n_thresh),
          sprintf("    matrix[%s, %s] Kc = cholesky_decompose(add_diag(square(sds_%s[t]) * Ism_%s[t], 1));",
                  nb, nb, base, base),
          sprintf("    s_%s[t, ] = sds_%s[t] * mdivide_right_tri_low(zs_%s[t, ], Kc);", base, base, base),
          "  }")
      } else {
        c(sprintf("  vector[%s] s_%s;", nb, base),
          "  {",
          sprintf("    matrix[%s, %s] Kc = cholesky_decompose(add_diag(square(sds_%s) * Ism_%s, 1));",
                  nb, nb, base, base),
          sprintf("    s_%s = sds_%s * mdivide_right_tri_low(zs_%s', Kc)';", base, base, base),
          "  }")
      }
    } else if (n_thresh > 0) {
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
    nb <- paste0("nbasis_", base)
    if (base %in% model_data$smooth_info_bases) {
      if (n_thresh > 0) {
        c(sprintf("  for (t in 1:%d) {", n_thresh),
          sprintf("    matrix[%s, %s] Kc = cholesky_decompose(add_diag(square(sds_%s[t]) * Ism_%s[t], 1));",
                  nb, nb, base, base),
          sprintf("    row_vector[%s] v = mdivide_right_tri_low(zs_%s[t, ], Kc);", nb, base),
          "    target += -0.5 * dot_self(v) - sum(log(diagonal(Kc)));",
          "  }",
          sprintf("  sds_%s ~ %s;", base, sds_prior))
      } else {
        c("  {",
          sprintf("    matrix[%s, %s] Kc = cholesky_decompose(add_diag(square(sds_%s) * Ism_%s, 1));",
                  nb, nb, base, base),
          sprintf("    row_vector[%s] v = mdivide_right_tri_low(zs_%s', Kc);", nb, base),
          "    target += -0.5 * dot_self(v) - sum(log(diagonal(Kc)));",
          "  }",
          sprintf("  sds_%s ~ %s;", base, sds_prior))
      }
    } else if (n_thresh > 0) {
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

# ============================================================================
# m-AFC position biases: Stan declarations, assembly and priors
# ============================================================================

#' Stan lines for the bias parameter in one program block
#' @noRd
.bias_stan <- function(model_data, block, pl = NULL) {
  b <- model_data$bias
  if (is.null(b)) return(character(0))
  out <- character(0)
  res <- b$random %||% list()
  if (block == "data") {
    return(c("", "  // position-bias fixed effects design matrix", "  int<lower=1> P_bias;", "  matrix[N, P_bias] X_bias;"))
  }
  if (block == "transformed_data") {
    for (group in names(res)) {
      if (is.null(res[[group]]$cor_id)) out <- c(out, sprintf("  int D_bias_%s = %d;", group, res[[group]]$dim))
    }
    return(out)
  }
  if (block == "parameters") {
    out <- "  matrix[K - 1, P_bias] beta_bias;  // position biases (last position the reference)"
    for (group in names(res)) {
      re <- res[[group]]
      if (!is.null(re$cor_id)) next
      out <- c(out, sprintf("  matrix[N_%s, D_bias_%s] u_bias_%s_raw;", group, group, group),
               sprintf("  vector<lower=0>[D_bias_%s] sigma_bias_%s;", group, group),
               .re_corr_decl(paste0("bias_", group), re))
    }
    return(out)
  }
  if (block == "transformed_parameters") {
    for (group in names(res)) {
      re <- res[[group]]
      if (!is.null(re$cor_id)) next
      if (isTRUE(re$correlated)) {
        out <- c(out, .re_corr_assemble(paste0("bias_", group), re), sprintf(
          "  matrix[N_%s, D_bias_%s] u_bias_%s = u_bias_%s_raw * diag_pre_multiply(sigma_bias_%s, L_corr_bias_%s)';",
          group, group, group, group, group, group))
      } else {
        out <- c(out, sprintf("  matrix[N_%s, D_bias_%s] u_bias_%s;", group, group, group),
                 sprintf("  for (d in 1:D_bias_%s) u_bias_%s[, d] = u_bias_%s_raw[, d] * sigma_bias_%s[d];",
                         group, group, group, group))
      }
    }
    return(out)
  }
  if (block == "model") {
    for (p in seq_along(b$coef_names)) {
      out <- c(out, sprintf("  beta_bias[, %d] ~ %s;", p, get_fixed_prior(pl, "bias", b$coef_names[p])))
    }
    for (group in names(res)) {
      re <- res[[group]]
      if (!is.null(re$cor_id)) next
      sdp <- .re_sd_priors(pl, "bias", group, re)
      if (length(unique(sdp)) == 1) {
        out <- c(out, sprintf("  sigma_bias_%s ~ %s;", group, sdp[1]))
      } else {
        out <- c(out, sprintf("  sigma_bias_%s[%d] ~ %s;", group, seq_along(sdp), sdp))
      }
      out <- c(out, sprintf("  to_vector(u_bias_%s_raw) ~ std_normal();", group),
               .re_corr_prior(paste0("bias_", group), re, get_cor_prior(pl, "bias", group)))
    }
    return(out)
  }
  out
}
