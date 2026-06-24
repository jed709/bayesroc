#' Generate Batch C++ Likelihood Code
#'
#' Generates a model-specific C++ function that computes the entire observation
#' loop in double arithmetic with analytic gradients, returning a single autodiff
#' node via stan::math::precomputed_gradients. Eliminates ~15N autodiff tape
#' entries per gradient evaluation.
#'
#' The generated function handles all RE patterns (scalar intercept-only, scalar
#' with term_idx, multi-dim with term_idx, Z-matrix scalar/multi, cross-parameter
#' correlation), UVSDT sigma with exp link, multi-predictor criterion, criterion
#' REs per threshold position, encoding variables, count-weighted likelihoods,
#' and threading support via start_idx/end_idx subsetting.
#'
#' @param model_data Model data list from build_model_data()
#' @param family Model family object
#' @return List with $cpp_code, $stan_decl, $stan_call, or NULL if unsupported
#' @noRd
generate_batch_cpp <- function(model_data, family) {
  family_name <- if (is.list(family)) family$family else family

  supported <- c("evsdt", "uvsdt", "cumulative", "dpsdt", "mixture", "source_mixture", "vrdp2d",
                  "bivariate_sdt", "bivariate_dp", "bivariate_cumulative", "cdp")
  if (!(family_name %in% supported)) return(NULL)

  # (varying_source_criteria and bounded are now supported in batch)

  # Collect structured info about every parameter

  params <- collect_batch_params(model_data, family_name)

  # Store link name for logit vs probit dispatch
  link_name <- if (is.list(family) && !is.null(family$link)) family$link$name else "probit"
  params$.link_name <- link_name

  # Build the canonical argument list (drives C++, Stan decl, and Stan call)
  sig <- build_batch_signature(model_data, family_name, params)

  # Generate C++ code
  cpp <- generate_batch_cpp_code(model_data, family_name, params, sig)

  # Generate Stan function declaration (goes in functions{} block)
  stan_decl <- generate_batch_stan_decl(sig)

  # Generate Stan model block call (non-threaded: passes 1, N)
  stan_call <- generate_batch_stan_call(sig, threaded = FALSE)

  # Generate threaded versions
  stan_partial <- generate_batch_partial_log_lik(sig)
  stan_call_threaded <- generate_batch_reduce_sum_call(sig)

  list(cpp_code = cpp, stan_decl = stan_decl, stan_call = stan_call,
       stan_partial = stan_partial, stan_call_threaded = stan_call_threaded)
}


# ============================================================================
# Parameter collection
# ============================================================================

#' Collect parameter metadata for batch function generation
#' @noRd
collect_batch_params <- function(model_data, family_name) {
  params <- list()

  # dprime (always present for SDT families and cumulative)
  params$dprime <- list(
    fixed  = model_data$dprime_fixed,
    random = model_data$dprime_random,
    link   = "identity"
  )

  is_bivariate <- family_name %in% c("bivariate_sdt", "bivariate_dp", "bivariate_cumulative")
  is_cdp <- family_name %in% c("cdp")

  # sigma (UVSDT, or optional in dpsdt/mixture/vrdp2d/bivariate/cdp)
  if (family_name %in% c("uvsdt", "dpsdt", "mixture", "vrdp2d", "bivariate_sdt", "bivariate_dp", "cdp") && isTRUE(model_data$has_sigma)) {
    params$sigma <- list(
      fixed  = model_data$sigma_fixed,
      random = model_data$sigma_random,
      link   = "exp"
    )
  }

  # lambda (dpsdt, mixture, source_mixture, vrdp2d)
  if (family_name %in% c("dpsdt", "mixture", "source_mixture", "vrdp2d") && isTRUE(model_data$has_lambda)) {
    params$lambda <- list(
      fixed  = model_data$lambda_fixed,
      random = model_data$lambda_random,
      link   = "inv_logit"
    )
  }

  # dprime2 (mixture, vrdp2d, cdp)
  if (family_name %in% c("mixture", "vrdp2d", "cdp") && (isTRUE(model_data$has_dprime2) || isTRUE(model_data$needs_ordered_dprime))) {
    params$dprime2 <- list(
      fixed  = model_data$dprime2_fixed,
      random = model_data$dprime2_random,
      link   = "identity"
    )
  }

  # sigma2 (mixture, vrdp2d, cdp)
  if (family_name %in% c("mixture", "vrdp2d", "cdp") && isTRUE(model_data$has_sigma2)) {
    params$sigma2 <- list(
      fixed  = model_data$sigma2_fixed,
      random = model_data$sigma2_random,
      link   = "exp"
    )
  }

  # discrim (vrdp2d or bivariate -- source discriminability for source A)
  if ((family_name == "vrdp2d" || is_bivariate) && isTRUE(model_data$has_discrim)) {
    params$discrim <- list(
      fixed  = model_data$discrim_fixed,
      random = model_data$discrim_random,
      link   = "identity"
    )
  }

  # discrim_B (vrdp2d or bivariate -- optional, asymmetric source B)
  if ((family_name == "vrdp2d" || is_bivariate) && isTRUE(model_data$has_discrim_B)) {
    params$discrim_B <- list(
      fixed  = model_data$discrim_B_fixed,
      random = model_data$discrim_B_random,
      link   = "identity"
    )
  }

  # dprime_B (source_mixture or bivariate)
  if ((family_name == "source_mixture" || is_bivariate) && isTRUE(model_data$has_dprime_B)) {
    params$dprime_B <- list(
      fixed  = model_data$dprime_B_fixed,
      random = model_data$dprime_B_random,
      link   = "identity"
    )
  }

  # lambda_B (source_mixture or bivariate_dp)
  if (family_name %in% c("source_mixture", "bivariate_dp") && isTRUE(model_data$has_lambda_B)) {
    params$lambda_B <- list(
      fixed  = model_data$lambda_B_fixed,
      random = model_data$lambda_B_random,
      link   = "inv_logit"
    )
  }

  # dprime_L (lure mixture within mixture family)
  if (family_name == "mixture" && isTRUE(model_data$has_lure_mixture)) {
    params$dprime_L <- list(
      fixed  = model_data$dprime_L_fixed,
      random = model_data$dprime_L_random,
      link   = "identity"
    )
  }

  # sigma_L (lure mixture)
  if (family_name == "mixture" && isTRUE(model_data$has_sigma_L)) {
    params$sigma_L <- list(
      fixed  = model_data$sigma_L_fixed,
      random = model_data$sigma_L_random,
      link   = "exp"
    )
  }

  # lambda_L (lure mixture)
  if (family_name == "mixture" && isTRUE(model_data$has_lambda_L)) {
    params$lambda_L <- list(
      fixed  = model_data$lambda_L_fixed,
      random = model_data$lambda_L_random,
      link   = "inv_logit"
    )
  }

  # --- Bivariate-specific parameters ---

  # sigma_B (bivariate -- Source B detection SD)
  if (is_bivariate && isTRUE(model_data$has_sigma_B)) {
    params$sigma_B <- list(
      fixed  = model_data$sigma_B_fixed,
      random = model_data$sigma_B_random,
      link   = "exp"
    )
  }

  # sigma2 (bivariate -- Source A discrimination SD)
  if (is_bivariate && isTRUE(model_data$has_sigma2)) {
    params$sigma2 <- list(
      fixed  = model_data$sigma2_fixed,
      random = model_data$sigma2_random,
      link   = "exp"
    )
  }

  # sigma2_B (bivariate -- Source B discrimination SD)
  if (is_bivariate && isTRUE(model_data$has_sigma2_B)) {
    params$sigma2_B <- list(
      fixed  = model_data$sigma2_B_fixed,
      random = model_data$sigma2_B_random,
      link   = "exp"
    )
  }

  # rho (bivariate -- correlation for Source A)
  if (is_bivariate && isTRUE(model_data$has_rho)) {
    params$rho <- list(
      fixed  = model_data$rho_fixed,
      random = model_data$rho_random,
      link   = "tanh"  # Fisher z -> correlation
    )
  }

  # rho_B (bivariate -- correlation for Source B)
  if (is_bivariate && isTRUE(model_data$has_rho_B)) {
    params$rho_B <- list(
      fixed  = model_data$rho_B_fixed,
      random = model_data$rho_B_random,
      link   = "tanh"
    )
  }

  # rho_N (bivariate -- correlation for new items)
  if (is_bivariate && isTRUE(model_data$has_rho_N)) {
    params$rho_N <- list(
      fixed  = model_data$rho_N_fixed,
      random = model_data$rho_N_random,
      link   = "tanh"
    )
  }

  # lambda (bivariate_dp -- item recollection R_I)
  if (family_name == "bivariate_dp" && isTRUE(model_data$has_lambda)) {
    params$lambda <- list(
      fixed  = model_data$lambda_fixed,
      random = model_data$lambda_random,
      link   = "inv_logit"
    )
  }

  # lambda2 (bivariate_dp -- source recollection R_S)
  if (family_name == "bivariate_dp" && isTRUE(model_data$has_lambda2)) {
    params$lambda2 <- list(
      fixed  = model_data$lambda2_fixed,
      random = model_data$lambda2_random,
      link   = "inv_logit"
    )
  }

  # lambda2_B (bivariate_dp -- Source-B source recollection R_S_B; optional)
  if (family_name == "bivariate_dp" && isTRUE(model_data$has_lambda2_B)) {
    params$lambda2_B <- list(
      fixed  = model_data$lambda2_B_fixed,
      random = model_data$lambda2_B_random,
      link   = "inv_logit"
    )
  }

  # rec_crit (cdp -- recollection criterion)
  if (is_cdp && isTRUE(model_data$has_rec_crit)) {
    params$rec_crit <- list(
      fixed  = model_data$rec_crit_fixed,
      random = model_data$rec_crit_random,
      link   = "identity"
    )
  }

  # know_crit (cdp -- know criterion for R/K/G)
  if (is_cdp && isTRUE(model_data$has_know_crit)) {
    params$know_crit <- list(
      fixed  = model_data$know_crit_fixed,
      random = model_data$know_crit_random,
      link   = "identity"
    )
  }

  # Criterion
  params$criterion <- list(
    fixed  = model_data$criterion,
    random = model_data$criterion$random
  )

  # Criterion2 (vrdp2d or bivariate -- source dimension thresholds)
  if (family_name == "vrdp2d" || is_bivariate) {
    params$criterion2 <- list(
      fixed  = model_data$criterion2,
      random = model_data$criterion2$random
    )
  }

  # Model-level flags
  params$.family      <- family_name
  params$.has_counts  <- isTRUE(model_data$has_counts)
  params$.has_sigma   <- isTRUE(model_data$has_sigma) && family_name %in% c("uvsdt", "dpsdt", "mixture", "vrdp2d", "bivariate_sdt", "bivariate_dp", "cdp")
  params$.has_lambda  <- isTRUE(model_data$has_lambda) && family_name %in% c("dpsdt", "mixture", "source_mixture", "vrdp2d", "bivariate_dp")
  params$.has_dprime2 <- (isTRUE(model_data$has_dprime2) || isTRUE(model_data$needs_ordered_dprime)) && family_name %in% c("mixture", "vrdp2d", "cdp")
  params$.has_sigma2  <- isTRUE(model_data$has_sigma2) && family_name %in% c("mixture", "vrdp2d", "bivariate_sdt", "bivariate_dp", "cdp")
  params$.has_dprime_B <- isTRUE(model_data$has_dprime_B) && (family_name == "source_mixture" || is_bivariate)
  # lambda_B is supported by source_mixture (its original use) and by
  # bivariate_dp (asymmetric Source-B item recollection). The data/RE
  # wiring is identical; only the per-row dispatch differs (source_mixture
  # uses a per-row `source` index; bivariate_dp dispatches on item_type
  # inside bivariate_dp_cell).
  params$.has_lambda_B <- isTRUE(model_data$has_lambda_B) &&
    family_name %in% c("source_mixture", "bivariate_dp")
  params$.has_lambda_B_dp <- isTRUE(model_data$has_lambda_B) && family_name == "bivariate_dp"
  params$.has_lure_mixture <- isTRUE(model_data$has_lure_mixture) && family_name == "mixture"
  params$.has_dprime_L <- isTRUE(model_data$has_dprime_L) && family_name == "mixture"
  params$.has_sigma_L  <- isTRUE(model_data$has_sigma_L) && family_name == "mixture"
  params$.has_lambda_L <- isTRUE(model_data$has_lambda_L) && family_name == "mixture"
  params$.is_cumulative <- family_name == "cumulative"
  params$.has_dp_fixed <- !is.null(model_data$dprime_fixed) && model_data$dprime_fixed$n_coef > 0
  params$.is_source_mixture <- family_name == "source_mixture"
  params$.is_vrdp2d   <- family_name == "vrdp2d"
  params$.is_bivariate <- is_bivariate
  params$.is_bivariate_dp <- family_name == "bivariate_dp"
  params$.has_discrim  <- isTRUE(model_data$has_discrim) && (family_name == "vrdp2d" || is_bivariate)
  params$.has_disc_fixed <- isTRUE(model_data$has_discrim) && !is.null(model_data$discrim_fixed) && model_data$discrim_fixed$n_coef > 0
  params$.has_discrim_B <- isTRUE(model_data$has_discrim_B) && (family_name == "vrdp2d" || is_bivariate)
  params$.has_sigma_B  <- isTRUE(model_data$has_sigma_B) && is_bivariate
  params$.has_sigma2_B <- isTRUE(model_data$has_sigma2_B) && is_bivariate
  params$.has_rho      <- isTRUE(model_data$has_rho) && is_bivariate
  params$.has_rho_B    <- isTRUE(model_data$has_rho_B) && is_bivariate
  params$.has_rho_N    <- isTRUE(model_data$has_rho_N) && is_bivariate
  params$.has_lambda2  <- isTRUE(model_data$has_lambda2) && family_name == "bivariate_dp"
  params$.has_lambda2_B <- isTRUE(model_data$has_lambda2_B) && family_name == "bivariate_dp"
  params$.is_cdp       <- family_name == "cdp"
  params$.is_cdp_family <- is_cdp
  params$.has_rec_crit <- isTRUE(model_data$has_rec_crit) && is_cdp
  params$.has_know_crit <- isTRUE(model_data$has_know_crit) && is_cdp
  params$.n_rkg        <- if (is_cdp) model_data$stan_data$n_rkg else NULL
  params$.K           <- model_data$K
  params$.K2          <- model_data$K2
  params$.gap_link    <- if (!is.null(model_data$gap_link)) model_data$gap_link else "log"
  params$.varying_source_criteria <- isTRUE(model_data$varying_source_criteria)
  params$.new_source_criteria <- model_data$new_source_criteria
  params$.bounded     <- isTRUE(model_data$bounded)
  params$.smooth_data <- model_data$smooth_data

  params
}


# ============================================================================
# Signature builder -- single source of truth for argument ordering
# ============================================================================

#' Build the batch function signature
#'
#' Returns a list of argument descriptors. Each has:
#'   $cpp_type   -- C++ parameter declaration (templated or concrete)
#'   $stan_type  -- Stan declaration string
#'   $stan_name  -- actual Stan variable name for the call
#'   $tpl_name   -- template type name (NULL if not templated)
#'   $is_var     -- TRUE if this is a var-typed operand (needs autodiff check)
#'   $role       -- "data", "param", "pstream"
#' @noRd
build_batch_signature <- function(model_data, family_name, params) {
  args <- list()
  tpl_idx <- 0  # counter for unique template names

  # --- helpers ---
  add_data <- function(cpp_type, stan_type, stan_name) {
    args[[length(args) + 1]] <<- list(
      cpp_type  = cpp_type,
      stan_type = stan_type,
      stan_name = stan_name,
      tpl_name  = NULL,
      is_var    = FALSE,
      role      = "data"
    )
  }
  add_param <- function(stan_type, stan_name, is_vec = FALSE) {
    tpl_idx <<- tpl_idx + 1
    tname <- sprintf("T%d_", tpl_idx)
    cpp_type <- sprintf("const %s&", tname)
    args[[length(args) + 1]] <<- list(
      cpp_type  = cpp_type,
      stan_type = stan_type,
      stan_name = stan_name,
      tpl_name  = tname,
      is_var    = TRUE,
      role      = "param"
    )
  }
  add_data_tpl <- function(stan_type, stan_name) {
    # Data that still needs a template (e.g., design matrices passed as Eigen)
    tpl_idx <<- tpl_idx + 1
    tname <- sprintf("T%d_", tpl_idx)
    cpp_type <- sprintf("const %s&", tname)
    args[[length(args) + 1]] <<- list(
      cpp_type  = cpp_type,
      stan_type = stan_type,
      stan_name = stan_name,
      tpl_name  = tname,
      is_var    = FALSE,
      role      = "data"
    )
  }

  # Track added group index arrays to avoid duplicates
  added_group_idx <- character(0)
  add_group_idx <- function(group) {
    if (!(group %in% added_group_idx)) {
      add_data("const std::vector<int>&", "array[] int", group)
      added_group_idx <<- c(added_group_idx, group)
    }
  }

  has_two_responses <- isTRUE(params$.is_vrdp2d) || isTRUE(params$.is_bivariate)
  is_cdp_family <- isTRUE(params$.is_cdp_family)

  # ---- Standard data ----
  add_data("const std::vector<int>&",    "array[] int",  "y")
  if (has_two_responses) {
    add_data("const std::vector<int>&",  "array[] int",  "y2")
    add_data("const std::vector<int>&",  "array[] int",  "item_type")
    add_data("const int&",               "int",          "K1")
    add_data("const int&",               "int",          "K2")
    add_data("const int&",               "int",          "mid_thresh1")
    add_data("const int&",               "int",          "mid_thresh2")
    if (identical(params$.new_source_criteria, "shared")) {
      add_data("const std::vector<int>&",  "array[] int",  "is_new_response")
    }
  } else if (isTRUE(params$.is_source_mixture)) {
    add_data("const std::vector<int>&",  "array[] int",  "source")
  } else if (!isTRUE(params$.is_cumulative)) {
    add_data("const std::vector<double>&", "array[] real", "is_old_d")
  }
  # CDP-specific data
  if (is_cdp_family) {
    add_data("const std::vector<int>&",  "array[] int",  "rk")
    if (isTRUE(params$.is_cdp)) {
      add_data("const int&",             "int",          "J")
      add_data("const int&",             "int",          "n_rkg")
      add_data("const std::vector<int>&","array[] int",  "old_level_map")
      add_data("const int&",             "int",          "mid_thresh_cdp")
    }
  }
  if (!has_two_responses) {
    add_data("const int&",               "int",          "K")
    if (!isTRUE(params$.is_cdp)) {
      # CDP uses mid_thresh_cdp (already added above); others use mid_thresh
      add_data("const int&",             "int",          "mid_thresh")
    }
  }
  # Threading: start/end observation indices (1-based, inclusive)
  add_data("const int&",                 "int",          "start_idx")
  add_data("const int&",                 "int",          "end_idx")

  if (params$.has_counts) {
    add_data("const std::vector<int>&", "array[] int", "counts_d")
  }

  # ---- dprime fixed ----
  has_dp_fixed <- !is.null(params$dprime$fixed) && params$dprime$fixed$n_coef > 0
  if (has_dp_fixed) {
    add_data_tpl("matrix", "X_dprime")
    add_param("vector", "beta_dprime")
  }

  # ---- dprime REs ----
  dp_re_idx <- 0
  if (!is.null(params$dprime$random)) {
    for (group in names(params$dprime$random)) {
      dp_re_idx <- dp_re_idx + 1
      re <- params$dprime$random[[group]]
      # group index array (dedup across params sharing same group)
      add_group_idx(group)
      # Z matrix if needed
      if (isTRUE(re$use_z_matrix)) {
        add_data_tpl("matrix", sprintf("Z_dprime_%s", group))
      }
      # RE vector/matrix
      var_name <- re_var_name("dprime", group, re$cor_id)
      if (re$dim == 1) {
        add_param("vector", var_name)
      } else {
        add_param("matrix", var_name)
      }
      # term index if needed
      if (!is.null(re$term_idx)) {
        add_data("const std::vector<int>&",
                 "array[] int",
                 sprintf("idx_dprime_%s", group))
      }
    }
  }

  # ---- sigma fixed + REs (UVSDT) ----
  if (isTRUE(params$.has_sigma)) {
    add_data_tpl("matrix", "X_sigma")
    add_param("vector", "beta_sigma")

    sig_re_idx <- 0
    if (!is.null(params$sigma$random)) {
      for (group in names(params$sigma$random)) {
        sig_re_idx <- sig_re_idx + 1
        re <- params$sigma$random[[group]]
        add_group_idx(group)
        if (isTRUE(re$use_z_matrix)) {
          add_data_tpl("matrix", sprintf("Z_sigma_%s", group))
        }
        var_name <- re_var_name("sigma", group, re$cor_id)
        if (re$dim == 1) {
          add_param("vector", var_name)
        } else {
          add_param("matrix", var_name)
        }
        if (!is.null(re$term_idx)) {
          add_data("const std::vector<int>&",
                   "array[] int",
                   sprintf("idx_sigma_%s", group))
        }
      }
    }
  }

  # ---- lambda fixed + REs (dpsdt, mixture) ----
  if (isTRUE(params$.has_lambda)) {
    add_data_tpl("matrix", "X_lambda")
    add_param("vector", "beta_lambda")

    lam_re_idx <- 0
    if (!is.null(params$lambda$random)) {
      for (group in names(params$lambda$random)) {
        lam_re_idx <- lam_re_idx + 1
        re <- params$lambda$random[[group]]
        add_group_idx(group)
        if (isTRUE(re$use_z_matrix)) {
          add_data_tpl("matrix", sprintf("Z_lambda_%s", group))
        }
        var_name <- re_var_name("lambda", group, re$cor_id)
        if (re$dim == 1) {
          add_param("vector", var_name)
        } else {
          add_param("matrix", var_name)
        }
        if (!is.null(re$term_idx)) {
          add_data("const std::vector<int>&",
                   "array[] int",
                   sprintf("idx_lambda_%s", group))
        }
      }
    }
  }

  # ---- dprime2 fixed + REs (mixture) ----
  if (isTRUE(params$.has_dprime2)) {
    add_data_tpl("matrix", "X_dprime2")
    add_param("vector", "beta_dprime2")

    dp2_re_idx <- 0
    if (!is.null(params$dprime2$random)) {
      for (group in names(params$dprime2$random)) {
        dp2_re_idx <- dp2_re_idx + 1
        re <- params$dprime2$random[[group]]
        add_group_idx(group)
        if (isTRUE(re$use_z_matrix)) {
          add_data_tpl("matrix", sprintf("Z_dprime2_%s", group))
        }
        var_name <- re_var_name("dprime2", group, re$cor_id)
        if (re$dim == 1) {
          add_param("vector", var_name)
        } else {
          add_param("matrix", var_name)
        }
        if (!is.null(re$term_idx)) {
          add_data("const std::vector<int>&",
                   "array[] int",
                   sprintf("idx_dprime2_%s", group))
        }
      }
    }
  }

  # ---- sigma2 fixed + REs (mixture) ----
  if (isTRUE(params$.has_sigma2)) {
    add_data_tpl("matrix", "X_sigma2")
    add_param("vector", "beta_sigma2")

    sig2_re_idx <- 0
    if (!is.null(params$sigma2$random)) {
      for (group in names(params$sigma2$random)) {
        sig2_re_idx <- sig2_re_idx + 1
        re <- params$sigma2$random[[group]]
        add_group_idx(group)
        if (isTRUE(re$use_z_matrix)) {
          add_data_tpl("matrix", sprintf("Z_sigma2_%s", group))
        }
        var_name <- re_var_name("sigma2", group, re$cor_id)
        if (re$dim == 1) {
          add_param("vector", var_name)
        } else {
          add_param("matrix", var_name)
        }
        if (!is.null(re$term_idx)) {
          add_data("const std::vector<int>&",
                   "array[] int",
                   sprintf("idx_sigma2_%s", group))
        }
      }
    }
  }

  # ---- dprime_B fixed + REs (source_mixture) ----
  if (isTRUE(params$.has_dprime_B)) {
    add_data_tpl("matrix", "X_dprime_B")
    add_param("vector", "beta_dprime_B")

    dpB_re_idx <- 0
    if (!is.null(params$dprime_B$random)) {
      for (group in names(params$dprime_B$random)) {
        dpB_re_idx <- dpB_re_idx + 1
        re <- params$dprime_B$random[[group]]
        add_group_idx(group)
        if (isTRUE(re$use_z_matrix)) {
          add_data_tpl("matrix", sprintf("Z_dprime_B_%s", group))
        }
        var_name <- re_var_name("dprime_B", group, re$cor_id)
        if (re$dim == 1) {
          add_param("vector", var_name)
        } else {
          add_param("matrix", var_name)
        }
        if (!is.null(re$term_idx)) {
          add_data("const std::vector<int>&",
                   "array[] int",
                   sprintf("idx_dprime_B_%s", group))
        }
      }
    }
  }

  # ---- lambda_B fixed + REs (source_mixture) ----
  if (isTRUE(params$.has_lambda_B)) {
    add_data_tpl("matrix", "X_lambda_B")
    add_param("vector", "beta_lambda_B")

    lamB_re_idx <- 0
    if (!is.null(params$lambda_B$random)) {
      for (group in names(params$lambda_B$random)) {
        lamB_re_idx <- lamB_re_idx + 1
        re <- params$lambda_B$random[[group]]
        add_group_idx(group)
        if (isTRUE(re$use_z_matrix)) {
          add_data_tpl("matrix", sprintf("Z_lambda_B_%s", group))
        }
        var_name <- re_var_name("lambda_B", group, re$cor_id)
        if (re$dim == 1) {
          add_param("vector", var_name)
        } else {
          add_param("matrix", var_name)
        }
        if (!is.null(re$term_idx)) {
          add_data("const std::vector<int>&",
                   "array[] int",
                   sprintf("idx_lambda_B_%s", group))
        }
      }
    }
  }

  # ---- discrim fixed + REs (vrdp2d / bivariate) ----
  has_disc_fixed <- isTRUE(params$.has_discrim) && !is.null(params$discrim$fixed) && params$discrim$fixed$n_coef > 0
  if (has_disc_fixed) {
    add_data_tpl("matrix", "X_discrim")
    add_param("vector", "beta_discrim")
  }
  if (isTRUE(params$.has_discrim)) {
    disc_re_idx <- 0
    if (!is.null(params$discrim$random)) {
      for (group in names(params$discrim$random)) {
        disc_re_idx <- disc_re_idx + 1
        re <- params$discrim$random[[group]]
        add_group_idx(group)
        if (isTRUE(re$use_z_matrix)) {
          add_data_tpl("matrix", sprintf("Z_discrim_%s", group))
        }
        var_name <- re_var_name("discrim", group, re$cor_id)
        if (re$dim == 1) {
          add_param("vector", var_name)
        } else {
          add_param("matrix", var_name)
        }
        if (!is.null(re$term_idx)) {
          add_data("const std::vector<int>&",
                   "array[] int",
                   sprintf("idx_discrim_%s", group))
        }
      }
    }
  }

  # ---- discrim_B fixed + REs (vrdp2d) ----
  if (isTRUE(params$.has_discrim_B)) {
    add_data_tpl("matrix", "X_discrim_B")
    add_param("vector", "beta_discrim_B")

    discB_re_idx <- 0
    if (!is.null(params$discrim_B$random)) {
      for (group in names(params$discrim_B$random)) {
        discB_re_idx <- discB_re_idx + 1
        re <- params$discrim_B$random[[group]]
        add_group_idx(group)
        if (isTRUE(re$use_z_matrix)) {
          add_data_tpl("matrix", sprintf("Z_discrim_B_%s", group))
        }
        var_name <- re_var_name("discrim_B", group, re$cor_id)
        if (re$dim == 1) {
          add_param("vector", var_name)
        } else {
          add_param("matrix", var_name)
        }
        if (!is.null(re$term_idx)) {
          add_data("const std::vector<int>&",
                   "array[] int",
                   sprintf("idx_discrim_B_%s", group))
        }
      }
    }
  }

  # ---- dprime_L fixed + REs (lure mixture) ----
  if (isTRUE(params$.has_dprime_L)) {
    add_data_tpl("matrix", "X_dprime_L")
    add_param("vector", "beta_dprime_L")

    dpL_re_idx <- 0
    if (!is.null(params$dprime_L$random)) {
      for (group in names(params$dprime_L$random)) {
        dpL_re_idx <- dpL_re_idx + 1
        re <- params$dprime_L$random[[group]]
        add_group_idx(group)
        if (isTRUE(re$use_z_matrix)) {
          add_data_tpl("matrix", sprintf("Z_dprime_L_%s", group))
        }
        var_name <- re_var_name("dprime_L", group, re$cor_id)
        if (re$dim == 1) {
          add_param("vector", var_name)
        } else {
          add_param("matrix", var_name)
        }
        if (!is.null(re$term_idx)) {
          add_data("const std::vector<int>&",
                   "array[] int",
                   sprintf("idx_dprime_L_%s", group))
        }
      }
    }
  }

  # ---- sigma_L fixed + REs (lure mixture) ----
  if (isTRUE(params$.has_sigma_L)) {
    add_data_tpl("matrix", "X_sigma_L")
    add_param("vector", "beta_sigma_L")

    sigL_re_idx <- 0
    if (!is.null(params$sigma_L$random)) {
      for (group in names(params$sigma_L$random)) {
        sigL_re_idx <- sigL_re_idx + 1
        re <- params$sigma_L$random[[group]]
        add_group_idx(group)
        if (isTRUE(re$use_z_matrix)) {
          add_data_tpl("matrix", sprintf("Z_sigma_L_%s", group))
        }
        var_name <- re_var_name("sigma_L", group, re$cor_id)
        if (re$dim == 1) {
          add_param("vector", var_name)
        } else {
          add_param("matrix", var_name)
        }
        if (!is.null(re$term_idx)) {
          add_data("const std::vector<int>&",
                   "array[] int",
                   sprintf("idx_sigma_L_%s", group))
        }
      }
    }
  }

  # ---- lambda_L fixed + REs (lure mixture) ----
  if (isTRUE(params$.has_lambda_L)) {
    add_data_tpl("matrix", "X_lambda_L")
    add_param("vector", "beta_lambda_L")

    lamL_re_idx <- 0
    if (!is.null(params$lambda_L$random)) {
      for (group in names(params$lambda_L$random)) {
        lamL_re_idx <- lamL_re_idx + 1
        re <- params$lambda_L$random[[group]]
        add_group_idx(group)
        if (isTRUE(re$use_z_matrix)) {
          add_data_tpl("matrix", sprintf("Z_lambda_L_%s", group))
        }
        var_name <- re_var_name("lambda_L", group, re$cor_id)
        if (re$dim == 1) {
          add_param("vector", var_name)
        } else {
          add_param("matrix", var_name)
        }
        if (!is.null(re$term_idx)) {
          add_data("const std::vector<int>&",
                   "array[] int",
                   sprintf("idx_lambda_L_%s", group))
        }
      }
    }
  }

  # ---- sigma_B fixed + REs (bivariate) ----
  if (isTRUE(params$.has_sigma_B)) {
    add_data_tpl("matrix", "X_sigma_B")
    add_param("vector", "beta_sigma_B")
    if (!is.null(params$sigma_B$random)) {
      for (group in names(params$sigma_B$random)) {
        re <- params$sigma_B$random[[group]]
        add_group_idx(group)
        if (isTRUE(re$use_z_matrix)) add_data_tpl("matrix", sprintf("Z_sigma_B_%s", group))
        var_name <- re_var_name("sigma_B", group, re$cor_id)
        if (re$dim == 1) add_param("vector", var_name) else add_param("matrix", var_name)
        if (!is.null(re$term_idx)) add_data("const std::vector<int>&", "array[] int", sprintf("idx_sigma_B_%s", group))
      }
    }
  }

  # ---- sigma2_B fixed + REs (bivariate) ----
  if (isTRUE(params$.has_sigma2_B)) {
    add_data_tpl("matrix", "X_sigma2_B")
    add_param("vector", "beta_sigma2_B")
    if (!is.null(params$sigma2_B$random)) {
      for (group in names(params$sigma2_B$random)) {
        re <- params$sigma2_B$random[[group]]
        add_group_idx(group)
        if (isTRUE(re$use_z_matrix)) add_data_tpl("matrix", sprintf("Z_sigma2_B_%s", group))
        var_name <- re_var_name("sigma2_B", group, re$cor_id)
        if (re$dim == 1) add_param("vector", var_name) else add_param("matrix", var_name)
        if (!is.null(re$term_idx)) add_data("const std::vector<int>&", "array[] int", sprintf("idx_sigma2_B_%s", group))
      }
    }
  }

  # ---- rho fixed + REs (bivariate) ----
  if (isTRUE(params$.has_rho)) {
    add_data_tpl("matrix", "X_rho")
    add_param("vector", "beta_rho")
    if (!is.null(params$rho$random)) {
      for (group in names(params$rho$random)) {
        re <- params$rho$random[[group]]
        add_group_idx(group)
        if (isTRUE(re$use_z_matrix)) add_data_tpl("matrix", sprintf("Z_rho_%s", group))
        var_name <- re_var_name("rho", group, re$cor_id)
        if (re$dim == 1) add_param("vector", var_name) else add_param("matrix", var_name)
        if (!is.null(re$term_idx)) add_data("const std::vector<int>&", "array[] int", sprintf("idx_rho_%s", group))
      }
    }
  }

  # ---- rho_B fixed + REs (bivariate) ----
  if (isTRUE(params$.has_rho_B)) {
    add_data_tpl("matrix", "X_rho_B")
    add_param("vector", "beta_rho_B")
    if (!is.null(params$rho_B$random)) {
      for (group in names(params$rho_B$random)) {
        re <- params$rho_B$random[[group]]
        add_group_idx(group)
        if (isTRUE(re$use_z_matrix)) add_data_tpl("matrix", sprintf("Z_rho_B_%s", group))
        var_name <- re_var_name("rho_B", group, re$cor_id)
        if (re$dim == 1) add_param("vector", var_name) else add_param("matrix", var_name)
        if (!is.null(re$term_idx)) add_data("const std::vector<int>&", "array[] int", sprintf("idx_rho_B_%s", group))
      }
    }
  }

  # ---- rho_N fixed + REs (bivariate) ----
  if (isTRUE(params$.has_rho_N)) {
    add_data_tpl("matrix", "X_rho_N")
    add_param("vector", "beta_rho_N")
    if (!is.null(params$rho_N$random)) {
      for (group in names(params$rho_N$random)) {
        re <- params$rho_N$random[[group]]
        add_group_idx(group)
        if (isTRUE(re$use_z_matrix)) add_data_tpl("matrix", sprintf("Z_rho_N_%s", group))
        var_name <- re_var_name("rho_N", group, re$cor_id)
        if (re$dim == 1) add_param("vector", var_name) else add_param("matrix", var_name)
        if (!is.null(re$term_idx)) add_data("const std::vector<int>&", "array[] int", sprintf("idx_rho_N_%s", group))
      }
    }
  }

  # ---- lambda2 fixed + REs (bivariate_dp) ----
  if (isTRUE(params$.has_lambda2)) {
    add_data_tpl("matrix", "X_lambda2")
    add_param("vector", "beta_lambda2")
    if (!is.null(params$lambda2$random)) {
      for (group in names(params$lambda2$random)) {
        re <- params$lambda2$random[[group]]
        add_group_idx(group)
        if (isTRUE(re$use_z_matrix)) add_data_tpl("matrix", sprintf("Z_lambda2_%s", group))
        var_name <- re_var_name("lambda2", group, re$cor_id)
        if (re$dim == 1) add_param("vector", var_name) else add_param("matrix", var_name)
        if (!is.null(re$term_idx)) add_data("const std::vector<int>&", "array[] int", sprintf("idx_lambda2_%s", group))
      }
    }
  }

  # ---- lambda2_B fixed + REs (bivariate_dp Source B; optional) ----
  if (isTRUE(params$.has_lambda2_B)) {
    add_data_tpl("matrix", "X_lambda2_B")
    add_param("vector", "beta_lambda2_B")
    if (!is.null(params$lambda2_B$random)) {
      for (group in names(params$lambda2_B$random)) {
        re <- params$lambda2_B$random[[group]]
        add_group_idx(group)
        if (isTRUE(re$use_z_matrix)) add_data_tpl("matrix", sprintf("Z_lambda2_B_%s", group))
        var_name <- re_var_name("lambda2_B", group, re$cor_id)
        if (re$dim == 1) add_param("vector", var_name) else add_param("matrix", var_name)
        if (!is.null(re$term_idx)) add_data("const std::vector<int>&", "array[] int", sprintf("idx_lambda2_B_%s", group))
      }
    }
  }

  # ---- rec_crit fixed + REs (cdp) ----
  if (isTRUE(params$.has_rec_crit)) {
    add_data_tpl("matrix", "X_rec_crit")
    add_param("vector", "beta_rec_crit")
    if (!is.null(params$rec_crit$random)) {
      for (group in names(params$rec_crit$random)) {
        re <- params$rec_crit$random[[group]]
        add_group_idx(group)
        if (isTRUE(re$use_z_matrix)) add_data_tpl("matrix", sprintf("Z_rec_crit_%s", group))
        var_name <- re_var_name("rec_crit", group, re$cor_id)
        if (re$dim == 1) add_param("vector", var_name) else add_param("matrix", var_name)
        if (!is.null(re$term_idx)) add_data("const std::vector<int>&", "array[] int", sprintf("idx_rec_crit_%s", group))
      }
    }
  }

  # ---- know_crit fixed + REs (cdp) ----
  if (isTRUE(params$.has_know_crit)) {
    add_data_tpl("matrix", "X_know_crit")
    add_param("vector", "beta_know_crit")
    if (!is.null(params$know_crit$random)) {
      for (group in names(params$know_crit$random)) {
        re <- params$know_crit$random[[group]]
        add_group_idx(group)
        if (isTRUE(re$use_z_matrix)) add_data_tpl("matrix", sprintf("Z_know_crit_%s", group))
        var_name <- re_var_name("know_crit", group, re$cor_id)
        if (re$dim == 1) add_param("vector", var_name) else add_param("matrix", var_name)
        if (!is.null(re$term_idx)) add_data("const std::vector<int>&", "array[] int", sprintf("idx_know_crit_%s", group))
      }
    }
  }

  # ---- Criterion population (dimension 1 for vrdp2d/bivariate, or standard) ----
  crit <- model_data$criterion
  is_crit_intercept_only <- isTRUE(crit$is_intercept_only) || crit$n_coef == 1
  # For CDP, criterion thresholds cover J levels; otherwise K levels
  K_for_crit <- if (has_two_responses) model_data$stan_data$K1
                else if (isTRUE(params$.is_cdp)) model_data$stan_data$J + 1  # J thresholds need J+1 "categories"
                else model_data$K
  if (!is_crit_intercept_only) {
    add_data_tpl("matrix", "X_criterion")
  }
  add_param("vector", "beta_thresh_mid")
  if (K_for_crit > 2) {
    add_param("matrix", "beta_log_gaps")
  }

  # ---- Criterion REs ----
  if (!is.null(params$criterion$random)) {
    for (group in names(params$criterion$random)) {
      re <- params$criterion$random[[group]]
      add_group_idx(group)
      var_name <- re_var_name("criterion", group, re$cor_id)
      add_param("matrix", var_name)

      n_re_terms <- if (!is.null(re$n_re_terms)) re$n_re_terms else 1
      if (n_re_terms > 1 && !isTRUE(re$use_z_matrix)) {
        add_data("const std::vector<int>&",
                 "array[] int",
                 sprintf("idx_criterion_%s", group))
      }
      if (isTRUE(re$use_z_matrix)) {
        add_data_tpl("matrix", sprintf("Z_criterion_%s", group))
      }
    }
  }

  # ---- Criterion2 population (vrdp2d/bivariate -- source dimension thresholds) ----
  if (has_two_responses) {
    varying_sc <- isTRUE(params$.varying_source_criteria)
    if (varying_sc) {
      # Varying source criteria: K1 sets of thresholds
      add_param("vector", "beta_thresh_mid_2_varying")
      K2_val <- model_data$stan_data$K2
      if (K2_val > 2) {
        add_param("matrix", "beta_log_gaps_2_varying")
      }
      # Shared new-item thresholds if applicable
      if (identical(params$.new_source_criteria, "shared")) {
        add_param("real", "beta_thresh_mid_2_new")
        if (K2_val > 2) {
          add_param("vector", "beta_log_gaps_2_new")
        }
      }
      # Criterion2 REs (if any -- varying mode)
      if (!is.null(params$criterion2$random)) {
        for (group in names(params$criterion2$random)) {
          re <- params$criterion2$random[[group]]
          add_group_idx(group)
          var_name <- re_var_name("criterion2", group, re$cor_id)
          crit2_re_type <- if (re$dim == 1) "vector" else "matrix"
          add_param(crit2_re_type, var_name)
        }
      }
    } else {
      crit2 <- model_data$criterion2
      is_crit2_intercept_only <- isTRUE(crit2$is_intercept_only) || crit2$n_coef == 1
      if (!is_crit2_intercept_only) {
        add_data_tpl("matrix", "X_criterion2")
      }
      add_param("vector", "beta_thresh_mid_2")
      K2_val <- model_data$stan_data$K2
      if (K2_val > 2) {
        add_param("matrix", "beta_log_gaps_2")
      }

      # ---- Criterion2 REs ----
      if (!is.null(params$criterion2$random)) {
        for (group in names(params$criterion2$random)) {
          re <- params$criterion2$random[[group]]
          add_group_idx(group)
          var_name <- re_var_name("criterion2", group, re$cor_id)
          # dim=1 intercept-only RE is a vector in Stan, multi-dim is a matrix
          crit2_re_type <- if (re$dim == 1) "vector" else "matrix"
          add_param(crit2_re_type, var_name)

          n_re_terms <- if (!is.null(re$n_re_terms)) re$n_re_terms else 1
          if (n_re_terms > 1 && !isTRUE(re$use_z_matrix)) {
            add_data("const std::vector<int>&",
                     "array[] int",
                     sprintf("idx_criterion2_%s", group))
          }
          if (isTRUE(re$use_z_matrix)) {
            add_data_tpl("matrix", sprintf("Z_criterion2_%s", group))
          }
        }
      }
    }
  }

  # ---- smooth terms: Zs data matrices and s_ parameter vectors/matrices ----
  if (!is.null(model_data$smooth_data)) {
    for (pname in names(model_data$smooth_data)) {
      for (sm in model_data$smooth_data[[pname]]) {
        n_thresh_sm <- if (!is.null(sm$n_thresh)) sm$n_thresh else 0L
        for (comp in sm$components) {
          for (k in seq_along(comp$Zs_list)) {
            zs_name <- paste0("Zs_", pname, "_", comp$san_label, "_", k)
            s_name <- paste0("s_", pname, "_", comp$san_label, "_", k)
            add_data_tpl("matrix", zs_name)
            if (n_thresh_sm > 0) {
              add_param("matrix", s_name)  # [n_thresh, nbasis]
            } else {
              add_param("vector", s_name)
            }
          }
        }
      }
    }
  }

  # ---- pstream ----
  args[[length(args) + 1]] <- list(
    cpp_type  = "std::ostream*",
    stan_type = NULL,
    stan_name = NULL,
    tpl_name  = NULL,
    is_var    = FALSE,
    role      = "pstream"
  )

  args
}


#' Helper: construct RE variable name
#' @noRd
re_var_name <- function(param_name, group, cor_id) {
  if (!is.null(cor_id)) {
    sprintf("u_%s_%s_from_%s", param_name, group, cor_id)
  } else {
    sprintf("u_%s_%s", param_name, group)
  }
}


# ============================================================================
# Stan declaration generator
# ============================================================================

#' Generate Stan function declaration from signature
#' @noRd
generate_batch_stan_decl <- function(sig) {
  parts <- character(0)
  for (a in sig) {
    if (a$role == "pstream") next
    parts <- c(parts, sprintf("%s %s", a$stan_type, a$stan_name))
  }
  arg_str <- paste(parts, collapse = ",\n      ")
  sprintf("  real batch_sdt_loglik(\n      %s);", arg_str)
}


# ============================================================================
# Stan call generator
# ============================================================================

#' Generate Stan model block call
#' @noRd
generate_batch_stan_call <- function(sig, threaded = FALSE) {
  # Deduplicate: same stan_name should only appear once
  seen <- character(0)
  parts <- character(0)
  for (a in sig) {
    if (a$role == "pstream") next
    nm <- a$stan_name
    stan_call_name <- nm
    if (nm == "is_old_d") stan_call_name <- "is_old"
    if (nm == "counts_d") stan_call_name <- "counts"
    # start_idx/end_idx: use 1/N for non-threaded, start/end for threaded
    if (nm == "start_idx") {
      stan_call_name <- if (threaded) "start" else "1"
    }
    if (nm == "end_idx") {
      stan_call_name <- if (threaded) "end" else "N"
    }
    if (stan_call_name %in% seen) next
    seen <- c(seen, stan_call_name)
    parts <- c(parts, stan_call_name)
  }
  arg_str <- paste(parts, collapse = ",\n      ")
  sprintf("  target += batch_sdt_loglik(\n      %s);", arg_str)
}


#' Generate partial_log_lik wrapper for reduce_sum threading
#' @noRd
generate_batch_partial_log_lik <- function(sig) {
  # Build the parameter declarations for partial_log_lik_lpmf
  # seq_n is sliced by reduce_sum; everything else (including y) is shared
  shared_decls <- character(0)
  shared_names <- character(0)
  seen <- character(0)

  for (a in sig) {
    if (a$role == "pstream") next
    nm <- a$stan_name
    if (nm %in% c("start_idx", "end_idx")) next

    stan_call_name <- nm
    if (nm == "is_old_d") stan_call_name <- "is_old"
    if (nm == "counts_d") stan_call_name <- "counts"
    if (stan_call_name %in% seen) next
    seen <- c(seen, stan_call_name)

    stan_type <- a$stan_type
    shared_decls <- c(shared_decls, sprintf("%s %s", stan_type, stan_call_name))
    shared_names <- c(shared_names, stan_call_name)
  }

  # Build the batch_sdt_loglik call inside the wrapper
  batch_call <- generate_batch_stan_call(sig, threaded = TRUE)
  # Replace "target +=" with "return" since partial returns a value
  batch_call <- sub("  target \\+=", "    return", batch_call)

  decl_str <- paste(shared_decls, collapse = ",\n      ")

  lines <- c(
    "  // Batch partial log-likelihood for reduce_sum threading",
    "  real partial_log_lik_lpmf(array[] int seq_n_slice, int start, int end,",
    paste0("      ", decl_str, ") {"),
    batch_call,
    "  }"
  )
  paste(lines, collapse = "\n")
}


#' Generate reduce_sum call for threaded batch likelihood
#' @noRd
generate_batch_reduce_sum_call <- function(sig) {
  # Arguments to reduce_sum: the shared params (same order as partial_log_lik)
  seen <- character(0)
  parts <- character(0)

  for (a in sig) {
    if (a$role == "pstream") next
    nm <- a$stan_name
    if (nm %in% c("start_idx", "end_idx")) next

    stan_call_name <- nm
    if (nm == "is_old_d") stan_call_name <- "is_old"
    if (nm == "counts_d") stan_call_name <- "counts"
    if (stan_call_name %in% seen) next
    seen <- c(seen, stan_call_name)
    parts <- c(parts, stan_call_name)
  }

  arg_str <- paste(parts, collapse = ",\n      ")
  paste(c(
    "  // Threaded batch likelihood via reduce_sum",
    "  target += reduce_sum(partial_log_lik_lpmf, seq_n, grainsize,",
    paste0("      ", arg_str, ");")
  ), collapse = "\n")
}


# ============================================================================
# C++ code generator
# ============================================================================

#' Generate the complete C++ batch function
#' @noRd
generate_batch_cpp_code <- function(model_data, family_name, params, sig) {
  L <- new_line_builder()

  # --- Include guard ---
  # Note: batch_sdt_core.hpp is included via the wrapper header written by fit_broc()
  L$add("#ifndef BATCH_SDT_LOGLIK_HPP")
  L$add("#define BATCH_SDT_LOGLIK_HPP")
  L$add("")

  # --- Template declaration ---
  tpl_names <- character(0)
  for (a in sig) {
    if (!is.null(a$tpl_name)) tpl_names <- c(tpl_names, a$tpl_name)
  }
  L$add(sprintf("template <%s>",
                paste(sprintf("typename %s", tpl_names), collapse = ", ")))

  # --- Return type + function name ---
  L$add("auto batch_sdt_loglik(")

  # --- Function parameters ---
  non_pstream <- Filter(function(a) a$role != "pstream", sig)
  pstream <- Filter(function(a) a$role == "pstream", sig)
  for (i in seq_along(non_pstream)) {
    a <- non_pstream[[i]]
    comma <- if (i < length(non_pstream) || length(pstream) > 0) "," else ""
    L$add(sprintf("    %s %s%s", a$cpp_type, a$stan_name, comma))
  }
  if (length(pstream) > 0) {
    L$add("    std::ostream* pstream__) {")
  } else {
    # shouldn't happen, but just in case
    L$add("    ) {")
  }

  L$add("")
  L$add("  using stan::math::var;")
  L$add("  using stan::math::value_of;")
  L$add("")

  # --- is_autodiff constexpr ---
  var_tpls <- character(0)
  for (a in sig) {
    if (isTRUE(a$is_var) && !is.null(a$tpl_name)) {
      var_tpls <- c(var_tpls, a$tpl_name)
    }
  }
  autodiff_checks <- paste(sprintf(
    "stan::is_var<typename stan::scalar_type<%s>::type>::value", var_tpls
  ), collapse = " ||\n      ")
  L$add(sprintf("  constexpr bool is_autodiff = %s;", autodiff_checks))
  L$add("")

  # --- Dimension constants ---
  L$add("  const int n_start = start_idx - 1;  // 0-based")
  L$add("  const int n_end = end_idx;            // exclusive upper bound")
  has_two_responses <- isTRUE(params$.is_vrdp2d) || isTRUE(params$.is_bivariate)
  if (has_two_responses) {
    L$add("  const int n_thresh1 = K1 - 1;")
    L$add("  const int n_gaps1 = n_thresh1 - 1;")
    L$add("  const int mid1 = mid_thresh1 - 1;")
    L$add("  const int n_upper1 = n_thresh1 - mid1 - 1;")
    L$add("  const int n_thresh2 = K2 - 1;")
    L$add("  const int n_gaps2 = n_thresh2 - 1;")
    L$add("  const int mid2 = mid_thresh2 - 1;")
    L$add("  const int n_upper2 = n_thresh2 - mid2 - 1;")
    # Aliases for criterion code that uses n_thresh/n_gaps/mid/n_upper
    L$add("  const int n_thresh = n_thresh1;")
    L$add("  const int n_gaps = n_gaps1;")
    L$add("  const int mid = mid1;")
    L$add("  const int n_upper = n_upper1;")
  } else if (isTRUE(params$.is_cdp)) {
    # CDP: J thresholds (tau), J-1 gaps
    L$add("  const int n_thresh = J;")
    L$add("  const int n_gaps = n_thresh - 1;")
    L$add("  const int mid = mid_thresh_cdp - 1;")
    L$add("  const int n_upper = n_thresh - mid - 1;")
  } else {
    L$add("  const int n_thresh = K - 1;")
    L$add("  const int n_gaps = n_thresh - 1;")
    L$add("  const int mid = mid_thresh - 1;")
    L$add("  const int n_upper = n_thresh - mid - 1;")
  }
  L$add("")

  # --- Extract doubles ---
  emit_value_extractions(L, model_data, family_name, params)
  L$add("")

  # --- Gradient accumulators ---
  emit_gradient_accumulators(L, model_data, family_name, params)
  L$add("")

  # --- Criterion RE: gap-to-column mapping ---
  has_crit_re <- !is.null(params$criterion$random) && length(params$criterion$random) > 0
  has_crit2_re <- has_two_responses && !is.null(params$criterion2$random) && length(params$criterion2$random) > 0
  if (has_crit_re) {
    L$add("  // Gap index -> threshold column mapping")
    L$add("  Eigen::VectorXi gap_to_col(n_gaps);")
    L$add("  for (int g = 0; g < n_upper; ++g) gap_to_col(g) = mid + 1 + g;")
    L$add("  for (int kd = 1; kd <= mid; ++kd) gap_to_col(n_upper + kd - 1) = mid - kd;")
    L$add("")
  }
  if (has_crit2_re) {
    L$add("  // Gap index -> threshold column mapping (dimension 2)")
    L$add("  Eigen::VectorXi gap_to_col2(n_gaps2);")
    L$add("  for (int g = 0; g < n_upper2; ++g) gap_to_col2(g) = mid2 + 1 + g;")
    L$add("  for (int kd = 1; kd <= mid2; ++kd) gap_to_col2(n_upper2 + kd - 1) = mid2 - kd;")
    L$add("")
  }

  # --- Pre-loop variables ---
  L$add("  double total_lp = 0.0;")
  if (has_two_responses) {
    L$add("  Eigen::VectorXd thresh1_n(n_thresh1);")
    L$add("  Eigen::VectorXd thresh2_n(n_thresh2);")
    # Also alias for criterion code
    L$add("  Eigen::VectorXd& thresh_n = thresh1_n;")
  } else {
    L$add("  Eigen::VectorXd thresh_n(n_thresh);")
  }
  crit <- model_data$criterion
  is_crit_intercept_only_early <- isTRUE(crit$is_intercept_only) || crit$n_coef == 1
  if (has_crit_re) {
    L$add("  Eigen::VectorXd exp_eff_gaps(n_gaps);")
  } else {
    L$add("  Eigen::VectorXd pop_exp_gaps(n_gaps);")
    gap_fn <- if (identical(params$.gap_link, "softplus")) "std::log1p(std::exp(pop_gap_d(g)))" else "std::exp(pop_gap_d(g))"
    L$add(sprintf("  for (int g = 0; g < n_gaps; ++g) pop_exp_gaps(g) = %s;", gap_fn))
    if (!is_crit_intercept_only_early) {
      L$add("  Eigen::VectorXd obs_exp_gaps(n_gaps);  // per-obs exp(gap) for multi-predictor")
    }
  }
  if (has_two_responses) {
    varying_sc <- isTRUE(params$.varying_source_criteria)
    if (varying_sc) {
      # Pre-compute K1 sets of source thresholds from varying parameters
      gap_fn2 <- if (identical(params$.gap_link, "softplus")) "std::log1p(std::exp(vary_gaps2_d(k1, g)))" else "std::exp(vary_gaps2_d(k1, g))"
      L$add("  // Pre-compute varying source thresholds: thresh2_all[k1][k2]")
      L$add("  Eigen::MatrixXd thresh2_all(K1, n_thresh2);")
      L$add("  for (int k1 = 0; k1 < K1; ++k1) {")
      L$add("    thresh2_all(k1, mid2) = vary_mid2_d(k1);")
      L$add("    for (int k = mid2 + 1; k < n_thresh2; ++k) {")
      L$add("      int g = k - mid2 - 1;")
      L$add(sprintf("      thresh2_all(k1, k) = thresh2_all(k1, k - 1) + %s;", gap_fn2))
      L$add("    }")
      L$add("    for (int kd = 1; kd <= mid2; ++kd) {")
      L$add("      int k = mid2 - kd;")
      L$add("      int g = n_upper2 + kd - 1;")
      L$add(sprintf("      thresh2_all(k1, k) = thresh2_all(k1, k + 1) - %s;", gap_fn2))
      L$add("    }")
      L$add("  }")
      if (identical(params$.new_source_criteria, "shared")) {
        gap_fn_new <- if (identical(params$.gap_link, "softplus")) "std::log1p(std::exp(new_gaps2_d(g)))" else "std::exp(new_gaps2_d(g))"
        L$add("  // Pre-compute shared new-item source thresholds")
        L$add("  Eigen::VectorXd thresh2_new_vec(n_thresh2);")
        L$add("  thresh2_new_vec(mid2) = new_mid2_d;")
        L$add("  for (int k = mid2 + 1; k < n_thresh2; ++k) {")
        L$add("    int g = k - mid2 - 1;")
        L$add(sprintf("    thresh2_new_vec(k) = thresh2_new_vec(k - 1) + %s;", gap_fn_new))
        L$add("  }")
        L$add("  for (int kd = 1; kd <= mid2; ++kd) {")
        L$add("    int k = mid2 - kd;")
        L$add("    int g = n_upper2 + kd - 1;")
        L$add(sprintf("    thresh2_new_vec(k) = thresh2_new_vec(k + 1) - %s;", gap_fn_new))
        L$add("  }")
      }
      # Pre-compute exp_gaps for gradient backpropagation
      L$add("  Eigen::MatrixXd vary_exp_gaps2(K1, n_gaps2);")
      gap_fn2_raw <- if (identical(params$.gap_link, "softplus")) "std::log1p(std::exp(vary_gaps2_d(k1, g)))" else "std::exp(vary_gaps2_d(k1, g))"
      L$add("  for (int k1 = 0; k1 < K1; ++k1)")
      L$add(sprintf("    for (int g = 0; g < n_gaps2; ++g) vary_exp_gaps2(k1, g) = %s;", gap_fn2_raw))
      if (identical(params$.new_source_criteria, "shared")) {
        gap_fn_new_raw <- if (identical(params$.gap_link, "softplus")) "std::log1p(std::exp(new_gaps2_d(g)))" else "std::exp(new_gaps2_d(g))"
        L$add("  Eigen::VectorXd new_exp_gaps2(n_gaps2);")
        L$add(sprintf("  for (int g = 0; g < n_gaps2; ++g) new_exp_gaps2(g) = %s;", gap_fn_new_raw))
      }
    } else if (has_crit2_re) {
      L$add("  Eigen::VectorXd exp_eff_gaps2(n_gaps2);")
    } else {
      L$add("  Eigen::VectorXd pop_exp_gaps2(n_gaps2);")
      gap_fn2 <- if (identical(params$.gap_link, "softplus")) "std::log1p(std::exp(pop_gap2_d(g)))" else "std::exp(pop_gap2_d(g))"
      L$add(sprintf("  for (int g = 0; g < n_gaps2; ++g) pop_exp_gaps2(g) = %s;", gap_fn2))
    }
  }
  L$add("")

  # --- Observation loop ---
  L$add("  for (int n = n_start; n < n_end; ++n) {")
  L$add("    int yn = y[n];")
  if (has_two_responses) {
    L$add("    int yn2 = y2[n];")
    L$add("    int item_type_n = item_type[n];")
  } else if (isTRUE(params$.is_source_mixture)) {
    L$add("    int source_n = source[n];")
  } else if (!isTRUE(params$.is_cumulative)) {
    L$add("    double is_old_n = is_old_d[n];")
  }
  if (isTRUE(params$.is_cdp_family)) {
    L$add("    int rk_n = rk[n];")
  }
  if (params$.has_counts) {
    L$add("    double count_n = counts_d[n];")
  }
  L$add("")

  # Group index extraction
  emit_group_indices(L, model_data, family_name, params)
  L$add("")

  # Per-observation parameter computation
  emit_param_computation(L, model_data, family_name, params)
  L$add("")

  # Threshold construction
  emit_thresh_computation(L, model_data, params, has_crit_re)
  if (has_two_responses) {
    L$add("")
    if (isTRUE(params$.varying_source_criteria)) {
      # Varying source criteria: copy from pre-computed matrix based on yn
      if (identical(params$.new_source_criteria, "shared")) {
        L$add("    // Varying source thresholds: use shared new-response thresholds for 'new' responses, yn-based for 'old' responses")
        L$add("    if (is_new_response[yn - 1] == 1) {")
        L$add("      thresh2_n = thresh2_new_vec;")
        L$add("    } else {")
        L$add("      thresh2_n = thresh2_all.row(yn - 1).transpose();")
        L$add("    }")
      } else {
        L$add("    // Varying source thresholds: index by detection response yn")
        L$add("    thresh2_n = thresh2_all.row(yn - 1).transpose();")
      }
    } else {
      emit_thresh2_computation(L, model_data, params, has_crit2_re)
    }
  }
  L$add("")

  # Likelihood
  if (family_name == "vrdp2d") {
    emit_likelihood_vrdp2d(L, params)
  } else if (isTRUE(params$.is_bivariate)) {
    emit_likelihood_bivariate(L, params)
  } else if (isTRUE(params$.is_cdp_family)) {
    emit_likelihood_cdp(L, params)
  } else {
    emit_likelihood(L, family_name, params)
  }
  L$add("")

  # Gradient accumulation (only in autodiff context -- skip in double context)
  L$add("    if constexpr (is_autodiff) {")
  if (family_name == "vrdp2d") {
    emit_gradient_accumulation_vrdp2d(L, model_data, params, has_crit_re, has_crit2_re)
  } else if (isTRUE(params$.is_bivariate)) {
    emit_gradient_accumulation_bivariate(L, model_data, params, has_crit_re, has_crit2_re)
  } else if (isTRUE(params$.is_cdp_family)) {
    emit_gradient_accumulation_cdp(L, model_data, params, has_crit_re)
  } else {
    emit_gradient_accumulation(L, model_data, family_name, params, has_crit_re)
  }
    # Smooth gradient accumulation (all parameters)
    if (!is.null(params$.smooth_data)) {
      smooth_d_vars <- c(dprime = "d_dprime_n", sigma = "d_sigma_n", lambda = "d_lambda_n",
                         dprime2 = "d_dprime2_n", sigma2 = "d_sigma2_n",
                         dprime_B = "d_dprime_B_n", lambda_B = "d_lambda_B_n",
                         dprime_L = "d_dprime_L_n", sigma_L = "d_sigma_L_n", lambda_L = "d_lambda_L_n",
                         discrim = "d_discrim_n", discrim_B = "d_discrim_B_n",
                         sigma_B = "d_sigma_B_n", sigma2_B = "d_sigma2_B_n",
                         rho = "d_rho_n", rho_B = "d_rho_B_n", rho_N = "d_rho_N_n",
                         lambda2 = "d_lambda2_n", rec_crit = "d_rec_crit_n", know_crit = "d_know_crit_n")
      for (pname in names(params$.smooth_data)) {
        # criterion/criterion2 gradients are handled in the threshold propagation
        if (pname %in% c("criterion", "criterion2")) next
        d_var <- smooth_d_vars[[pname]]
        if (!is.null(d_var)) {
          emit_smooth_gradient(L, params$.smooth_data, pname, d_var)
        }
      }
    }

  L$add("    }")

  # Close observation loop
  L$add("  }")
  L$add("")

  # Operand packing
  emit_operand_packing(L, model_data, family_name, params, has_crit_re,
                       has_crit2_re = has_crit2_re)

  # Close function
  L$add("}")
  L$add("")
  L$add("#endif  // BATCH_SDT_LOGLIK_HPP")

  L$get()
}


# ============================================================================
# Line builder helper
# ============================================================================

#' Simple line accumulator
#' @noRd
new_line_builder <- function() {
  lines <- character(0)
  list(
    add = function(x) lines <<- c(lines, x),
    get = function() paste(lines, collapse = "\n")
  )
}


# ============================================================================
# Value extraction emitter
# ============================================================================

#' Emit value_of() extraction for all parameters
#' @noRd
emit_value_extractions <- function(L, model_data, family_name, params) {
  L$add("  // Extract double values from autodiff types")

  # dprime fixed
  has_dp_fixed <- isTRUE(params$.has_dp_fixed)
  if (has_dp_fixed) {
    L$add("  const Eigen::MatrixXd X_dp_d = value_of(X_dprime);")
    L$add("  const Eigen::VectorXd beta_dp_d = value_of(beta_dprime);")
    L$add("  const int P_dp = beta_dp_d.size();")
  }

  # dprime REs
  dp_re_idx <- 0
  if (!is.null(params$dprime$random)) {
    for (group in names(params$dprime$random)) {
      dp_re_idx <- dp_re_idx + 1
      re <- params$dprime$random[[group]]
      tag <- dp_re_tag(dp_re_idx)
      if (re$dim == 1) {
        L$add(sprintf("  const Eigen::VectorXd %s_d = value_of(%s);",
                       tag, re_stan_arg_name("dprime", group, re, dp_re_idx, "dp")))
      } else {
        L$add(sprintf("  const Eigen::MatrixXd %s_d = value_of(%s);",
                       tag, re_stan_arg_name("dprime", group, re, dp_re_idx, "dp")))
      }
      if (isTRUE(re$use_z_matrix)) {
        L$add(sprintf("  const Eigen::MatrixXd Z_%s_d = value_of(Z_dprime_%s);",
                       tag, group))
      }
    }
  }

  # sigma fixed + REs
  if (isTRUE(params$.has_sigma)) {
    L$add("  const Eigen::MatrixXd X_sig_d = value_of(X_sigma);")
    L$add("  const Eigen::VectorXd beta_sig_d = value_of(beta_sigma);")
    L$add("  const int P_sig = beta_sig_d.size();")

    sig_re_idx <- 0
    if (!is.null(params$sigma$random)) {
      for (group in names(params$sigma$random)) {
        sig_re_idx <- sig_re_idx + 1
        re <- params$sigma$random[[group]]
        tag <- sig_re_tag(sig_re_idx)
        if (re$dim == 1) {
          L$add(sprintf("  const Eigen::VectorXd %s_d = value_of(%s);",
                         tag, re_stan_arg_name("sigma", group, re, sig_re_idx, "sig")))
        } else {
          L$add(sprintf("  const Eigen::MatrixXd %s_d = value_of(%s);",
                         tag, re_stan_arg_name("sigma", group, re, sig_re_idx, "sig")))
        }
        if (isTRUE(re$use_z_matrix)) {
          L$add(sprintf("  const Eigen::MatrixXd Z_%s_d = value_of(Z_sigma_%s);",
                         tag, group))
        }
      }
    }
  }

  # lambda fixed + REs
  if (isTRUE(params$.has_lambda)) {
    L$add("  const Eigen::MatrixXd X_lam_d = value_of(X_lambda);")
    L$add("  const Eigen::VectorXd beta_lam_d = value_of(beta_lambda);")
    L$add("  const int P_lam = beta_lam_d.size();")

    lam_re_idx <- 0
    if (!is.null(params$lambda$random)) {
      for (group in names(params$lambda$random)) {
        lam_re_idx <- lam_re_idx + 1
        re <- params$lambda$random[[group]]
        tag <- lam_re_tag(lam_re_idx)
        if (re$dim == 1) {
          L$add(sprintf("  const Eigen::VectorXd %s_d = value_of(%s);",
                         tag, re_stan_arg_name("lambda", group, re, lam_re_idx, "lam")))
        } else {
          L$add(sprintf("  const Eigen::MatrixXd %s_d = value_of(%s);",
                         tag, re_stan_arg_name("lambda", group, re, lam_re_idx, "lam")))
        }
        if (isTRUE(re$use_z_matrix)) {
          L$add(sprintf("  const Eigen::MatrixXd Z_%s_d = value_of(Z_lambda_%s);",
                         tag, group))
        }
      }
    }
  }

  # dprime2 fixed + REs
  if (isTRUE(params$.has_dprime2)) {
    L$add("  const Eigen::MatrixXd X_dp2_d = value_of(X_dprime2);")
    L$add("  const Eigen::VectorXd beta_dp2_d = value_of(beta_dprime2);")
    L$add("  const int P_dp2 = beta_dp2_d.size();")

    dp2_re_idx <- 0
    if (!is.null(params$dprime2$random)) {
      for (group in names(params$dprime2$random)) {
        dp2_re_idx <- dp2_re_idx + 1
        re <- params$dprime2$random[[group]]
        tag <- dp2_re_tag(dp2_re_idx)
        if (re$dim == 1) {
          L$add(sprintf("  const Eigen::VectorXd %s_d = value_of(%s);",
                         tag, re_stan_arg_name("dprime2", group, re, dp2_re_idx, "dp2")))
        } else {
          L$add(sprintf("  const Eigen::MatrixXd %s_d = value_of(%s);",
                         tag, re_stan_arg_name("dprime2", group, re, dp2_re_idx, "dp2")))
        }
        if (isTRUE(re$use_z_matrix)) {
          L$add(sprintf("  const Eigen::MatrixXd Z_%s_d = value_of(Z_dprime2_%s);",
                         tag, group))
        }
      }
    }
  }

  # sigma2 fixed + REs
  if (isTRUE(params$.has_sigma2)) {
    L$add("  const Eigen::MatrixXd X_sig2_d = value_of(X_sigma2);")
    L$add("  const Eigen::VectorXd beta_sig2_d = value_of(beta_sigma2);")
    L$add("  const int P_sig2 = beta_sig2_d.size();")

    sig2_re_idx <- 0
    if (!is.null(params$sigma2$random)) {
      for (group in names(params$sigma2$random)) {
        sig2_re_idx <- sig2_re_idx + 1
        re <- params$sigma2$random[[group]]
        tag <- sig2_re_tag(sig2_re_idx)
        if (re$dim == 1) {
          L$add(sprintf("  const Eigen::VectorXd %s_d = value_of(%s);",
                         tag, re_stan_arg_name("sigma2", group, re, sig2_re_idx, "sig2")))
        } else {
          L$add(sprintf("  const Eigen::MatrixXd %s_d = value_of(%s);",
                         tag, re_stan_arg_name("sigma2", group, re, sig2_re_idx, "sig2")))
        }
        if (isTRUE(re$use_z_matrix)) {
          L$add(sprintf("  const Eigen::MatrixXd Z_%s_d = value_of(Z_sigma2_%s);",
                         tag, group))
        }
      }
    }
  }

  # dprime_B fixed + REs (source_mixture)
  if (isTRUE(params$.has_dprime_B)) {
    L$add("  const Eigen::MatrixXd X_dpB_d = value_of(X_dprime_B);")
    L$add("  const Eigen::VectorXd beta_dpB_d = value_of(beta_dprime_B);")
    L$add("  const int P_dpB = beta_dpB_d.size();")

    dpB_re_idx <- 0
    if (!is.null(params$dprime_B$random)) {
      for (group in names(params$dprime_B$random)) {
        dpB_re_idx <- dpB_re_idx + 1
        re <- params$dprime_B$random[[group]]
        tag <- dpB_re_tag(dpB_re_idx)
        if (re$dim == 1) {
          L$add(sprintf("  const Eigen::VectorXd %s_d = value_of(%s);",
                         tag, re_stan_arg_name("dprime_B", group, re, dpB_re_idx, "dpB")))
        } else {
          L$add(sprintf("  const Eigen::MatrixXd %s_d = value_of(%s);",
                         tag, re_stan_arg_name("dprime_B", group, re, dpB_re_idx, "dpB")))
        }
        if (isTRUE(re$use_z_matrix)) {
          L$add(sprintf("  const Eigen::MatrixXd Z_%s_d = value_of(Z_dprime_B_%s);",
                         tag, group))
        }
      }
    }
  }

  # lambda_B fixed + REs (source_mixture)
  if (isTRUE(params$.has_lambda_B)) {
    L$add("  const Eigen::MatrixXd X_lamB_d = value_of(X_lambda_B);")
    L$add("  const Eigen::VectorXd beta_lamB_d = value_of(beta_lambda_B);")
    L$add("  const int P_lamB = beta_lamB_d.size();")

    lamB_re_idx <- 0
    if (!is.null(params$lambda_B$random)) {
      for (group in names(params$lambda_B$random)) {
        lamB_re_idx <- lamB_re_idx + 1
        re <- params$lambda_B$random[[group]]
        tag <- lamB_re_tag(lamB_re_idx)
        if (re$dim == 1) {
          L$add(sprintf("  const Eigen::VectorXd %s_d = value_of(%s);",
                         tag, re_stan_arg_name("lambda_B", group, re, lamB_re_idx, "lamB")))
        } else {
          L$add(sprintf("  const Eigen::MatrixXd %s_d = value_of(%s);",
                         tag, re_stan_arg_name("lambda_B", group, re, lamB_re_idx, "lamB")))
        }
        if (isTRUE(re$use_z_matrix)) {
          L$add(sprintf("  const Eigen::MatrixXd Z_%s_d = value_of(Z_lambda_B_%s);",
                         tag, group))
        }
      }
    }
  }

  # dprime_L fixed + REs (lure mixture)
  if (isTRUE(params$.has_dprime_L)) {
    L$add("  const Eigen::MatrixXd X_dpL_d = value_of(X_dprime_L);")
    L$add("  const Eigen::VectorXd beta_dpL_d = value_of(beta_dprime_L);")
    L$add("  const int P_dpL = beta_dpL_d.size();")

    dpL_re_idx <- 0
    if (!is.null(params$dprime_L$random)) {
      for (group in names(params$dprime_L$random)) {
        dpL_re_idx <- dpL_re_idx + 1
        re <- params$dprime_L$random[[group]]
        tag <- dpL_re_tag(dpL_re_idx)
        if (re$dim == 1) {
          L$add(sprintf("  const Eigen::VectorXd %s_d = value_of(%s);",
                         tag, re_stan_arg_name("dprime_L", group, re, dpL_re_idx, "dpL")))
        } else {
          L$add(sprintf("  const Eigen::MatrixXd %s_d = value_of(%s);",
                         tag, re_stan_arg_name("dprime_L", group, re, dpL_re_idx, "dpL")))
        }
        if (isTRUE(re$use_z_matrix)) {
          L$add(sprintf("  const Eigen::MatrixXd Z_%s_d = value_of(Z_dprime_L_%s);",
                         tag, group))
        }
      }
    }
  }

  # sigma_L fixed + REs (lure mixture)
  if (isTRUE(params$.has_sigma_L)) {
    L$add("  const Eigen::MatrixXd X_sigL_d = value_of(X_sigma_L);")
    L$add("  const Eigen::VectorXd beta_sigL_d = value_of(beta_sigma_L);")
    L$add("  const int P_sigL = beta_sigL_d.size();")

    sigL_re_idx <- 0
    if (!is.null(params$sigma_L$random)) {
      for (group in names(params$sigma_L$random)) {
        sigL_re_idx <- sigL_re_idx + 1
        re <- params$sigma_L$random[[group]]
        tag <- sigL_re_tag(sigL_re_idx)
        if (re$dim == 1) {
          L$add(sprintf("  const Eigen::VectorXd %s_d = value_of(%s);",
                         tag, re_stan_arg_name("sigma_L", group, re, sigL_re_idx, "sigL")))
        } else {
          L$add(sprintf("  const Eigen::MatrixXd %s_d = value_of(%s);",
                         tag, re_stan_arg_name("sigma_L", group, re, sigL_re_idx, "sigL")))
        }
        if (isTRUE(re$use_z_matrix)) {
          L$add(sprintf("  const Eigen::MatrixXd Z_%s_d = value_of(Z_sigma_L_%s);",
                         tag, group))
        }
      }
    }
  }

  # lambda_L fixed + REs (lure mixture)
  if (isTRUE(params$.has_lambda_L)) {
    L$add("  const Eigen::MatrixXd X_lamL_d = value_of(X_lambda_L);")
    L$add("  const Eigen::VectorXd beta_lamL_d = value_of(beta_lambda_L);")
    L$add("  const int P_lamL = beta_lamL_d.size();")

    lamL_re_idx <- 0
    if (!is.null(params$lambda_L$random)) {
      for (group in names(params$lambda_L$random)) {
        lamL_re_idx <- lamL_re_idx + 1
        re <- params$lambda_L$random[[group]]
        tag <- lamL_re_tag(lamL_re_idx)
        if (re$dim == 1) {
          L$add(sprintf("  const Eigen::VectorXd %s_d = value_of(%s);",
                         tag, re_stan_arg_name("lambda_L", group, re, lamL_re_idx, "lamL")))
        } else {
          L$add(sprintf("  const Eigen::MatrixXd %s_d = value_of(%s);",
                         tag, re_stan_arg_name("lambda_L", group, re, lamL_re_idx, "lamL")))
        }
        if (isTRUE(re$use_z_matrix)) {
          L$add(sprintf("  const Eigen::MatrixXd Z_%s_d = value_of(Z_lambda_L_%s);",
                         tag, group))
        }
      }
    }
  }

  # discrim fixed + REs (vrdp2d / bivariate)
  has_disc_fixed_cpp <- isTRUE(params$.has_disc_fixed)
  if (isTRUE(params$.has_discrim)) {
    if (has_disc_fixed_cpp) {
      L$add("  const Eigen::MatrixXd X_disc_d = value_of(X_discrim);")
      L$add("  const Eigen::VectorXd beta_disc_d = value_of(beta_discrim);")
      L$add("  const int P_disc = beta_disc_d.size();")
    }

    disc_re_idx <- 0
    if (!is.null(params$discrim$random)) {
      for (group in names(params$discrim$random)) {
        disc_re_idx <- disc_re_idx + 1
        re <- params$discrim$random[[group]]
        tag <- disc_re_tag(disc_re_idx)
        if (re$dim == 1) {
          L$add(sprintf("  const Eigen::VectorXd %s_d = value_of(%s);",
                         tag, re_stan_arg_name("discrim", group, re, disc_re_idx, "disc")))
        } else {
          L$add(sprintf("  const Eigen::MatrixXd %s_d = value_of(%s);",
                         tag, re_stan_arg_name("discrim", group, re, disc_re_idx, "disc")))
        }
        if (isTRUE(re$use_z_matrix)) {
          L$add(sprintf("  const Eigen::MatrixXd Z_%s_d = value_of(Z_discrim_%s);",
                         tag, group))
        }
      }
    }
  }

  # discrim_B fixed + REs (vrdp2d)
  if (isTRUE(params$.has_discrim_B)) {
    L$add("  const Eigen::MatrixXd X_discB_d = value_of(X_discrim_B);")
    L$add("  const Eigen::VectorXd beta_discB_d = value_of(beta_discrim_B);")
    L$add("  const int P_discB = beta_discB_d.size();")

    discB_re_idx <- 0
    if (!is.null(params$discrim_B$random)) {
      for (group in names(params$discrim_B$random)) {
        discB_re_idx <- discB_re_idx + 1
        re <- params$discrim_B$random[[group]]
        tag <- discB_re_tag(discB_re_idx)
        if (re$dim == 1) {
          L$add(sprintf("  const Eigen::VectorXd %s_d = value_of(%s);",
                         tag, re_stan_arg_name("discrim_B", group, re, discB_re_idx, "discB")))
        } else {
          L$add(sprintf("  const Eigen::MatrixXd %s_d = value_of(%s);",
                         tag, re_stan_arg_name("discrim_B", group, re, discB_re_idx, "discB")))
        }
        if (isTRUE(re$use_z_matrix)) {
          L$add(sprintf("  const Eigen::MatrixXd Z_%s_d = value_of(Z_discrim_B_%s);",
                         tag, group))
        }
      }
    }
  }

  # sigma_B fixed + REs (bivariate)
  if (isTRUE(params$.has_sigma_B)) {
    emit_param_value_extraction(L, "sigma_B", "sigB", params$sigma_B$random, "sigma_B")
  }

  # sigma2_B fixed + REs (bivariate)
  if (isTRUE(params$.has_sigma2_B)) {
    emit_param_value_extraction(L, "sigma2_B", "sig2B", params$sigma2_B$random, "sigma2_B")
  }

  # rho fixed + REs (bivariate)
  if (isTRUE(params$.has_rho)) {
    emit_param_value_extraction(L, "rho", "rho", params$rho$random, "rho")
  }

  # rho_B fixed + REs (bivariate)
  if (isTRUE(params$.has_rho_B)) {
    emit_param_value_extraction(L, "rho_B", "rhoB", params$rho_B$random, "rho_B")
  }

  # rho_N fixed + REs (bivariate)
  if (isTRUE(params$.has_rho_N)) {
    emit_param_value_extraction(L, "rho_N", "rhoN", params$rho_N$random, "rho_N")
  }

  # lambda2 fixed + REs (bivariate_dp)
  if (isTRUE(params$.has_lambda2)) {
    emit_param_value_extraction(L, "lambda2", "lam2", params$lambda2$random, "lambda2")
  }

  # lambda2_B fixed + REs (bivariate_dp Source B)
  if (isTRUE(params$.has_lambda2_B)) {
    emit_param_value_extraction(L, "lambda2_B", "lam2B", params$lambda2_B$random, "lambda2_B")
  }

  # rec_crit fixed + REs (cdp)
  if (isTRUE(params$.has_rec_crit)) {
    emit_param_value_extraction(L, "rec_crit", "rc", params$rec_crit$random, "rec_crit")
  }

  # know_crit fixed + REs (cdp)
  if (isTRUE(params$.has_know_crit)) {
    emit_param_value_extraction(L, "know_crit", "kc", params$know_crit$random, "know_crit")
  }

  # CDP: extract old_level_map as int array
  if (isTRUE(params$.is_cdp)) {
    L$add("  // CDP old_level_map (already int)")
  }

  # Criterion population
  crit <- model_data$criterion
  is_crit_intercept_only <- isTRUE(crit$is_intercept_only) || crit$n_coef == 1
  if (!is_crit_intercept_only) {
    L$add("  const Eigen::MatrixXd X_crit_d = value_of(X_criterion);")
    L$add("  const int P_crit = X_crit_d.cols();")
  }
  L$add("  const Eigen::VectorXd thresh_mid_d_vec = value_of(beta_thresh_mid);")

  has_two_responses <- isTRUE(params$.is_vrdp2d) || isTRUE(params$.is_bivariate)
  L$add("  Eigen::VectorXd pop_gap_d(n_gaps);")
  if (is_crit_intercept_only) {
    K_for_crit <- if (has_two_responses) model_data$stan_data$K1
                  else if (isTRUE(params$.is_cdp)) model_data$stan_data$J + 1
                  else model_data$K
    if (K_for_crit > 2) {
      L$add("  {")
      L$add("    const Eigen::MatrixXd gaps_val = value_of(beta_log_gaps);")
      L$add("    for (int g = 0; g < n_gaps; ++g) pop_gap_d(g) = gaps_val(g, 0);")
      L$add("  }")
    }
  } else {
    # Multi-predictor: gaps are a matrix [n_gaps x P_crit]
    L$add("  const Eigen::MatrixXd gaps_mat_d = value_of(beta_log_gaps);")
  }

  # Criterion REs
  crit_re_idx <- 0
  if (!is.null(params$criterion$random)) {
    for (group in names(params$criterion$random)) {
      crit_re_idx <- crit_re_idx + 1
      re <- params$criterion$random[[group]]
      L$add(sprintf("  const Eigen::MatrixXd u_crit%d_d = value_of(%s);",
                     crit_re_idx, re_var_name("criterion", group, re$cor_id)))
      if (isTRUE(re$use_z_matrix)) {
        L$add(sprintf("  const Eigen::MatrixXd Z_crit%d_d = value_of(Z_criterion_%s);",
                       crit_re_idx, group))
      }
    }
  }

  # Criterion2 population (vrdp2d/bivariate -- source dimension)
  if (isTRUE(params$.is_vrdp2d) || isTRUE(params$.is_bivariate)) {
    varying_sc <- isTRUE(params$.varying_source_criteria)

    if (varying_sc) {
      # Varying source criteria: extract per-bin threshold parameters
      L$add("  // Varying source criteria: K1 sets of thresholds")
      L$add("  const Eigen::VectorXd vary_mid2_d = value_of(beta_thresh_mid_2_varying);")
      K2_val <- model_data$stan_data$K2
      if (K2_val > 2) {
        L$add("  const Eigen::MatrixXd vary_gaps2_d = value_of(beta_log_gaps_2_varying);")
      }
      if (identical(params$.new_source_criteria, "shared")) {
        L$add("  const double new_mid2_d = value_of(beta_thresh_mid_2_new);")
        if (K2_val > 2) {
          L$add("  const Eigen::VectorXd new_gaps2_d = value_of(beta_log_gaps_2_new);")
        }
      }
      # Criterion2 REs (varying mode)
      crit2_re_idx <- 0
      if (!is.null(params$criterion2$random)) {
        for (group in names(params$criterion2$random)) {
          crit2_re_idx <- crit2_re_idx + 1
          re <- params$criterion2$random[[group]]
          L$add(sprintf("  const Eigen::MatrixXd u_crit2_%d_d = value_of(%s);",
                         crit2_re_idx, re_var_name("criterion2", group, re$cor_id)))
        }
      }
    } else {
      crit2 <- model_data$criterion2
      is_crit2_intercept_only <- is.null(crit2) || isTRUE(crit2$is_intercept_only) || crit2$n_coef == 1
      if (!is_crit2_intercept_only) {
        L$add("  const Eigen::MatrixXd X_crit2_d = value_of(X_criterion2);")
        L$add("  const int P_crit2 = X_crit2_d.cols();")
      }
      L$add("  const Eigen::VectorXd thresh_mid2_d_vec = value_of(beta_thresh_mid_2);")

      L$add("  Eigen::VectorXd pop_gap2_d(n_gaps2);")
      K2_val <- model_data$stan_data$K2
      if (is_crit2_intercept_only) {
        if (K2_val > 2) {
          L$add("  {")
          L$add("    const Eigen::MatrixXd gaps2_val = value_of(beta_log_gaps_2);")
          L$add("    for (int g = 0; g < n_gaps2; ++g) pop_gap2_d(g) = gaps2_val(g, 0);")
          L$add("  }")
        }
      } else {
        L$add("  const Eigen::MatrixXd gaps2_mat_d = value_of(beta_log_gaps_2);")
      }

      # Criterion2 REs
      crit2_re_idx <- 0
      if (!is.null(params$criterion2$random)) {
        for (group in names(params$criterion2$random)) {
          crit2_re_idx <- crit2_re_idx + 1
          re <- params$criterion2$random[[group]]
          L$add(sprintf("  const Eigen::MatrixXd u_crit2_%d_d = value_of(%s);",
                         crit2_re_idx, re_var_name("criterion2", group, re$cor_id)))
          if (isTRUE(re$use_z_matrix)) {
            L$add(sprintf("  const Eigen::MatrixXd Z_crit2_%d_d = value_of(Z_criterion2_%s);",
                           crit2_re_idx, group))
          }
        }
      }
    }
  }

  # Smooth value_of extractions
  if (!is.null(params$.smooth_data)) {
    for (pname in names(params$.smooth_data)) {
      for (sm in params$.smooth_data[[pname]]) {
        n_thresh_sm <- if (!is.null(sm$n_thresh)) sm$n_thresh else 0L
        for (comp in sm$components) {
          for (k in seq_along(comp$Zs_list)) {
            zs_name <- paste0("Zs_", pname, "_", comp$san_label, "_", k)
            s_name <- paste0("s_", pname, "_", comp$san_label, "_", k)
            L$add(sprintf("  const Eigen::MatrixXd %s_d = value_of(%s);", zs_name, zs_name))
            if (n_thresh_sm > 0) {
              L$add(sprintf("  const Eigen::MatrixXd %s_d = value_of(%s);", s_name, s_name))
            } else {
              L$add(sprintf("  const Eigen::VectorXd %s_d = value_of(%s);", s_name, s_name))
            }
          }
        }
      }
    }
  }
}


# ============================================================================
# Gradient accumulator emitter
# ============================================================================

#' Emit gradient accumulator declarations
#' @noRd
emit_gradient_accumulators <- function(L, model_data, family_name, params) {
  L$add("  // Gradient accumulators")

  # dprime fixed
  has_dp_fixed <- isTRUE(params$.has_dp_fixed)
  has_disc_fixed_cpp <- isTRUE(params$.has_disc_fixed)
  if (has_dp_fixed) {
    L$add("  Eigen::VectorXd grad_beta_dp = Eigen::VectorXd::Zero(P_dp);")
  }

  # dprime REs
  dp_re_idx <- 0
  if (!is.null(params$dprime$random)) {
    for (group in names(params$dprime$random)) {
      dp_re_idx <- dp_re_idx + 1
      re <- params$dprime$random[[group]]
      tag <- dp_re_tag(dp_re_idx)
      if (re$dim == 1) {
        L$add(sprintf("  Eigen::VectorXd grad_%s = Eigen::VectorXd::Zero(%s_d.size());", tag, tag))
      } else {
        L$add(sprintf("  Eigen::MatrixXd grad_%s = Eigen::MatrixXd::Zero(%s_d.rows(), %s_d.cols());", tag, tag, tag))
      }
    }
  }

  # sigma
  if (isTRUE(params$.has_sigma)) {
    L$add("  Eigen::VectorXd grad_beta_sig = Eigen::VectorXd::Zero(P_sig);")
    sig_re_idx <- 0
    if (!is.null(params$sigma$random)) {
      for (group in names(params$sigma$random)) {
        sig_re_idx <- sig_re_idx + 1
        re <- params$sigma$random[[group]]
        tag <- sig_re_tag(sig_re_idx)
        if (re$dim == 1) {
          L$add(sprintf("  Eigen::VectorXd grad_%s = Eigen::VectorXd::Zero(%s_d.size());", tag, tag))
        } else {
          L$add(sprintf("  Eigen::MatrixXd grad_%s = Eigen::MatrixXd::Zero(%s_d.rows(), %s_d.cols());", tag, tag, tag))
        }
      }
    }
  }

  # lambda
  if (isTRUE(params$.has_lambda)) {
    L$add("  Eigen::VectorXd grad_beta_lam = Eigen::VectorXd::Zero(P_lam);")
    lam_re_idx <- 0
    if (!is.null(params$lambda$random)) {
      for (group in names(params$lambda$random)) {
        lam_re_idx <- lam_re_idx + 1
        re <- params$lambda$random[[group]]
        tag <- lam_re_tag(lam_re_idx)
        if (re$dim == 1) {
          L$add(sprintf("  Eigen::VectorXd grad_%s = Eigen::VectorXd::Zero(%s_d.size());", tag, tag))
        } else {
          L$add(sprintf("  Eigen::MatrixXd grad_%s = Eigen::MatrixXd::Zero(%s_d.rows(), %s_d.cols());", tag, tag, tag))
        }
      }
    }
  }

  # dprime2
  if (isTRUE(params$.has_dprime2)) {
    L$add("  Eigen::VectorXd grad_beta_dp2 = Eigen::VectorXd::Zero(P_dp2);")
    dp2_re_idx <- 0
    if (!is.null(params$dprime2$random)) {
      for (group in names(params$dprime2$random)) {
        dp2_re_idx <- dp2_re_idx + 1
        re <- params$dprime2$random[[group]]
        tag <- dp2_re_tag(dp2_re_idx)
        if (re$dim == 1) {
          L$add(sprintf("  Eigen::VectorXd grad_%s = Eigen::VectorXd::Zero(%s_d.size());", tag, tag))
        } else {
          L$add(sprintf("  Eigen::MatrixXd grad_%s = Eigen::MatrixXd::Zero(%s_d.rows(), %s_d.cols());", tag, tag, tag))
        }
      }
    }
  }

  # sigma2
  if (isTRUE(params$.has_sigma2)) {
    L$add("  Eigen::VectorXd grad_beta_sig2 = Eigen::VectorXd::Zero(P_sig2);")
    sig2_re_idx <- 0
    if (!is.null(params$sigma2$random)) {
      for (group in names(params$sigma2$random)) {
        sig2_re_idx <- sig2_re_idx + 1
        re <- params$sigma2$random[[group]]
        tag <- sig2_re_tag(sig2_re_idx)
        if (re$dim == 1) {
          L$add(sprintf("  Eigen::VectorXd grad_%s = Eigen::VectorXd::Zero(%s_d.size());", tag, tag))
        } else {
          L$add(sprintf("  Eigen::MatrixXd grad_%s = Eigen::MatrixXd::Zero(%s_d.rows(), %s_d.cols());", tag, tag, tag))
        }
      }
    }
  }

  # dprime_B (source_mixture)
  if (isTRUE(params$.has_dprime_B)) {
    L$add("  Eigen::VectorXd grad_beta_dpB = Eigen::VectorXd::Zero(P_dpB);")
    dpB_re_idx <- 0
    if (!is.null(params$dprime_B$random)) {
      for (group in names(params$dprime_B$random)) {
        dpB_re_idx <- dpB_re_idx + 1
        re <- params$dprime_B$random[[group]]
        tag <- dpB_re_tag(dpB_re_idx)
        if (re$dim == 1) {
          L$add(sprintf("  Eigen::VectorXd grad_%s = Eigen::VectorXd::Zero(%s_d.size());", tag, tag))
        } else {
          L$add(sprintf("  Eigen::MatrixXd grad_%s = Eigen::MatrixXd::Zero(%s_d.rows(), %s_d.cols());", tag, tag, tag))
        }
      }
    }
  }

  # lambda_B (source_mixture)
  if (isTRUE(params$.has_lambda_B)) {
    L$add("  Eigen::VectorXd grad_beta_lamB = Eigen::VectorXd::Zero(P_lamB);")
    lamB_re_idx <- 0
    if (!is.null(params$lambda_B$random)) {
      for (group in names(params$lambda_B$random)) {
        lamB_re_idx <- lamB_re_idx + 1
        re <- params$lambda_B$random[[group]]
        tag <- lamB_re_tag(lamB_re_idx)
        if (re$dim == 1) {
          L$add(sprintf("  Eigen::VectorXd grad_%s = Eigen::VectorXd::Zero(%s_d.size());", tag, tag))
        } else {
          L$add(sprintf("  Eigen::MatrixXd grad_%s = Eigen::MatrixXd::Zero(%s_d.rows(), %s_d.cols());", tag, tag, tag))
        }
      }
    }
  }

  # dprime_L (lure mixture)
  if (isTRUE(params$.has_dprime_L)) {
    L$add("  Eigen::VectorXd grad_beta_dpL = Eigen::VectorXd::Zero(P_dpL);")
    dpL_re_idx <- 0
    if (!is.null(params$dprime_L$random)) {
      for (group in names(params$dprime_L$random)) {
        dpL_re_idx <- dpL_re_idx + 1
        re <- params$dprime_L$random[[group]]
        tag <- dpL_re_tag(dpL_re_idx)
        if (re$dim == 1) {
          L$add(sprintf("  Eigen::VectorXd grad_%s = Eigen::VectorXd::Zero(%s_d.size());", tag, tag))
        } else {
          L$add(sprintf("  Eigen::MatrixXd grad_%s = Eigen::MatrixXd::Zero(%s_d.rows(), %s_d.cols());", tag, tag, tag))
        }
      }
    }
  }

  # sigma_L (lure mixture)
  if (isTRUE(params$.has_sigma_L)) {
    L$add("  Eigen::VectorXd grad_beta_sigL = Eigen::VectorXd::Zero(P_sigL);")
    sigL_re_idx <- 0
    if (!is.null(params$sigma_L$random)) {
      for (group in names(params$sigma_L$random)) {
        sigL_re_idx <- sigL_re_idx + 1
        re <- params$sigma_L$random[[group]]
        tag <- sigL_re_tag(sigL_re_idx)
        if (re$dim == 1) {
          L$add(sprintf("  Eigen::VectorXd grad_%s = Eigen::VectorXd::Zero(%s_d.size());", tag, tag))
        } else {
          L$add(sprintf("  Eigen::MatrixXd grad_%s = Eigen::MatrixXd::Zero(%s_d.rows(), %s_d.cols());", tag, tag, tag))
        }
      }
    }
  }

  # lambda_L (lure mixture)
  if (isTRUE(params$.has_lambda_L)) {
    L$add("  Eigen::VectorXd grad_beta_lamL = Eigen::VectorXd::Zero(P_lamL);")
    lamL_re_idx <- 0
    if (!is.null(params$lambda_L$random)) {
      for (group in names(params$lambda_L$random)) {
        lamL_re_idx <- lamL_re_idx + 1
        re <- params$lambda_L$random[[group]]
        tag <- lamL_re_tag(lamL_re_idx)
        if (re$dim == 1) {
          L$add(sprintf("  Eigen::VectorXd grad_%s = Eigen::VectorXd::Zero(%s_d.size());", tag, tag))
        } else {
          L$add(sprintf("  Eigen::MatrixXd grad_%s = Eigen::MatrixXd::Zero(%s_d.rows(), %s_d.cols());", tag, tag, tag))
        }
      }
    }
  }

  # discrim (vrdp2d / bivariate)
  if (isTRUE(params$.has_discrim)) {
    if (has_disc_fixed_cpp) {
      L$add("  Eigen::VectorXd grad_beta_disc = Eigen::VectorXd::Zero(P_disc);")
    }
    disc_re_idx <- 0
    if (!is.null(params$discrim$random)) {
      for (group in names(params$discrim$random)) {
        disc_re_idx <- disc_re_idx + 1
        re <- params$discrim$random[[group]]
        tag <- disc_re_tag(disc_re_idx)
        if (re$dim == 1) {
          L$add(sprintf("  Eigen::VectorXd grad_%s = Eigen::VectorXd::Zero(%s_d.size());", tag, tag))
        } else {
          L$add(sprintf("  Eigen::MatrixXd grad_%s = Eigen::MatrixXd::Zero(%s_d.rows(), %s_d.cols());", tag, tag, tag))
        }
      }
    }
  }

  # discrim_B (vrdp2d)
  if (isTRUE(params$.has_discrim_B)) {
    L$add("  Eigen::VectorXd grad_beta_discB = Eigen::VectorXd::Zero(P_discB);")
    discB_re_idx <- 0
    if (!is.null(params$discrim_B$random)) {
      for (group in names(params$discrim_B$random)) {
        discB_re_idx <- discB_re_idx + 1
        re <- params$discrim_B$random[[group]]
        tag <- discB_re_tag(discB_re_idx)
        if (re$dim == 1) {
          L$add(sprintf("  Eigen::VectorXd grad_%s = Eigen::VectorXd::Zero(%s_d.size());", tag, tag))
        } else {
          L$add(sprintf("  Eigen::MatrixXd grad_%s = Eigen::MatrixXd::Zero(%s_d.rows(), %s_d.cols());", tag, tag, tag))
        }
      }
    }
  }

  # rec_crit (cdp)
  if (isTRUE(params$.has_rec_crit)) {
    L$add("  Eigen::VectorXd grad_beta_rc = Eigen::VectorXd::Zero(P_rc);")
    rc_re_idx <- 0
    if (!is.null(params$rec_crit$random)) {
      for (group in names(params$rec_crit$random)) {
        rc_re_idx <- rc_re_idx + 1
        re <- params$rec_crit$random[[group]]
        tag <- rc_re_tag(rc_re_idx)
        if (re$dim == 1) {
          L$add(sprintf("  Eigen::VectorXd grad_%s = Eigen::VectorXd::Zero(%s_d.size());", tag, tag))
        } else {
          L$add(sprintf("  Eigen::MatrixXd grad_%s = Eigen::MatrixXd::Zero(%s_d.rows(), %s_d.cols());", tag, tag, tag))
        }
      }
    }
  }

  # know_crit (cdp)
  if (isTRUE(params$.has_know_crit)) {
    L$add("  Eigen::VectorXd grad_beta_kc = Eigen::VectorXd::Zero(P_kc);")
    kc_re_idx <- 0
    if (!is.null(params$know_crit$random)) {
      for (group in names(params$know_crit$random)) {
        kc_re_idx <- kc_re_idx + 1
        re <- params$know_crit$random[[group]]
        tag <- kc_re_tag(kc_re_idx)
        if (re$dim == 1) {
          L$add(sprintf("  Eigen::VectorXd grad_%s = Eigen::VectorXd::Zero(%s_d.size());", tag, tag))
        } else {
          L$add(sprintf("  Eigen::MatrixXd grad_%s = Eigen::MatrixXd::Zero(%s_d.rows(), %s_d.cols());", tag, tag, tag))
        }
      }
    }
  }

  # Criterion population
  crit <- model_data$criterion
  is_crit_intercept_only <- isTRUE(crit$is_intercept_only) || crit$n_coef == 1
  if (is_crit_intercept_only) {
    L$add("  double grad_thresh_mid_pop = 0.0;")
    L$add("  Eigen::VectorXd grad_log_gaps_pop = Eigen::VectorXd::Zero(n_gaps);")
  } else {
    # Multi-predictor: accumulate per-coefficient gradients
    L$add("  Eigen::VectorXd grad_thresh_mid_pop = Eigen::VectorXd::Zero(P_crit);")
    L$add("  Eigen::MatrixXd grad_log_gaps_pop = Eigen::MatrixXd::Zero(n_gaps, P_crit);")
  }

  # Criterion REs
  crit_re_idx <- 0
  if (!is.null(params$criterion$random)) {
    for (group in names(params$criterion$random)) {
      crit_re_idx <- crit_re_idx + 1
      L$add(sprintf("  Eigen::MatrixXd grad_u_crit%d = Eigen::MatrixXd::Zero(u_crit%d_d.rows(), u_crit%d_d.cols());",
                     crit_re_idx, crit_re_idx, crit_re_idx))
    }
  }

  # Bivariate-specific gradient accumulators
  if (isTRUE(params$.has_sigma_B)) emit_param_grad_accum(L, "sigma_B", "sigB", params$sigma_B$random, "sigma_B")
  if (isTRUE(params$.has_sigma2_B)) emit_param_grad_accum(L, "sigma2_B", "sig2B", params$sigma2_B$random, "sigma2_B")
  if (isTRUE(params$.has_rho)) emit_param_grad_accum(L, "rho", "rho", params$rho$random, "rho")
  if (isTRUE(params$.has_rho_B)) emit_param_grad_accum(L, "rho_B", "rhoB", params$rho_B$random, "rho_B")
  if (isTRUE(params$.has_rho_N)) emit_param_grad_accum(L, "rho_N", "rhoN", params$rho_N$random, "rho_N")
  if (isTRUE(params$.has_lambda2)) emit_param_grad_accum(L, "lambda2", "lam2", params$lambda2$random, "lambda2")
  if (isTRUE(params$.has_lambda2_B)) emit_param_grad_accum(L, "lambda2_B", "lam2B", params$lambda2_B$random, "lambda2_B")

  # Criterion2 population (vrdp2d/bivariate -- source dimension)
  if (isTRUE(params$.is_vrdp2d) || isTRUE(params$.is_bivariate)) {
    varying_sc <- isTRUE(params$.varying_source_criteria)
    if (varying_sc) {
      # Varying source criteria: gradient accumulators for per-bin thresholds
      L$add("  Eigen::VectorXd grad_vary_mid2 = Eigen::VectorXd::Zero(K1);")
      L$add("  Eigen::MatrixXd grad_vary_gaps2 = Eigen::MatrixXd::Zero(K1, n_gaps2);")
      if (identical(params$.new_source_criteria, "shared")) {
        L$add("  double grad_new_mid2 = 0.0;")
        L$add("  Eigen::VectorXd grad_new_gaps2 = Eigen::VectorXd::Zero(n_gaps2);")
      }
      # Criterion2 REs
      crit2_re_idx <- 0
      if (!is.null(params$criterion2$random)) {
        for (group in names(params$criterion2$random)) {
          crit2_re_idx <- crit2_re_idx + 1
          L$add(sprintf("  Eigen::MatrixXd grad_u_crit2_%d = Eigen::MatrixXd::Zero(u_crit2_%d_d.rows(), u_crit2_%d_d.cols());",
                         crit2_re_idx, crit2_re_idx, crit2_re_idx))
        }
      }
    } else {
      crit2 <- model_data$criterion2
      is_crit2_intercept_only <- is.null(crit2) || isTRUE(crit2$is_intercept_only) || crit2$n_coef == 1
      if (is_crit2_intercept_only) {
        L$add("  double grad_thresh_mid2_pop = 0.0;")
        L$add("  Eigen::VectorXd grad_log_gaps2_pop = Eigen::VectorXd::Zero(n_gaps2);")
      } else {
        L$add("  Eigen::VectorXd grad_thresh_mid2_pop = Eigen::VectorXd::Zero(P_crit2);")
        L$add("  Eigen::MatrixXd grad_log_gaps2_pop = Eigen::MatrixXd::Zero(n_gaps2, P_crit2);")
      }

      # Criterion2 REs
      crit2_re_idx <- 0
      if (!is.null(params$criterion2$random)) {
        for (group in names(params$criterion2$random)) {
          crit2_re_idx <- crit2_re_idx + 1
          L$add(sprintf("  Eigen::MatrixXd grad_u_crit2_%d = Eigen::MatrixXd::Zero(u_crit2_%d_d.rows(), u_crit2_%d_d.cols());",
                         crit2_re_idx, crit2_re_idx, crit2_re_idx))
        }
      }
    }
  }

  # Smooth gradient accumulators
  if (!is.null(params$.smooth_data)) {
    for (pname in names(params$.smooth_data)) {
      for (sm in params$.smooth_data[[pname]]) {
        n_thresh_sm <- if (!is.null(sm$n_thresh)) sm$n_thresh else 0L
        for (comp in sm$components) {
          for (k in seq_along(comp$Zs_list)) {
            s_name <- paste0("s_", pname, "_", comp$san_label, "_", k)
            if (n_thresh_sm > 0) {
              L$add(sprintf("  Eigen::MatrixXd grad_%s = Eigen::MatrixXd::Zero(%s_d.rows(), %s_d.cols());", s_name, s_name, s_name))
            } else {
              L$add(sprintf("  Eigen::VectorXd grad_%s = Eigen::VectorXd::Zero(%s_d.size());", s_name, s_name))
            }
          }
        }
      }
    }
  }
}


#' Emit smooth forward-pass contribution for a parameter
#' @noRd
emit_smooth_contribution <- function(L, smooth_data, pname, target_var) {
  if (is.null(smooth_data) || is.null(smooth_data[[pname]])) return()
  for (sm in smooth_data[[pname]]) {
    for (comp in sm$components) {
      for (k in seq_along(comp$Zs_list)) {
        zs_name <- paste0("Zs_", pname, "_", comp$san_label, "_", k)
        s_name <- paste0("s_", pname, "_", comp$san_label, "_", k)
        L$add(sprintf("    for (int j = 0; j < %s_d.size(); ++j) %s += %s_d(n, j) * %s_d(j);",
                       s_name, target_var, zs_name, s_name))
      }
    }
  }
}


#' Emit smooth backward-pass gradient for a parameter
#' @noRd
emit_smooth_gradient <- function(L, smooth_data, pname, d_var) {
  if (is.null(smooth_data) || is.null(smooth_data[[pname]])) return()
  for (sm in smooth_data[[pname]]) {
    for (comp in sm$components) {
      for (k in seq_along(comp$Zs_list)) {
        zs_name <- paste0("Zs_", pname, "_", comp$san_label, "_", k)
        s_name <- paste0("s_", pname, "_", comp$san_label, "_", k)
        L$add(sprintf("    for (int j = 0; j < %s_d.size(); ++j) grad_%s(j) += %s * %s_d(n, j);",
                       s_name, s_name, d_var, zs_name))
      }
    }
  }
}


#' Emit per-threshold smooth contribution (row-indexed) for criterion
#' @param row_expr C++ expression for the row index (0-based)
#' @noRd
emit_criterion_smooth_contribution <- function(L, smooth_data, target_var, row_expr) {
  if (is.null(smooth_data) || is.null(smooth_data[["criterion"]])) return()
  for (sm in smooth_data[["criterion"]]) {
    for (comp in sm$components) {
      for (k in seq_along(comp$Zs_list)) {
        zs_name <- paste0("Zs_criterion_", comp$san_label, "_", k)
        s_name <- paste0("s_criterion_", comp$san_label, "_", k)
        L$add(sprintf("    for (int j = 0; j < %s_d.cols(); ++j) %s += %s_d(n, j) * %s_d(%s, j);",
                       s_name, target_var, zs_name, s_name, row_expr))
      }
    }
  }
}


#' Emit per-threshold smooth gradient (row-indexed) for criterion
#' @noRd
emit_criterion_smooth_gradient <- function(L, smooth_data, d_var, row_expr) {
  if (is.null(smooth_data) || is.null(smooth_data[["criterion"]])) return()
  for (sm in smooth_data[["criterion"]]) {
    for (comp in sm$components) {
      for (k in seq_along(comp$Zs_list)) {
        zs_name <- paste0("Zs_criterion_", comp$san_label, "_", k)
        s_name <- paste0("s_criterion_", comp$san_label, "_", k)
        L$add(sprintf("    for (int j = 0; j < %s_d.cols(); ++j) grad_%s(%s, j) += %s * %s_d(n, j);",
                       s_name, s_name, row_expr, d_var, zs_name))
      }
    }
  }
}


# ============================================================================
# Group index extraction
# ============================================================================

#' Emit per-observation group index extraction
#' @noRd
emit_group_indices <- function(L, model_data, family_name, params) {
  # Track emitted groups to avoid duplicates (same group used by multiple params)
  emitted <- character(0)

  dp_re_idx <- 0
  if (!is.null(params$dprime$random)) {
    for (group in names(params$dprime$random)) {
      dp_re_idx <- dp_re_idx + 1
      if (!(group %in% emitted)) {
        L$add(sprintf("    int g_%s = %s[n] - 1;", group, group))
        emitted <- c(emitted, group)
      }
    }
  }

  if (isTRUE(params$.has_sigma) && !is.null(params$sigma$random)) {
    for (group in names(params$sigma$random)) {
      if (!(group %in% emitted)) {
        L$add(sprintf("    int g_%s = %s[n] - 1;", group, group))
        emitted <- c(emitted, group)
      }
    }
  }

  if (isTRUE(params$.has_lambda) && !is.null(params$lambda$random)) {
    for (group in names(params$lambda$random)) {
      if (!(group %in% emitted)) {
        L$add(sprintf("    int g_%s = %s[n] - 1;", group, group))
        emitted <- c(emitted, group)
      }
    }
  }

  if (isTRUE(params$.has_dprime2) && !is.null(params$dprime2$random)) {
    for (group in names(params$dprime2$random)) {
      if (!(group %in% emitted)) {
        L$add(sprintf("    int g_%s = %s[n] - 1;", group, group))
        emitted <- c(emitted, group)
      }
    }
  }

  if (isTRUE(params$.has_sigma2) && !is.null(params$sigma2$random)) {
    for (group in names(params$sigma2$random)) {
      if (!(group %in% emitted)) {
        L$add(sprintf("    int g_%s = %s[n] - 1;", group, group))
        emitted <- c(emitted, group)
      }
    }
  }

  if (isTRUE(params$.has_dprime_B) && !is.null(params$dprime_B$random)) {
    for (group in names(params$dprime_B$random)) {
      if (!(group %in% emitted)) {
        L$add(sprintf("    int g_%s = %s[n] - 1;", group, group))
        emitted <- c(emitted, group)
      }
    }
  }

  if (isTRUE(params$.has_lambda_B) && !is.null(params$lambda_B$random)) {
    for (group in names(params$lambda_B$random)) {
      if (!(group %in% emitted)) {
        L$add(sprintf("    int g_%s = %s[n] - 1;", group, group))
        emitted <- c(emitted, group)
      }
    }
  }

  if (isTRUE(params$.has_dprime_L) && !is.null(params$dprime_L$random)) {
    for (group in names(params$dprime_L$random)) {
      if (!(group %in% emitted)) {
        L$add(sprintf("    int g_%s = %s[n] - 1;", group, group))
        emitted <- c(emitted, group)
      }
    }
  }

  if (isTRUE(params$.has_sigma_L) && !is.null(params$sigma_L$random)) {
    for (group in names(params$sigma_L$random)) {
      if (!(group %in% emitted)) {
        L$add(sprintf("    int g_%s = %s[n] - 1;", group, group))
        emitted <- c(emitted, group)
      }
    }
  }

  if (isTRUE(params$.has_lambda_L) && !is.null(params$lambda_L$random)) {
    for (group in names(params$lambda_L$random)) {
      if (!(group %in% emitted)) {
        L$add(sprintf("    int g_%s = %s[n] - 1;", group, group))
        emitted <- c(emitted, group)
      }
    }
  }

  if (isTRUE(params$.has_discrim) && !is.null(params$discrim$random)) {
    for (group in names(params$discrim$random)) {
      if (!(group %in% emitted)) {
        L$add(sprintf("    int g_%s = %s[n] - 1;", group, group))
        emitted <- c(emitted, group)
      }
    }
  }

  if (isTRUE(params$.has_discrim_B) && !is.null(params$discrim_B$random)) {
    for (group in names(params$discrim_B$random)) {
      if (!(group %in% emitted)) {
        L$add(sprintf("    int g_%s = %s[n] - 1;", group, group))
        emitted <- c(emitted, group)
      }
    }
  }

  if (!is.null(params$criterion$random)) {
    for (group in names(params$criterion$random)) {
      if (!(group %in% emitted)) {
        L$add(sprintf("    int g_%s = %s[n] - 1;", group, group))
        emitted <- c(emitted, group)
      }
    }
  }

  if ((isTRUE(params$.is_vrdp2d) || isTRUE(params$.is_bivariate)) && !is.null(params$criterion2$random)) {
    for (group in names(params$criterion2$random)) {
      if (!(group %in% emitted)) {
        L$add(sprintf("    int g_%s = %s[n] - 1;", group, group))
        emitted <- c(emitted, group)
      }
    }
  }

  # CDP-specific group indices (rec_crit, know_crit)
  for (pname in c("rec_crit", "know_crit")) {
    flag <- paste0(".has_", pname)
    if (isTRUE(params[[flag]]) && !is.null(params[[pname]]$random)) {
      for (group in names(params[[pname]]$random)) {
        if (!(group %in% emitted)) {
          L$add(sprintf("    int g_%s = %s[n] - 1;", group, group))
          emitted <- c(emitted, group)
        }
      }
    }
  }

  # Bivariate-specific group indices
  for (pname in c("sigma_B", "sigma2_B", "rho", "rho_B", "rho_N", "lambda2", "lambda2_B")) {
    flag <- paste0(".has_", pname)
    if (isTRUE(params[[flag]]) && !is.null(params[[pname]]$random)) {
      for (group in names(params[[pname]]$random)) {
        if (!(group %in% emitted)) {
          L$add(sprintf("    int g_%s = %s[n] - 1;", group, group))
          emitted <- c(emitted, group)
        }
      }
    }
  }
}


# ============================================================================
# Parameter computation
# ============================================================================

#' Emit per-observation parameter computation (dprime, sigma)
#' @noRd
emit_param_computation <- function(L, model_data, family_name, params) {
  # --- dprime ---
  has_dp_fixed <- isTRUE(params$.has_dp_fixed)
  has_disc_fixed_cpp <- isTRUE(params$.has_disc_fixed)
  has_encoding_dp <- isTRUE(params$dprime$fixed$has_encoding)
  L$add("    double dprime_n = 0.0;")
  if (has_dp_fixed) {
    if (has_encoding_dp) {
      L$add("    if (is_old_n > 0) {")
      L$add("      for (int p = 0; p < P_dp; ++p) dprime_n += X_dp_d(n, p) * beta_dp_d(p);")
      L$add("    }")
    } else {
      L$add("    for (int p = 0; p < P_dp; ++p) dprime_n += X_dp_d(n, p) * beta_dp_d(p);")
    }
  }

  # dprime REs
  dp_re_idx <- 0
  if (!is.null(params$dprime$random)) {
    for (group in names(params$dprime$random)) {
      dp_re_idx <- dp_re_idx + 1
      re <- params$dprime$random[[group]]
      tag <- dp_re_tag(dp_re_idx)
      emit_re_contribution(L, tag, re, group, "dprime_n", "dprime", dp_re_idx, "dp")
    }
  }

  # dprime smooth contributions (must come before mu computation)
  emit_smooth_contribution(L, params$.smooth_data, "dprime", "dprime_n")

  # mu computation: cumulative/source_mixture/vrdp2d/bivariate use dprime_n directly,
  # CDP families use dprime_n as mu_R (not multiplied by is_old),
  # standard SDT uses is_old_n * dprime_n
  if (isTRUE(params$.is_cdp_family)) {
    # CDP: dprime_n is the recollection mean mu_R (used directly by cell function)
  } else if (isTRUE(params$.is_cumulative) || isTRUE(params$.is_source_mixture) || isTRUE(params$.is_vrdp2d) || isTRUE(params$.is_bivariate)) {
    L$add("    double mu = dprime_n;")
  } else {
    L$add("    double mu = is_old_n * dprime_n;")
  }

  # --- sigma (UVSDT, dpsdt, mixture) ---
  if (isTRUE(params$.has_sigma)) {
    has_encoding_sig <- isTRUE(params$sigma$fixed$has_encoding)
    if (has_encoding_sig) {
      L$add("    double sigma_n = 0.0;")
      L$add("    if (is_old_n > 0) {")
      L$add("      for (int p = 0; p < P_sig; ++p) sigma_n += X_sig_d(n, p) * beta_sig_d(p);")
      L$add("    }")
    } else {
      L$add("    double sigma_n = 0.0;")
      L$add("    for (int p = 0; p < P_sig; ++p) sigma_n += X_sig_d(n, p) * beta_sig_d(p);")
    }

    sig_re_idx <- 0
    if (!is.null(params$sigma$random)) {
      for (group in names(params$sigma$random)) {
        sig_re_idx <- sig_re_idx + 1
        re <- params$sigma$random[[group]]
        tag <- sig_re_tag(sig_re_idx)
        emit_re_contribution(L, tag, re, group, "sigma_n", "sigma", sig_re_idx, "sig")
      }
    }

    emit_smooth_contribution(L, params$.smooth_data, "sigma", "sigma_n")
    L$add("    double sigma_val = std::exp(sigma_n);")
    if (isTRUE(params$.is_vrdp2d) || isTRUE(params$.is_bivariate) || isTRUE(params$.is_cdp_family)) {
      # VRDP2D/bivariate/CDP: sigma used directly by cell function (no s alias)
    } else {
      L$add("    double s = is_old_n > 0 ? sigma_val : 1.0;")
    }
  }

  # --- lambda (dpsdt, mixture) ---
  if (isTRUE(params$.has_lambda)) {
    has_encoding_lam <- isTRUE(params$lambda$fixed$has_encoding)
    if (has_encoding_lam) {
      L$add("    double lambda_n = 0.0;")
      L$add("    if (is_old_n > 0) {")
      L$add("      for (int p = 0; p < P_lam; ++p) lambda_n += X_lam_d(n, p) * beta_lam_d(p);")
      L$add("    }")
    } else {
      L$add("    double lambda_n = 0.0;")
      L$add("    for (int p = 0; p < P_lam; ++p) lambda_n += X_lam_d(n, p) * beta_lam_d(p);")
    }

    lam_re_idx <- 0
    if (!is.null(params$lambda$random)) {
      for (group in names(params$lambda$random)) {
        lam_re_idx <- lam_re_idx + 1
        re <- params$lambda$random[[group]]
        tag <- lam_re_tag(lam_re_idx)
        emit_re_contribution(L, tag, re, group, "lambda_n", "lambda", lam_re_idx, "lam")
      }
    }

    # inv_logit link
    emit_smooth_contribution(L, params$.smooth_data, "lambda", "lambda_n")
    L$add("    double lambda_val = 1.0 / (1.0 + std::exp(-lambda_n));")
  }

  # --- dprime2 (mixture) ---
  if (isTRUE(params$.has_dprime2)) {
    has_encoding_dp2 <- isTRUE(params$dprime2$fixed$has_encoding)
    if (has_encoding_dp2) {
      L$add("    double dprime2_n = 0.0;")
      L$add("    if (is_old_n > 0) {")
      L$add("      for (int p = 0; p < P_dp2; ++p) dprime2_n += X_dp2_d(n, p) * beta_dp2_d(p);")
      L$add("    }")
    } else {
      L$add("    double dprime2_n = 0.0;")
      L$add("    for (int p = 0; p < P_dp2; ++p) dprime2_n += X_dp2_d(n, p) * beta_dp2_d(p);")
    }

    dp2_re_idx <- 0
    if (!is.null(params$dprime2$random)) {
      for (group in names(params$dprime2$random)) {
        dp2_re_idx <- dp2_re_idx + 1
        re <- params$dprime2$random[[group]]
        tag <- dp2_re_tag(dp2_re_idx)
        emit_re_contribution(L, tag, re, group, "dprime2_n", "dprime2", dp2_re_idx, "dp2")
      }
    }
    emit_smooth_contribution(L, params$.smooth_data, "dprime2", "dprime2_n")
  }

  # --- sigma2 (mixture) ---
  if (isTRUE(params$.has_sigma2)) {
    has_encoding_sig2 <- isTRUE(params$sigma2$fixed$has_encoding)
    if (has_encoding_sig2) {
      L$add("    double sigma2_n = 0.0;")
      L$add("    if (is_old_n > 0) {")
      L$add("      for (int p = 0; p < P_sig2; ++p) sigma2_n += X_sig2_d(n, p) * beta_sig2_d(p);")
      L$add("    }")
    } else {
      L$add("    double sigma2_n = 0.0;")
      L$add("    for (int p = 0; p < P_sig2; ++p) sigma2_n += X_sig2_d(n, p) * beta_sig2_d(p);")
    }

    sig2_re_idx <- 0
    if (!is.null(params$sigma2$random)) {
      for (group in names(params$sigma2$random)) {
        sig2_re_idx <- sig2_re_idx + 1
        re <- params$sigma2$random[[group]]
        tag <- sig2_re_tag(sig2_re_idx)
        emit_re_contribution(L, tag, re, group, "sigma2_n", "sigma2", sig2_re_idx, "sig2")
      }
    }

    emit_smooth_contribution(L, params$.smooth_data, "sigma2", "sigma2_n")
    L$add("    double sigma2_val = std::exp(sigma2_n);")
    if (isTRUE(params$.is_vrdp2d) || isTRUE(params$.is_bivariate) || isTRUE(params$.is_cdp_family)) {
      # VRDP2D/bivariate/CDP: sigma2 used by cell function directly (no s2 alias)
    } else {
      L$add("    double s2 = is_old_n > 0 ? sigma2_val : 1.0;")
    }
  }

  # --- dprime_B (source_mixture) ---
  if (isTRUE(params$.has_dprime_B)) {
    L$add("    double dprime_B_n = 0.0;")
    L$add("    for (int p = 0; p < P_dpB; ++p) dprime_B_n += X_dpB_d(n, p) * beta_dpB_d(p);")

    dpB_re_idx <- 0
    if (!is.null(params$dprime_B$random)) {
      for (group in names(params$dprime_B$random)) {
        dpB_re_idx <- dpB_re_idx + 1
        re <- params$dprime_B$random[[group]]
        tag <- dpB_re_tag(dpB_re_idx)
        emit_re_contribution(L, tag, re, group, "dprime_B_n", "dprime_B", dpB_re_idx, "dpB")
      }
    }
    emit_smooth_contribution(L, params$.smooth_data, "dprime_B", "dprime_B_n")
  }

  # --- lambda_B (source_mixture) ---
  if (isTRUE(params$.has_lambda_B)) {
    L$add("    double lambda_B_n = 0.0;")
    L$add("    for (int p = 0; p < P_lamB; ++p) lambda_B_n += X_lamB_d(n, p) * beta_lamB_d(p);")

    lamB_re_idx <- 0
    if (!is.null(params$lambda_B$random)) {
      for (group in names(params$lambda_B$random)) {
        lamB_re_idx <- lamB_re_idx + 1
        re <- params$lambda_B$random[[group]]
        tag <- lamB_re_tag(lamB_re_idx)
        emit_re_contribution(L, tag, re, group, "lambda_B_n", "lambda_B", lamB_re_idx, "lamB")
      }
    }

    emit_smooth_contribution(L, params$.smooth_data, "lambda_B", "lambda_B_n")
    L$add("    double lambda_B_val = 1.0 / (1.0 + std::exp(-lambda_B_n));")
  }

  # --- dprime_L (lure mixture) ---
  if (isTRUE(params$.has_dprime_L)) {
    L$add("    double dprime_L_n = 0.0;")
    L$add("    for (int p = 0; p < P_dpL; ++p) dprime_L_n += X_dpL_d(n, p) * beta_dpL_d(p);")

    dpL_re_idx <- 0
    if (!is.null(params$dprime_L$random)) {
      for (group in names(params$dprime_L$random)) {
        dpL_re_idx <- dpL_re_idx + 1
        re <- params$dprime_L$random[[group]]
        tag <- dpL_re_tag(dpL_re_idx)
        emit_re_contribution(L, tag, re, group, "dprime_L_n", "dprime_L", dpL_re_idx, "dpL")
      }
    }
    emit_smooth_contribution(L, params$.smooth_data, "dprime_L", "dprime_L_n")
  }

  # --- sigma_L (lure mixture) ---
  if (isTRUE(params$.has_sigma_L)) {
    L$add("    double sigma_L_n = 0.0;")
    L$add("    for (int p = 0; p < P_sigL; ++p) sigma_L_n += X_sigL_d(n, p) * beta_sigL_d(p);")

    sigL_re_idx <- 0
    if (!is.null(params$sigma_L$random)) {
      for (group in names(params$sigma_L$random)) {
        sigL_re_idx <- sigL_re_idx + 1
        re <- params$sigma_L$random[[group]]
        tag <- sigL_re_tag(sigL_re_idx)
        emit_re_contribution(L, tag, re, group, "sigma_L_n", "sigma_L", sigL_re_idx, "sigL")
      }
    }

    emit_smooth_contribution(L, params$.smooth_data, "sigma_L", "sigma_L_n")
    L$add("    double sigma_L_val = std::exp(sigma_L_n);")
  }

  # --- lambda_L (lure mixture) ---
  if (isTRUE(params$.has_lambda_L)) {
    L$add("    double lambda_L_n = 0.0;")
    L$add("    for (int p = 0; p < P_lamL; ++p) lambda_L_n += X_lamL_d(n, p) * beta_lamL_d(p);")

    lamL_re_idx <- 0
    if (!is.null(params$lambda_L$random)) {
      for (group in names(params$lambda_L$random)) {
        lamL_re_idx <- lamL_re_idx + 1
        re <- params$lambda_L$random[[group]]
        tag <- lamL_re_tag(lamL_re_idx)
        emit_re_contribution(L, tag, re, group, "lambda_L_n", "lambda_L", lamL_re_idx, "lamL")
      }
    }

    emit_smooth_contribution(L, params$.smooth_data, "lambda_L", "lambda_L_n")
    L$add("    double lambda_L_val = 1.0 / (1.0 + std::exp(-lambda_L_n));")
  }

  # --- discrim (vrdp2d / bivariate -- source A discriminability) ---
  if (isTRUE(params$.has_discrim)) {
    L$add("    double discrim_n = 0.0;")
    if (has_disc_fixed_cpp) {
      L$add("    for (int p = 0; p < P_disc; ++p) discrim_n += X_disc_d(n, p) * beta_disc_d(p);")
    }

    disc_re_idx <- 0
    if (!is.null(params$discrim$random)) {
      for (group in names(params$discrim$random)) {
        disc_re_idx <- disc_re_idx + 1
        re <- params$discrim$random[[group]]
        tag <- disc_re_tag(disc_re_idx)
        emit_re_contribution(L, tag, re, group, "discrim_n", "discrim", disc_re_idx, "disc")
      }
    }
    emit_smooth_contribution(L, params$.smooth_data, "discrim", "discrim_n")
  }

  # --- discrim_B (vrdp2d/bivariate -- source B discriminability, optional) ---
  if (isTRUE(params$.has_discrim_B)) {
    L$add("    double discrim_B_n = 0.0;")
    L$add("    for (int p = 0; p < P_discB; ++p) discrim_B_n += X_discB_d(n, p) * beta_discB_d(p);")

    discB_re_idx <- 0
    if (!is.null(params$discrim_B$random)) {
      for (group in names(params$discrim_B$random)) {
        discB_re_idx <- discB_re_idx + 1
        re <- params$discrim_B$random[[group]]
        tag <- discB_re_tag(discB_re_idx)
        emit_re_contribution(L, tag, re, group, "discrim_B_n", "discrim_B", discB_re_idx, "discB")
      }
    }
    emit_smooth_contribution(L, params$.smooth_data, "discrim_B", "discrim_B_n")
  }

  # --- Bivariate-specific parameters ---
  emit_generic_param_computation(L, params, "sigma_B", "sigB", "has_sigma_B", link = "exp", val_name = "sigma_B_val")
  emit_generic_param_computation(L, params, "sigma2_B", "sig2B", "has_sigma2_B", link = "exp", val_name = "sigma2_B_val")
  # Link for rho follows the family$bounded flag. bivariate_dp now supports
  # bounded = FALSE (the BDP model from Starns/Rotello/Hautus 2014 appendix),
  # in which case rho uses Fisher z (tanh) just like unbounded sdt.
  rho_link <- if (isTRUE(params$.bounded)) "inv_logit" else "tanh"
  emit_generic_param_computation(L, params, "rho", "rho", "has_rho", link = rho_link, val_name = "rho_val")
  emit_generic_param_computation(L, params, "rho_B", "rhoB", "has_rho_B", link = rho_link, val_name = "rho_B_val")
  emit_generic_param_computation(L, params, "rho_N", "rhoN", "has_rho_N", link = "tanh", val_name = "rho_N_val")
  emit_generic_param_computation(L, params, "lambda2", "lam2", "has_lambda2", link = "inv_logit", val_name = "R_S_val")
  # lambda2_B (Source-B source recollection); produces R_S_B_val if user
  # specified lambda2_B, otherwise R_S_B_val is aliased to R_S_val below.
  emit_generic_param_computation(L, params, "lambda2_B", "lam2B", "has_lambda2_B", link = "inv_logit", val_name = "R_S_B_val")

  # If bivariate_dp, also alias lambda_val as R_I_val (and lambda_B_val as
  # R_I_B_val when present). When the user did not supply lambda_B /
  # lambda2_B, fall back to the matching A-side value so the cell function
  # sees R_I_B_eff = R_I_val (constrained-equal default -- produces the
  # same likelihood and gradient routing as the pre-asymmetric path).
  if (isTRUE(params$.is_bivariate_dp) && isTRUE(params$.has_lambda)) {
    L$add("    double R_I_val = lambda_val;")
    if (isTRUE(params$.has_lambda_B_dp)) {
      L$add("    double R_I_B_val = lambda_B_val;")
    } else {
      L$add("    double R_I_B_val = lambda_val;  // constrained equal to lambda")
    }
    if (!isTRUE(params$.has_lambda2_B)) {
      L$add("    double R_S_B_val = R_S_val;  // constrained equal to lambda2")
    }
  }

  # --- CDP-specific: rec_crit and know_crit ---
  if (isTRUE(params$.has_rec_crit)) {
    emit_generic_param_computation(L, params, "rec_crit", "rc", "has_rec_crit")
    L$add("    double rec_crit_val = rec_crit_n;")
  }

  if (isTRUE(params$.has_know_crit)) {
    emit_generic_param_computation(L, params, "know_crit", "kc", "has_know_crit")
    L$add("    double know_crit_val = know_crit_n;")
  }
}


#' Emit generic parameter computation (fixed + REs + link)
#' @noRd
emit_generic_param_computation <- function(L, params, full_name, short, flag_name, link = "identity", val_name = NULL) {
  if (!isTRUE(params[[paste0(".", flag_name)]])) return()
  tag_fn <- get(paste0(short, "_re_tag"))

  L$add(sprintf("    double %s_n = 0.0;", full_name))
  L$add(sprintf("    for (int p = 0; p < P_%s; ++p) %s_n += X_%s_d(n, p) * beta_%s_d(p);",
                 short, full_name, short, short))

  re_idx <- 0
  param_data <- params[[full_name]]
  if (!is.null(param_data$random)) {
    for (group in names(param_data$random)) {
      re_idx <- re_idx + 1
      re <- param_data$random[[group]]
      tag <- tag_fn(re_idx)
      emit_re_contribution(L, tag, re, group, sprintf("%s_n", full_name), full_name, re_idx, short)
    }
  }

  # Smooth contributions
  emit_smooth_contribution(L, params$.smooth_data, full_name, sprintf("%s_n", full_name))

  if (!is.null(val_name)) {
    if (link == "exp") {
      L$add(sprintf("    double %s = std::exp(%s_n);", val_name, full_name))
    } else if (link == "tanh") {
      L$add(sprintf("    double %s = std::tanh(%s_n);", val_name, full_name))
    } else if (link == "inv_logit") {
      L$add(sprintf("    double %s = 1.0 / (1.0 + std::exp(-%s_n));", val_name, full_name))
    }
  }
}


#' Emit a single RE contribution to the linear predictor
#' @noRd
emit_re_contribution <- function(L, tag, re, group, target_var, param_name, re_idx, prefix) {
  g_var <- sprintf("g_%s", group)

  if (isTRUE(re$use_z_matrix)) {
    if (re$dim == 1) {
      # Z-matrix scalar: Z[n,0] * u[group]
      L$add(sprintf("    %s += Z_%s_d(n, 0) * %s_d(%s);",
                     target_var, tag, tag, g_var))
    } else {
      # Z-matrix multi-dim: sum_j Z[n,j] * u[group, j]
      L$add(sprintf("    for (int j = 0; j < %s_d.cols(); ++j) %s += Z_%s_d(n, j) * %s_d(%s, j);",
                     tag, target_var, tag, tag, g_var))
    }
  } else if (re$dim == 1) {
    if (!is.null(re$term_idx)) {
      # Scalar with term_idx guard
      L$add(sprintf("    if (idx_%s_%s[n] > 0) %s += %s_d(%s);",
                     param_name, group, target_var, tag, g_var))
    } else {
      # Scalar intercept-only
      L$add(sprintf("    %s += %s_d(%s);", target_var, tag, g_var))
    }
  } else if (!is.null(re$term_idx)) {
    # Multi-dim index-based: u[group, idx-1]
    L$add(sprintf("    if (idx_%s_%s[n] > 0) %s += %s_d(%s, idx_%s_%s[n] - 1);",
                   param_name, group, target_var, tag, g_var, param_name, group))
  }
}


# ============================================================================
# Threshold computation
# ============================================================================

#' Emit threshold construction code
#' @noRd
emit_thresh_computation <- function(L, model_data, params, has_crit_re) {
  crit <- model_data$criterion
  is_crit_intercept_only <- isTRUE(crit$is_intercept_only) || crit$n_coef == 1
  gap_link <- if (!is.null(params$.gap_link)) params$.gap_link else "log"
  gap_fn <- if (identical(gap_link, "softplus")) "std::log1p(std::exp(%s))" else "std::exp(%s)"

  if (has_crit_re) {
    # --- Thresholds with criterion RE ---
    # Mid anchor
    if (is_crit_intercept_only) {
      L$add("    double thresh_mid_val = thresh_mid_d_vec(0);")
    } else {
      L$add("    double thresh_mid_val = 0.0;")
      L$add("    for (int p = 0; p < P_crit; ++p) thresh_mid_val += X_crit_d(n, p) * thresh_mid_d_vec(p);")
    }

    # Add criterion RE contributions to mid threshold
    crit_re_idx <- 0
    for (group in names(params$criterion$random)) {
      crit_re_idx <- crit_re_idx + 1
      re <- params$criterion$random[[group]]
      emit_crit_re_mid(L, crit_re_idx, re, group, "thresh_mid_val")
    }
    # Criterion smooth contribution to mid-anchor (row 0)
    emit_criterion_smooth_contribution(L, params$.smooth_data, "thresh_mid_val", "0")
    L$add("    thresh_n(mid) = thresh_mid_val;")

    # Upper gaps
    L$add("    for (int k = mid + 1; k < n_thresh; ++k) {")
    L$add("      int gi = k - mid - 1;")
    if (is_crit_intercept_only) {
      L$add("      double eff_gap = pop_gap_d(gi);")
    } else {
      L$add("      double eff_gap = 0.0;")
      L$add("      for (int p = 0; p < P_crit; ++p) eff_gap += X_crit_d(n, p) * gaps_mat_d(gi, p);")
    }
    # Add criterion RE to gap
    crit_re_idx <- 0
    for (group in names(params$criterion$random)) {
      crit_re_idx <- crit_re_idx + 1
      re <- params$criterion$random[[group]]
      emit_crit_re_gap(L, crit_re_idx, re, group, "eff_gap", "k + 1")  # k+1 for 1-based Stan-like indexing
    }
    # Criterion smooth per-gap (upper, row = 1 + gi)
    emit_criterion_smooth_contribution(L, params$.smooth_data, "eff_gap", "1 + gi")
    L$add(sprintf("      exp_eff_gaps(gi) = %s;", sprintf(gap_fn, "eff_gap")))
    L$add("      thresh_n(k) = thresh_n(k - 1) + exp_eff_gaps(gi);")
    L$add("    }")

    # Lower gaps
    L$add("    for (int kd = 1; kd <= mid; ++kd) {")
    L$add("      int k = mid - kd;")
    L$add("      int gi = n_upper + kd - 1;")
    if (is_crit_intercept_only) {
      L$add("      double eff_gap = pop_gap_d(gi);")
    } else {
      L$add("      double eff_gap = 0.0;")
      L$add("      for (int p = 0; p < P_crit; ++p) eff_gap += X_crit_d(n, p) * gaps_mat_d(gi, p);")
    }
    crit_re_idx <- 0
    for (group in names(params$criterion$random)) {
      crit_re_idx <- crit_re_idx + 1
      re <- params$criterion$random[[group]]
      emit_crit_re_gap(L, crit_re_idx, re, group, "eff_gap", "k + 1")
    }
    # Criterion smooth per-gap (lower, row = 1 + gi)
    emit_criterion_smooth_contribution(L, params$.smooth_data, "eff_gap", "1 + gi")
    L$add(sprintf("      exp_eff_gaps(gi) = %s;", sprintf(gap_fn, "eff_gap")))
    L$add("      thresh_n(k) = thresh_n(k + 1) - exp_eff_gaps(gi);")
    L$add("    }")

  } else {
    # --- Population-only thresholds (precomputed exp_gaps) ---
    has_crit_smooth <- !is.null(params$.smooth_data[["criterion"]])
    if (is_crit_intercept_only && !has_crit_smooth) {
      L$add("    thresh_n(mid) = thresh_mid_d_vec(0);")
    } else if (is_crit_intercept_only && has_crit_smooth) {
      L$add("    double thresh_mid_val = thresh_mid_d_vec(0);")
      emit_smooth_contribution(L, params$.smooth_data, "criterion", "thresh_mid_val")
      L$add("    thresh_n(mid) = thresh_mid_val;")
    } else {
      L$add("    double thresh_mid_val = 0.0;")
      L$add("    for (int p = 0; p < P_crit; ++p) thresh_mid_val += X_crit_d(n, p) * thresh_mid_d_vec(p);")
      emit_criterion_smooth_contribution(L, params$.smooth_data, "thresh_mid_val", "0")
      L$add("    thresh_n(mid) = thresh_mid_val;")
    }
    L$add("    for (int k = mid + 1; k < n_thresh; ++k) {")
    L$add("      int gi = k - mid - 1;")
    if (!is_crit_intercept_only || has_crit_smooth) {
      if (!is_crit_intercept_only) {
        L$add("      double gval = 0.0;")
        L$add("      for (int p = 0; p < P_crit; ++p) gval += X_crit_d(n, p) * gaps_mat_d(gi, p);")
      } else {
        L$add("      double gval = pop_gap_d(gi);")
      }
      emit_criterion_smooth_contribution(L, params$.smooth_data, "gval", "1 + gi")
      L$add(sprintf("      double egval = %s;", sprintf(gap_fn, "gval")))
      L$add("      thresh_n(k) = thresh_n(k - 1) + egval;")
      L$add("      obs_exp_gaps(gi) = egval;")
    } else {
      L$add("      thresh_n(k) = thresh_n(k - 1) + pop_exp_gaps(gi);")
    }
    L$add("    }")
    L$add("    for (int kd = 1; kd <= mid; ++kd) {")
    L$add("      int k = mid - kd;")
    L$add("      int gi = n_upper + kd - 1;")
    if (!is_crit_intercept_only || has_crit_smooth) {
      if (!is_crit_intercept_only) {
        L$add("      double gval = 0.0;")
        L$add("      for (int p = 0; p < P_crit; ++p) gval += X_crit_d(n, p) * gaps_mat_d(gi, p);")
      } else {
        L$add("      double gval = pop_gap_d(gi);")
      }
      emit_criterion_smooth_contribution(L, params$.smooth_data, "gval", "1 + gi")
      L$add(sprintf("      double egval = %s;", sprintf(gap_fn, "gval")))
      L$add("      thresh_n(k) = thresh_n(k + 1) - egval;")
      L$add("      obs_exp_gaps(gi) = egval;")
    } else {
      L$add("      thresh_n(k) = thresh_n(k + 1) - pop_exp_gaps(gi);")
    }
    L$add("    }")
  }
}


#' Emit criterion RE contribution to mid-threshold
#' @noRd
emit_crit_re_mid <- function(L, crit_re_idx, re, group, target) {
  g_var <- sprintf("g_%s", group)
  n_re_terms <- if (!is.null(re$n_re_terms)) re$n_re_terms else 1

  if (isTRUE(re$use_z_matrix)) {
    # Z_crit[n, ] * u_crit[group, ((mid+1-1)*nrt+1):((mid+1)*nrt)]
    # In 0-based: cols mid*nrt .. (mid+1)*nrt - 1
    L$add(sprintf("    for (int j = 0; j < %d; ++j) %s += Z_crit%d_d(n, j) * u_crit%d_d(%s, mid * %d + j);",
                   n_re_terms, target, crit_re_idx, crit_re_idx, g_var, n_re_terms))
  } else if (n_re_terms == 1) {
    # u_crit[group, mid]
    L$add(sprintf("    %s += u_crit%d_d(%s, mid);", target, crit_re_idx, g_var))
  } else {
    # Multi-term with idx: u_crit[group, (mid)*n_re_terms + idx - 1]
    idx_var <- sprintf("idx_criterion_%s", group)
    L$add(sprintf("    if (%s[n] > 0) %s += u_crit%d_d(%s, mid * %d + %s[n] - 1);",
                   idx_var, target, crit_re_idx, g_var, n_re_terms, idx_var))
  }
}


#' Emit criterion RE contribution to a gap
#' thresh_k_1based is a C++ expression for the 1-based threshold index (Stan convention)
#' @noRd
emit_crit_re_gap <- function(L, crit_re_idx, re, group, target, thresh_k_1based) {
  g_var <- sprintf("g_%s", group)
  n_re_terms <- if (!is.null(re$n_re_terms)) re$n_re_terms else 1

  # Convert 1-based thresh_k to 0-based column: col = (thresh_k - 1) for n_re_terms==1
  # For n_re_terms > 1: col_start = (thresh_k - 1) * n_re_terms

  if (isTRUE(re$use_z_matrix)) {
    # Z_crit[n, j] * u_crit[group, (k)*nrt + j]  where k is 0-based threshold = k (from loop var)
    L$add(sprintf("      for (int j = 0; j < %d; ++j) %s += Z_crit%d_d(n, j) * u_crit%d_d(%s, k * %d + j);",
                   n_re_terms, target, crit_re_idx, crit_re_idx, g_var, n_re_terms))
  } else if (n_re_terms == 1) {
    # u_crit[group, k]  (k is the 0-based threshold index from loop)
    L$add(sprintf("      %s += u_crit%d_d(%s, k);", target, crit_re_idx, g_var))
  } else {
    # Multi-term: u_crit[group, k * n_re_terms + idx - 1]
    idx_var <- sprintf("idx_criterion_%s", group)
    L$add(sprintf("      if (%s[n] > 0) %s += u_crit%d_d(%s, k * %d + %s[n] - 1);",
                   idx_var, target, crit_re_idx, g_var, n_re_terms, idx_var))
  }
}


# ============================================================================
# Likelihood computation
# ============================================================================

#' Emit likelihood (cell log-probability) computation
#'
#' For all families, this emits code that produces:
#' - total_lp accumulation
#' - k_lo, k_hi (threshold indices used for cell boundaries)
#' - d_thresh_lo, d_thresh_hi: d_lp/d_thresh[k_lo] and d_lp/d_thresh[k_hi]
#'   (direct derivatives w.r.t. the threshold values, no further scaling needed)
#'
#' For dpsdt/mixture, also emits d_mu_raw, d_lambda_raw, d_sigma_raw, etc.
#' @noRd
emit_likelihood <- function(L, family_name, params) {
  if (family_name %in% c("evsdt", "uvsdt", "cumulative")) {
    emit_likelihood_standard(L, family_name, params)
  } else if (family_name == "dpsdt") {
    emit_likelihood_dpsdt(L, params)
  } else if (family_name == "mixture") {
    emit_likelihood_mixture(L, params)
  } else if (family_name == "source_mixture") {
    emit_likelihood_source_mixture(L, params)
  }
}


#' Get logit suffix for C++ function calls based on link
#' @noRd
logit_suffix <- function(params) {
  if (identical(params$.link_name, "logit")) "_logit" else ""
}

#' Emit standard likelihood (EVSDT, UVSDT, cumulative)
#' @noRd
emit_likelihood_standard <- function(L, family_name, params) {
  has_sigma <- isTRUE(params$.has_sigma)
  lsuf <- logit_suffix(params)

  z_expr <- if (has_sigma) {
    function(thresh) sprintf("(%s - mu) / s", thresh)
  } else {
    function(thresh) sprintf("%s - mu", thresh)
  }

  L$add("    batch_sdt::CellResult res;")
  L$add("    int k_lo = -1, k_hi = -1;")
  L$add("    if (yn == 1) {")
  L$add(sprintf("      res = batch_sdt::edge_lo%s(%s);", lsuf, z_expr("thresh_n(0)")))
  L$add("      k_hi = 0;")
  L$add("    } else if (yn == K) {")
  L$add(sprintf("      res = batch_sdt::edge_hi%s(%s);", lsuf, z_expr("thresh_n(n_thresh - 1)")))
  L$add("      k_lo = n_thresh - 1;")
  L$add("    } else {")
  L$add(sprintf("      res = batch_sdt::cell_log_prob%s(%s, %s);",
                lsuf, z_expr("thresh_n(yn - 2)"), z_expr("thresh_n(yn - 1)")))
  L$add("      k_lo = yn - 2;")
  L$add("      k_hi = yn - 1;")
  L$add("    }")

  if (params$.has_counts) {
    L$add("    total_lp += res.lp * count_n;")
  } else {
    L$add("    total_lp += res.lp;")
  }
}


#' Emit DPSDT likelihood
#' @noRd
emit_likelihood_dpsdt <- function(L, params) {
  has_sigma <- isTRUE(params$.has_sigma)
  lsuf <- logit_suffix(params)

  L$add("    // DPSDT likelihood")
  L$add("    double obs_lp = 0.0;")
  L$add("    double d_mu_raw = 0.0;")
  L$add("    double d_lambda_raw = 0.0;")
  L$add("    double d_thresh_lo = 0.0, d_thresh_hi = 0.0;")
  if (has_sigma) {
    L$add("    double d_sigma_raw = 0.0;")
  }
  L$add("    int k_lo = -1, k_hi = -1;")
  L$add("")
  L$add("    if (is_old_n > 0) {")

  s_var <- if (has_sigma) "s" else "1.0"

  # --- Old item, y == K: recollection + familiarity ---
  L$add("      if (yn == K) {")
  L$add(sprintf("        double z_K = (thresh_n(n_thresh - 1) - mu) / %s;", s_var))
  L$add(sprintf("        auto dr = batch_sdt::dpsdt_top_cell%s(z_K, lambda_val);", lsuf))
  L$add("        obs_lp = dr.lp;")
  L$add(sprintf("        d_mu_raw = dr.d_z * (-1.0 / %s);", s_var))
  L$add("        d_lambda_raw = dr.d_lambda;")
  L$add("        d_thresh_lo = -d_mu_raw;")
  L$add("        k_lo = n_thresh - 1;")
  if (has_sigma) {
    L$add("        d_sigma_raw = dr.d_z * (-(thresh_n(n_thresh - 1) - mu) / (s * s)) * sigma_val;")
  }

  # --- Old item, y < K: familiarity only ---
  L$add("      } else {")
  L$add(sprintf("        double z_hi_val = (thresh_n(yn == 1 ? 0 : yn - 1) - mu) / %s;", s_var))
  L$add(sprintf("        double z_lo_val = yn > 1 ? (thresh_n(yn - 2) - mu) / %s : 0.0;", s_var))
  L$add(sprintf("        auto dr = batch_sdt::dpsdt_non_top_cell%s(z_lo_val, z_hi_val, lambda_val, yn, K);", lsuf))
  L$add("        obs_lp = dr.lp;")
  L$add(sprintf("        d_mu_raw = -(yn > 1 ? dr.d_z_lo : 0.0) / %s - dr.d_z_hi / %s;", s_var, s_var))
  L$add("        d_lambda_raw = dr.d_lambda;")
  L$add(sprintf("        d_thresh_lo = yn > 1 ? dr.d_z_lo / %s : 0.0;", s_var))
  L$add(sprintf("        d_thresh_hi = dr.d_z_hi / %s;", s_var))
  L$add("        if (yn == 1) { k_hi = 0; } else { k_lo = yn - 2; k_hi = yn - 1; }")
  if (has_sigma) {
    L$add("        d_sigma_raw = 0.0;")
    L$add("        if (yn > 1) d_sigma_raw -= dr.d_z_lo * (thresh_n(yn - 2) - mu) / (s * s) * sigma_val;")
    L$add("        d_sigma_raw -= dr.d_z_hi * (thresh_n(yn == 1 ? 0 : yn - 1) - mu) / (s * s) * sigma_val;")
  }
  L$add("      }")

  # --- New item: standard EVSDT (mu=0, s=1) ---
  L$add("    } else {")
  L$add("      batch_sdt::CellResult res;")
  L$add("      if (yn == 1) {")
  L$add(sprintf("        res = batch_sdt::edge_lo%s(thresh_n(0));", lsuf))
  L$add("        k_hi = 0;")
  L$add("      } else if (yn == K) {")
  L$add(sprintf("        res = batch_sdt::edge_hi%s(thresh_n(n_thresh - 1));", lsuf))
  L$add("        k_lo = n_thresh - 1;")
  L$add("      } else {")
  L$add(sprintf("        res = batch_sdt::cell_log_prob%s(thresh_n(yn - 2), thresh_n(yn - 1));", lsuf))
  L$add("        k_lo = yn - 2;")
  L$add("        k_hi = yn - 1;")
  L$add("      }")
  L$add("      obs_lp = res.lp;")
  L$add("      d_thresh_lo = res.d_z_lo;")
  L$add("      d_thresh_hi = res.d_z_hi;")
  L$add("      d_mu_raw = 0.0;")
  L$add("      d_lambda_raw = 0.0;")
  L$add("    }")

  if (params$.has_counts) {
    L$add("    total_lp += obs_lp * count_n;")
  } else {
    L$add("    total_lp += obs_lp;")
  }
}


#' Emit mixture SDT likelihood
#' @noRd
emit_likelihood_mixture <- function(L, params) {
  has_sigma <- isTRUE(params$.has_sigma)
  has_sigma2 <- isTRUE(params$.has_sigma2)
  has_dprime2 <- isTRUE(params$.has_dprime2)
  has_lure <- isTRUE(params$.has_lure_mixture)
  has_sigma_L <- isTRUE(params$.has_sigma_L)
  lsuf <- logit_suffix(params)

  s1 <- if (has_sigma) "s" else "1.0"
  s2 <- if (has_sigma2) "s2" else "1.0"

  L$add("    // Mixture SDT likelihood")
  L$add("    double obs_lp = 0.0;")
  L$add("    double d_mu1_raw = 0.0;")
  L$add("    double d_mu2_raw = 0.0;")
  L$add("    double d_lambda_raw = 0.0;")
  L$add("    double d_thresh_lo = 0.0, d_thresh_hi = 0.0;")
  if (has_sigma) {
    L$add("    double d_sigma_raw = 0.0;")
  }
  if (has_sigma2) {
    L$add("    double d_sigma2_raw = 0.0;")
  }
  if (has_lure) {
    L$add("    double d_dprime_L_raw = 0.0;")
    if (has_sigma_L) {
      L$add("    double d_sigma_L_raw = 0.0;")
    }
    L$add("    double d_lambda_L_raw = 0.0;")
  }
  L$add("    int k_lo = -1, k_hi = -1;")
  L$add("")

  has_lambda <- isTRUE(params$.has_lambda)

  L$add("    if (is_old_n > 0) {")

  if (has_lambda) {
    # Signal mixture: P = lambda * p1(dprime) + (1-lambda) * p2(dprime2)
    if (has_dprime2) {
      L$add("      double mu2 = dprime2_n;")
    } else {
      L$add("      double mu2 = 0.0;")
    }

    L$add("      double z1_lo = 0.0, z1_hi = 0.0, z2_lo = 0.0, z2_hi = 0.0;")
    L$add("      if (yn == 1) {")
    L$add(sprintf("        z1_hi = (thresh_n(0) - mu) / %s;", s1))
    L$add(sprintf("        z2_hi = (thresh_n(0) - mu2) / %s;", s2))
    L$add("        k_hi = 0;")
    L$add("      } else if (yn == K) {")
    L$add(sprintf("        z1_lo = (thresh_n(n_thresh - 1) - mu) / %s;", s1))
    L$add(sprintf("        z2_lo = (thresh_n(n_thresh - 1) - mu2) / %s;", s2))
    L$add("        k_lo = n_thresh - 1;")
    L$add("      } else {")
    L$add(sprintf("        z1_lo = (thresh_n(yn - 2) - mu) / %s;", s1))
    L$add(sprintf("        z1_hi = (thresh_n(yn - 1) - mu) / %s;", s1))
    L$add(sprintf("        z2_lo = (thresh_n(yn - 2) - mu2) / %s;", s2))
    L$add(sprintf("        z2_hi = (thresh_n(yn - 1) - mu2) / %s;", s2))
    L$add("        k_lo = yn - 2;")
    L$add("        k_hi = yn - 1;")
    L$add("      }")

    L$add(sprintf("      auto mr = batch_sdt::mixture_cell%s(z1_lo, z1_hi, z2_lo, z2_hi, lambda_val, yn, K);", lsuf))
    L$add("      obs_lp = mr.lp;")

    L$add(sprintf("      d_mu1_raw = -(mr.d_z1_lo + mr.d_z1_hi) / %s;", s1))
    L$add(sprintf("      d_mu2_raw = -(mr.d_z2_lo + mr.d_z2_hi) / %s;", s2))
    L$add("      d_lambda_raw = mr.d_lambda;")

    L$add(sprintf("      d_thresh_lo = mr.d_z1_lo / %s + mr.d_z2_lo / %s;", s1, s2))
    L$add(sprintf("      d_thresh_hi = mr.d_z1_hi / %s + mr.d_z2_hi / %s;", s1, s2))

    if (has_sigma) {
      L$add("      d_sigma_raw = 0.0;")
      L$add(sprintf("      if (k_lo >= 0) d_sigma_raw -= mr.d_z1_lo * (thresh_n(k_lo) - mu) / (%s * %s) * sigma_val;", s1, s1))
      L$add(sprintf("      if (k_hi >= 0) d_sigma_raw -= mr.d_z1_hi * (thresh_n(k_hi) - mu) / (%s * %s) * sigma_val;", s1, s1))
    }
    if (has_sigma2) {
      L$add("      d_sigma2_raw = 0.0;")
      L$add(sprintf("      if (k_lo >= 0) d_sigma2_raw -= mr.d_z2_lo * (thresh_n(k_lo) - mu2) / (%s * %s) * sigma2_val;", s2, s2))
      L$add(sprintf("      if (k_hi >= 0) d_sigma2_raw -= mr.d_z2_hi * (thresh_n(k_hi) - mu2) / (%s * %s) * sigma2_val;", s2, s2))
    }
  } else {
    # Lure-only mixture: old items use standard EVSDT/UVSDT (no signal mixture)
    L$add("      if (yn == 1) {")
    L$add(sprintf("        auto res = batch_sdt::edge_lo%s((thresh_n(0) - mu) / %s);", lsuf, s1))
    L$add("        obs_lp = res.lp;")
    L$add(sprintf("        d_thresh_hi = res.d_z_hi / %s;", s1))
    L$add("        k_hi = 0;")
    L$add("      } else if (yn == K) {")
    L$add(sprintf("        auto res = batch_sdt::edge_hi%s((thresh_n(n_thresh - 1) - mu) / %s);", lsuf, s1))
    L$add("        obs_lp = res.lp;")
    L$add(sprintf("        d_thresh_lo = res.d_z_lo / %s;", s1))
    L$add("        k_lo = n_thresh - 1;")
    L$add("      } else {")
    L$add(sprintf("        auto res = batch_sdt::cell_log_prob%s((thresh_n(yn - 2) - mu) / %s, (thresh_n(yn - 1) - mu) / %s);", lsuf, s1, s1))
    L$add("        obs_lp = res.lp;")
    L$add(sprintf("        d_thresh_lo = res.d_z_lo / %s;", s1))
    L$add(sprintf("        d_thresh_hi = res.d_z_hi / %s;", s1))
    L$add("        k_lo = yn - 2;")
    L$add("        k_hi = yn - 1;")
    L$add("      }")
    L$add(sprintf("      d_mu1_raw = -(d_thresh_lo + d_thresh_hi);"))
    if (has_sigma) {
      L$add("      d_sigma_raw = 0.0;")
      L$add(sprintf("      if (k_lo >= 0) d_sigma_raw -= d_thresh_lo * (thresh_n(k_lo) - mu) / %s * sigma_val;", s1))
      L$add(sprintf("      if (k_hi >= 0) d_sigma_raw -= d_thresh_hi * (thresh_n(k_hi) - mu) / %s * sigma_val;", s1))
    }
  }

  L$add("    } else {")

  if (has_lure) {
    # --- New items: lure mixture ---
    # P(y|new) = lambda_L * cell_prob(z_lure) + (1 - lambda_L) * cell_prob(z_noise)
    # z_lure = (thresh + dprime_L) / sigma_L  (note: +dprime_L because lure shifts distribution)
    # z_noise = thresh  (standard noise, mu=0, s=1)
    sL <- if (has_sigma_L) "sigma_L_val" else "1.0"

    L$add("      // Lure mixture for new items")
    L$add("      double zL_lo = 0.0, zL_hi = 0.0, zR_lo = 0.0, zR_hi = 0.0;")
    L$add("      if (yn == 1) {")
    L$add(sprintf("        zL_hi = (thresh_n(0) + dprime_L_n) / %s;", sL))
    L$add("        zR_hi = thresh_n(0);")
    L$add("        k_hi = 0;")
    L$add("      } else if (yn == K) {")
    L$add(sprintf("        zL_lo = (thresh_n(n_thresh - 1) + dprime_L_n) / %s;", sL))
    L$add("        zR_lo = thresh_n(n_thresh - 1);")
    L$add("        k_lo = n_thresh - 1;")
    L$add("      } else {")
    L$add(sprintf("        zL_lo = (thresh_n(yn - 2) + dprime_L_n) / %s;", sL))
    L$add(sprintf("        zL_hi = (thresh_n(yn - 1) + dprime_L_n) / %s;", sL))
    L$add("        zR_lo = thresh_n(yn - 2);")
    L$add("        zR_hi = thresh_n(yn - 1);")
    L$add("        k_lo = yn - 2;")
    L$add("        k_hi = yn - 1;")
    L$add("      }")

    # Reuse source_mixture_cell (same structure: P = lambda * comp1 + (1-lambda) * comp2)
    L$add(sprintf("      auto lr = batch_sdt::source_mixture_cell%s(zL_lo, zL_hi, zR_lo, zR_hi, lambda_L_val, yn, K);", lsuf))
    L$add("      obs_lp = lr.lp;")

    # d_dprime_L: dz_lure/d_dprime_L = 1/sigma_L
    L$add(sprintf("      d_dprime_L_raw = (lr.d_z1_lo + lr.d_z1_hi) / %s;", sL))

    # d_lambda_L
    L$add("      d_lambda_L_raw = lr.d_lambda;")

    # d_sigma_L: dz/d_sigma_L_val = -(thresh+dprime_L)/(sigma_L^2), chain through exp
    if (has_sigma_L) {
      L$add("      d_sigma_L_raw = 0.0;")
      L$add(sprintf("      if (k_lo >= 0) d_sigma_L_raw -= lr.d_z1_lo * (thresh_n(k_lo) + dprime_L_n) / (%s * %s) * sigma_L_val;", sL, sL))
      L$add(sprintf("      if (k_hi >= 0) d_sigma_L_raw -= lr.d_z1_hi * (thresh_n(k_hi) + dprime_L_n) / (%s * %s) * sigma_L_val;", sL, sL))
    }

    # d_thresh for lure mixture: from lure component dz/d_thresh = 1/sigma_L, from ref component dz/d_thresh = 1
    L$add(sprintf("      d_thresh_lo = lr.d_z1_lo / %s + lr.d_z2_lo;", sL))
    L$add(sprintf("      d_thresh_hi = lr.d_z1_hi / %s + lr.d_z2_hi;", sL))

    L$add("      d_mu1_raw = 0.0;")
    L$add("      d_mu2_raw = 0.0;")
    L$add("      d_lambda_raw = 0.0;")
  } else {
    # New items: standard EVSDT (mu=0, s=1)
    L$add("      batch_sdt::CellResult res;")
    L$add("      if (yn == 1) {")
    L$add(sprintf("        res = batch_sdt::edge_lo%s(thresh_n(0));", lsuf))
    L$add("        k_hi = 0;")
    L$add("      } else if (yn == K) {")
    L$add(sprintf("        res = batch_sdt::edge_hi%s(thresh_n(n_thresh - 1));", lsuf))
    L$add("        k_lo = n_thresh - 1;")
    L$add("      } else {")
    L$add(sprintf("        res = batch_sdt::cell_log_prob%s(thresh_n(yn - 2), thresh_n(yn - 1));", lsuf))
    L$add("        k_lo = yn - 2;")
    L$add("        k_hi = yn - 1;")
    L$add("      }")
    L$add("      obs_lp = res.lp;")
    L$add("      d_thresh_lo = res.d_z_lo;")
    L$add("      d_thresh_hi = res.d_z_hi;")
    L$add("      d_mu1_raw = 0.0;")
    L$add("      d_mu2_raw = 0.0;")
    L$add("      d_lambda_raw = 0.0;")
  }

  L$add("    }")

  if (params$.has_counts) {
    L$add("    total_lp += obs_lp * count_n;")
  } else {
    L$add("    total_lp += obs_lp;")
  }
}


#' Emit source mixture SDT likelihood
#' @noRd
emit_likelihood_source_mixture <- function(L, params) {
  has_dprime_B <- isTRUE(params$.has_dprime_B)
  has_lambda_B <- isTRUE(params$.has_lambda_B)
  lsuf <- logit_suffix(params)

  L$add("    // Source mixture SDT likelihood")
  L$add("    double obs_lp = 0.0;")
  L$add("    double d_mu_raw = 0.0;")  # gradient for dprime (source A, or symmetric)
  L$add("    double d_lambda_raw = 0.0;")
  L$add("    double d_thresh_lo = 0.0, d_thresh_hi = 0.0;")
  if (has_dprime_B) {
    L$add("    double d_mu_B_raw = 0.0;")
  }
  if (has_lambda_B) {
    L$add("    double d_lambda_B_raw = 0.0;")
  }
  L$add("    int k_lo = -1, k_hi = -1;")
  L$add("")

  # Select d and lambda based on source
  L$add("    // Select parameters based on source")
  if (has_dprime_B) {
    L$add("    double d_eff = source_n == 0 ? mu : dprime_B_n;")
  } else {
    # Symmetric: d_B = -dprime
    L$add("    double d_eff = source_n == 0 ? mu : -mu;")
  }
  if (has_lambda_B) {
    L$add("    double lam_eff = source_n == 0 ? lambda_val : lambda_B_val;")
  } else {
    L$add("    double lam_eff = lambda_val;")
  }

  # z_signal = thresh - d_eff (signal at d), z_noise = thresh (noise at 0)
  L$add("")
  L$add("    double zS_lo = 0.0, zS_hi = 0.0, zN_lo = 0.0, zN_hi = 0.0;")
  L$add("    if (yn == 1) {")
  L$add("      zS_hi = thresh_n(0) - d_eff;")
  L$add("      zN_hi = thresh_n(0);")
  L$add("      k_hi = 0;")
  L$add("    } else if (yn == K) {")
  L$add("      zS_lo = thresh_n(n_thresh - 1) - d_eff;")
  L$add("      zN_lo = thresh_n(n_thresh - 1);")
  L$add("      k_lo = n_thresh - 1;")
  L$add("    } else {")
  L$add("      zS_lo = thresh_n(yn - 2) - d_eff;")
  L$add("      zS_hi = thresh_n(yn - 1) - d_eff;")
  L$add("      zN_lo = thresh_n(yn - 2);")
  L$add("      zN_hi = thresh_n(yn - 1);")
  L$add("      k_lo = yn - 2;")
  L$add("      k_hi = yn - 1;")
  L$add("    }")

  L$add(sprintf("    auto sr = batch_sdt::source_mixture_cell%s(zS_lo, zS_hi, zN_lo, zN_hi, lam_eff, yn, K);", lsuf))
  L$add("    obs_lp = sr.lp;")

  # d_mu/d_mu_B: dz_signal/d_d = -1
  # sr.d_z1_lo and sr.d_z1_hi are d_lp/d_z_signal_{lo,hi}
  # d_lp/d_d_eff = -(sr.d_z1_lo + sr.d_z1_hi)
  L$add("    double d_d_eff = -(sr.d_z1_lo + sr.d_z1_hi);")
  L$add("    double d_lam_eff = sr.d_lambda;")

  # Route d_d_eff to dprime or dprime_B
  if (has_dprime_B) {
    L$add("    if (source_n == 0) {")
    L$add("      d_mu_raw = d_d_eff;")
    L$add("    } else {")
    L$add("      d_mu_B_raw = d_d_eff;")
    L$add("    }")
  } else {
    # Symmetric: d_A = dprime, d_B = -dprime
    # d_lp/d_dprime = d_lp/d_d_eff * d_d_eff/d_dprime
    # For source A: d_d_eff/d_dprime = 1
    # For source B: d_d_eff/d_dprime = -1
    L$add("    d_mu_raw = source_n == 0 ? d_d_eff : -d_d_eff;")
  }

  # Route d_lam_eff to lambda or lambda_B
  if (has_lambda_B) {
    L$add("    if (source_n == 0) {")
    L$add("      d_lambda_raw = d_lam_eff;")
    L$add("    } else {")
    L$add("      d_lambda_B_raw = d_lam_eff;")
    L$add("    }")
  } else {
    L$add("    d_lambda_raw = d_lam_eff;")
  }

  # d_thresh: from signal component dz/d_thresh = 1, from noise dz/d_thresh = 1
  L$add("    d_thresh_lo = sr.d_z1_lo + sr.d_z2_lo;")
  L$add("    d_thresh_hi = sr.d_z1_hi + sr.d_z2_hi;")

  if (params$.has_counts) {
    L$add("    total_lp += obs_lp * count_n;")
  } else {
    L$add("    total_lp += obs_lp;")
  }
}


# ============================================================================
# VRDP2D: Threshold2 computation
# ============================================================================

#' Emit source dimension (dimension 2) threshold construction for vrdp2d
#' @noRd
emit_thresh2_computation <- function(L, model_data, params, has_crit2_re) {
  crit2 <- model_data$criterion2
  if (is.null(crit2)) return()
  is_crit2_intercept_only <- isTRUE(crit2$is_intercept_only) || crit2$n_coef == 1
  gap_link <- if (!is.null(params$.gap_link)) params$.gap_link else "log"
  gap_fn <- if (identical(gap_link, "softplus")) "std::log1p(std::exp(%s))" else "std::exp(%s)"

  if (has_crit2_re) {
    # --- Thresholds with criterion2 RE ---
    if (is_crit2_intercept_only) {
      L$add("    double thresh2_mid_val = thresh_mid2_d_vec(0);")
    } else {
      L$add("    double thresh2_mid_val = 0.0;")
      L$add("    for (int p = 0; p < P_crit2; ++p) thresh2_mid_val += X_crit2_d(n, p) * thresh_mid2_d_vec(p);")
    }

    crit2_re_idx <- 0
    for (group in names(params$criterion2$random)) {
      crit2_re_idx <- crit2_re_idx + 1
      re <- params$criterion2$random[[group]]
      emit_crit2_re_mid(L, crit2_re_idx, re, group, "thresh2_mid_val")
    }
    L$add("    thresh2_n(mid2) = thresh2_mid_val;")

    L$add("    for (int k = mid2 + 1; k < n_thresh2; ++k) {")
    L$add("      int gi = k - mid2 - 1;")
    if (is_crit2_intercept_only) {
      L$add("      double eff_gap2 = pop_gap2_d(gi);")
    } else {
      L$add("      double eff_gap2 = 0.0;")
      L$add("      for (int p = 0; p < P_crit2; ++p) eff_gap2 += X_crit2_d(n, p) * gaps2_mat_d(gi, p);")
    }
    crit2_re_idx <- 0
    for (group in names(params$criterion2$random)) {
      crit2_re_idx <- crit2_re_idx + 1
      re <- params$criterion2$random[[group]]
      emit_crit2_re_gap(L, crit2_re_idx, re, group, "eff_gap2", "k + 1")
    }
    L$add(sprintf("      exp_eff_gaps2(gi) = %s;", sprintf(gap_fn, "eff_gap2")))
    L$add("      thresh2_n(k) = thresh2_n(k - 1) + exp_eff_gaps2(gi);")
    L$add("    }")

    L$add("    for (int kd = 1; kd <= mid2; ++kd) {")
    L$add("      int k = mid2 - kd;")
    L$add("      int gi = n_upper2 + kd - 1;")
    if (is_crit2_intercept_only) {
      L$add("      double eff_gap2 = pop_gap2_d(gi);")
    } else {
      L$add("      double eff_gap2 = 0.0;")
      L$add("      for (int p = 0; p < P_crit2; ++p) eff_gap2 += X_crit2_d(n, p) * gaps2_mat_d(gi, p);")
    }
    crit2_re_idx <- 0
    for (group in names(params$criterion2$random)) {
      crit2_re_idx <- crit2_re_idx + 1
      re <- params$criterion2$random[[group]]
      emit_crit2_re_gap(L, crit2_re_idx, re, group, "eff_gap2", "k + 1")
    }
    L$add(sprintf("      exp_eff_gaps2(gi) = %s;", sprintf(gap_fn, "eff_gap2")))
    L$add("      thresh2_n(k) = thresh2_n(k + 1) - exp_eff_gaps2(gi);")
    L$add("    }")

  } else {
    # --- Population-only thresholds ---
    if (is_crit2_intercept_only) {
      L$add("    thresh2_n(mid2) = thresh_mid2_d_vec(0);")
    } else {
      L$add("    double thresh2_mid_val = 0.0;")
      L$add("    for (int p = 0; p < P_crit2; ++p) thresh2_mid_val += X_crit2_d(n, p) * thresh_mid2_d_vec(p);")
      L$add("    thresh2_n(mid2) = thresh2_mid_val;")
    }
    L$add("    for (int k = mid2 + 1; k < n_thresh2; ++k) {")
    L$add("      int gi = k - mid2 - 1;")
    if (!is_crit2_intercept_only) {
      L$add("      double gval2 = 0.0;")
      L$add("      for (int p = 0; p < P_crit2; ++p) gval2 += X_crit2_d(n, p) * gaps2_mat_d(gi, p);")
      L$add(sprintf("      thresh2_n(k) = thresh2_n(k - 1) + %s;", sprintf(gap_fn, "gval2")))
    } else {
      L$add("      thresh2_n(k) = thresh2_n(k - 1) + pop_exp_gaps2(gi);")
    }
    L$add("    }")
    L$add("    for (int kd = 1; kd <= mid2; ++kd) {")
    L$add("      int k = mid2 - kd;")
    L$add("      int gi = n_upper2 + kd - 1;")
    if (!is_crit2_intercept_only) {
      L$add("      double gval2 = 0.0;")
      L$add("      for (int p = 0; p < P_crit2; ++p) gval2 += X_crit2_d(n, p) * gaps2_mat_d(gi, p);")
      L$add(sprintf("      thresh2_n(k) = thresh2_n(k + 1) - %s;", sprintf(gap_fn, "gval2")))
    } else {
      L$add("      thresh2_n(k) = thresh2_n(k + 1) - pop_exp_gaps2(gi);")
    }
    L$add("    }")
  }
}


#' Emit criterion2 RE contribution to mid-threshold
#' @noRd
emit_crit2_re_mid <- function(L, crit2_re_idx, re, group, target) {
  g_var <- sprintf("g_%s", group)
  n_re_terms <- if (!is.null(re$n_re_terms)) re$n_re_terms else 1

  if (isTRUE(re$use_z_matrix)) {
    L$add(sprintf("    for (int j = 0; j < %d; ++j) %s += Z_crit2_%d_d(n, j) * u_crit2_%d_d(%s, mid2 * %d + j);",
                   n_re_terms, target, crit2_re_idx, crit2_re_idx, g_var, n_re_terms))
  } else if (n_re_terms == 1) {
    L$add(sprintf("    %s += u_crit2_%d_d(%s, mid2);", target, crit2_re_idx, g_var))
  } else {
    idx_var <- sprintf("idx_criterion2_%s", group)
    L$add(sprintf("    if (%s[n] > 0) %s += u_crit2_%d_d(%s, mid2 * %d + %s[n] - 1);",
                   idx_var, target, crit2_re_idx, g_var, n_re_terms, idx_var))
  }
}


#' Emit criterion2 RE contribution to a gap
#' @noRd
emit_crit2_re_gap <- function(L, crit2_re_idx, re, group, target, thresh_k_1based) {
  g_var <- sprintf("g_%s", group)
  n_re_terms <- if (!is.null(re$n_re_terms)) re$n_re_terms else 1

  if (isTRUE(re$use_z_matrix)) {
    L$add(sprintf("      for (int j = 0; j < %d; ++j) %s += Z_crit2_%d_d(n, j) * u_crit2_%d_d(%s, k * %d + j);",
                   n_re_terms, target, crit2_re_idx, crit2_re_idx, g_var, n_re_terms))
  } else if (n_re_terms == 1) {
    L$add(sprintf("      %s += u_crit2_%d_d(%s, k);", target, crit2_re_idx, g_var))
  } else {
    idx_var <- sprintf("idx_criterion2_%s", group)
    L$add(sprintf("      if (%s[n] > 0) %s += u_crit2_%d_d(%s, k * %d + %s[n] - 1);",
                   idx_var, target, crit2_re_idx, g_var, n_re_terms, idx_var))
  }
}


# ============================================================================
# VRDP2D: Likelihood computation
# ============================================================================

#' Emit VRDP2D likelihood -- 2D mixture with analytic gradients
#' @noRd
emit_likelihood_vrdp2d <- function(L, params) {
  has_sigma <- isTRUE(params$.has_sigma)
  has_sigma2 <- isTRUE(params$.has_sigma2)
  has_discrim_B <- isTRUE(params$.has_discrim_B)
  has_counts <- params$.has_counts

  L$add("    // VRDP2D likelihood")
  L$add("    // Compute source_d and effective parameters")
  if (has_discrim_B) {
    L$add("    double source_d = (item_type_n == 2) ? discrim_n : discrim_B_n;")
  } else {
    L$add("    double source_d = (item_type_n == 2) ? discrim_n : -discrim_n;")
  }

  sigma_item_expr <- if (has_sigma) "sigma_val" else "1.0"
  sigma_S_expr <- if (has_sigma2) "sigma2_val" else "1.0"

  L$add(sprintf("    double sigma_item_val = %s;", sigma_item_expr))
  L$add(sprintf("    double sigma_S_val = %s;", sigma_S_expr))
  L$add("    double R_val = lambda_val;")
  L$add("")

  L$add("    auto vr = batch_sdt::vrdp2d_cell(")
  L$add("        yn, yn2, item_type_n, K1, K2,")
  L$add("        dprime_n, dprime2_n, source_d,")
  L$add("        R_val, sigma_item_val, sigma_S_val,")
  L$add("        thresh1_n.data(), n_thresh1,")
  L$add("        thresh2_n.data(), n_thresh2);")
  L$add("")

  L$add("    double obs_lp = vr.lp;")

  if (has_counts) {
    L$add("    total_lp += obs_lp * count_n;")
  } else {
    L$add("    total_lp += obs_lp;")
  }
}


# ============================================================================
# VRDP2D: Gradient accumulation
# ============================================================================

#' Emit VRDP2D gradient accumulation
#' @noRd
emit_gradient_accumulation_vrdp2d <- function(L, model_data, params, has_crit_re, has_crit2_re) {
  has_sigma <- isTRUE(params$.has_sigma)
  has_sigma2 <- isTRUE(params$.has_sigma2)
  has_discrim_B <- isTRUE(params$.has_discrim_B)
  has_counts <- params$.has_counts
  count_mult <- if (has_counts) " * count_n" else ""

  # ---- dprime (d'_F familiarity) gradient ----
  has_dp_fixed <- isTRUE(params$.has_dp_fixed)
  L$add(sprintf("    double d_dprime_n = vr.d_dprime_F%s;", count_mult))
  if (has_dp_fixed) {
    L$add("    for (int p = 0; p < P_dp; ++p) grad_beta_dp(p) += d_dprime_n * X_dp_d(n, p);")
  }
  dp_re_idx <- 0
  if (!is.null(params$dprime$random)) {
    for (group in names(params$dprime$random)) {
      dp_re_idx <- dp_re_idx + 1
      re <- params$dprime$random[[group]]
      tag <- dp_re_tag(dp_re_idx)
      emit_re_gradient(L, tag, re, group, "d_dprime_n", "dprime", dp_re_idx, "dp")
    }
  }

  # ---- dprime2 (d'_R recollection boost) gradient ----
  if (isTRUE(params$.has_dprime2)) {
    L$add("")
    L$add(sprintf("    double d_dprime2_n = vr.d_dprime_R%s;", count_mult))
    L$add("    for (int p = 0; p < P_dp2; ++p) grad_beta_dp2(p) += d_dprime2_n * X_dp2_d(n, p);")
    dp2_re_idx <- 0
    if (!is.null(params$dprime2$random)) {
      for (group in names(params$dprime2$random)) {
        dp2_re_idx <- dp2_re_idx + 1
        re <- params$dprime2$random[[group]]
        tag <- dp2_re_tag(dp2_re_idx)
        emit_re_gradient(L, tag, re, group, "d_dprime2_n", "dprime2", dp2_re_idx, "dp2")
      }
    }
  }

  # ---- discrim gradient ----
  # d_source_d chains to discrim_n or discrim_B_n
  L$add("")
  if (has_discrim_B) {
    L$add("    if (item_type_n == 2) {")
    L$add(sprintf("      double d_discrim_n = vr.d_source_d%s;", count_mult))
    L$add("      for (int p = 0; p < P_disc; ++p) grad_beta_disc(p) += d_discrim_n * X_disc_d(n, p);")
    disc_re_idx <- 0
    if (!is.null(params$discrim$random)) {
      for (group in names(params$discrim$random)) {
        disc_re_idx <- disc_re_idx + 1
        re <- params$discrim$random[[group]]
        tag <- disc_re_tag(disc_re_idx)
        emit_re_gradient(L, tag, re, group, "d_discrim_n", "discrim", disc_re_idx, "disc", indent = "      ")
      }
    }
    L$add("    } else if (item_type_n == 3) {")
    L$add(sprintf("      double d_discrim_B_n = vr.d_source_d%s;", count_mult))
    L$add("      for (int p = 0; p < P_discB; ++p) grad_beta_discB(p) += d_discrim_B_n * X_discB_d(n, p);")
    discB_re_idx <- 0
    if (!is.null(params$discrim_B$random)) {
      for (group in names(params$discrim_B$random)) {
        discB_re_idx <- discB_re_idx + 1
        re <- params$discrim_B$random[[group]]
        tag <- discB_re_tag(discB_re_idx)
        emit_re_gradient(L, tag, re, group, "d_discrim_B_n", "discrim_B", discB_re_idx, "discB", indent = "      ")
      }
    }
    L$add("    }")
  } else {
    # Symmetric: source_d = discrim_n for A, -discrim_n for B
    # d_logP/d_discrim_n = d_logP/d_source_d * d_source_d/d_discrim_n
    # For A: d_source_d/d_discrim_n = 1, For B: = -1
    L$add("    {")
    L$add(sprintf("      double d_discrim_n = (item_type_n == 2 ? vr.d_source_d : -vr.d_source_d)%s;", count_mult))
    L$add("      for (int p = 0; p < P_disc; ++p) grad_beta_disc(p) += d_discrim_n * X_disc_d(n, p);")
    disc_re_idx <- 0
    if (!is.null(params$discrim$random)) {
      for (group in names(params$discrim$random)) {
        disc_re_idx <- disc_re_idx + 1
        re <- params$discrim$random[[group]]
        tag <- disc_re_tag(disc_re_idx)
        emit_re_gradient(L, tag, re, group, "d_discrim_n", "discrim", disc_re_idx, "disc", indent = "      ")
      }
    }
    L$add("    }")
  }

  # ---- lambda (R recollection probability) gradient ----
  # d_logP/d_lambda_n = d_logP/d_R * d_R/d_lambda_n (inv_logit chain rule)
  L$add("")
  L$add(sprintf("    double d_lambda_n = vr.d_lambda * lambda_val * (1.0 - lambda_val)%s;", count_mult))
  L$add("    for (int p = 0; p < P_lam; ++p) grad_beta_lam(p) += d_lambda_n * X_lam_d(n, p);")
  lam_re_idx <- 0
  if (!is.null(params$lambda$random)) {
    for (group in names(params$lambda$random)) {
      lam_re_idx <- lam_re_idx + 1
      re <- params$lambda$random[[group]]
      tag <- lam_re_tag(lam_re_idx)
      emit_re_gradient(L, tag, re, group, "d_lambda_n", "lambda", lam_re_idx, "lam")
    }
  }

  # ---- sigma (sigma_item) gradient ----
  if (has_sigma) {
    # d_logP/d_sigma_n = d_logP/d_sigma_item * d_sigma_item/d_sigma_n (exp chain)
    # = d_logP/d_sigma_item * sigma_item
    L$add("")
    L$add(sprintf("    double d_sigma_n = vr.d_sigma_item * sigma_val%s;", count_mult))
    L$add("    for (int p = 0; p < P_sig; ++p) grad_beta_sig(p) += d_sigma_n * X_sig_d(n, p);")
    sig_re_idx <- 0
    if (!is.null(params$sigma$random)) {
      for (group in names(params$sigma$random)) {
        sig_re_idx <- sig_re_idx + 1
        re <- params$sigma$random[[group]]
        tag <- sig_re_tag(sig_re_idx)
        emit_re_gradient(L, tag, re, group, "d_sigma_n", "sigma", sig_re_idx, "sig")
      }
    }
  }

  # ---- sigma2 (sigma_S) gradient ----
  if (has_sigma2) {
    L$add("")
    L$add(sprintf("    double d_sigma2_n = vr.d_sigma_S * sigma2_val%s;", count_mult))
    L$add("    for (int p = 0; p < P_sig2; ++p) grad_beta_sig2(p) += d_sigma2_n * X_sig2_d(n, p);")
    sig2_re_idx <- 0
    if (!is.null(params$sigma2$random)) {
      for (group in names(params$sigma2$random)) {
        sig2_re_idx <- sig2_re_idx + 1
        re <- params$sigma2$random[[group]]
        tag <- sig2_re_tag(sig2_re_idx)
        emit_re_gradient(L, tag, re, group, "d_sigma2_n", "sigma2", sig2_re_idx, "sig2")
      }
    }
  }

  # ---- Threshold1 gradient (item dimension) ----
  L$add("")
  L$add("    // Threshold1 gradient (item dimension)")
  L$add("    double d_thresh1_lo = vr.d_thresh1_lo;")
  L$add("    double d_thresh1_hi = vr.d_thresh1_hi;")
  L$add("    int k_lo = vr.k1_lo;")
  L$add("    int k_hi = vr.k1_hi;")

  if (has_counts) {
    d_lo1 <- "d_thresh1_lo * count_n"
    d_hi1 <- "d_thresh1_hi * count_n"
  } else {
    d_lo1 <- "d_thresh1_lo"
    d_hi1 <- "d_thresh1_hi"
  }

  # Use the existing threshold propagation infrastructure
  crit <- model_data$criterion
  is_crit_intercept_only <- isTRUE(crit$is_intercept_only) || crit$n_coef == 1
  emit_thresh_propagation(L, model_data, params, has_crit_re, d_lo1, d_hi1, is_crit_intercept_only)

  # ---- Threshold2 gradient (source dimension) ----
  L$add("")
  L$add("    // Threshold2 gradient (source dimension)")

  if (has_counts) {
    d_lo2 <- "vr.d_thresh2_lo * count_n"
    d_hi2 <- "vr.d_thresh2_hi * count_n"
  } else {
    d_lo2 <- "vr.d_thresh2_lo"
    d_hi2 <- "vr.d_thresh2_hi"
  }

  varying_sc <- isTRUE(params$.varying_source_criteria)

  if (varying_sc) {
    # Varying source criteria: propagate gradients to per-bin accumulators
    L$add("    {")
    L$add("      int bin = yn - 1;")
    L$add("      Eigen::VectorXd bin_exp_gaps = vary_exp_gaps2.row(bin).transpose();")
    L$add(sprintf("      double& mid_ref = grad_vary_mid2(bin);"))
    L$add("      auto propagate_vary = [&](int k, double d) {")
    L$add("        mid_ref += d;")
    L$add("        if (k > mid2) { for (int j = 0; j <= k-mid2-1; ++j) grad_vary_gaps2(bin, j) += d * bin_exp_gaps(j); }")
    L$add("        else if (k < mid2) { for (int j = n_upper2; j <= n_upper2+(mid2-k-1); ++j) grad_vary_gaps2(bin, j) += d * (-bin_exp_gaps(j)); }")
    L$add("      };")
    L$add(sprintf("      if (vr.k2_lo >= 0) propagate_vary(vr.k2_lo, %s);", d_lo2))
    L$add(sprintf("      if (vr.k2_hi >= 0) propagate_vary(vr.k2_hi, %s);", d_hi2))
    L$add("    }")
  } else {
    crit2 <- model_data$criterion2
    is_crit2_intercept_only <- is.null(crit2) || isTRUE(crit2$is_intercept_only) || crit2$n_coef == 1

    # Inline threshold2 propagation (similar to thresh1 but for dim2)
    if (has_crit2_re) {
      L$add("    double g_crit2_mid = 0.0;")
      L$add("    Eigen::VectorXd g_crit2_gaps = Eigen::VectorXd::Zero(n_gaps2);")

      if (is_crit2_intercept_only) {
        L$add("    if (vr.k2_lo >= 0) {")
        L$add(sprintf("      batch_sdt::propagate_thresh_grad_with_re(vr.k2_lo, %s, mid2, n_thresh2, n_upper2,", d_lo2))
        L$add("          exp_eff_gaps2, grad_thresh_mid2_pop, grad_log_gaps2_pop, g_crit2_mid, g_crit2_gaps);")
        L$add("    }")
        L$add("    if (vr.k2_hi >= 0) {")
        L$add(sprintf("      batch_sdt::propagate_thresh_grad_with_re(vr.k2_hi, %s, mid2, n_thresh2, n_upper2,", d_hi2))
        L$add("          exp_eff_gaps2, grad_thresh_mid2_pop, grad_log_gaps2_pop, g_crit2_mid, g_crit2_gaps);")
        L$add("    }")
      } else {
        # Multi-predictor with RE
        L$add("    auto propagate_mp2 = [&](int k, double d) {")
        L$add("      for (int p = 0; p < P_crit2; ++p) grad_thresh_mid2_pop(p) += d * X_crit2_d(n, p);")
        L$add("      g_crit2_mid += d;")
        L$add("      if (k > mid2) {")
        L$add("        for (int j = 0; j <= k - mid2 - 1; ++j) {")
        L$add("          double g = d * exp_eff_gaps2(j);")
        L$add("          for (int p = 0; p < P_crit2; ++p) grad_log_gaps2_pop(j, p) += g * X_crit2_d(n, p);")
        L$add("          g_crit2_gaps(j) += g;")
        L$add("        }")
        L$add("      } else if (k < mid2) {")
        L$add("        for (int j = n_upper2; j <= n_upper2 + (mid2 - k - 1); ++j) {")
        L$add("          double g = d * (-exp_eff_gaps2(j));")
        L$add("          for (int p = 0; p < P_crit2; ++p) grad_log_gaps2_pop(j, p) += g * X_crit2_d(n, p);")
        L$add("          g_crit2_gaps(j) += g;")
        L$add("        }")
        L$add("      }")
        L$add("    };")
        L$add(sprintf("    if (vr.k2_lo >= 0) propagate_mp2(vr.k2_lo, %s);", d_lo2))
        L$add(sprintf("    if (vr.k2_hi >= 0) propagate_mp2(vr.k2_hi, %s);", d_hi2))
      }

      # Scatter criterion2 RE gradients
      emit_crit2_re_grad_scatter(L, model_data, params)

    } else {
      # No criterion2 RE
      if (is_crit2_intercept_only) {
        L$add("    if (vr.k2_lo >= 0) {")
        L$add(sprintf("      batch_sdt::propagate_thresh_grad(vr.k2_lo, %s, mid2, n_thresh2, n_upper2,", d_lo2))
        L$add("          pop_exp_gaps2, grad_thresh_mid2_pop, grad_log_gaps2_pop);")
        L$add("    }")
        L$add("    if (vr.k2_hi >= 0) {")
        L$add(sprintf("      batch_sdt::propagate_thresh_grad(vr.k2_hi, %s, mid2, n_thresh2, n_upper2,", d_hi2))
        L$add("          pop_exp_gaps2, grad_thresh_mid2_pop, grad_log_gaps2_pop);")
        L$add("    }")
      } else {
        # Multi-predictor, no RE
        L$add("    auto propagate_mp2 = [&](int k, double d) {")
        L$add("      for (int p = 0; p < P_crit2; ++p) grad_thresh_mid2_pop(p) += d * X_crit2_d(n, p);")
        L$add("      if (k > mid2) {")
        L$add("        for (int j = 0; j <= k - mid2 - 1; ++j) {")
        L$add("          double g = d * pop_exp_gaps2(j);")
        L$add("          for (int p = 0; p < P_crit2; ++p) grad_log_gaps2_pop(j, p) += g * X_crit2_d(n, p);")
        L$add("        }")
        L$add("      } else if (k < mid2) {")
        L$add("        for (int j = n_upper2; j <= n_upper2 + (mid2 - k - 1); ++j) {")
        L$add("          double g = d * (-pop_exp_gaps2(j));")
        L$add("          for (int p = 0; p < P_crit2; ++p) grad_log_gaps2_pop(j, p) += g * X_crit2_d(n, p);")
        L$add("        }")
        L$add("      }")
        L$add("    };")
        L$add(sprintf("    if (vr.k2_lo >= 0) propagate_mp2(vr.k2_lo, %s);", d_lo2))
        L$add(sprintf("    if (vr.k2_hi >= 0) propagate_mp2(vr.k2_hi, %s);", d_hi2))
      }
    }
  }
}


#' Emit criterion2 RE gradient scatter (from g_crit2_mid/g_crit2_gaps to grad_u_crit2)
#' @noRd
emit_crit2_re_grad_scatter <- function(L, model_data, params) {
  if (is.null(params$criterion2$random)) return()

  crit2_re_idx <- 0
  for (group in names(params$criterion2$random)) {
    crit2_re_idx <- crit2_re_idx + 1
    re <- params$criterion2$random[[group]]
    g_var <- sprintf("g_%s", group)
    n_re_terms <- if (!is.null(re$n_re_terms)) re$n_re_terms else 1

    if (n_re_terms == 1 && !isTRUE(re$use_z_matrix)) {
      L$add(sprintf("    grad_u_crit2_%d(%s, mid2) += g_crit2_mid;", crit2_re_idx, g_var))
      L$add("    for (int g = 0; g < n_gaps2; ++g) {")
      L$add(sprintf("      grad_u_crit2_%d(%s, gap_to_col2(g)) += g_crit2_gaps(g);", crit2_re_idx, g_var))
      L$add("    }")
    } else if (isTRUE(re$use_z_matrix)) {
      L$add(sprintf("    for (int j = 0; j < %d; ++j) grad_u_crit2_%d(%s, mid2 * %d + j) += g_crit2_mid * Z_crit2_%d_d(n, j);",
                     n_re_terms, crit2_re_idx, g_var, n_re_terms, crit2_re_idx))
      L$add("    for (int g = 0; g < n_gaps2; ++g) {")
      L$add("      int col = gap_to_col2(g);")
      L$add(sprintf("      for (int j = 0; j < %d; ++j) grad_u_crit2_%d(%s, col * %d + j) += g_crit2_gaps(g) * Z_crit2_%d_d(n, j);",
                     n_re_terms, crit2_re_idx, g_var, n_re_terms, crit2_re_idx))
      L$add("    }")
    } else {
      idx_var <- sprintf("idx_criterion2_%s", group)
      L$add(sprintf("    if (%s[n] > 0) {", idx_var))
      L$add(sprintf("      grad_u_crit2_%d(%s, mid2 * %d + %s[n] - 1) += g_crit2_mid;",
                     crit2_re_idx, g_var, n_re_terms, idx_var))
      L$add("      for (int g = 0; g < n_gaps2; ++g) {")
      L$add("        int col = gap_to_col2(g);")
      L$add(sprintf("        grad_u_crit2_%d(%s, col * %d + %s[n] - 1) += g_crit2_gaps(g);",
                     crit2_re_idx, g_var, n_re_terms, idx_var))
      L$add("      }")
      L$add("    }")
    }
  }
}


# ============================================================================
# Gradient accumulation
# ============================================================================

#' Emit gradient accumulation for all parameters
#' @noRd
emit_gradient_accumulation <- function(L, model_data, family_name, params, has_crit_re) {
  if (family_name %in% c("evsdt", "uvsdt", "cumulative")) {
    emit_gradient_accumulation_standard(L, model_data, family_name, params, has_crit_re)
  } else if (family_name == "dpsdt") {
    emit_gradient_accumulation_dpsdt(L, model_data, params, has_crit_re)
  } else if (family_name == "mixture") {
    emit_gradient_accumulation_mixture(L, model_data, params, has_crit_re)
  } else if (family_name == "source_mixture") {
    emit_gradient_accumulation_source_mixture(L, model_data, params, has_crit_re)
  }
}


#' Emit standard gradient accumulation (EVSDT, UVSDT, cumulative)
#' @noRd
emit_gradient_accumulation_standard <- function(L, model_data, family_name, params, has_crit_re) {
  has_sigma <- isTRUE(params$.has_sigma)
  has_counts <- params$.has_counts
  is_cumulative <- isTRUE(params$.is_cumulative)
  crit <- model_data$criterion
  is_crit_intercept_only <- isTRUE(crit$is_intercept_only) || crit$n_coef == 1

  # ---- d_mu from z-gradients ----
  L$add("    // Gradient: d_lp / d_mu")
  L$add("    double d_mu = 0.0;")
  L$add("    if (k_lo >= 0) d_mu -= res.d_z_lo;")
  L$add("    if (k_hi >= 0) d_mu -= res.d_z_hi;")

  if (has_sigma) {
    L$add("    d_mu /= s;")
  }
  if (has_counts) {
    L$add("    d_mu *= count_n;")
  }

  # ---- dprime gradient ----
  has_dp_fixed <- isTRUE(params$.has_dp_fixed)
  has_encoding_dp <- isTRUE(params$dprime$fixed$has_encoding)
  if (is_cumulative) {
    L$add("    double d_dprime_n = d_mu;")
  } else {
    L$add("    double d_dprime_n = d_mu * is_old_n;")
  }

  if (has_dp_fixed) {
    if (has_encoding_dp) {
      L$add("    if (is_old_n > 0) {")
      L$add("      for (int p = 0; p < P_dp; ++p) grad_beta_dp(p) += d_dprime_n * X_dp_d(n, p);")
      L$add("    }")
    } else {
      L$add("    for (int p = 0; p < P_dp; ++p) grad_beta_dp(p) += d_dprime_n * X_dp_d(n, p);")
    }
  }

  dp_re_idx <- 0
  if (!is.null(params$dprime$random)) {
    for (group in names(params$dprime$random)) {
      dp_re_idx <- dp_re_idx + 1
      re <- params$dprime$random[[group]]
      tag <- dp_re_tag(dp_re_idx)
      emit_re_gradient(L, tag, re, group, "d_dprime_n", "dprime", dp_re_idx, "dp")
    }
  }

  # ---- sigma gradient (UVSDT) ----
  if (has_sigma) {
    L$add("")
    L$add("    // Gradient: d_lp / d_sigma_n")
    L$add("    if (is_old_n > 0) {")
    L$add("      double d_sigma_val = 0.0;")
    L$add("      if (k_lo >= 0) d_sigma_val -= res.d_z_lo * (thresh_n(k_lo) - mu) / (s * s);")
    L$add("      if (k_hi >= 0) d_sigma_val -= res.d_z_hi * (thresh_n(k_hi) - mu) / (s * s);")
    if (has_counts) {
      L$add("      d_sigma_val *= count_n;")
    }
    L$add("      double d_sigma_n = d_sigma_val * sigma_val;")  # chain through exp

    L$add("      for (int p = 0; p < P_sig; ++p) grad_beta_sig(p) += d_sigma_n * X_sig_d(n, p);")

    sig_re_idx <- 0
    if (!is.null(params$sigma$random)) {
      for (group in names(params$sigma$random)) {
        sig_re_idx <- sig_re_idx + 1
        re <- params$sigma$random[[group]]
        tag <- sig_re_tag(sig_re_idx)
        emit_re_gradient(L, tag, re, group, "d_sigma_n", "sigma", sig_re_idx, "sig", indent = "      ")
      }
    }
    L$add("    }")
  }

  # ---- Threshold gradient ----
  emit_thresh_gradient_standard(L, model_data, params, has_crit_re)
}


#' Emit DPSDT gradient accumulation
#' @noRd
emit_gradient_accumulation_dpsdt <- function(L, model_data, params, has_crit_re) {
  has_sigma <- isTRUE(params$.has_sigma)
  has_counts <- params$.has_counts

  count_mult <- if (has_counts) " * count_n" else ""

  # ---- dprime gradient ----
  has_dp_fixed <- isTRUE(params$.has_dp_fixed)
  has_encoding_dp <- isTRUE(params$dprime$fixed$has_encoding)
  L$add(sprintf("    double d_dprime_n = d_mu_raw * is_old_n%s;", count_mult))

  if (has_dp_fixed) {
    if (has_encoding_dp) {
      L$add("    if (is_old_n > 0) {")
      L$add("      for (int p = 0; p < P_dp; ++p) grad_beta_dp(p) += d_dprime_n * X_dp_d(n, p);")
      L$add("    }")
    } else {
      L$add("    for (int p = 0; p < P_dp; ++p) grad_beta_dp(p) += d_dprime_n * X_dp_d(n, p);")
    }
  }

  dp_re_idx <- 0
  if (!is.null(params$dprime$random)) {
    for (group in names(params$dprime$random)) {
      dp_re_idx <- dp_re_idx + 1
      re <- params$dprime$random[[group]]
      tag <- dp_re_tag(dp_re_idx)
      emit_re_gradient(L, tag, re, group, "d_dprime_n", "dprime", dp_re_idx, "dp")
    }
  }

  # ---- sigma gradient ----
  if (has_sigma) {
    L$add("")
    L$add("    if (is_old_n > 0) {")
    L$add(sprintf("      double d_sigma_n = d_sigma_raw%s;", count_mult))
    L$add("      for (int p = 0; p < P_sig; ++p) grad_beta_sig(p) += d_sigma_n * X_sig_d(n, p);")
    sig_re_idx <- 0
    if (!is.null(params$sigma$random)) {
      for (group in names(params$sigma$random)) {
        sig_re_idx <- sig_re_idx + 1
        re <- params$sigma$random[[group]]
        tag <- sig_re_tag(sig_re_idx)
        emit_re_gradient(L, tag, re, group, "d_sigma_n", "sigma", sig_re_idx, "sig", indent = "      ")
      }
    }
    L$add("    }")
  }

  # ---- lambda gradient ----
  L$add("")
  L$add("    if (is_old_n > 0) {")
  L$add(sprintf("      double d_lambda_n = d_lambda_raw * lambda_val * (1.0 - lambda_val)%s;", count_mult))
  L$add("      for (int p = 0; p < P_lam; ++p) grad_beta_lam(p) += d_lambda_n * X_lam_d(n, p);")
  lam_re_idx <- 0
  if (!is.null(params$lambda$random)) {
    for (group in names(params$lambda$random)) {
      lam_re_idx <- lam_re_idx + 1
      re <- params$lambda$random[[group]]
      tag <- lam_re_tag(lam_re_idx)
      emit_re_gradient(L, tag, re, group, "d_lambda_n", "lambda", lam_re_idx, "lam", indent = "      ")
    }
  }
  L$add("    }")

  # ---- Threshold gradient ----
  emit_thresh_gradient_extended(L, model_data, params, has_crit_re)
}


#' Emit mixture gradient accumulation
#' @noRd
emit_gradient_accumulation_mixture <- function(L, model_data, params, has_crit_re) {
  has_sigma <- isTRUE(params$.has_sigma)
  has_sigma2 <- isTRUE(params$.has_sigma2)
  has_dprime2 <- isTRUE(params$.has_dprime2)
  has_counts <- params$.has_counts

  count_mult <- if (has_counts) " * count_n" else ""

  # ---- dprime (component 1) gradient ----
  has_dp_fixed <- isTRUE(params$.has_dp_fixed)
  has_encoding_dp <- isTRUE(params$dprime$fixed$has_encoding)
  L$add(sprintf("    double d_dprime_n = d_mu1_raw * is_old_n%s;", count_mult))

  if (has_dp_fixed) {
    if (has_encoding_dp) {
      L$add("    if (is_old_n > 0) {")
      L$add("      for (int p = 0; p < P_dp; ++p) grad_beta_dp(p) += d_dprime_n * X_dp_d(n, p);")
      L$add("    }")
    } else {
      L$add("    for (int p = 0; p < P_dp; ++p) grad_beta_dp(p) += d_dprime_n * X_dp_d(n, p);")
    }
  }

  dp_re_idx <- 0
  if (!is.null(params$dprime$random)) {
    for (group in names(params$dprime$random)) {
      dp_re_idx <- dp_re_idx + 1
      re <- params$dprime$random[[group]]
      tag <- dp_re_tag(dp_re_idx)
      emit_re_gradient(L, tag, re, group, "d_dprime_n", "dprime", dp_re_idx, "dp")
    }
  }

  # ---- sigma (component 1) gradient ----
  if (has_sigma) {
    L$add("")
    L$add("    if (is_old_n > 0) {")
    L$add(sprintf("      double d_sigma_n = d_sigma_raw%s;", count_mult))
    L$add("      for (int p = 0; p < P_sig; ++p) grad_beta_sig(p) += d_sigma_n * X_sig_d(n, p);")
    sig_re_idx <- 0
    if (!is.null(params$sigma$random)) {
      for (group in names(params$sigma$random)) {
        sig_re_idx <- sig_re_idx + 1
        re <- params$sigma$random[[group]]
        tag <- sig_re_tag(sig_re_idx)
        emit_re_gradient(L, tag, re, group, "d_sigma_n", "sigma", sig_re_idx, "sig", indent = "      ")
      }
    }
    L$add("    }")
  }

  # ---- lambda gradient (only when signal mixture is active) ----
  if (isTRUE(params$.has_lambda)) {
    L$add("")
    L$add("    if (is_old_n > 0) {")
    L$add(sprintf("      double d_lambda_n = d_lambda_raw * lambda_val * (1.0 - lambda_val)%s;", count_mult))
    L$add("      for (int p = 0; p < P_lam; ++p) grad_beta_lam(p) += d_lambda_n * X_lam_d(n, p);")
    lam_re_idx <- 0
    if (!is.null(params$lambda$random)) {
      for (group in names(params$lambda$random)) {
        lam_re_idx <- lam_re_idx + 1
        re <- params$lambda$random[[group]]
        tag <- lam_re_tag(lam_re_idx)
        emit_re_gradient(L, tag, re, group, "d_lambda_n", "lambda", lam_re_idx, "lam", indent = "      ")
      }
    }
    L$add("    }")
  }

  # ---- dprime2 (component 2) gradient ----
  if (has_dprime2) {
    L$add("")
    has_encoding_dp2 <- isTRUE(params$dprime2$fixed$has_encoding)
    L$add(sprintf("    double d_dprime2_n = d_mu2_raw * is_old_n%s;", count_mult))

    if (has_encoding_dp2) {
      L$add("    if (is_old_n > 0) {")
      L$add("      for (int p = 0; p < P_dp2; ++p) grad_beta_dp2(p) += d_dprime2_n * X_dp2_d(n, p);")
      L$add("    }")
    } else {
      L$add("    for (int p = 0; p < P_dp2; ++p) grad_beta_dp2(p) += d_dprime2_n * X_dp2_d(n, p);")
    }

    dp2_re_idx <- 0
    if (!is.null(params$dprime2$random)) {
      for (group in names(params$dprime2$random)) {
        dp2_re_idx <- dp2_re_idx + 1
        re <- params$dprime2$random[[group]]
        tag <- dp2_re_tag(dp2_re_idx)
        emit_re_gradient(L, tag, re, group, "d_dprime2_n", "dprime2", dp2_re_idx, "dp2")
      }
    }
  }

  # ---- sigma2 (component 2) gradient ----
  if (has_sigma2) {
    L$add("")
    L$add("    if (is_old_n > 0) {")
    L$add(sprintf("      double d_sigma2_n = d_sigma2_raw%s;", count_mult))
    L$add("      for (int p = 0; p < P_sig2; ++p) grad_beta_sig2(p) += d_sigma2_n * X_sig2_d(n, p);")
    sig2_re_idx <- 0
    if (!is.null(params$sigma2$random)) {
      for (group in names(params$sigma2$random)) {
        sig2_re_idx <- sig2_re_idx + 1
        re <- params$sigma2$random[[group]]
        tag <- sig2_re_tag(sig2_re_idx)
        emit_re_gradient(L, tag, re, group, "d_sigma2_n", "sigma2", sig2_re_idx, "sig2", indent = "      ")
      }
    }
    L$add("    }")
  }

  # ---- Lure mixture gradient (dprime_L, sigma_L, lambda_L) ----
  has_lure <- isTRUE(params$.has_lure_mixture)
  if (has_lure) {
    has_sigma_L <- isTRUE(params$.has_sigma_L)

    L$add("")
    L$add("    if (is_old_n <= 0) {")

    # dprime_L gradient
    L$add(sprintf("      double d_dprime_L_n = d_dprime_L_raw%s;", count_mult))
    L$add("      for (int p = 0; p < P_dpL; ++p) grad_beta_dpL(p) += d_dprime_L_n * X_dpL_d(n, p);")
    dpL_re_idx <- 0
    if (!is.null(params$dprime_L$random)) {
      for (group in names(params$dprime_L$random)) {
        dpL_re_idx <- dpL_re_idx + 1
        re <- params$dprime_L$random[[group]]
        tag <- dpL_re_tag(dpL_re_idx)
        emit_re_gradient(L, tag, re, group, "d_dprime_L_n", "dprime_L", dpL_re_idx, "dpL", indent = "      ")
      }
    }

    # sigma_L gradient
    if (has_sigma_L) {
      L$add(sprintf("      double d_sigma_L_n = d_sigma_L_raw%s;", count_mult))
      L$add("      for (int p = 0; p < P_sigL; ++p) grad_beta_sigL(p) += d_sigma_L_n * X_sigL_d(n, p);")
      sigL_re_idx <- 0
      if (!is.null(params$sigma_L$random)) {
        for (group in names(params$sigma_L$random)) {
          sigL_re_idx <- sigL_re_idx + 1
          re <- params$sigma_L$random[[group]]
          tag <- sigL_re_tag(sigL_re_idx)
          emit_re_gradient(L, tag, re, group, "d_sigma_L_n", "sigma_L", sigL_re_idx, "sigL", indent = "      ")
        }
      }
    }

    # lambda_L gradient (inv_logit chain rule)
    L$add(sprintf("      double d_lambda_L_n = d_lambda_L_raw * lambda_L_val * (1.0 - lambda_L_val)%s;", count_mult))
    L$add("      for (int p = 0; p < P_lamL; ++p) grad_beta_lamL(p) += d_lambda_L_n * X_lamL_d(n, p);")
    lamL_re_idx <- 0
    if (!is.null(params$lambda_L$random)) {
      for (group in names(params$lambda_L$random)) {
        lamL_re_idx <- lamL_re_idx + 1
        re <- params$lambda_L$random[[group]]
        tag <- lamL_re_tag(lamL_re_idx)
        emit_re_gradient(L, tag, re, group, "d_lambda_L_n", "lambda_L", lamL_re_idx, "lamL", indent = "      ")
      }
    }

    L$add("    }")
  }

  # ---- Threshold gradient ----
  emit_thresh_gradient_extended(L, model_data, params, has_crit_re)
}


#' Emit source mixture gradient accumulation
#' @noRd
emit_gradient_accumulation_source_mixture <- function(L, model_data, params, has_crit_re) {
  has_dprime_B <- isTRUE(params$.has_dprime_B)
  has_lambda_B <- isTRUE(params$.has_lambda_B)
  has_counts <- params$.has_counts
  count_mult <- if (has_counts) " * count_n" else ""

  # ---- dprime gradient (source A or symmetric) ----
  has_dp_fixed <- isTRUE(params$.has_dp_fixed)
  L$add(sprintf("    double d_dprime_n = d_mu_raw%s;", count_mult))
  if (has_dp_fixed) {
    L$add("    for (int p = 0; p < P_dp; ++p) grad_beta_dp(p) += d_dprime_n * X_dp_d(n, p);")
  }

  dp_re_idx <- 0
  if (!is.null(params$dprime$random)) {
    for (group in names(params$dprime$random)) {
      dp_re_idx <- dp_re_idx + 1
      re <- params$dprime$random[[group]]
      tag <- dp_re_tag(dp_re_idx)
      emit_re_gradient(L, tag, re, group, "d_dprime_n", "dprime", dp_re_idx, "dp")
    }
  }

  # ---- dprime_B gradient ----
  if (has_dprime_B) {
    L$add("")
    L$add(sprintf("    double d_dprime_B_n = d_mu_B_raw%s;", count_mult))
    L$add("    for (int p = 0; p < P_dpB; ++p) grad_beta_dpB(p) += d_dprime_B_n * X_dpB_d(n, p);")

    dpB_re_idx <- 0
    if (!is.null(params$dprime_B$random)) {
      for (group in names(params$dprime_B$random)) {
        dpB_re_idx <- dpB_re_idx + 1
        re <- params$dprime_B$random[[group]]
        tag <- dpB_re_tag(dpB_re_idx)
        emit_re_gradient(L, tag, re, group, "d_dprime_B_n", "dprime_B", dpB_re_idx, "dpB")
      }
    }
  }

  # ---- lambda gradient (inv_logit chain rule) ----
  L$add("")
  L$add(sprintf("    double d_lambda_n = d_lambda_raw * lambda_val * (1.0 - lambda_val)%s;", count_mult))
  L$add("    for (int p = 0; p < P_lam; ++p) grad_beta_lam(p) += d_lambda_n * X_lam_d(n, p);")
  lam_re_idx <- 0
  if (!is.null(params$lambda$random)) {
    for (group in names(params$lambda$random)) {
      lam_re_idx <- lam_re_idx + 1
      re <- params$lambda$random[[group]]
      tag <- lam_re_tag(lam_re_idx)
      emit_re_gradient(L, tag, re, group, "d_lambda_n", "lambda", lam_re_idx, "lam")
    }
  }

  # ---- lambda_B gradient ----
  if (has_lambda_B) {
    L$add("")
    L$add(sprintf("    double d_lambda_B_n = d_lambda_B_raw * lambda_B_val * (1.0 - lambda_B_val)%s;", count_mult))
    L$add("    for (int p = 0; p < P_lamB; ++p) grad_beta_lamB(p) += d_lambda_B_n * X_lamB_d(n, p);")
    lamB_re_idx <- 0
    if (!is.null(params$lambda_B$random)) {
      for (group in names(params$lambda_B$random)) {
        lamB_re_idx <- lamB_re_idx + 1
        re <- params$lambda_B$random[[group]]
        tag <- lamB_re_tag(lamB_re_idx)
        emit_re_gradient(L, tag, re, group, "d_lambda_B_n", "lambda_B", lamB_re_idx, "lamB")
      }
    }
  }

  # ---- Threshold gradient ----
  emit_thresh_gradient_extended(L, model_data, params, has_crit_re)
}


#' Emit standard threshold gradient (uses res.d_z_lo / res.d_z_hi)
#' @noRd
emit_thresh_gradient_standard <- function(L, model_data, params, has_crit_re) {
  has_sigma <- isTRUE(params$.has_sigma)
  has_counts <- params$.has_counts
  crit <- model_data$criterion
  is_crit_intercept_only <- isTRUE(crit$is_intercept_only) || crit$n_coef == 1

  L$add("")
  L$add("    // Gradient: d_lp / d_thresh")

  needs_scale <- has_sigma || has_counts
  if (has_sigma) {
    L$add("    double d_thresh_scale = is_old_n > 0 ? 1.0 / s : 1.0;")
    if (has_counts) {
      L$add("    d_thresh_scale *= count_n;")
    }
  } else if (has_counts) {
    L$add("    double d_thresh_scale = count_n;")
  }

  d_lo <- if (needs_scale) "res.d_z_lo * d_thresh_scale" else "res.d_z_lo"
  d_hi <- if (needs_scale) "res.d_z_hi * d_thresh_scale" else "res.d_z_hi"

  emit_thresh_propagation(L, model_data, params, has_crit_re, d_lo, d_hi, is_crit_intercept_only)
}


#' Emit threshold gradient for dpsdt/mixture (uses pre-computed d_thresh_lo/d_thresh_hi)
#' @noRd
emit_thresh_gradient_extended <- function(L, model_data, params, has_crit_re) {
  has_counts <- params$.has_counts
  crit <- model_data$criterion
  is_crit_intercept_only <- isTRUE(crit$is_intercept_only) || crit$n_coef == 1

  L$add("")
  L$add("    // Gradient: d_lp / d_thresh (using precomputed d_thresh_lo/hi)")

  if (has_counts) {
    d_lo <- "d_thresh_lo * count_n"
    d_hi <- "d_thresh_hi * count_n"
  } else {
    d_lo <- "d_thresh_lo"
    d_hi <- "d_thresh_hi"
  }

  emit_thresh_propagation(L, model_data, params, has_crit_re, d_lo, d_hi, is_crit_intercept_only)
}


#' Emit threshold gradient propagation (shared by standard and extended)
#' @noRd
emit_thresh_propagation <- function(L, model_data, params, has_crit_re, d_lo, d_hi, is_crit_intercept_only) {
  if (has_crit_re) {
    L$add("    double g_crit_mid = 0.0;")
    L$add("    Eigen::VectorXd g_crit_gaps = Eigen::VectorXd::Zero(n_gaps);")

    if (is_crit_intercept_only) {
      L$add("    if (k_lo >= 0) {")
      L$add(sprintf("      batch_sdt::propagate_thresh_grad_with_re(k_lo, %s, mid, n_thresh, n_upper,", d_lo))
      L$add("          exp_eff_gaps, grad_thresh_mid_pop, grad_log_gaps_pop, g_crit_mid, g_crit_gaps);")
      L$add("    }")
      L$add("    if (k_hi >= 0) {")
      L$add(sprintf("      batch_sdt::propagate_thresh_grad_with_re(k_hi, %s, mid, n_thresh, n_upper,", d_hi))
      L$add("          exp_eff_gaps, grad_thresh_mid_pop, grad_log_gaps_pop, g_crit_mid, g_crit_gaps);")
      L$add("    }")
    } else {
      emit_multi_predictor_thresh_grad_with_re(L, d_lo, d_hi, params)
    }

    # Criterion smooth gradient: extract per-obs contribution from g_crit_mid/g_crit_gaps
    # by saving pre-propagation values and computing the delta
    if (!is.null(params$.smooth_data[["criterion"]])) {
      # g_crit_mid and g_crit_gaps were accumulated by propagate_thresh_grad_with_re
      # which is called BEFORE this point. The per-obs delta is the change from this obs.
      # Since pre-obs values were saved, delta = current - saved.
      # Use the per-obs d_lo/d_hi directly with the correct chain rule.
      # The derivative of log-lik w.r.t. thresh_mid for this observation:
      # For EACH boundary (lo/hi): d flows through ALL thresholds between it and mid,
      # each contributing d * product_of_exp_gaps. The propagate function handles this.
      # Compute it locally:
      L$add("    double g_sm_mid = 0.0;")
      L$add("    Eigen::VectorXd g_sm_gaps = Eigen::VectorXd::Zero(n_gaps);")
      # Compute per-obs contributions using the same logic as propagate_thresh_grad_with_re
      # but into local variables only
      L$add("    {")
      L$add("      auto scatter_sm = [&](int k, double d) {")
      L$add("        g_sm_mid += d;")
      L$add("        if (k > mid) {")
      L$add("          for (int j = 0; j <= k - mid - 1; ++j) g_sm_gaps(j) += d * exp_eff_gaps(j);")
      L$add("        } else if (k < mid) {")
      L$add("          for (int j = n_upper; j <= n_upper + (mid - k - 1); ++j) g_sm_gaps(j) += d * (-exp_eff_gaps(j));")
      L$add("        }")
      L$add("      };")
      L$add(sprintf("      if (k_lo >= 0) scatter_sm(k_lo, %s);", d_lo))
      L$add(sprintf("      if (k_hi >= 0) scatter_sm(k_hi, %s);", d_hi))
      L$add("    }")
      emit_criterion_smooth_gradient(L, params$.smooth_data, "g_sm_mid", "0")
      L$add("    for (int gi = 0; gi < n_gaps; ++gi) {")
      emit_criterion_smooth_gradient(L, params$.smooth_data, "g_sm_gaps(gi)", "1 + gi")
      L$add("    }")
    }

    emit_crit_re_grad_scatter(L, model_data, params)

  } else {
    if (is_crit_intercept_only) {
      L$add("    if (k_lo >= 0) {")
      L$add(sprintf("      batch_sdt::propagate_thresh_grad(k_lo, %s, mid, n_thresh, n_upper,", d_lo))
      L$add("          pop_exp_gaps, grad_thresh_mid_pop, grad_log_gaps_pop);")
      L$add("    }")
      L$add("    if (k_hi >= 0) {")
      L$add(sprintf("      batch_sdt::propagate_thresh_grad(k_hi, %s, mid, n_thresh, n_upper,", d_hi))
      L$add("          pop_exp_gaps, grad_thresh_mid_pop, grad_log_gaps_pop);")
      L$add("    }")
    } else {
      emit_multi_predictor_thresh_grad_no_re(L, d_lo, d_hi, params)
    }
    # Criterion smooth gradient for no-RE path: compute per-obs derivatives
    if (!is.null(params$.smooth_data[["criterion"]])) {
      # Per-obs mid derivative = d_lo + d_hi (same as what flows to thresh_mid_pop)
      L$add(sprintf("    double g_sm_mid = (k_lo >= 0 ? %s : 0.0) + (k_hi >= 0 ? %s : 0.0);", d_lo, d_hi))
      emit_criterion_smooth_gradient(L, params$.smooth_data, "g_sm_mid", "0")
      # Per-obs gap derivatives: chain through exp(gap) and gap direction
      L$add("    for (int gi = 0; gi < n_gaps; ++gi) {")
      L$add("      double g_sm_gap = 0.0;")
      if (is_crit_intercept_only) {
        L$add(sprintf("      if (k_lo >= 0 && k_lo > mid && gi <= k_lo - mid - 1) g_sm_gap += %s * pop_exp_gaps(gi);", d_lo))
        L$add(sprintf("      if (k_hi >= 0 && k_hi > mid && gi <= k_hi - mid - 1) g_sm_gap += %s * pop_exp_gaps(gi);", d_hi))
        L$add(sprintf("      if (k_lo >= 0 && k_lo < mid && gi >= n_upper && gi <= n_upper + (mid - k_lo - 1)) g_sm_gap += %s * (-pop_exp_gaps(gi));", d_lo))
        L$add(sprintf("      if (k_hi >= 0 && k_hi < mid && gi >= n_upper && gi <= n_upper + (mid - k_hi - 1)) g_sm_gap += %s * (-pop_exp_gaps(gi));", d_hi))
      } else {
        L$add(sprintf("      if (k_lo >= 0 && k_lo > mid && gi <= k_lo - mid - 1) g_sm_gap += %s * obs_exp_gaps(gi);", d_lo))
        L$add(sprintf("      if (k_hi >= 0 && k_hi > mid && gi <= k_hi - mid - 1) g_sm_gap += %s * obs_exp_gaps(gi);", d_hi))
        L$add(sprintf("      if (k_lo >= 0 && k_lo < mid && gi >= n_upper && gi <= n_upper + (mid - k_lo - 1)) g_sm_gap += %s * (-obs_exp_gaps(gi));", d_lo))
        L$add(sprintf("      if (k_hi >= 0 && k_hi < mid && gi >= n_upper && gi <= n_upper + (mid - k_hi - 1)) g_sm_gap += %s * (-obs_exp_gaps(gi));", d_hi))
      }
      emit_criterion_smooth_gradient(L, params$.smooth_data, "g_sm_gap", "1 + gi")
      L$add("    }")
    }
  }
}


#' Emit RE gradient for a single group
#' @noRd
emit_re_gradient <- function(L, tag, re, group, d_var, param_name, re_idx, prefix, indent = "    ") {
  g_var <- sprintf("g_%s", group)

  if (isTRUE(re$use_z_matrix)) {
    if (re$dim == 1) {
      L$add(sprintf("%sgrad_%s(%s) += %s * Z_%s_d(n, 0);", indent, tag, g_var, d_var, tag))
    } else {
      L$add(sprintf("%sfor (int j = 0; j < %s_d.cols(); ++j) grad_%s(%s, j) += %s * Z_%s_d(n, j);",
                     indent, tag, tag, g_var, d_var, tag))
    }
  } else if (re$dim == 1) {
    if (!is.null(re$term_idx)) {
      L$add(sprintf("%sif (idx_%s_%s[n] > 0) grad_%s(%s) += %s;",
                     indent, param_name, group, tag, g_var, d_var))
    } else {
      L$add(sprintf("%sgrad_%s(%s) += %s;", indent, tag, g_var, d_var))
    }
  } else if (!is.null(re$term_idx)) {
    L$add(sprintf("%sif (idx_%s_%s[n] > 0) grad_%s(%s, idx_%s_%s[n] - 1) += %s;",
                   indent, param_name, group, tag, g_var, param_name, group, d_var))
  } else if (!is.null(re$varying_re_mode) && re$varying_re_mode %in% c("per_bin", "full")) {
    # Per-bin varying RE: gradient goes to bin yn-1
    L$add(sprintf("%sgrad_%s(%s, yn - 1) += %s;", indent, tag, g_var, d_var))
  }
}


#' Emit multi-predictor threshold gradient with criterion RE
#' @noRd
emit_multi_predictor_thresh_grad_with_re <- function(L, d_lo, d_hi, params = list()) {
  # For multi-predictor criterion with RE, inline the propagation logic
  # since per-predictor gradient accumulation is needed
  L$add("    // Multi-predictor threshold gradient propagation (with RE)")
  L$add("    auto propagate_mp = [&](int k, double d) {")
  L$add("      // Population: per-predictor")
  L$add("      for (int p = 0; p < P_crit; ++p) grad_thresh_mid_pop(p) += d * X_crit_d(n, p);")
  L$add("      g_crit_mid += d;")
  L$add("      if (k > mid) {")
  L$add("        for (int j = 0; j <= k - mid - 1; ++j) {")
  L$add("          double g = d * exp_eff_gaps(j);")
  L$add("          for (int p = 0; p < P_crit; ++p) grad_log_gaps_pop(j, p) += g * X_crit_d(n, p);")
  L$add("          g_crit_gaps(j) += g;")
  L$add("        }")
  L$add("      } else if (k < mid) {")
  L$add("        for (int j = n_upper; j <= n_upper + (mid - k - 1); ++j) {")
  L$add("          double g = d * (-exp_eff_gaps(j));")
  L$add("          for (int p = 0; p < P_crit; ++p) grad_log_gaps_pop(j, p) += g * X_crit_d(n, p);")
  L$add("          g_crit_gaps(j) += g;")
  L$add("        }")
  L$add("      }")
  L$add("    };")
  L$add(sprintf("    if (k_lo >= 0) propagate_mp(k_lo, %s);", d_lo))
  L$add(sprintf("    if (k_hi >= 0) propagate_mp(k_hi, %s);", d_hi))
  # NOTE: criterion smooth gradient is handled by emit_thresh_propagation's
  # per-obs scatter_sm block, NOT here (g_crit_mid/g_crit_gaps accumulate
  # across observations, which is wrong for per-obs smooth gradients)
}


#' Emit multi-predictor threshold gradient without criterion RE
#' @noRd
emit_multi_predictor_thresh_grad_no_re <- function(L, d_lo, d_hi, params = list()) {
  L$add("    // Multi-predictor threshold gradient propagation (no RE)")
  L$add("    // Uses obs_exp_gaps (per-obs gap values) instead of pop_exp_gaps")
  L$add("    auto propagate_mp = [&](int k, double d) {")
  L$add("      for (int p = 0; p < P_crit; ++p) grad_thresh_mid_pop(p) += d * X_crit_d(n, p);")
  L$add("      if (k > mid) {")
  L$add("        for (int j = 0; j <= k - mid - 1; ++j) {")
  L$add("          double g = d * obs_exp_gaps(j);")
  L$add("          for (int p = 0; p < P_crit; ++p) grad_log_gaps_pop(j, p) += g * X_crit_d(n, p);")
  L$add("        }")
  L$add("      } else if (k < mid) {")
  L$add("        for (int j = n_upper; j <= n_upper + (mid - k - 1); ++j) {")
  L$add("          double g = d * (-obs_exp_gaps(j));")
  L$add("          for (int p = 0; p < P_crit; ++p) grad_log_gaps_pop(j, p) += g * X_crit_d(n, p);")
  L$add("        }")
  L$add("      }")
  L$add("    };")
  L$add(sprintf("    if (k_lo >= 0) propagate_mp(k_lo, %s);", d_lo))
  L$add(sprintf("    if (k_hi >= 0) propagate_mp(k_hi, %s);", d_hi))
  # NOTE: criterion smooth gradient is handled by emit_thresh_propagation's
  # per-obs gradient block, NOT here
}


#' Emit criterion RE gradient scatter (from g_crit_mid/g_crit_gaps to grad_u_crit)
#' @noRd
emit_crit_re_grad_scatter <- function(L, model_data, params) {
  crit_re_idx <- 0
  if (is.null(params$criterion$random)) return()

  for (group in names(params$criterion$random)) {
    crit_re_idx <- crit_re_idx + 1
    re <- params$criterion$random[[group]]
    g_var <- sprintf("g_%s", group)
    n_re_terms <- if (!is.null(re$n_re_terms)) re$n_re_terms else 1

    if (n_re_terms == 1 && !isTRUE(re$use_z_matrix)) {
      # Simple: mid column gets g_crit_mid, gap columns get g_crit_gaps
      L$add(sprintf("    grad_u_crit%d(%s, mid) += g_crit_mid;", crit_re_idx, g_var))
      L$add("    for (int g = 0; g < n_gaps; ++g) {")
      L$add(sprintf("      grad_u_crit%d(%s, gap_to_col(g)) += g_crit_gaps(g);", crit_re_idx, g_var))
      L$add("    }")
    } else if (isTRUE(re$use_z_matrix)) {
      # Z-matrix criterion RE: gradient goes through Z weights
      # Mid threshold
      L$add(sprintf("    for (int j = 0; j < %d; ++j) grad_u_crit%d(%s, mid * %d + j) += g_crit_mid * Z_crit%d_d(n, j);",
                     n_re_terms, crit_re_idx, g_var, n_re_terms, crit_re_idx))
      # Gap thresholds
      L$add("    for (int g = 0; g < n_gaps; ++g) {")
      L$add("      int col = gap_to_col(g);")
      L$add(sprintf("      for (int j = 0; j < %d; ++j) grad_u_crit%d(%s, col * %d + j) += g_crit_gaps(g) * Z_crit%d_d(n, j);",
                     n_re_terms, crit_re_idx, g_var, n_re_terms, crit_re_idx))
      L$add("    }")
    } else {
      # Multi-term idx: mid and gap columns use idx_criterion
      idx_var <- sprintf("idx_criterion_%s", group)
      L$add(sprintf("    if (%s[n] > 0) {", idx_var))
      L$add(sprintf("      grad_u_crit%d(%s, mid * %d + %s[n] - 1) += g_crit_mid;",
                     crit_re_idx, g_var, n_re_terms, idx_var))
      L$add("      for (int g = 0; g < n_gaps; ++g) {")
      L$add("        int col = gap_to_col(g);")
      L$add(sprintf("        grad_u_crit%d(%s, col * %d + %s[n] - 1) += g_crit_gaps(g);",
                     crit_re_idx, g_var, n_re_terms, idx_var))
      L$add("      }")
      L$add("    }")
    }
  }
}


# ============================================================================
# Operand packing (precomputed_gradients)
# ============================================================================

#' Emit operand/gradient packing for precomputed_gradients
#' @noRd
emit_operand_packing <- function(L, model_data, family_name, params, has_crit_re, has_crit2_re = FALSE) {
  crit <- model_data$criterion
  is_crit_intercept_only <- isTRUE(crit$is_intercept_only) || crit$n_coef == 1
  has_dp_fixed <- isTRUE(params$.has_dp_fixed)
  has_disc_fixed_cpp <- isTRUE(params$.has_disc_fixed)

  L$add("  if constexpr (is_autodiff) {")
  L$add("    std::vector<var> operands;")
  L$add("    std::vector<double> gradients;")
  L$add("")

  # beta_dprime
  has_dp_fixed <- isTRUE(params$.has_dp_fixed)
  if (has_dp_fixed) {
    emit_pack_vec(L, "beta_dprime", "grad_beta_dp")
  }

  # dprime REs
  dp_re_idx <- 0
  if (!is.null(params$dprime$random)) {
    for (group in names(params$dprime$random)) {
      dp_re_idx <- dp_re_idx + 1
      re <- params$dprime$random[[group]]
      stan_name <- re_var_name("dprime", group, re$cor_id)
      tag <- dp_re_tag(dp_re_idx)
      if (re$dim == 1) {
        emit_pack_vec(L, stan_name, sprintf("grad_%s", tag))
      } else {
        emit_pack_mat(L, stan_name, sprintf("grad_%s", tag))
      }
    }
  }

  # sigma
  if (isTRUE(params$.has_sigma)) {
    emit_pack_vec(L, "beta_sigma", "grad_beta_sig")
    sig_re_idx <- 0
    if (!is.null(params$sigma$random)) {
      for (group in names(params$sigma$random)) {
        sig_re_idx <- sig_re_idx + 1
        re <- params$sigma$random[[group]]
        stan_name <- re_var_name("sigma", group, re$cor_id)
        tag <- sig_re_tag(sig_re_idx)
        if (re$dim == 1) {
          emit_pack_vec(L, stan_name, sprintf("grad_%s", tag))
        } else {
          emit_pack_mat(L, stan_name, sprintf("grad_%s", tag))
        }
      }
    }
  }

  # lambda
  if (isTRUE(params$.has_lambda)) {
    emit_pack_vec(L, "beta_lambda", "grad_beta_lam")
    lam_re_idx <- 0
    if (!is.null(params$lambda$random)) {
      for (group in names(params$lambda$random)) {
        lam_re_idx <- lam_re_idx + 1
        re <- params$lambda$random[[group]]
        stan_name <- re_var_name("lambda", group, re$cor_id)
        tag <- lam_re_tag(lam_re_idx)
        if (re$dim == 1) {
          emit_pack_vec(L, stan_name, sprintf("grad_%s", tag))
        } else {
          emit_pack_mat(L, stan_name, sprintf("grad_%s", tag))
        }
      }
    }
  }

  # dprime2
  if (isTRUE(params$.has_dprime2)) {
    emit_pack_vec(L, "beta_dprime2", "grad_beta_dp2")
    dp2_re_idx <- 0
    if (!is.null(params$dprime2$random)) {
      for (group in names(params$dprime2$random)) {
        dp2_re_idx <- dp2_re_idx + 1
        re <- params$dprime2$random[[group]]
        stan_name <- re_var_name("dprime2", group, re$cor_id)
        tag <- dp2_re_tag(dp2_re_idx)
        if (re$dim == 1) {
          emit_pack_vec(L, stan_name, sprintf("grad_%s", tag))
        } else {
          emit_pack_mat(L, stan_name, sprintf("grad_%s", tag))
        }
      }
    }
  }

  # sigma2
  if (isTRUE(params$.has_sigma2)) {
    emit_pack_vec(L, "beta_sigma2", "grad_beta_sig2")
    sig2_re_idx <- 0
    if (!is.null(params$sigma2$random)) {
      for (group in names(params$sigma2$random)) {
        sig2_re_idx <- sig2_re_idx + 1
        re <- params$sigma2$random[[group]]
        stan_name <- re_var_name("sigma2", group, re$cor_id)
        tag <- sig2_re_tag(sig2_re_idx)
        if (re$dim == 1) {
          emit_pack_vec(L, stan_name, sprintf("grad_%s", tag))
        } else {
          emit_pack_mat(L, stan_name, sprintf("grad_%s", tag))
        }
      }
    }
  }

  # dprime_B (source_mixture)
  if (isTRUE(params$.has_dprime_B)) {
    emit_pack_vec(L, "beta_dprime_B", "grad_beta_dpB")
    dpB_re_idx <- 0
    if (!is.null(params$dprime_B$random)) {
      for (group in names(params$dprime_B$random)) {
        dpB_re_idx <- dpB_re_idx + 1
        re <- params$dprime_B$random[[group]]
        stan_name <- re_var_name("dprime_B", group, re$cor_id)
        tag <- dpB_re_tag(dpB_re_idx)
        if (re$dim == 1) {
          emit_pack_vec(L, stan_name, sprintf("grad_%s", tag))
        } else {
          emit_pack_mat(L, stan_name, sprintf("grad_%s", tag))
        }
      }
    }
  }

  # lambda_B (source_mixture)
  if (isTRUE(params$.has_lambda_B)) {
    emit_pack_vec(L, "beta_lambda_B", "grad_beta_lamB")
    lamB_re_idx <- 0
    if (!is.null(params$lambda_B$random)) {
      for (group in names(params$lambda_B$random)) {
        lamB_re_idx <- lamB_re_idx + 1
        re <- params$lambda_B$random[[group]]
        stan_name <- re_var_name("lambda_B", group, re$cor_id)
        tag <- lamB_re_tag(lamB_re_idx)
        if (re$dim == 1) {
          emit_pack_vec(L, stan_name, sprintf("grad_%s", tag))
        } else {
          emit_pack_mat(L, stan_name, sprintf("grad_%s", tag))
        }
      }
    }
  }

  # dprime_L (lure mixture)
  if (isTRUE(params$.has_dprime_L)) {
    emit_pack_vec(L, "beta_dprime_L", "grad_beta_dpL")
    dpL_re_idx <- 0
    if (!is.null(params$dprime_L$random)) {
      for (group in names(params$dprime_L$random)) {
        dpL_re_idx <- dpL_re_idx + 1
        re <- params$dprime_L$random[[group]]
        stan_name <- re_var_name("dprime_L", group, re$cor_id)
        tag <- dpL_re_tag(dpL_re_idx)
        if (re$dim == 1) {
          emit_pack_vec(L, stan_name, sprintf("grad_%s", tag))
        } else {
          emit_pack_mat(L, stan_name, sprintf("grad_%s", tag))
        }
      }
    }
  }

  # sigma_L (lure mixture)
  if (isTRUE(params$.has_sigma_L)) {
    emit_pack_vec(L, "beta_sigma_L", "grad_beta_sigL")
    sigL_re_idx <- 0
    if (!is.null(params$sigma_L$random)) {
      for (group in names(params$sigma_L$random)) {
        sigL_re_idx <- sigL_re_idx + 1
        re <- params$sigma_L$random[[group]]
        stan_name <- re_var_name("sigma_L", group, re$cor_id)
        tag <- sigL_re_tag(sigL_re_idx)
        if (re$dim == 1) {
          emit_pack_vec(L, stan_name, sprintf("grad_%s", tag))
        } else {
          emit_pack_mat(L, stan_name, sprintf("grad_%s", tag))
        }
      }
    }
  }

  # lambda_L (lure mixture)
  if (isTRUE(params$.has_lambda_L)) {
    emit_pack_vec(L, "beta_lambda_L", "grad_beta_lamL")
    lamL_re_idx <- 0
    if (!is.null(params$lambda_L$random)) {
      for (group in names(params$lambda_L$random)) {
        lamL_re_idx <- lamL_re_idx + 1
        re <- params$lambda_L$random[[group]]
        stan_name <- re_var_name("lambda_L", group, re$cor_id)
        tag <- lamL_re_tag(lamL_re_idx)
        if (re$dim == 1) {
          emit_pack_vec(L, stan_name, sprintf("grad_%s", tag))
        } else {
          emit_pack_mat(L, stan_name, sprintf("grad_%s", tag))
        }
      }
    }
  }

  # discrim (vrdp2d / bivariate)
  if (isTRUE(params$.has_discrim)) {
    if (has_disc_fixed_cpp) {
      emit_pack_vec(L, "beta_discrim", "grad_beta_disc")
    }
    disc_re_idx <- 0
    if (!is.null(params$discrim$random)) {
      for (group in names(params$discrim$random)) {
        disc_re_idx <- disc_re_idx + 1
        re <- params$discrim$random[[group]]
        stan_name <- re_var_name("discrim", group, re$cor_id)
        tag <- disc_re_tag(disc_re_idx)
        if (re$dim == 1) {
          emit_pack_vec(L, stan_name, sprintf("grad_%s", tag))
        } else {
          emit_pack_mat(L, stan_name, sprintf("grad_%s", tag))
        }
      }
    }
  }

  # discrim_B (vrdp2d)
  if (isTRUE(params$.has_discrim_B)) {
    emit_pack_vec(L, "beta_discrim_B", "grad_beta_discB")
    discB_re_idx <- 0
    if (!is.null(params$discrim_B$random)) {
      for (group in names(params$discrim_B$random)) {
        discB_re_idx <- discB_re_idx + 1
        re <- params$discrim_B$random[[group]]
        stan_name <- re_var_name("discrim_B", group, re$cor_id)
        tag <- discB_re_tag(discB_re_idx)
        if (re$dim == 1) {
          emit_pack_vec(L, stan_name, sprintf("grad_%s", tag))
        } else {
          emit_pack_mat(L, stan_name, sprintf("grad_%s", tag))
        }
      }
    }
  }

  # Bivariate-specific operand packing
  emit_generic_operand_packing(L, params, "sigma_B", "sigB", "has_sigma_B")
  emit_generic_operand_packing(L, params, "sigma2_B", "sig2B", "has_sigma2_B")
  emit_generic_operand_packing(L, params, "rho", "rho", "has_rho")
  emit_generic_operand_packing(L, params, "rho_B", "rhoB", "has_rho_B")
  emit_generic_operand_packing(L, params, "rho_N", "rhoN", "has_rho_N")
  emit_generic_operand_packing(L, params, "lambda2", "lam2", "has_lambda2")
  emit_generic_operand_packing(L, params, "lambda2_B", "lam2B", "has_lambda2_B")

  # rec_crit (cdp)
  if (isTRUE(params$.has_rec_crit)) {
    emit_pack_vec(L, "beta_rec_crit", "grad_beta_rc")
    rc_re_idx <- 0
    if (!is.null(params$rec_crit$random)) {
      for (group in names(params$rec_crit$random)) {
        rc_re_idx <- rc_re_idx + 1
        re <- params$rec_crit$random[[group]]
        stan_name <- re_var_name("rec_crit", group, re$cor_id)
        tag <- rc_re_tag(rc_re_idx)
        if (re$dim == 1) {
          emit_pack_vec(L, stan_name, sprintf("grad_%s", tag))
        } else {
          emit_pack_mat(L, stan_name, sprintf("grad_%s", tag))
        }
      }
    }
  }

  # know_crit (cdp)
  if (isTRUE(params$.has_know_crit)) {
    emit_pack_vec(L, "beta_know_crit", "grad_beta_kc")
    kc_re_idx <- 0
    if (!is.null(params$know_crit$random)) {
      for (group in names(params$know_crit$random)) {
        kc_re_idx <- kc_re_idx + 1
        re <- params$know_crit$random[[group]]
        stan_name <- re_var_name("know_crit", group, re$cor_id)
        tag <- kc_re_tag(kc_re_idx)
        if (re$dim == 1) {
          emit_pack_vec(L, stan_name, sprintf("grad_%s", tag))
        } else {
          emit_pack_mat(L, stan_name, sprintf("grad_%s", tag))
        }
      }
    }
  }

  # Criterion population: thresh_mid
  if (is_crit_intercept_only) {
    L$add("    operands.push_back(beta_thresh_mid.coeff(0));")
    L$add("    gradients.push_back(grad_thresh_mid_pop);")
  } else {
    emit_pack_vec(L, "beta_thresh_mid", "grad_thresh_mid_pop")
  }

  has_two_responses <- isTRUE(params$.is_vrdp2d) || isTRUE(params$.is_bivariate)
  # Criterion population: log_gaps
  K_for_crit <- if (has_two_responses) model_data$stan_data$K1
                else if (isTRUE(params$.is_cdp)) model_data$stan_data$J + 1
                else model_data$K
  if (K_for_crit > 2) {
    if (is_crit_intercept_only) {
      L$add("    for (int g = 0; g < n_gaps; ++g) {")
      L$add("      operands.push_back(beta_log_gaps.coeff(g, 0));")
      L$add("      gradients.push_back(grad_log_gaps_pop(g));")
      L$add("    }")
    } else {
      emit_pack_mat(L, "beta_log_gaps", "grad_log_gaps_pop")
    }
  }

  # Criterion REs
  crit_re_idx <- 0
  if (!is.null(params$criterion$random)) {
    for (group in names(params$criterion$random)) {
      crit_re_idx <- crit_re_idx + 1
      stan_name <- re_var_name("criterion", group, params$criterion$random[[group]]$cor_id)
      emit_pack_mat(L, stan_name, sprintf("grad_u_crit%d", crit_re_idx))
    }
  }

  # Criterion2 population (vrdp2d/bivariate -- source dimension)
  if (has_two_responses) {
    varying_sc <- isTRUE(params$.varying_source_criteria)
    K2_val <- model_data$stan_data$K2

    if (varying_sc) {
      # Varying source criteria: pack per-bin threshold gradients
      emit_pack_vec(L, "beta_thresh_mid_2_varying", "grad_vary_mid2")
      if (K2_val > 2) {
        emit_pack_mat(L, "beta_log_gaps_2_varying", "grad_vary_gaps2")
      }
      if (identical(params$.new_source_criteria, "shared")) {
        # Push the autodiff var itself, NOT value_of(...): wrapping it strips the
        # var to a constant so operands_and_partials cannot attach grad_new_mid2,
        # silently dropping the beta_thresh_mid_2_new gradient.
        L$add("    operands.push_back(beta_thresh_mid_2_new);")
        L$add("    gradients.push_back(grad_new_mid2);")
        if (K2_val > 2) {
          emit_pack_vec(L, "beta_log_gaps_2_new", "grad_new_gaps2")
        }
      }
      # Criterion2 REs
      crit2_re_idx <- 0
      if (!is.null(params$criterion2$random)) {
        for (group in names(params$criterion2$random)) {
          crit2_re_idx <- crit2_re_idx + 1
          stan_name <- re_var_name("criterion2", group, params$criterion2$random[[group]]$cor_id)
          emit_pack_mat(L, stan_name, sprintf("grad_u_crit2_%d", crit2_re_idx))
        }
      }
    } else {
      crit2 <- model_data$criterion2
      is_crit2_intercept_only <- is.null(crit2) || isTRUE(crit2$is_intercept_only) || crit2$n_coef == 1

      if (is_crit2_intercept_only) {
        L$add("    operands.push_back(beta_thresh_mid_2.coeff(0));")
        L$add("    gradients.push_back(grad_thresh_mid2_pop);")
      } else {
        emit_pack_vec(L, "beta_thresh_mid_2", "grad_thresh_mid2_pop")
      }

      if (K2_val > 2) {
        if (is_crit2_intercept_only) {
          L$add("    for (int g = 0; g < n_gaps2; ++g) {")
          L$add("      operands.push_back(beta_log_gaps_2.coeff(g, 0));")
          L$add("      gradients.push_back(grad_log_gaps2_pop(g));")
          L$add("    }")
        } else {
          emit_pack_mat(L, "beta_log_gaps_2", "grad_log_gaps2_pop")
        }
      }

      # Criterion2 REs
      crit2_re_idx <- 0
      if (!is.null(params$criterion2$random)) {
        for (group in names(params$criterion2$random)) {
          crit2_re_idx <- crit2_re_idx + 1
          stan_name <- re_var_name("criterion2", group, params$criterion2$random[[group]]$cor_id)
          emit_pack_mat(L, stan_name, sprintf("grad_u_crit2_%d", crit2_re_idx))
        }
      }
    }
  }

  # Smooth operand packing
  if (!is.null(params$.smooth_data)) {
    for (pname in names(params$.smooth_data)) {
      for (sm in params$.smooth_data[[pname]]) {
        n_thresh_sm <- if (!is.null(sm$n_thresh)) sm$n_thresh else 0L
        for (comp in sm$components) {
          for (k in seq_along(comp$Zs_list)) {
            s_name <- paste0("s_", pname, "_", comp$san_label, "_", k)
            if (n_thresh_sm > 0) {
              emit_pack_mat(L, s_name, paste0("grad_", s_name))
            } else {
              emit_pack_vec(L, s_name, paste0("grad_", s_name))
            }
          }
        }
      }
    }
  }

  L$add("")
  L$add("    // NaN guard: return -inf with no gradient operands for clean rejection")
  L$add("    if (!std::isfinite(total_lp)) {")
  L$add("      return stan::math::var(-std::numeric_limits<double>::infinity());")
  L$add("    }")
  L$add("    return stan::math::precomputed_gradients(total_lp, operands, gradients);")
  L$add("  } else {")
  L$add("    return total_lp;")
  L$add("  }")
}


#' Emit operand packing for a generic parameter (fixed + REs)
#' @noRd
emit_generic_operand_packing <- function(L, params, full_name, short, flag_name) {
  if (!isTRUE(params[[paste0(".", flag_name)]])) return()
  tag_fn <- get(paste0(short, "_re_tag"))

  emit_pack_vec(L, sprintf("beta_%s", full_name), sprintf("grad_beta_%s", short))
  re_idx <- 0
  param_data <- params[[full_name]]
  if (!is.null(param_data$random)) {
    for (group in names(param_data$random)) {
      re_idx <- re_idx + 1
      re <- param_data$random[[group]]
      stan_name <- re_var_name(full_name, group, re$cor_id)
      tag <- tag_fn(re_idx)
      if (re$dim == 1) {
        emit_pack_vec(L, stan_name, sprintf("grad_%s", tag))
      } else {
        emit_pack_mat(L, stan_name, sprintf("grad_%s", tag))
      }
    }
  }
}


#' Emit operand packing for an Eigen vector parameter
#' @noRd
emit_pack_vec <- function(L, param_name, grad_name) {
  L$add(sprintf("    for (int i = 0; i < %s.size(); ++i) {", param_name))
  L$add(sprintf("      operands.push_back(%s.coeff(i));", param_name))
  L$add(sprintf("      gradients.push_back(%s(i));", grad_name))
  L$add("    }")
}


#' Emit operand packing for an Eigen matrix parameter (column-major)
#' @noRd
emit_pack_mat <- function(L, param_name, grad_name) {
  L$add(sprintf("    for (int c = 0; c < %s.cols(); ++c) {", param_name))
  L$add(sprintf("      for (int r = 0; r < %s.rows(); ++r) {", param_name))
  L$add(sprintf("        operands.push_back(%s.coeff(r, c));", param_name))
  L$add(sprintf("        gradients.push_back(%s(r, c));", grad_name))
  L$add("      }")
  L$add("    }")
}


# ============================================================================
# Tag helpers -- short internal names for RE groups
# ============================================================================

#' @noRd
dp_re_tag <- function(idx) sprintf("u_dp%d", idx)

#' @noRd
sig_re_tag <- function(idx) sprintf("u_sig%d", idx)

#' @noRd
lam_re_tag <- function(idx) sprintf("u_lam%d", idx)

#' @noRd
dp2_re_tag <- function(idx) sprintf("u_dp2_%d", idx)

#' @noRd
sig2_re_tag <- function(idx) sprintf("u_sig2_%d", idx)

#' @noRd
dpB_re_tag <- function(idx) sprintf("u_dpB%d", idx)

#' @noRd
lamB_re_tag <- function(idx) sprintf("u_lamB%d", idx)

#' @noRd
dpL_re_tag <- function(idx) sprintf("u_dpL%d", idx)

#' @noRd
sigL_re_tag <- function(idx) sprintf("u_sigL%d", idx)

#' @noRd
lamL_re_tag <- function(idx) sprintf("u_lamL%d", idx)

#' @noRd
disc_re_tag <- function(idx) sprintf("u_disc%d", idx)

#' @noRd
discB_re_tag <- function(idx) sprintf("u_discB%d", idx)

#' @noRd
rc_re_tag <- function(idx) sprintf("u_rc%d", idx)

#' @noRd
kc_re_tag <- function(idx) sprintf("u_kc%d", idx)

#' Map internal tag to the Stan argument name used in the signature
#' @noRd
re_stan_arg_name <- function(param_name, group, re, re_idx, prefix) {
  re_var_name(param_name, group, re$cor_id)
}

#' @noRd
sigB_re_tag <- function(idx) sprintf("u_sigB%d", idx)

#' @noRd
sig2B_re_tag <- function(idx) sprintf("u_sig2B%d", idx)

#' @noRd
rho_re_tag <- function(idx) sprintf("u_rho%d", idx)

#' @noRd
rhoB_re_tag <- function(idx) sprintf("u_rhoB%d", idx)

#' @noRd
rhoN_re_tag <- function(idx) sprintf("u_rhoN%d", idx)

#' @noRd
lam2_re_tag <- function(idx) sprintf("u_lam2_%d", idx)

#' @noRd
lam2B_re_tag <- function(idx) sprintf("u_lam2B_%d", idx)


# ============================================================================
# CDP: Likelihood and gradient accumulation
# ============================================================================

#' Emit CDP likelihood computation
#' @noRd
emit_likelihood_cdp <- function(L, params) {
  is_cdp <- isTRUE(params$.is_cdp)
  has_sigma <- isTRUE(params$.has_sigma)
  has_sigma2 <- isTRUE(params$.has_sigma2)
  has_dprime2 <- isTRUE(params$.has_dprime2)
  has_know_crit <- isTRUE(params$.has_know_crit)
  has_counts <- params$.has_counts
  n_rkg <- params$.n_rkg

  # Resolve parameter values
  sigma_R_expr <- if (has_sigma) "sigma_val" else "1.0"
  sigma_F_expr <- if (has_sigma2) "sigma2_val" else "1.0"
  dprime2_expr <- if (has_dprime2) "dprime2_n" else "0.0"

  L$add("    // CDP likelihood")
  L$add(sprintf("    double mu_R_n = dprime_n;"))
  L$add(sprintf("    double sigma_R_n = %s;", sigma_R_expr))
  L$add(sprintf("    double mu_F_n = %s;", dprime2_expr))
  L$add(sprintf("    double sigma_F_n = %s;", sigma_F_expr))

  if (isTRUE(params$.is_cdp)) {
    # rec_crit value
    L$add("    double c_R_val = rec_crit_val;")

    if (isTRUE(n_rkg == 3) && has_know_crit) {
      L$add("    double c_K_val = know_crit_val;")
      L$add("")
      L$add("    int is_old_int = (is_old_n > 0.5) ? 1 : 0;")
      L$add("    auto cdp_res = batch_sdt::cdp_rkg_cell(")
      L$add("        yn, rk_n, is_old_int,")
      L$add("        mu_R_n, sigma_R_n, mu_F_n, sigma_F_n,")
      L$add("        c_R_val, c_K_val,")
      L$add("        thresh_n.data(), n_thresh,")
      L$add("        old_level_map.data());")
    } else {
      L$add("")
      L$add("    int is_old_int = (is_old_n > 0.5) ? 1 : 0;")
      L$add("    auto cdp_res = batch_sdt::cdp_rk_cell(")
      L$add("        yn, rk_n, is_old_int,")
      L$add("        mu_R_n, sigma_R_n, mu_F_n, sigma_F_n,")
      L$add("        c_R_val,")
      L$add("        thresh_n.data(), n_thresh,")
      L$add("        old_level_map.data());")
    }
  }

  L$add("")
  L$add("    double obs_lp = cdp_res.lp;")
  L$add("    int k_lo = cdp_res.k_lo;")
  L$add("    int k_hi = cdp_res.k_hi;")
  L$add("    double d_thresh_lo = cdp_res.d_thresh_lo;")
  L$add("    double d_thresh_hi = cdp_res.d_thresh_hi;")
  if (has_counts) {
    L$add("    total_lp += obs_lp * count_n;")
  } else {
    L$add("    total_lp += obs_lp;")
  }
}


#' Emit CDP gradient accumulation
#' @noRd
emit_gradient_accumulation_cdp <- function(L, model_data, params, has_crit_re) {
  has_sigma <- isTRUE(params$.has_sigma)
  has_sigma2 <- isTRUE(params$.has_sigma2)
  has_dprime2 <- isTRUE(params$.has_dprime2)
  has_know_crit <- isTRUE(params$.has_know_crit)
  has_rec_crit <- isTRUE(params$.has_rec_crit)
  has_counts <- params$.has_counts
  count_mult <- if (has_counts) " * count_n" else ""

  # ---- dprime gradient (recollection mean mu_R) ----
  has_dp_fixed <- isTRUE(params$.has_dp_fixed)
  L$add("")
  L$add(sprintf("    double d_dprime_n = cdp_res.d_dprime%s;", count_mult))
  if (has_dp_fixed) {
    L$add("    for (int p = 0; p < P_dp; ++p) grad_beta_dp(p) += d_dprime_n * X_dp_d(n, p);")
  }
  dp_re_idx <- 0
  if (!is.null(params$dprime$random)) {
    for (group in names(params$dprime$random)) {
      dp_re_idx <- dp_re_idx + 1
      re <- params$dprime$random[[group]]
      tag <- dp_re_tag(dp_re_idx)
      emit_re_gradient(L, tag, re, group, "d_dprime_n", "dprime", dp_re_idx, "dp")
    }
  }

  # ---- dprime2 gradient (familiarity mean mu_F) ----
  if (has_dprime2) {
    L$add("")
    L$add(sprintf("    double d_dprime2_n = cdp_res.d_dprime2%s;", count_mult))
    L$add("    for (int p = 0; p < P_dp2; ++p) grad_beta_dp2(p) += d_dprime2_n * X_dp2_d(n, p);")
    dp2_re_idx <- 0
    if (!is.null(params$dprime2$random)) {
      for (group in names(params$dprime2$random)) {
        dp2_re_idx <- dp2_re_idx + 1
        re <- params$dprime2$random[[group]]
        tag <- dp2_re_tag(dp2_re_idx)
        emit_re_gradient(L, tag, re, group, "d_dprime2_n", "dprime2", dp2_re_idx, "dp2")
      }
    }
  }

  # ---- sigma gradient (sigma_R, log-link) ----
  if (has_sigma) {
    L$add("")
    L$add(sprintf("    double d_sigma_n = cdp_res.d_sigma%s;", count_mult))
    L$add("    for (int p = 0; p < P_sig; ++p) grad_beta_sig(p) += d_sigma_n * X_sig_d(n, p);")
    sig_re_idx <- 0
    if (!is.null(params$sigma$random)) {
      for (group in names(params$sigma$random)) {
        sig_re_idx <- sig_re_idx + 1
        re <- params$sigma$random[[group]]
        tag <- sig_re_tag(sig_re_idx)
        emit_re_gradient(L, tag, re, group, "d_sigma_n", "sigma", sig_re_idx, "sig")
      }
    }
  }

  # ---- sigma2 gradient (sigma_F, log-link) ----
  if (has_sigma2) {
    L$add("")
    L$add(sprintf("    double d_sigma2_n = cdp_res.d_sigma2%s;", count_mult))
    L$add("    for (int p = 0; p < P_sig2; ++p) grad_beta_sig2(p) += d_sigma2_n * X_sig2_d(n, p);")
    sig2_re_idx <- 0
    if (!is.null(params$sigma2$random)) {
      for (group in names(params$sigma2$random)) {
        sig2_re_idx <- sig2_re_idx + 1
        re <- params$sigma2$random[[group]]
        tag <- sig2_re_tag(sig2_re_idx)
        emit_re_gradient(L, tag, re, group, "d_sigma2_n", "sigma2", sig2_re_idx, "sig2")
      }
    }
  }

  # ---- rec_crit gradient ----
  if (has_rec_crit) {
    L$add("")
    L$add(sprintf("    double d_rec_crit_n = cdp_res.d_rec_crit%s;", count_mult))
    L$add("    for (int p = 0; p < P_rc; ++p) grad_beta_rc(p) += d_rec_crit_n * X_rc_d(n, p);")
    rc_re_idx <- 0
    if (!is.null(params$rec_crit$random)) {
      for (group in names(params$rec_crit$random)) {
        rc_re_idx <- rc_re_idx + 1
        re <- params$rec_crit$random[[group]]
        tag <- rc_re_tag(rc_re_idx)
        emit_re_gradient(L, tag, re, group, "d_rec_crit_n", "rec_crit", rc_re_idx, "rc")
      }
    }
  }

  # ---- know_crit gradient ----
  if (has_know_crit) {
    L$add("")
    L$add(sprintf("    double d_know_crit_n = cdp_res.d_know_crit%s;", count_mult))
    L$add("    for (int p = 0; p < P_kc; ++p) grad_beta_kc(p) += d_know_crit_n * X_kc_d(n, p);")
    kc_re_idx <- 0
    if (!is.null(params$know_crit$random)) {
      for (group in names(params$know_crit$random)) {
        kc_re_idx <- kc_re_idx + 1
        re <- params$know_crit$random[[group]]
        tag <- kc_re_tag(kc_re_idx)
        emit_re_gradient(L, tag, re, group, "d_know_crit_n", "know_crit", kc_re_idx, "kc")
      }
    }
  }

  # ---- Threshold gradient ----
  # Uses precomputed d_thresh_lo/d_thresh_hi from CDP cell function
  emit_thresh_gradient_extended(L, model_data, params, has_crit_re)
}


# ============================================================================
# Generic parameter helpers for bivariate
# ============================================================================

#' Emit value_of extraction for a parameter (fixed + REs)
#' @noRd
emit_param_value_extraction <- function(L, param_name, short, random, full_name) {
  tag_fn <- get(paste0(short, "_re_tag"))
  L$add(sprintf("  const Eigen::MatrixXd X_%s_d = value_of(X_%s);", short, full_name))
  L$add(sprintf("  const Eigen::VectorXd beta_%s_d = value_of(beta_%s);", short, full_name))
  L$add(sprintf("  const int P_%s = beta_%s_d.size();", short, short))
  re_idx <- 0
  if (!is.null(random)) {
    for (group in names(random)) {
      re_idx <- re_idx + 1
      re <- random[[group]]
      tag <- tag_fn(re_idx)
      if (re$dim == 1) {
        L$add(sprintf("  const Eigen::VectorXd %s_d = value_of(%s);",
                       tag, re_var_name(full_name, group, re$cor_id)))
      } else {
        L$add(sprintf("  const Eigen::MatrixXd %s_d = value_of(%s);",
                       tag, re_var_name(full_name, group, re$cor_id)))
      }
      if (isTRUE(re$use_z_matrix)) {
        L$add(sprintf("  const Eigen::MatrixXd Z_%s_d = value_of(Z_%s_%s);",
                       tag, full_name, group))
      }
    }
  }
}

#' Emit gradient accumulator for a parameter (fixed + REs)
#' @noRd
emit_param_grad_accum <- function(L, param_name, short, random, full_name) {
  tag_fn <- get(paste0(short, "_re_tag"))
  L$add(sprintf("  Eigen::VectorXd grad_beta_%s = Eigen::VectorXd::Zero(P_%s);", short, short))
  re_idx <- 0
  if (!is.null(random)) {
    for (group in names(random)) {
      re_idx <- re_idx + 1
      re <- random[[group]]
      tag <- tag_fn(re_idx)
      if (re$dim == 1) {
        L$add(sprintf("  Eigen::VectorXd grad_%s = Eigen::VectorXd::Zero(%s_d.size());", tag, tag))
      } else {
        L$add(sprintf("  Eigen::MatrixXd grad_%s = Eigen::MatrixXd::Zero(%s_d.rows(), %s_d.cols());", tag, tag, tag))
      }
    }
  }
}


# ============================================================================
# Bivariate SDT: Likelihood computation
# ============================================================================

#' Emit bivariate SDT/DP likelihood
#' @noRd
emit_likelihood_bivariate <- function(L, params) {
  has_sigma <- isTRUE(params$.has_sigma)
  has_sigma_B <- isTRUE(params$.has_sigma_B)
  has_sigma2 <- isTRUE(params$.has_sigma2)
  has_sigma2_B <- isTRUE(params$.has_sigma2_B)
  has_rho <- isTRUE(params$.has_rho)
  has_rho_B <- isTRUE(params$.has_rho_B)
  has_rho_N <- isTRUE(params$.has_rho_N)
  has_dprime_B <- isTRUE(params$.has_dprime_B)
  has_discrim_B <- isTRUE(params$.has_discrim_B)
  is_dp <- isTRUE(params$.is_bivariate_dp)
  bounded <- isTRUE(params$.bounded)
  has_counts <- params$.has_counts

  L$add("    // Bivariate SDT likelihood: resolve per-source parameters")

  # bivariate_dp always uses exp link on dprime/discrim (positivity constraint);

  # bounded mode uses log link on dprime/discrim and logistic on rho. Under
  # the new convention (A on negative source axis, B on positive), bounded
  # places the sign by negating A's parameters in the call to bivariate_cell.
  needs_exp <- bounded

  # Resolve dprime for this source
  if (has_dprime_B) {
    if (needs_exp) {
      L$add("    double dprime_B_val = std::exp(dprime_B_n);  // exp link")
    } else {
      L$add("    double dprime_B_val = dprime_B_n;")
    }
  } else {
    if (needs_exp) {
      L$add("    double dprime_B_val = std::exp(dprime_n);  // exp link (equal detection)")
    } else {
      L$add("    double dprime_B_val = dprime_n;  // equal detection")
    }
  }

  # Resolve discrim_B_val.
  # Symmetric default depends on bounded:
  #   bounded:   discrim_B_val = exp(discrim_n) -- same positive magnitude
  #   unbounded: discrim_B_val = -discrim_n     -- opposite sign for mirror
  if (has_discrim_B) {
    if (needs_exp) {
      L$add("    double discrim_B_val = std::exp(discrim_B_n);  // exp link")
    } else {
      L$add("    double discrim_B_val = discrim_B_n;")
    }
  } else {
    if (needs_exp) {
      L$add("    double discrim_B_val = std::exp(discrim_n);  // bounded symmetric: same magnitude")
    } else {
      L$add("    double discrim_B_val = -discrim_n;  // unbounded symmetric: mirror across origin")
    }
  }

  # Apply exp() to dprime_n and discrim_n for positivity
  if (needs_exp) {
    L$add("    double dprime_bounded = std::exp(dprime_n);")
    L$add("    double discrim_bounded = std::exp(discrim_n);")
  }

  # Resolve sigma parameters
  sigma1_A <- if (has_sigma) "sigma_val" else "1.0"
  sigma1_B <- if (has_sigma_B) "sigma_B_val" else sigma1_A
  sigma2_A <- if (has_sigma2) "sigma2_val" else "1.0"
  sigma2_B <- if (has_sigma2_B) "sigma2_B_val" else sigma2_A

  # Resolve rho parameters (rho_val is tanh for unbounded, inv_logit for bounded).
  # Symmetric default for rho_B_expr depends on bounded:
  #   bounded:   rho_B_expr = rho_val     -- same magnitude (sign placed in call)
  #   unbounded: rho_B_expr = -rho_val    -- opposite sign for mirror
  rho_A <- if (has_rho) "rho_val" else "0.0"
  if (has_rho_B) {
    rho_B_expr <- "rho_B_val"
  } else if (needs_exp) {
    rho_B_expr <- rho_A
  } else {
    rho_B_expr <- if (has_rho) "(-rho_val)" else "0.0"
  }
  rho_N_expr <- if (has_rho_N) "rho_N_val" else "0.0"

  # Per-source mu2/rho_eff under new convention:
  #   bounded:   mu2_A = -discrim, rho_eff_A = -rho;   mu2_B = discrim_B, rho_eff_B = rho_B
  #   unbounded: mu2_A =  discrim, rho_eff_A =  rho;   mu2_B = discrim_B, rho_eff_B = rho_B
  L$add("")
  L$add("    // Select parameters based on item_type (1=new, 2=A, 3=B)")
  L$add("    double mu1, mu2, s1, s2, rho_eff;")
  L$add("    if (item_type_n == 1) {")
  L$add(sprintf("      mu1 = 0.0; mu2 = 0.0; s1 = 1.0; s2 = 1.0; rho_eff = %s;", rho_N_expr))
  L$add("    } else if (item_type_n == 2) {")
  if (needs_exp) {
    L$add(sprintf("      mu1 = dprime_bounded; mu2 = -discrim_bounded; s1 = %s; s2 = %s; rho_eff = -%s;", sigma1_A, sigma2_A, rho_A))
  } else {
    L$add(sprintf("      mu1 = dprime_n; mu2 = discrim_n; s1 = %s; s2 = %s; rho_eff = %s;", sigma1_A, sigma2_A, rho_A))
  }
  L$add("    } else {")
  L$add(sprintf("      mu1 = dprime_B_val; mu2 = discrim_B_val; s1 = %s; s2 = %s; rho_eff = %s;",
                 sigma1_B, sigma2_B, rho_B_expr))
  L$add("    }")
  L$add("")

  if (is_dp && bounded) {
    L$add("    auto bv = batch_sdt::bounded_bivariate_dp_cell(")
    L$add("        yn, yn2, item_type_n, K1, K2,")
    L$add("        mu1, mu2, s1, s2, rho_eff,")
    L$add("        R_I_val, R_S_val, R_I_B_val, R_S_B_val,")
    L$add("        thresh1_n.data(), n_thresh1,")
    L$add("        thresh2_n.data(), n_thresh2);")
  } else if (is_dp) {
    L$add("    auto bv = batch_sdt::bivariate_dp_cell(")
    L$add("        yn, yn2, item_type_n, K1, K2,")
    L$add("        mu1, mu2, s1, s2, rho_eff,")
    L$add("        R_I_val, R_S_val, R_I_B_val, R_S_B_val,")
    L$add("        thresh1_n.data(), n_thresh1,")
    L$add("        thresh2_n.data(), n_thresh2);")
  } else if (bounded) {
    L$add("    auto bv = batch_sdt::bounded_bivariate_sdt_cell(")
    L$add("        yn, yn2, K1, K2,")
    L$add("        mu1, mu2, s1, s2, rho_eff,")
    L$add("        thresh1_n.data(), n_thresh1,")
    L$add("        thresh2_n.data(), n_thresh2);")
  } else {
    L$add("    auto bv = batch_sdt::bivariate_sdt_cell(")
    L$add("        yn, yn2, K1, K2,")
    L$add("        mu1, mu2, s1, s2, rho_eff,")
    L$add("        thresh1_n.data(), n_thresh1,")
    L$add("        thresh2_n.data(), n_thresh2);")
  }
  L$add("")

  L$add("    double obs_lp = bv.lp;")

  if (has_counts) {
    L$add("    total_lp += obs_lp * count_n;")
  } else {
    L$add("    total_lp += obs_lp;")
  }
}


# ============================================================================
# Bivariate SDT: Gradient accumulation
# ============================================================================

#' Emit bivariate gradient accumulation
#' @noRd
emit_gradient_accumulation_bivariate <- function(L, model_data, params, has_crit_re, has_crit2_re) {
  has_sigma <- isTRUE(params$.has_sigma)
  has_sigma_B <- isTRUE(params$.has_sigma_B)
  has_sigma2 <- isTRUE(params$.has_sigma2)
  has_sigma2_B <- isTRUE(params$.has_sigma2_B)
  has_rho <- isTRUE(params$.has_rho)
  has_rho_B <- isTRUE(params$.has_rho_B)
  has_rho_N <- isTRUE(params$.has_rho_N)
  has_dprime_B <- isTRUE(params$.has_dprime_B)
  has_discrim_B <- isTRUE(params$.has_discrim_B)
  has_dp_fixed <- !is.null(params$dprime$fixed) && params$dprime$fixed$n_coef > 0
  has_disc_fixed <- isTRUE(params$.has_discrim) && !is.null(params$discrim$fixed) && params$discrim$fixed$n_coef > 0
  is_dp <- isTRUE(params$.is_bivariate_dp)
  bounded <- isTRUE(params$.bounded)
  has_counts <- params$.has_counts
  count_mult <- if (has_counts) " * count_n" else ""

  # For bivariate_dp or bounded bivariate_sdt, gradients need exp/inv_logit chain rules:
  # dprime_linear -> exp(dprime_linear) = mu1, so d/d_linear = d/d_mu1 * mu1
  # discrim_linear -> exp(discrim_linear) = mu2, so d/d_linear = d/d_mu2 * mu2
  # rho_linear -> inv_logit(rho_linear) = rho, so d/d_linear = d/d_rho * rho*(1-rho)
  # These chain rule factors are applied as multipliers to bv.d_dprime, bv.d_discrim, bv.d_rho
  needs_exp <- bounded
  bounded_dp_mult <- if (needs_exp) " * dprime_bounded" else ""
  bounded_disc_mult <- if (needs_exp) " * discrim_bounded" else ""
  # Sign of d_logP/d_discrim_n on the A side: +1 in unbounded (mu2_A = discrim_n),
  # -1 in bounded (mu2_A = -exp(discrim_n)). Multiplied into chain rule.
  sign_A_disc <- if (bounded) "-" else ""
  sign_A_rho  <- if (bounded) "-" else ""

  # ---- dprime gradient ----
  # For source A: d_logP/d_dprime_n = bv.d_dprime (which is d_logP/d_mu1)
  # For source B (if shared dprime): d_logP/d_dprime_n = bv.d_dprime (mu1 = dprime_n)
  # For new items: bv.d_dprime = 0 (mu1=0, no dprime dependence)
  L$add("")
  if (has_dprime_B) {
    L$add("    // dprime gradient (only from source A, new items contribute 0)")
    L$add(sprintf("    double d_dprime_n = (item_type_n == 2) ? bv.d_dprime%s%s : 0.0;", bounded_dp_mult, count_mult))
  } else {
    L$add("    // dprime gradient (from source A and B, new items contribute 0)")
    L$add(sprintf("    double d_dprime_n = (item_type_n != 1) ? bv.d_dprime%s%s : 0.0;", bounded_dp_mult, count_mult))
  }
  has_dp_fixed <- isTRUE(params$.has_dp_fixed)
  if (has_dp_fixed) {
    L$add("    for (int p = 0; p < P_dp; ++p) grad_beta_dp(p) += d_dprime_n * X_dp_d(n, p);")
  }
  dp_re_idx <- 0
  if (!is.null(params$dprime$random)) {
    for (group in names(params$dprime$random)) {
      dp_re_idx <- dp_re_idx + 1
      re <- params$dprime$random[[group]]
      tag <- dp_re_tag(dp_re_idx)
      emit_re_gradient(L, tag, re, group, "d_dprime_n", "dprime", dp_re_idx, "dp")
    }
  }

  # ---- dprime_B gradient ----
  if (has_dprime_B) {
    L$add("")
    bounded_dpB_mult <- if (needs_exp) " * dprime_B_val" else ""
    L$add(sprintf("    double d_dprime_B_n = (item_type_n == 3) ? bv.d_dprime%s%s : 0.0;", bounded_dpB_mult, count_mult))
    L$add("    for (int p = 0; p < P_dpB; ++p) grad_beta_dpB(p) += d_dprime_B_n * X_dpB_d(n, p);")
    dpB_re_idx <- 0
    if (!is.null(params$dprime_B$random)) {
      for (group in names(params$dprime_B$random)) {
        dpB_re_idx <- dpB_re_idx + 1
        re <- params$dprime_B$random[[group]]
        tag <- dpB_re_tag(dpB_re_idx)
        emit_re_gradient(L, tag, re, group, "d_dprime_B_n", "dprime_B", dpB_re_idx, "dpB")
      }
    }
  }

  # ---- discrim gradient (NEW convention: A on negative source axis) ----
  # bv.d_discrim = d_logP/d_mu2.
  # - bounded:   mu2_A = -exp(discrim_n) -> d/d(discrim_n) = -exp(discrim_n) = -discrim_bounded
  #              mu2_B = +exp(discrim_B_n) -> d/d(discrim_B_n) = +discrim_B_val
  # - unbounded: mu2_A = discrim_n -> d/d(discrim_n) = +1
  #              mu2_B = discrim_B_n -> d/d(discrim_B_n) = +1
  # - unbounded symmetric (no discrim_B): mu2_B = -discrim_n -> d/d(discrim_n) for B = -1
  L$add("")
  if (has_discrim_B) {
    L$add("    if (item_type_n == 2) {")
    L$add(sprintf("      double d_discrim_n = %sbv.d_discrim%s%s;", sign_A_disc, bounded_disc_mult, count_mult))
    if (has_disc_fixed) {
      L$add("      for (int p = 0; p < P_disc; ++p) grad_beta_disc(p) += d_discrim_n * X_disc_d(n, p);")
    }
    disc_re_idx <- 0
    if (!is.null(params$discrim$random)) {
      for (group in names(params$discrim$random)) {
        disc_re_idx <- disc_re_idx + 1
        re <- params$discrim$random[[group]]
        tag <- disc_re_tag(disc_re_idx)
        emit_re_gradient(L, tag, re, group, "d_discrim_n", "discrim", disc_re_idx, "disc", indent = "      ")
      }
    }
    L$add("    } else if (item_type_n == 3) {")
    # B side never negates under new convention (chain factor +1 in both modes)
    bounded_discB_mult <- if (needs_exp) " * discrim_B_val" else ""
    L$add(sprintf("      double d_discrim_B_n = bv.d_discrim%s%s;", bounded_discB_mult, count_mult))
    L$add("      for (int p = 0; p < P_discB; ++p) grad_beta_discB(p) += d_discrim_B_n * X_discB_d(n, p);")
    discB_re_idx <- 0
    if (!is.null(params$discrim_B$random)) {
      for (group in names(params$discrim_B$random)) {
        discB_re_idx <- discB_re_idx + 1
        re <- params$discrim_B$random[[group]]
        tag <- discB_re_tag(discB_re_idx)
        emit_re_gradient(L, tag, re, group, "d_discrim_B_n", "discrim_B", discB_re_idx, "discB", indent = "      ")
      }
    }
    L$add("    }")
  } else {
    # Symmetric:
    #   bounded:   A: -bv.d_discrim * discrim_bounded ; B: +bv.d_discrim * discrim_bounded
    #   unbounded: A: +bv.d_discrim                   ; B: -bv.d_discrim   (B's mu2 = -discrim_n)
    L$add("    {")
    if (needs_exp) {
      L$add(sprintf("      double d_discrim_n = (item_type_n == 2 ? -bv.d_discrim : bv.d_discrim) * discrim_bounded%s;", count_mult))
    } else {
      L$add(sprintf("      double d_discrim_n = (item_type_n == 2 ? bv.d_discrim : -bv.d_discrim)%s;", count_mult))
    }
    L$add("      if (item_type_n != 1) {")
    if (has_disc_fixed) {
      L$add("        for (int p = 0; p < P_disc; ++p) grad_beta_disc(p) += d_discrim_n * X_disc_d(n, p);")
    }
    disc_re_idx <- 0
    if (!is.null(params$discrim$random)) {
      for (group in names(params$discrim$random)) {
        disc_re_idx <- disc_re_idx + 1
        re <- params$discrim$random[[group]]
        tag <- disc_re_tag(disc_re_idx)
        emit_re_gradient(L, tag, re, group, "d_discrim_n", "discrim", disc_re_idx, "disc", indent = "        ")
      }
    }
    L$add("      }")
    L$add("    }")
  }

  # ---- sigma gradient (detection SD for source A) ----
  if (has_sigma) {
    # bv.d_sigma1 is d_logP/d_sigma1. Chain rule for exp: d/d_sigma_n = d_sigma1 * sigma_val
    # Only contributes for source A (item_type==2), or also B if sigma_B not separate
    L$add("")
    if (has_sigma_B) {
      L$add(sprintf("    double d_sigma_n = (item_type_n == 2) ? bv.d_sigma1 * sigma_val%s : 0.0;", count_mult))
    } else {
      L$add(sprintf("    double d_sigma_n = (item_type_n != 1) ? bv.d_sigma1 * sigma_val%s : 0.0;", count_mult))
    }
    L$add("    for (int p = 0; p < P_sig; ++p) grad_beta_sig(p) += d_sigma_n * X_sig_d(n, p);")
    sig_re_idx <- 0
    if (!is.null(params$sigma$random)) {
      for (group in names(params$sigma$random)) {
        sig_re_idx <- sig_re_idx + 1
        re <- params$sigma$random[[group]]
        tag <- sig_re_tag(sig_re_idx)
        emit_re_gradient(L, tag, re, group, "d_sigma_n", "sigma", sig_re_idx, "sig")
      }
    }
  }

  # ---- sigma_B gradient (detection SD for source B) ----
  if (has_sigma_B) {
    L$add("")
    L$add(sprintf("    double d_sigma_B_n = (item_type_n == 3) ? bv.d_sigma1 * sigma_B_val%s : 0.0;", count_mult))
    L$add("    for (int p = 0; p < P_sigB; ++p) grad_beta_sigB(p) += d_sigma_B_n * X_sigB_d(n, p);")
    sigB_re_idx <- 0
    if (!is.null(params$sigma_B$random)) {
      for (group in names(params$sigma_B$random)) {
        sigB_re_idx <- sigB_re_idx + 1
        re <- params$sigma_B$random[[group]]
        tag <- sigB_re_tag(sigB_re_idx)
        emit_re_gradient(L, tag, re, group, "d_sigma_B_n", "sigma_B", sigB_re_idx, "sigB")
      }
    }
  }

  # ---- sigma2 gradient (discrimination SD for source A) ----
  if (has_sigma2) {
    L$add("")
    if (has_sigma2_B) {
      L$add(sprintf("    double d_sigma2_n = (item_type_n == 2) ? bv.d_sigma2 * sigma2_val%s : 0.0;", count_mult))
    } else {
      L$add(sprintf("    double d_sigma2_n = (item_type_n != 1) ? bv.d_sigma2 * sigma2_val%s : 0.0;", count_mult))
    }
    L$add("    for (int p = 0; p < P_sig2; ++p) grad_beta_sig2(p) += d_sigma2_n * X_sig2_d(n, p);")
    sig2_re_idx <- 0
    if (!is.null(params$sigma2$random)) {
      for (group in names(params$sigma2$random)) {
        sig2_re_idx <- sig2_re_idx + 1
        re <- params$sigma2$random[[group]]
        tag <- sig2_re_tag(sig2_re_idx)
        emit_re_gradient(L, tag, re, group, "d_sigma2_n", "sigma2", sig2_re_idx, "sig2")
      }
    }
  }

  # ---- sigma2_B gradient (discrimination SD for source B) ----
  if (has_sigma2_B) {
    L$add("")
    L$add(sprintf("    double d_sigma2_B_n = (item_type_n == 3) ? bv.d_sigma2 * sigma2_B_val%s : 0.0;", count_mult))
    L$add("    for (int p = 0; p < P_sig2B; ++p) grad_beta_sig2B(p) += d_sigma2_B_n * X_sig2B_d(n, p);")
    sig2B_re_idx <- 0
    if (!is.null(params$sigma2_B$random)) {
      for (group in names(params$sigma2_B$random)) {
        sig2B_re_idx <- sig2B_re_idx + 1
        re <- params$sigma2_B$random[[group]]
        tag <- sig2B_re_tag(sig2B_re_idx)
        emit_re_gradient(L, tag, re, group, "d_sigma2_B_n", "sigma2_B", sig2B_re_idx, "sig2B")
      }
    }
  }

  # ---- rho gradient (NEW convention) ----
  # bv.d_rho = d_logP/d_rho_eff. Link chain factor:
  #   bounded (logistic): d(rho_val)/d(rho_n) = rho_val * (1 - rho_val)
  #   unbounded (tanh):   d(rho_val)/d(rho_n) = 1 - rho_val^2
  # mu sign chain (rho_eff = sign * rho_val):
  #   bounded   A: rho_eff = -rho_val -> -1
  #   unbounded A: rho_eff =  rho_val -> +1
  #   B side, asymmetric: rho_eff = +rho_B_val -> +1
  #   B side, symmetric (no rho_B): bounded uses rho_B_expr = +rho_val (sign from A's
  #     negation gives mirror), unbounded uses rho_B_expr = -rho_val (mirror in code)
  if (has_rho) {
    L$add("")
    if (bounded) {
      rho_chain <- "rho_val * (1.0 - rho_val)"
    } else {
      rho_chain <- "(1.0 - rho_val * rho_val)"
    }
    if (has_rho_B) {
      # Only A contributes via rho parameter
      L$add(sprintf("    double d_rho_n = (item_type_n == 2) ? %sbv.d_rho * %s%s : 0.0;",
                     sign_A_rho, rho_chain, count_mult))
    } else {
      # Symmetric: A gets sign_A_rho * chain; B's contribution depends on bounded:
      #   bounded:   B uses rho_B_expr = +rho_val -> chain factor +1 (no negation)
      #   unbounded: B uses rho_B_expr = -rho_val -> chain factor -1
      sign_B_rho_sym <- if (bounded) "" else "-"
      L$add("    double d_rho_n = 0.0;")
      L$add(sprintf("    if (item_type_n == 2) d_rho_n = %sbv.d_rho * %s%s;",
                     sign_A_rho, rho_chain, count_mult))
      L$add(sprintf("    else if (item_type_n == 3) d_rho_n = %sbv.d_rho * %s%s;",
                     sign_B_rho_sym, rho_chain, count_mult))
    }
    L$add("    for (int p = 0; p < P_rho; ++p) grad_beta_rho(p) += d_rho_n * X_rho_d(n, p);")
    rho_re_idx <- 0
    if (!is.null(params$rho$random)) {
      for (group in names(params$rho$random)) {
        rho_re_idx <- rho_re_idx + 1
        re <- params$rho$random[[group]]
        tag <- rho_re_tag(rho_re_idx)
        emit_re_gradient(L, tag, re, group, "d_rho_n", "rho", rho_re_idx, "rho")
      }
    }
  }

  # ---- rho_B gradient (NEW convention) ----
  # rho_eff for B = +rho_B_val (no negation in either bounded or unbounded), so
  # chain factor on the B side is +1 in both modes.
  if (has_rho_B) {
    L$add("")
    if (bounded) {
      rho_B_chain <- "rho_B_val * (1.0 - rho_B_val)"  # rho_B_val = inv_logit(rho_B_n)
    } else {
      rho_B_chain <- "(1.0 - rho_B_val * rho_B_val)"
    }
    L$add(sprintf("    double d_rho_B_n = (item_type_n == 3) ? bv.d_rho * %s%s : 0.0;", rho_B_chain, count_mult))
    L$add("    for (int p = 0; p < P_rhoB; ++p) grad_beta_rhoB(p) += d_rho_B_n * X_rhoB_d(n, p);")
    rhoB_re_idx <- 0
    if (!is.null(params$rho_B$random)) {
      for (group in names(params$rho_B$random)) {
        rhoB_re_idx <- rhoB_re_idx + 1
        re <- params$rho_B$random[[group]]
        tag <- rhoB_re_tag(rhoB_re_idx)
        emit_re_gradient(L, tag, re, group, "d_rho_B_n", "rho_B", rhoB_re_idx, "rhoB")
      }
    }
  }

  # ---- rho_N gradient ----
  if (has_rho_N) {
    L$add("")
    L$add(sprintf("    double d_rho_N_n = (item_type_n == 1) ? bv.d_rho * (1.0 - rho_N_val * rho_N_val)%s : 0.0;", count_mult))
    L$add("    for (int p = 0; p < P_rhoN; ++p) grad_beta_rhoN(p) += d_rho_N_n * X_rhoN_d(n, p);")
    rhoN_re_idx <- 0
    if (!is.null(params$rho_N$random)) {
      for (group in names(params$rho_N$random)) {
        rhoN_re_idx <- rhoN_re_idx + 1
        re <- params$rho_N$random[[group]]
        tag <- rhoN_re_tag(rhoN_re_idx)
        emit_re_gradient(L, tag, re, group, "d_rho_N_n", "rho_N", rhoN_re_idx, "rhoN")
      }
    }
  }

  # ---- lambda gradient (bivariate_dp: R_I -- Source A R_I) ----
  # bv.d_lambda is already routed by item_type inside bivariate_dp_cell:
  # nonzero only for Source-A items. The chain rule factor d(R_I)/d(lambda_n)
  # = R_I_val * (1 - R_I_val) (inv_logit derivative).
  if (is_dp && isTRUE(params$.has_lambda)) {
    L$add("")
    if (!isTRUE(params$.has_lambda_B_dp)) {
      # lambda_B constrained equal to lambda: Source-B recollection (bv.d_lambda_B,
      # nonzero only for item_type == 3) shares the lambda linear predictor, so it
      # must fold into the same accumulator. Dropping it halves the gradient.
      L$add(sprintf("    double d_lambda_n = (bv.d_lambda + bv.d_lambda_B) * R_I_val * (1.0 - R_I_val)%s;", count_mult))
    } else {
      L$add(sprintf("    double d_lambda_n = bv.d_lambda * R_I_val * (1.0 - R_I_val)%s;", count_mult))
    }
    L$add("    for (int p = 0; p < P_lam; ++p) grad_beta_lam(p) += d_lambda_n * X_lam_d(n, p);")
    lam_re_idx <- 0
    if (!is.null(params$lambda$random)) {
      for (group in names(params$lambda$random)) {
        lam_re_idx <- lam_re_idx + 1
        re <- params$lambda$random[[group]]
        tag <- lam_re_tag(lam_re_idx)
        emit_re_gradient(L, tag, re, group, "d_lambda_n", "lambda", lam_re_idx, "lam")
      }
    }
  }

  # ---- lambda_B gradient (bivariate_dp: R_I_B -- Source B R_I) ----
  # bv.d_lambda_B is nonzero only for Source-B items (item_type == 3).
  # Chain rule uses R_I_B_val (which equals lambda_B_val when the user
  # specified lambda_B; constrained-equal default has has_lambda_B_dp = FALSE
  # so this block doesn't run).
  if (is_dp && isTRUE(params$.has_lambda_B_dp)) {
    L$add("")
    L$add(sprintf("    double d_lambda_B_n = bv.d_lambda_B * R_I_B_val * (1.0 - R_I_B_val)%s;", count_mult))
    L$add("    for (int p = 0; p < P_lamB; ++p) grad_beta_lamB(p) += d_lambda_B_n * X_lamB_d(n, p);")
    lamB_re_idx <- 0
    if (!is.null(params$lambda_B$random)) {
      for (group in names(params$lambda_B$random)) {
        lamB_re_idx <- lamB_re_idx + 1
        re <- params$lambda_B$random[[group]]
        tag <- lamB_re_tag(lamB_re_idx)
        emit_re_gradient(L, tag, re, group, "d_lambda_B_n", "lambda_B", lamB_re_idx, "lamB")
      }
    }
  }

  # ---- lambda2 gradient (bivariate_dp: R_S -- Source A R_S) ----
  if (is_dp && isTRUE(params$.has_lambda2)) {
    L$add("")
    if (!isTRUE(params$.has_lambda2_B)) {
      # lambda2_B constrained equal to lambda2: fold Source-B source-recollection
      # gradient (bv.d_lambda2_B) into the shared lambda2 accumulator (see lambda).
      L$add(sprintf("    double d_lambda2_n = (bv.d_lambda2 + bv.d_lambda2_B) * R_S_val * (1.0 - R_S_val)%s;", count_mult))
    } else {
      L$add(sprintf("    double d_lambda2_n = bv.d_lambda2 * R_S_val * (1.0 - R_S_val)%s;", count_mult))
    }
    L$add("    for (int p = 0; p < P_lam2; ++p) grad_beta_lam2(p) += d_lambda2_n * X_lam2_d(n, p);")
    lam2_re_idx <- 0
    if (!is.null(params$lambda2$random)) {
      for (group in names(params$lambda2$random)) {
        lam2_re_idx <- lam2_re_idx + 1
        re <- params$lambda2$random[[group]]
        tag <- lam2_re_tag(lam2_re_idx)
        emit_re_gradient(L, tag, re, group, "d_lambda2_n", "lambda2", lam2_re_idx, "lam2")
      }
    }
  }

  # ---- lambda2_B gradient (bivariate_dp: R_S_B -- Source B R_S) ----
  if (is_dp && isTRUE(params$.has_lambda2_B)) {
    L$add("")
    L$add(sprintf("    double d_lambda2_B_n = bv.d_lambda2_B * R_S_B_val * (1.0 - R_S_B_val)%s;", count_mult))
    L$add("    for (int p = 0; p < P_lam2B; ++p) grad_beta_lam2B(p) += d_lambda2_B_n * X_lam2B_d(n, p);")
    lam2B_re_idx <- 0
    if (!is.null(params$lambda2_B$random)) {
      for (group in names(params$lambda2_B$random)) {
        lam2B_re_idx <- lam2B_re_idx + 1
        re <- params$lambda2_B$random[[group]]
        tag <- lam2B_re_tag(lam2B_re_idx)
        emit_re_gradient(L, tag, re, group, "d_lambda2_B_n", "lambda2_B", lam2B_re_idx, "lam2B")
      }
    }
  }

  # ---- Threshold1 gradient (detection dimension) ----
  L$add("")
  L$add("    // Threshold1 gradient (detection dimension)")
  L$add("    double d_thresh1_lo = bv.d_thresh1_lo;")
  L$add("    double d_thresh1_hi = bv.d_thresh1_hi;")
  L$add("    int k_lo = bv.k1_lo;")
  L$add("    int k_hi = bv.k1_hi;")

  if (has_counts) {
    d_lo1 <- "d_thresh1_lo * count_n"
    d_hi1 <- "d_thresh1_hi * count_n"
  } else {
    d_lo1 <- "d_thresh1_lo"
    d_hi1 <- "d_thresh1_hi"
  }

  crit <- model_data$criterion
  is_crit_intercept_only <- isTRUE(crit$is_intercept_only) || crit$n_coef == 1
  emit_thresh_propagation(L, model_data, params, has_crit_re, d_lo1, d_hi1, is_crit_intercept_only)

  # ---- Threshold2 gradient (discrimination dimension) ----
  L$add("")
  L$add("    // Threshold2 gradient (discrimination dimension)")

  if (has_counts) {
    d_lo2 <- "bv.d_thresh2_lo * count_n"
    d_hi2 <- "bv.d_thresh2_hi * count_n"
  } else {
    d_lo2 <- "bv.d_thresh2_lo"
    d_hi2 <- "bv.d_thresh2_hi"
  }

  varying_sc <- isTRUE(params$.varying_source_criteria)

  if (varying_sc) {
    # Varying source criteria: propagate to per-bin threshold gradient accumulators
    new_shared <- identical(params$.new_source_criteria, "shared")
    if (new_shared) {
      # For new responses, propagate to shared new-response thresholds
      L$add("    if (is_new_response[yn - 1] == 1) {")
      L$add("      // New response: propagate to shared new-response thresholds")
      L$add(sprintf("      if (bv.k2_lo >= 0) batch_sdt::propagate_thresh_grad(bv.k2_lo, %s, mid2, n_thresh2, n_upper2, new_exp_gaps2, grad_new_mid2, grad_new_gaps2);", d_lo2))
      L$add(sprintf("      if (bv.k2_hi >= 0) batch_sdt::propagate_thresh_grad(bv.k2_hi, %s, mid2, n_thresh2, n_upper2, new_exp_gaps2, grad_new_mid2, grad_new_gaps2);", d_hi2))
      L$add("    } else {")
      L$add("      // Old response: propagate to varying thresholds for this detection bin")
      L$add("      int bin = yn - 1;")
      L$add("      Eigen::VectorXd bin_exp_gaps = vary_exp_gaps2.row(bin).transpose();")
      L$add(sprintf("      double& mid_ref = grad_vary_mid2(bin);"))
      L$add("      auto propagate_vary = [&](int k, double d) {")
      L$add("        mid_ref += d;")
      L$add("        if (k > mid2) { for (int j = 0; j <= k-mid2-1; ++j) grad_vary_gaps2(bin, j) += d * bin_exp_gaps(j); }")
      L$add("        else if (k < mid2) { for (int j = n_upper2; j <= n_upper2+(mid2-k-1); ++j) grad_vary_gaps2(bin, j) += d * (-bin_exp_gaps(j)); }")
      L$add("      };")
      L$add(sprintf("      if (bv.k2_lo >= 0) propagate_vary(bv.k2_lo, %s);", d_lo2))
      L$add(sprintf("      if (bv.k2_hi >= 0) propagate_vary(bv.k2_hi, %s);", d_hi2))
      L$add("    }")
    } else {
      # All items use varying thresholds indexed by yn
      L$add("    {")
      L$add("      int bin = yn - 1;")
      L$add("      Eigen::VectorXd bin_exp_gaps = vary_exp_gaps2.row(bin).transpose();")
      L$add(sprintf("      double& mid_ref = grad_vary_mid2(bin);"))
      L$add("      auto propagate_vary = [&](int k, double d) {")
      L$add("        mid_ref += d;")
      L$add("        if (k > mid2) { for (int j = 0; j <= k-mid2-1; ++j) grad_vary_gaps2(bin, j) += d * bin_exp_gaps(j); }")
      L$add("        else if (k < mid2) { for (int j = n_upper2; j <= n_upper2+(mid2-k-1); ++j) grad_vary_gaps2(bin, j) += d * (-bin_exp_gaps(j)); }")
      L$add("      };")
      L$add(sprintf("      if (bv.k2_lo >= 0) propagate_vary(bv.k2_lo, %s);", d_lo2))
      L$add(sprintf("      if (bv.k2_hi >= 0) propagate_vary(bv.k2_hi, %s);", d_hi2))
      L$add("    }")
    }
  } else {
    crit2 <- model_data$criterion2
    is_crit2_intercept_only <- is.null(crit2) || isTRUE(crit2$is_intercept_only) || crit2$n_coef == 1

    # Reuse the same thresh2 propagation infrastructure as vrdp2d
    if (has_crit2_re) {
      L$add("    double g_crit2_mid = 0.0;")
      L$add("    Eigen::VectorXd g_crit2_gaps = Eigen::VectorXd::Zero(n_gaps2);")

      if (is_crit2_intercept_only) {
        L$add("    if (bv.k2_lo >= 0) {")
        L$add(sprintf("      batch_sdt::propagate_thresh_grad_with_re(bv.k2_lo, %s, mid2, n_thresh2, n_upper2,", d_lo2))
        L$add("          exp_eff_gaps2, grad_thresh_mid2_pop, grad_log_gaps2_pop, g_crit2_mid, g_crit2_gaps);")
        L$add("    }")
        L$add("    if (bv.k2_hi >= 0) {")
        L$add(sprintf("      batch_sdt::propagate_thresh_grad_with_re(bv.k2_hi, %s, mid2, n_thresh2, n_upper2,", d_hi2))
        L$add("          exp_eff_gaps2, grad_thresh_mid2_pop, grad_log_gaps2_pop, g_crit2_mid, g_crit2_gaps);")
        L$add("    }")
      } else {
        L$add("    auto propagate_mp2 = [&](int k, double d) {")
        L$add("      for (int p = 0; p < P_crit2; ++p) grad_thresh_mid2_pop(p) += d * X_crit2_d(n, p);")
        L$add("      g_crit2_mid += d;")
        L$add("      if (k > mid2) {")
        L$add("        for (int j = 0; j <= k - mid2 - 1; ++j) {")
        L$add("          double g = d * exp_eff_gaps2(j);")
        L$add("          for (int p = 0; p < P_crit2; ++p) grad_log_gaps2_pop(j, p) += g * X_crit2_d(n, p);")
        L$add("          g_crit2_gaps(j) += g;")
        L$add("        }")
        L$add("      } else if (k < mid2) {")
        L$add("        for (int j = n_upper2; j <= n_upper2 + (mid2 - k - 1); ++j) {")
        L$add("          double g = d * (-exp_eff_gaps2(j));")
        L$add("          for (int p = 0; p < P_crit2; ++p) grad_log_gaps2_pop(j, p) += g * X_crit2_d(n, p);")
        L$add("          g_crit2_gaps(j) += g;")
        L$add("        }")
        L$add("      }")
        L$add("    };")
        L$add(sprintf("    if (bv.k2_lo >= 0) propagate_mp2(bv.k2_lo, %s);", d_lo2))
        L$add(sprintf("    if (bv.k2_hi >= 0) propagate_mp2(bv.k2_hi, %s);", d_hi2))
      }
      emit_crit2_re_grad_scatter(L, model_data, params)
    } else {
      if (is_crit2_intercept_only) {
        L$add("    if (bv.k2_lo >= 0) {")
        L$add(sprintf("      batch_sdt::propagate_thresh_grad(bv.k2_lo, %s, mid2, n_thresh2, n_upper2,", d_lo2))
        L$add("          pop_exp_gaps2, grad_thresh_mid2_pop, grad_log_gaps2_pop);")
        L$add("    }")
        L$add("    if (bv.k2_hi >= 0) {")
        L$add(sprintf("      batch_sdt::propagate_thresh_grad(bv.k2_hi, %s, mid2, n_thresh2, n_upper2,", d_hi2))
        L$add("          pop_exp_gaps2, grad_thresh_mid2_pop, grad_log_gaps2_pop);")
        L$add("    }")
      } else {
        L$add("    auto propagate_mp2 = [&](int k, double d) {")
        L$add("      for (int p = 0; p < P_crit2; ++p) grad_thresh_mid2_pop(p) += d * X_crit2_d(n, p);")
        L$add("      if (k > mid2) {")
        L$add("        for (int j = 0; j <= k - mid2 - 1; ++j) {")
        L$add("          double g = d * pop_exp_gaps2(j);")
        L$add("          for (int p = 0; p < P_crit2; ++p) grad_log_gaps2_pop(j, p) += g * X_crit2_d(n, p);")
        L$add("        }")
        L$add("      } else if (k < mid2) {")
        L$add("        for (int j = n_upper2; j <= n_upper2 + (mid2 - k - 1); ++j) {")
        L$add("          double g = d * (-pop_exp_gaps2(j));")
        L$add("          for (int p = 0; p < P_crit2; ++p) grad_log_gaps2_pop(j, p) += g * X_crit2_d(n, p);")
        L$add("        }")
        L$add("      }")
        L$add("    };")
        L$add(sprintf("    if (bv.k2_lo >= 0) propagate_mp2(bv.k2_lo, %s);", d_lo2))
        L$add(sprintf("    if (bv.k2_hi >= 0) propagate_mp2(bv.k2_hi, %s);", d_hi2))
      }
    }
  }
}
