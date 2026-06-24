#' Build Fixed Effects Design Matrix
#'
#' Uses model.matrix() to properly handle all fixed effects structures
#' including interactions, contrasts, etc.
#'
#' @param formula One-sided formula for fixed effects (e.g., ~ 0 + condition:exp)
#' @param data Full data frame
#' @param encoding_vars Character vector of encoding variable names
#' @param is_old_var Name of the is_old indicator variable
#' @return List with design matrix and metadata
#' @noRd

build_fixed_effects_matrix <- function(formula, data, encoding_vars = NULL, is_old_var = NULL,
                                       conditional_terms = NULL, parsed_fixed = NULL) {

  # If conditional terms are present, delegate to the conditional builder
  if (!is.null(conditional_terms) && length(conditional_terms) > 0 && !is.null(parsed_fixed)) {
    return(build_conditional_fixed_matrix(parsed_fixed, formula, data,
                                           encoding_vars, is_old_var))
  }

  # Check if any encoding variables are in the formula
  formula_vars <- all.vars(formula)
  has_encoding <- !is.null(encoding_vars) && any(encoding_vars %in% formula_vars)
  
  if (has_encoding && !is.null(is_old_var)) {
    # Build matrix on old items only, then map back
    old_idx <- which(data[[is_old_var]] > 0)  # target rows (1 treatment / 0.5 centered)
    old_data <- data[old_idx, , drop = FALSE]
    
    # Drop unused levels for encoding factors in the formula,
    # but only if 2+ levels remain (1-level factors crash model.matrix)
    for (v in formula_vars) {
      if (v %in% names(old_data) && is.factor(old_data[[v]])) {
        dropped <- droplevels(old_data[[v]])
        if (nlevels(dropped) >= 2) {
          old_data[[v]] <- dropped
        }
      }
    }
    
    # Build matrix on old data
    X_old <- model.matrix(formula, data = old_data)
    
    # Create full matrix with zeros for new items
    N <- nrow(data)
    P <- ncol(X_old)
    X <- matrix(0, nrow = N, ncol = P)
    colnames(X) <- colnames(X_old)
    X[old_idx, ] <- X_old
    
  } else {
    # No encoding variables - use full data
    X <- model.matrix(formula, data = data)
  }
  
  list(
    X = X,
    n_coef = ncol(X),
    coef_names = colnames(X),
    has_encoding = has_encoding,
    formula = formula
  )
}


#' Build Random Effects Structure
#'
#' Parses random effects and builds the necessary structures.
#' Now supports both:
#' - Simple categorical: uses index arrays for fast lookup (beta[idx])
#' - General case: uses full Z matrices (Z[n,] * u[s,]')
#'
#' @param re_terms Random effects formula component
#' @param data Full data frame
#' @param encoding_vars Character vector of encoding variable names
#' @param is_old_var Name of the is_old indicator variable
#' @return List with random effects structure
#' @noRd

build_random_effects_structure <- function(re_terms, data, encoding_vars = NULL, is_old_var = NULL) {
  
  re_list <- list()
  
  for (re in re_terms) {
    group <- re$group
    
    # Get group indices
    if (is.factor(data[[group]])) {
      group_idx <- as.integer(data[[group]])
      n_groups <- nlevels(data[[group]])
    } else {
      group_fac <- as.factor(data[[group]])
      group_idx <- as.integer(group_fac)
      n_groups <- nlevels(group_fac)
    }
    
    # Get the terms (accounting for encoding variables)
    re_terms_clean <- re$terms
    
    # Check for encoding - need to look at term_info if available
    is_encoding <- FALSE
    if (!is.null(re$encoding_vars) && length(re$encoding_vars) > 0) {
      is_encoding <- TRUE
    } else if (!is.null(encoding_vars)) {
      # Check if any RE terms are encoding vars
      for (t in re_terms_clean) {
        # Strip encoding() wrapper if present
        t_clean <- gsub("encoding\\s*\\(([^)]+)\\)", "\\1", t)
        if (t_clean %in% encoding_vars) {
          is_encoding <- TRUE
          break
        }
      }
    }
    
    N <- nrow(data)
    
    # Determine dimension of random effects
    if (length(re_terms_clean) == 0 || (length(re_terms_clean) == 1 && re_terms_clean[1] == "")) {
      # Intercept only - no Z matrix needed, just use group index
      re_list[[group]] <- list(
        group = group,
        group_idx = group_idx,
        n_groups = n_groups,
        dim = 1,
        level_names = "(Intercept)",
        type = "intercept",
        term = NULL,
        term_idx = NULL,
        Z = NULL,  # No Z matrix for intercept-only
        use_z_matrix = FALSE,
        correlated = re$correlated,
        cor_id = re$cor_id,
        is_encoding = FALSE
      )
    } else {
      # Has varying slopes - build Z matrix using model.matrix
      # Clean up the terms (remove encoding() wrapper)
      terms_clean <- sapply(re_terms_clean, function(t) {
        gsub("encoding\\s*\\(([^)]+)\\)", "\\1", t)
      })

      # Check if any RE term contains conditional()
      re_cond_info <- NULL
      re_has_conditional <- FALSE
      for (ti in re$term_info) {
        if (!is.null(ti$conditional_info)) {
          re_has_conditional <- TRUE
          re_cond_info <- ti$conditional_info
          break
        }
      }

      if (re_has_conditional && !is.null(re_cond_info)) {
        # Build Z via conditional columns (always one-hot -> fast index path)
        cond_result <- build_conditional_columns(
          term = re_cond_info$term,
          where_str = re_cond_info$where_str,
          data = data,
          interaction_partners = re_cond_info$interaction_partners,
          intercept = re$intercept
        )
        Z <- cond_result$X
        dim <- ncol(Z)
        level_names <- colnames(Z)
        primary_term <- re_cond_info$term

        can_use_index <- check_simple_categorical(Z)
        if (can_use_index) {
          term_idx <- build_term_index(Z)
          use_z_matrix <- FALSE
        } else {
          term_idx <- NULL
          use_z_matrix <- TRUE
        }

        re_list[[group]] <- list(
          group = group,
          group_idx = group_idx,
          n_groups = n_groups,
          dim = dim,
          level_names = level_names,
          type = "varying_slope",
          term = primary_term,
          term_idx = term_idx,
          Z = if (use_z_matrix) Z else NULL,
          use_z_matrix = use_z_matrix,
          correlated = re$correlated,
          cor_id = re$cor_id,
          is_encoding = is_encoding,
          re_formula = NULL,
          conditional_info = cond_result$conditional_info
        )
      } else {
      # Build formula for RE - respect intercept flag for treatment vs cell-means coding
      if (re$intercept) {
        re_formula <- as.formula(paste("~ 1 +", paste(terms_clean, collapse = " + ")))
      } else {
        re_formula <- as.formula(paste("~ 0 +", paste(terms_clean, collapse = " + ")))
      }

      if (is_encoding && !is.null(is_old_var)) {
        # Build on old items only, zeros for new items
        old_idx <- which(data[[is_old_var]] > 0)  # target rows (1 treatment / 0.5 centered)
        old_data <- data[old_idx, , drop = FALSE]

        # Drop unused levels for factors in formula, only if 2+ remain
        formula_vars <- all.vars(re_formula)
        for (v in formula_vars) {
          if (v %in% names(old_data) && is.factor(old_data[[v]])) {
            dropped <- droplevels(old_data[[v]])
            if (nlevels(dropped) >= 2) {
              old_data[[v]] <- dropped
            }
          }
        }

        Z_old <- model.matrix(re_formula, data = old_data)
        dim <- ncol(Z_old)
        level_names <- colnames(Z_old)

        # Create full Z matrix with zeros for new items
        Z <- matrix(0, nrow = N, ncol = dim)
        colnames(Z) <- level_names
        Z[old_idx, ] <- Z_old

        formula_vars <- all.vars(re_formula)
        primary_term <- formula_vars[1]

      } else {
        # Use full data
        Z <- model.matrix(re_formula, data = data)
        dim <- ncol(Z)
        level_names <- colnames(Z)

        formula_vars <- all.vars(re_formula)
        primary_term <- formula_vars[1]
      }

      # Determine whether the fast index-based approach applies
      # This is possible when:
      # 1. Each row has exactly one 1 and rest zeros (simple categorical)
      # 2. OR each row is all zeros (for encoding with new items)
      can_use_index <- check_simple_categorical(Z)

      if (can_use_index) {
        # Build index array for fast lookup
        term_idx <- build_term_index(Z)
        use_z_matrix <- FALSE
      } else {
        # Must use full Z matrix
        term_idx <- NULL
        use_z_matrix <- TRUE
      }

      re_list[[group]] <- list(
        group = group,
        group_idx = group_idx,
        n_groups = n_groups,
        dim = dim,
        level_names = level_names,
        type = if (length(terms_clean) > 1 || grepl(":", terms_clean[1])) "interaction" else "varying_slope",
        term = primary_term,
        term_idx = term_idx,
        Z = if (use_z_matrix) Z else NULL,
        use_z_matrix = use_z_matrix,
        correlated = re$correlated,
        cor_id = re$cor_id,
        is_encoding = is_encoding,
        re_formula = re_formula
      )
      }
    }
  }
  
  re_list
}


#' Check if Z matrix has simple categorical structure
#' 
#' Returns TRUE if each row has at most one non-zero entry and that entry is 1.
#' This allows using fast index-based lookup instead of matrix multiplication.
#' @noRd
check_simple_categorical <- function(Z) {
  for (i in 1:nrow(Z)) {
    row <- Z[i, ]
    nonzero <- which(row != 0)
    if (length(nonzero) == 0) {
      # All zeros - OK (e.g., new items for encoding)
      next
    } else if (length(nonzero) == 1 && row[nonzero] == 1) {
      # Exactly one 1 - OK
      next
    } else {
      # Multiple non-zeros or non-1 value - need full Z matrix
      return(FALSE)
    }
  }
  TRUE
}


#' Build term index array from simple categorical Z matrix
#' 
#' For each row, returns the column index with value 1, or 0 if all zeros.
#' @noRd
build_term_index <- function(Z) {
  N <- nrow(Z)
  term_idx <- integer(N)
  for (i in 1:N) {
    row_idx <- which(Z[i, ] == 1)
    if (length(row_idx) == 1) {
      term_idx[i] <- row_idx
    } else {
      term_idx[i] <- 0L
    }
  }
  term_idx
}


#' Build Criterion (Threshold) Structure
#'
#' Thresholds never use encoding variables - all items respond
#' Now properly handles continuous random slopes using Z matrices.
#'
#' @param formula One-sided formula for criterion fixed effects
#' @param re_terms Random effects terms for criterion
#' @param data Data frame
#' @param K Number of response categories
#' @return List with criterion structure
#' @noRd

build_criterion_structure <- function(formula, re_terms, data, n_thresh,
                                     conditional_terms = NULL, parsed_fixed = NULL) {

  # First, strip random effects from formula to get just fixed effects
  formula_str <- as.character(formula)
  if (length(formula_str) == 2) {
    rhs <- formula_str[2]
  } else {
    rhs <- formula_str[3]
  }
  fixed_str <- remove_random_effects(rhs)
  fixed_str <- trimws(fixed_str)
  if (fixed_str == "" || is.na(fixed_str)) {
    fixed_str <- "1"
  }
  fixed_formula <- as.formula(paste("~", fixed_str), env = environment(formula))

  # If conditional terms present, build fixed effects via conditional builder
  crit_cond_info <- NULL
  if (!is.null(conditional_terms) && length(conditional_terms) > 0 && !is.null(parsed_fixed)) {
    crit_fixed <- build_conditional_fixed_matrix(parsed_fixed, formula, data, NULL, NULL)
    X <- crit_fixed$X
    n_coef <- crit_fixed$n_coef
    coef_names <- crit_fixed$coef_names
    crit_cond_info <- crit_fixed$conditional_info
  } else {
    # Drop any unused factor levels in the full data for formula vars
    formula_vars <- all.vars(fixed_formula)
    data_clean <- data
    for (v in formula_vars) {
      if (v %in% names(data_clean) && is.factor(data_clean[[v]])) {
        dropped <- droplevels(data_clean[[v]])
        if (nlevels(dropped) >= 2) {
          data_clean[[v]] <- dropped
        }
      }
    }

    # Fixed effects - use model.matrix with the cleaned formula
    X <- model.matrix(fixed_formula, data = data_clean)
    n_coef <- ncol(X)
    coef_names <- colnames(X)
  }

  # Check if it's intercept only
  is_intercept_only <- n_coef == 1 && coef_names[1] == "(Intercept)"

  N <- nrow(data)

  # Random effects
  re_list <- list()
  for (re in re_terms) {
    group <- re$group

    if (is.factor(data[[group]])) {
      group_idx <- as.integer(data[[group]])
      n_groups <- nlevels(data[[group]])
    } else {
      group_fac <- as.factor(data[[group]])
      group_idx <- as.integer(group_fac)
      n_groups <- nlevels(group_fac)
    }

    if (length(re$terms) == 0 || (length(re$terms) == 1 && re$terms[1] == "")) {
      # Intercept only - n_thresh correlated REs per group
      # No Z matrix needed, dimension is just n_thresh
      re_list[[group]] <- list(
        group = group,
        group_idx = group_idx,
        n_groups = n_groups,
        dim = n_thresh,
        n_cond_levels = 1,
        n_re_terms = 1,  # Just intercept
        term = NULL,
        term_idx = NULL,
        Z = NULL,
        use_z_matrix = FALSE,
        correlated = re$correlated,
        cor_id = re$cor_id
      )
    } else {
      # Check if RE terms contain conditional()
      re_cond_info <- NULL
      for (ti in re$term_info) {
        if (!is.null(ti$conditional_info)) {
          re_cond_info <- ti$conditional_info
          break
        }
      }

      if (!is.null(re_cond_info)) {
        # Build Z via conditional columns
        cond_result <- build_conditional_columns(
          term = re_cond_info$term,
          where_str = re_cond_info$where_str,
          data = data,
          interaction_partners = re_cond_info$interaction_partners,
          intercept = re$intercept
        )
        Z <- cond_result$X
        n_re_terms <- ncol(Z)
        level_names <- colnames(Z)

        can_use_index <- check_simple_categorical(Z)
        if (can_use_index) {
          term_idx <- build_term_index(Z)
          n_cond_levels <- n_re_terms
          re_list[[group]] <- list(
            group = group,
            group_idx = group_idx,
            n_groups = n_groups,
            dim = n_thresh * n_cond_levels,
            n_cond_levels = n_cond_levels,
            n_re_terms = n_re_terms,
            level_names = level_names,
            term = re_cond_info$term,
            term_idx = term_idx,
            Z = NULL,
            use_z_matrix = FALSE,
            correlated = re$correlated,
            cor_id = re$cor_id,
            conditional_info = cond_result$conditional_info
          )
        } else {
          re_list[[group]] <- list(
            group = group,
            group_idx = group_idx,
            n_groups = n_groups,
            dim = n_thresh * n_re_terms,
            n_cond_levels = n_re_terms,
            n_re_terms = n_re_terms,
            level_names = level_names,
            term = NULL,
            term_idx = NULL,
            Z = Z,
            use_z_matrix = TRUE,
            correlated = re$correlated,
            cor_id = re$cor_id,
            conditional_info = cond_result$conditional_info
          )
        }
      } else {
      # Has varying slopes - build Z matrix using model.matrix
      # Clean up the terms
      terms_clean <- re$terms

      # Build formula for RE (include intercept if specified)
      if (re$intercept) {
        re_formula <- as.formula(paste("~ 1 +", paste(terms_clean, collapse = " + ")))
      } else {
        re_formula <- as.formula(paste("~ 0 +", paste(terms_clean, collapse = " + ")))
      }

      # Build the Z matrix
      Z <- model.matrix(re_formula, data = data)
      n_re_terms <- ncol(Z)  # Number of RE terms (e.g., intercept + slope = 2)
      level_names <- colnames(Z)

      # Check whether simple index-based lookup applies
      # This only works for simple categorical (one-hot) structures
      can_use_index <- check_simple_categorical(Z)

      if (can_use_index) {
        # Simple categorical - use index-based lookup
        term_idx <- build_term_index(Z)
        n_cond_levels <- n_re_terms

        re_list[[group]] <- list(
          group = group,
          group_idx = group_idx,
          n_groups = n_groups,
          dim = n_thresh * n_cond_levels,  # Total dimension for all thresholds
          n_cond_levels = n_cond_levels,
          n_re_terms = n_re_terms,
          level_names = level_names,
          term = re$terms[1],
          term_idx = term_idx,
          Z = NULL,
          use_z_matrix = FALSE,
          correlated = re$correlated,
          cor_id = re$cor_id
        )
      } else {
        # General case - use Z matrix
        # Total dimension: n_thresh thresholds x n_re_terms
        re_list[[group]] <- list(
          group = group,
          group_idx = group_idx,
          n_groups = n_groups,
          dim = n_thresh * n_re_terms,  # e.g., 3 thresholds x 2 RE terms = 6
          n_cond_levels = n_re_terms,  # For compatibility
          n_re_terms = n_re_terms,
          level_names = level_names,
          term = NULL,
          term_idx = NULL,
          Z = Z,
          use_z_matrix = TRUE,
          correlated = re$correlated,
          cor_id = re$cor_id
        )
      }
      }
    }
  }

  list(
    X = X,
    n_coef = n_coef,
    coef_names = coef_names,
    is_intercept_only = is_intercept_only,
    random = re_list,
    n_thresh = n_thresh,
    fixed_formula = fixed_formula,
    conditional_info = crit_cond_info
  )
}
