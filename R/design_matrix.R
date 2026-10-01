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

  if (!is.null(conditional_terms) && length(conditional_terms) > 0 && !is.null(parsed_fixed)) {
    return(build_conditional_fixed_matrix(parsed_fixed, formula, data,
                                           encoding_vars, is_old_var))
  }

  formula_vars <- all.vars(formula)
  has_encoding <- !is.null(encoding_vars) && any(encoding_vars %in% formula_vars)
  
  if (has_encoding && !is.null(is_old_var)) {
    old_idx <- which(data[[is_old_var]] > 0)
    old_data <- data[old_idx, , drop = FALSE]
    
    for (v in formula_vars) {
      if (v %in% names(old_data) && is.factor(old_data[[v]])) {
        dropped <- droplevels(old_data[[v]])
        if (nlevels(dropped) >= 2) {
          old_data[[v]] <- dropped
        }
      }
    }
    
    X_old <- model.matrix(formula, data = old_data)
    
    N <- nrow(data)
    P <- ncol(X_old)
    X <- matrix(0, nrow = N, ncol = P)
    colnames(X) <- colnames(X_old)
    X[old_idx, ] <- X_old
    
  } else {
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
#' - Simple categorical: uses index arrays for fast lookup (`beta[idx]`)
#' - General case: uses full Z matrices (`Z[n,] * u[s,]'`)
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
    if (any(!vapply(c(list(re), re$parts), function(p) isTRUE(p$cor_thresholds %||% TRUE), logical(1)))) {
      stop("gr(", re$group, ", cor_thresholds = FALSE) applies only to threshold parameters ",
           "(criterion, criterion2): other parameters have one effect per term.", call. = FALSE)
    }
    if (!is.null(re$parts)) {
      structs <- lapply(re$parts, function(p)
        build_random_effects_structure(list(p), data, encoding_vars, is_old_var)[[p$group]])
      re_list[[re$group]] <- .combine_re_parts(structs, re$parts, nrow(data))
      next
    }
    group <- re$group
    
    if (is.factor(data[[group]])) {
      group_idx <- as.integer(data[[group]])
      n_groups <- nlevels(data[[group]])
    } else {
      group_fac <- as.factor(data[[group]])
      group_idx <- as.integer(group_fac)
      n_groups <- nlevels(group_fac)
    }
    
    re_terms_clean <- re$terms
    
    is_encoding <- FALSE
    if (!is.null(re$encoding_vars) && length(re$encoding_vars) > 0) {
      is_encoding <- TRUE
    } else if (!is.null(encoding_vars)) {
      for (t in re_terms_clean) {
        t_clean <- gsub("encoding\\s*\\(([^)]+)\\)", "\\1", t)
        if (t_clean %in% encoding_vars) {
          is_encoding <- TRUE
          break
        }
      }
    }
    
    N <- nrow(data)
    
    if (length(re_terms_clean) == 0 || (length(re_terms_clean) == 1 && re_terms_clean[1] == "")) {
      re_list[[group]] <- list(
        group = group,
        group_idx = group_idx,
        n_groups = n_groups,
        dim = 1,
        level_names = "(Intercept)",
        type = "intercept",
        term = NULL,
        term_idx = NULL,
        Z = NULL,
        use_z_matrix = FALSE,
        correlated = re$correlated,
        cor_id = re$cor_id,
        is_encoding = FALSE
      )
    } else {
      terms_clean <- sapply(re_terms_clean, function(t) {
        gsub("encoding\\s*\\(([^)]+)\\)", "\\1", t)
      })

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
      if (re$intercept) {
        re_formula <- as.formula(paste("~ 1 +", paste(terms_clean, collapse = " + ")))
      } else {
        re_formula <- as.formula(paste("~ 0 +", paste(terms_clean, collapse = " + ")))
      }

      if (is_encoding && !is.null(is_old_var)) {
        old_idx <- which(data[[is_old_var]] > 0)
        old_data <- data[old_idx, , drop = FALSE]

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

        Z <- matrix(0, nrow = N, ncol = dim)
        colnames(Z) <- level_names
        Z[old_idx, ] <- Z_old

        formula_vars <- all.vars(re_formula)
        primary_term <- formula_vars[1]

      } else {
        Z <- model.matrix(re_formula, data = data)
        dim <- ncol(Z)
        level_names <- colnames(Z)

        formula_vars <- all.vars(re_formula)
        primary_term <- formula_vars[1]
      }

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
  
  for (g in names(re_list)) re_list[[g]] <- .finalize_re_blocks(re_list[[g]], re_list[[g]]$dim, 1L)
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
      next
    } else if (length(nonzero) == 1 && row[nonzero] == 1) {
      next
    } else {
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

  crit_cond_info <- NULL
  if (!is.null(conditional_terms) && length(conditional_terms) > 0 && !is.null(parsed_fixed)) {
    crit_fixed <- build_conditional_fixed_matrix(parsed_fixed, formula, data, NULL, NULL)
    X <- crit_fixed$X
    n_coef <- crit_fixed$n_coef
    coef_names <- crit_fixed$coef_names
    crit_cond_info <- crit_fixed$conditional_info
  } else {
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

    X <- model.matrix(fixed_formula, data = data_clean)
    n_coef <- ncol(X)
    coef_names <- colnames(X)
  }

  is_intercept_only <- n_coef == 1 && coef_names[1] == "(Intercept)"

  N <- nrow(data)

  re_list <- list()
  for (re in re_terms) {
    if (!is.null(re$parts)) {
      structs <- lapply(re$parts, function(p)
        build_criterion_structure(formula, list(p), data, n_thresh, conditional_terms,
                                  parsed_fixed)$random[[p$group]])
      re_list[[re$group]] <- .combine_re_parts(structs, re$parts, nrow(data), n_thresh = n_thresh)
      next
    }
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
      re_list[[group]] <- list(
        group = group,
        group_idx = group_idx,
        n_groups = n_groups,
        dim = n_thresh,
        n_cond_levels = 1,
        n_re_terms = 1,
        term = NULL,
        term_idx = NULL,
        Z = NULL,
        use_z_matrix = FALSE,
        correlated = re$correlated,
        cor_id = re$cor_id
      )
    } else {
      re_cond_info <- NULL
      for (ti in re$term_info) {
        if (!is.null(ti$conditional_info)) {
          re_cond_info <- ti$conditional_info
          break
        }
      }

      if (!is.null(re_cond_info)) {
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
      terms_clean <- re$terms

      if (re$intercept) {
        re_formula <- as.formula(paste("~ 1 +", paste(terms_clean, collapse = " + ")))
      } else {
        re_formula <- as.formula(paste("~ 0 +", paste(terms_clean, collapse = " + ")))
      }

      Z <- model.matrix(re_formula, data = data)
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
          term = re$terms[1],
          term_idx = term_idx,
          Z = NULL,
          use_z_matrix = FALSE,
          correlated = re$correlated,
          cor_id = re$cor_id,
          re_formula = re_formula
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
          re_formula = re_formula
        )
      }
      }
    }
  }

  for (re in re_terms) {
    st <- re_list[[re$group]]
    re_list[[re$group]] <- .finalize_re_blocks(st, st$n_re_terms %||% 1L, n_thresh,
                                               cor_thresholds = re$cor_thresholds %||% TRUE)
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

# =============================================================================
# Random-effect correlation blocks
# =============================================================================

.re_part_Z <- function(st, N) {
  if (isTRUE(st$use_z_matrix)) return(as.matrix(st$Z))
  R <- st$n_re_terms %||% st$dim
  Z <- matrix(0, N, R, dimnames = list(NULL, st$level_names %||% "(Intercept)"))
  if (!is.null(st$term_idx)) {
    ok <- which(st$term_idx > 0)
    Z[cbind(ok, st$term_idx[ok])] <- 1
  } else {
    Z[, 1] <- 1
  }
  Z
}

.re_col_set <- function(n_cols, correlated, offset = 0L) {
  if (isTRUE(correlated)) rep(offset + 1L, n_cols) else offset + seq_len(n_cols)
}

.re_cor_blocks <- function(R, C, col_set, set_cor_thresholds) {
  blocks <- list()
  for (s in unique(col_set)) {
    js <- which(col_set == s) - 1L
    if (isTRUE(set_cor_thresholds[[as.character(s)]])) {
      blocks[[length(blocks) + 1L]] <- sort(as.vector(outer(js, (seq_len(C) - 1L) * R, "+"))) + 1L
    } else {
      for (k in seq_len(C) - 1L) blocks[[length(blocks) + 1L]] <- k * R + js + 1L
    }
  }
  blocks[lengths(blocks) > 1L]
}

.finalize_re_blocks <- function(st, R, C, cor_thresholds = TRUE) {
  if (is.null(st$col_set)) {
    st$col_set <- .re_col_set(R, st$correlated)
    st$set_cor_thresholds <- stats::setNames(rep(isTRUE(cor_thresholds), length(unique(st$col_set))),
                                             unique(st$col_set))
  }
  st$cor_blocks <- .re_cor_blocks(R, C, st$col_set, st$set_cor_thresholds)
  st$correlated <- length(st$cor_blocks) > 0
  st
}

.combine_re_parts <- function(structs, parts, N, n_thresh = NULL) {
  Zs <- lapply(structs, .re_part_Z, N = N)
  col_set <- integer(0)
  set_ct <- logical(0)
  for (i in seq_along(Zs)) {
    cs <- .re_col_set(ncol(Zs[[i]]), parts[[i]]$correlated, offset = max(c(0L, col_set)))
    col_set <- c(col_set, cs)
    set_ct[as.character(unique(cs))] <- isTRUE(parts[[i]]$cor_thresholds %||% TRUE)
  }
  Z <- do.call(cbind, Zs)
  first <- structs[[1]]
  R <- ncol(Z)
  st <- list(group = first$group, group_idx = first$group_idx, n_groups = first$n_groups,
             level_names = colnames(Z), term = NULL, term_idx = NULL, Z = Z, use_z_matrix = TRUE,
             correlated = TRUE, cor_id = NULL, col_set = col_set, set_cor_thresholds = set_ct,
             is_encoding = any(vapply(structs, function(x) isTRUE(x$is_encoding), logical(1))),
             part_structs = lapply(structs, function(x) { x$group_idx <- NULL; x }))
  if (is.null(n_thresh)) {
    st$dim <- R
    st$type <- "interaction"
    st$re_formula <- NULL
  } else {
    st$dim <- n_thresh * R
    st$n_cond_levels <- R
    st$n_re_terms <- R
  }
  st
}
