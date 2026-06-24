#' Conditional Design Matrix Functions
#'
#' Support for conditional() in SDT formulas. Allows predictors to apply
#' only to a subset of observations defined by a where condition.


#' Conditional Term Marker
#'
#' Used in formulas to indicate that a predictor should only vary for
#' observations matching the where condition. Observations not matching
#' share a single intercept column.
#'
#' @param term A factor variable name
#' @param where A logical expression evaluated on the data
#' @return This function should not be called directly
#' @noRd
conditional <- function(term, where) {
  stop("conditional() should only be used inside formulas, not called directly.")
}


#' Detect conditional() in a Term String
#'
#' Parses a term string like 'conditional(production, where = group == "matched")'
#' and extracts the term name and where expression string.
#'
#' @param term_str Character string of the conditional() call
#' @return List with term (character) and where_str (character), or NULL if not conditional
#' @noRd
detect_conditional <- function(term_str) {
  if (!grepl("conditional\\s*\\(", term_str)) return(NULL)

  # Find the opening paren of conditional(
  match_start <- regexpr("conditional\\s*\\(", term_str)
  paren_start <- match_start + attr(match_start, "match.length") - 1
  chars <- strsplit(term_str, "")[[1]]

  # Find matching close paren
  depth <- 1
  i <- paren_start + 1
  while (i <= length(chars) && depth > 0) {
    if (chars[i] == "(") depth <- depth + 1
    if (chars[i] == ")") depth <- depth - 1
    i <- i + 1
  }
  paren_end <- i - 1

  # Extract content inside conditional(...)
  inner <- substr(term_str, paren_start + 1, paren_end - 1)

  # Split on first comma that is outside nested parens to get term and where=...
  # Need to find the comma that separates term from where = ...
  inner_chars <- strsplit(inner, "")[[1]]
  comma_pos <- NULL
  depth <- 0
  for (j in seq_along(inner_chars)) {
    if (inner_chars[j] == "(") depth <- depth + 1
    if (inner_chars[j] == ")") depth <- depth - 1
    if (inner_chars[j] == "," && depth == 0) {
      comma_pos <- j
      break
    }
  }

  if (is.null(comma_pos)) {
    stop("conditional() requires both a term and a where argument: ",
         "conditional(term, where = expr)")
  }

  term_part <- trimws(substr(inner, 1, comma_pos - 1))
  where_part <- trimws(substr(inner, comma_pos + 1, nchar(inner)))

  # Strip "where =" or "where=" prefix if present; also accept positional arg
  where_str <- sub("^where\\s*=\\s*", "", where_part)

  list(term = term_part, where_str = where_str)
}


#' Parenthesis-Aware Split of Conditional Interaction
#'
#' Splits a term like 'conditional(production, where = group == "matched"):session'
#' into the conditional part and interaction partners. Handles : and * outside
#' the conditional() parens.
#'
#' @param term_str Full term string potentially containing conditional() and operators
#' @return List with conditional_part (character) and partners (character vector)
#' @noRd
split_conditional_interaction <- function(term_str) {
  chars <- strsplit(term_str, "")[[1]]
  n <- length(chars)

  # Track positions of : and * that are at depth 0
  depth <- 0
  split_positions <- integer(0)
  split_types <- character(0)

  for (i in seq_len(n)) {
    if (chars[i] == "(") depth <- depth + 1
    if (chars[i] == ")") depth <- depth - 1
    if (depth == 0 && chars[i] %in% c(":", "*")) {
      split_positions <- c(split_positions, i)
      split_types <- c(split_types, chars[i])
    }
  }

  if (length(split_positions) == 0) {
    return(list(conditional_part = term_str, partners = character(0)))
  }

  # Split the string at the operator positions
  parts <- character(0)
  prev <- 1
  for (pos in split_positions) {
    parts <- c(parts, trimws(substr(term_str, prev, pos - 1)))
    prev <- pos + 1
  }
  parts <- c(parts, trimws(substr(term_str, prev, n)))

  # Find which part contains conditional(
  cond_idx <- grep("conditional\\s*\\(", parts)
  if (length(cond_idx) != 1) {
    stop("Expected exactly one conditional() in term: ", term_str)
  }

  conditional_part <- parts[cond_idx]
  partners <- parts[-cond_idx]

  # If any split was *, also need to expand main effects
  # For now, just return the partners; parse_fixed_effects handles * expansion
  has_star <- any(split_types == "*")

  list(conditional_part = conditional_part, partners = partners,
       has_star = has_star)
}


#' Build Conditional Design Matrix Columns
#'
#' Core function: evaluates where mask, builds special design matrix.
#' FALSE observations get a single intercept column; TRUE observations
#' get one column per level of the conditional term (cell-means coded).
#'
#' @param term Character: the factor variable name inside conditional()
#' @param where_str Character: the where expression as string
#' @param data Data frame
#' @param interaction_partners Character vector of partner variable names (may be empty)
#' @param intercept Logical: whether an intercept is present in the formula
#' @return List with X (matrix), col_names (character), conditional_info (metadata)
#' @noRd
build_conditional_columns <- function(term, where_str, data, interaction_partners = NULL,
                                      intercept = TRUE) {
  N <- nrow(data)

  # Evaluate the where mask
  mask <- eval(parse(text = where_str), envir = data, enclos = parent.frame())
  if (!is.logical(mask) || length(mask) != N) {
    stop("conditional() where expression must evaluate to a logical vector ",
         "of length nrow(data). Got length ", length(mask))
  }

  true_idx <- which(mask)
  false_idx <- which(!mask)

  # Get levels of the conditional term
  if (is.factor(data[[term]])) {
    term_levels <- levels(data[[term]])
  } else {
    term_levels <- sort(unique(as.character(data[[term]])))
  }
  n_term_levels <- length(term_levels)

  # Build partner interaction levels if partners exist
  has_partners <- !is.null(interaction_partners) && length(interaction_partners) > 0

  if (!has_partners) {
    # Simple case: FALSE intercept + TRUE level columns
    # Always cell-means coded (one 1 per row)
    n_cols <- 1 + n_term_levels  # FALSE + TRUE levels
    col_names <- c("cond_FALSE", paste0("cond_", term, ":", term_levels))
    X <- matrix(0, nrow = N, ncol = n_cols)
    colnames(X) <- col_names

    # FALSE rows get column 1
    X[false_idx, 1] <- 1

    # TRUE rows get their level column
    term_vals <- as.character(data[[term]])
    for (i in seq_along(term_levels)) {
      lev <- term_levels[i]
      match_idx <- which(mask & term_vals == lev)
      X[match_idx, 1 + i] <- 1
    }

  } else {
    # Interaction case: cross conditional columns with partner levels
    # Get partner levels
    partner_level_list <- lapply(interaction_partners, function(p) {
      if (is.factor(data[[p]])) {
        levels(data[[p]])
      } else {
        sort(unique(as.character(data[[p]])))
      }
    })
    names(partner_level_list) <- interaction_partners

    # Expand partner combinations (grid of all partners)
    partner_grid <- expand.grid(partner_level_list, stringsAsFactors = FALSE)
    n_partner_combos <- nrow(partner_grid)
    partner_labels <- apply(partner_grid, 1, function(row) paste(row, collapse = ":"))

    # FALSE group: partner-only columns
    false_col_names <- paste0("cond_FALSE:", partner_labels)

    # TRUE group: term x partner crossed columns
    true_col_names <- character(0)
    for (lev in term_levels) {
      for (pl in partner_labels) {
        true_col_names <- c(true_col_names, paste0("cond_", term, ":", lev, ":", pl))
      }
    }

    n_false_cols <- length(false_col_names)
    n_true_cols <- length(true_col_names)
    n_cols <- n_false_cols + n_true_cols
    col_names <- c(false_col_names, true_col_names)

    X <- matrix(0, nrow = N, ncol = n_cols)
    colnames(X) <- col_names

    # Get partner values for each row
    partner_vals <- as.data.frame(lapply(interaction_partners, function(p) {
      as.character(data[[p]])
    }), stringsAsFactors = FALSE)
    names(partner_vals) <- interaction_partners
    row_partner_labels <- apply(partner_vals, 1, function(row) paste(row, collapse = ":"))

    # FALSE rows: match on partner label only
    for (i in seq_along(partner_labels)) {
      pl <- partner_labels[i]
      match_idx <- intersect(false_idx, which(row_partner_labels == pl))
      X[match_idx, i] <- 1
    }

    # TRUE rows: match on term level + partner label
    term_vals <- as.character(data[[term]])
    col_offset <- n_false_cols
    for (ti in seq_along(term_levels)) {
      for (pi in seq_along(partner_labels)) {
        col_idx <- col_offset + (ti - 1) * n_partner_combos + pi
        match_idx <- which(mask & term_vals == term_levels[ti] &
                            row_partner_labels == partner_labels[pi])
        X[match_idx, col_idx] <- 1
      }
    }
  }

  # Store metadata for predict rebuild
  conditional_info <- list(
    term = term,
    where_str = where_str,
    term_levels = term_levels,
    interaction_partners = if (has_partners) interaction_partners else NULL,
    partner_level_list = if (has_partners) partner_level_list else NULL,
    col_names = col_names,
    n_cols = n_cols
  )

  list(X = X, col_names = col_names, conditional_info = conditional_info)
}


#' Rebuild Conditional Columns for New Data
#'
#' Uses stored metadata from build_conditional_columns() to rebuild the
#' design matrix for prediction with new data.
#'
#' @param conditional_info Metadata list from build_conditional_columns()
#' @param newdata New data frame
#' @return Design matrix
#' @noRd
rebuild_conditional_columns <- function(conditional_info, newdata) {
  result <- build_conditional_columns(
    term = conditional_info$term,
    where_str = conditional_info$where_str,
    data = newdata,
    interaction_partners = conditional_info$interaction_partners
  )
  # Align to original column names
  X <- align_matrix_columns(result$X, conditional_info$col_names)
  X
}


#' Build Full Fixed Effects Matrix with Conditional Terms
#'
#' Called when conditional terms are present. Builds conditional columns
#' for conditional terms and normal model.matrix columns for regular terms,
#' then column-binds them.
#'
#' @param parsed_fixed Parsed fixed effects structure (from parse_fixed_effects)
#' @param formula Original formula
#' @param data Data frame
#' @param encoding_vars Encoding variables
#' @param is_old_var Is-old variable name
#' @return Standard fixed effects list: X, n_coef, coef_names, has_encoding, formula, conditional_info
#' @noRd
build_conditional_fixed_matrix <- function(parsed_fixed, formula, data,
                                            encoding_vars, is_old_var) {
  # Separate conditional and non-conditional terms
  cond_terms <- list()
  regular_terms <- character(0)
  regular_term_info <- list()

  for (ti in parsed_fixed$term_info) {
    if (!is.null(ti$conditional_info)) {
      cond_terms[[length(cond_terms) + 1]] <- ti
    } else {
      regular_terms <- c(regular_terms, ti$term)
      regular_term_info[[length(regular_term_info) + 1]] <- ti
    }
  }

  N <- nrow(data)
  blocks <- list()
  all_cond_info <- list()

  # Build regular term columns (if any)
  # Note: When ONLY conditional terms exist, skip the intercept column --
  # the conditional cond_FALSE column already provides the baseline.
  # Only add regular columns if there are actual non-conditional terms.
  if (length(regular_terms) > 0) {
    terms_str <- paste(regular_terms, collapse = " + ")
    if (parsed_fixed$intercept) {
      reg_formula <- as.formula(paste("~", terms_str))
    } else {
      reg_formula <- as.formula(paste("~ 0 +", terms_str))
    }

    has_encoding <- !is.null(encoding_vars) && any(encoding_vars %in% all.vars(reg_formula))
    if (has_encoding && !is.null(is_old_var)) {
      old_idx <- which(data[[is_old_var]] == 1)
      old_data <- data[old_idx, , drop = FALSE]
      for (v in all.vars(reg_formula)) {
        if (v %in% names(old_data) && is.factor(old_data[[v]])) {
          dropped <- droplevels(old_data[[v]])
          if (nlevels(dropped) >= 2) {
            old_data[[v]] <- dropped
          }
        }
      }
      X_old <- model.matrix(reg_formula, data = old_data)
      X_reg <- matrix(0, nrow = N, ncol = ncol(X_old))
      colnames(X_reg) <- colnames(X_old)
      X_reg[old_idx, ] <- X_old
    } else {
      X_reg <- model.matrix(reg_formula, data = data)
    }
    blocks[[length(blocks) + 1]] <- X_reg
  }

  # Build conditional term columns
  for (ct in cond_terms) {
    ci <- ct$conditional_info
    result <- build_conditional_columns(
      term = ci$term,
      where_str = ci$where_str,
      data = data,
      interaction_partners = ci$interaction_partners,
      intercept = parsed_fixed$intercept
    )
    blocks[[length(blocks) + 1]] <- result$X
    all_cond_info[[length(all_cond_info) + 1]] <- result$conditional_info
  }

  # Column-bind all blocks
  X <- do.call(cbind, blocks)

  has_encoding_overall <- !is.null(encoding_vars) &&
    any(encoding_vars %in% all.vars(formula))

  list(
    X = X,
    n_coef = ncol(X),
    coef_names = colnames(X),
    has_encoding = has_encoding_overall,
    formula = formula,
    conditional_info = if (length(all_cond_info) > 0) all_cond_info else NULL
  )
}


#' Extract Conditional Terms from Parsed Formula
#'
#' Checks parsed fixed effects for conditional terms and returns them,
#' or NULL if none exist.
#'
#' @param parsed_param Parsed parameter (e.g., parsed$criterion)
#' @return List of conditional terms, or NULL
#' @noRd
extract_conditional_from_parsed <- function(parsed_param) {
  if (is.null(parsed_param) || is.null(parsed_param$fixed)) return(NULL)
  if (is.null(parsed_param$fixed$conditional_terms)) return(NULL)
  if (length(parsed_param$fixed$conditional_terms) == 0) return(NULL)
  parsed_param$fixed$conditional_terms
}
