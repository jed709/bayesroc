#' Parse SDT Model Formula
#'
#' Takes a formula for a single SDT parameter (dprime, criterion, theta) and
#' extracts fixed effects, random effects, and their correlation structure.
#'
#' @param formula A one-sided formula (e.g., ~ condition + (condition | subject))
#' @return A list with fixed and random effects specifications
#' @noRd

parse_broc_parameter_formula <- function(formula) {
  formula_str <- as.character(formula)
  if (length(formula_str) == 3) {
    rhs <- formula_str[3]
  } else if (length(formula_str) == 2) {
    rhs <- formula_str[2]
  } else {
    stop("Invalid formula structure")
  }

  random_effects <- extract_random_effects(rhs)
  fixed_str <- remove_random_effects(rhs)
  fixed <- parse_fixed_effects(fixed_str)
  random <- lapply(random_effects, parse_random_effect)

  result <- list(fixed = fixed, random = random)

  # Pass through smooth terms if any were detected
  if (length(fixed$smooth_terms) > 0) {
    result$smooth <- fixed$smooth_terms
  }

  result
}


#' Extract Random Effect Strings from Formula RHS
#'
#' @param rhs Character string of formula right-hand side
#' @return Character vector of random effect specifications (without outer parens)
#' @noRd

extract_random_effects <- function(rhs) {
  random_effects <- character(0)
  chars <- strsplit(rhs, "")[[1]]
  n <- length(chars)
  
  i <- 1
  while (i <= n) {
    if (chars[i] == "(") {
      depth <- 1
      start <- i
      i <- i + 1
      while (i <= n && depth > 0) {
        if (chars[i] == "(") depth <- depth + 1
        if (chars[i] == ")") depth <- depth - 1
        i <- i + 1
      }
      content <- paste(chars[(start + 1):(i - 2)], collapse = "")
      
      # Check if this is a random effect (contains | outside of any function)
      # Simple check: if it contains |, it's a random effect
      # encoding() with | inside would be invalid syntax anyway
      if (grepl("\\|", content)) {
        random_effects <- c(random_effects, content)
      }
    } else {
      i <- i + 1
    }
  }
  
  random_effects
}

remove_random_effects <- function(rhs) {
  chars <- strsplit(rhs, "")[[1]]
  n <- length(chars)
  
  result <- character(0)
  i <- 1
  
  while (i <= n) {
    if (chars[i] == "(") {
      depth <- 1
      start <- i
      i <- i + 1
      while (i <= n && depth > 0) {
        if (chars[i] == "(") depth <- depth + 1
        if (chars[i] == ")") depth <- depth - 1
        i <- i + 1
      }
      content <- paste(chars[(start + 1):(i - 2)], collapse = "")
      if (!grepl("\\|", content)) {
        # Not a random effect, keep it
        result <- c(result, chars[start:(i - 1)])
      }
      # If it has |, skip it (it's a random effect)
    } else {
      result <- c(result, chars[i])
      i <- i + 1
    }
  }
  
  fixed_str <- paste(result, collapse = "")
  fixed_str <- gsub("^[\\s+]+|[\\s+]+$", "", fixed_str, perl = TRUE)
  fixed_str <- gsub("\\+\\s*\\+", "+", fixed_str)
  fixed_str <- gsub("\\s+", " ", trimws(fixed_str))
  
  fixed_str
}

# Parse Fixed Effects String
# Handles single terms, interactions (*), and nested effects (:)
# Also detects encoding() wrapper for within-subject encoding manipulations
# @param fixed_str Character string of fixed effects
# @return List with terms, intercept flag, and encoding info

# Parenthesis-aware split on +
# Splits a string by + signs that are NOT inside parentheses
split_on_plus <- function(str) {
  chars <- strsplit(str, "")[[1]]
  n <- length(chars)
  parts <- character(0)
  current <- character(0)
  depth <- 0
  in_string <- FALSE
  string_char <- NULL

  for (i in seq_len(n)) {
    ch <- chars[i]

    # Track quoted strings (to avoid matching + inside strings)
    if (!in_string && ch %in% c('"', "'")) {
      in_string <- TRUE
      string_char <- ch
    } else if (in_string && ch == string_char) {
      in_string <- FALSE
    }

    if (!in_string) {
      if (ch == "(") depth <- depth + 1
      if (ch == ")") depth <- depth - 1
      if (ch == "+" && depth == 0) {
        parts <- c(parts, paste(current, collapse = ""))
        current <- character(0)
        next
      }
    }
    current <- c(current, ch)
  }
  # Add last part
  parts <- c(parts, paste(current, collapse = ""))
  parts
}


#' Detect Smooth Term
#'
#' Returns TRUE if a term string is a smooth term call: s() or t2().
#' Uses parenthesis-aware check to avoid matching variable names like "score".
#'
#' @param term_str A single term string
#' @return Logical
#' @noRd
detect_smooth_term <- function(term_str) {
 grepl("^\\s*(s|t2)\\s*\\(", term_str)
}


parse_fixed_effects <- function(fixed_str) {
  # Check for explicit no-intercept
  has_minus_one <- grepl("-\\s*1", fixed_str)
  has_zero_plus <- grepl("^\\s*0\\s*\\+", fixed_str)
  has_just_zero <- grepl("^\\s*0\\s*$", fixed_str)  # Just "0" with optional whitespace
  no_intercept <- has_minus_one || has_zero_plus || has_just_zero
  
  # Remove intercept modifiers
  fixed_str <- gsub("-\\s*1", "", fixed_str)
  fixed_str <- gsub("^\\s*0\\s*\\+", "", fixed_str)
  fixed_str <- gsub("^\\s*0\\s*$", "", fixed_str)  # Remove standalone 0
  
  # Split by + to get terms -- parenthesis-aware to handle conditional()
  terms <- split_on_plus(fixed_str)
  terms <- trimws(terms)
  terms <- terms[terms != "" & terms != "1"]

  # Detect and extract smooth terms (s(), t2()) -- they must not reach model.matrix()
  smooth_mask <- vapply(terms, detect_smooth_term, logical(1))
  smooth_terms <- terms[smooth_mask]
  terms <- terms[!smooth_mask]

  # Process each term to detect encoding(), conditional(), and expand interactions
  processed_terms <- list()
  encoding_vars <- character(0)
  conditional_terms <- list()

  for (term in terms) {
    is_encoding_term <- FALSE
    unwrapped_term <- term
    encoding_factor <- NULL

    # Check for conditional() -- must happen BEFORE * / : splitting
    if (grepl("conditional\\s*\\(", term)) {
      # Use paren-aware split to separate conditional(...) from :partner or *partner
      split_result <- split_conditional_interaction(term)
      cond_info <- detect_conditional(split_result$conditional_part)
      interaction_partners <- split_result$partners
      cond_info$interaction_partners <- interaction_partners

      conditional_terms[[length(conditional_terms) + 1]] <- cond_info
      processed_terms[[length(processed_terms) + 1]] <- list(
        term = term,
        type = if (length(interaction_partners) > 0) "interaction" else "main",
        factors = c(cond_info$term, interaction_partners),
        is_encoding = FALSE,
        conditional_info = cond_info
      )
      next
    }

    # Check for encoding() wrapper - can wrap a single factor or be part of interaction
    # e.g., encoding(condition) or encoding(condition):encoding(scale_wf)
    if (grepl("encoding\\s*\\(", term)) {
      # Extract ALL factors wrapped in encoding(): an interaction term may carry
      # several (encoding(a):encoding(b)). Grabbing only the first would leave the
      # rest unreported in the encoding-vars summary.
      enc_hits <- regmatches(term, gregexpr("encoding\\s*\\(([^)]+)\\)", term))[[1]]
      if (length(enc_hits) > 0) {
        facs <- trimws(gsub("encoding\\s*\\(([^)]+)\\)", "\\1", enc_hits))
        encoding_vars <- c(encoding_vars, facs)
        is_encoding_term <- TRUE
        # Replace encoding(X) with just X in the term
        unwrapped_term <- gsub("encoding\\s*\\(([^)]+)\\)", "\\1", term)
      }
    }

    # Check for interaction (*)
    if (grepl("\\*", unwrapped_term)) {
      # a*b expands to a + b + a:b
      factors <- trimws(strsplit(unwrapped_term, "\\*")[[1]])
      # Add main effects
      for (f in factors) {
        processed_terms[[length(processed_terms) + 1]] <- list(
          term = f,
          type = "main",
          factors = f,
          is_encoding = f %in% encoding_vars
        )
      }
      # Add interaction
      processed_terms[[length(processed_terms) + 1]] <- list(
        term = paste(factors, collapse = ":"),
        type = "interaction",
        factors = factors,
        is_encoding = any(factors %in% encoding_vars)
      )
    } else if (grepl(":", unwrapped_term)) {
      # Explicit interaction a:b
      factors <- trimws(strsplit(unwrapped_term, ":")[[1]])
      processed_terms[[length(processed_terms) + 1]] <- list(
        term = unwrapped_term,
        type = "interaction",
        factors = factors,
        is_encoding = any(factors %in% encoding_vars)
      )
    } else {
      # Simple main effect
      processed_terms[[length(processed_terms) + 1]] <- list(
        term = unwrapped_term,
        type = "main",
        factors = unwrapped_term,
        is_encoding = is_encoding_term
      )
    }
  }

  # Deduplicate terms (in case a*b added 'a' when 'a' was also explicit)
  seen_terms <- character(0)
  unique_terms <- list()
  for (pt in processed_terms) {
    if (!(pt$term %in% seen_terms)) {
      seen_terms <- c(seen_terms, pt$term)
      unique_terms[[length(unique_terms) + 1]] <- pt
    }
  }

  # Determine intercept
  if (length(unique_terms) == 0 && !no_intercept) {
    intercept <- TRUE
  } else if (no_intercept) {
    intercept <- FALSE
  } else {
    intercept <- TRUE
  }

  # Extract just the term strings for backwards compatibility
  term_strings <- sapply(unique_terms, `[[`, "term")
  if (length(term_strings) == 0) term_strings <- character(0)

  list(
    terms = term_strings,
    term_info = unique_terms,
    intercept = intercept,
    encoding_vars = encoding_vars,
    conditional_terms = conditional_terms,
    smooth_terms = smooth_terms
  )
}

parse_random_effect <- function(re_str) {
  pipe_split <- strsplit(re_str, "\\|")[[1]]
  n_parts <- length(pipe_split)
  
  if (grepl("\\|\\|", re_str)) {
    parts <- strsplit(re_str, "\\|\\|")[[1]]
    correlated <- FALSE
    cor_id <- NULL
    terms_str <- trimws(parts[1])
    group <- trimws(parts[2])
  } else if (n_parts == 3) {
    terms_str <- trimws(pipe_split[1])
    cor_id <- trimws(pipe_split[2])
    group <- trimws(pipe_split[3])
    correlated <- TRUE
  } else if (n_parts == 2) {
    terms_str <- trimws(pipe_split[1])
    group <- trimws(pipe_split[2])
    correlated <- TRUE
    cor_id <- NULL
  } else {
    stop("Invalid random effect specification: ", re_str)
  }
  
  terms_parsed <- parse_fixed_effects(terms_str)
  
  list(
    group = group,
    terms = terms_parsed$terms,
    term_info = terms_parsed$term_info,
    intercept = terms_parsed$intercept,
    correlated = correlated,
    cor_id = cor_id,
    encoding_vars = terms_parsed$encoding_vars
  )
}

#' Parse resp() Arguments
#'
#' Receives the content inside resp(...), e.g. "det_conf, src_conf".
#' Both arguments are positional. The family determines whether the second
#' variable is a source response or R/K variable.
#'
#' @param inner_str The string inside resp(...)
#' @return List with confidence and response2
#' @noRd
parse_resp_args <- function(inner_str) {
  # Split on commas at depth 0
  chars <- strsplit(inner_str, "")[[1]]
  depth <- 0
  splits <- c(0)
  for (i in seq_along(chars)) {
    if (chars[i] == "(") depth <- depth + 1
    if (chars[i] == ")") depth <- depth - 1
    if (chars[i] == "," && depth == 0) splits <- c(splits, i)
  }
  splits <- c(splits, nchar(inner_str) + 1)

  args <- character(0)
  for (k in seq_len(length(splits) - 1)) {
    args <- c(args, trimws(substr(inner_str, splits[k] + 1, splits[k + 1] - 1)))
  }

  if (length(args) < 2 || nchar(args[1]) == 0 || nchar(args[2]) == 0) {
    stop("resp() requires two variables: resp(response1, response2)")
  }
  if (length(args) > 2) {
    stop("resp() takes exactly two variables: resp(response1, response2)")
  }
  # Reject old named-arg syntax (src = ..., rk = ...)
  if (grepl("=", args[1]) || grepl("=", args[2])) {
    stop("resp() now uses positional arguments only: resp(response1, response2)\n",
         "  Old syntax like resp(conf, src = src_var) or resp(conf, rk = rk_var) ",
         "is no longer supported.")
  }

  list(confidence = trimws(args[1]), response2 = trimws(args[2]))
}


#' Parse Response Variable with Conditioning Variable
#'
#' Parses response specifications like "y | signal_var", "resp(det, src) | type_var".
#' The role of the conditioning variable (old/new indicator, source variable,
#' item type) is determined by the family, not by wrapper functions.
#'
#' @param response_str Character string of response specification
#' @param family_name Family name string (determines how the conditioning variable is used)
#' @return List with response, condition_var, response2, rk
#' @noRd

parse_response <- function(response_str, family_name = NULL) {
  response_str <- trimws(response_str)
  is_cumulative <- family_name %in% c("cumulative", "bivariate_cumulative")
  is_bivariate <- family_name %in% c("bivariate_sdt", "bivariate_dp", "vrdp2d", "bivariate_cumulative")
  is_cdp <- identical(family_name, "cdp")
  is_source <- identical(family_name, "source_mixture")

  # --- Detect resp() wrapper using paren-depth-aware parsing ---
  resp_info <- NULL
  if (grepl("^resp\\s*\\(", response_str)) {
    # Find the matching close paren for resp(
    match_start <- regexpr("resp\\s*\\(", response_str)
    paren_start <- match_start + attr(match_start, "match.length") - 1
    chars <- strsplit(response_str, "")[[1]]
    depth <- 1
    i <- paren_start + 1
    while (i <= length(chars) && depth > 0) {
      if (chars[i] == "(") depth <- depth + 1
      if (chars[i] == ")") depth <- depth - 1
      i <- i + 1
    }
    paren_end <- i - 1

    inner <- substr(response_str, paren_start + 1, paren_end - 1)
    resp_info <- parse_resp_args(inner)

    # Replace response_str: use confidence var + everything after resp(...)
    remainder <- trimws(substr(response_str, paren_end + 1, nchar(response_str)))
    response_str <- paste0(resp_info$confidence, remainder)
  }

  if (grepl("\\|", response_str)) {
    parts <- strsplit(response_str, "\\|")[[1]]
    response_var <- trimws(parts[1])
    condition_var <- trimws(parts[2])

    if (length(parts) > 2) {
      stop("Only one conditioning variable is supported after |: response | variable ~ ...")
    }

    # Map conditioning variable to the appropriate role based on family
    result <- list(response = response_var, is_old = NULL, source = NULL,
                   item_type = NULL, rk = NULL, response2 = NULL)

    if (is_bivariate) {
      result$item_type <- condition_var
    } else if (is_source) {
      result$source <- condition_var
    } else {
      # SDT families (evsdt, uvsdt, dpsdt, mixture) and CDP
      result$is_old <- condition_var
    }

    # If resp() was used, store response2 and rk (based on family)
    if (!is.null(resp_info)) {
      if (is_cdp) {
        result$rk <- resp_info$response2
      } else {
        result$response2 <- resp_info$response2
      }
    }

    return(result)
  } else if (!is.null(resp_info)) {
    stop("resp() formulas must include a conditioning variable after |: ",
         "resp(var1, var2) | condition_var ~ ...")
  } else if (is_cumulative) {
    # Bare response (no |) -- allowed for cumulative
    return(list(response = response_str, is_old = NULL, source = NULL,
                item_type = NULL, rk = NULL, response2 = NULL))
  } else {
    stop("Formula requires a conditioning variable: response | variable ~ predictors\n",
         "The variable after | depends on the family:\n",
         "  SDT families: response | old_new_var ~ ...\n",
         "  source_mixture: response | source_var ~ ...\n",
         "  bivariate families: resp(det, src) | item_type_var ~ ...\n",
         "  CDP families: resp(conf, rk) | old_new_var ~ ...")
  }
}


#' Parse Complete SDT Formula
#'
#' Main function to parse a full SDT model specification
#'
#' @param dprime Formula for d' parameter
#' @param criterion Formula for criterion parameter
#' @param sigma Formula for sigma parameter (optional, NULL to fix at 1)
#' @param lambda Formula for lambda parameter (optional, for DPSDT/mixture)
#' @param dprime2 Formula for second d' (optional, for mixture)
#' @param sigma2 Formula for second sigma (optional, for mixture or bivariate_gaussian)
#' @param dprime_B Formula for d' of Source B (optional, for source_mixture or bivariate_gaussian)
#' @param lambda_B Formula for lambda of Source B (optional, for source_mixture)
#' @param discrim Formula for discrimination parameter (required for bivariate_gaussian)
#' @param discrim_B Formula for discrimination of Source B (optional, for bivariate_gaussian)
#' @param sigma_B Formula for sigma of Source B (optional, for bivariate_gaussian)
#' @param sigma2_B Formula for sigma2 of Source B (optional, for bivariate_gaussian)
#' @param rho Formula for correlation parameter (required for bivariate_gaussian)
#' @param rho_B Formula for correlation of Source B (optional, for bivariate_gaussian)
#' @param rho_N Formula for correlation of new items (optional, for bivariate_gaussian)
#' @param criterion2 Formula for second criterion/threshold (required for bivariate_gaussian)
#' @param data Data frame (used to extract factor levels)
#' @param family_name Family name string (determines role of conditioning variable)
#' @return Parsed formula structure
#' @noRd

parse_broc_formula <- function(dprime, criterion, sigma = NULL, lambda = NULL,
                              dprime2 = NULL, sigma2 = NULL,
                              dprime_B = NULL, lambda_B = NULL,
                              discrim = NULL, discrim_B = NULL,
                              sigma_B = NULL, sigma2_B = NULL,
                              rho = NULL, rho_B = NULL, rho_N = NULL,
                              criterion2 = NULL,
                              rec_crit = NULL,
                              know_crit = NULL,
                              dprime_L = NULL, sigma_L = NULL, lambda_L = NULL,
                              lambda2 = NULL, lambda2_B = NULL,
                              data = NULL, family_name = NULL) {
  dprime_str <- as.character(dprime)
  if (length(dprime_str) != 3) {
    stop("Primary formula must be two-sided: response | variable ~ predictors\n",
         "  SDT families: conf | old_new_var ~ predictors\n",
         "  source_mixture: conf | source_var ~ predictors\n",
         "  bivariate families: resp(det, src) | item_type_var ~ predictors\n",
         "  CDP families: resp(conf, rk) | old_new_var ~ predictors\n",
         "  cumulative: response ~ predictors (no conditioning variable)")
  }

  response_info <- parse_response(dprime_str[2], family_name = family_name)
  
  parsed <- list(
    response = response_info,
    dprime = parse_broc_parameter_formula(dprime),
    criterion = parse_broc_parameter_formula(criterion)
  )
  
  # Parse optional parameters (only if not NULL)
  if (!is.null(sigma)) {
    parsed$sigma <- parse_broc_parameter_formula(sigma)
  }
  
  if (!is.null(lambda)) {
    parsed$lambda <- parse_broc_parameter_formula(lambda)
  }
  
  if (!is.null(dprime2)) {
    parsed$dprime2 <- parse_broc_parameter_formula(dprime2)
  }
  
  if (!is.null(sigma2)) {
    parsed$sigma2 <- parse_broc_parameter_formula(sigma2)
  }
  
  if (!is.null(dprime_B)) {
    parsed$dprime_B <- parse_broc_parameter_formula(dprime_B)
  }
  
  if (!is.null(lambda_B)) {
    parsed$lambda_B <- parse_broc_parameter_formula(lambda_B)
  }
  
  # Bivariate SDT specific parameters
  if (!is.null(discrim)) {
    parsed$discrim <- parse_broc_parameter_formula(discrim)
  }
  
  if (!is.null(discrim_B)) {
    parsed$discrim_B <- parse_broc_parameter_formula(discrim_B)
  }
  
  if (!is.null(sigma_B)) {
    parsed$sigma_B <- parse_broc_parameter_formula(sigma_B)
  }
  
  if (!is.null(sigma2_B)) {
    parsed$sigma2_B <- parse_broc_parameter_formula(sigma2_B)
  }
  
  if (!is.null(rho)) {
    parsed$rho <- parse_broc_parameter_formula(rho)
  }
  
  if (!is.null(rho_B)) {
    parsed$rho_B <- parse_broc_parameter_formula(rho_B)
  }
  
  if (!is.null(rho_N)) {
    parsed$rho_N <- parse_broc_parameter_formula(rho_N)
  }
  
  if (!is.null(criterion2)) {
    parsed$criterion2 <- parse_broc_parameter_formula(criterion2)
  }
  
  # CDP specific parameters
  if (!is.null(rec_crit)) {
    parsed$rec_crit <- parse_broc_parameter_formula(rec_crit)
  }
  if (!is.null(know_crit)) {
    parsed$know_crit <- parse_broc_parameter_formula(know_crit)
  }

  # Lure mixture parameters (mixture family)
  if (!is.null(dprime_L)) {
    parsed$dprime_L <- parse_broc_parameter_formula(dprime_L)
  }
  
  if (!is.null(sigma_L)) {
    parsed$sigma_L <- parse_broc_parameter_formula(sigma_L)
  }
  
  if (!is.null(lambda_L)) {
    parsed$lambda_L <- parse_broc_parameter_formula(lambda_L)
  }

  # Bivariate DP specific: lambda2 (source recollection)
  if (!is.null(lambda2)) {
    parsed$lambda2 <- parse_broc_parameter_formula(lambda2)
  }

  # Bivariate DP specific: lambda2_B (source recollection for Source B)
  if (!is.null(lambda2_B)) {
    parsed$lambda2_B <- parse_broc_parameter_formula(lambda2_B)
  }

  # For bivariate families, extract the second response variable from resp()
  if (!is.null(family_name) && family_name %in% c("bivariate_sdt", "bivariate_dp", "vrdp2d", "bivariate_cumulative")) {
    if (!is.null(response_info$response2)) {
      parsed$response2 <- response_info$response2
    } else {
      stop("Bivariate family requires resp() with two response variables.\n",
           "Use resp(det_conf, src_conf) | item_type_var ~ predictors")
    }
  }

  # For CDP, validate rk is present (comes from resp())
  if (identical(family_name, "cdp")) {
    if (is.null(response_info$rk)) {
      stop("cdp family requires resp() with confidence and R/K variables.\n",
           "Use resp(conf, rk_var) | old_new_var ~ predictors")
    }
  }
  
  # Collect all encoding variables (those that only apply to old items)
  # All possible parameters for encoding vars
  all_param_names <- c("dprime", "sigma", "lambda", "dprime2", "sigma2",
                       "dprime_B", "lambda_B", "discrim", "discrim_B",
                       "sigma_B", "sigma2_B", "rho", "rho_B", "rho_N",
                       "rec_crit", "know_crit", "lambda2", "lambda2_B",
                       "dprime_L", "sigma_L", "lambda_L")
  
  encoding_vars <- character(0)
  for (pname in all_param_names) {
    if (!is.null(parsed[[pname]]) && !is.null(parsed[[pname]]$fixed$encoding_vars)) {
      encoding_vars <- c(encoding_vars, parsed[[pname]]$fixed$encoding_vars)
    }
  }
  parsed$encoding_vars <- unique(encoding_vars)
  
  # Collect all unique grouping factors
  # Include criterion and criterion2 in grouping
  all_param_names_with_crit <- c(all_param_names, "criterion", "criterion2")
  
  all_groups <- character(0)
  for (pname in all_param_names_with_crit) {
    if (!is.null(parsed[[pname]]) && !is.null(parsed[[pname]]$random)) {
      groups <- sapply(parsed[[pname]]$random, `[[`, "group")
      all_groups <- c(all_groups, groups)
    }
  }
  parsed$grouping_factors <- unique(all_groups)
  
  # Collect all correlation IDs
  all_cor_ids <- character(0)
  for (pname in all_param_names_with_crit) {
    if (!is.null(parsed[[pname]]) && !is.null(parsed[[pname]]$random)) {
      cor_ids <- sapply(parsed[[pname]]$random, `[[`, "cor_id")
      cor_ids <- cor_ids[!sapply(cor_ids, is.null)]
      all_cor_ids <- c(all_cor_ids, unlist(cor_ids))
    }
  }
  parsed$cross_cor_ids <- unique(all_cor_ids)
  
  if (!is.null(data)) {
    parsed$data_info <- extract_data_info(parsed, data)
  }
  
  class(parsed) <- c("broc_formula", "list")
  parsed
}


#' Extract Data Information for Formula Terms
#'
#' @param parsed Parsed formula structure
#' @param data Data frame
#' @return List with factor levels and dimensions
#' @noRd

extract_data_info <- function(parsed, data) {
  info <- list()
  
  # Get all unique base terms (not interactions) across all parameters
  all_terms <- character(0)
  
  # All possible parameters
  param_names <- c("dprime", "criterion", "sigma", "lambda", "dprime2", "sigma2",
                   "dprime_B", "lambda_B", "discrim", "discrim_B",
                   "sigma_B", "sigma2_B", "rho", "rho_B", "rho_N", "criterion2",
                   "rec_crit", "know_crit", "dprime_L", "sigma_L", "lambda_L",
                   "lambda2", "lambda2_B")
  
  # From fixed effects
  for (param in param_names) {
    if (!is.null(parsed[[param]])) {
      for (ti in parsed[[param]]$fixed$term_info) {
        all_terms <- c(all_terms, ti$factors)
      }
    }
  }
  
  # From random effects
  for (param in param_names) {
    if (!is.null(parsed[[param]])) {
      for (re in parsed[[param]]$random) {
        all_terms <- c(all_terms, re$terms)
      }
    }
  }
  
  all_terms <- unique(all_terms)
  
  # Get levels for each term
  info$term_levels <- list()
  for (term in all_terms) {
    if (term %in% names(data)) {
      if (is.factor(data[[term]])) {
        info$term_levels[[term]] <- levels(data[[term]])
      } else {
        info$term_levels[[term]] <- sort(unique(data[[term]]))
      }
    }
  }
  
  # Get levels for grouping factors
  info$group_levels <- list()
  info$group_n <- list()
  for (group in parsed$grouping_factors) {
    if (group %in% names(data)) {
      info$group_levels[[group]] <- sort(unique(data[[group]]))
      info$group_n[[group]] <- length(info$group_levels[[group]])
    }
  }
  
  info$K <- length(unique(data[[parsed$response$response]]))
  info$N <- nrow(data)
  
  info
}


#' Print Method for SDT Formula
#'
#' @param x An broc_formula object
#' @param ... Additional arguments (ignored)
#' @noRd

print.broc_formula <- function(x, ...) {
  cat("bayesroc model formula\n")
  cat("======================\n\n")

  # If print.broc_model attached the family on the way in, use its
  # external_aliases to choose user-facing labels per parameter section
  # (cdp -> rec/fam/sigma_R, cumulative -> mu/cutpoints, etc.).
  fam <- attr(x, ".broc_family")
  ext <- if (!is.null(fam$external_aliases)) fam$external_aliases else list()
  family_name <- if (!is.null(fam)) fam$family else NA
  is_cumulative <- identical(family_name, "cumulative")
  is_bivariate_cumulative <- identical(family_name, "bivariate_cumulative")

  # Section label resolver: prefer external alias, fall back to a hand-curated
  # label, finally to a Title-cased internal name.
  section_label <- function(internal, fallback = NULL) {
    ext_name <- ext[[internal]]
    if (!is.null(ext_name)) {
      # e.g. "rec" -> "Rec (Recollection d')", "mu" -> "Mu (location)"
      pretty <- switch(ext_name,
        rec       = "Rec (Recollection d')",
        fam       = "Fam (Familiarity d')",
        sigma_R   = "Sigma_R (Recollection SD, log link)",
        sigma_F   = "Sigma_F (Familiarity SD, log link)",
        mu        = "Mu (location)",
        mu1       = "Mu1 (dimension 1 location)",
        mu2       = "Mu2 (dimension 2 location)",
        cutpoints = "Cutpoints",
        cutpoints1 = "Cutpoints1",
        cutpoints2 = "Cutpoints2",
        ext_name
      )
      return(pretty)
    }
    if (!is.null(fallback)) fallback else internal
  }

  cat("Response:", x$response$response, "\n")
  # Skip the .cumulative_is_old fake column that was injected internally --
  # it's an implementation detail, not something the user wrote.
  if (!is.null(x$response$is_old) && x$response$is_old != ".cumulative_is_old") {
    cat("Condition variable:", x$response$is_old, "(old/new)\n")
  }
  if (!is.null(x$response$source)) cat("Condition variable:", x$response$source, "(source)\n")
  if (!is.null(x$response$item_type)) cat("Condition variable:", x$response$item_type, "(item type)\n")
  if (!is.null(x$response$rk)) cat("R/K variable:", x$response$rk, "\n")
  if (!is.null(x$response$response2)) cat("Second response:", x$response$response2, "\n")

  if (length(x$encoding_vars) > 0) {
    cat("Encoding-only variables:", paste(x$encoding_vars, collapse = ", "), "\n")
  }
  cat("\n")

  cat(section_label("dprime", "d' (dprime)"), ":\n", sep = "")
  print_parameter_formula(x$dprime)

  cat("\n", section_label("criterion", "Criterion"), ":\n", sep = "")
  print_parameter_formula(x$criterion)

  if (!is.null(x[["sigma"]])) {
    cat("\n", section_label("sigma", "Sigma (old item SD)"), ":\n", sep = "")
    print_parameter_formula(x[["sigma"]])
  }

  if (!is.null(x[["lambda"]])) {
    cat("\nLambda (recollection/mixture probability):\n")
    print_parameter_formula(x[["lambda"]])
  }

  if (!is.null(x[["dprime2"]])) {
    cat("\n", section_label("dprime2", "d'2 (second state d')"), ":\n", sep = "")
    print_parameter_formula(x[["dprime2"]])
  }

  if (!is.null(x[["sigma2"]])) {
    cat("\n", section_label("sigma2", "Sigma2 (second state SD)"), ":\n", sep = "")
    print_parameter_formula(x[["sigma2"]])
  }

  # Bivariate/source parameters
  if (!is.null(x[["dprime_B"]])) {
    cat("\nd'_B (Source B detection):\n")
    print_parameter_formula(x[["dprime_B"]])
  }
  if (!is.null(x[["lambda_B"]])) {
    cat("\nLambda_B (Source B attention):\n")
    print_parameter_formula(x[["lambda_B"]])
  }
  if (!is.null(x[["discrim"]])) {
    cat("\nDiscrim (source discrimination):\n")
    print_parameter_formula(x[["discrim"]])
  }
  if (!is.null(x[["discrim_B"]])) {
    cat("\nDiscrim_B (Source B discrimination):\n")
    print_parameter_formula(x[["discrim_B"]])
  }
  if (!is.null(x[["sigma_B"]])) {
    cat("\nSigma_B (Source B detection SD):\n")
    print_parameter_formula(x[["sigma_B"]])
  }
  if (!is.null(x[["sigma2_B"]])) {
    cat("\nSigma2_B (Source B discrimination SD):\n")
    print_parameter_formula(x[["sigma2_B"]])
  }
  if (!is.null(x[["rho"]])) {
    cat("\nRho (detection-discrimination correlation):\n")
    print_parameter_formula(x[["rho"]])
  }
  if (!is.null(x[["rho_B"]])) {
    cat("\nRho_B (Source B correlation):\n")
    print_parameter_formula(x[["rho_B"]])
  }
  if (!is.null(x[["rho_N"]])) {
    cat("\nRho_N (new item correlation):\n")
    print_parameter_formula(x[["rho_N"]])
  }
  if (!is.null(x[["lambda2"]])) {
    cat("\nLambda2 (source recollection probability):\n")
    print_parameter_formula(x[["lambda2"]])
  }
  if (!is.null(x[["lambda2_B"]])) {
    cat("\nLambda2_B (Source B source recollection probability):\n")
    print_parameter_formula(x[["lambda2_B"]])
  }

  # Lure mixture parameters
  if (!is.null(x[["dprime_L"]])) {
    cat("\nd'_L (lure discriminability):\n")
    print_parameter_formula(x[["dprime_L"]])
  }
  if (!is.null(x[["sigma_L"]])) {
    cat("\nSigma_L (lure SD):\n")
    print_parameter_formula(x[["sigma_L"]])
  }
  if (!is.null(x[["lambda_L"]])) {
    cat("\nLambda_L (lure mixture weight):\n")
    print_parameter_formula(x[["lambda_L"]])
  }

  # CDP parameters
  if (!is.null(x[["rec_crit"]])) {
    cat("\nRec_crit (recollection criterion):\n")
    print_parameter_formula(x[["rec_crit"]])
  }
  if (!is.null(x[["know_crit"]])) {
    cat("\nKnow_crit (know criterion):\n")
    print_parameter_formula(x[["know_crit"]])
  }

  # Criterion2 (bivariate/vrdp2d)
  if (!is.null(x[["criterion2"]])) {
    cat("\nCriterion2 (source thresholds):\n")
    print_parameter_formula(x[["criterion2"]])
  }

  if (length(x[["cross_cor_ids"]]) > 0) {
    cat("\nCross-parameter correlation groups:",
        paste(x[["cross_cor_ids"]], collapse = ", "), "\n")
  }

  invisible(x)
}


#' Helper to Print Parameter Formula
#'
#' @param param Parsed parameter specification
#' @noRd

print_parameter_formula <- function(param) {
  # Fixed effects. Encoding-only variables are named once in the header line;
  # we deliberately don't repeat encoding() wrappers on every term here.
  display_terms <- param$fixed$terms

  if (param$fixed$intercept && length(display_terms) == 0) {
    cat("  Fixed: ~ 1 (intercept only)\n")
  } else if (param$fixed$intercept) {
    cat("  Fixed: ~ 1 +", paste(display_terms, collapse = " + "), "\n")
  } else {
    cat("  Fixed: ~ 0 +", paste(display_terms, collapse = " + "), "\n")
  }
  
  # Random effects
  if (length(param$random) > 0) {
    cat("  Random:\n")
    for (re in param$random) {
      terms_str <- if (re$intercept && length(re$terms) == 0) {
        "1"
      } else if (re$intercept) {
        paste(c("1", re$terms), collapse = " + ")
      } else {
        paste(c("0", re$terms), collapse = " + ")
      }
      
      bar <- if (!re$correlated) {
        "||"
      } else if (!is.null(re$cor_id)) {
        paste0("|", re$cor_id, "|")
      } else {
        "|"
      }
      
      cat("    (", terms_str, " ", bar, " ", re$group, ")\n", sep = "")
    }
  } else {
    cat("  Random: none\n")
  }
}
