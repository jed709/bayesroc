#' Smooth Term Support via mgcv
#'
#' Functions for building smooth basis matrices using mgcv::smoothCon()
#' and decomposing them via mgcv::smooth2random() into unpenalized (fixed)
#' and penalized (random) components.


#' Sanitize Smooth Label for Variable Naming
#'
#' Converts mgcv smooth labels like "s(age)" to valid Stan/Python variable
#' name fragments like "sage".
#'
#' @param label Character string, e.g. "s(age)", "t2(x,z)"
#' @return Sanitized string suitable for use in variable names
#' @noRd
sanitize_smooth_label <- function(label) {
  # Remove all non-alphanumeric characters
  gsub("[^a-zA-Z0-9]", "", label)
}


#' Build Smooth Term Structures
#'
#' For each smooth term string, calls mgcv::smoothCon() and
#' mgcv::smooth2random() to decompose the smooth into unpenalized (Xs)
#' and penalized (Zs) components.
#'
#' @param smooth_specs Character vector of smooth term strings (e.g., "s(age, k=10)")
#' @param data Data frame
#' @param encoding_vars Character vector of encoding variable names (or NULL)
#' @param is_old_var Name of is_old indicator variable (or NULL)
#' @return List of smooth structures, one per term
#' @noRd
build_smooth_terms <- function(smooth_specs, data, encoding_vars = NULL, is_old_var = NULL) {
  if (!requireNamespace("mgcv", quietly = TRUE)) {
    stop("Package 'mgcv' is required for smooth terms (s(), t2()). ",
         "Install it with install.packages('mgcv').")
  }

  results <- list()

  for (i in seq_along(smooth_specs)) {
    term_str <- trimws(smooth_specs[i])

    # Evaluate the smooth specification in an environment with mgcv's s() and t2()
    sm_env <- new.env(parent = baseenv())
    sm_env$s <- mgcv::s
    sm_env$t2 <- mgcv::t2
    spec <- eval(parse(text = term_str), envir = sm_env)

    # Determine smooth function name
    sfun <- sub("\\s*\\(.*", "", term_str)

    # Check if this smooth involves encoding variables
    by_var_name <- if (!is.null(spec$by) && spec$by != "NA") spec$by else NULL
    smooth_vars <- c(spec$term, by_var_name)
    has_encoding <- !is.null(encoding_vars) && any(encoding_vars %in% smooth_vars)

    if (has_encoding && !is.null(is_old_var)) {
      # Build on old items only, then zero-pad for new items
      old_idx <- which(data[[is_old_var]] == 1)
      old_data <- data[old_idx, , drop = FALSE]
      # Drop unused factor levels in old subset
      for (v in smooth_vars) {
        if (!is.null(v) && v %in% names(old_data) && is.factor(old_data[[v]])) {
          old_data[[v]] <- droplevels(old_data[[v]])
        }
      }
      sm_data <- sm_prepare_data(spec, old_data)
    } else {
      sm_data <- sm_prepare_data(spec, data)
    }

    # Build basis via mgcv::smoothCon
    sm <- mgcv::smoothCon(
      spec, data = sm_data,
      knots = NULL,
      absorb.cons = TRUE,
      diagonal.penalty = TRUE
    )

    # sm is a list -- multiple elements when by-variable is a factor
    # Use the base label (first element) for the overall smooth
    label <- sm[[1]]$label
    # For by-factor smooths, each component has its own label (e.g., s(age):condA)
    # so bylevel need not be appended separately
    san_label <- sanitize_smooth_label(label)
    covars <- spec$term
    by_var <- if (!is.null(spec$by) && spec$by != "NA") spec$by else NA

    # Determine by-levels
    bylevels <- NULL
    if (!is.na(by_var)) {
      bls <- vapply(sm, function(s) {
        if (length(s$by.level)) s$by.level else NA_character_
      }, character(1))
      if (!all(is.na(bls))) {
        bylevels <- bls[!is.na(bls)]
      }
    }

    # Process each component (one per by-level, or one if no by-factor)
    N_full <- nrow(data)
    components <- list()
    for (j in seq_along(sm)) {
      re <- mgcv::smooth2random(sm[[j]], names(sm_data), type = 2)

      # Unpenalized part (null space of penalty)
      Xs <- re$Xf
      if (NCOL(Xs) > 0) {
        # Generate column names
        comp_label <- sm[[j]]$label
        Xs_colnames <- paste0("sXs", comp_label, "_", seq_len(ncol(Xs)))
        colnames(Xs) <- Xs_colnames
      } else {
        Xs_colnames <- character(0)
      }

      # Penalized parts (one Zs matrix per penalty component)
      Zs_list <- re$rand
      Zs_dims <- vapply(Zs_list, ncol, integer(1))
      if (length(Zs_list) == 0) {
        Zs_dims <- integer(0)
      }

      # Zero-pad for encoding variables (expand old-only matrices to full N)
      if (has_encoding && !is.null(is_old_var)) {
        if (NCOL(Xs) > 0) {
          Xs_full <- matrix(0, nrow = N_full, ncol = ncol(Xs))
          colnames(Xs_full) <- Xs_colnames
          Xs_full[old_idx, ] <- Xs
          Xs <- Xs_full
        }
        for (zi in seq_along(Zs_list)) {
          Zs_full <- matrix(0, nrow = N_full, ncol = ncol(Zs_list[[zi]]))
          Zs_full[old_idx, ] <- Zs_list[[zi]]
          Zs_list[[zi]] <- Zs_full
        }
      }

      bylevel <- if (length(sm[[j]]$by.level)) sm[[j]]$by.level else NULL
      # Each component gets its own sanitized label (includes bylevel for by-factor smooths)
      comp_san_label <- sanitize_smooth_label(sm[[j]]$label)

      components[[j]] <- list(
        bylevel = bylevel,
        san_label = comp_san_label,
        Xs = Xs,
        Xs_colnames = Xs_colnames,
        Zs_list = Zs_list,
        Zs_dims = Zs_dims,
        sm_obj = sm[[j]],
        re_obj = re
      )
    }

    results[[i]] <- list(
      term_str = term_str,
      label = label,
      san_label = san_label,
      sfun = sfun,
      covars = covars,
      by = by_var,
      bylevels = bylevels,
      components = components
    )
  }

  results
}


#' Prepare Data for smoothCon
#'
#' Strips terms attribute and ensures factor-like variables are proper factors.
#'
#' @param spec A smooth specification object (from mgcv::s() or mgcv::t2())
#' @param data Data frame
#' @return Cleaned data frame
#' @noRd
sm_prepare_data <- function(spec, data) {
  data <- data
  attr(data, "terms") <- NULL
  vars <- setdiff(c(spec$term, spec$by), "NA")
  for (v in vars) {
    if (v %in% names(data)) {
      if (is.character(data[[v]])) {
        data[[v]] <- as.factor(data[[v]])
      }
    }
  }
  data
}


#' Build Smooth Prediction Basis for New Data
#'
#' Uses cached smoothCon objects and smooth2random transformation info to
#' construct Xs and Zs matrices for new data.
#'
#' @param component A single smooth component (element of smooth_info$components)
#' @param newdata Data frame of new observations
#' @return List with Xs (unpenalized matrix) and Zs_list (list of penalized matrices)
#' @noRd
build_smooth_prediction_basis <- function(component, newdata) {
  sm_obj <- component$sm_obj
  re_obj <- component$re_obj

  # Get full prediction basis for new data
  X <- mgcv::PredictMat(sm_obj, newdata)

  # Apply smooth2random transformation
  if (!is.null(re_obj$trans.U)) {
    X <- X %*% re_obj$trans.U
  }

  if (is.null(re_obj$trans.D)) {
    # No penalization -- everything is unpenalized
    Xs <- X
    Zs_list <- list()
  } else {
    # Scale columns
    X <- t(t(X) * re_obj$trans.D)

    # Reorder columns
    X[, re_obj$rind] <- X[, re_obj$pen.ind != 0]
    pen.ind <- re_obj$pen.ind
    pen.ind[re_obj$rind] <- pen.ind[pen.ind > 0]

    # Separate fixed and random parts
    Xs <- X[, which(re_obj$pen.ind == 0), drop = FALSE]
    Zs_list <- list()
    for (k in seq_along(re_obj$rand)) {
      Zs_list[[k]] <- X[, which(pen.ind == k), drop = FALSE]
      attr(Zs_list[[k]], "s.label") <- attr(re_obj$rand[[k]], "s.label")
    }
    names(Zs_list) <- names(re_obj$rand)
  }

  # Preserve column names
  if (NCOL(Xs) > 0) {
    colnames(Xs) <- component$Xs_colnames
  }

  list(Xs = Xs, Zs_list = Zs_list)
}


#' Penalized smooth basis for newdata rows in a NEW by-level (unseen in training)
#'
#' Reuses a reference component's basis (the smooth is the same function of the
#' covariates; only the by-indicator differs) by temporarily assigning the
#' reference's trained by-level, then zeros rows outside `new_level`. Returns the
#' penalized Zs_list only; coefficients for the new level are sampled from the
#' prior by the caller.
#' @noRd
build_new_bylevel_basis <- function(ref_component, newdata, by_var, new_level) {
  nd <- newdata
  lv <- union(levels(factor(nd[[by_var]])), ref_component$bylevel)
  nd[[by_var]] <- factor(ref_component$bylevel, levels = lv)
  Zs_list <- build_smooth_prediction_basis(ref_component, nd)$Zs_list
  keep <- as.character(newdata[[by_var]]) == new_level
  lapply(Zs_list, function(Z) { if (NROW(Z)) Z[!keep, ] <- 0; Z })
}
