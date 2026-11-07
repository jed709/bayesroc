# =============================================================================
# Data coding: missing values and the coding of response and indicator columns
# =============================================================================

.resp_vars <- function(brf_obj) {
  lhs <- brf_obj$primary
  if (is.null(lhs) || length(lhs) < 3L) return(character(0))
  s <- paste(deparse(lhs[[2]], width.cutoff = 500L), collapse = " ")
  m <- regmatches(s, regexec("resp\\(\\s*([^,]+?)\\s*,\\s*([^)]+?)\\s*\\)", s))[[1]]
  if (length(m) == 3L) m[2:3] else character(0)
}

# factor levels with no rows (left over from subsetting) would otherwise become
# prior-only coefficients and empty random-effect groups
.drop_unused_levels <- function(data, brf_obj, encoding_vars) {
  vars <- unique(c(unlist(lapply(c(list(brf_obj$primary), brf_obj$params), function(f) {
    if (is.null(f)) character(0) else tryCatch(all.vars(f), error = function(e) character(0))
  })), encoding_vars))
  scale_vars <- .scale_resp_vars(brf_obj)
  for (v in intersect(vars, names(data))) {
    x <- data[[v]]
    if (is.ordered(x) && v %in% scale_vars) next
    if (is.factor(x) && !all(levels(x) %in% as.character(unique(x)))) data[[v]] <- droplevels(x)
  }
  data
}

.scale_resp_vars <- function(brf_obj) {
  rv <- .resp_vars(brf_obj)
  if (length(rv)) return(rv)
  lhs <- brf_obj$primary
  if (is.null(lhs) || length(lhs) < 3L) return(character(0))
  e <- lhs[[2]]
  if (is.call(e) && identical(e[[1]], as.name("|"))) e <- e[[2]]
  all.vars(e)
}

.drop_incomplete_rows <- function(data, brf_obj, family, encoding_vars, counts) {
  vars <- unique(c(unlist(lapply(c(list(brf_obj$primary), brf_obj$params), function(f) {
    if (is.null(f)) character(0) else tryCatch(all.vars(f), error = function(e) character(0))
  })), encoding_vars, counts))
  exempt <- character(0)
  rv <- .resp_vars(brf_obj)
  if (length(rv) == 2L && family$family %in% c("cdp", "bivariate_sdt", "bivariate_dp", "vrdp2d", "bivariate_cumulative")) exempt <- rv[2]
  vars <- setdiff(intersect(vars, names(data)), exempt)
  if (!length(vars)) return(list(data = data, dropped = integer(0)))
  ok <- stats::complete.cases(data[vars])
  if (all(ok)) return(list(data = data, dropped = integer(0)))
  bad_vars <- vars[vapply(vars, function(v) anyNA(data[[v]]), logical(1))]
  dropped <- which(!ok)
  message("Dropping ", length(dropped), " row(s) with missing values in: ", paste(bad_vars, collapse = ", "),
          ". Their row numbers are stored in model$dropped_rows.")
  if (!any(ok)) stop("Every row has a missing value in a variable the model uses.", call. = FALSE)
  out <- data[ok, , drop = FALSE]
  rownames(out) <- NULL
  list(data = out, dropped = dropped)
}

.code_response <- function(v, name) {
  if (is.ordered(v)) {
    return(list(y = as.integer(v), K = nlevels(v), labels = levels(v), offset = 0))
  }
  if (is.factor(v) || is.character(v)) {
    ch <- as.character(v)
    num <- suppressWarnings(as.numeric(ch))
    if (any(is.na(num) & !is.na(ch))) {
      stop("Response '", name, "' has text labels (", paste(utils::head(unique(ch[is.na(num) & !is.na(ch)]), 4), collapse = ", "),
           "). Use integer codes, or an ordered factor whose level order is the response scale ",
           "(e.g. factor(x, levels = c(...), ordered = TRUE)).", call. = FALSE)
    }
    v <- num
  }
  if (is.logical(v)) v <- as.integer(v)
  if (!is.numeric(v)) stop("Response '", name, "' must be integer codes or an ordered factor.", call. = FALSE)
  obs <- v[!is.na(v)]
  if (any(abs(obs - round(obs)) > 1e-8)) {
    stop("Response '", name, "' must be whole numbers (got e.g. ", obs[abs(obs - round(obs)) > 1e-8][1], ").",
         call. = FALSE)
  }
  lo <- min(obs); hi <- max(obs)
  if (lo > 1) {
    message("Response '", name, "' ranges from ", lo, " to ", hi, "; it is treated as ", hi - lo + 1,
            " categories (", lo, " is the lowest). If the scale starts below ", lo, ", that category is not represented.")
  }
  if (lo < 0) stop("Response '", name, "' must not be negative.", call. = FALSE)
  if (lo == 0) message("0-indexed responses detected (range 0..", hi, ") -- converting to 1..", hi + 1, ".")
  list(y = as.integer(round(v - lo + 1)), K = as.integer(hi - lo + 1), labels = as.character(lo:hi), offset = lo - 1)
}

.code_is_old <- function(v, name, allow_centered = TRUE) {
  if (is.logical(v)) return(as.numeric(v))
  num <- if (is.factor(v) || is.character(v)) suppressWarnings(as.numeric(as.character(v))) else v
  if (!is.numeric(num) || any(is.na(num) & !is.na(v))) {
    stop("The old/new indicator '", name, "' must be numeric 0/1 (new/old), logical",
         if (allow_centered) ", or centered -0.5/0.5" else "", "; ",
         "text labels such as ", paste(utils::head(unique(as.character(v)), 3), collapse = "/"),
         " are not accepted. Recode it, e.g. as.integer(x == \"old\").", call. = FALSE)
  }
  vals <- sort(unique(num[!is.na(num)]))
  ok01 <- all(vals %in% c(0, 1))
  okc <- allow_centered && all(vals %in% c(-0.5, 0.5))
  if (!ok01 && !okc) {
    stop("The old/new indicator '", name, "' must be coded 0/1 (new/old)",
         if (allow_centered) " or -0.5/0.5 (centered)" else "", ". Got: ", paste(utils::head(vals, 6), collapse = ", "),
         ".", call. = FALSE)
  }
  as.numeric(num)
}

.code_item_type <- function(v, name) {
  if (is.factor(v)) {
    v <- droplevels(v)
    lev <- levels(v)
    newish <- grepl("^(new|lure|lures|foil|foils|distractor|distractors|unstudied|nonstudied|non-studied|novel)$",
                    tolower(lev))
    if (nlevels(v) > 3L) stop("Item type '", name, "' has more than three levels.", call. = FALSE)
    if (any(newish) && !newish[1]) {
      stop("Item type '", name, "' is a factor whose first level is '", lev[1], "', but the level order must be ",
           "new, source A, source B (level '", lev[newish][1], "' looks like the new items). Reorder it, e.g. ",
           "factor(x, levels = c(\"", lev[newish][1], "\", ...)).", call. = FALSE)
    }
    return(as.integer(v))
  }
  if (is.character(v)) {
    stop("Item type '", name, "' is text; give it as codes 1/2/3 (new, source A, source B) or as a factor ",
         "with levels in that order.", call. = FALSE)
  }
  if (!is.numeric(v) || any(abs(v - round(v)) > 1e-8, na.rm = TRUE)) {
    stop("Item type '", name, "' must be codes 1/2/3 (new, source A, source B).", call. = FALSE)
  }
  vals <- sort(unique(v[!is.na(v)]))
  if (all(vals %in% 0:2) && 0 %in% vals) v <- v + 1
  else if (!all(vals %in% 1:3)) {
    stop("Item type '", name, "' must be codes 1/2/3 or 0/1/2 (new, source A, source B). Got: ",
         paste(vals, collapse = ", "), ".", call. = FALSE)
  }
  as.integer(v)
}

.code_source <- function(v, name) {
  if (is.factor(v)) {
    v <- droplevels(v)
    if (nlevels(v) != 2L) {
      stop("Source '", name, "' must have exactly two levels (source A first, source B second); got ",
           nlevels(v), ".", call. = FALSE)
    }
    return(as.integer(v) - 1L)
  }
  if (is.character(v)) {
    stop("Source '", name, "' is text; give it as a factor (first level = source A) or as codes 0/1 or 1/2.",
         call. = FALSE)
  }
  if (is.logical(v)) return(as.integer(v))
  vals <- sort(unique(v[!is.na(v)]))
  if (all(vals %in% c(0, 1))) return(as.integer(v))
  if (all(vals %in% c(1, 2))) return(as.integer(v) - 1L)
  stop("Source '", name, "' must be coded 0/1 or 1/2 (source A, source B). Got: ", paste(vals, collapse = ", "), ".",
       call. = FALSE)
}

.code_like_training <- function(new, train, name, coder) {
  if (is.factor(train)) {
    lev <- levels(droplevels(train))
    idx <- match(as.character(new), lev)
    if (any(is.na(idx) & !is.na(new))) {
      stop("'", name, "' in newdata has values not seen in the training data: ",
           paste(utils::head(unique(as.character(new)[is.na(idx) & !is.na(new)]), 4), collapse = ", "), ".",
           call. = FALSE)
    }
    return(coder(factor(lev[idx], levels = lev), name))
  }
  coder(new, name)
}

.code_position <- function(v, name) {
  num <- if (is.factor(v) || is.character(v)) suppressWarnings(as.numeric(as.character(v))) else as.numeric(v)
  if (any(is.na(num) & !is.na(v)) || any(abs(num - round(num)) > 1e-8, na.rm = TRUE) || any(num < 1, na.rm = TRUE)) {
    stop("'", name, "' must hold positions 1, 2, ..., m (whole numbers from 1).", call. = FALSE)
  }
  as.integer(round(num))
}

.code_mafc <- function(data, choice_var, target_var, alternatives) {
  if (is.null(target_var) || !target_var %in% names(data)) {
    stop("mafc() needs `choice | target ~ ...`: the target position is missing from the data.", call. = FALSE)
  }
  choice <- .code_position(data[[choice_var]], choice_var)
  target <- .code_position(data[[target_var]], target_var)
  if (is.null(alternatives)) {
    M <- max(choice, target)
    m <- rep(M, length(choice))
  } else {
    if (!alternatives %in% names(data)) {
      stop("Column '", alternatives, "' (mafc(alternatives = )) not found in the data.", call. = FALSE)
    }
    m <- .code_position(data[[alternatives]], alternatives)
    M <- max(m)
    bad <- which(choice > m | target > m)
    if (length(bad)) {
      stop("Row ", bad[1], ": choice ", choice[bad[1]], " / target ", target[bad[1]], " exceeds its ", m[bad[1]],
           " alternatives.", call. = FALSE)
    }
  }
  if (M < 2L || any(m < 2L)) stop("mafc() needs at least 2 alternatives per trial.", call. = FALSE)
  if (M - 1L > 64L) stop("mafc() supports up to 65 alternatives.", call. = FALSE)
  list(choice = choice, target = target, m = m, M = as.integer(M))
}
