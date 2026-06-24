# Prior-based initial values for fit_broc(init = "prior"): one draw per chain
# from each parameter's prior, returned as per-chain init lists. Scale-
# appropriate for the log-linked threshold gaps and RE SDs, where cmdstanr's
# U(-2, 2) "random" init can blow up after exp(). <lower=0> parameters are drawn
# from the positive half; non-centred z's from N(0, 1).
#' @noRd
prior_init <- function(model, chains = 4, seed = NULL) {
  if (!inherits(model, "broc_model")) {
    stop("`model` must be a broc_model object built with broc().", call. = FALSE)
  }
  if (!is.null(seed)) set.seed(seed)

  # Parse Stan param block + transformed-data block directly from the generated
  # Stan code string. NO compilation needed: shapes are recoverable from the
  # source plus `stan_data` (cmdstanr's mod$variables() reports `dimensions =
  # number of dims`, not actual sizes, so it wouldn't help anyway). Avoiding
  # the compile here saves ~20-60 s per call on typical batch-likelihood models
  # since fit_broc() will compile the same code path again for sampling.
  parsed <- .parse_stan_param_block(model$stan_code, model$stan_data)

  # Map each Stan param to a prior string via model$prior_lookup
  pl <- model$prior_lookup

  inits <- vector("list", chains)
  for (ch in seq_len(chains)) {
    init <- list()
    for (p in names(parsed)) {
      spec <- parsed[[p]]
      prior_str <- .resolve_prior_for_param(p, spec, pl)
      init[[p]] <- .sample_prior(prior_str, spec$shape, spec$type,
                                  spec$has_lower_zero)
    }
    inits[[ch]] <- init
  }
  inits
}


# =========================================================================
# Stan param block parser
# =========================================================================

#' @noRd
.parse_stan_param_block <- function(stan_code, stan_data) {
  lines <- strsplit(stan_code, "\n")[[1]]

  # Build extended size context by parsing `transformed data` block. Common
  # bayesroc declarations like `int D_criterion_subj = K - 1` and
  # `int mid_thresh = ...` aren't in stan_data but are needed to resolve
  # parameter dimensions.
  td_ctx <- list2env(stan_data, parent = baseenv())
  td_start <- grep("^transformed data\\s*\\{", lines)
  if (length(td_start) > 0) {
    td_start <- td_start[1]
    td_end <- td_start + which(grepl("^\\}", lines[td_start:length(lines)]))[1] - 1L
    td_body <- lines[(td_start + 1):(td_end - 1)]
    td_body <- sub("//.*$", "", td_body)
    td_body <- trimws(td_body)
    td_body <- td_body[nzchar(td_body)]
    for (decl in td_body) {
      decl <- sub(";\\s*$", "", decl)
      # Pattern: int [<...>] NAME = EXPR
      m <- regmatches(decl, regexec(
        "^int(?:<[^>]*>)?\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*=\\s*(.+)$",
        decl, perl = TRUE))[[1]]
      if (length(m) >= 3) {
        nm <- m[2]; expr <- m[3]
        val <- tryCatch(eval(parse(text = expr), envir = td_ctx),
                        error = function(e) NULL)
        if (!is.null(val) && is.numeric(val)) {
          assign(nm, as.integer(val), envir = td_ctx)
        }
      }
    }
  }

  start <- grep("^parameters\\s*\\{", lines)[1]
  end   <- start + which(grepl("^\\}", lines[start:length(lines)]))[1] - 1L
  body  <- lines[(start + 1):(end - 1)]

  # Pattern: TYPE[constraints] NAME[dims]; or vector[N] NAME; etc.
  # Strip comments and trim
  body <- sub("//.*$", "", body)
  body <- trimws(body)
  body <- body[nzchar(body)]

  parsed <- list()
  for (decl in body) {
    decl <- sub(";\\s*$", "", decl)
    # Match: TYPE [<...>] [size] NAME [array dims]
    m <- regmatches(decl, regexec(
      "^(real|int|vector|row_vector|matrix|cholesky_factor_corr|cholesky_factor_cov|corr_matrix|cov_matrix|simplex|positive_ordered|ordered|unit_vector)(?:<[^>]*>)?(\\[[^\\]]+\\])?\\s+([A-Za-z_][A-Za-z0-9_]*)(\\[[^\\]]+\\])?$",
      decl, perl = TRUE))[[1]]
    if (length(m) < 4 || m[1] == "") next  # skip unparseable lines
    type_  <- m[2]
    sz_str <- if (nzchar(m[3])) m[3] else NULL
    name_  <- m[4]
    arr_str <- if (length(m) >= 5 && nzchar(m[5])) m[5] else NULL

    # Resolve numeric size from Stan data + transformed-data context
    sz <- .resolve_size_expr(sz_str, td_ctx)
    arr <- .resolve_size_expr(arr_str, td_ctx)

    # Build shape vector
    shape <- switch(type_,
      real      = if (is.null(arr)) integer(0) else as.integer(arr),
      int       = if (is.null(arr)) integer(0) else as.integer(arr),
      vector    = c(as.integer(sz), if (!is.null(arr)) as.integer(arr) else NULL),
      row_vector = c(as.integer(sz), if (!is.null(arr)) as.integer(arr) else NULL),
      matrix    = as.integer(sz),
      cholesky_factor_corr = c(as.integer(sz), as.integer(sz)),
      cholesky_factor_cov  = c(as.integer(sz), as.integer(sz)),
      corr_matrix          = c(as.integer(sz), as.integer(sz)),
      cov_matrix           = c(as.integer(sz), as.integer(sz)),
      simplex   = as.integer(sz),
      ordered   = as.integer(sz),
      positive_ordered = as.integer(sz),
      unit_vector = as.integer(sz),
      integer(0)
    )

    # Detect lower=0 constraint
    has_lower_zero <- grepl("<.*lower\\s*=\\s*0", decl)

    parsed[[name_]] <- list(
      type           = type_,
      shape          = shape,
      has_lower_zero = has_lower_zero,
      decl           = decl
    )
  }
  parsed
}

#' @noRd
.resolve_size_expr <- function(sz_str, ctx) {
  if (is.null(sz_str) || !nzchar(sz_str)) return(NULL)
  # Strip brackets
  expr <- gsub("[\\[\\]]", "", sz_str, perl = TRUE)
  # Split on commas -> multi-dim; for now only one or two dims handled
  parts <- strsplit(expr, ",")[[1]]
  envir <- if (is.environment(ctx)) ctx else list2env(ctx)
  vals <- vapply(parts, function(part) {
    part <- trimws(part)
    e <- tryCatch(eval(parse(text = part), envir = envir),
                  error = function(err) NA_integer_)
    if (is.na(e)) NA_integer_ else as.integer(e)
  }, integer(1), USE.NAMES = FALSE)
  if (any(is.na(vals))) return(NULL)
  vals
}


# =========================================================================
# Prior lookup
# =========================================================================

#' @noRd
.resolve_prior_for_param <- function(stan_name, spec, pl) {
  # Stan param names follow conventions:
  #   beta_<dpar>           -> fixed-effect prior (class "b")
  #   sigma_<dpar>_<group>  -> RE SD prior (class "sd")
  #   L_corr_<dpar>_<group> -> correlation prior (class "cor")
  #   u_<dpar>_<group>_raw  -> std_normal (always -- non-centered z's)
  #   z_<dpar>_<group>      -> std_normal (always)
  #
  # Uses the package's get_fixed_prior / get_sd_prior / get_cor_prior helpers
  # which handle hierarchical resolution (coef > group > dpar > default).

  # Non-centered z-scores
  if (grepl("_raw$", stan_name) || grepl("^z_", stan_name)) {
    return("normal(0, 1)")
  }

  # L_corr_<dpar>_<group> -> LKJ Cholesky prior
  m <- regmatches(stan_name, regexec("^L_corr_(.+)_([A-Za-z0-9_]+)$", stan_name))[[1]]
  if (length(m) == 3) {
    dpar <- m[2]; group <- m[3]
    return(get_cor_prior(pl, dpar, group))
  }

  # sigma_<dpar>_<group> -> RE SD prior
  m <- regmatches(stan_name, regexec("^sigma_(.+)_([A-Za-z0-9_]+)$", stan_name))[[1]]
  if (length(m) == 3) {
    dpar <- m[2]; group <- m[3]
    return(get_sd_prior(pl, dpar, group))
  }

  # beta_<dpar> -> fixed effect prior
  if (startsWith(stan_name, "beta_")) {
    dpar <- sub("^beta_", "", stan_name)
    return(get_fixed_prior(pl, dpar))
  }

  # Fallback: weakly informative
  "normal(0, 1)"
}


# =========================================================================
# Distribution samplers
# =========================================================================

#' @noRd
.sample_prior <- function(prior_str, shape, type_, has_lower_zero = FALSE) {
  # Total element count
  n_elem <- if (length(shape) == 0) 1L else prod(shape)

  # Special types -- sample whole objects
  if (type_ == "cholesky_factor_corr") {
    eta <- .extract_lkj_eta(prior_str)
    K <- shape[1]
    return(.rlkj_cholesky(K, eta))
  }
  if (type_ %in% c("corr_matrix", "cov_matrix",
                    "simplex", "ordered", "positive_ordered", "unit_vector",
                    "cholesky_factor_cov")) {
    # Fall back to safe default value -- not commonly used in bayesroc.
    K <- shape[1]
    return(switch(type_,
      corr_matrix = diag(K),
      cov_matrix = diag(K),
      cholesky_factor_cov = diag(K),
      simplex = rep(1/K, K),
      ordered = seq_len(K) - (K+1)/2,
      positive_ordered = seq_len(K),
      unit_vector = c(1, rep(0, K - 1))
    ))
  }

  # Univariate samplers
  vals <- .sample_univariate(prior_str, n_elem)

  # Enforce <lower=0> constraint by reflecting around 0 (half-normal etc.).
  # This matches Stan's implicit treatment: when a `<lower=0>` parameter has
  # `normal(0, sigma)` prior, the effective distribution is half-normal (the
  # density is normalized over [0, Inf)).
  if (has_lower_zero) {
    vals <- abs(vals)
  }

  if (length(shape) == 0) {
    as.numeric(vals[1])
  } else if (length(shape) == 1) {
    as.numeric(vals)
  } else if (length(shape) == 2) {
    matrix(vals, nrow = shape[1], ncol = shape[2])
  } else {
    array(vals, dim = shape)
  }
}

#' @noRd
.sample_univariate <- function(prior_str, n) {
  # Parse "name(arg1, arg2, ...)"
  m <- regmatches(prior_str, regexec("^\\s*([a-zA-Z_]+)\\s*\\(\\s*([^)]*)\\s*\\)\\s*$",
                                       prior_str))[[1]]
  if (length(m) < 3) return(rnorm(n, 0, 1))  # unknown -> standard normal
  fam <- m[2]
  args <- as.numeric(strsplit(m[3], ",")[[1]])
  switch(fam,
    normal      = rnorm(n, args[1], args[2]),
    half_normal = abs(rnorm(n, 0, args[1])),
    student_t   = args[2] + args[3] * rt(n, df = args[1]),
    cauchy      = rcauchy(n, args[1], args[2]),
    exponential = rexp(n, rate = args[1]),
    gamma       = rgamma(n, shape = args[1], rate = args[2]),
    inv_gamma   = 1 / rgamma(n, shape = args[1], rate = args[2]),
    beta        = rbeta(n, args[1], args[2]),
    uniform     = runif(n, args[1], args[2]),
    rnorm(n, 0, 1)  # default fallback
  )
}

#' @noRd
.extract_lkj_eta <- function(prior_str) {
  m <- regmatches(prior_str, regexec("lkj_corr_cholesky\\(\\s*([0-9.]+)\\s*\\)",
                                       prior_str))[[1]]
  if (length(m) >= 2) as.numeric(m[2]) else 1.0
}


# =========================================================================
# Onion-method LKJ Cholesky sampler (Lewandowski-Kurowicka-Joe 2009)
# =========================================================================

#' Sample an LKJ Cholesky factor via the onion method
#'
#' @param K dimension of the K x K correlation matrix
#' @param eta LKJ concentration parameter (eta > 0)
#' @return Lower-triangular Cholesky factor L such that L %*% t(L) is a valid
#'   correlation matrix drawn from LKJ(eta).
#' @noRd
.rlkj_cholesky <- function(K, eta = 1) {
  if (K <= 1) return(matrix(1, K, K))
  L <- matrix(0, K, K)
  L[1, 1] <- 1
  # First off-diagonal: r_21 ~ marginal Beta-on-(-1,1)
  beta_param <- eta + (K - 2) / 2
  r <- 2 * rbeta(1, beta_param, beta_param) - 1
  L[2, 1] <- r
  L[2, 2] <- sqrt(1 - r^2)
  if (K == 2) return(L)
  for (m in 3:K) {
    beta_param <- beta_param - 0.5
    # Squared distance from origin
    y <- rbeta(1, (m - 1) / 2, beta_param)
    # Uniform direction on unit (m-1)-sphere
    u <- rnorm(m - 1)
    u <- u / sqrt(sum(u^2))
    L[m, 1:(m - 1)] <- sqrt(y) * u
    L[m, m] <- sqrt(1 - y)
  }
  L
}
