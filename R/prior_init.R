#' @noRd
prior_init <- function(model, chains = 4, seed = NULL) {
  if (!inherits(model, "broc_model")) {
    stop("`model` must be a broc_model object built with broc().", call. = FALSE)
  }
  restore_seed <- .seed_scope(seed)
  on.exit(restore_seed(), add = TRUE)

  parsed <- .parse_stan_param_block(model$stan_code, model$stan_data)

  pl <- model$prior_lookup
  stmts <- .model_block_priors(model$stan_code, names(parsed))

  inits <- vector("list", chains)
  for (ch in seq_len(chains)) {
    init <- list()
    for (p in names(parsed)) {
      spec <- parsed[[p]]
      st <- stmts[[p]]
      prior_str <- if (!is.null(st$whole)) st$whole else .resolve_prior_for_param(p, spec, pl)
      v <- .sample_prior(prior_str, spec$shape, spec$type, spec$has_lower_zero)
      if (length(st$elem) && !spec$type %in% c("cholesky_factor_corr", "corr_matrix", "cov_matrix", "simplex",
                                                "ordered", "positive_ordered", "unit_vector", "cholesky_factor_cov")) {
        dims <- if (length(spec$shape)) spec$shape else 1L
        for (e in st$elem) {
          if (length(e$idx) != length(dims) || any(e$idx > dims)) next
          x <- .sample_univariate(e$prior, 1L)
          if (spec$has_lower_zero) x <- abs(x)
          k <- if (length(dims) == 1L) e$idx else e$idx[1] + (e$idx[2] - 1L) * dims[1]
          v[k] <- x
        }
      }
      init[[p]] <- v
    }
    inits[[ch]] <- init
  }
  inits
}

# Prior statements of the generated model block: a whole-variable statement (`x ~ d;`,
# `to_vector(x) ~ d;`) and element statements (`x[k] ~ d;`, `x[i, j] ~ d;`) per parameter
.model_block_priors <- function(stan_code, params) {
  L <- strsplit(stan_code, "\n")[[1]]
  s <- grep("^model\\s*\\{", L)
  if (!length(s)) return(list())
  L <- trimws(sub("//.*$", "", L[s[1]:length(L)]))
  pat <- "^(to_vector\\()?([A-Za-z_][A-Za-z0-9_]*)(\\[([0-9 ,]+)\\])?\\)?\\s*~\\s*(.+?)\\s*;$"
  out <- list()
  for (ln in L) {
    m <- regmatches(ln, regexec(pat, ln, perl = TRUE))[[1]]
    if (length(m) < 6 || !(m[3] %in% params)) next
    nm <- m[3]
    if (!nzchar(m[5])) {
      out[[nm]]$whole <- m[6]
    } else {
      idx <- as.integer(strsplit(gsub(" ", "", m[5]), ",")[[1]])
      out[[nm]]$elem[[length(out[[nm]]$elem) + 1L]] <- list(idx = idx, prior = m[6])
    }
  }
  out
}

# =========================================================================
# Stan param block parser
# =========================================================================

#' @noRd
.parse_stan_param_block <- function(stan_code, stan_data) {
  lines <- strsplit(stan_code, "\n")[[1]]

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

  body <- sub("//.*$", "", body)
  body <- trimws(body)
  body <- body[nzchar(body)]

  parsed <- list()
  for (decl in body) {
    decl <- sub(";\\s*$", "", decl)
    m <- regmatches(decl, regexec(
      "^(real|int|vector|row_vector|matrix|cholesky_factor_corr|cholesky_factor_cov|corr_matrix|cov_matrix|simplex|positive_ordered|ordered|unit_vector)(?:<[^>]*>)?(\\[[^\\]]+\\])?\\s+([A-Za-z_][A-Za-z0-9_]*)(\\[[^\\]]+\\])?$",
      decl, perl = TRUE))[[1]]
    if (length(m) < 4 || m[1] == "") next
    type_  <- m[2]
    sz_str <- if (nzchar(m[3])) m[3] else NULL
    name_  <- m[4]
    arr_str <- if (length(m) >= 5 && nzchar(m[5])) m[5] else NULL

    sz <- .resolve_size_expr(sz_str, td_ctx)
    arr <- .resolve_size_expr(arr_str, td_ctx)

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
  expr <- gsub("[\\[\\]]", "", sz_str, perl = TRUE)
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

  if (grepl("_raw$", stan_name) || grepl("^z_", stan_name)) {
    return("normal(0, 1)")
  }

  m <- regmatches(stan_name, regexec("^L_corr_(.+)_([A-Za-z0-9_]+)$", stan_name))[[1]]
  if (length(m) == 3) {
    dpar <- m[2]; group <- m[3]
    return(get_cor_prior(pl, dpar, group))
  }

  m <- regmatches(stan_name, regexec("^sigma_(.+)_([A-Za-z0-9_]+)$", stan_name))[[1]]
  if (length(m) == 3) {
    dpar <- m[2]; group <- m[3]
    return(get_sd_prior(pl, dpar, group))
  }

  if (startsWith(stan_name, "beta_")) {
    dpar <- sub("^beta_", "", stan_name)
    return(get_fixed_prior(pl, dpar))
  }

  "normal(0, 1)"
}

# =========================================================================
# Distribution samplers
# =========================================================================

#' @noRd
.sample_prior <- function(prior_str, shape, type_, has_lower_zero = FALSE) {
  n_elem <- if (length(shape) == 0) 1L else prod(shape)

  if (type_ == "cholesky_factor_corr") {
    eta <- .extract_lkj_eta(prior_str)
    K <- shape[1]
    return(.rlkj_cholesky(K, eta))
  }
  if (type_ %in% c("corr_matrix", "cov_matrix",
                    "simplex", "ordered", "positive_ordered", "unit_vector",
                    "cholesky_factor_cov")) {
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

  vals <- .sample_univariate(prior_str, n_elem)

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
  m <- regmatches(prior_str, regexec("^\\s*([a-zA-Z_]+)\\s*\\(\\s*([^)]*)\\s*\\)\\s*$",
                                       prior_str))[[1]]
  if (length(m) < 3) return(rnorm(n, 0, 1))
  fam <- m[2]
  args <- as.numeric(strsplit(m[3], ",")[[1]])
  switch(fam,
    normal      = rnorm(n, args[1], args[2]),
    std_normal  = rnorm(n),
    half_normal = abs(rnorm(n, 0, args[1])),
    student_t   = args[2] + args[3] * rt(n, df = args[1]),
    half_student_t = abs(args[2] * rt(n, df = args[1])),
    cauchy      = rcauchy(n, args[1], args[2]),
    half_cauchy = abs(rcauchy(n, 0, args[1])),
    logistic    = rlogis(n, args[1], args[2]),
    gumbel      = args[1] - args[2] * log(-log(runif(n))),
    double_exponential = ,
    laplace     = args[1] + args[2] * (rexp(n) - rexp(n)),
    lognormal   = rlnorm(n, args[1], args[2]),
    exponential = rexp(n, rate = args[1]),
    gamma       = rgamma(n, shape = args[1], rate = args[2]),
    inv_gamma   = 1 / rgamma(n, shape = args[1], rate = args[2]),
    weibull     = rweibull(n, shape = args[1], scale = args[2]),
    beta        = rbeta(n, args[1], args[2]),
    uniform     = runif(n, args[1], args[2]),
    stop("init = \"prior\" cannot draw starting values from ", fam, "(); use init = 0 or set them yourself.",
         call. = FALSE)
  )
}

#' @noRd
.extract_lkj_eta <- function(prior_str) {
  m <- regmatches(prior_str, regexec("lkj_corr_cholesky\\(\\s*([0-9.eE+-]+)\\s*\\)",
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
  beta_param <- eta + (K - 2) / 2
  r <- 2 * rbeta(1, beta_param, beta_param) - 1
  L[2, 1] <- r
  L[2, 2] <- sqrt(1 - r^2)
  if (K == 2) return(L)
  for (m in 3:K) {
    beta_param <- beta_param - 0.5
    y <- rbeta(1, (m - 1) / 2, beta_param)
    u <- rnorm(m - 1)
    u <- u / sqrt(sum(u^2))
    L[m, 1:(m - 1)] <- sqrt(y) * u
    L[m, m] <- sqrt(1 - y)
  }
  L
}
