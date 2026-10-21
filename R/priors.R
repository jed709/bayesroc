#' Set a Prior on an SDT Model Parameter
#'
#' Specify a prior for one class of parameters in a [broc()] model: a fixed
#' effect, a random-effect standard deviation, or a random-effect correlation.
#' Pass one or more `broc_prior()` calls to `broc(..., priors = ...)`; combine
#' several with [c()].
#'
#' Priors are matched from most to least specific: a `coef`-level prior overrides
#' a `group`-level one, which overrides a `dpar`-level one, which overrides the
#' global class default. Call [get_broc_prior()] on a [broc()] or [brf()] object
#' (with data) to see which coefficients exist and the current prior for each.
#'
#' Confidence thresholds are not set directly. They are reparameterized as a
#' central anchor (`thresh_mid`) plus strictly-positive, log-scale spacings
#' (`log_gaps`) between successive thresholds, which enforces ordering. Set
#' criterion priors via `dpar = "thresh_mid"` (overall criterion location) and
#' `dpar = "log_gaps"` (threshold spread). With `gap_link = "softplus"` the
#' `log_gaps` prior applies to the unconstrained gap parameters that the
#' softplus maps to spacings.
#'
#' The same names set the random-effect SDs of the criteria (`class = "sd"`):
#' `dpar = "thresh_mid"` for the midpoint SD, `dpar = "log_gaps"` for the gap
#' SDs, and `coef = "gap2"` for one gap (`"gap2_x"` for one random-slope term).
#' Gaps are numbered as in the fixed effects: outward from the midpoint, upper
#' gaps first, then lower ones. With varying source criteria, a component prior
#' applies to that component in every detection bin. `dpar = "criterion"` (or
#' `"criterion2"`) still sets every criterion SD at once; component priors
#' override it.
#'
#' @param prior The prior as a distribution call, e.g. `"normal(0, 1)"`,
#'   `"student_t(3, 0, 2.5)"`, or `"lkj(2)"`. Supported on both backends:
#'   `normal`, `std_normal`, `student_t`, `cauchy`, `logistic`, `gumbel`,
#'   `double_exponential` (= `laplace`), `uniform`, `beta`, `exponential`,
#'   `gamma`, `inv_gamma`, `weibull`, `lognormal`, the half forms `half_normal`,
#'   `half_cauchy`, `half_student_t`, and `lkj` (`class = "cor"` only). For SD
#'   priors (`class = "sd"`/`"sds"`) a symmetric distribution is automatically
#'   restricted to positive values. The Stan backend additionally accepts any
#'   distribution in the Stan language.
#' @param class Parameter class: `"b"` (fixed effects), `"sd"` (random-effect
#'   SDs), `"cor"` (random-effect correlations), or `"sds"` (smooth penalty SDs,
#'   only for models with `s()`/`t2()` smooth terms).
#' @param dpar Distributional parameter the prior applies to, e.g. `"dprime"`,
#'   `"sigma"`, `"lambda"`. For criterion priors use `"thresh_mid"` / `"log_gaps"`
#'   (`"thresh_mid2"` / `"log_gaps2"` for the second response axis in bivariate
#'   families). Family-specific aliases (`cutpoints`, `cutpoints1`,
#'   `cutpoints2`, `cutpoints_scale`, `rec`, `fam`, `sigma_R`, `sigma_F`) are also
#'   accepted. The correlation prior of a cross-parameter group `(... | id | g)` is
#'   `dpar = "cross_<id>"`. `NULL` applies the prior to every parameter in
#'   `class`. [get_broc_prior()] lists every settable prior with its `dpar` and
#'   `coef`.
#' @param coef Optional specific coefficient name (e.g., `"conditionread"`). For
#'   criterion gaps, `"gap<k>"` or `"gap<k>_<term>"`.
#' @param group Optional grouping factor for random effects (e.g., `"participant"`)
#' @return A `broc_prior` object. Combine several with [c()] and pass to [broc()].
#' @seealso [get_broc_prior()]
#'
#' @examples
#' # Prior on all d' fixed effects
#' broc_prior("normal(1.5, 0.75)", class = "b", dpar = "dprime")
#' 
#' # Prior on specific d' coefficient
#' broc_prior("normal(2, 0.5)", class = "b", dpar = "dprime", coef = "conditionread")
#' 
#' # Criterion SDs by component: midpoint, all gaps, and one source gap
#' broc_prior("normal(0, 0.4)", class = "sd", dpar = "thresh_mid", group = "participant")
#' broc_prior("normal(0, 0.2)", class = "sd", dpar = "log_gaps", group = "participant")
#' broc_prior("normal(0, 0.1)", class = "sd", dpar = "log_gaps2", coef = "gap2")
#'
#' # Prior on every random-effect standard deviation
#' broc_prior("normal(0, 0.5)", class = "sd")
#' 
#' # Prior on random effect SDs for d' only
#' broc_prior("normal(0, 0.5)", class = "sd", dpar = "dprime")
#' 
#' # Prior on random effect SDs for d' for participant group only
#' broc_prior("normal(0, 0.3)", class = "sd", dpar = "dprime", group = "participant")
#' 
#' # Prior on every correlation
#' broc_prior("lkj(2)", class = "cor")
#' 
#' # Prior on criterion correlations for participant
#' broc_prior("lkj(4)", class = "cor", dpar = "criterion", group = "participant")
#' @export
broc_prior <- function(prior, class, dpar = NULL, coef = NULL, group = NULL) {
  
  valid_classes <- c("b", "sd", "cor", "sds")
  if (!class %in% valid_classes) {
    stop("Invalid class '", class, "'. Valid classes: ", 
         paste(valid_classes, collapse = ", "))
  }
  
  if (!is.null(dpar)) {
    valid_dpars <- c("dprime", "sigma", "lambda", "criterion", "dprime2", "sigma2",
                     "dprime_B", "lambda_B",
                     "discrim", "discrim_B", "sigma_B", "sigma2_B",
                     "rho", "rho_B", "rho_N", "criterion2",
                     "dprime_L", "sigma_L", "lambda_L",
                     "lambda2", "lambda2_B",
                     "rec", "fam", "sigma_R", "sigma_F", "rec_crit", "know_crit",
                     "thresh_mid", "log_gaps", "thresh_mid2", "log_gaps2",
                     "mu", "mu1", "mu2", "cutpoints", "cutpoints1", "cutpoints2",
                     "criterion_scale", "cutpoints_scale", "criterion2_scale", "cutpoints2_scale")
    if (!dpar %in% valid_dpars && !grepl("^cross_[A-Za-z0-9_.]+$", dpar)) {
      stop("Invalid dpar '", dpar, "'. Valid dpars: ",
           paste(valid_dpars, collapse = ", "))
    }

    if (dpar == "mu" || dpar == "mu1") dpar <- "dprime"
    if (dpar == "mu2") dpar <- "discrim"
    if (dpar == "cutpoints" || dpar == "cutpoints1") dpar <- "criterion"
    if (dpar == "cutpoints2") dpar <- "criterion2"
    if (dpar == "cutpoints_scale") dpar <- "criterion_scale"
    if (dpar == "cutpoints2_scale") dpar <- "criterion2_scale"

    if (dpar == "rec") dpar <- "dprime"
    if (dpar == "fam") dpar <- "dprime2"
    if (dpar == "sigma_R") dpar <- "sigma"
    if (dpar == "sigma_F") dpar <- "sigma2"
  }
  
  if (class == "cor") prior <- sub("^\\s*lkj(_corr)?\\s*\\(", "lkj_corr_cholesky(", prior)

  if (!is.null(coef) && !class %in% c("b", "sd", "sds")) {
    warning("coef argument is ignored for class '", class, "'")
    coef <- NULL
  }
  
  if (!is.null(group) && class == "b") {
    warning("group argument is ignored for class 'b'")
    group <- NULL
  }
  
  prior_info <- parse_prior_string(prior)
  
  structure(
    list(
      prior = prior,
      class = class,
      dpar = dpar,
      coef = coef,
      group = group,
      distribution = prior_info$distribution,
      args = prior_info$args
    ),
    class = "broc_prior"
  )
}

#' Parse a Prior String
#'
#' @param prior_str Prior specification string (e.g., "normal(0, 1)")
#' @return List with distribution name and arguments
#' @noRd
parse_prior_string <- function(prior_str) {
  if (prior_str == "" || is.na(prior_str)) {
    return(list(distribution = "flat", args = list()))
  }
  
  match <- regmatches(prior_str, regexec("^([a-zA-Z_]+)\\((.*)\\)$", prior_str))[[1]]
  
  if (length(match) != 3) {
    stop("Invalid prior specification: '", prior_str, 
         "'. Expected format: distribution(args)")
  }
  
  distribution <- match[2]
  args_str <- match[3]
  
  if (args_str == "") {
    args <- list()
  } else {
    args <- tryCatch(
      eval(parse(text = paste0("list(", args_str, ")"))),
      error = function(e) {
        stop("Could not parse prior arguments: '", args_str, "'")
      }
    )
  }
  
  list(distribution = distribution, args = args)
}

#' Combine Multiple Priors
#'
#' @param ... Prior specifications from broc_prior(), or lists of them made by
#'   earlier calls to `c()`.
#' @return A list of prior specifications
#' @export
c.broc_prior <- function(...) {
  parts <- lapply(list(...), function(p) if (inherits(p, "broc_prior_list")) unclass(p) else list(p))
  priors <- do.call(c, parts)
  bad <- !vapply(priors, inherits, logical(1), "broc_prior")
  if (any(bad)) stop("c() combines priors made with broc_prior().", call. = FALSE)
  class(priors) <- "broc_prior_list"
  priors
}

#' @rdname c.broc_prior
#' @export
c.broc_prior_list <- c.broc_prior

#' Print Method for Prior
#' @param x The object to print.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @export
print.broc_prior <- function(x, ...) {
  cat("Prior:", x$prior, "\n")
  cat("  class:", x$class)
  if (!is.null(x$dpar)) cat(", dpar:", x$dpar)
  if (!is.null(x$coef)) cat(", coef:", x$coef)
  if (!is.null(x$group)) cat(", group:", x$group)
  cat("\n")
  invisible(x)
}

#' Print Method for Prior List
#' @param x The object to print.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @export
print.broc_prior_list <- function(x, ...) {
  cat("Prior specifications:\n")
  for (i in seq_along(x)) {
    cat("\n[", i, "] ", sep = "")
    print(x[[i]])
  }
  invisible(x)
}

#' Build Prior Lookup Structure
#'
#' Converts user priors into a lookup structure keyed by specificity; more
#' specific priors (with `coef`) override less specific ones.
#'
#' @param user_priors List of broc_prior objects or single broc_prior
#' @return A prior_lookup object
#' @noRd
build_prior_lookup <- function(user_priors, family = NULL) {
  
  lookup <- list(
    b = list(
      dprime = "normal(1.5, 0.75)",
      sigma = "normal(0, 0.5)",
      lambda = "normal(0, 1.5)",
      dprime2 = "normal(0, 1)",
      sigma2 = "normal(0, 0.5)",
      dprime_B = "normal(-1, 1)",
      lambda_B = "normal(0, 1.5)",
      lambda2 = "normal(0, 1.5)",
      lambda2_B = "normal(0, 1.5)",
      discrim = "normal(0, 1)",
      discrim_B = "normal(0, 1)",
      sigma_B = "normal(0, 0.5)",
      sigma2_B = "normal(0, 0.5)",
      rho = "normal(0, 0.5)",
      rho_B = "normal(0, 0.5)",
      rho_N = "normal(0, 0.5)",
      rec_crit = "normal(2, 1)",
      know_crit = "normal(0, 1)",
      dprime_L = "normal(0, 1)",
      sigma_L = "normal(0, 0.5)",
      lambda_L = "normal(0, 1.5)",
      thresh_mid = "normal(0.5, 0.75)",
      log_gaps = "normal(-0.5, 0.5)",
      thresh_mid2 = "normal(0, 0.75)",
      log_gaps2 = "normal(0, 0.75)",
      criterion_scale = "normal(0, 0.5)",
      criterion2_scale = "normal(0, 0.5)"
    ),
    b_coef = list(),
    
    sd = "normal(0, 0.5)",
    sd_dpar = list(
      dprime = "normal(0, 0.5)",
      sigma = "normal(0, 0.3)",
      lambda = "normal(0, 0.5)",
      dprime2 = "normal(0, 0.5)",
      sigma2 = "normal(0, 0.3)",
      dprime_B = "normal(0, 0.5)",
      lambda_B = "normal(0, 0.5)",
      lambda2 = "normal(0, 0.5)",
      lambda2_B = "normal(0, 0.5)",
      discrim = "normal(0, 0.5)",
      discrim_B = "normal(0, 0.5)",
      sigma_B = "normal(0, 0.3)",
      sigma2_B = "normal(0, 0.3)",
      rho = "normal(0, 0.3)",
      rho_B = "normal(0, 0.3)",
      rho_N = "normal(0, 0.3)",
      rec_crit = "normal(0, 0.5)",
      know_crit = "normal(0, 0.3)",
      dprime_L = "normal(0, 0.5)",
      sigma_L = "normal(0, 0.3)",
      lambda_L = "normal(0, 0.5)",
      criterion = "normal(0, 0.5)",
      criterion2 = "normal(0, 0.5)",
      criterion_scale = "normal(0, 0.3)",
      criterion2_scale = "normal(0, 0.3)"
    ),
    sd_group = list(),
    sd_dpar_group = list(),
    sd_dpar_coef = list(),
    sd_dpar_group_coef = list(),
    sd_group_coef = list(),
    sd_coef = list(),
    
    cor = "lkj_corr_cholesky(1)",
    cor_dpar = list(),
    cor_group = list(),
    cor_dpar_group = list(),

    sds = "student_t(3, 0, 2.5)",
    sds_dpar = list(),
    sds_coef = list(),
    sds_label = list()
  )
  
  lookup$user_set <- list(
    b = character(0),
    b_all = NULL,
    sd = FALSE,
    sd_dpar = character(0),
    cor = FALSE
  )

  if (!is.null(user_priors) && inherits(user_priors, "broc_prior")) {
    user_priors <- list(user_priors)
  }

  for (p in if (!is.null(user_priors)) user_priors else list()) {
    if (p$class == "b") {
      if (!is.null(p$coef) && !is.null(p$dpar)) {
        key <- paste0(p$dpar, "_", p$coef)
        lookup$b_coef[[key]] <- p$prior
      } else if (!is.null(p$dpar)) {
        lookup$b[[p$dpar]] <- p$prior
        lookup$user_set$b <- unique(c(lookup$user_set$b, p$dpar))
      } else if (is.null(p$coef)) {
        lookup$user_set$b_all <- p$prior
      }

    } else if (p$class == "sd") {
      if (!is.null(p$dpar) && !is.null(p$group) && !is.null(p$coef)) {
        key <- paste0(p$dpar, "_", p$group, "_", p$coef)
        lookup$sd_dpar_group_coef[[key]] <- p$prior
      } else if (!is.null(p$dpar) && !is.null(p$coef)) {
        key <- paste0(p$dpar, "_", p$coef)
        lookup$sd_dpar_coef[[key]] <- p$prior
      } else if (!is.null(p$dpar) && !is.null(p$group)) {
        key <- paste0(p$dpar, "_", p$group)
        lookup$sd_dpar_group[[key]] <- p$prior
      } else if (!is.null(p$group) && !is.null(p$coef)) {
        lookup$sd_group_coef[[paste0(p$group, "_", p$coef)]] <- p$prior
      } else if (!is.null(p$coef)) {
        lookup$sd_coef[[p$coef]] <- p$prior
      } else if (!is.null(p$group)) {
        lookup$sd_group[[p$group]] <- p$prior
      } else if (!is.null(p$dpar)) {
        lookup$sd_dpar[[p$dpar]] <- p$prior
        lookup$user_set$sd_dpar <- unique(c(lookup$user_set$sd_dpar, p$dpar))
      } else {
        lookup$sd <- p$prior
        lookup$user_set$sd <- TRUE
      }

    } else if (p$class == "cor") {
      if (!is.null(p$dpar) && !is.null(p$group)) {
        key <- paste0(p$dpar, "_", p$group)
        lookup$cor_dpar_group[[key]] <- p$prior
      } else if (!is.null(p$group)) {
        lookup$cor_group[[p$group]] <- p$prior
      } else if (!is.null(p$dpar)) {
        lookup$cor_dpar[[p$dpar]] <- p$prior
      } else {
        lookup$cor <- p$prior
        lookup$user_set$cor <- TRUE
      }

    } else if (p$class == "sds") {
      if (!is.null(p$group)) {
        stop("class = \"sds\" priors (smooth penalty SDs) cannot take a group: smooths have no grouping factor.",
             call. = FALSE)
      }
      if (is.null(p$dpar) && !is.null(p$coef)) {
        lookup$sds_label[[p$coef]] <- p$prior
      } else if (!is.null(p$dpar) && !is.null(p$coef)) {
        key <- paste0(p$dpar, "_", p$coef)
        lookup$sds_coef[[key]] <- p$prior
      } else if (!is.null(p$dpar)) {
        lookup$sds_dpar[[p$dpar]] <- p$prior
      } else {
        lookup$sds <- p$prior
        lookup$user_set$sds <- TRUE
      }
    }
  }

  if (!is.null(lookup$user_set$b_all)) {
    for (k in setdiff(names(lookup$b), lookup$user_set$b)) lookup$b[[k]] <- lookup$user_set$b_all
    lookup$user_set$b <- names(lookup$b)
  }
  if (isTRUE(lookup$user_set$sd)) {
    for (k in setdiff(names(lookup$sd_dpar), lookup$user_set$sd_dpar)) lookup$sd_dpar[[k]] <- lookup$sd
    lookup$user_set$sd_dpar <- names(lookup$sd_dpar)
  }

  fam_defaults <- switch(family$family %||% "",
    source_mixture = list(dprime = "normal(-1, 1)", dprime_B = "normal(1, 1)", thresh_mid = "normal(0, 0.75)"),
    cumulative = list(dprime = "normal(0, 1)", thresh_mid = "normal(0, 1.5)"),
    bivariate_cumulative = list(dprime = "normal(0, 1)", thresh_mid = "normal(0, 1.5)",
                                thresh_mid2 = "normal(0, 1.5)", log_gaps2 = "normal(-0.5, 0.5)"),
    bivariate_sdt = list(discrim = "normal(-1, 1)", discrim_B = "normal(1, 1)", dprime_B = "normal(1.5, 0.75)",
                         rho = "normal(-0.5, 0.5)", rho_B = "normal(0.5, 0.5)"),
    bivariate_dp = list(discrim = "normal(-1, 1)", discrim_B = "normal(1, 1)", dprime_B = "normal(1.5, 0.75)"),
    vrdp2d = list(discrim = "normal(-1, 1)", discrim_B = "normal(1, 1)", dprime2 = "normal(2, 1)"),
    list())
  for (pname in setdiff(names(fam_defaults), lookup$user_set$b)) lookup$b[[pname]] <- fam_defaults[[pname]]

  if (!is.null(family) && inherits(family, "broc_family") && !is.null(family$params)) {
    for (pname in names(family$params)) {
      plink <- family$params[[pname]]$link
      if (is.null(plink)) next

      if (!(pname %in% lookup$user_set$b)) {
        if (plink == "log" && pname %in% c("dprime", "dprime_B")) {
          lookup$b[[pname]] <- "normal(0, 0.5)"
        }
        if (plink == "log" && pname %in% c("discrim", "discrim_B")) {
          lookup$b[[pname]] <- "normal(-1.5, 0.75)"
        }
        if (plink == "logis" && pname %in% c("rho", "rho_B")) {
          lookup$b[[pname]] <- "normal(0, 1)"
        }
        if (plink == "logit" && pname %in% c("lambda", "lambda2", "lambda_B", "lambda2_B")) {
          lookup$b[[pname]] <- "normal(0, 1.5)"
        }
      }

      if (!(pname %in% lookup$user_set$sd_dpar)) {
        if (plink == "log" && pname %in% c("dprime", "dprime_B", "discrim", "discrim_B")) {
          lookup$sd_dpar[[pname]] <- "normal(0, 0.3)"
        }
        if (plink == "logis" && pname %in% c("rho", "rho_B")) {
          lookup$sd_dpar[[pname]] <- "normal(0, 0.5)"
        }
      }
    }
  }

  class(lookup) <- "prior_lookup"
  lookup
}

#' Get Prior for Fixed Effect
#'
#' @param lookup prior_lookup object
#' @param dpar Distributional parameter (dprime, sigma, etc.)
#' @param coef Optional coefficient name
#' @return Prior string
#' @noRd
get_fixed_prior <- function(lookup, dpar, coef = NULL) {
  if (!is.null(coef)) {
    key <- paste0(dpar, "_", coef)
    if (!is.null(lookup$b_coef[[key]])) {
      return(lookup$b_coef[[key]])
    }
  }

  if (!is.null(coef) && grepl("^sXs", coef)) {
    return("normal(0, 1)")
  }

  prior <- lookup$b[[dpar]] %||% "normal(0, 1)"
  if (!is.null(coef) && !(dpar %in% lookup$user_set$b) && coef %in% lookup$coefs[[dpar]] &&
      !(coef %in% lookup$location[[dpar]])) prior <- .zero_centre_prior(prior)
  prior
}

.zero_centre_prior <- function(prior) {
  sub("^\\s*normal\\(\\s*[^,]+,", "normal(0,", prior)
}

.location_dpars <- c(dprime = "dprime", sigma = "sigma", lambda = "lambda", dprime2 = "dprime2",
                     sigma2 = "sigma2", dprime_B = "dprime_B", lambda_B = "lambda_B", lambda2 = "lambda2",
                     lambda2_B = "lambda2_B", discrim = "discrim", discrim_B = "discrim_B",
                     sigma_B = "sigma_B", sigma2_B = "sigma2_B", rho = "rho", rho_B = "rho_B", rho_N = "rho_N",
                     rec_crit = "rec_crit", know_crit = "know_crit", dprime_L = "dprime_L", sigma_L = "sigma_L",
                     lambda_L = "lambda_L", thresh_mid = "criterion", thresh_mid2 = "criterion2",
                     criterion_scale = "criterion_scale", criterion2_scale = "criterion2_scale")

.finish_prior_lookup <- function(pl, model_data) {
  pl$location <- list()
  pl$coefs <- list()
  for (dp in names(.location_dpars)) {
    st <- model_data[[paste0(.location_dpars[[dp]], if (dp %in% c("thresh_mid", "thresh_mid2")) "" else "_fixed")]]
    if (is.null(st$coef_names)) next
    pl$coefs[[dp]] <- st$coef_names
    pl$location[[dp]] <- st$location_coefs %||% intersect("(Intercept)", st$coef_names)
  }
  for (g in c("", "2")) {
    pl$coefs[[paste0("log_gaps", g)]] <- pl$coefs[[paste0("thresh_mid", g)]]
    pl$location[[paste0("log_gaps", g)]] <- pl$location[[paste0("thresh_mid", g)]]
  }
  if (isTRUE(model_data$centered_coding) && !("thresh_mid" %in% pl$user_set$b))
    pl$b$thresh_mid <- "normal(0, 0.5)"
  if (isTRUE(model_data$K2 %% 2L == 1L) && !("thresh_mid2" %in% pl$user_set$b) &&
      isTRUE(model_data$family %in% c("bivariate_sdt", "bivariate_dp", "vrdp2d")))
    pl$b$thresh_mid2 <- "normal(-0.5, 0.75)"
  pl
}

#' Get Prior for Random Effect SD
#'
#' @param lookup prior_lookup object
#' @param dpar Distributional parameter (dprime, sigma, criterion, etc.)
#' @param group Grouping factor (participant, words, etc.)
#' @param coef Optional coefficient name for coefficient-level priors
#' @return Prior string
#' @noRd
get_sd_prior <- function(lookup, dpar, group, coef = NULL) {
  if (!is.null(coef)) {
    key <- paste0(dpar, "_", group, "_", coef)
    if (!is.null(lookup$sd_dpar_group_coef[[key]])) {
      return(lookup$sd_dpar_group_coef[[key]])
    }
    
    key <- paste0(dpar, "_", coef)
    if (!is.null(lookup$sd_dpar_coef[[key]])) {
      return(lookup$sd_dpar_coef[[key]])
    }
    if (!is.null(lookup$sd_group_coef[[paste0(group, "_", coef)]])) {
      return(lookup$sd_group_coef[[paste0(group, "_", coef)]])
    }
    if (!is.null(lookup$sd_coef[[coef]])) {
      return(lookup$sd_coef[[coef]])
    }
  }
  
  key <- paste0(dpar, "_", group)
  if (!is.null(lookup$sd_dpar_group[[key]])) {
    return(lookup$sd_dpar_group[[key]])
  }
  
  if (!is.null(lookup$sd_group[[group]])) {
    return(lookup$sd_group[[group]])
  }
  
  if (!is.null(lookup$sd_dpar[[dpar]])) {
    return(lookup$sd_dpar[[dpar]])
  }
  
  lookup$sd
}

#' Get Prior for Correlation
#'
#' @param lookup prior_lookup object
#' @param dpar Distributional parameter (dprime, sigma, criterion, etc.)
#' @param group Grouping factor (participant, words, etc.)
#' @return Prior string
#' @noRd
get_cor_prior <- function(lookup, dpar, group) {
  key <- paste0(dpar, "_", group)
  if (!is.null(lookup$cor_dpar_group[[key]])) {
    return(lookup$cor_dpar_group[[key]])
  }
  
  if (!is.null(lookup$cor_group[[group]])) {
    return(lookup$cor_group[[group]])
  }
  
  if (!is.null(lookup$cor_dpar[[dpar]])) {
    return(lookup$cor_dpar[[dpar]])
  }
  
  lookup$cor
}

#' Get Smooth Penalty SD Prior
#'
#' Resolves the prior for a smooth penalty SD parameter.
#' Resolution order: coef-specific (sds_dpar_label) > dpar-specific (sds_dpar) > global default.
#'
#' @param lookup Prior lookup object
#' @param dpar Parameter name (e.g., "dprime")
#' @param smooth_label Sanitized smooth label (e.g., "sage_1")
#' @return Prior string
#' @noRd
get_sds_prior <- function(lookup, dpar, smooth_label = NULL) {
  if (!is.null(smooth_label)) {
    key <- paste0(dpar, "_", smooth_label)
    if (!is.null(lookup$sds_coef[[key]])) {
      return(lookup$sds_coef[[key]])
    }
    if (!is.null(lookup$sds_label[[smooth_label]])) {
      return(lookup$sds_label[[smooth_label]])
    }
  }

  if (!is.null(lookup$sds_dpar[[dpar]])) {
    return(lookup$sds_dpar[[dpar]])
  }

  if (!is.null(lookup$sds)) {
    return(lookup$sds)
  }

  "student_t(3, 0, 2.5)"
}

#' Print Method for Prior Lookup
#' @param x The object to print.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @export
print.prior_lookup <- function(x, ...) {
  cat("Prior lookup\n")
  cat("============\n\n")
  
  cat("Fixed Effects (class = 'b'):\n")
  for (dpar in names(x$b)) {
    cat(sprintf("  %s: %s\n", dpar, x$b[[dpar]]))
  }
  
  if (length(x$b_coef) > 0) {
    cat("\nCoefficient-specific:\n")
    for (key in names(x$b_coef)) {
      cat(sprintf("  %s: %s\n", key, x$b_coef[[key]]))
    }
  }
  
  cat("\nRandom Effect SDs (class = 'sd'):\n")
  cat(sprintf("  (default): %s\n", x$sd))
  for (dpar in names(x$sd_dpar)) {
    cat(sprintf("  %s: %s\n", dpar, x$sd_dpar[[dpar]]))
  }
  if (length(x$sd_group) > 0) {
    for (grp in names(x$sd_group)) {
      cat(sprintf("  group=%s: %s\n", grp, x$sd_group[[grp]]))
    }
  }
  if (length(x$sd_dpar_group) > 0) {
    for (key in names(x$sd_dpar_group)) {
      cat(sprintf("  %s: %s\n", key, x$sd_dpar_group[[key]]))
    }
  }
  
  cat("\nCorrelations (class = 'cor'):\n")
  cat(sprintf("  (default): %s\n", x$cor))
  if (length(x$cor_dpar) > 0) {
    for (dpar in names(x$cor_dpar)) {
      cat(sprintf("  %s: %s\n", dpar, x$cor_dpar[[dpar]]))
    }
  }
  if (length(x$cor_group) > 0) {
    for (grp in names(x$cor_group)) {
      cat(sprintf("  group=%s: %s\n", grp, x$cor_group[[grp]]))
    }
  }
  if (length(x$cor_dpar_group) > 0) {
    for (key in names(x$cor_dpar_group)) {
      cat(sprintf("  %s: %s\n", key, x$cor_dpar_group[[key]]))
    }
  }
  
  invisible(x)
}

#' Preview the Priors for an SDT Model
#'
#' Show every prior a model will use as a data frame. Accepts either a built
#' [broc()] model or a [brf()] object that specifies a formula, data, and a
#' family.
#'
#' @param object A [broc()] model, or a [brf()] specification / bare formula.
#'   When a formula or `brf`, the model is built internally from `data`,
#'   `family`, and any `...`.
#' @param data Data frame. Required when `object` is a formula or `brf`, because
#'   coefficient names and the number of thresholds are read from it. Ignored
#'   when `object` is already a model.
#' @param family Optional family (e.g. [evsd()], [uvsd()]) for the formula/`brf`
#'   path; defaults to the family set in the `brf`, or [evsd()]. Ignored when
#'   `object` is already a model.
#' @param priors Optional priors from [broc_prior()] to override defaults (as in [broc()]).
#' @param ... Further arguments forwarded to [broc()] on the formula/`brf` path
#'   (e.g. `encoding_vars`, `counts`, `gap_link`).
#' @return A data frame with columns `class`, `dpar`, `coef`, `group`, `prior`,
#'   and `source`, where `source` flags whether each row is a package default
#'   (`"default"`) or was overridden by `priors` (`"user"`).
#' @seealso [broc_prior()]
#' @export
#'
#' @examples
#' \dontrun{
#' # From a formula
#' get_broc_prior(conf | old ~ condition + (1 | subject),
#'                data = my_data, family = evsd())
#'
#' # From a brf() specification (family carried on the brf)
#' spec <- brf(conf | old ~ condition + (1 | subject),
#'             criterion ~ 1 + (1 | subject), family = uvsd())
#' get_broc_prior(spec, data = my_data)
#'
#' # From an already-built model, with custom overrides
#' model <- broc(spec, data = my_data)
#' my_priors <- c(
#'   broc_prior("normal(2, 0.5)", class = "b", dpar = "dprime"),
#'   broc_prior("normal(0, 1)", class = "sd")
#' )
#' get_broc_prior(model, priors = my_priors)
#' }
get_broc_prior <- function(object, data = NULL, family = NULL,
                           priors = NULL, ...) {

  if (inherits(object, "broc_model")) {
    model <- object
  } else if (inherits(object, "brf") || inherits(object, "formula")) {
    if (is.null(data)) {
      stop("`data` is required when `object` is a formula or brf() object: ",
           "coefficient names and the number of thresholds are read from it.")
    }
    model <- broc(object, data = data, family = family, priors = priors, ...)
  } else {
    stop("`object` must be a broc_model (from broc()), a brf() object, or a formula.")
  }

  pl <- if (!is.null(priors)) {
    .finish_prior_lookup(build_prior_lookup(priors, family = model$family), model$model_data)
  } else if (!is.null(model$prior_lookup)) {
    model$prior_lookup
  } else {
    build_prior_lookup(NULL)
  }
  
  is_cumulative <- model$family$family == "cumulative"
  
  priors <- list()
  
  add_prior <- function(class, dpar, coef, group, prior, source) {
    priors[[length(priors) + 1]] <<- data.frame(
      class = class,
      dpar = if (is.null(dpar)) "" else dpar,
      coef = if (is.null(coef)) "" else coef,
      group = if (is.null(group)) "" else group,
      prior = prior,
      source = source,
      stringsAsFactors = FALSE
    )
  }
  
  # --- Fixed effects (class = "b") ---

  is_user_b <- function(dpar_name) dpar_name %in% pl$user_set$b

  add_fixed_priors <- function(internal_dpar, display_dpar, coef_names) {
    for (coef in coef_names) {
      key <- paste0(internal_dpar, "_", coef)
      prior <- get_fixed_prior(pl, internal_dpar, coef)
      if (!is.null(pl$b_coef[[key]])) {
        add_prior("b", display_dpar, coef, NA, pl$b_coef[[key]], "user")
      } else if (is_user_b(internal_dpar)) {
        add_prior("b", display_dpar, coef, NA, prior, "user")
      } else {
        add_prior("b", display_dpar, coef, NA, prior, "default")
      }
    }
  }

  is_bivariate_cumulative <- model$family$family == "bivariate_cumulative"
  dprime_dpar <- if (is_cumulative || is_bivariate_cumulative) "mu" else "dprime"
  has_dprime_fixed <- isTRUE(model$model_data$has_dprime_fixed)

  if (has_dprime_fixed && model$model_data$dprime_fixed$n_coef > 0) {
    add_fixed_priors("dprime", dprime_dpar, model$model_data$dprime_fixed$coef_names)
  }

  crit_dpar <- if (is_cumulative || is_bivariate_cumulative) "cutpoints" else "criterion"
  crit2_dpar <- if (is_bivariate_cumulative) "cutpoints2" else "criterion2"
  add_thresh_priors <- function(mid_key, gap_key, coefs, n_gaps, ls = FALSE) {
    tg <- .thresh_prior_grid(pl, mid_key, gap_key, coefs, n_gaps)
    gap_cols <- if (ls) seq_len(min(1L, length(tg$coefs))) else seq_along(tg$coefs)
    for (p in seq_along(tg$coefs)) {
      add_prior("b", mid_key, tg$coefs[p], NA, tg$mid[p],
                if (is_user_b(mid_key) || !is.null(pl$b_coef[[paste0(mid_key, "_", tg$coefs[p])]])) "user" else "default")
    }
    for (k in seq_len(max(0L, n_gaps))) {
      for (p in gap_cols) {
        cf <- paste0("gap", k, "_", tg$coefs[p])
        user <- is_user_b(gap_key) || !is.null(pl$b_coef[[paste0(gap_key, "_", cf)]]) ||
          !is.null(pl$b_coef[[paste0(gap_key, "_gap", k)]]) ||
          !is.null(pl$b_coef[[paste0(gap_key, "_[", k, "]")]])
        add_prior("b", gap_key, cf, NA, tg$gaps[k, p], if (user) "user" else "default")
      }
    }
  }
  add_thresh_priors("thresh_mid", "log_gaps", model$model_data$criterion$coef_names, model$model_data$K - 2L,
                    isTRUE(model$model_data$has_criterion_scale))
  
  if (model$model_data$has_sigma)
    add_fixed_priors("sigma", "sigma", model$model_data$sigma_fixed$coef_names)
  if (isTRUE(model$model_data$has_dprime2))
    add_fixed_priors("dprime2", "dprime2", model$model_data$dprime2_fixed$coef_names)
  if (model$model_data$has_lambda)
    add_fixed_priors("lambda", "lambda", model$model_data$lambda_fixed$coef_names)
  if (isTRUE(model$model_data$has_dprime_B))
    add_fixed_priors("dprime_B", "dprime_B", model$model_data$dprime_B_fixed$coef_names)
  if (isTRUE(model$model_data$has_lambda_B))
    add_fixed_priors("lambda_B", "lambda_B", model$model_data$lambda_B_fixed$coef_names)
  if (isTRUE(model$model_data$has_discrim))
    add_fixed_priors("discrim", if (is_bivariate_cumulative) "mu2" else "discrim",
                     model$model_data$discrim_fixed$coef_names)
  if (isTRUE(model$model_data$has_discrim_B))
    add_fixed_priors("discrim_B", "discrim_B", model$model_data$discrim_B_fixed$coef_names)
  if (isTRUE(model$model_data$has_sigma_B))
    add_fixed_priors("sigma_B", "sigma_B", model$model_data$sigma_B_fixed$coef_names)
  if (isTRUE(model$model_data$has_sigma2))
    add_fixed_priors("sigma2", "sigma2", model$model_data$sigma2_fixed$coef_names)
  if (isTRUE(model$model_data$has_sigma2_B))
    add_fixed_priors("sigma2_B", "sigma2_B", model$model_data$sigma2_B_fixed$coef_names)
  if (isTRUE(model$model_data$has_rho))
    add_fixed_priors("rho", "rho", model$model_data$rho_fixed$coef_names)
  if (isTRUE(model$model_data$has_rho_B))
    add_fixed_priors("rho_B", "rho_B", model$model_data$rho_B_fixed$coef_names)
  if (isTRUE(model$model_data$has_rho_N))
    add_fixed_priors("rho_N", "rho_N", model$model_data$rho_N_fixed$coef_names)
  if (isTRUE(model$model_data$has_rec_crit))
    add_fixed_priors("rec_crit", "rec_crit", model$model_data$rec_crit_fixed$coef_names)
  if (isTRUE(model$model_data$has_know_crit))
    add_fixed_priors("know_crit", "know_crit", model$model_data$know_crit_fixed$coef_names)
  if (isTRUE(model$model_data$has_dprime_L))
    add_fixed_priors("dprime_L", "dprime_L", model$model_data$dprime_L_fixed$coef_names)
  if (isTRUE(model$model_data$has_sigma_L))
    add_fixed_priors("sigma_L", "sigma_L", model$model_data$sigma_L_fixed$coef_names)
  if (isTRUE(model$model_data$has_lambda_L))
    add_fixed_priors("lambda_L", "lambda_L", model$model_data$lambda_L_fixed$coef_names)
  if (isTRUE(model$model_data$has_lambda2))
    add_fixed_priors("lambda2", "lambda2", model$model_data$lambda2_fixed$coef_names)
  if (isTRUE(model$model_data$has_lambda2_B))
    add_fixed_priors("lambda2_B", "lambda2_B", model$model_data$lambda2_B_fixed$coef_names)
  if (isTRUE(model$model_data$has_criterion_scale) && model$model_data$criterion_scale_fixed$n_coef > 0)
    add_fixed_priors("criterion_scale", "criterion_scale", model$model_data$criterion_scale_fixed$coef_names)
  if (isTRUE(model$model_data$has_criterion2_scale) && model$model_data$criterion2_scale_fixed$n_coef > 0)
    add_fixed_priors("criterion2_scale", "criterion2_scale", model$model_data$criterion2_scale_fixed$coef_names)

  if (isTRUE(model$model_data$has_criterion2)) {
    add_thresh_priors("thresh_mid2", "log_gaps2", model$model_data$criterion2$coef_names,
                      model$model_data$K2 - 2L, isTRUE(model$model_data$has_criterion2_scale))
  }
  
  # --- Random effect SDs (class = "sd") and correlations (class = "cor") ---

  name_map <- .param_name_map(model$family$family)
  label <- function(p) name_map[[p]] %||% p
  user_sd <- function(p, group, coef) {
    !is.null(pl$sd_dpar_group_coef[[paste0(p, "_", group, "_", coef)]]) ||
      !is.null(pl$sd_dpar_coef[[paste0(p, "_", coef)]]) ||
      !is.null(pl$sd_group_coef[[paste0(group, "_", coef)]]) || !is.null(pl$sd_coef[[coef]]) ||
      is_user_sd_prior(pl, p, group)
  }
  add_crit_sd <- function(p, group, re) {
    r <- .crit_re_sd_resolve(pl, p, group, re)
    r <- r[!duplicated(r[c("comp", "term")]), , drop = FALSE]
    r <- r[order(r$comp > 0L, r$comp), , drop = FALSE]
    for (i in seq_len(nrow(r))) {
      add_prior("sd", r$dpar[i], if (r$comp[i] == 0L) r$term[i] else paste0("gap", r$comp[i], "_", r$term[i]),
                group, r$prior[i], if (r$user[i]) "user" else "default")
    }
  }
  for (p in .re_param_names()) {
    res <- .re_list_of(model$model_data, p)
    for (group in names(res)) {
      re <- res[[group]]
      if (!is.null(re$cor_id)) next
      if (p %in% c("criterion", "criterion2")) { add_crit_sd(p, group, re); next }
      for (coef in .re_sd_coef_names(p, re)) {
        add_prior("sd", label(p), coef, group, get_sd_prior(pl, p, group, coef),
                  if (user_sd(p, group, coef)) "user" else "default")
      }
    }
  }
  for (p in .re_param_names()) {
    res <- .re_list_of(model$model_data, p)
    for (group in names(res)) {
      re <- res[[group]]
      if (!is.null(re$cor_id) || !isTRUE(re$correlated) || !isTRUE(re$dim > 1)) next
      add_prior("cor", label(p), NA, group, get_cor_prior(pl, p, group),
                if (is_user_cor_prior(pl, p, group)) "user" else "default")
    }
  }
  for (cor_id in names(model$model_data$cross_cor)) {
    cc <- model$model_data$cross_cor[[cor_id]]
    for (member in cc$members) {
      re <- .re_list_of(model$model_data, member$param)[[cc$group]]
      if (is.null(re)) next
      if (member$param %in% c("criterion", "criterion2")) { add_crit_sd(member$param, cc$group, re); next }
      for (coef in .re_sd_coef_names(member$param, re)) {
        add_prior("sd", label(member$param), coef, cc$group, get_sd_prior(pl, member$param, cc$group, coef),
                  if (user_sd(member$param, cc$group, coef)) "user" else "default")
      }
    }
    ck <- paste0("cross_", cor_id)
    add_prior("cor", ck, NA, cc$group, .cross_cor_prior(pl, cor_id, cc$group),
              if (is_user_cor_prior(pl, ck, cc$group) || is_user_cor_prior(pl, "cross", cc$group)) "user" else "default")
  }

  # --- Smooth penalty SDs (class = "sds") ---
  sm_data <- model$model_data$smooth_data
  if (!is.null(sm_data) && length(sm_data) > 0) {
    is_user_sds <- function(dpar_, key_) {
      !is.null(pl$sds_coef[[paste0(dpar_, "_", key_)]]) || !is.null(pl$sds_label[[key_]]) ||
      !is.null(pl$sds_dpar[[dpar_]]) || isTRUE(pl$user_set$sds)
    }
    for (pname in names(sm_data)) {
      display_pname <- if (is_cumulative && pname == "dprime") "mu"
                       else if (is_cumulative && pname == "criterion") "cutpoints"
                       else pname
      sms <- sm_data[[pname]]
      if (is.null(sms)) next
      for (sm in sms) {
        for (comp in sm$components) {
          n_zs <- length(comp$Zs_list)
          for (k in seq_len(n_zs)) {
            label <- paste0(comp$san_label, "_", k)
            sds_prior <- get_sds_prior(pl, pname, label)
            src <- if (is_user_sds(pname, label)) "user" else "default"
            add_prior("sds", display_pname, label, NA, sds_prior, src)
          }
        }
      }
    }
  }

  if (length(priors) == 0) {
    return(data.frame(
      class = character(0),
      dpar = character(0),
      coef = character(0),
      group = character(0),
      prior = character(0),
      source = character(0),
      stringsAsFactors = FALSE
    ))
  }
  
  result <- do.call(rbind, priors)
  rownames(result) <- NULL
  
  
  if (model$family$family == "cdp") {
    result$dpar <- gsub("^dprime$", "rec", result$dpar)
    result$dpar <- gsub("^dprime2$", "fam", result$dpar)
    result$dpar <- gsub("^sigma$", "sigma_R", result$dpar)
    result$dpar <- gsub("^sigma2$", "sigma_F", result$dpar)
  }
  
  result
}

#' Check if user specified an SD prior (not just using defaults)
#' @noRd
is_user_sd_prior <- function(pl, dpar, group) {
  key <- paste0(dpar, "_", group)
  !is.null(pl$sd_dpar_group[[key]]) ||
    !is.null(pl$sd_group[[group]]) ||
    (dpar %in% pl$user_set$sd_dpar) ||
    isTRUE(pl$user_set$sd)
}

#' Check if user specified a correlation prior (not just using defaults)
#' @noRd
is_user_cor_prior <- function(pl, dpar, group) {
  key <- paste0(dpar, "_", group)
  !is.null(pl$cor_dpar_group[[key]]) || !is.null(pl$cor_dpar[[dpar]]) ||
    !is.null(pl$cor_group[[group]]) || isTRUE(pl$user_set$cor)
}

#' Print method for get_broc_prior output
#' @param x The object to print.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @export
print.broc_prior_table <- function(x, ...) {
  cat("bayesroc model priors\n")
  cat("=====================\n\n")
  
  fixed <- x[x$class == "b", ]
  if (nrow(fixed) > 0) {
    cat("Fixed Effects (class = 'b'):\n")
    print(fixed[, c("dpar", "coef", "prior", "source")], row.names = FALSE)
    cat("\n")
  }
  
  sds <- x[x$class == "sd", ]
  if (nrow(sds) > 0) {
    cat("Random Effect SDs (class = 'sd'):\n")
    print(sds[, c("dpar", "group", "prior", "source")], row.names = FALSE)
    cat("\n")
  }
  
  cors <- x[x$class == "cor", ]
  if (nrow(cors) > 0) {
    cat("Correlations (class = 'cor'):\n")
    print(cors[, c("dpar", "group", "prior", "source")], row.names = FALSE)
  }
  
  invisible(x)
}

.crit_re_coef_names <- function(re) {
  R <- as.integer(re$n_re_terms %||% 1L)
  if (R == 1L && is.null(re$term_idx) && !isTRUE(re$use_z_matrix)) return("(Intercept)")
  re$level_names %||% paste0("term", seq_len(R))
}

.crit_re_components <- function(re) {
  R <- as.integer(re$n_re_terms %||% 1L)
  nb <- as.integer(re$dim) %/% R
  mode <- re$varying_re_mode %||% "none"
  J <- if (mode == "full") nb %/% as.integer(re$n_bins) else if (mode == "none") nb else 1L
  k <- ((seq_len(nb) - 1L) %% J) + 1L
  m <- (J + 1L) %/% 2L
  rep(ifelse(k == m, 0L, ifelse(k > m, k - m, J - k)), each = R)
}

.first_prior <- function(...) {
  for (x in list(...)) {
    x <- unlist(x, use.names = FALSE)
    x <- x[!is.na(x)]
    if (length(x)) return(x[[1]])
  }
  NA_character_
}

.crit_re_sd_resolve <- function(pl, dpar, group, re) {
  terms <- rep_len(.crit_re_coef_names(re), re$dim)
  comp <- .crit_re_components(re)
  sfx <- if (dpar == "criterion2") "2" else ""
  out <- lapply(seq_len(re$dim), function(d) {
    cf <- terms[d]
    cd <- if (comp[d] == 0L) paste0("thresh_mid", sfx) else paste0("log_gaps", sfx)
    cc <- if (comp[d] == 0L) cf else c(paste0("gap", comp[d], "_", cf), paste0("gap", comp[d]), cf)
    user <- .first_prior(pl$sd_dpar_group_coef[paste0(cd, "_", group, "_", cc)],
                         pl$sd_dpar_group_coef[[paste0(dpar, "_", group, "_", cf)]],
                         pl$sd_dpar_coef[paste0(cd, "_", cc)],
                         pl$sd_dpar_coef[[paste0(dpar, "_", cf)]],
                         pl$sd_group_coef[paste0(group, "_", cc)],
                         pl$sd_coef[cc],
                         pl$sd_dpar_group[[paste0(cd, "_", group)]],
                         pl$sd_dpar_group[[paste0(dpar, "_", group)]],
                         pl$sd_group[[group]],
                         pl$sd_dpar[[cd]])
    if (!is.na(user)) return(list(prior = user, user = TRUE))
    list(prior = .first_prior(pl$sd_dpar[[dpar]], pl$sd), user = is_user_sd_prior(pl, dpar, group))
  })
  data.frame(comp = comp, term = terms, dpar = ifelse(comp == 0L, paste0("thresh_mid", sfx), paste0("log_gaps", sfx)),
             prior = vapply(out, `[[`, "", "prior"), user = vapply(out, `[[`, TRUE, "user"),
             stringsAsFactors = FALSE)
}

.check_priors_matched <- function(model, priors) {
  if (is.null(priors)) return(invisible(NULL))
  ps <- if (inherits(priors, "broc_prior")) list(priors) else unclass(priors)
  all_tab <- NULL
  unused <- character(0)
  for (p in ps) {
    tab <- get_broc_prior(model, priors = p)
    if (any(tab$source == "user")) next
    if (is.null(all_tab)) all_tab <- get_broc_prior(model)
    cand <- all_tab[all_tab$class == p$class & (is.null(p$dpar) | all_tab$dpar == (p$dpar %||% "")), , drop = FALSE]
    spec <- paste0("class = \"", p$class, "\"",
                   if (!is.null(p$dpar)) paste0(", dpar = \"", p$dpar, "\""),
                   if (!is.null(p$coef)) paste0(", coef = \"", p$coef, "\""),
                   if (!is.null(p$group)) paste0(", group = \"", p$group, "\""))
    if (!nrow(cand)) {
      hint <- if (identical(p$class, "b") && isTRUE(p$dpar %in% c("criterion", "cutpoints", "criterion2", "cutpoints2")))
        paste0(" (threshold effects take priors through dpar = \"thresh_mid", if (grepl("2$", p$dpar)) "2",
               "\" and \"log_gaps", if (grepl("2$", p$dpar)) "2", "\")") else ""
      unused <- c(unused, paste0(p$prior, " (", spec, ")", hint))
      next
    }
    lab <- unique(paste0(ifelse(nzchar(cand$coef) & !is.na(cand$coef), paste0("coef = \"", cand$coef, "\""), ""),
                         ifelse(nzchar(cand$group) & !is.na(cand$group),
                                paste0(ifelse(nzchar(cand$coef) & !is.na(cand$coef), ", ", ""), "group = \"", cand$group, "\""), "")))
    lab <- lab[nzchar(lab)]
    stop("Prior ", p$prior, " (", spec, ") matches no parameter in this model.",
         if (length(lab)) paste0(" Valid targets: ", paste(utils::head(lab, 12), collapse = "; "),
                                 if (length(lab) > 12) "; ..." else "", "."),
         " See get_broc_prior(model).", call. = FALSE)
  }
  if (length(unused)) {
    message("Priors for parameters this model does not have (not used): ", paste(unused, collapse = "; "), ".")
  }
  invisible(NULL)
}

.crit_re_sd_priors <- function(pl, dpar, group, re) {
  .crit_re_sd_resolve(pl, dpar, group, re)$prior
}

.thresh_prior_grid <- function(pl, mid_key, gap_key, coefs, n_gaps) {
  if (length(coefs) == 0) coefs <- "(Intercept)"
  mid <- vapply(coefs, function(cf) get_fixed_prior(pl, mid_key, cf), character(1), USE.NAMES = FALSE)
  base <- get_fixed_prior(pl, gap_key)
  gaps <- matrix(base, max(0L, n_gaps), length(coefs))
  for (k in seq_len(max(0L, n_gaps))) {
    for (p in seq_along(coefs)) {
      gaps[k, p] <- pl$b_coef[[paste0(gap_key, "_gap", k, "_", coefs[p])]] %||%
        pl$b_coef[[paste0(gap_key, "_gap", k)]] %||%
        pl$b_coef[[paste0(gap_key, "_[", k, "]")]] %||% get_fixed_prior(pl, gap_key, coefs[p])
    }
  }
  list(mid = mid, gaps = gaps, coefs = coefs)
}

.crit2_vary_prior_grid <- function(pl, model_data) {
  g <- .thresh_prior_grid(pl, "thresh_mid2", "log_gaps2", model_data$criterion2$coef_names,
                          model_data$K2 - 2L)
  cells <- model_data$criterion2$vcell_names %||% "(Intercept)"
  cov <- match(model_data$criterion2$cov_names %||% setdiff(g$coefs, "(Intercept)"), g$coefs)
  ic <- match(cells, g$coefs)
  base_mid <- if ("(Intercept)" %in% g$coefs) g$mid[match("(Intercept)", g$coefs)] else get_fixed_prior(pl, "thresh_mid2")
  base_gaps <- if ("(Intercept)" %in% g$coefs) g$gaps[, match("(Intercept)", g$coefs)]
               else vapply(seq_len(nrow(g$gaps)), function(k)
                 pl$b_coef[[paste0("log_gaps2_gap", k)]] %||% pl$b_coef[[paste0("log_gaps2_[", k, "]")]] %||%
                   get_fixed_prior(pl, "log_gaps2"), "")
  n_bin <- max(.vbin_of(model_data))
  mid <- rep(ifelse(is.na(ic), base_mid, g$mid[ic]), each = n_bin)
  gaps <- matrix(base_gaps, nrow(g$gaps), length(cells))
  gaps[, !is.na(ic)] <- g$gaps[, ic[!is.na(ic)]]
  list(mid = mid, gaps = gaps[, rep(seq_along(cells), each = n_bin), drop = FALSE],
       cov_mid = g$mid[cov], cov_gaps = g$gaps[, cov, drop = FALSE])
}

.stan_prior_vec <- function(name, priors) {
  if (length(priors) == 0) return(character(0))
  if (length(unique(priors)) == 1L) return(sprintf("  %s ~ %s;", name, priors[1]))
  sprintf("  %s[%d] ~ %s;", name, seq_along(priors), priors)
}

.stan_prior_mat <- function(name, priors) {
  if (length(priors) == 0) return(character(0))
  if (length(unique(as.vector(priors))) == 1L) return(sprintf("  to_vector(%s) ~ %s;", name, priors[1]))
  sprintf("  %s[%d, %d] ~ %s;", name, as.vector(row(priors)), as.vector(col(priors)), as.vector(priors))
}

.re_param_names <- function() {
  c("dprime", "criterion", "sigma", "lambda", "dprime2", "sigma2", "dprime_B", "lambda_B",
    "discrim", "discrim_B", "sigma_B", "sigma2_B", "rho", "rho_B", "rho_N", "criterion2",
    "rec_crit", "know_crit", "dprime_L", "sigma_L", "lambda_L", "lambda2", "lambda2_B", "criterion_scale",
    "criterion2_scale")
}

.re_list_of <- function(model_data, p) {
  if (p %in% c("criterion", "criterion2")) model_data[[p]]$random else model_data[[paste0(p, "_random")]]
}

.re_sd_coef_names <- function(p, re) {
  if (p %in% c("criterion", "criterion2")) return(.crit_re_coef_names(re))
  lv <- re$level_names
  if (is.null(lv) || length(lv) == 0) "(Intercept)" else lv
}

.re_sd_priors <- function(pl, p, group, re) {
  if (p %in% c("criterion", "criterion2")) return(.crit_re_sd_priors(pl, p, group, re))
  per <- vapply(.re_sd_coef_names(p, re), function(coef) get_sd_prior(pl, p, group, coef),
                character(1), USE.NAMES = FALSE)
  rep_len(per, re$dim)
}

.cross_cor_prior <- function(pl, cor_id, group) {
  k <- paste0("cross_", cor_id)
  pl$cor_dpar_group[[paste0(k, "_", group)]] %||% pl$cor_dpar[[k]] %||% get_cor_prior(pl, "cross", group)
}
