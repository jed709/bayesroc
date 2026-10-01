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
#'   and CDP families). Family-specific aliases (`mu`, `mu1`, `mu2`, `rec`, `fam`,
#'   `sigma_R`, `sigma_F`) are also accepted. `NULL` applies the prior to every
#'   parameter in `class`.
#' @param coef Optional specific coefficient name (e.g., `"conditionread"`)
#' @param group Optional grouping factor for random effects (e.g., `"participant"`)
#' @return A `broc_prior` object. Combine several with [c()] and pass to [broc()].
#' @seealso [get_broc_prior()]
#'
#' @examples
#' # Prior on all d' fixed effects
#' broc_prior("normal(1, 1)", class = "b", dpar = "dprime")
#' 
#' # Prior on specific d' coefficient
#' broc_prior("normal(2, 0.5)", class = "b", dpar = "dprime", coef = "conditionread")
#' 
#' # Prior on ALL random effect standard deviations
#' broc_prior("normal(0, 0.5)", class = "sd")
#' 
#' # Prior on random effect SDs for d' only
#' broc_prior("normal(0, 0.5)", class = "sd", dpar = "dprime")
#' 
#' # Prior on random effect SDs for d' for participant group only
#' broc_prior("normal(0, 0.3)", class = "sd", dpar = "dprime", group = "participant")
#' 
#' # Prior on ALL correlations
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
                     "mu", "mu1", "mu2")
    if (!dpar %in% valid_dpars) {
      stop("Invalid dpar '", dpar, "'. Valid dpars: ",
           paste(valid_dpars, collapse = ", "))
    }

    if (dpar == "mu" || dpar == "mu1") dpar <- "dprime"
    if (dpar == "mu2") dpar <- "discrim"
    
    if (dpar == "rec") dpar <- "dprime"
    if (dpar == "fam") dpar <- "dprime2"
    if (dpar == "sigma_R") dpar <- "sigma"
    if (dpar == "sigma_F") dpar <- "sigma2"
  }
  
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
#' @param ... Prior specifications from broc_prior()
#' @return A list of prior specifications
#' @export
c.broc_prior <- function(...) {
  priors <- list(...)
  class(priors) <- "broc_prior_list"
  priors
}

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

#' @noRd
get_default_broc_priors <- function() {
  data.frame(
    class = c("b", "b", "b", "b", "b", "b", "b", "b", "b",
              "b", "b", "b", "b", "b", "b", "b", "b",
              "b", "b", "b",
              "sd", "sd", "sd", "sd", "sd", "sd", "sd", "sd",
              "sd", "sd", "sd", "sd", "sd", "sd", "sd", "sd",
              "sd", "sd", "sd",
              "cor"),
    dpar = c("dprime", "sigma", "lambda", "dprime2", "sigma2", 
             "dprime_B", "lambda_B",
             "thresh_mid", "log_gaps",
             "discrim", "discrim_B", "sigma_B", "sigma2_B",
             "rho", "rho_B", "rho_N", "criterion2",
             "dprime_L", "sigma_L", "lambda_L",
             "dprime", "sigma", "lambda", "dprime2", "sigma2", 
             "dprime_B", "lambda_B", "criterion",
             "discrim", "discrim_B", "sigma_B", "sigma2_B",
             "rho", "rho_B", "rho_N", "criterion2",
             "dprime_L", "sigma_L", "lambda_L",
             NA),
    prior = c("normal(1, 1)", "normal(0, 0.5)", "normal(0, 1.5)", 
              "normal(0, 1)", "normal(0, 0.5)", 
              "normal(-1, 1)", "normal(0, 1.5)",
              "normal(0, 1.5)", "normal(0, 1)",
              "normal(0, 1)", "normal(0, 1)",
              "normal(0, 0.5)", "normal(0, 0.5)",
              "normal(0, 0.5)", "normal(0, 0.5)", "normal(0, 0.5)",
              "normal(0, 1.5)",
              "normal(0, 1)", "normal(0, 0.5)", "normal(0, 1.5)",
              "normal(0, 0.5)", "normal(0, 0.3)", "normal(0, 0.5)", 
              "normal(0, 0.5)", "normal(0, 0.3)", 
              "normal(0, 0.5)", "normal(0, 0.5)",
              "normal(0, 0.5)",
              "normal(0, 0.5)", "normal(0, 0.5)",
              "normal(0, 0.3)", "normal(0, 0.3)",
              "normal(0, 0.3)", "normal(0, 0.3)", "normal(0, 0.3)",
              "normal(0, 0.5)",
              "normal(0, 0.5)", "normal(0, 0.3)", "normal(0, 0.5)",
              "lkj_corr_cholesky(1)"),
    stringsAsFactors = FALSE
  )
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
      dprime = "normal(1, 1)",
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
      rec = "normal(1, 1)",
      fam = "normal(0, 1)",
      sigma_R = "normal(0, 0.5)",
      sigma_F = "normal(0, 0.5)",
      rec_crit = "normal(1.5, 1)",
      know_crit = "normal(1.0, 1)",
      dprime_L = "normal(0, 1)",
      sigma_L = "normal(0, 0.5)",
      lambda_L = "normal(0, 1.5)",
      thresh_mid = "normal(0, 1.5)",
      log_gaps = "normal(0, 1)",
      thresh_mid2 = "normal(0, 1.5)",
      log_gaps2 = "normal(0, 1)"
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
      rec = "normal(0, 0.5)",
      fam = "normal(0, 0.5)",
      sigma_R = "normal(0, 0.3)",
      sigma_F = "normal(0, 0.3)",
      rec_crit = "normal(0, 0.5)",
      know_crit = "normal(0, 0.3)",
      dprime_L = "normal(0, 0.5)",
      sigma_L = "normal(0, 0.3)",
      lambda_L = "normal(0, 0.5)",
      criterion = "normal(0, 0.5)",
      criterion2 = "normal(0, 0.5)"
    ),
    sd_group = list(),
    sd_dpar_group = list(),
    sd_dpar_coef = list(),
    sd_dpar_group_coef = list(),
    
    cor = "lkj_corr_cholesky(1)",
    cor_dpar = list(),
    cor_group = list(),
    cor_dpar_group = list(),

    sds = "student_t(3, 0, 2.5)",
    sds_dpar = list(),
    sds_coef = list()
  )
  
  lookup$user_set <- list(
    b = character(0),
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
      if (!is.null(p$dpar) && !is.null(p$coef)) {
        key <- paste0(p$dpar, "_", p$coef)
        lookup$sds_coef[[key]] <- p$prior
      } else if (!is.null(p$dpar)) {
        lookup$sds_dpar[[p$dpar]] <- p$prior
      } else {
        lookup$sds <- p$prior
      }
    }
  }

  if (!is.null(family) && inherits(family, "broc_family") && !is.null(family$params)) {
    for (pname in names(family$params)) {
      plink <- family$params[[pname]]$link
      if (is.null(plink)) next
      if (pname %in% lookup$user_set$b) next

      if (plink == "log" && pname %in% c("dprime", "dprime_B", "discrim", "discrim_B")) {
        lookup$b[[pname]] <- "normal(0, 0.5)"
      }
      if (plink == "logis" && pname %in% c("rho", "rho_B")) {
        lookup$b[[pname]] <- "normal(0, 1)"
      }
      if (plink == "logit" && pname %in% c("lambda", "lambda2", "lambda_B", "lambda2_B")) {
        lookup$b[[pname]] <- "normal(0, 1.5)"
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

  if (!is.null(lookup$b[[dpar]])) {
    return(lookup$b[[dpar]])
  }

  "normal(0, 1)"
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
#' # Straight from a formula -- no need to build the model first
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
    build_prior_lookup(priors)
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
  dprime_dpar <- if (is_cumulative) "mu" else if (is_bivariate_cumulative) "mu1" else "dprime"
  has_dprime_fixed <- isTRUE(model$model_data$has_dprime_fixed)

  if (has_dprime_fixed && model$model_data$dprime_fixed$n_coef > 0) {
    add_fixed_priors("dprime", dprime_dpar, model$model_data$dprime_fixed$coef_names)
  }

  crit_dpar <- if (is_cumulative) "cutpoints" else "criterion"
  thresh_mid_prior <- get_fixed_prior(pl, "thresh_mid")
  log_gaps_prior <- get_fixed_prior(pl, "log_gaps")

  crit_coefs <- model$model_data$criterion$coef_names
  for (coef in crit_coefs) {
    add_prior("b", "thresh_mid", coef, NA, thresh_mid_prior,
              if (is_user_b("thresh_mid")) "user" else "default")
  }

  K <- model$model_data$K
  if (K > 2) {
    for (k in 1:(K-2)) {
      for (coef in crit_coefs) {
        add_prior("b", "log_gaps", paste0("gap", k, "_", coef), NA,
                  log_gaps_prior, if (is_user_b("log_gaps")) "user" else "default")
      }
    }
  }
  
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

  if (isTRUE(model$model_data$has_criterion2)) {
    thresh_mid2_prior <- get_fixed_prior(pl, "thresh_mid2")
    log_gaps2_prior <- get_fixed_prior(pl, "log_gaps2")

    crit2_coefs <- model$model_data$criterion2$coef_names
    for (coef in crit2_coefs) {
      add_prior("b", "thresh_mid2", coef, NA, thresh_mid2_prior,
                if (is_user_b("thresh_mid2")) "user" else "default")
    }

    K2 <- model$model_data$K2
    if (K2 > 2) {
      for (k in 1:(K2-2)) {
        for (coef in crit2_coefs) {
          add_prior("b", "log_gaps2", paste0("gap", k, "_", coef), NA,
                    log_gaps2_prior, if (is_user_b("log_gaps2")) "user" else "default")
        }
      }
    }
  }
  
  # --- Random effect SDs (class = "sd") ---
  
  is_user_sd_prior_coef <- function(pl, dpar, group, coef) {
    key_full <- paste0(dpar, "_", group, "_", coef)
    key_no_group <- paste0(dpar, "_", coef)
    !is.null(pl$sd_dpar_group_coef[[key_full]]) || !is.null(pl$sd_dpar_coef[[key_no_group]])
  }
  
  re_list <- model$model_data$dprime_random
  if (!is.null(re_list)) {
    for (group in names(re_list)) {
      re <- re_list[[group]]
      if (is.null(re$cor_id)) {
        level_names <- re$level_names
        if (is.null(level_names) || length(level_names) == 0) {
          level_names <- "(Intercept)"
        }
        for (coef in level_names) {
          sd_prior <- get_sd_prior(pl, "dprime", group, coef)
          source <- if (is_user_sd_prior_coef(pl, "dprime", group, coef)) {
            "user"
          } else if (is_user_sd_prior(pl, "dprime", group)) {
            "user"
          } else {
            "default"
          }
          add_prior("sd", dprime_dpar, coef, group, sd_prior, source)
        }
      }
    }
  }
  
  crit_re <- model$model_data$criterion$random
  if (!is.null(crit_re)) {
    for (group in names(crit_re)) {
      re <- crit_re[[group]]
      if (is.null(re$cor_id)) {
        n_re_terms <- if (!is.null(re$n_re_terms)) re$n_re_terms else 1
        
        if (n_re_terms == 1) {
          level_names <- "(Intercept)"
        } else {
          level_names <- re$level_names
          if (is.null(level_names)) {
            level_names <- paste0("term", 1:n_re_terms)
          }
        }
        
        for (coef in level_names) {
          sd_prior <- get_sd_prior(pl, "criterion", group, coef)
          source <- if (is_user_sd_prior_coef(pl, "criterion", group, coef)) {
            "user"
          } else if (is_user_sd_prior(pl, "criterion", group)) {
            "user"
          } else {
            "default"
          }
          add_prior("sd", crit_dpar, coef, group, sd_prior, source)
        }
      }
    }
  }
  
  if (model$model_data$has_sigma) {
    sigma_re <- model$model_data$sigma_random
    if (!is.null(sigma_re)) {
      for (group in names(sigma_re)) {
        re <- sigma_re[[group]]
        if (is.null(re$cor_id)) {
          level_names <- re$level_names
          if (is.null(level_names) || length(level_names) == 0) {
            level_names <- "(Intercept)"
          }
          for (coef in level_names) {
            sd_prior <- get_sd_prior(pl, "sigma", group, coef)
            source <- if (is_user_sd_prior_coef(pl, "sigma", group, coef)) {
              "user"
            } else if (is_user_sd_prior(pl, "sigma", group)) {
              "user"
            } else {
              "default"
            }
            add_prior("sd", "sigma", coef, group, sd_prior, source)
          }
        }
      }
    }
  }
  
  if (model$model_data$has_lambda) {
    lambda_re <- model$model_data$lambda_random
    if (!is.null(lambda_re)) {
      for (group in names(lambda_re)) {
        re <- lambda_re[[group]]
        if (is.null(re$cor_id)) {
          level_names <- re$level_names
          if (is.null(level_names) || length(level_names) == 0) {
            level_names <- "(Intercept)"
          }
          for (coef in level_names) {
            sd_prior <- get_sd_prior(pl, "lambda", group, coef)
            source <- if (is_user_sd_prior_coef(pl, "lambda", group, coef)) {
              "user"
            } else if (is_user_sd_prior(pl, "lambda", group)) {
              "user"
            } else {
              "default"
            }
            add_prior("sd", "lambda", coef, group, sd_prior, source)
          }
        }
      }
    }
  }
  
  if (isTRUE(model$model_data$has_lambda2)) {
    lambda2_re <- model$model_data$lambda2_random
    if (!is.null(lambda2_re)) {
      for (group in names(lambda2_re)) {
        re <- lambda2_re[[group]]
        if (is.null(re$cor_id)) {
          level_names <- re$level_names
          if (is.null(level_names) || length(level_names) == 0) {
            level_names <- "(Intercept)"
          }
          for (coef in level_names) {
            sd_prior <- get_sd_prior(pl, "lambda2", group, coef)
            source <- if (is_user_sd_prior_coef(pl, "lambda2", group, coef)) {
              "user"
            } else if (is_user_sd_prior(pl, "lambda2", group)) {
              "user"
            } else {
              "default"
            }
            add_prior("sd", "lambda2", coef, group, sd_prior, source)
          }
        }
      }
    }
  }

  if (isTRUE(model$model_data$has_lambda2_B)) {
    lambda2_B_re <- model$model_data$lambda2_B_random
    if (!is.null(lambda2_B_re)) {
      for (group in names(lambda2_B_re)) {
        re <- lambda2_B_re[[group]]
        if (is.null(re$cor_id)) {
          level_names <- re$level_names
          if (is.null(level_names) || length(level_names) == 0) {
            level_names <- "(Intercept)"
          }
          for (coef in level_names) {
            sd_prior <- get_sd_prior(pl, "lambda2_B", group, coef)
            source <- if (is_user_sd_prior_coef(pl, "lambda2_B", group, coef)) {
              "user"
            } else if (is_user_sd_prior(pl, "lambda2_B", group)) {
              "user"
            } else {
              "default"
            }
            add_prior("sd", "lambda2_B", coef, group, sd_prior, source)
          }
        }
      }
    }
  }

  if (isTRUE(model$model_data$has_dprime_B)) {
    dprime_B_re <- model$model_data$dprime_B_random
    if (!is.null(dprime_B_re)) {
      for (group in names(dprime_B_re)) {
        re <- dprime_B_re[[group]]
        if (is.null(re$cor_id)) {
          level_names <- re$level_names
          if (is.null(level_names) || length(level_names) == 0) {
            level_names <- "(Intercept)"
          }
          for (coef in level_names) {
            sd_prior <- get_sd_prior(pl, "dprime_B", group, coef)
            source <- if (is_user_sd_prior_coef(pl, "dprime_B", group, coef)) {
              "user"
            } else if (is_user_sd_prior(pl, "dprime_B", group)) {
              "user"
            } else {
              "default"
            }
            add_prior("sd", "dprime_B", coef, group, sd_prior, source)
          }
        }
      }
    }
  }
  
  if (isTRUE(model$model_data$has_lambda_B)) {
    lambda_B_re <- model$model_data$lambda_B_random
    if (!is.null(lambda_B_re)) {
      for (group in names(lambda_B_re)) {
        re <- lambda_B_re[[group]]
        if (is.null(re$cor_id)) {
          level_names <- re$level_names
          if (is.null(level_names) || length(level_names) == 0) {
            level_names <- "(Intercept)"
          }
          for (coef in level_names) {
            sd_prior <- get_sd_prior(pl, "lambda_B", group, coef)
            source <- if (is_user_sd_prior_coef(pl, "lambda_B", group, coef)) {
              "user"
            } else if (is_user_sd_prior(pl, "lambda_B", group)) {
              "user"
            } else {
              "default"
            }
            add_prior("sd", "lambda_B", coef, group, sd_prior, source)
          }
        }
      }
    }
  }
  
  
  if (isTRUE(model$model_data$has_discrim)) {
    discrim_re <- model$model_data$discrim_random
    if (!is.null(discrim_re)) {
      for (group in names(discrim_re)) {
        re <- discrim_re[[group]]
        if (is.null(re$cor_id)) {
          level_names <- re$level_names
          if (is.null(level_names) || length(level_names) == 0) level_names <- "(Intercept)"
          for (coef in level_names) {
            sd_prior <- get_sd_prior(pl, "discrim", group, coef)
            source <- if (is_user_sd_prior_coef(pl, "discrim", group, coef)) "user" 
            else if (is_user_sd_prior(pl, "discrim", group)) "user" else "default"
            add_prior("sd", "discrim", coef, group, sd_prior, source)
          }
        }
      }
    }
  }
  
  if (isTRUE(model$model_data$has_discrim_B)) {
    discrim_B_re <- model$model_data$discrim_B_random
    if (!is.null(discrim_B_re)) {
      for (group in names(discrim_B_re)) {
        re <- discrim_B_re[[group]]
        if (is.null(re$cor_id)) {
          level_names <- re$level_names
          if (is.null(level_names) || length(level_names) == 0) level_names <- "(Intercept)"
          for (coef in level_names) {
            sd_prior <- get_sd_prior(pl, "discrim_B", group, coef)
            source <- if (is_user_sd_prior_coef(pl, "discrim_B", group, coef)) "user" 
            else if (is_user_sd_prior(pl, "discrim_B", group)) "user" else "default"
            add_prior("sd", "discrim_B", coef, group, sd_prior, source)
          }
        }
      }
    }
  }
  
  if (isTRUE(model$model_data$has_sigma_B)) {
    sigma_B_re <- model$model_data$sigma_B_random
    if (!is.null(sigma_B_re)) {
      for (group in names(sigma_B_re)) {
        re <- sigma_B_re[[group]]
        if (is.null(re$cor_id)) {
          level_names <- re$level_names
          if (is.null(level_names) || length(level_names) == 0) level_names <- "(Intercept)"
          for (coef in level_names) {
            sd_prior <- get_sd_prior(pl, "sigma_B", group, coef)
            source <- if (is_user_sd_prior_coef(pl, "sigma_B", group, coef)) "user" 
            else if (is_user_sd_prior(pl, "sigma_B", group)) "user" else "default"
            add_prior("sd", "sigma_B", coef, group, sd_prior, source)
          }
        }
      }
    }
  }
  
  if (isTRUE(model$model_data$has_sigma2)) {
    sigma2_re <- model$model_data$sigma2_random
    if (!is.null(sigma2_re)) {
      for (group in names(sigma2_re)) {
        re <- sigma2_re[[group]]
        if (is.null(re$cor_id)) {
          level_names <- re$level_names
          if (is.null(level_names) || length(level_names) == 0) level_names <- "(Intercept)"
          for (coef in level_names) {
            sd_prior <- get_sd_prior(pl, "sigma2", group, coef)
            source <- if (is_user_sd_prior_coef(pl, "sigma2", group, coef)) "user" 
            else if (is_user_sd_prior(pl, "sigma2", group)) "user" else "default"
            add_prior("sd", "sigma2", coef, group, sd_prior, source)
          }
        }
      }
    }
  }
  
  if (isTRUE(model$model_data$has_sigma2_B)) {
    sigma2_B_re <- model$model_data$sigma2_B_random
    if (!is.null(sigma2_B_re)) {
      for (group in names(sigma2_B_re)) {
        re <- sigma2_B_re[[group]]
        if (is.null(re$cor_id)) {
          level_names <- re$level_names
          if (is.null(level_names) || length(level_names) == 0) level_names <- "(Intercept)"
          for (coef in level_names) {
            sd_prior <- get_sd_prior(pl, "sigma2_B", group, coef)
            source <- if (is_user_sd_prior_coef(pl, "sigma2_B", group, coef)) "user" 
            else if (is_user_sd_prior(pl, "sigma2_B", group)) "user" else "default"
            add_prior("sd", "sigma2_B", coef, group, sd_prior, source)
          }
        }
      }
    }
  }
  
  if (isTRUE(model$model_data$has_rho)) {
    rho_re <- model$model_data$rho_random
    if (!is.null(rho_re)) {
      for (group in names(rho_re)) {
        re <- rho_re[[group]]
        if (is.null(re$cor_id)) {
          level_names <- re$level_names
          if (is.null(level_names) || length(level_names) == 0) level_names <- "(Intercept)"
          for (coef in level_names) {
            sd_prior <- get_sd_prior(pl, "rho", group, coef)
            source <- if (is_user_sd_prior_coef(pl, "rho", group, coef)) "user" 
            else if (is_user_sd_prior(pl, "rho", group)) "user" else "default"
            add_prior("sd", "rho", coef, group, sd_prior, source)
          }
        }
      }
    }
  }
  
  if (isTRUE(model$model_data$has_rho_B)) {
    rho_B_re <- model$model_data$rho_B_random
    if (!is.null(rho_B_re)) {
      for (group in names(rho_B_re)) {
        re <- rho_B_re[[group]]
        if (is.null(re$cor_id)) {
          level_names <- re$level_names
          if (is.null(level_names) || length(level_names) == 0) level_names <- "(Intercept)"
          for (coef in level_names) {
            sd_prior <- get_sd_prior(pl, "rho_B", group, coef)
            source <- if (is_user_sd_prior_coef(pl, "rho_B", group, coef)) "user" 
            else if (is_user_sd_prior(pl, "rho_B", group)) "user" else "default"
            add_prior("sd", "rho_B", coef, group, sd_prior, source)
          }
        }
      }
    }
  }
  
  if (isTRUE(model$model_data$has_rho_N)) {
    rho_N_re <- model$model_data$rho_N_random
    if (!is.null(rho_N_re)) {
      for (group in names(rho_N_re)) {
        re <- rho_N_re[[group]]
        if (is.null(re$cor_id)) {
          level_names <- re$level_names
          if (is.null(level_names) || length(level_names) == 0) level_names <- "(Intercept)"
          for (coef in level_names) {
            sd_prior <- get_sd_prior(pl, "rho_N", group, coef)
            source <- if (is_user_sd_prior_coef(pl, "rho_N", group, coef)) "user" 
            else if (is_user_sd_prior(pl, "rho_N", group)) "user" else "default"
            add_prior("sd", "rho_N", coef, group, sd_prior, source)
          }
        }
      }
    }
  }
  
  if (isTRUE(model$model_data$has_rec_crit)) {
    rec_crit_re <- model$model_data$rec_crit_random
    if (!is.null(rec_crit_re)) {
      for (group in names(rec_crit_re)) {
        re <- rec_crit_re[[group]]
        if (is.null(re$cor_id)) {
          level_names <- re$level_names
          if (is.null(level_names) || length(level_names) == 0) {
            level_names <- "(Intercept)"
          }
          for (coef in level_names) {
            sd_prior <- get_sd_prior(pl, "rec_crit", group, coef)
            source <- if (is_user_sd_prior_coef(pl, "rec_crit", group, coef)) {
              "user"
            } else if (is_user_sd_prior(pl, "rec_crit", group)) {
              "user"
            } else {
              "default"
            }
            add_prior("sd", "rec_crit", coef, group, sd_prior, source)
          }
        }
      }
    }
  }
  
  if (isTRUE(model$model_data$has_know_crit)) {
    know_crit_re <- model$model_data$know_crit_random
    if (!is.null(know_crit_re)) {
      for (group in names(know_crit_re)) {
        re <- know_crit_re[[group]]
        if (is.null(re$cor_id)) {
          level_names <- re$level_names
          if (is.null(level_names) || length(level_names) == 0) {
            level_names <- "(Intercept)"
          }
          for (coef in level_names) {
            sd_prior <- get_sd_prior(pl, "know_crit", group, coef)
            source <- if (is_user_sd_prior_coef(pl, "know_crit", group, coef)) {
              "user"
            } else if (is_user_sd_prior(pl, "know_crit", group)) {
              "user"
            } else {
              "default"
            }
            add_prior("sd", "know_crit", coef, group, sd_prior, source)
          }
        }
      }
    }
  }

  if (isTRUE(model$model_data$has_dprime_L)) {
    dprime_L_re <- model$model_data$dprime_L_random
    if (!is.null(dprime_L_re)) {
      for (group in names(dprime_L_re)) {
        re <- dprime_L_re[[group]]
        if (is.null(re$cor_id)) {
          level_names <- re$level_names
          if (is.null(level_names) || length(level_names) == 0) level_names <- "(Intercept)"
          for (coef in level_names) {
            sd_prior <- get_sd_prior(pl, "dprime_L", group, coef)
            source <- if (is_user_sd_prior_coef(pl, "dprime_L", group, coef)) "user" 
            else if (is_user_sd_prior(pl, "dprime_L", group)) "user" else "default"
            add_prior("sd", "dprime_L", coef, group, sd_prior, source)
          }
        }
      }
    }
  }
  
  if (isTRUE(model$model_data$has_sigma_L)) {
    sigma_L_re <- model$model_data$sigma_L_random
    if (!is.null(sigma_L_re)) {
      for (group in names(sigma_L_re)) {
        re <- sigma_L_re[[group]]
        if (is.null(re$cor_id)) {
          level_names <- re$level_names
          if (is.null(level_names) || length(level_names) == 0) level_names <- "(Intercept)"
          for (coef in level_names) {
            sd_prior <- get_sd_prior(pl, "sigma_L", group, coef)
            source <- if (is_user_sd_prior_coef(pl, "sigma_L", group, coef)) "user" 
            else if (is_user_sd_prior(pl, "sigma_L", group)) "user" else "default"
            add_prior("sd", "sigma_L", coef, group, sd_prior, source)
          }
        }
      }
    }
  }
  
  if (isTRUE(model$model_data$has_lambda_L)) {
    lambda_L_re <- model$model_data$lambda_L_random
    if (!is.null(lambda_L_re)) {
      for (group in names(lambda_L_re)) {
        re <- lambda_L_re[[group]]
        if (is.null(re$cor_id)) {
          level_names <- re$level_names
          if (is.null(level_names) || length(level_names) == 0) level_names <- "(Intercept)"
          for (coef in level_names) {
            sd_prior <- get_sd_prior(pl, "lambda_L", group, coef)
            source <- if (is_user_sd_prior_coef(pl, "lambda_L", group, coef)) "user" 
            else if (is_user_sd_prior(pl, "lambda_L", group)) "user" else "default"
            add_prior("sd", "lambda_L", coef, group, sd_prior, source)
          }
        }
      }
    }
  }
  
  if (isTRUE(model$model_data$has_criterion2)) {
    crit2_re <- model$model_data$criterion2$random
    if (!is.null(crit2_re)) {
      for (group in names(crit2_re)) {
        re <- crit2_re[[group]]
        if (is.null(re$cor_id)) {
          n_re_terms <- if (!is.null(re$n_re_terms)) re$n_re_terms else 1
          if (n_re_terms == 1) {
            level_names <- "(Intercept)"
          } else {
            level_names <- re$level_names
            if (is.null(level_names)) level_names <- paste0("term", 1:n_re_terms)
          }
          for (coef in level_names) {
            sd_prior <- get_sd_prior(pl, "criterion", group, coef)
            source <- if (is_user_sd_prior_coef(pl, "criterion", group, coef)) "user" 
            else if (is_user_sd_prior(pl, "criterion", group)) "user" else "default"
            add_prior("sd", "criterion2", coef, group, sd_prior, source)
          }
        }
      }
    }
  }
  
  # --- Correlations (class = "cor") ---
  
  if (!is.null(re_list)) {
    for (group in names(re_list)) {
      re <- re_list[[group]]
      if (is.null(re$cor_id) && re$dim > 1) {
        cor_prior <- get_cor_prior(pl, "dprime", group)
        source <- if (is_user_cor_prior(pl, "dprime", group)) "user" else "default"
        add_prior("cor", dprime_dpar, NA, group, cor_prior, source)
      }
    }
  }
  
  if (!is.null(crit_re)) {
    for (group in names(crit_re)) {
      re <- crit_re[[group]]
      if (is.null(re$cor_id) && re$dim > 1) {
        cor_prior <- get_cor_prior(pl, "criterion", group)
        source <- if (is_user_cor_prior(pl, "criterion", group)) "user" else "default"
        add_prior("cor", crit_dpar, NA, group, cor_prior, source)
      }
    }
  }
  
  if (model$model_data$has_sigma) {
    sigma_re <- model$model_data$sigma_random
    if (!is.null(sigma_re)) {
      for (group in names(sigma_re)) {
        re <- sigma_re[[group]]
        if (is.null(re$cor_id) && re$dim > 1) {
          cor_prior <- get_cor_prior(pl, "sigma", group)
          source <- if (is_user_cor_prior(pl, "sigma", group)) "user" else "default"
          add_prior("cor", "sigma", NA, group, cor_prior, source)
        }
      }
    }
  }
  
  if (model$model_data$has_lambda) {
    lambda_re <- model$model_data$lambda_random
    if (!is.null(lambda_re)) {
      for (group in names(lambda_re)) {
        re <- lambda_re[[group]]
        if (is.null(re$cor_id) && re$dim > 1) {
          cor_prior <- get_cor_prior(pl, "lambda", group)
          source <- if (is_user_cor_prior(pl, "lambda", group)) "user" else "default"
          add_prior("cor", "lambda", NA, group, cor_prior, source)
        }
      }
    }
  }
  
  if (isTRUE(model$model_data$has_lambda2)) {
    lambda2_re <- model$model_data$lambda2_random
    if (!is.null(lambda2_re)) {
      for (group in names(lambda2_re)) {
        re <- lambda2_re[[group]]
        if (is.null(re$cor_id) && re$dim > 1) {
          cor_prior <- get_cor_prior(pl, "lambda2", group)
          source <- if (is_user_cor_prior(pl, "lambda2", group)) "user" else "default"
          add_prior("cor", "lambda2", NA, group, cor_prior, source)
        }
      }
    }
  }

  if (isTRUE(model$model_data$has_lambda2_B)) {
    lambda2_B_re <- model$model_data$lambda2_B_random
    if (!is.null(lambda2_B_re)) {
      for (group in names(lambda2_B_re)) {
        re <- lambda2_B_re[[group]]
        if (is.null(re$cor_id) && re$dim > 1) {
          cor_prior <- get_cor_prior(pl, "lambda2_B", group)
          source <- if (is_user_cor_prior(pl, "lambda2_B", group)) "user" else "default"
          add_prior("cor", "lambda2_B", NA, group, cor_prior, source)
        }
      }
    }
  }

  if (isTRUE(model$model_data$has_dprime_B)) {
    dprime_B_re <- model$model_data$dprime_B_random
    if (!is.null(dprime_B_re)) {
      for (group in names(dprime_B_re)) {
        re <- dprime_B_re[[group]]
        if (is.null(re$cor_id) && re$dim > 1) {
          cor_prior <- get_cor_prior(pl, "dprime_B", group)
          source <- if (is_user_cor_prior(pl, "dprime_B", group)) "user" else "default"
          add_prior("cor", "dprime_B", NA, group, cor_prior, source)
        }
      }
    }
  }
  
  if (isTRUE(model$model_data$has_lambda_B)) {
    lambda_B_re <- model$model_data$lambda_B_random
    if (!is.null(lambda_B_re)) {
      for (group in names(lambda_B_re)) {
        re <- lambda_B_re[[group]]
        if (is.null(re$cor_id) && re$dim > 1) {
          cor_prior <- get_cor_prior(pl, "lambda_B", group)
          source <- if (is_user_cor_prior(pl, "lambda_B", group)) "user" else "default"
          add_prior("cor", "lambda_B", NA, group, cor_prior, source)
        }
      }
    }
  }
  
  if (isTRUE(model$model_data$has_dprime_L)) {
    dprime_L_re <- model$model_data$dprime_L_random
    if (!is.null(dprime_L_re)) {
      for (group in names(dprime_L_re)) {
        re <- dprime_L_re[[group]]
        if (is.null(re$cor_id) && re$dim > 1) {
          cor_prior <- get_cor_prior(pl, "dprime_L", group)
          source <- if (is_user_cor_prior(pl, "dprime_L", group)) "user" else "default"
          add_prior("cor", "dprime_L", NA, group, cor_prior, source)
        }
      }
    }
  }
  
  if (isTRUE(model$model_data$has_sigma_L)) {
    sigma_L_re <- model$model_data$sigma_L_random
    if (!is.null(sigma_L_re)) {
      for (group in names(sigma_L_re)) {
        re <- sigma_L_re[[group]]
        if (is.null(re$cor_id) && re$dim > 1) {
          cor_prior <- get_cor_prior(pl, "sigma_L", group)
          source <- if (is_user_cor_prior(pl, "sigma_L", group)) "user" else "default"
          add_prior("cor", "sigma_L", NA, group, cor_prior, source)
        }
      }
    }
  }
  
  if (isTRUE(model$model_data$has_lambda_L)) {
    lambda_L_re <- model$model_data$lambda_L_random
    if (!is.null(lambda_L_re)) {
      for (group in names(lambda_L_re)) {
        re <- lambda_L_re[[group]]
        if (is.null(re$cor_id) && re$dim > 1) {
          cor_prior <- get_cor_prior(pl, "lambda_L", group)
          source <- if (is_user_cor_prior(pl, "lambda_L", group)) "user" else "default"
          add_prior("cor", "lambda_L", NA, group, cor_prior, source)
        }
      }
    }
  }
  
  for (cor_id in names(model$model_data$cross_cor)) {
    cc <- model$model_data$cross_cor[[cor_id]]
    
    for (member in cc$members) {
      param <- member$param
      group <- member$group
      dim <- member$dim
      
      if (param == "criterion") {
        re <- model$model_data$criterion$random[[group]]
      } else {
        re_list <- model$model_data[[paste0(param, "_random")]]
        re <- if (!is.null(re_list)) re_list[[group]] else NULL
      }
      
      if (!is.null(re)) {
        level_names <- re$level_names
        if (is.null(level_names) || length(level_names) == 0) {
          if (param == "criterion") {
            n_thresh <- re$dim
            level_names <- paste0("thresh", 1:n_thresh)
          } else {
            level_names <- "(Intercept)"
          }
        }
        
        for (coef in level_names) {
          sd_prior <- get_sd_prior(pl, param, group, coef)
          source <- if (is_user_sd_prior_coef(pl, param, group, coef)) {
            "user"
          } else if (is_user_sd_prior(pl, param, group)) {
            "user"
          } else {
            "default"
          }
          add_prior("sd", param, coef, group, sd_prior, source)
        }
      }
    }
    
    cor_prior <- get_cor_prior(pl, "cross", cc$group)
    source <- "default"
    add_prior("cor", paste0("cross_", cor_id), NA, cc$group, cor_prior, source)
  }

  # --- Smooth penalty SDs (class = "sds") ---
  sm_data <- model$model_data$smooth_data
  if (!is.null(sm_data) && length(sm_data) > 0) {
    is_user_sds <- function(dpar_, key_) {
      !is.null(pl$sds_coef[[paste0(dpar_, "_", key_)]]) ||
      !is.null(pl$sds_dpar[[dpar_]])
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
