#' SDT Model Family Specifications
#'
#' These functions define the structure of different SDT model families.


# =============================================================================
# Internal <-> external parameter name map
# =============================================================================
# All families store `family$params` keyed by INTERNAL Stan/JAX names. Some
# families also expose user-facing EXTERNAL names that appear in formulas:
#
#   cdp:                    dprime/dprime2/sigma/sigma2 displayed as rec/fam/sigma_R/sigma_F
#   cumulative:             dprime/criterion            displayed as mu/cutpoints
#   bivariate_cumulative:   dprime/discrim/criterion/criterion2 displayed as mu1/mu2/cutpoints1/cutpoints2
#
# The mapping below is the single source of truth. Family constructors copy
# it into `family$external_aliases`, and the print method consults that to
# show user-facing names. `external_to_internal_param()` below maps a
# user-facing formula name back to the internal name (used by posterior.R).

#' @noRd
.param_name_map <- function(family_name) {
  switch(family_name,
    cdp                  = list(dprime = "rec", dprime2 = "fam",
                                sigma  = "sigma_R", sigma2 = "sigma_F"),
    cumulative           = list(dprime = "mu", criterion = "cutpoints"),
    bivariate_cumulative = list(dprime = "mu1", discrim = "mu2",
                                criterion = "cutpoints1",
                                criterion2 = "cutpoints2"),
    list()
  )
}

#' @noRd
external_to_internal_param <- function(external_name, family_name) {
  m <- .param_name_map(family_name)
  if (length(m) == 0) return(external_name)
  inv <- setNames(names(m), unlist(m, use.names = FALSE))
  hit <- inv[external_name]  # named-vector lookup: missing -> NA, not error
  if (is.na(hit)) external_name else unname(hit)
}


# =============================================================================
# Link Functions
# =============================================================================

#' Probit Link Function (default)
#'
#' Uses the standard normal CDF.
#' @return A `broc_link` object to pass to a family's `link` argument.
#' @export
link_probit <- function() {
  structure(list(name = "probit"), class = "broc_link")
}

#' Logistic Link Function
#'
#' Uses the logistic CDF.
#' @return A `broc_link` object to pass to a family's `link` argument.
#' @export
link_logit <- function() {
  structure(list(name = "logit"), class = "broc_link")
}

#' Print method for a bayesroc link
#' @param x The object to print.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @export
print.broc_link <- function(x, ...) {
  cat("Link:", x$name, "\n")
  invisible(x)
}


# =============================================================================
# SDT Families
# =============================================================================

#' Equal Variance Signal Detection Model
#'
#' Canonical signal detection model. Signal and noise distributions are assumed
#' normal with equal variance fixed at 1 (Macmillan & Creelman, 2005).
#'
#' @param link Link function (default: [link_probit()]). Also supports
#'   [link_logit()] for logistic CDF.
#'
#' @details
#' ## Likelihood
#' For an old/new judgment Y on a K-point confidence scale (`old` in {0, 1})
#' with criteria c_1 < c_2 < ... < c_{K-1}:
#' \deqn{P(Y \le k \mid \mathrm{old}) = \Phi(c_k - \mathrm{old} \cdot d')}
#' for k = 1, ..., K-1, where \eqn{\Phi} is the link CDF (probit by default).
#' Cell probabilities by differencing:
#' \eqn{P(Y = k \mid \mathrm{old}) = P(Y \le k \mid \mathrm{old}) - P(Y \le k - 1 \mid \mathrm{old})}
#' with \eqn{P(Y \le 0) = 0} and \eqn{P(Y \le K) = 1}. Both new- and
#' old-item distributions have unit SD.
#'
#' ## Parameters
#' \describe{
#'   \item{dprime}{(required) Sensitivity. Identity link.}
#'   \item{criterion}{(required) K-1 confidence thresholds, parameterized
#'     internally as a mid-anchor `thresh_mid` plus K-2 ordered gaps. Identity
#'     link on the gap parameterization.}
#' }
#'
#' ## Formula syntax
#' ```
#' broc(brf(conf | old ~ condition + (1 + condition | subj),
#'          criterion ~ 1 + (1 | subj),
#'          family = evsd()),
#'     data = dat)
#' ```
#'
#' @references
#' Macmillan, N. A., & Creelman, C. D. (2005). *Detection theory: A user's
#'   guide* (2nd ed.). Lawrence Erlbaum.
#'
#' @return A `broc_family` object.
#' @seealso [uvsd()] for unequal-variance extension, [dpsd()] for dual-process,
#'   [mixture()] for mixture variants.
#' @export
evsd <- function(link = link_probit()) {
  if (!inherits(link, "broc_link")) {
    stop("link must be created with link_probit() or link_logit()")
  }
  structure(
    list(
      family = "evsdt",
      link = link,
      params = list(
        dprime = list(required = TRUE, default = NULL, link = "identity"),
        criterion = list(required = TRUE, default = NULL, link = "identity")
      ),
      description = "Equal Variance Signal Detection Model"
    ),
    class = "broc_family"
  )
}

#' Unequal Variance Signal Detection Model
#'
#' Extends [evsd()] by allowing the signal (old-item) distribution to have a
#' different variance than the noise (new-item) distribution (Macmillan &
#' Creelman, 2005).
#'
#' @param link Link function (default: [link_probit()]).
#'
#' @details
#' ## Likelihood
#' Noise distribution N(0, 1); signal distribution N(d', sigma). For an
#' old/new judgment Y on a K-point scale (`old` in {0, 1}) with criteria
#' c_1 < ... < c_{K-1}:
#' \deqn{P(Y \le k \mid \mathrm{old} = 0) = \Phi(c_k)}
#' \deqn{P(Y \le k \mid \mathrm{old} = 1) = \Phi\!\Big(\frac{c_k - d'}{\sigma}\Big)}
#' for k = 1, ..., K-1.
#'
#' ## Parameters
#' \describe{
#'   \item{dprime}{(required) Sensitivity. Identity link.}
#'   \item{criterion}{(required) K-1 confidence thresholds (mid-anchor +
#'     K-2 ordered gaps). Identity link.}
#'   \item{sigma}{(required) SD of the signal distribution. Log link. To fix
#'     sigma at 1, use [evsd()] instead.}
#' }
#'
#' ## Formula syntax
#' ```
#' broc(brf(conf | old ~ condition + (1 | subj),
#'          criterion ~ 1 + (1 | subj),
#'          sigma ~ 1 + (1 | subj),
#'          family = uvsd()),
#'     data = dat)
#' ```
#'
#' @references
#' Macmillan, N. A., & Creelman, C. D. (2005). *Detection theory: A user's
#'   guide* (2nd ed.). Lawrence Erlbaum.
#'
#' @return A `broc_family` object.
#' @seealso [evsd()], [dpsd()], [mixture()].
#' @export
uvsd <- function(link = link_probit()) {
  if (!inherits(link, "broc_link")) {
    stop("link must be created with link_probit() or link_logit()")
  }
  structure(
    list(
      family = "uvsdt",
      link = link,
      params = list(
        dprime = list(required = TRUE, default = NULL, link = "identity"),
        criterion = list(required = TRUE, default = NULL, link = "identity"),
        sigma = list(required = TRUE, default = NULL, link = "log")
      ),
      description = "Unequal Variance Signal Detection Model"
    ),
    class = "broc_family"
  )
}

#' Dual Process Signal Detection Model
#'
#' Recognition-memory model combining discrete recollection and continuous (UVSD)
#' familiarity (Yonelinas, 1994).
#'
#' @param link Link function (default: [link_probit()]).
#'
#' @details
#' ## Likelihood
#' For new items, responses follow standard EVSD (no recollection):
#' \deqn{P(Y \le k \mid \mathrm{old} = 0) = \Phi(c_k)}
#' For old items, recollection succeeds with probability lambda, producing the
#' highest-confidence "old" response Y = K. Otherwise, the familiarity-only
#' distribution is UVSD with mean d' and SD sigma:
#' \deqn{P(Y \le k \mid \mathrm{old} = 1) = (1 - \lambda)\,\Phi\!\Big(\frac{c_k - d'}{\sigma}\Big)
#'       \quad (k < K)}
#' \deqn{P(Y \le K \mid \mathrm{old} = 1) = 1}
#'
#' ## Parameters
#' \describe{
#'   \item{dprime}{(required) Familiarity sensitivity. Identity link.}
#'   \item{criterion}{(required) K-1 thresholds. Identity link.}
#'   \item{lambda}{(required) Recollection probability. Logit link.}
#'   \item{sigma}{(optional) SD of the familiarity signal distribution. If
#'     omitted, fixed at 1 (= EVSD familiarity). Log link.}
#' }
#'
#' ## Formula syntax
#' ```
#' broc(brf(conf | old ~ 1 + (1 | subj),
#'          criterion ~ 1 + (1 | subj),
#'          lambda ~ condition + (1 | subj),
#'          family = dpsd()),
#'     data = dat)
#' ```
#'
#' @references
#' Yonelinas, A. P. (1994). Receiver-operating characteristics in recognition
#'   memory: Evidence for a dual-process model. *Journal of Experimental
#'   Psychology: Learning, Memory, and Cognition*, *20*(6), 1341-1354.
#'   https://doi.org/10.1037/0278-7393.20.6.1341
#'
#' @return A `broc_family` object.
#' @seealso [evsd()], [uvsd()], [mixture()], [bivariate_dp()].
#' @export
dpsd <- function(link = link_probit()) {
  if (!inherits(link, "broc_link")) {
    stop("link must be created with link_probit() or link_logit()")
  }
  structure(
    list(
      family = "dpsdt",
      link = link,
      params = list(
        dprime = list(required = TRUE, default = NULL, link = "identity"),
        criterion = list(required = TRUE, default = NULL, link = "identity"),
        lambda = list(required = TRUE, default = NULL, link = "logit",
                      description = "recollection probability"),
        sigma = list(required = FALSE, default = NULL, link = "log",
                     null_value = 1, null_description = "fixed at 1")
      ),
      description = "Dual Process Signal Detection Model"
    ),
    class = "broc_family"
  )
}

#' Mixture Signal Detection Model
#'
#' Two-state mixture model for old items, combining a high- and a low-strength
#' Gaussian component (DeCarlo, 2002).
#'
#' @param link Link function (default: [link_probit()]).
#'
#' @details
#' ## Likelihood
#' Mixture of two latent classes (DeCarlo, 2002), expressed cumulatively:
#' \deqn{P(Y \le k \mid \mathrm{old}) = \lambda\,\Phi\!\Big(\tfrac{c_k - d'_1 \cdot \mathrm{old}}{\sigma_1}\Big)
#'       + (1 - \lambda)\,\Phi\!\Big(\tfrac{c_k - d'_2 \cdot \mathrm{old}}{\sigma_2}\Big)}
#' where lambda is the mixing weight on state 1 (`lambda`),
#' \eqn{(d'_1, \sigma_1)} = (`dprime`, `sigma`), and
#' \eqn{(d'_2, \sigma_2)} = (`dprime2`, `sigma2`). At `old = 0` both
#' Gaussians collapse to the noise N(0, 1) baseline, so new items reduce to
#' standard EVSD.
#'
#' If a lure mixture is specified (both `dprime_L` and `lambda_L`), new
#' items get their own mixture:
#' \deqn{P(Y \le k \mid \mathrm{old} = 0) = (1 - \lambda_L)\,\Phi(c_k)
#'       + \lambda_L\,\Phi\!\Big(\tfrac{c_k - d'_L}{\sigma_L}\Big)}
#'
#' When both `dprime` and `dprime2` are free, an ordered constraint
#' (`dprime > dprime2`) is imposed for identifiability.
#'
#' ## Parameters
#' The `_L` suffix marks the optional lure-item mixture component (applied to
#' new items); the unsuffixed parameters describe the old-item mixture.
#' \describe{
#'   \item{dprime}{(required) \eqn{d'_1} for the high (state 1) old-item
#'     component. Identity link.}
#'   \item{criterion}{(required) K-1 thresholds. Identity link.}
#'   \item{lambda}{(required) Mixing weight on state 1. Logit link.}
#'   \item{dprime2}{(optional) \eqn{d'_2} for state 2. If omitted, fixed at 0.
#'     Identity link.}
#'   \item{sigma}{(optional) SD \eqn{\sigma_1} of state 1. If omitted, fixed at 1.
#'     Log link.}
#'   \item{sigma2}{(optional) SD \eqn{\sigma_2} of state 2. If omitted, fixed at 1.
#'     Log link.}
#'   \item{dprime_L}{(optional) d' for an additional lure-item mixture
#'     component (new items only). Identity link. If omitted, no lure mixture
#'     (new items are standard EVSD N(0, 1)).}
#'   \item{lambda_L}{(optional) Mixing weight on the lure-mixture component
#'     (new items only). Logit link. Required (with `dprime_L`) to enable the
#'     lure mixture.}
#'   \item{sigma_L}{(optional) SD of the lure-mixture component. If omitted,
#'     fixed at 1. Log link.}
#' }
#'
#' ## Formula syntax
#' ```
#' broc(brf(conf | old ~ 1 + (1 | subj),
#'          criterion ~ 1 + (1 | subj),
#'          lambda ~ condition + (1 | subj),
#'          dprime2 ~ 1,
#'          family = mixture()),
#'     data = dat)
#' ```
#'
#' @references
#' DeCarlo, L. T. (2002). Signal detection theory with finite mixture
#'   distributions: Theoretical developments with applications to recognition
#'   memory. *Psychological Review*, *109*(4), 710-721.
#'   https://doi.org/10.1037/0033-295X.109.4.710
#'
#' @return A `broc_family` object.
#' @seealso [dpsd()], [uvsd()], [evsd()].
#' @export
mixture <- function(link = link_probit()) {
  if (!inherits(link, "broc_link")) {
    stop("link must be created with link_probit() or link_logit()")
  }
  structure(
    list(
      family = "mixture",
      link = link,
      params = list(
        dprime = list(required = TRUE, default = NULL, link = "identity",
                      description = "d' for state 1 (high state)"),
        criterion = list(required = TRUE, default = NULL, link = "identity"),
        lambda = list(required = TRUE, default = NULL, link = "logit",
                      description = "probability of state 1"),
        sigma = list(required = FALSE, default = NULL, link = "log",
                     null_value = 1, null_description = "sigma1 fixed at 1"),
        dprime2 = list(required = FALSE, default = NULL, link = "identity",
                       null_value = 0, null_description = "fixed at 0",
                       description = "d' for state 2 (low state)"),
        sigma2 = list(required = FALSE, default = NULL, link = "log",
                      null_value = 1, null_description = "fixed at 1"),
        dprime_L = list(required = FALSE, default = NULL, link = "identity",
                        null_value = NULL, null_description = "lure mixture disabled",
                        description = "d' for lure mixture state"),
        sigma_L = list(required = FALSE, default = NULL, link = "log",
                       null_value = 1, null_description = "sigma_L fixed at 1",
                       description = "sigma for lure mixture state"),
        lambda_L = list(required = FALSE, default = NULL, link = "logit",
                        null_value = NULL, null_description = "lure mixture disabled",
                        description = "probability of lure mixture state")
      ),
      description = "Mixture Signal Detection Model"
    ),
    class = "broc_family"
  )
}


#' Cumulative Ordinal Regression Model
#'
#' Ordinal regression with cumulative-link cutpoints and no SDT structure
#' (Agresti, 2010).
#'
#' @param link Link function (default: [link_probit()]). [link_logit()] also
#'   supported; GEV link not implemented for this family.
#'
#' @details
#' ## Likelihood
#' For an ordinal response Y on a K-point scale with cutpoints
#' c_1 < ... < c_{K-1}:
#' \deqn{P(Y \le k \mid x) = F(c_k - \mu(x))}
#' with \eqn{c_0 = -\infty} and \eqn{c_K = +\infty}, where F is the link CDF
#' (probit or logit) and \eqn{\mu(x)} is the linear predictor. Cell
#' probabilities by differencing:
#' \eqn{P(Y = k \mid x) = F(c_k - \mu(x)) - F(c_{k-1} - \mu(x))}.
#'
#' ## Parameters
#' \describe{
#'   \item{mu}{(required) Linear predictor of covariate effects; the intercept is
#'     absorbed by the cutpoints, so `rating ~ 1` has no `mu` coefficients.
#'     Identity link.}
#'   \item{cutpoints}{(required) K-1 ordered cutpoints (mid-anchor + K-2 ordered
#'     gaps). Identity link. Set priors via `thresh_mid` and `log_gaps`.}
#' }
#'
#' ## Formula syntax
#' ```
#' broc(brf(rating ~ condition + (condition | subj),
#'          cutpoints ~ 1 + (1 | subj),
#'          family = cumulative()),
#'     data = dat)
#' ```
#'
#' @references
#' Agresti, A. (2010). *Analysis of ordinal categorical data* (2nd ed.). Wiley.
#'
#' @return A `broc_family` object.
#' @seealso [evsd()] for the simplest SDT model;
#'   [bivariate_cumulative()] for the bivariate extension.
#' @export
cumulative <- function(link = link_probit()) {
  if (!inherits(link, "broc_link")) {
    stop("link must be created with link_probit() or link_logit()")
  }
  structure(
    list(
      family = "cumulative",
      link = link,
      is_sdt = FALSE,  # Flag to indicate this is NOT an SDT model
      # params keyed by INTERNAL Stan/JAX names. external_aliases below maps
      # them to the user-facing names ("mu", "cutpoints") used in formulas.
      params = list(
        dprime    = list(required = TRUE, default = NULL, link = "identity",
                         description = "linear predictor for location"),
        criterion = list(required = TRUE, default = NULL, link = "identity",
                         description = "threshold/cutpoint parameters")
      ),
      external_aliases = .param_name_map("cumulative"),
      description = "Cumulative Ordinal Regression Model"
    ),
    class = "broc_family"
  )
}


#' Bivariate Cumulative Ordinal Model
#'
#' Two correlated ordinal responses modelled jointly via the bivariate normal
#' CDF, with no SDT structure (Agresti, 2010).
#'
#' @details
#' ## Likelihood
#' For two ordinal responses (Y_1, Y_2) with cutpoints
#' c1_1 < ... < c1_{K_1-1} and c2_1 < ... < c2_{K_2-1}, the joint cumulative
#' is the bivariate normal CDF:
#' \deqn{P(Y_1 \le j,\; Y_2 \le k \mid x) = \Phi_2\big(c1_j - \mu_1(x),\;
#'       c2_k - \mu_2(x);\; \rho\big)}
#' with \eqn{c_0 = -\infty}, \eqn{c_K = +\infty}. Variance fixed at 1 on
#' both dimensions; \eqn{\mu_1} and \eqn{\mu_2} have no intercept (absorbed
#' into cutpoints) for identifiability. Cell probabilities by 4-corner
#' differencing:
#' \deqn{P(Y_1 = j,\; Y_2 = k \mid x) = P(\le j, \le k) - P(\le j-1, \le k)
#'       - P(\le j, \le k-1) + P(\le j-1, \le k-1)}
#'
#' ## Parameters
#' \describe{
#'   \item{mu1}{(required) Linear predictor for dimension 1; the intercept is
#'     absorbed by the cutpoints. Identity link.}
#'   \item{mu2}{(required) Linear predictor for dimension 2. Identity link.}
#'   \item{cutpoints1}{(required) K1-1 ordered cutpoints for dimension 1
#'     (mid-anchor + ordered gaps). Identity link. Set priors via `thresh_mid`
#'     and `log_gaps`.}
#'   \item{cutpoints2}{(required) K2-1 ordered cutpoints for dimension 2.
#'     Identity link. Set priors via `thresh_mid2` and `log_gaps2`.}
#'   \item{rho}{(required) Correlation between dimensions; effective
#'     \eqn{\rho = \tanh(\rho_{\text{param}})}. Fisher z link.}
#' }
#'
#' ## Formula syntax
#' ```
#' broc(brf(resp(y1, y2) ~ x1,
#'          mu2 ~ x2,
#'          cutpoints1 ~ 1,
#'          cutpoints2 ~ 1,
#'          rho ~ 1,
#'          family = bivariate_cumulative()),
#'     data = dat)
#' ```
#'
#' @return A `broc_family` object.
#' @seealso [cumulative()] for univariate ordinal,
#'   [bivariate_gaussian()] for the SDT version with new-item baseline.
#' @export
bivariate_cumulative <- function() {
  link <- link_probit()
  structure(
    list(
      family = "bivariate_cumulative",
      link = link,
      is_sdt = FALSE,
      # params keyed by INTERNAL Stan/JAX names. external_aliases below maps
      # them to the user-facing names ("mu1", "mu2", "cutpoints1", "cutpoints2")
      # used in formulas.
      params = list(
        dprime     = list(required = TRUE, default = NULL, link = "identity",
                          description = "linear predictor for dimension 1 location"),
        discrim    = list(required = TRUE, default = NULL, link = "identity",
                          description = "linear predictor for dimension 2 location"),
        criterion  = list(required = TRUE, default = NULL, link = "identity",
                          description = "dimension 1 threshold/cutpoint parameters"),
        criterion2 = list(required = TRUE, default = NULL, link = "identity",
                          description = "dimension 2 threshold/cutpoint parameters"),
        rho        = list(required = TRUE, default = NULL, link = "fisherz",
                          description = "correlation between dimensions (Fisher z scale)")
      ),
      external_aliases = .param_name_map("bivariate_cumulative"),
      description = "Bivariate Cumulative Ordinal Model"
    ),
    class = "broc_family"
  )
}


#' Source Mixture Signal Detection Model
#'
#' Finite-mixture signal detection model for source-discrimination tasks
#' (DeCarlo, 2003).
#'
#' @param link Link function (default: [link_probit()]).
#'
#' @details
#' ## Likelihood
#' Three latent Gaussian components, all with unit variance: nonattended at
#' 0 (shared), attended Source A at \eqn{d'_A} (typically negative), and attended
#' Source B at \eqn{d'_B} (typically positive). For each source, the source-rating
#' cumulative is a mixture of attended and nonattended:
#' \deqn{P(Y \le k \mid \mathrm{source} = A) = \lambda_A\,\Phi(c_k - d'_A)
#'       + (1 - \lambda_A)\,\Phi(c_k)}
#' \deqn{P(Y \le k \mid \mathrm{source} = B) = \lambda_B\,\Phi(c_k - d'_B)
#'       + (1 - \lambda_B)\,\Phi(c_k)}
#' where lambda_A, lambda_B are the source-conditional attention probabilities
#' (`lambda`, `lambda_B`).
#'
#' The overall source discriminability is \eqn{d_{AB} = d'_B - d'_A}.
#'
#' ## Parameters
#' \describe{
#'   \item{dprime}{(required) \eqn{d'_A}: location of the attended Source A
#'     distribution. Identity link.}
#'   \item{criterion}{(required) K-1 ordered confidence thresholds (mid-anchor +
#'     ordered gaps). Identity link.}
#'   \item{lambda}{(required) Attention probability for Source A. Logit link.}
#'   \item{dprime_B}{(optional) \eqn{d'_B}: location of the attended Source B
#'     distribution. Identity link. If omitted, constrained to
#'     `dprime_B = -dprime`.}
#'   \item{lambda_B}{(optional) Attention probability for Source B. Logit link.
#'     If omitted, constrained to `lambda_B = lambda`.}
#' }
#'
#' ## Formula syntax
#' The conditioning variable after `|` is the source variable (2-level
#' factor or 0/1 indicator for Source A vs Source B).
#' ```
#' broc(brf(conf | source ~ 1 + (1 | subj),
#'          criterion ~ 1 + (1 | subj),
#'          lambda ~ 1 + (1 | subj),
#'          family = source_mixture()),
#'     data = dat)
#' ```
#'
#' @references
#' DeCarlo, L. T. (2003a). An application of signal detection theory with finite
#'   mixture distributions to source discrimination. *Journal of Experimental
#'   Psychology: Learning, Memory, and Cognition*, *29*(5), 767-778.
#'
#' @return A `broc_family` object.
#' @seealso [bivariate_gaussian()] for the bivariate-normal approach to source
#'   monitoring (separate detection + source ratings).
#' @export
source_mixture <- function(link = link_probit()) {
  if (!inherits(link, "broc_link")) {
    stop("link must be created with link_probit() or link_logit()")
  }
  structure(
    list(
      family = "source_mixture",
      link = link,
      params = list(
        dprime = list(required = TRUE, default = NULL, link = "identity",
                      description = "d' for attended Source A items (Psi_A location)"),
        criterion = list(required = TRUE, default = NULL, link = "identity"),
        lambda = list(required = TRUE, default = NULL, link = "logit",
                      description = "attention/encoding probability for Source A"),
        dprime_B = list(required = FALSE, default = NULL, link = "identity",
                        null_value = "symmetric", 
                        null_description = "constrained to -dprime (symmetric)",
                        description = "d' for attended Source B items (Psi_B location)"),
        lambda_B = list(required = FALSE, default = NULL, link = "logit",
                        null_value = "equal",
                        null_description = "constrained equal to lambda",
                        description = "attention/encoding probability for Source B")
      ),
      description = "Source Mixture Signal Detection Model"
    ),
    class = "broc_family"
  )
}


#' Bivariate Gaussian Signal Detection Model
#'
#' Bivariate Gaussian signal detection model for source monitoring, with
#' unbounded and bounded (BBG) variants (DeCarlo, 2003; Starns et al., 2014).
#'
#' ## Source labeling convention
#' Source A is the source associated with **low source ratings** (left end
#' of the source confidence scale); Source B with **high source ratings**.
#' In bounded mode, A items are anchored at the negative source-axis end and
#' B items at the positive end.
#'
#' @param varying_source_criteria If `TRUE`, estimate separate source criteria
#'   for each detection confidence level.
#' @param bounded If `TRUE`, fit the Bounded Bivariate Gaussian (BBG; Starns
#'   et al. 2014) variant: dprime/discrim use log links (positive magnitudes),
#'   rho uses logistic link (in (0,1)), and the model anchors A on the
#'   negative source axis / B on the positive axis. Default `FALSE` fits the
#'   standard unbounded BG model (DeCarlo, 2003).
#' @param new_source_criteria With `varying_source_criteria = TRUE`, how source
#'   criteria are structured across detection responses (ignored otherwise).
#'   `"full"` (default): every detection response level has its own source
#'   criteria (K1 x (K2-1) in total). `"shared"`: responses that judge an item
#'   "new" share a single set of source criteria; requires `old_levels`.
#' @param old_levels Integer vector of detection response levels (on the
#'   1..K1 scale, e.g. `old_levels = 4:6`) that count as "old" judgements; the
#'   rest are "new". Required when `new_source_criteria = "shared"`; ignored
#'   otherwise.
#' @param varying_re Random-effect structure for varying source criteria
#'   (distinct from `new_source_criteria`, which governs the fixed criteria).
#'   `"shared"` (default): one source-criterion RE shift shared across all
#'   detection bins. `"per_bin"`: a separate RE shift per detection bin.
#'   `"full"`: a separate RE for every source threshold within every detection
#'   bin. Requires `varying_source_criteria = TRUE` unless `"shared"`.
#'
#' @details
#' ## Likelihood
#' For each item type \eqn{X \in \{N, A, B\}}, the joint cumulative on the
#' (detection, source) ratings is the bivariate normal CDF:
#' \deqn{P(Y_1 \le j,\; Y_2 \le k \mid X) = \Phi_2\!\Big(\tfrac{c1_j - \mu_1(X)}{\sigma_1(X)},\;
#'       \tfrac{c2_k - \mu_2(X)}{\sigma_2(X)};\; \rho(X)\Big)}
#' where the per-source means/SDs/correlation are:
#' \describe{
#'   \item{X = N:}{\eqn{\mu_1 = 0,\; \mu_2 = 0,\; \sigma_1 = \sigma_2 = 1,\; \rho = \rho_N}}
#'   \item{X = A:}{\eqn{\mu_1 = d'_A,\; \mu_2 = \psi_A,\; \sigma_1 = \sigma_{1A},\; \sigma_2 = \sigma_{2A},\; \rho = \rho_A}}
#'   \item{X = B:}{\eqn{\mu_1 = d'_B,\; \mu_2 = \psi_B,\; \sigma_1 = \sigma_{1B},\; \sigma_2 = \sigma_{2B},\; \rho = \rho_B}}
#' }
#' with \eqn{d'_A} = `dprime`, \eqn{d'_B} = `dprime_B`,
#' \eqn{\psi_A} = `discrim`, \eqn{\psi_B} = `discrim_B`. Cell probabilities
#' by 4-corner differencing:
#' \deqn{P(Y_1 = j,\; Y_2 = k \mid X) = P(\le j, \le k) - P(\le j-1, \le k)
#'       - P(\le j, \le k-1) + P(\le j-1, \le k-1)}
#' In **bounded** mode, the conditional source mean is clamped at 0 below
#' the per-source crossover point, so source items never have below-chance
#' source discrimination at low item evidence (Starns et al., 2014).
#'
#' ## Parameters
#' \describe{
#'   \item{dprime}{(required) \eqn{d'_A} on the detection axis. Identity link
#'     (unbounded) or log link (bounded).}
#'   \item{discrim}{(required) Signed source-axis position of A items.
#'     Identity link (unbounded) or log link (bounded -- positive magnitude,
#'     placed on the negative source axis via internal negation).}
#'   \item{criterion}{(required) K1-1 detection thresholds. Identity link.}
#'   \item{criterion2}{(required) K2-1 source thresholds. Identity link.}
#'   \item{rho}{(required) Detection-source correlation for A items. Fisher z
#'     link (unbounded) or logistic (bounded).}
#'   \item{dprime_B}{(optional) \eqn{d'_B} for B items. If omitted, constrained
#'     `dprime_B = dprime` (equal detection sensitivity).}
#'   \item{discrim_B}{(optional) Signed source-axis position of B items.
#'     If omitted, constrained mirror-symmetric (unbounded:
#'     `discrim_B = -discrim`; bounded: same magnitude as `discrim`,
#'     placed on the positive source axis).}
#'   \item{sigma}{(optional) SD on detection axis for A items. If
#'     omitted, fixed at 1. Log link.}
#'   \item{sigma_B}{(optional) SD on detection axis for B items. If
#'     omitted, constrained `sigma_B = sigma`.}
#'   \item{sigma2}{(optional) SD on source axis for A items. If
#'     omitted, fixed at 1. Log link.}
#'   \item{sigma2_B}{(optional) SD on source axis for B items. If
#'     omitted, constrained `sigma2_B = sigma2`.}
#'   \item{rho_B}{(optional) Detection-source correlation for B items.
#'     If omitted, constrained mirror-symmetric to `rho`.}
#'   \item{rho_N}{(optional) Detection-source correlation for new items.
#'     Fisher z link. If omitted, fixed at 0.}
#' }
#'
#' ## Formula syntax
#' Uses `resp(det_conf, src_conf)` with a 3-level item type variable
#' (new = reference baseline, A, B) as the conditioning variable:
#' ```
#' broc(brf(resp(det_conf, src_conf) | type ~ 1 + (1 | subj),
#'          discrim ~ 1 + (1 | subj),
#'          criterion ~ 1 + (1 | subj),
#'          criterion2 ~ 1 + (1 | subj),
#'          rho ~ 1,
#'          family = bivariate_gaussian()),
#'     data = dat)
#' ```
#'
#' @references
#' DeCarlo, L. T. (2003b). Source monitoring and multivariate signal detection
#'   theory, with a model for selection. *Journal of Mathematical Psychology*,
#'   *47*(3), 292-303.
#'
#' Starns, J. J., Rotello, C. M., & Hautus, M. J. (2014). Recognition memory
#'   zROC slopes for items with correct versus incorrect source decisions
#'   discriminate the dual process and unequal variance signal detection
#'   models. *Journal of Experimental Psychology: Learning, Memory, and
#'   Cognition*, *40*(5), 1205-1225.
#'
#' @return A `broc_family` object.
#' @seealso [bivariate_dp()] for dual-process extension, [vrdp2d()] for
#'   variable recollection, [source_mixture()] for the single-response
#'   mixture approach.
#' @export
bivariate_gaussian <- function(varying_source_criteria = FALSE,
                          bounded = FALSE, new_source_criteria = c("full", "shared"),
                          old_levels = NULL, varying_re = "shared") {
  link <- link_probit()
  varying_re <- match.arg(varying_re, c("shared", "per_bin", "full"))
  if (varying_re != "shared" && !varying_source_criteria) {
    stop("varying_re = '", varying_re, "' requires varying_source_criteria = TRUE")
  }
  new_source_criteria <- match.arg(new_source_criteria)
  if (new_source_criteria == "shared") {
    if (!varying_source_criteria) {
      stop("new_source_criteria = 'shared' requires varying_source_criteria = TRUE")
    }
    if (is.null(old_levels) || !is.numeric(old_levels) || length(old_levels) < 1) {
      stop("old_levels must be a numeric vector of detection response levels ",
           "that count as 'old' (e.g., old_levels = 4:6) when ",
           "new_source_criteria = 'shared'")
    }
    old_levels <- sort(as.integer(old_levels))
  } else if (!is.null(old_levels)) {
    stop("old_levels is only used when new_source_criteria = 'shared'")
  }
  # Bounded models require positive dprime, discrim, and rho in (0,1)
  dprime_link <- if (bounded) "log" else "identity"
  discrim_link <- if (bounded) "log" else "identity"
  rho_link <- if (bounded) "logis" else "fisherz"
  rho_desc <- if (bounded) {
    "rho_A correlation for Source A (logistic scale, bounded to (0,1))"
  } else {
    "rho_A correlation for Source A (Fisher z scale)"
  }

  structure(
    list(
      family = "bivariate_sdt",
      link = link,
      varying_source_criteria = varying_source_criteria,
      bounded = bounded,
      new_source_criteria = new_source_criteria,
      old_levels = old_levels,
      varying_re = varying_re,
      params = list(
        # Detection dimension (dimension 1)
        dprime = list(required = TRUE, default = NULL, link = dprime_link,
                      description = "d' for Source A detection (dimension 1 mean)"),
        dprime_B = list(required = FALSE, default = NULL, link = dprime_link,
                        null_value = "equal",
                        null_description = "constrained equal to dprime",
                        description = "d' for Source B detection"),
        sigma = list(required = FALSE, default = NULL, link = "log",
                     null_value = 1, null_description = "fixed at 1",
                     description = "sigma1 for Source A (detection SD)"),
        sigma_B = list(required = FALSE, default = NULL, link = "log",
                       null_value = "equal",
                       null_description = "constrained equal to sigma",
                       description = "sigma1 for Source B (detection SD)"),

        # Discrimination dimension (dimension 2)
        discrim = list(required = TRUE, default = NULL, link = discrim_link,
                       description = "Source A discrimination (dimension 2 mean)"),
        discrim_B = list(required = FALSE, default = NULL, link = discrim_link,
                         null_value = "symmetric",
                         null_description = "constrained to -discrim (symmetric sources)",
                         description = "Source B discrimination"),
        sigma2 = list(required = FALSE, default = NULL, link = "log",
                      null_value = 1, null_description = "fixed at 1",
                      description = "sigma2 for Source A (discrimination SD)"),
        sigma2_B = list(required = FALSE, default = NULL, link = "log",
                        null_value = "equal",
                        null_description = "constrained equal to sigma2",
                        description = "sigma2 for Source B (discrimination SD)"),

        # Correlations
        rho = list(required = TRUE, default = NULL, link = rho_link,
                   description = rho_desc),
        rho_B = list(required = FALSE, default = NULL, link = rho_link,
                     null_value = "symmetric",
                     null_description = "constrained to -rho (symmetric)",
                     description = "rho_B correlation for Source B"),
        rho_N = list(required = FALSE, default = NULL, link = "fisherz",
                     null_value = 0, null_description = "fixed at 0",
                     description = "rho_N correlation for new items"),

        # Thresholds
        criterion = list(required = TRUE, default = NULL, link = "identity",
                         description = "Detection thresholds (c1)"),
        criterion2 = list(required = TRUE, default = NULL, link = "identity",
                          description = "Discrimination thresholds (c2)")
      ),
      description = if (bounded) {
        "Bounded Bivariate Gaussian Signal Detection Model"
      } else {
        "Bivariate Gaussian Signal Detection Model"
      }
    ),
    class = "broc_family"
  )
}


#' Bivariate Dual Process Signal Detection Model
#'
#' Bivariate dual-process model: Gaussian familiarity plus discrete item and
#' source recollection (Starns et al., 2014).
#'
#' Same source labeling convention as [bivariate_gaussian()]: source A = low
#' source ratings, source B = high.
#'
#' @param varying_source_criteria If `TRUE`, estimate separate source
#'   criteria for each detection confidence level.
#' @param bounded If `TRUE`, fit the BBDP variant (conditional source mean
#'   clamped at the new-item baseline). If `FALSE` (default), fit the BDP
#'   variant with standard bivariate Gaussian familiarity.
#' @inheritParams bivariate_gaussian
#'
#' @details
#' ## Likelihood
#' For each old item, the cell probability is a 3-component mixture over
#' recollection states:
#' \deqn{P(Y_1 = j,\; Y_2 = k \mid X) = (1 - \lambda)\,P_{\mathrm{F}}(j, k \mid X)
#'       + \lambda(1 - \lambda_2)\,\mathbf{1}[j = K_1]\,P_{\mathrm{S}}(k \mid X)
#'       + \lambda \lambda_2\,\mathbf{1}[j = K_1, k = k_X^*]}
#' where:
#' - lambda is the item-recollection probability, lambda_2 is the source-recollection
#'   probability (`lambda`, `lambda2`).
#' - \eqn{P_{\mathrm{F}}(j, k \mid X)} is the bivariate-Gaussian familiarity
#'   cell probability, defined cumulatively as
#'   \eqn{P_{\mathrm{F}}(Y_1 \le j, Y_2 \le k \mid X) = \Phi_2\big((c1_j - \mu_1)/\sigma_1,\;
#'   (c2_k - \mu_2)/\sigma_2;\; \rho\big)} with 4-corner differencing, taking
#'   per-source \eqn{(\mu_1, \mu_2, \sigma_1, \sigma_2, \rho)} from
#'   [bivariate_gaussian()].
#' - \eqn{P_{\mathrm{S}}(k \mid X)} is the marginal source cell probability
#'   (familiarity integrated out): \eqn{P_{\mathrm{S}}(Y_2 \le k \mid X) =
#'   \Phi((c2_k - \mu_2)/\sigma_2)} with 1D differencing.
#' - \eqn{k_X^*} is the recollection-success source response: \eqn{k_X^* = 1}
#'   for X = A, \eqn{k_X^* = K_2} for X = B. Both signal a correctly
#'   identified source under recollection.
#' - For new items: lambda = lambda_2 = 0, so the likelihood reduces to
#'   \eqn{P_{\mathrm{F}}} alone with new-item parameters.
#'
#' In the bounded (BBDP) variant, the bivariate piece uses the source-mean
#' clamping of Starns et al. (2014); otherwise (BDP) the standard
#' bivariate normal applies.
#'
#' ## Parameters
#' All parameters of [bivariate_gaussian()] (`dprime`, `dprime_B`, `discrim`,
#' `discrim_B`, `sigma`, `sigma_B`, `sigma2`, `sigma2_B`, `rho`, `rho_B`,
#' `rho_N`, `criterion`, `criterion2`) plus:
#' \describe{
#'   \item{lambda}{(required) Item-recollection probability for Source A. Logit
#'     link.}
#'   \item{lambda_B}{(optional) Item-recollection probability for Source B. If
#'     omitted, constrained `lambda_B = lambda`.}
#'   \item{lambda2}{(required) Source-recollection probability for Source A.
#'     Logit link.}
#'   \item{lambda2_B}{(optional) Source-recollection probability for Source B. If
#'     omitted, constrained `lambda2_B = lambda2`.}
#' }
#' Defaults for the bivariate parameters match [bivariate_gaussian()] -- e.g.
#' `dprime_B` defaults to `dprime`, `sigma` defaults to fixed at 1,
#' `discrim_B` defaults to mirror-symmetric.
#'
#' ## Formula syntax
#' ```
#' broc(brf(resp(det_conf, src_conf) | type ~ 1 + (1 | subj),
#'          discrim ~ 1 + (1 | subj),
#'          criterion ~ 1 + (1 | subj),
#'          criterion2 ~ 1 + (1 | subj),
#'          rho ~ 1,
#'          lambda ~ 1 + (1 | subj),
#'          lambda2 ~ 1 + (1 | subj),
#'          family = bivariate_dp()),
#'     data = dat)
#' ```
#'
#' @references
#' Starns, J. J., Rotello, C. M., & Hautus, M. J. (2014). Recognition memory
#'   zROC slopes for items with correct versus incorrect source decisions
#'   discriminate the dual process and unequal variance signal detection
#'   models. *Journal of Experimental Psychology: Learning, Memory, and
#'   Cognition*, *40*(5), 1205-1225.
#'
#' @return A `broc_family` object.
#' @seealso [bivariate_gaussian()], [vrdp2d()].
#' @export
bivariate_dp <- function(varying_source_criteria = FALSE,
                         bounded = FALSE, new_source_criteria = c("full", "shared"),
                         old_levels = NULL, varying_re = "shared") {
  link <- link_probit()
  varying_re <- match.arg(varying_re, c("shared", "per_bin", "full"))
  if (varying_re != "shared" && !varying_source_criteria) {
    stop("varying_re = '", varying_re, "' requires varying_source_criteria = TRUE")
  }
  new_source_criteria <- match.arg(new_source_criteria)
  if (new_source_criteria == "shared") {
    if (!varying_source_criteria) {
      stop("new_source_criteria = 'shared' requires varying_source_criteria = TRUE")
    }
    if (is.null(old_levels) || !is.numeric(old_levels) || length(old_levels) < 1) {
      stop("old_levels must be a numeric vector of detection response levels ",
           "that count as 'old' (e.g., old_levels = 4:6) when ",
           "new_source_criteria = 'shared'")
    }
    old_levels <- sort(as.integer(old_levels))
  } else if (!is.null(old_levels)) {
    stop("old_levels is only used when new_source_criteria = 'shared'")
  }
  # Conditional links: bounded = log/logistic, unbounded = identity/Fisher z.
  dprime_link <- if (bounded) "log" else "identity"
  discrim_link <- if (bounded) "log" else "identity"
  rho_link <- if (bounded) "logis" else "fisherz"
  dprime_desc <- if (bounded) "d' for Source A detection (log scale, always positive)" else "d' for Source A detection"
  discrim_desc <- if (bounded) "Source A discrimination (log scale, always positive)" else "Source A discrimination (signed source-axis position)"
  rho_desc <- if (bounded) "rho_A correlation for Source A (logistic scale, bounded to (0,1))" else "rho_A correlation for Source A (Fisher z)"
  structure(
    list(
      family = "bivariate_dp",
      link = link,
      varying_source_criteria = varying_source_criteria,
      bounded = bounded,
      new_source_criteria = new_source_criteria,
      old_levels = old_levels,
      varying_re = varying_re,
      params = list(
        # Detection dimension (dimension 1)
        dprime = list(required = TRUE, default = NULL, link = dprime_link,
                      description = dprime_desc),
        dprime_B = list(required = FALSE, default = NULL, link = dprime_link,
                        null_value = "equal",
                        null_description = "constrained equal to dprime",
                        description = "d' for Source B detection"),
        sigma = list(required = FALSE, default = NULL, link = "log",
                     null_value = 1, null_description = "fixed at 1",
                     description = "sigma_I for Source A (detection SD)"),
        sigma_B = list(required = FALSE, default = NULL, link = "log",
                       null_value = "equal",
                       null_description = "constrained equal to sigma",
                       description = "sigma_I for Source B (detection SD)"),

        # Discrimination dimension (dimension 2)
        discrim = list(required = TRUE, default = NULL, link = discrim_link,
                       description = discrim_desc),
        discrim_B = list(required = FALSE, default = NULL, link = discrim_link,
                         null_value = "symmetric",
                         null_description = "constrained mirror of discrim (symmetric sources)",
                         description = "Source B discrimination"),
        sigma2 = list(required = FALSE, default = NULL, link = "log",
                      null_value = 1, null_description = "fixed at 1",
                      description = "sigma_S for Source A (discrimination SD)"),
        sigma2_B = list(required = FALSE, default = NULL, link = "log",
                        null_value = "equal",
                        null_description = "constrained equal to sigma2",
                        description = "sigma_S for Source B (discrimination SD)"),

        # Correlations
        rho = list(required = TRUE, default = NULL, link = rho_link,
                   description = rho_desc),
        rho_B = list(required = FALSE, default = NULL, link = rho_link,
                     null_value = "symmetric",
                     null_description = "constrained mirror of rho (symmetric)",
                     description = "rho_B correlation for Source B"),
        rho_N = list(required = FALSE, default = NULL, link = "fisherz",
                     null_value = 0, null_description = "fixed at 0",
                     description = "rho_N correlation for new items"),

        # Recollection parameters
        lambda = list(required = TRUE, default = NULL, link = "logit",
                      description = "item recollection probability for Source A (logit scale)"),
        lambda_B = list(required = FALSE, default = NULL, link = "logit",
                        null_value = "equal",
                        null_description = "constrained equal to lambda",
                        description = "item recollection probability for Source B"),
        lambda2 = list(required = TRUE, default = NULL, link = "logit",
                       description = "source recollection probability for Source A (logit scale)"),
        lambda2_B = list(required = FALSE, default = NULL, link = "logit",
                         null_value = "equal",
                         null_description = "constrained equal to lambda2",
                         description = "source recollection probability for Source B"),

        # Thresholds
        criterion = list(required = TRUE, default = NULL, link = "identity",
                         description = "Detection thresholds (c1)"),
        criterion2 = list(required = TRUE, default = NULL, link = "identity",
                          description = "Discrimination thresholds (c2)")
      ),
      description = if (bounded) {
        "Bounded Bivariate Dual Process Signal Detection Model"
      } else {
        "Bivariate Dual Process Signal Detection Model"
      }
    ),
    class = "broc_family"
  )
}


#' Continuous Dual Process Signal Detection Model
#'
#' Continuous dual-process model for Remember/Know paradigms, with independent
#' recollection and familiarity strengths (Wixted & Mickes, 2010).
#'
#' @param old_levels Integer vector specifying which confidence levels
#'   correspond to "old" responses (where R/K judgments apply). If `NULL`
#'   (default), inferred at fit time from the data as the bins where `rk`
#'   is non-NA. Pass explicitly only to override that inference (e.g. when
#'   some "old" bin happens to have no observed responses).
#'
#' @details
#' ## Likelihood
#' Recollection R and familiarity F are independent Gaussians:
#' \eqn{R \sim \mathcal{N}(\mu_R, \sigma_R)}, \eqn{F \sim \mathcal{N}(\mu_F, \sigma_F)}
#' for targets, with \eqn{(\mu_R, \mu_F) = (0, 0)} and
#' \eqn{(\sigma_R, \sigma_F) = (1, 1)} for lures.
#'
#' On the F (familiarity) dimension the cumulative is:
#' \deqn{P(Y \le k \mid \mathrm{old} = 0) = \Phi(c_k), \quad
#'       P(Y \le k \mid \mathrm{old} = 1) = \Phi\!\Big(\tfrac{c_k - \mu_F}{\sigma_F}\Big)}
#' For ratings k not in `old_levels` (new-side responses), the F cell
#' (by differencing) is the full likelihood. For ratings k in `old_levels`
#' (old-side responses), the cell factors across F (bin) and R (Remember
#' vs Know):
#' \deqn{P(Y = k,\; \mathrm{rk} \mid \mathrm{old}) =
#'       \big[P(Y \le k) - P(Y \le k - 1)\big] \cdot P(\mathrm{rk} \mid \mathrm{old})}
#' with the rk factor from the cumulative on the R dimension. For targets:
#' \deqn{P(\mathrm{rk} = K \mid \mathrm{old} = 1) = \Phi\!\Big(\tfrac{c_R - \mu_R}{\sigma_R}\Big), \quad
#'       P(\mathrm{rk} = R \mid \mathrm{old} = 1) = 1 - \Phi\!\Big(\tfrac{c_R - \mu_R}{\sigma_R}\Big)}
#' (lures use the same form with \eqn{\mu_R = 0, \sigma_R = 1}). For 3-level
#' R/K/G with Know criterion \eqn{c_K}, the K row becomes
#' \eqn{\Phi((c_R - \mu_R)/\sigma_R) - \Phi((c_K - \mu_R)/\sigma_R)} and
#' \eqn{P(\mathrm{rk} = G) = \Phi((c_K - \mu_R)/\sigma_R)}.
#'
#' ## Parameters
#' \describe{
#'   \item{rec}{(required) Recollection mean \eqn{\mu_R} for targets. Identity
#'     link.}
#'   \item{fam}{(required) Familiarity mean \eqn{\mu_F} for targets. Identity
#'     link.}
#'   \item{criterion}{(required) Confidence thresholds on the F dimension.
#'     Identity link.}
#'   \item{rec_crit}{(required) Recollection criterion \eqn{c_R} separating
#'     Remember vs Know responses on the R dimension. Identity link.}
#'   \item{sigma_R}{(optional) SD of recollection for targets. If omitted, fixed
#'     at 1. Log link.}
#'   \item{sigma_F}{(optional) SD of familiarity for targets. If omitted, fixed
#'     at 1. Log link.}
#'   \item{know_crit}{(optional) Know criterion \eqn{c_K} on the R dimension
#'     for the 3-level R/K/G case (when `rk` has values 1=R, 2=K, 3=G).
#'     Identity link. If omitted, only R/K is modeled.}
#' }
#'
#' ## Response coding
#' `rk` values: 1 = Remember, 2 = Know, (3 = Guess for R/K/G mode).
#'
#' ## Formula syntax
#' ```
#' broc(brf(resp(conf, rk_resp) | old ~ 1 + (1 | subj),
#'          fam ~ 1 + (1 | subj),
#'          criterion ~ 1 + (1 | subj),
#'          rec_crit ~ 1 + (1 | subj),
#'          family = cdp(old_levels = 4:6)),
#'     data = dat)
#' ```
#'
#' @references
#' Wixted, J. T., & Mickes, L. (2010). A continuous dual-process model of
#'   remember/know judgments. *Psychological Review*, *117*(4), 1025-1054.
#'
#' @return A `broc_family` object.
#' @export
cdp <- function(old_levels = NULL) {
  link <- link_probit()

  # old_levels is optional: if NULL, broc() will infer from data as
  # the confidence bins where rk is non-NA (rk should be NA for new-side
  # responses by data construction). Pass explicitly to override.
  if (!is.null(old_levels)) {
    if (!is.numeric(old_levels) || length(old_levels) < 1) {
      stop("old_levels must be a numeric vector of confidence levels (or NULL to infer from data)")
    }
    old_levels <- sort(as.integer(old_levels))
  }
  J <- if (is.null(old_levels)) NULL else length(old_levels)

  structure(
    list(
      family = "cdp",
      link = link,
      old_levels = old_levels,
      J = J,
      # params keyed by INTERNAL Stan/JAX names. external_aliases below maps
      # them to the user-facing names (rec/fam/sigma_R/sigma_F) used in formulas.
      params = list(
        # Recollection mean (rec) -> internal: dprime
        dprime    = list(required = TRUE, default = NULL, link = "identity",
                         description = "Recollection mean (mu_R) for targets"),
        # Recollection SD (sigma_R) -> internal: sigma
        sigma     = list(required = FALSE, default = NULL, link = "log",
                         null_value = 0, null_description = "fixed at 1",
                         description = "Recollection SD for targets"),
        # Familiarity mean (fam) -> internal: dprime2
        dprime2   = list(required = TRUE, default = NULL, link = "identity",
                         description = "Familiarity mean (mu_F) for targets"),
        # Familiarity SD (sigma_F) -> internal: sigma2
        sigma2    = list(required = FALSE, default = NULL, link = "log",
                         null_value = 0, null_description = "fixed at 1",
                         description = "Familiarity SD for targets"),
        # Criteria (no aliasing for these)
        rec_crit  = list(required = TRUE, default = NULL, link = "identity",
                         description = "Recollection criterion (c_R)"),
        know_crit = list(required = FALSE, default = NULL, link = "identity",
                         description = "Know criterion (c_K) for R/K/G mode"),
        criterion = list(required = TRUE, default = NULL, link = "identity",
                         description = "Confidence thresholds (tau)")
      ),
      external_aliases = .param_name_map("cdp"),
      description = "Continuous Dual Process Signal Detection Model"
    ),
    class = "broc_family"
  )
}


#' Two-Dimensional Variable Recollection Dual Process Model
#'
#' Two-dimensional variable-recollection dual-process model for conjoint
#' item-source recognition, with recollection-gated source discrimination
#' (Onyper, Zhang & Howard, 2010).
#'
#' @param varying_source_criteria If `TRUE`, estimate separate source
#'   criteria for each item confidence level (K1 x (K2-1) source thresholds);
#'   the full model of Onyper et al. (2010). If `FALSE` (default), a single
#'   set of source criteria (K2-1 thresholds) shared across detection levels.
#' @param varying_re Random-effect structure for varying source criteria.
#'   `"shared"` (default): one source-criterion RE shift shared across all
#'   detection bins. `"per_bin"`: a separate RE shift per detection bin.
#'   `"full"`: a separate RE for every source threshold within every detection
#'   bin. Requires `varying_source_criteria = TRUE` unless `"shared"`.
#'
#' @details
#' ## Likelihood
#' Both states have diagonal covariance (zero detection-source correlation),
#' so the joint cumulative factors across dimensions. For new items:
#' \deqn{P(Y_1 \le j,\; Y_2 \le k \mid X = N) = \Phi(c1_j)\,\Phi(c2_k)}
#' For old items, recollection succeeds with probability lambda. The cumulative
#' is a 2-component mixture over recollection:
#' \deqn{P(Y_1 \le j,\; Y_2 \le k \mid X = \mathrm{old}) =
#'       (1 - \lambda)\,\Phi\!\Big(\tfrac{c1_j - d'_F}{\sigma_1}\Big)\Phi(c2_k)
#'       + \lambda\,\Phi\!\Big(\tfrac{c1_j - d'_F - d'_R}{\sigma_1}\Big)
#'         \Phi\!\Big(\tfrac{c2_k - \psi_X}{\sigma_2}\Big)}
#' where \eqn{\psi_A = +d'_S, \psi_B = -d'_S} (the recollection-conditional
#' source location for A vs B), and lambda is the recollection probability
#' (`lambda`). The non-recollected component has zero source mean (no source
#' info); the recollected component carries source discriminability.
#'
#' ## Parameters
#' \describe{
#'   \item{dprime}{(required) \eqn{d'_F}: familiarity strength on the item
#'     dimension. Identity link.}
#'   \item{dprime2}{(required) \eqn{d'_R}: recollection boost on the item
#'     dimension. Identity link.}
#'   \item{discrim}{(required) \eqn{d'_S}: source discriminability for
#'     recollected items only (non-recollected items center at 0 on source).
#'     Identity link.}
#'   \item{lambda}{(required) Recollection probability. Logit link.}
#'   \item{criterion}{(required) K1-1 item-dimension thresholds. Identity link.}
#'   \item{criterion2}{(required) K2-1 source-dimension thresholds. Identity link.}
#'   \item{sigma}{(optional) SD on the item dimension. If omitted, fixed at 1.
#'     Log link.}
#'   \item{sigma2}{(optional) SD on the source dimension for recollected items.
#'     If omitted, fixed at 1. Log link.}
#'   \item{discrim_B}{(optional) Source discriminability for B items. If omitted,
#'     constrained to mirror A: `discrim_B = -discrim`.}
#' }
#'
#' ## Formula syntax
#' Uses `resp(det_conf, src_conf)` with a 3-level item type conditioning
#' variable (new, A, B):
#' ```
#' broc(brf(resp(det_conf, src_conf) | type ~ 1 + (1 | subj),
#'          dprime2 ~ 1,
#'          discrim ~ 1 + (1 | subj),
#'          criterion ~ 1 + (1 | subj),
#'          criterion2 ~ 1 + (1 | subj),
#'          lambda ~ 1 + (1 | subj),
#'          family = vrdp2d()),
#'     data = dat)
#' ```
#'
#' @references
#' Onyper, S. V., Zhang, Y. X., & Howard, M. W. (2010). Some-or-none
#'   recollection: Evidence from item and source memory. *Journal of
#'   Experimental Psychology: General*, *139*(2), 341-364.
#'   https://doi.org/10.1037/a0018926
#'
#' @return A `broc_family` object.
#' @seealso [bivariate_gaussian()], [bivariate_dp()].
#' @export
vrdp2d <- function(varying_source_criteria = FALSE,
                   varying_re = "shared") {
  link <- link_probit()
  varying_re <- match.arg(varying_re, c("shared", "per_bin", "full"))
  if (varying_re != "shared" && !varying_source_criteria) {
    stop("varying_re = '", varying_re, "' requires varying_source_criteria = TRUE")
  }
  structure(
    list(
      family = "vrdp2d",
      link = link,
      varying_source_criteria = varying_source_criteria,
      varying_re = varying_re,
      params = list(
        # Item dimension - familiarity (use dprime)
        dprime = list(required = TRUE, default = NULL, link = "identity",
                      description = "Familiarity strength (item dimension, new vs old)"),
        
        # Item dimension - recollection boost (use dprime2)
        dprime2 = list(required = TRUE, default = NULL, link = "identity",
                       description = "Recollection strength boost on item dimension"),
        
        # Source dimension (use discrim for Source A/symmetric)
        discrim = list(required = TRUE, default = NULL, link = "identity",
                       description = "Source discriminability for Source A (only for recollected items)"),
        
        # Source dimension B (optional, for asymmetric sources)
        discrim_B = list(required = FALSE, default = NULL, link = "identity",
                         null_value = "symmetric",
                         null_description = "constrained to -discrim (symmetric sources)",
                         description = "Source discriminability for Source B"),
        
        # Recollection probability
        lambda = list(required = TRUE, default = NULL, link = "logit",
                      description = "R: Probability of recollection for old items"),
        
        # Optional: SD on item dimension for old items (use sigma)
        sigma = list(required = FALSE, default = NULL, link = "log",
                     null_value = 1, null_description = "fixed at 1",
                     description = "sigma_item: SD on item dimension for old items"),
        
        # Optional: SD on source dimension for recollected items (use sigma2)
        sigma2 = list(required = FALSE, default = NULL, link = "log",
                      null_value = 1, null_description = "fixed at 1",
                      description = "sigma_S: SD on source dimension for recollected items"),
        
        # Thresholds
        criterion = list(required = TRUE, default = NULL, link = "identity",
                         description = "Item dimension thresholds (K1-1 thresholds)"),
        criterion2 = list(required = TRUE, default = NULL, link = "identity",
                          description = "Source dimension thresholds (K2-1 thresholds, or K1x(K2-1) if varying)")
      ),
      description = "Two-Dimensional Variable Recollection Dual Process Model"
    ),
    class = "broc_family"
  )
}


#' Validate that required parameters are provided
#' @param family The SDT family
#' @param provided_params Named list of provided parameters (NULL means fixed)
#' @noRd
validate_family_params <- function(family, provided_params) {
  for (param_name in names(family$params)) {
    param_info <- family$params[[param_name]]
    
    if (param_info$required && !param_name %in% names(provided_params)) {
      stop(sprintf("Parameter '%s' is required for family '%s'", 
                   param_name, family$family))
    }
    
    # Check if optional param is provided but family doesn't support it
    if (param_name %in% names(provided_params) && 
        !param_name %in% names(family$params)) {
      stop(sprintf("Parameter '%s' is not supported by family '%s'",
                   param_name, family$family))
    }
  }
  
  # Check for unknown parameters
  for (param_name in names(provided_params)) {
    if (!param_name %in% names(family$params) && 
        !param_name %in% c("data", "family", "priors")) {
      stop(sprintf("Unknown parameter '%s' for family '%s'",
                   param_name, family$family))
    }
  }
  
  invisible(TRUE)
}


#' Map an internal family name to its user-facing constructor name.
#'
#' Some families store an internal dispatch name that differs from the
#' constructor the user calls (e.g. `evsd()` -> `"evsdt"`,
#' `bivariate_gaussian()` -> `"bivariate_sdt"`); printed output should show the
#' constructor name.
#' @noRd
family_display_name <- function(family) {
  map <- c(evsdt = "evsd", uvsdt = "uvsd", dpsdt = "dpsd",
           bivariate_sdt = "bivariate_gaussian")
  if (!is.null(family) && family %in% names(map)) unname(map[family]) else family
}

#' Print method for a bayesroc family
#' @param x The object to print.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @export
print.broc_family <- function(x, ...) {
  cat("Family:", family_display_name(x$family), "\n")
  cat(x$description, "\n\n")
  cat("Parameters:\n")
  for (nm in names(x$params)) {
    p <- x$params[[nm]]
    # For aliased families, show the user-facing external name (rec/fam/mu/...)
    # rather than the internal Stan name (dprime/dprime2/...) since the external
    # name is what the user writes in formulas.
    display_nm <- if (!is.null(x$external_aliases) &&
                     !is.null(x$external_aliases[[nm]])) {
      x$external_aliases[[nm]]
    } else {
      nm
    }
    req <- if (p$required) "required" else "optional"
    desc <- if (!is.null(p$description)) paste0(" - ", p$description) else ""
    null_desc <- if (!is.null(p$null_description)) paste0(" (NULL = ", p$null_description, ")") else ""
    cat(sprintf("  %s: %s%s%s\n", display_nm, req, desc, null_desc))
  }
  invisible(x)
}
