#' SDT Model Family Specifications
#'
#' These functions define the structure of different SDT model families.

# =============================================================================
# =============================================================================

#' @noRd
.crit_scale_spec <- function() {
  list(required = FALSE, default = NULL, link = "log",
       null_value = "k-1", null_description = "criteria parameterized by a midpoint and K-2 gaps",
       description = "criterion scale (log link): multiplies every gap, so the criteria stretch about their mean")
}

.param_name_map <- function(family_name) {
  switch(family_name,
    cdp                  = list(dprime = "rec", dprime2 = "fam",
                                sigma  = "sigma_R", sigma2 = "sigma_F",
                                rho = "rho_target", rho_N = "rho_lure"),
    metadt               = list(mratio = "mratio_old", mratio_N = "mratio_new"),
    cumulative           = list(dprime = "mu", criterion = "cutpoints", criterion_scale = "cutpoints_scale"),
    bivariate_cumulative = list(dprime = "mu", discrim = "mu2",
                                criterion = "cutpoints", criterion_scale = "cutpoints_scale",
                                criterion2 = "cutpoints2", criterion2_scale = "cutpoints2_scale"),
    list()
  )
}

#' @noRd
external_to_internal_param <- function(external_name, family_name) {
  m <- .param_name_map(family_name)
  if (length(m) == 0) return(external_name)
  inv <- setNames(names(m), unlist(m, use.names = FALSE))
  hit <- inv[external_name]
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

#' Complementary Log-Log Link Function
#'
#' Uses the CDF of the minimum extreme-value (Gumbel) distribution,
#' \eqn{F(x) = 1 - \exp(-\exp(x))}. With this link the equal-variance model
#' produces a zROC slope below 1 (Meyer-Grant et al., 2026).
#' @return A `broc_link` object to pass to a family's `link` argument.
#' @references Meyer-Grant, Kellen, Harding & Singmann (2026), Psychological
#'   Review. doi:10.1037/rev0000615
#' @export
link_cloglog <- function() {
  structure(list(name = "cloglog"), class = "broc_link")
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
#' @param link Link function: [link_probit()] (default), [link_logit()] or
#'   [link_cloglog()].
#' @param meta_noise Form of the meta-noise when `sigma_meta` is modelled:
#'   `"gaussian"` (default; Bang et al., 2019) adds normal noise to the
#'   evidence read by the rating; `"lognormal"` (Shekhar & Rahnev, 2021) makes
#'   the confidence criteria lognormal in their distance from c0.
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
#'   \item{sigma_meta}{(optional) Meta-noise SD: of the Gaussian noise added to
#'     the decision evidence before the rating (Bang et al., 2019), or of the
#'     log-distance of each confidence criterion from c0 (Shekhar & Rahnev,
#'     2021), as set by `meta_noise`. Log link. If omitted, the rating is read
#'     from the decision evidence against fixed criteria.}
#' }
#'
#' ## Gaussian meta-noise
#' With `sigma_meta`, the decision uses the evidence x (noise N(0, 1), signal
#' N(d', 1)) against the type-1 criterion c0, the middle threshold, and the
#' rating uses y = x + e, e ~ N(0, sigma_meta^2), against the confidence
#' criteria on the side of the decision; when y falls on the other side of c0
#' the rating is the lowest confidence. Each cell is a bivariate normal
#' rectangle with correlation 1 / sqrt(1 + sigma_meta^2). Requires the probit
#' link and an even number of response categories (at least 4). As
#' `sigma_meta` goes to 0 the model becomes the ordinary rating model.
#'
#' ## Lognormal meta-noise
#' With `meta_noise = "lognormal"`, the rating is read from the decision
#' evidence x, but on each trial the confidence criteria are
#' c0 + (c_k - c0) exp(sigma_meta z) on the side of the decision, with one
#' z ~ N(0, 1) shared by all criteria so that they never cross (Shekhar &
#' Rahnev, 2021). The thresholds c_k are the median criteria. Each cell is a
#' one-dimensional integral, computed by quadrature. Requires the probit link
#' and an even number of response categories (at least 4); as `sigma_meta`
#' goes to 0 the model becomes the ordinary rating model.
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
#' Bang, J. W., Shekhar, M., & Rahnev, D. (2019). Sensory noise increases
#'   metacognitive efficiency. *Journal of Experimental Psychology: General*,
#'   *148*(3), 437-452.
#'
#' Shekhar, M., & Rahnev, D. (2021). The nature of metacognitive inefficiency
#'   in perceptual decision making. *Psychological Review*, *128*(1), 45-70.
#'
#' @return A `broc_family` object.
#' @seealso [uvsd()] for unequal-variance extension, [dpsd()] for dual-process,
#'   [mixture()] for mixture variants.
#' @export
evsd <- function(link = link_probit(), meta_noise = c("gaussian", "lognormal")) {
  if (!inherits(link, "broc_link")) {
    stop("link must be created with link_probit(), link_logit(), or link_cloglog()")
  }
  meta_noise <- match.arg(meta_noise)
  structure(
    list(
      family = "evsdt",
      link = link,
      meta_noise = meta_noise,
      params = list(
        dprime = list(required = TRUE, default = NULL, link = "identity"),
        sigma_meta = list(required = FALSE, default = NULL, link = "log",
                          null_description = "ratings read from the decision evidence",
                          description = if (meta_noise == "gaussian")
                            "SD of the Gaussian meta-noise between decision and rating"
                          else "SD of the lognormal noise in the confidence criteria"),
        criterion_scale = .crit_scale_spec(),
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
#' @param link Link function: [link_probit()] (default), [link_logit()] or
#'   [link_cloglog()].
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
    stop("link must be created with link_probit(), link_logit(), or link_cloglog()")
  }
  structure(
    list(
      family = "uvsdt",
      link = link,
      params = list(
        dprime = list(required = TRUE, default = NULL, link = "identity"),
        criterion_scale = .crit_scale_spec(),
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
#' @param link Link function: [link_probit()] (default), [link_logit()] or
#'   [link_cloglog()].
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
    stop("link must be created with link_probit(), link_logit(), or link_cloglog()")
  }
  structure(
    list(
      family = "dpsdt",
      link = link,
      params = list(
        dprime = list(required = TRUE, default = NULL, link = "identity"),
        criterion_scale = .crit_scale_spec(),
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
#' @param link Link function: [link_probit()] (default), [link_logit()] or
#'   [link_cloglog()].
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
    stop("link must be created with link_probit(), link_logit(), or link_cloglog()")
  }
  structure(
    list(
      family = "mixture",
      link = link,
      params = list(
        dprime = list(required = TRUE, default = NULL, link = "identity",
                      description = "d' for state 1 (high state)"),
        criterion_scale = .crit_scale_spec(),
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
#' @param link Link function: [link_probit()] (default), [link_logit()] or
#'   [link_cloglog()].
#'
#' @details
#' ## Likelihood
#' For an ordinal response Y on a K-point scale with cutpoints
#' c_1(x) < ... < c_{K-1}(x):
#' \deqn{P(Y \le k \mid x) = F(c_k(x))}
#' with \eqn{c_0 = -\infty} and \eqn{c_K = +\infty}, where F is the link CDF.
#' Cell probabilities by differencing:
#' \eqn{P(Y = k \mid x) = F(c_k(x)) - F(c_{k-1}(x))}. The latent distribution is fixed
#' (mean 0, SD 1), so every effect is an effect on the cutpoints and the response formula
#' takes no predictors (`rating ~ 1`).
#'
#' ## Parameters
#' \describe{
#'   \item{cutpoints}{K-1 ordered cutpoints (a midpoint and K-2 ordered gaps; default `~ 1`).
#'     Without `cutpoints_scale`, every term has one coefficient per cutpoint (the midpoint
#'     and each log gap) and a random effect is K-1 correlated shifts per group. Identity link.
#'     Set priors via `thresh_mid` and `log_gaps`.}
#'   \item{cutpoints_scale}{(optional) Makes the cutpoints location-scale:
#'     \eqn{c_k(x) = L(x) + e^{s(x)} (t_k - \bar t)}, with the shape \eqn{t} set by the gap
#'     intercepts. `cutpoints` then gives the location \eqn{L} (the mean cutpoint), one
#'     coefficient per term, and `cutpoints_scale` gives the log scale \eqn{s}, which has no
#'     intercept of its own. A positive location coefficient moves responses down.}
#' }
#'
#' ## Formula syntax
#' ```
#' # random-intercept ordinal model with a condition shift
#' broc(brf(rating ~ 1,
#'          cutpoints ~ condition + (1 | subj) + (1 | item),
#'          cutpoints_scale ~ 1,
#'          family = cumulative()), data = dat)
#' # per-cutpoint condition and subject effects
#' broc(brf(rating ~ 1,
#'          cutpoints ~ condition + (1 | subj),
#'          family = cumulative()), data = dat)
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
    stop("link must be created with link_probit(), link_logit(), or link_cloglog()")
  }
  structure(
    list(
      family = "cumulative",
      link = link,
      is_sdt = FALSE,
      params = list(
        dprime    = list(required = TRUE, default = NULL, link = "identity",
                         description = "linear predictor for location"),
        criterion_scale = .crit_scale_spec(),
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
#' with \eqn{c_0 = -\infty}, \eqn{c_K = +\infty}. Both latent dimensions have mean 0
#' and variance 1, so effects go on the cutpoints and the response formula takes no
#' predictors. Cell probabilities by 4-corner differencing:
#' \deqn{P(Y_1 = j,\; Y_2 = k \mid x) = P(\le j, \le k) - P(\le j-1, \le k)
#'       - P(\le j, \le k-1) + P(\le j-1, \le k-1)}
#'
#' ## Parameters
#' \describe{
#'   \item{cutpoints}{K1-1 ordered cutpoints for dimension 1 (a midpoint and ordered
#'     gaps; default `~ 1`), one coefficient per cutpoint unless `cutpoints_scale` is given,
#'     as in [cumulative()]. Identity link. Set priors via `thresh_mid` and `log_gaps`.}
#'   \item{cutpoints_scale}{(optional) Location-scale cutpoints for dimension 1; see
#'     [cumulative()].}
#'   \item{cutpoints2}{K2-1 ordered cutpoints for dimension 2 (default `~ 1`).
#'     Identity link. Set priors via `thresh_mid2` and `log_gaps2`.}
#'   \item{cutpoints2_scale}{(optional) Location-scale cutpoints for dimension 2.}
#'   \item{rho}{(required) Correlation between dimensions; effective
#'     \eqn{\rho = \tanh(\rho_{\text{param}})}. Fisher z link.}
#' }
#'
#' ## Formula syntax
#' ```
#' broc(brf(resp(y1, y2) ~ 1,
#'          cutpoints ~ x1 + (1 | subj),
#'          cutpoints_scale ~ 1,
#'          cutpoints2 ~ x2 + (1 | subj),
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
      params = list(
        dprime     = list(required = TRUE, default = NULL, link = "identity",
                          description = "linear predictor for dimension 1 location"),
        discrim    = list(required = TRUE, default = NULL, link = "identity",
                          description = "linear predictor for dimension 2 location"),
        criterion  = list(required = TRUE, default = NULL, link = "identity",
                          description = "dimension 1 threshold/cutpoint parameters"),
        criterion2_scale = .crit_scale_spec(),
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
#' @param link Link function: [link_probit()] (default), [link_logit()] or
#'   [link_cloglog()].
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
    stop("link must be created with link_probit(), link_logit(), or link_cloglog()")
  }
  structure(
    list(
      family = "source_mixture",
      link = link,
      params = list(
        dprime = list(required = TRUE, default = NULL, link = "identity",
                      description = "d' for attended Source A items (Psi_A location)"),
        criterion_scale = .crit_scale_spec(),
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
# The source-criteria structure shared by the two-response SDT families: one set for every
# detection response ("common"), one set per detection response ("varying"), or one set per
# "old" response plus a single set for all "new" responses ("varying_shared_new").
.source_criteria_arg <- function(source_criteria, old_levels) {
  source_criteria <- match.arg(source_criteria, c("common", "varying", "varying_shared_new"))
  if (source_criteria == "varying_shared_new") {
    if (is.null(old_levels) || !is.numeric(old_levels) || length(old_levels) < 1) {
      stop("old_levels must be a numeric vector of the detection responses that count as ",
           "'old' (e.g. old_levels = 4:6) when source_criteria = \"varying_shared_new\".",
           call. = FALSE)
    }
    old_levels <- sort(as.integer(old_levels))
  } else if (!is.null(old_levels)) {
    stop("old_levels is only used with source_criteria = \"varying_shared_new\".", call. = FALSE)
  }
  list(source_criteria = source_criteria, varying = source_criteria != "common", old_levels = old_levels)
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
#' @param source_criteria How the source criteria are structured across
#'   detection responses. `"common"` (default): one set of source criteria for
#'   every detection response. `"varying"`: a separate set for each detection
#'   response (K1 x (K2-1) criteria in total). `"varying_shared_new"`: a separate
#'   set for each "old" detection response and a single set shared by all "new"
#'   responses; requires `old_levels`. Covariates, smooths and random effects on
#'   `criterion2` apply to every set, and a random effect on `criterion2` gives
#'   each source threshold of each set its own effect.
#' @param old_levels Integer vector of detection response values that count as
#'   "old" judgements, in the data's own coding (e.g. `old_levels = 4:6` for a
#'   1-6 scale, or `3:5` if the same scale is coded 0-5); the rest are "new".
#'   Required when `source_criteria = "varying_shared_new"`; not used otherwise.
#' @param bounded If `TRUE`, fit the Bounded Bivariate Gaussian (BBG; Starns
#'   et al. 2014) variant: dprime/discrim use log links (positive magnitudes),
#'   rho uses logistic link (in (0,1)), and the model anchors A on the
#'   negative source axis / B on the positive axis. Default `FALSE` fits the
#'   standard unbounded BG model (DeCarlo, 2003).
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
#'     Identity link (unbounded) or log link (bounded: positive magnitude,
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
bivariate_gaussian <- function(source_criteria = c("common", "varying", "varying_shared_new"),
                               bounded = FALSE, old_levels = NULL) {
  link <- link_probit()
  sc <- .source_criteria_arg(source_criteria, old_levels)
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
      source_criteria = sc$source_criteria,
      varying_source_criteria = sc$varying,
      bounded = bounded,
      old_levels = sc$old_levels,
      params = list(
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

        rho = list(required = TRUE, default = NULL, link = rho_link,
                   description = rho_desc),
        rho_B = list(required = FALSE, default = NULL, link = rho_link,
                     null_value = "symmetric",
                     null_description = "constrained to -rho (symmetric)",
                     description = "rho_B correlation for Source B"),
        rho_N = list(required = FALSE, default = NULL, link = "fisherz",
                     null_value = 0, null_description = "fixed at 0",
                     description = "rho_N correlation for new items"),

        criterion_scale = .crit_scale_spec(),
        criterion = list(required = TRUE, default = NULL, link = "identity",
                         description = "Detection thresholds (c1)"),
        criterion2_scale = .crit_scale_spec(),
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

#' Bounded Bivariate Dual Process Signal Detection Model
#'
#' Bounded bivariate dual-process model (BBDP): Gaussian familiarity with the
#' conditional source mean clamped at the new-item baseline, plus discrete item
#' and source recollection (Starns et al., 2014). The model is published only in
#' its bounded form, so there is no unbounded variant.
#'
#' Same source labeling convention as [bivariate_gaussian()]: source A = low
#' source ratings, source B = high.
#'
#' @param source_criteria How the source criteria are structured across
#'   detection responses. `"common"` (default): one set of source criteria for
#'   every detection response. `"varying"`: a separate set for each detection
#'   response (K1 x (K2-1) criteria in total). `"varying_shared_new"`: a separate
#'   set for each "old" detection response and a single set shared by all "new"
#'   responses; requires `old_levels`. Covariates, smooths and random effects on
#'   `criterion2` apply to every set, and a random effect on `criterion2` gives
#'   each source threshold of each set its own effect.
#' @param old_levels Integer vector of detection response values that count as
#'   "old" judgements, in the data's own coding (e.g. `old_levels = 4:6` for a
#'   1-6 scale, or `3:5` if the same scale is coded 0-5); the rest are "new".
#'   Required when `source_criteria = "varying_shared_new"`; not used otherwise.
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
#' The bivariate piece uses the source-mean clamping of Starns et al. (2014),
#' as in [bivariate_gaussian()] with `bounded = TRUE`.
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
#' As in the bounded [bivariate_gaussian()], `dprime` and `discrim` use log
#' links (positive magnitudes; A on the negative source axis, B on the positive)
#' and `rho` a logistic link on (0, 1).
#' Defaults for the bivariate parameters match [bivariate_gaussian()], e.g.
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
bivariate_dp <- function(source_criteria = c("common", "varying", "varying_shared_new"),
                         old_levels = NULL) {
  link <- link_probit()
  sc <- .source_criteria_arg(source_criteria, old_levels)
  dprime_link <- "log"
  discrim_link <- "log"
  rho_link <- "logis"
  dprime_desc <- "d' for Source A detection (log scale, always positive)"
  discrim_desc <- "Source A discrimination (log scale, always positive)"
  rho_desc <- "rho_A correlation for Source A (logistic scale, bounded to (0,1))"
  structure(
    list(
      family = "bivariate_dp",
      link = link,
      source_criteria = sc$source_criteria,
      varying_source_criteria = sc$varying,
      bounded = TRUE,
      old_levels = sc$old_levels,
      params = list(
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

        rho = list(required = TRUE, default = NULL, link = rho_link,
                   description = rho_desc),
        rho_B = list(required = FALSE, default = NULL, link = rho_link,
                     null_value = "symmetric",
                     null_description = "constrained mirror of rho (symmetric)",
                     description = "rho_B correlation for Source B"),
        rho_N = list(required = FALSE, default = NULL, link = "fisherz",
                     null_value = 0, null_description = "fixed at 0",
                     description = "rho_N correlation for new items"),

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

        criterion_scale = .crit_scale_spec(),
        criterion = list(required = TRUE, default = NULL, link = "identity",
                         description = "Detection thresholds (c1)"),
        criterion2_scale = .crit_scale_spec(),
        criterion2 = list(required = TRUE, default = NULL, link = "identity",
                          description = "Discrimination thresholds (c2)")
      ),
      description = "Bounded Bivariate Dual Process Signal Detection Model"
    ),
    class = "broc_family"
  )
}

#' Continuous Dual Process Signal Detection Model
#'
#' Continuous dual-process model for Remember/Know paradigms: memory strength is
#' the sum of a recollection and a familiarity signal, confidence is rated on the
#' summed strength, and the Remember judgment is a criterion on the recollection
#' signal (Wixted & Mickes, 2010).
#'
#'
#' @details
#' ## Likelihood
#' Recollection and familiarity are Gaussians,
#' \eqn{R \sim \mathcal{N}(\mu_R, \sigma_R^2)} and
#' \eqn{F \sim \mathcal{N}(\mu_F, \sigma_F^2)} for targets,
#' \eqn{R, F \sim \mathcal{N}(0, 1)} for lures, with correlation \eqn{r} (\eqn{r_T} for
#' targets, \eqn{r_L} for lures; 0 unless estimated, see below). Confidence is driven by
#' their sum,
#' \deqn{M = R + F \sim \mathcal{N}(\mu_R + \mu_F,\; \sigma_R^2 + \sigma_F^2 + 2 r \sigma_R \sigma_F),}
#' so \eqn{(R, M)} is bivariate normal with
#' \eqn{\rho = (\sigma_R + r \sigma_F) / \sigma_M}
#' (lures: \eqn{\sigma_M = \sqrt{2 + 2 r_L}}, \eqn{\rho = (1 + r_L) / \sigma_M}).
#'
#' `criterion` supplies J thresholds \eqn{\tau_1 < \dots < \tau_J} on M, where
#' J = `length(old_levels)`. A response falls in old bin j when
#' \eqn{\tau_j < M < \tau_{j+1}} (\eqn{\tau_{J+1} = \infty}); every rating outside
#' `old_levels` is a single collapsed "new" response,
#' \deqn{P(\mathrm{new}) = \Phi\left(\frac{\tau_1 - \mu_M}{\sigma_M}\right),}
#' so graded new-side ratings are pooled rather than modelled separately.
#'
#' ## Judgements on "new" responses
#' The remember/know rule is defined on R and F for every item, so it applies to
#' any confidence band. When the data carry a judgement on responses outside
#' `old_levels`, the model switches to the full confidence scale automatically:
#' `criterion` then has K - 1 thresholds for K confidence levels, the lowest band
#' is \eqn{M < \tau_1}, and every response with a judgement contributes the cell
#' \eqn{P(\tau_j < M < \tau_{j+1}, \mathrm{judgement})} with the same criteria
#' \eqn{c_R} and \eqn{c_K}. No parameters are added. A response with a missing
#' judgement, at any confidence level, contributes its band probability
#' \eqn{P(\tau_j < M < \tau_{j+1})} alone.
#'
#' With two `rk` levels the joint cell is a strip of the \eqn{(R, M)} plane:
#' \deqn{P(Y = j, \mathrm{R}) = P(R > c_R,\; \tau_j < M < \tau_{j+1}), \quad
#'       P(Y = j, \mathrm{K}) = P(R \le c_R,\; \tau_j < M < \tau_{j+1}).}
#' With three `rk` levels a second criterion \eqn{c_K} sits on the familiarity
#' axis and the judgment is sequential: Remember if \eqn{R > c_R}, else Know if
#' \eqn{F > c_K}, else Guess. The Guess cell is \eqn{G(\tau_{j+1}) - G(\tau_j)}
#' with \eqn{G(c) = P(R \le c_R, F \le c_K, M < c)}, and Know is the remainder of
#' the band.
#'
#' Lure means and SDs are fixed at 0 and 1 on both axes.
#'
#' ## Recollection-familiarity correlation
#' Wixted and Mickes (2010) define separate R-F correlations for targets and lures
#' and fix both at 0; Moran and Goshen-Gottstein (2015) report a positive one.
#' Which correlations are estimated follows from the formulas given:
#' \itemize{
#'   \item none of `rho`, `rho_target`, `rho_lure`: both correlations fixed at 0;
#'   \item `rho`: one correlation shared by targets and lures;
#'   \item `rho_target` and `rho_lure`: both estimated separately;
#'   \item `rho_target` alone: the lure correlation is fixed at 0;
#'   \item `rho_lure` alone: the target correlation is fixed at 0.
#' }
#' `rho` cannot be combined with `rho_target` or `rho_lure`. All three use a
#' Fisher z link, \eqn{r = \tanh(\eta)}. Priors use the formulas' spelling
#' (`dpar = "rho"` for the shared correlation, `"rho_target"` / `"rho_lure"` for
#' separate ones).
#'
#' ## Parameters
#' \describe{
#'   \item{rec}{(required) Recollection mean \eqn{\mu_R} for targets. Identity link.}
#'   \item{fam}{(required) Familiarity mean \eqn{\mu_F} for targets. Identity link.}
#'   \item{criterion}{(required) J thresholds \eqn{\tau} on memory strength
#'     \eqn{M = R + F}, J = `length(old_levels)`. Identity link.}
#'   \item{rec_crit}{(required) Remember criterion \eqn{c_R} on the recollection
#'     axis. Identity link.}
#'   \item{sigma_R}{(optional) Recollection SD for targets. If omitted, fixed at 1.
#'     Log link.}
#'   \item{sigma_F}{(optional) Familiarity SD for targets. If omitted, fixed at 1.
#'     Log link.}
#'   \item{know_crit}{(optional) Know criterion \eqn{c_K} on the familiarity axis,
#'     required when `rk` has three levels. Identity link.}
#'   \item{rho}{(optional) R-F correlation shared by targets and lures. Fisher z link.}
#'   \item{rho_target}{(optional) R-F correlation for targets. Fisher z link.}
#'   \item{rho_lure}{(optional) R-F correlation for lures. Fisher z link.}
#' }
#'
#' ## Response coding
#' `rk` values: 1 = Remember, 2 = Know, (3 = Guess for R/K/G mode). Centered
#' `is_old` coding is not supported.
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
#' Moran, R., & Goshen-Gottstein, Y. (2015). Old processes, new perspectives:
#'   Familiarity is correlated with (not independent of) recollection and is more
#'   (not equally) variable for targets than for lures. *Cognitive Psychology*,
#'   *79*, 40-67.
#'
#' @return A `broc_family` object.
#' @export
cdp <- function(old_levels = NULL) {
  link <- link_probit()

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
      params = list(
        dprime    = list(required = TRUE, default = NULL, link = "identity",
                         description = "Recollection mean (mu_R) for targets"),
        sigma     = list(required = FALSE, default = NULL, link = "log",
                         null_value = 0, null_description = "fixed at 1",
                         description = "Recollection SD for targets"),
        dprime2   = list(required = TRUE, default = NULL, link = "identity",
                         description = "Familiarity mean (mu_F) for targets"),
        sigma2    = list(required = FALSE, default = NULL, link = "log",
                         null_value = 0, null_description = "fixed at 1",
                         description = "Familiarity SD for targets"),
        rec_crit  = list(required = TRUE, default = NULL, link = "identity",
                         description = "Remember criterion (c_R) on the recollection axis"),
        know_crit = list(required = FALSE, default = NULL, link = "identity",
                         description = "Know criterion (c_K) on the familiarity axis (R/K/G mode)"),
        rho       = list(required = FALSE, default = NULL, link = "fisherz",
                         null_value = 0, null_description = "fixed at 0",
                         description = "Correlation of recollection and familiarity for targets"),
        rho_N     = list(required = FALSE, default = NULL, link = "fisherz",
                         null_value = 0, null_description = "fixed at 0",
                         description = "Correlation of recollection and familiarity for lures"),
        criterion_scale = .crit_scale_spec(),
        criterion = list(required = TRUE, default = NULL, link = "identity",
                         description = "Confidence thresholds (tau) on memory strength M = R + F")
      ),
      external_aliases = .param_name_map("cdp"),
      description = "Continuous Dual Process Signal Detection Model"
    ),
    class = "broc_family"
  )
}

#' Meta-d' Model of Confidence Ratings
#'
#' The meta-d' model of type-2 sensitivity (Maniscalco & Lau, 2012, 2014),
#' fitted jointly to the old/new decision and the confidence ratings, with
#' crossed random effects available on every parameter.
#'
#' @param metac Where the meta criterion sits. `"absolute"` (default): at the
#'   type-1 criterion, as in HMeta-d (Fleming, 2017), the hmetad package and
#'   bmm. `"relative"`: scaled with the M-ratio about the midpoint of the two
#'   distributions, so that meta-c' = c' (Maniscalco & Lau, 2014; hmetad with
#'   `metac_absolute = FALSE`).
#' @param link Latent-distribution link: [link_probit()] (default, the
#'   published model), [link_logit()] or [link_cloglog()]. Both the type-1 and
#'   the meta-level distributions take the link's shape, so `dprime` and the
#'   criteria are on that link's scale, and the M-ratio keeps its meaning as
#'   meta-d'/d' on it.
#'
#' @details
#' ## Likelihood
#' Responses are a K-point scale with K even and at least 4: responses
#' 1, ..., K/2 are "new" at decreasing confidence, K/2 + 1, ..., K are "old"
#' at increasing confidence. The decision follows equal-variance SDT with new
#' items N(0, 1), old items N(d', 1) and the type-1 criterion c, the middle
#' threshold \eqn{\tau_{K/2}} (shown for the probit link; `link` swaps
#' \eqn{\Phi} for the logistic or Gumbel-min CDF throughout). With m = d'/2 the midpoint, the confidence
#' rating given the decision comes from a second, meta-level SDT model with
#' means \eqn{m \pm M d'/2} and confidence criteria \eqn{t_k}, truncated at
#' the meta criterion \eqn{t_{K/2}}:
#' \deqn{P(Y = k \mid S) = P(R \mid S) \frac{\Phi(t_k - \mu'_S) - \Phi(t_{k-1} - \mu'_S)}{P'(R \mid S)}}
#' where \eqn{P'(R \mid S)} is the meta-level probability of the same
#' decision. For `metac = "absolute"`, \eqn{t_k = \tau_k}. For `"relative"`,
#' every criterion on the response side shifts by \eqn{(M - 1)(c - m)}, which
#' puts the meta criterion at \eqn{m + M(c - m)} and keeps the confidence
#' gaps. With M = 1 the model is [evsd()] on the same thresholds.
#'
#' ## Parameters
#' \describe{
#'   \item{dprime}{(required) Type-1 sensitivity. Identity link.}
#'   \item{criterion}{(required) The K-1 thresholds: the middle one is the
#'     type-1 criterion c, the others are the confidence criteria
#'     (mid-anchor + K-2 ordered gaps). Identity link.}
#'   \item{mratio}{(optional) M-ratio meta-d'/d', shared by both responses.
#'     Log link.}
#'   \item{mratio_old}{(optional) M-ratio for "old" responses. Log link.}
#'   \item{mratio_new}{(optional) M-ratio for "new" responses. Log link.}
#' }
#' Give `mratio` for one M-ratio, or `mratio_old` and/or `mratio_new` for
#' response-specific M-ratios (Maniscalco & Lau, 2014); a side that is not
#' given is fixed at M = 1. With no M-ratio formula, M = 1 everywhere.
#'
#' ## Formula syntax
#' ```
#' broc(brf(conf | old ~ 1 + (1 | subj) + (1 | item),
#'          criterion ~ 1 + (1 | subj) + (1 | item),
#'          mratio ~ 1 + (1 | subj) + (1 | item),
#'          family = metad()),
#'     data = dat)
#' ```
#'
#' @references
#' Maniscalco, B., & Lau, H. (2012). A signal detection theoretic approach for
#'   estimating metacognitive sensitivity from confidence ratings.
#'   *Consciousness and Cognition*, *21*(1), 422-430.
#'
#' Maniscalco, B., & Lau, H. (2014). Signal detection theory analysis of type 1
#'   and type 2 data: Meta-d', response-specific meta-d', and the
#'   unequal variance SDT model. In S. M. Fleming & C. D. Frith (Eds.),
#'   *The cognitive neuroscience of metacognition* (pp. 25-66). Springer.
#'
#' Fleming, S. M. (2017). HMeta-d: Hierarchical Bayesian estimation of
#'   metacognitive efficiency from confidence ratings. *Neuroscience of
#'   Consciousness*, *2017*(1), nix007.
#'
#' @return A `broc_family` object.
#' @seealso [evsd()], the type-1 model it reduces to at M = 1.
#' @export
metad <- function(metac = c("absolute", "relative"), link = link_probit()) {
  metac <- match.arg(metac)
  if (!inherits(link, "broc_link")) {
    stop("link must be created with link_probit(), link_logit(), or link_cloglog()")
  }
  structure(
    list(
      family = "metadt",
      link = link,
      metac = metac,
      params = list(
        dprime = list(required = TRUE, default = NULL, link = "identity",
                      description = "Type-1 sensitivity d'"),
        mratio = list(required = FALSE, default = NULL, link = "log",
                      null_value = 0, null_description = "fixed at 1",
                      description = "M-ratio (meta-d'/d') for \"old\" responses"),
        mratio_N = list(required = FALSE, default = NULL, link = "log",
                        null_value = 0, null_description = "fixed at 1",
                        description = "M-ratio (meta-d'/d') for \"new\" responses"),
        criterion_scale = .crit_scale_spec(),
        criterion = list(required = TRUE, default = NULL, link = "identity",
                         description = "Thresholds: the middle one is the type-1 criterion, the rest confidence criteria")
      ),
      external_aliases = .param_name_map("metadt"),
      description = paste0("Meta-d' Model (", metac, " meta criterion)")
    ),
    class = "broc_family"
  )
}

#' m-Alternative Forced Choice Model
#'
#' Signal detection model for m-alternative forced choice with position bias
#' (DeCarlo, 2012). Each of the m positions yields evidence X_i; the target's is
#' N(d', 1), every lure's N(0, 1); the observer chooses the position with the
#' largest X_i + b_i.
#'
#' @param alternatives `NULL` (default): every trial has m alternatives, m the
#'   largest position in the data. Or the name of a column holding each trial's
#'   number of alternatives, for designs that mix m.
#'
#' @details
#' ## Likelihood
#' With the target at position s, choice k and position biases b (the last
#' position the reference, b_M = 0), a correct choice has probability
#' \deqn{P(k = s) = \int \phi(z) \prod_{j \ne s} \Phi(z + d' + b_s - b_j) dz}
#' and an error
#' \deqn{P(k) = \int \phi(z) \Phi(z - d' + b_k - b_s) \prod_{j \ne k, s} \Phi(z + b_k - b_j) dz.}
#' The data are the full target-by-choice table, fitted with its categorical
#' likelihood. DeCarlo's own estimates use a product of binary likelihoods (a
#' composite likelihood), so they differ slightly from these. Cells are
#' computed by quadrature; m = 2 is closed form.
#'
#' ## Equal variance
#' The target's evidence has the lures' SD. A first choice among m alternatives
#' does not identify an unequal-variance target: at m = 2 only d'/sqrt(1 +
#' sigma^2) enters, and at m >= 3 without position bias only the proportion
#' correct does. Identifying sigma takes second choices or rankings (Kellen &
#' Klauer, 2011), which this family does not model.
#'
#' ## Parameters
#' \describe{
#'   \item{dprime}{(required) Sensitivity. Identity link.}
#'   \item{bias}{(optional) Position biases b_1, ..., b_(M-1), one formula for
#'     all: each position gets its own coefficients and random effects,
#'     correlated within a grouping factor unless
#'     `gr(g, cor_thresholds = FALSE)`. Identity link. If omitted, 0.}
#' }
#'
#' ## Formula syntax
#' ```
#' broc(brf(choice | target ~ cond + (1 | subj),
#'          bias ~ 1 + (1 | subj),
#'          family = mafc()),
#'     data = dat)
#' ```
#' `choice` and `target` are positions 1, ..., m.
#'
#' @references
#' DeCarlo, L. T. (2012). On a signal detection approach to m-alternative
#'   forced choice with bias, with maximum likelihood and Bayesian approaches to
#'   estimation. *Journal of Mathematical Psychology*, *56*(3), 196-207.
#'
#' Kellen, D., & Klauer, K. C. (2011). Evaluating models of recognition memory
#'   using first- and second-choice responses. *Journal of Mathematical
#'   Psychology*, *55*(3), 251-266.
#'
#' @return A `broc_family` object.
#' @seealso [evsd()], the yes/no model with the same d'.
#' @export
mafc <- function(alternatives = NULL) {
  if (!is.null(alternatives) && !(is.character(alternatives) && length(alternatives) == 1L)) {
    stop("alternatives must be NULL or the name of a column of each trial's number of alternatives.", call. = FALSE)
  }
  structure(
    list(
      family = "mafc",
      link = link_probit(),
      alternatives = alternatives,
      params = list(
        dprime = list(required = TRUE, default = NULL, link = "identity",
                      description = "Sensitivity d'"),
        bias = list(required = FALSE, default = NULL, link = "identity",
                    null_description = "no position bias",
                    description = "Position biases, the last position the reference")
      ),
      description = "m-Alternative Forced Choice Model (DeCarlo 2012)"
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
#' Same source labeling convention as [bivariate_gaussian()]: source A is the
#' source associated with low source ratings, source B with high ratings.
#'
#'
#' @param source_criteria How the source criteria are structured across
#'   detection responses. `"common"` (default): one set of source criteria for
#'   every detection response. `"varying"`: a separate set for each detection
#'   response (K1 x (K2-1) criteria in total). `"varying_shared_new"`: a separate
#'   set for each "old" detection response and a single set shared by all "new"
#'   responses; requires `old_levels`. Covariates, smooths and random effects on
#'   `criterion2` apply to every set, and a random effect on `criterion2` gives
#'   each source threshold of each set its own effect.
#' @param old_levels Integer vector of detection response values that count as
#'   "old" judgements, in the data's own coding (e.g. `old_levels = 4:6` for a
#'   1-6 scale, or `3:5` if the same scale is coded 0-5); the rest are "new".
#'   Required when `source_criteria = "varying_shared_new"`; not used otherwise.
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
#' where \eqn{\psi_A = d'_S, \psi_B = -d'_S} (the recollection-conditional
#' source location for A vs B; a negative \eqn{d'_S} places A on the low side),
#' and lambda is the recollection probability
#' (`lambda`). The non-recollected component has zero source mean (no source
#' info); the recollected component carries source discriminability.
#'
#' ## Parameters
#' \describe{
#'   \item{dprime}{(required) \eqn{d'_F}: familiarity strength on the item
#'     dimension. Identity link.}
#'   \item{dprime2}{(required) \eqn{d'_R}: recollection boost on the item
#'     dimension. Identity link.}
#'   \item{discrim}{(required) \eqn{d'_S}: signed source-axis position of
#'     recollected A items (non-recollected items center at 0 on source);
#'     negative when A items get low source ratings. Identity link.}
#'   \item{lambda}{(required) Recollection probability. Logit link.}
#'   \item{criterion}{(required) K1-1 item-dimension thresholds. Identity link.}
#'   \item{criterion2}{(required) K2-1 source-dimension thresholds. Identity link.}
#'   \item{sigma}{(optional) SD on the item dimension. If omitted, fixed at 1.
#'     Log link.}
#'   \item{sigma2}{(optional) SD on the source dimension for recollected items.
#'     If omitted, fixed at 1. Log link.}
#'   \item{discrim_B}{(optional) Signed source-axis position of recollected B
#'     items. If omitted,
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
vrdp2d <- function(source_criteria = c("common", "varying", "varying_shared_new"), old_levels = NULL) {
  link <- link_probit()
  sc <- .source_criteria_arg(source_criteria, old_levels)
  structure(
    list(
      family = "vrdp2d",
      link = link,
      source_criteria = sc$source_criteria,
      varying_source_criteria = sc$varying,
      old_levels = sc$old_levels,
      params = list(
        dprime = list(required = TRUE, default = NULL, link = "identity",
                      description = "Familiarity strength (item dimension, new vs old)"),
        
        dprime2 = list(required = TRUE, default = NULL, link = "identity",
                       description = "Recollection strength boost on item dimension"),
        
        discrim = list(required = TRUE, default = NULL, link = "identity",
                       description = "Signed source-axis position of recollected Source A items"),
        
        discrim_B = list(required = FALSE, default = NULL, link = "identity",
                         null_value = "symmetric",
                         null_description = "constrained to -discrim (symmetric sources)",
                         description = "Signed source-axis position of recollected Source B items"),
        
        lambda = list(required = TRUE, default = NULL, link = "logit",
                      description = "R: Probability of recollection for old items"),
        
        sigma = list(required = FALSE, default = NULL, link = "log",
                     null_value = 1, null_description = "fixed at 1",
                     description = "sigma_item: SD on item dimension for old items"),
        
        sigma2 = list(required = FALSE, default = NULL, link = "log",
                      null_value = 1, null_description = "fixed at 1",
                      description = "sigma_S: SD on source dimension for recollected items"),
        
        criterion_scale = .crit_scale_spec(),
        criterion = list(required = TRUE, default = NULL, link = "identity",
                         description = "Item dimension thresholds (K1-1 thresholds)"),
        criterion2_scale = .crit_scale_spec(),
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
    
    if (param_name %in% names(provided_params) && 
        !param_name %in% names(family$params)) {
      stop(sprintf("Parameter '%s' is not supported by family '%s'",
                   param_name, family$family))
    }
  }
  
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
  map <- c(evsdt = "evsd", uvsdt = "uvsd", dpsdt = "dpsd", metadt = "metad",
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
