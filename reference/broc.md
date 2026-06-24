# Specify a Bayesian SDT Model

Builds a model specification from a formula, family, and (optional)
priors, and generates the backend code needed to fit it.
[`fit_broc()`](https://jed709.github.io/bayesroc/reference/fit_broc.md)
is used to fit the object created by `broc()`.

## Usage

``` r
broc(
  formula = NULL,
  data = NULL,
  family = NULL,
  priors = NULL,
  encoding_vars = NULL,
  counts = NULL,
  cor_threshold = NULL,
  threads = NULL,
  gap_link = NULL,
  batch_likelihood = TRUE
)
```

## Arguments

- formula:

  A [`brf()`](https://jed709.github.io/bayesroc/reference/brf.md)
  object, or a bare two-sided formula `response | is_old ~ predictors`.

- data:

  A data frame containing the response, the `is_old` indicator, and
  every predictor and grouping factor named in `formula`.

- family:

  The SDT model family: one of
  [`evsd()`](https://jed709.github.io/bayesroc/reference/evsd.md),
  [`uvsd()`](https://jed709.github.io/bayesroc/reference/uvsd.md),
  [`dpsd()`](https://jed709.github.io/bayesroc/reference/dpsd.md),
  [`mixture()`](https://jed709.github.io/bayesroc/reference/mixture.md),
  [`source_mixture()`](https://jed709.github.io/bayesroc/reference/source_mixture.md),
  [`bivariate_gaussian()`](https://jed709.github.io/bayesroc/reference/bivariate_gaussian.md),
  [`bivariate_dp()`](https://jed709.github.io/bayesroc/reference/bivariate_dp.md),
  [`vrdp2d()`](https://jed709.github.io/bayesroc/reference/vrdp2d.md),
  [`cdp()`](https://jed709.github.io/bayesroc/reference/cdp.md),
  [`cumulative()`](https://jed709.github.io/bayesroc/reference/cumulative.md),
  or
  [`bivariate_cumulative()`](https://jed709.github.io/bayesroc/reference/bivariate_cumulative.md).
  May also be set in
  [`brf()`](https://jed709.github.io/bayesroc/reference/brf.md); a value
  in broc() takes precedence.

- priors:

  A prior specification from
  [`broc_prior()`](https://jed709.github.io/bayesroc/reference/broc_prior.md).
  If `NULL`, the family's weakly-informative default priors are used
  (inspect them with
  [`get_broc_prior()`](https://jed709.github.io/bayesroc/reference/get_broc_prior.md)).

- encoding_vars:

  Character vector of column names to treat as encoding-phase (study)
  manipulations for which lure items do not have a meaningful value.
  Listing a column here auto-wraps it in `encoding()`. May also be set
  in [`brf()`](https://jed709.github.io/bayesroc/reference/brf.md); a
  value in broc() takes precedence.

- counts:

  Name of a column giving per-row trial counts, for aggregated (count)
  data rather than one row per trial. May also be set in
  [`brf()`](https://jed709.github.io/bayesroc/reference/brf.md).

- cor_threshold:

  If `TRUE` (default), random effects on the K-1 thresholds are modeled
  jointly (correlated) via the threshold parameterization; `FALSE`
  treats them as independent.

- threads:

  If `TRUE`, enable within-chain `reduce_sum` threading for the Stan
  backend (set the thread count per chain via
  `fit_broc(threads_per_chain=)`). Default `FALSE`.

- gap_link:

  Link for threshold gaps: `"log"` (default) or `"softplus"`. `NULL`
  uses the family default.

- batch_likelihood:

  If `TRUE` (default), build the fused batch likelihood with analytic
  gradients for the Stan backend.

## Value

A `broc_model` object, ready to pass to
[`fit_broc()`](https://jed709.github.io/bayesroc/reference/fit_broc.md).

## Examples

``` r
if (FALSE) { # \dontrun{
# Unequal-variance model (construction only; pass to fit_broc() to fit)
m <- broc(brf(conf | old ~ cond + (1 | subj),
              criterion ~ 1 + (1 | subj),
              sigma ~ 1,
              family = uvsd()),
          data = dat)

# Bare-formula shorthand (criterion defaults to ~ 1)
broc(conf | old ~ cond, data = dat, family = evsd())
} # }
```
