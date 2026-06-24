# Posterior Predictions from a Fitted Bayesian ROC Model

Generate posterior predictive response probabilities or simulated
responses from a fitted model.

## Usage

``` r
# S3 method for class 'broc_fit'
predict(
  object,
  newdata = NULL,
  type = c("response", "prediction"),
  re_formula = NULL,
  ndraws = NULL,
  summary = TRUE,
  prob = 0.95,
  cores = 1,
  response = 1,
  seed = NULL,
  allow_new_levels = FALSE,
  ...
)
```

## Arguments

- object:

  A `broc_fit` object from
  [`fit_broc()`](https://jed709.github.io/bayesroc/reference/fit_broc.md).

- newdata:

  Optional data frame for out-of-sample prediction. If `NULL` (default),
  uses the training data.

- type:

  `"response"` (default) returns posterior category probabilities P(Y =
  k); `"prediction"` returns simulated categorical responses.

- re_formula:

  Controls random effects inclusion. `NULL` (default) includes all
  random effects; `NA` excludes all (population-level predictions only).

- ndraws:

  Number of posterior draws to use. `NULL` (default) uses all available
  draws.

- summary:

  If `TRUE` (default), returns posterior mean and credible intervals. If
  `FALSE`, returns the full S x N x K array of draws.

- prob:

  Width of credible intervals when `summary = TRUE` (default 0.95).

- cores:

  Number of cores for parallel probability computation (default 1).

- response:

  Which response dimension. For bivariate families: `1` (detection) or
  `2` (source). For CDP: `1` (confidence, default) or `2` (R/K/G
  probabilities). Ignored for other families.

- seed:

  Optional integer for reproducibility. Controls the random subsampling
  of `ndraws` and (for `type = "prediction"`) the categorical draws used
  to simulate responses. The previous `.Random.seed` is restored on
  exit. Default `NULL` leaves the RNG untouched.

- allow_new_levels:

  If `TRUE`, allow `newdata` to contain random-effect levels not seen
  during fitting; their effects are drawn from the estimated population
  distribution. Default `FALSE` errors if `newdata` introduces new
  levels.

- ...:

  Ignored.

## Value

Depends on `type` and `summary`:

- `type = "response"`, `summary = TRUE`:

  A data frame with columns for each category's posterior mean
  probability plus lower/upper CI bounds.

- `type = "response"`, `summary = FALSE`:

  An S x N x K array where S = draws, N = observations, K = response
  categories.

- `type = "prediction"`, `summary = TRUE`:

  A vector of modal predicted categories.

- `type = "prediction"`, `summary = FALSE`:

  An S x N matrix of simulated categorical responses.

## See also

[`pp_check()`](https://mc-stan.org/bayesplot/reference/pp_check.html)
for visual posterior predictive checks.

## Examples

``` r
if (FALSE) { # \dontrun{
fit <- fit_broc(model)
# Category probabilities (summarized)
preds <- predict(fit)
# Full posterior array
preds_full <- predict(fit, summary = FALSE, ndraws = 100)
# Population-level only (no random effects)
preds_pop <- predict(fit, re_formula = NA)
} # }
```
