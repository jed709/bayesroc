# Posterior draws of a linear predictor (dpar)

Returns posterior draws of a distributional parameter at each row of
`newdata` on the response scale.

## Usage

``` r
# S3 method for class 'broc_fit'
posterior_linpred(
  object,
  transform = FALSE,
  newdata = NULL,
  dpar = "dprime",
  re_formula = NULL,
  ndraws = NULL,
  summary = FALSE,
  prob = 0.95,
  allow_new_levels = FALSE,
  seed = NULL,
  ...
)
```

## Arguments

- object:

  A `broc_fit` object from
  [`fit_broc()`](https://jed709.github.io/bayesroc/reference/fit_broc.md).

- transform:

  Ignored.

- newdata:

  Optional data frame. If `NULL`, uses the training data.

- dpar:

  Distributional parameter to extract (default `"dprime"`). Valid names
  depend on the family – the same names used in
  [`brf()`](https://jed709.github.io/bayesroc/reference/brf.md) and
  [`broc_prior()`](https://jed709.github.io/bayesroc/reference/broc_prior.md).

- re_formula:

  `NULL` (default) includes random effects; `NA` excludes them
  (population-level prediction).

- ndraws:

  Optional number of posterior draws to subsample. Default uses all
  draws.

- summary:

  If `TRUE`, returns a data frame with `mean`, `lower`, `upper`,
  `trial`. If `FALSE` (default), returns the full S x N matrix.

- prob:

  CI width when `summary = TRUE` (default 0.95).

- allow_new_levels:

  If `TRUE`, sample random effects from their prior for grouping levels
  in `newdata` not seen during training. Default `FALSE` errors if
  `newdata` introduces new levels.

- seed:

  Optional integer for reproducibility (subsampling and new-level RE
  draws). Restores caller's RNG state on exit.

- ...:

  Ignored.

## Value

If `summary = FALSE`, an S x N matrix. If `TRUE`, a data frame with
`trial`, `mean`, `lower`, `upper`.

## See also

[`posterior_epred()`](https://mc-stan.org/rstantools/reference/posterior_epred.html)
for expected response value;
[`predict()`](https://rdrr.io/r/stats/predict.html) for category
probabilities.
