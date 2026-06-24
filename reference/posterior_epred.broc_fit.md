# Posterior expected response value

For ordinal models, returns `E[Y | X] = sum_k k * P(Y = k)` for each row
of `newdata`, per posterior draw.

## Usage

``` r
# S3 method for class 'broc_fit'
posterior_epred(
  object,
  newdata = NULL,
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

- newdata:

  Optional data frame. If `NULL`, uses the training data.

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

  Further arguments passed to
  [`predict.broc_fit()`](https://jed709.github.io/bayesroc/reference/predict.broc_fit.md)
  (e.g. `cores`).

## Value

If `summary = FALSE`, an S x N matrix of expected category indices. If
`TRUE`, a data frame with `trial`, `mean`, `lower`, `upper`.

## See also

[`posterior_linpred()`](https://mc-stan.org/rstantools/reference/posterior_linpred.html),
[`predict()`](https://rdrr.io/r/stats/predict.html).
