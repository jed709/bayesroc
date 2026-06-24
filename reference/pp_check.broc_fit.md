# Posterior Predictive Check for SDT Models

Compare observed response distributions against draws from the posterior
predictive distribution.

## Usage

``` r
# S3 method for class 'broc_fit'
pp_check(
  object,
  type = c("bars", "bars_grouped", "dens_overlay", "stat", "rootogram"),
  group = NULL,
  ndraws = 200,
  prob = 0.95,
  stat_fun = mean,
  re_formula = NULL,
  response = 1,
  cores = 1,
  ...
)
```

## Arguments

- object:

  A `broc_fit` object from
  [`fit_broc()`](https://jed709.github.io/bayesroc/reference/fit_broc.md).

- type:

  Plot type. `"bars"`: observed vs predicted counts per response
  category, with predictive intervals. `"bars_grouped"`: the same, split
  by `group`. `"rootogram"`: a hanging rootogram, emphasizing fit in
  low-count categories. `"stat"`: posterior distribution of a summary
  statistic (`stat_fun`) against its observed value. `"dens_overlay"`:
  observed density with overlaid predictive densities.

- group:

  Column name (from the original data) to facet by; `NULL` for no
  grouping. `type = "bars_grouped"` groups by this column.

- ndraws:

  Number of replicated datasets (default 200)

- prob:

  CI width for intervals on predicted counts (default 0.95)

- stat_fun:

  Summary statistic function for `type = "stat"` (default: mean)

- re_formula:

  Which random effects to include in prediction. `NULL` (default)
  includes all random effects; `NA` excludes them, giving
  population-level predictions.

- response:

  For bivariate families (bivariate_gaussian, vrdp2d): which dimension
  to check (1 = item/detection, 2 = source/discrimination). Default 1.

- cores:

  Number of cores used for posterior prediction (default 1).

- ...:

  Additional arguments (unused).

## Value

A ggplot object.

## Examples

``` r
if (FALSE) { # \dontrun{
fit <- fit_broc(model)
pp_check(fit, type = "bars")
pp_check(fit, type = "bars_grouped", group = "condition")
pp_check(fit, type = "rootogram")
} # }
```
