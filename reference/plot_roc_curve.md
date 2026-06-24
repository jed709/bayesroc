# Plot Model-Implied ROC Curves

Generates receiver operating characteristic (ROC) curves from a fitted
Bayesian SDT model, with posterior credible bands and optional empirical
ROC points, on standard or zROC (probit-transformed) scales.

## Usage

``` r
plot_roc_curve(
  fit,
  response = 1,
  group = NULL,
  empirical = TRUE,
  scale = c("probability", "z"),
  ndraws = 200,
  prob = 0.95,
  re_formula = NULL,
  cores = 1
)
```

## Arguments

- fit:

  A `broc_fit` object from
  [`fit_broc()`](https://jed709.github.io/bayesroc/reference/fit_broc.md).

- response:

  Which response dimension to plot for bivariate families: `1` (default)
  for detection, `2` for source. Ignored for univariate families. For
  `response = 2`, two ROC curves are plotted (Source A vs New, Source B
  vs New).

- group:

  Optional character vector of column name(s) from the original data for
  per-condition ROC curves (faceted). E.g., `group = "condition"`.

- empirical:

  If `TRUE` (default), overlay empirical ROC points computed from
  observed data.

- scale:

  `"probability"` (default) for standard ROC, `"z"` for zROC with
  qnorm-transformed axes (linear zROC = Gaussian SDT; slope = 1/sigma
  for UVSDT).

- ndraws:

  Number of posterior draws to use (default 200).

- prob:

  Width of credible interval for the ribbon (default 0.95).

- re_formula:

  `NULL` (default) includes all random effects; `NA` for
  population-level ROC only.

- cores:

  Number of cores for parallel computation.

## Value

A ggplot2 object.

## Examples

``` r
if (FALSE) { # \dontrun{
# Standard ROC
plot_roc_curve(fit)

# zROC (probit-transformed)
plot_roc_curve(fit, scale = "z")

# Bivariate: source dimension ROC
plot_roc_curve(fit_bv, response = 2)

# Per-condition ROC curves
plot_roc_curve(fit, group = "condition")
} # }
```
