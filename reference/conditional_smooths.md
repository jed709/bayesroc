# Conditional Smooth Effects

Evaluate smooth terms at a grid of covariate values using posterior
draws. Returns a data frame with posterior mean and credible intervals
for each smooth, suitable for plotting with ggplot2. Requires a model
fit with `s()`/`t2()` smooth terms in a parameter formula.

## Usage

``` r
conditional_smooths(
  fit,
  smooths = NULL,
  dpar = NULL,
  resolution = 100,
  prob = 0.95,
  ndraws = NULL
)
```

## Arguments

- fit:

  A `broc_fit` object from
  [`fit_broc()`](https://jed709.github.io/bayesroc/reference/fit_broc.md).

- smooths:

  Character vector of smooth terms to evaluate (e.g., "s(age)"). If NULL
  (default), all smooth terms are evaluated.

- dpar:

  Optional distributional parameter to filter to (e.g., `"dprime"`).

- resolution:

  Number of grid points per covariate (default 100).

- prob:

  Credible interval width (default 0.95).

- ndraws:

  Number of posterior draws to use (default: all).

## Value

A data frame of class `conditional_smooths` with covariate value(s),
parameter, smooth label, estimate, lower, upper.
