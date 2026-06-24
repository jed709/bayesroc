# Load a previously-saved broc_fit

Counterpart to
[`save_broc_fit()`](https://jed709.github.io/bayesroc/reference/save_broc_fit.md).
Reconstructs the broc_fit wrapper from the saved data bundle, restoring
all methods (`$draws()`, `$summary()`, `$loo()`, etc.) and the
`broc_model` attribute that downstream functions need.

## Usage

``` r
load_broc_fit(file)
```

## Arguments

- file:

  Path to a .rds file written by
  [`save_broc_fit()`](https://jed709.github.io/bayesroc/reference/save_broc_fit.md).

## Value

A `broc_fit` object.

## See also

[`save_broc_fit()`](https://jed709.github.io/bayesroc/reference/save_broc_fit.md).
