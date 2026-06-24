# Summarize a Fitted SDT Model

Produce a labeled table of posterior estimates for a fitted model. The
printed summary groups parameters into sections: fixed effects (per
distributional parameter, e.g. d', criterion, sigma), random-effect
standard deviations, random-effect correlations, and population-level
thresholds. The print method also reports sampler diagnostics.

## Usage

``` r
# S3 method for class 'broc_fit'
summary(object, prob = 0.95, digits = 3, threshold_natural = FALSE, ...)
```

## Arguments

- object:

  A `broc_fit` object from
  [`fit_broc()`](https://jed709.github.io/bayesroc/reference/fit_broc.md).

- prob:

  Probability mass for credible intervals (default 0.95)

- digits:

  Number of digits for rounding (default 3)

- threshold_natural:

  If `TRUE`, additionally report criterion random-effect SDs and
  correlations on the natural threshold scale rather than the internal
  log-gap scale.

- ...:

  Unused; present for S3 generic compatibility.

## Value

A structured summary object of class `broc_summary`. Its `print` method
also shows a sampler-diagnostics banner (number of divergent
transitions, max-treedepth hits, and minimum E-BFMI across chains).
