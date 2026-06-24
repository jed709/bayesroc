# LOO-CV for Fitted SDT Models

Approximate leave-one-out cross-validation (PSIS-LOO) for model
comparison via
[`loo::loo_compare()`](https://mc-stan.org/loo/reference/loo_compare.html).

## Usage

``` r
# S3 method for class 'broc_fit'
loo(x, cores = 4, ...)
```

## Arguments

- x:

  A `broc_fit` object from
  [`fit_broc()`](https://jed709.github.io/bayesroc/reference/fit_broc.md).

- cores:

  Number of cores for the PSOCK cluster (default 4)

- ...:

  Additional arguments passed to
  [`loo::loo()`](https://mc-stan.org/loo/reference/loo.html)

## Value

A `loo` object.

## Details

For trial-level data, passes log_lik directly to loo. For aggregated
data with counts, `log_lik[n]` is the per-row log-probability (without
count weighting).

Computed in parallel across observations with a PSOCK cluster.
