# Extract Stan Data List

Returns the data list that would be passed to Stan's `$sample()` method.

## Usage

``` r
get_stan_data(x)
```

## Arguments

- x:

  A `broc_model` object created by
  [`broc()`](https://jed709.github.io/bayesroc/reference/broc.md).

## Value

A named list of data values formatted for Stan.

## See also

[`get_stan_code()`](https://jed709.github.io/bayesroc/reference/get_stan_code.md)
for the generated program;
[`get_numpyro_config()`](https://jed709.github.io/bayesroc/reference/get_numpyro_config.md)
for the equivalent artifact used by the JAX backend.

## Examples

``` r
if (FALSE) { # \dontrun{
model <- broc(brf(conf | old ~ cond), data = dat, family = evsd())
str(get_stan_data(model))
} # }
```
