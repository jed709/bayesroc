# Extract the Generated NumPyro (JAX) Config

Returns the config list that
[`broc()`](https://jed709.github.io/bayesroc/reference/broc.md)
generates for the JAX backend, the counterpart to
[`get_stan_code()`](https://jed709.github.io/bayesroc/reference/get_stan_code.md)/[`get_stan_data()`](https://jed709.github.io/bayesroc/reference/get_stan_data.md).
It describes the model (family, link, response data, design matrices,
priors, threshold and random-effect structure) and is serialized to JSON
for
[`fit_broc()`](https://jed709.github.io/bayesroc/reference/fit_broc.md)
with `backend = "jax"`.

## Usage

``` r
get_numpyro_config(x, json = FALSE)
```

## Arguments

- x:

  A `broc_model` object created by
  [`broc()`](https://jed709.github.io/bayesroc/reference/broc.md).

- json:

  If `TRUE`, return the pretty-printed JSON string actually written for
  the Python backend (matrices expanded to row lists), of class `json`.
  If `FALSE` (default), return the R list.

## Value

The config as a named list, or, when `json = TRUE`, a length-one
character vector of class `json`.

## See also

[`get_stan_code()`](https://jed709.github.io/bayesroc/reference/get_stan_code.md),
[`get_stan_data()`](https://jed709.github.io/bayesroc/reference/get_stan_data.md).

## Examples

``` r
if (FALSE) { # \dontrun{
model <- broc(brf(conf | old ~ cond), data = dat, family = evsd())
cfg <- get_numpyro_config(model)
str(cfg)
# Exactly what fit_broc(backend = "jax") sends to Python:
cat(get_numpyro_config(model, json = TRUE))
} # }
```
