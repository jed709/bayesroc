# Override the Python executable used by the JAX backend

Sets `BAYESROC_PYTHON` for the current R session. Future
`fit_broc(backend = "jax")` calls will use this Python without going
through auto-discovery.

## Usage

``` r
set_jax_python(python)
```

## Arguments

- python:

  Path to a Python executable. Must already have `jax` and `numpyro`
  installed (use
  [`check_jax_backend()`](https://jed709.github.io/bayesroc/reference/check_jax_backend.md)
  to verify after).

## Value

Invisibly returns the new path.
