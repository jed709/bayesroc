# Diagnose the JAX backend setup

Reports the resolved Python, whether `jax` and `numpyro` import, and
whether their installed versions are compatible.

## Usage

``` r
check_jax_backend()
```

## Value

Invisibly, a list with `status` (one of `"ok"`, `"compat_warning"`,
`"missing_numpyro"`, `"no_jax"`, `"no_python"`), `python`,
`jax_version`, `numpyro_version`, and `message`.

## See also

[`install_jax_backend()`](https://jed709.github.io/bayesroc/reference/install_jax_backend.md).
