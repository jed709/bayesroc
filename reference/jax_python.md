# Resolve the Python executable used by the JAX backend

Returns the Python path that `fit_broc(backend = "jax")` will currently
use. Resolution order:

1.  `Sys.getenv("BAYESROC_PYTHON")` if set and the file exists (set by
    [`set_jax_python()`](https://jed709.github.io/bayesroc/reference/set_jax_python.md)
    or auto-cached during the last fit / install).

2.  The managed venv at `jax_backend_dir()` if it exists.

3.  `Sys.getenv("RETICULATE_PYTHON")` if set and the file exists.

4.  Auto-discovery: scans common Python locations and PATH for an
    install that can `import jax`. (See `find_python_with_jax`.) Returns
    `NULL` if nothing is found – call
    [`install_jax_backend()`](https://jed709.github.io/bayesroc/reference/install_jax_backend.md)
    then.

## Usage

``` r
jax_python()
```

## Value

Character path to a Python executable, or `NULL`.

## See also

[`set_jax_python()`](https://jed709.github.io/bayesroc/reference/set_jax_python.md),
[`check_jax_backend()`](https://jed709.github.io/bayesroc/reference/check_jax_backend.md),
[`install_jax_backend()`](https://jed709.github.io/bayesroc/reference/install_jax_backend.md).
