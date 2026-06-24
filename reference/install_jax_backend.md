# Install the JAX/NumPyro backend into a managed venv

Creates an isolated Python virtual environment under `jax_backend_dir()`
and installs `jax` and `numpyro` at known-compatible pinned versions.
This never touches the user's system or global Python install. The
cmdstanr analogue is
[`cmdstanr::install_cmdstan()`](https://mc-stan.org/cmdstanr/reference/install_cmdstan.html).

## Usage

``` r
install_jax_backend(
  dir = jax_backend_dir(),
  python = NULL,
  jax_version = JAX_VERSION_PIN,
  numpyro_version = NUMPYRO_VERSION_PIN,
  overwrite = FALSE,
  quiet = FALSE
)
```

## Arguments

- dir:

  Directory to install into. Default: `jax_backend_dir()`.

- python:

  Path to the Python executable used to *create* the venv. Default:
  auto-detect from PATH (must be Python 3.10+). The venv itself uses its
  own bundled interpreter regardless of this choice.

- jax_version, numpyro_version:

  Versions to pin. Defaults to `JAX_VERSION_PIN` / `NUMPYRO_VERSION_PIN`
  constants in the package (the latest mutually-compatible pair as of
  release).

- overwrite:

  If `TRUE`, recreate the venv even if it already exists. Default
  `FALSE` errors if a managed venv is already present.

- quiet:

  If `TRUE`, suppress pip output.

## Value

Invisibly returns the path to the installed venv's Python.

## Details

Subsequent JAX-backend fits via `fit_broc(backend = "jax")` will prefer
this managed venv automatically; no further configuration is needed.

## See also

[`check_jax_backend()`](https://jed709.github.io/bayesroc/reference/check_jax_backend.md),
[`jax_python()`](https://jed709.github.io/bayesroc/reference/jax_python.md),
[`set_jax_python()`](https://jed709.github.io/bayesroc/reference/set_jax_python.md).
