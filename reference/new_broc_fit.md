# SDT Fit Wrapper Class

Backend-agnostic S3 wrapper around CmdStanMCMC (Stan) or NumPyro fit
objects. Uses environment-based S3 class so \$draws(), \$summary(), etc.
work like CmdStanMCMC's R6 interface. Create an broc_fit object

## Usage

``` r
new_broc_fit(
  backend_fit,
  model,
  backend = "stan",
  num_chains = 4L,
  iter_sampling = 1000L,
  elapsed = NULL,
  draws_array = NULL,
  max_treedepth = 10L,
  save_loglik = TRUE,
  diagnostics = NULL
)
```

## Arguments

- backend_fit:

  Raw backend object (CmdStanMCMC for Stan, list for JAX)

- model:

  broc_model object

- backend:

  "stan" or "jax"

- num_chains:

  Number of chains used

- iter_sampling:

  Number of post-warmup iterations per chain

- elapsed:

  Elapsed time in seconds (numeric)

- draws_array:

  For JAX: pre-converted posterior::draws_array (NULL for Stan)

- max_treedepth:

  Maximum NUTS tree depth used in the fit (default 10).

## Value

An broc_fit object (environment with S3 class)
