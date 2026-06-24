# Compile and Fit a Bayesian ROC Model

Takes a `broc_model` object (from
[`broc()`](https://jed709.github.io/bayesroc/reference/broc.md)) and
runs MCMC sampling via Stan (cmdstanr) or JAX (NumPyro). Returns a
`broc_fit` object that wraps the backend-specific result and provides a
uniform interface for downstream analysis (`summary(fit)`,
`predict(fit)`, `pp_check(fit)`, etc.).

## Usage

``` r
fit_broc(
  model,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 2000,
  iter_sampling = 2000,
  thin = 1,
  adapt_delta = 0.8,
  max_treedepth = 10,
  init = 0,
  refresh = 100,
  threads_per_chain = NULL,
  backend = c("stan", "jax"),
  python = NULL,
  progress_bar = TRUE,
  seed = NULL,
  file = NULL,
  file_refit = FALSE,
  save_loglik = TRUE,
  num_warmup = NULL,
  num_samples = NULL,
  num_chains = NULL,
  ...
)
```

## Arguments

- model:

  A `broc_model` object created by
  [`broc()`](https://jed709.github.io/bayesroc/reference/broc.md).

- chains:

  Number of MCMC chains (default 4).

- parallel_chains:

  Number of chains to run in parallel (default 4). For the JAX backend
  this is all-or-nothing: chains run in parallel across devices when
  `parallel_chains >= chains`, and sequentially on one device when
  `parallel_chains < chains`.

- iter_warmup:

  Number of warmup/adaptation iterations per chain (default 2000).

- iter_sampling:

  Number of post-warmup sampling iterations per chain (default 2000).

- thin:

  Period for saving samples (default 1). `thin = k` keeps every `k`-th
  post-warmup draw.

- adapt_delta:

  Target acceptance probability for NUTS (default 0.8).

- max_treedepth:

  Maximum tree depth for NUTS (default 10).

- init:

  Initial-value strategy. Default `0`: all parameters start at 0 on the
  unconstrained scale. Other options: `"prior"` samples each parameter
  from its prior; `"random"` uses Stan's U(-2, 2) on the unconstrained
  scale (can be unstable for log-linked parameters); a single number `x`
  sets a U(-x, x) initialization radius; or a function or list of
  per-chain init lists.

- refresh:

  How often the Stan backend prints progress, in iterations (default
  100). Stan only; the JAX progress bar has no refresh-rate control.

- threads_per_chain:

  Number of threads per chain for within-chain parallelism via
  `reduce_sum`. Only used when the model was built with
  `threads = TRUE`.

- backend:

  Inference backend: `"stan"` (default, via cmdstanr) or `"jax"` (via
  NumPyro). The JAX backend requires a configured Python environment
  (see
  [`install_jax_backend()`](https://jed709.github.io/bayesroc/reference/install_jax_backend.md)).

- python:

  Optional path to the Python executable for the JAX backend. `NULL`
  (default) uses the managed environment (see
  [`install_jax_backend()`](https://jed709.github.io/bayesroc/reference/install_jax_backend.md)).

- progress_bar:

  If `TRUE` (default), show the sampler progress bar.

- seed:

  Optional integer seed for reproducible sampling. `NULL` (default) uses
  a random seed.

- file:

  Optional path to save the fitted model to. When set, the fit is
  written there with
  [`save_broc_fit()`](https://jed709.github.io/bayesroc/reference/save_broc_fit.md)
  once sampling finishes (a `.rds` extension is appended if absent). If
  the file already exists, the saved fit is loaded and returned instead
  of refitting – unless `file_refit = TRUE`. `NULL` (default) does not
  save.

- file_refit:

  If `FALSE` (default), an existing `file` is loaded and sampling is
  skipped; if `TRUE`, the model is refit and the file overwritten.
  Ignored when `file` is `NULL`.

- save_loglik:

  If `TRUE` (default), the per-observation log-likelihood is kept so the
  saved fit supports
  [`loo()`](https://mc-stan.org/loo/reference/loo.html). If `FALSE`,
  `log_lik` is dropped when the fit is saved.

- num_warmup, num_samples, num_chains:

  NumPyro-style aliases for `iter_warmup`, `iter_sampling`, and
  `chains`. When supplied, each overrides its canonical counterpart.

- ...:

  Additional arguments forwarded to cmdstanr's `$sample()` (Stan backend
  only; a warning is issued if any are passed with `backend = "jax"`).

## Value

A `broc_fit` object. It carries low-level accessors `$draws()`,
`$summary()`, `$diagnostic_summary()`, `$loo()`, `$num_chains()`,
`$metadata()`, and `$save_object()`, but S3 methods
[summary()](https://jed709.github.io/bayesroc/reference/summary.broc_fit.md),
[plot()](https://jed709.github.io/bayesroc/reference/plot.broc_fit.md),
[predict()](https://jed709.github.io/bayesroc/reference/predict.broc_fit.md),
[pp_check()](https://jed709.github.io/bayesroc/reference/pp_check.broc_fit.md),
and [loo()](https://jed709.github.io/bayesroc/reference/loo.broc_fit.md)
are also provided.

## See also

[`broc()`](https://jed709.github.io/bayesroc/reference/broc.md) for
model specification, `summary(fit)` for results.

## Examples

``` r
if (FALSE) { # \dontrun{
model <- broc(brf(conf | old ~ cond + (1|subj),
                  sigma ~ 1), data = dat, family = uvsd())
fit <- fit_broc(model)
fit <- fit_broc(model, backend = "jax", chains = 2)

# Save on first run; reload instead of refitting on later runs.
fit <- fit_broc(model, file = "my_fit")          # writes my_fit.rds
fit <- fit_broc(model, file = "my_fit")          # loads my_fit.rds
fit <- fit_broc(model, file = "my_fit", file_refit = TRUE)      # refits
} # }
```
