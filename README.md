# bayesroc

Bayesian signal detection models in R, with a formula-based API. Compiles models
to Stan (via [`cmdstanr`](https://mc-stan.org/cmdstanr/)) or [`NumPyro`](https://github.com/pyro-ppl/numpyro)
and supports multiple univariate and bivariate model families with fixed and
random effects on any model parameter.

Docs and the getting-started vignette: <https://jed709.github.io/bayesroc/>

## Installation

bayesroc is not yet on CRAN. Install from GitHub:

```r
# install.packages("remotes")
remotes::install_github("jed709/bayesroc")
```

The Stan backend requires `cmdstanr` and a working CmdStan installation:

```r
install.packages("cmdstanr",
  repos = c("https://stan-dev.r-universe.dev", getOption("repos")))
cmdstanr::install_cmdstan()
```

The optional JAX backend runs in a managed Python virtual environment. It
requires a **system Python (>= 3.10) installed first** — bayesroc creates the
venv but does not install Python itself. On macOS/Linux a `python.org`,
Homebrew (`brew install python`), pyenv, or conda interpreter all work and are
auto-discovered. On Windows, install Python from
[python.org](https://www.python.org/downloads/) rather than the Microsoft Store
(the Store "python" stub is not a usable interpreter). Then:

```r
bayesroc::install_jax_backend()   # creates a managed venv with jax + numpyro
bayesroc::check_jax_backend()     # confirms the backend is ready ("ok")
```

Once installed, switch backends with `fit_broc(model, backend = "jax")`.

If `install_jax_backend()` cannot find Python, point it at one explicitly with
`install_jax_backend(python = "/path/to/python")`, or set the interpreter used
for fitting with `set_jax_python()`. See `?install_jax_backend` and
`?check_jax_backend` for troubleshooting.

## A simple example

Formulas provided to univariate bayesroc models must supply both a response variable 
and a binary indicator variable using either 0/1 or -0.5/0.5 coding that denotes 
whether the trial is a lure or a target. These variables should be provided in the form `response | is_old ~ predictors`, where this formula implictly applies to d' and subsequent formulae are provided for other model parameters (e.g., `criterion ~ predictors`, `sigma ~ predictors`). 

The code below shows how a hierarchical unequal variance signal detection model (UVSD) 
can be fit to the bundled `prm09` dataset (97 subjects, 480 items, 6-point confidence;
Pratte et al., 2010), with correlated subject- and item-level random intercepts on 
every parameter that are cross-correlated between d' and sigma. We then view the model
summary and plot the model-implied ROC curve. Because the dataset is very large, we fit 
this basic example to the first 20 participants. Here, `resp` is the rating response on 
a 6-point scale (with higher responses indicating "sure old") and `cond` is the 0/1 coded 
indicator variable. 

```r
library(bayesroc)
data(prm09)

first20 <- prm09[prm09$sub<20,]

m <- broc(brf(resp | cond ~ 1 + (1|s|sub) + (1|i|item),
              criterion ~ 1 + (1|sub) + (1|item),
              sigma ~ 1 + (1|s|sub) + (1|i|item),
              family = uvsd()),
          data = first20)
fit <- fit_broc(m, backend = "stan")

summary(fit)
plot(fit)          # parameter estimates on their natural scale
plot_roc(fit)      # model-implied ROC with observed operating points
```

If we are not interested in modeling trial-level data (e.g., random effects on items), 
we can aggregate the data to per-subject counts and fit a quick version of the model.

```r
agg <- as.data.frame(table(first20[c("sub", "cond", "resp")]), 
                     responseName = "count")

m <- broc(brf(resp | cond ~ 1 + (1|s|sub),
              criterion ~ 1 + (1|sub),
              sigma ~ 1 + (1|s|sub),
              family = uvsd()),
              counts = "count",
          data = agg)
fit <- fit_broc(m, backend = "stan")
```

See the [vignette](https://jed709.github.io/bayesroc/articles/univariate-recognition.html) for an in-depth example.

### Backends and speed

The Stan backend uses a fused C++ likelihood with analytic gradients, which is
much faster than Stan-language autodiff and needs no extra toolchain beyond
`cmdstanr`. The Stan backend is single-threaded
by default; `fit_broc(threads_per_chain = N)` with N > 1 compiles a `reduce_sum`
version of the model and runs each chain on N threads.

The NumPyro backend (`fit_broc(m, backend = "jax")`) is generally faster than Stan and 
uses analytic custom JVPs. This backend only supports multithreading and the number
of threads spawned for each chain depends on the model family, fixed- and random-
effects structure, and dataset size. Threads per chain cannot be set directly. At 
present, the NumPyro backend is CPU-only, but GPU support may be added in future releases.

## Supported families

| Family                | Description                                                            |
|-----------------------|------------------------------------------------------------------------|
| `evsd()`             | Equal variance signal detection model                                                     |
| `uvsd()`             | Unequal variance signal detection model                                                  |
| `dpsd()`             | Dual process signal detection model (Yonelinas, 1994)                                           |
| `mixture()`           | Mixture signal detection model (with optional lure mixture; DeCarlo, 2002)                                    |
| `source_mixture()`    | Source-discrimination mixture (DeCarlo, 2003a)                          |
| `bivariate_gaussian()`     | Bivariate Gaussian signal detection model (DeCarlo, 2003b; Starns et al., 2014)                                   |
| `bivariate_dp()`      | Bivariate dual process signal detection model (Starns et al., 2014)                                                |
| `vrdp2d()`            | 2D variable recollection dual process model (Onyper et al., 2010)  |
| `cdp()`               | Continuous dual process model for Remember/Know (Wixted & Mickes, 2010)      |
| `cumulative()`        | Plain cumulative ordinal regression                                    |
| `bivariate_cumulative()` | Bivariate cumulative ordinal regression |

## Bundled datasets

bayesroc ships five datasets:

| Dataset | Format | Use |
|---|---|---|
| `prm09` | trial-level, 46,495 × 5 (97 subjects, 480 items, 6-point confidence) | Pratte et al. (2010) — hierarchical univariate models |
| `whitridge_2024` | trial-level, 7,560 × 6 (42 subjects, 360 items, 6-point confidence) | Whitridge et al. (2024) — hierarchical univariate models with within-subject effects |
| `yonelinas_1999` | aggregated counts, 108 × 4 (3 item types × 6×6 detect/source ratings) | Yonelinas (1999) — item/source judgements for bivariate SDT models (`bivariate_gaussian`, `bivariate_dp`, `vrdp2d`) |
| `hilford_2002` | aggregated counts, 12 × 3 (2 sources × 6 ratings) | Hilford et al. (2002) — `source_mixture()` models |
| `rotello_2005` | aggregated counts, 44 × 5 | Rotello et al. (2005) — Remember/Know data for `cdp()` |

```r
data(prm09)
data(whitridge_2024)
data(yonelinas_1999)
data(hilford_2002)
data(rotello_2005)
```

See the corresponding help pages (e.g. `?yonelinas_1999`, `?hilford_2002`) for column descriptions.

## Documentation

- `?broc` — main entry point
- `?brf` — formula composition
- `?fit_broc` — fitting (Stan or JAX)
- `summary(fit)`, `plot(fit)`, `predict(fit)`, `pp_check(fit)`, `loo(fit)`,
  `posterior_epred(fit)`, `posterior_linpred(fit)` — S3 methods on the fitted model
- Plots (all return ggplot objects and share `by`, `ndraws`, `prob`, `re_formula`,
  `seed`): `?plot_estimates` (also `plot(fit)`), `?plot_roc` (ROC and zROC),
  `?plot_distributions` (latent distributions with criteria), `?pp_check.broc_fit`,
  `?plot_random_effects`, `?plot_smooths`; `?theme_broc` for the shared theme and palette
- `?conditional_smooths` — smooth-term evaluation (mgcv `s()` / `t2()`
  are supported in any parameter formula). If a model with smooth terms reports
  divergences, `broc(..., smooth_info = TRUE)` samples the population-level smooth
  coefficients on a data-informed scale; the posterior is unchanged
- `?broc_prior` — prior specification
- `?autostop_broc` — sample until ESS and R-hat targets are met (JAX backend)
- `?drop_loglik` — reclaim a fit's log-likelihood sidecar

## Log-likelihood and LOO

`log_lik` is written to an `.npy` sidecar rather than held in the draws array,
and `loo()` streams it in column chunks sized to available RAM. Both backends
write the same sidecar format and `loo()` treats them the same way. Stan's values
come from its CSV output, so they carry CmdStan's output precision (`sig_figs`).

```r
fit <- fit_broc(model, loo = TRUE)   # LOO at fit time, sidecar reclaimed
fit$.loo

fit <- fit_broc(model)               # sidecar kept for the session
loo(fit, cores = 4)
fit <- drop_loglik(fit)              # reclaim when finished
```

For aggregated (`counts =`) fits, `loo()` defaults to leave-one-trial-out;
`type = "cell"` gives the leave-one-row-out partition that brms `weights()`
produces. The two are not comparable via `loo_compare()`.

`save_broc_fit()` (and `fit_broc(file = )`) writes the sidecar next to the saved
fit as `<file>_loglik.npy` whenever the fit kept its log-likelihood, so `loo()`
works after `load_broc_fit()`; pass `include_loglik = FALSE` to skip it.

## Sampling until convergence and precision targets are met

`autostop_broc()` runs one set of chains and extends them in chunks until every
monitored parameter has rank-normalised bulk and tail ESS of at least
`ess_target` and R-hat of at most `rhat_target` (defaults 400 and 1.01, following
Vehtari et al., 2021), or a cap on draws or time is reached. Chains continue
exactly where the previous chunk stopped, so the result is identical to one run of
the final length. Targets can differ by parameter class:

```r
fit <- autostop_broc(model,
                     ess_target = c(pop = 2000, sd = 400, cor = 200),
                     max_draws = 50000, max_time = 3600)
attr(fit, "stop_reason")   # "targets", "max_draws" or "max_time"
attr(fit, "ess_trace")     # one row per check
fit <- autostop_broc(model, resume = fit, max_draws = 100000)  # continue a capped run
```

`autostop_broc()` is JAX-only (exact continuation needs sampler state that
cmdstanr cannot return). A time cap makes the stopping point machine-dependent;
the draws themselves are not affected.

## License

MIT. See `LICENSE.md`.
