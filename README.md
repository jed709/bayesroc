# bayesroc

Bayesian signal detection models in R, with a formula-based API. Compiles models
to Stan (via [`cmdstanr`](https://mc-stan.org/cmdstanr/)) or [`NumPyro`](https://github.com/pyro-ppl/numpyro)
and supports multiple univariate and bivariate model families with fixed and
random effects on any model parameter.

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
plot_roc_curve(fit)
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

See the [vignette](vignettes/univariate-recognition.Rmd) for an in-depth example.

### Backends and speed

The Stan backend uses a fused likelihood with analytic gradients by default
(`batch_likelihood = TRUE`), which is much faster than plain Stan with autodiff
and needs no extra toolchain beyond `cmdstanr`. Pass `batch_likelihood = FALSE` 
to force the plain path (e.g. for cross-checking). The Stan backend is single-threaded
by default but supports multithreading via `reduce_sum` by passing `threads = TRUE` 
to `broc()` and specifying `threads_per_chain = N_threads` when calling `fit_broc()`.

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
- `?plot_roc_curve` — ROC curve plot
- `?conditional_smooths` — smooth-term evaluation/plotting (mgcv `s()` / `t2()`
  are supported in any parameter formula)
- `?broc_prior` — prior specification

## License

MIT. See `LICENSE.md`.
