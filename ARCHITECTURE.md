# bayesroc architecture

Developer/maintainer notes. Not shipped to users (build-ignored). The single
most important thing to understand before editing the likelihood or gradient
code is the **three-generator design** and the **parity contract** (below).

## Pipeline

```
brf(...)            compose a multi-parameter formula spec  (R/brf.R)
   |
broc(spec, data)    parse formula -> build design matrices -> generate backend code
   |                (R/parse_formula.R, R/design_matrix.R, R/broc_model.R,
   |                 R/generate_stan.R, R/generate_batch_cpp.R)
   v
broc_model          carries: parsed spec, model_data (Stan data), generated Stan
   |                code, optional batch C++, priors, family
fit_broc(model)     compile + sample via Stan (cmdstanr) or JAX (NumPyro subprocess)
   |                (R/broc_model.R, R/jax_backend.R)
   v
broc_fit            S3 wrapper (environment) over the backend result
   |                (R/broc_fit.R)
   v
summary / predict / plot / pp_check / loo / posterior_*   (R/summary.R, predict.R, ...)
```

## The three-generator design (read this)

The same SDT **likelihood + gradient algebra** for each family is implemented
**three independent times**, because the package supports three compute paths
and you cannot share a hand-coded JAX JVP with emitted C++ or Stan strings:

| Path | File | What it emits / does |
|---|---|---|
| Stan (plain) | `R/generate_stan.R` (~4.4k lines) | Generated Stan; vanilla Stan autodiff. |
| Stan (batch) | `R/generate_batch_cpp.R` (~6.4k lines) | A per-model C++ function with **analytic** gradients via `stan::math::precomputed_gradients`, injected through cmdstanr's `user_header`. ~10-16x faster than plain. |
| JAX | `inst/python/numpyro_backend.py` (~4.8k lines) | Fused per-observation likelihoods with hand-coded `@custom_jvp` analytic gradients; batch-vectorized (no vmap). |

Plain Stan and the batch path share **one** generator (`generate_stan_code_v2`);
`batch_info` swaps only the likelihood line in the model block. The JAX backend
is wholly separate (different language + transport).

**Consequence:** any change to the likelihood, the gradient math, the threshold
parameterization, or the probability-floor strategy must be made in ~8-12 places
across these three generators and kept **bit-compatible by hand**. This is the
largest maintainability cost in the package and is intrinsic to the 3-backend goal.

## The parity contract

All three paths must produce **identical** posteriors for the same model. This
is enforced by tests, which are the guardrail for any likelihood/gradient edit:

- `tests/testthat/test-stan_jax_agreement.R` — Stan(batch) vs JAX posterior means.
- `tests/testthat/test-batch_parity.R` — Stan(batch) vs Stan(plain). Plain Stan is
  the **independent autodiff oracle**: batch-C++ and JAX are both *hand-derived*
  gradients, so plain Stan (vanilla autodiff) is the one independent check that
  catches a shared derivation bug. **Do not remove plain Stan.**

When you touch likelihood/gradient code in one generator, change all three and
run the full suite.

## Backends and `batch_likelihood`

- `broc(..., batch_likelihood = TRUE)` is the **default**. It builds the batch
  C++ (no extra toolchain beyond cmdstanr -- it uses `user_header`, not FFI).
- `generate_batch_cpp()` returns `NULL` for any family/config it can't emit, and
  the model **silently falls back to plain Stan**. That NULL return is the single
  source of truth for batch support -- there is no separate allowlist.
- `backend = "jax"` runs NumPyro in a **subprocess** (`processx`, not reticulate,
  to avoid RStudio DLL conflicts). R writes a JSON config + reads `.npy` results;
  Python discovery is shared by `.python_candidate_paths()` (R/jax_backend.R).

## Shared modeling conventions

- **Mid-anchor thresholds:** the K-1 ordered thresholds are parameterized as a
  middle anchor `thresh_mid` plus ordered positive **gaps** under a `gap_link`
  (`"log"` default, or `"softplus"`). All three generators must build thresholds
  the same way (cumsum from the mid anchor).
- **Bivariate cell probabilities:** analytic via Owen's-T binormal CDF
  (`inst/python/owens_t_jax.py`, and the Stan/C++ equivalents) and GL10 rectangle
  integrals where needed.

## Numerical landmines (don't relearn these the hard way)

- **Clip-floor JVP guard:** a fused JVP whose primal returns `log(clip(prob, 1e-20))`
  is *flat* where prob underflows, so the true gradient there is 0. If the JVP
  multiplies a finite analytic `dp` by `inv_prob = 1/clip(prob, 1e-20)` it
  produces ~1e20 gradients that silently freeze NUTS (0 divergences, but ESS
  collapses). Guard every such site: `jnp.where(prob > 1e-15, 1/clip(prob, 1e-20), 0)`.
  Primals that use `log(prob)` *without* a clip are fine -- `1/prob` is the
  consistent gradient there; do not add a guard.
- **Owen's-T tail stability / inverse-Mills:** SDT JVPs must use tail-stable forms;
  avoid `phi * exp(-logcdf)` patterns that evaluate to `0 * inf`.

## File map (orientation)

| File | Role |
|---|---|
| `R/brf.R`, `R/parse_formula.R` | formula composition + parsing |
| `R/design_matrix.R` | fixed/random/threshold design matrices |
| `R/families.R` | family definitions, parameter specs, the per-family JVP emit |
| `R/generate_stan.R` | Stan codegen (plain + batch-likelihood line) |
| `R/generate_batch_cpp.R` | per-model batch C++ (analytic gradients) |
| `inst/stan/batch_sdt_core.hpp` | batch C++ support header |
| `inst/python/numpyro_backend.py` | JAX fused custom-JVP likelihoods |
| `R/jax_backend.R`, `R/install_jax.R` | JAX subprocess + Python discovery/venv mgmt |
| `R/broc_model.R` | `broc()` + `fit_broc()` orchestration |
| `R/broc_fit.R` | `broc_fit` S3 object + save/load |
| `R/summary.R`, `R/predict.R`, `R/pp_check.R`, `R/posterior.R`, `R/plot.R` | post-processing |
| `R/priors.R`, `R/prior_init.R` | prior specification + defaults |

## Things deliberately NOT done (and why)

- **No symbolic IR unifying the three generators.** It would be the "right" DRY
  fix but means rewriting the validated gradients; the risk/ROI is poor. The
  triplication is documented here instead.
- The god-functions (`sdt_model`, `build_model_data`, `generate_functions_block_v2`)
  are long but are codegen orchestrators; left as-is.
