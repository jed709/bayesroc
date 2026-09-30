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
| Stan (plain) | `R/generate_stan.R`, `R/stan_bvn.R` | Generated Stan; vanilla Stan autodiff. |
| Stan (batch) | `R/generate_batch_cpp.R`, `inst/stan/batch_sdt_core.hpp` | A per-model C++ function with **analytic** gradients via `stan::math::precomputed_gradients`, injected through cmdstanr's `user_header`. ~10-16x faster than plain. |
| JAX | `inst/python/numpyro_backend.py` | Fused per-observation likelihoods with hand-coded `@custom_jvp` analytic gradients; batch-vectorized (no vmap). Bivariate-normal and f64 probit cells run natively through XLA FFI (`inst/ffi/bvn_ffi.cc`, `inst/include/bayesroc/bvn_simd.h`). The native kernel is required: it is built on first use with the C++ toolchain cmdstanr uses, and the JAX backend stops with a fix-it message if it cannot be built. The pure-JAX kernels remain as test references (`BAYESROC_PURE_JAX_REFERENCE=1`). |

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
  the model falls back to plain Stan with a message at compile time. That NULL
  return is the single source of truth for batch support -- there is no separate
  allowlist.
- `backend = "jax"` runs NumPyro in a **subprocess** (`processx`, not reticulate,
  to avoid RStudio DLL conflicts). R writes a JSON config + reads `.npy` results;
  Python discovery is shared by `.python_candidate_paths()` (R/jax_backend.R).

## Shared modeling conventions

- **Mid-anchor thresholds:** the K-1 ordered thresholds are parameterized as a
  middle anchor `thresh_mid` plus ordered positive **gaps** under a `gap_link`
  (`"log"` default, or `"softplus"`). All three generators must build thresholds
  the same way (cumsum from the mid anchor).
- **Bivariate cell probabilities:** every cell is one rectangle probability
  P(a1 < Z1 < b1, a2 < Z2 < b2; rho), never an inclusion-exclusion of CDFs built by
  the caller. The rectangle is Drezner-Wesolowsky Gauss-Legendre in the angle with a
  node count tiered on |rho| (8/10/12/14/16/20 at |rho| edges .75/.85/.9/.925/.97;
  <=1e-6 relative on cells >= 1e-6), corners at infinity skipped, and a tail path for
  cells < 1e-6 (log-space Gauss-Legendre over the log-concave conditional integrand
  around its Newton-located mode; ~1e-9 relative down to exp(-700)). One algorithm,
  four implementations kept in lockstep: `bvn_simd.h` (native SIMD, JAX),
  `bvn_tail.h` (scalar, batch C++ and the native tail), `_dw_*`/`_rect_tail_jax`
  (pure JAX), `R/stan_bvn.R` (Stan language, plain Stan).
- **Univariate cells** are differenced on the tail side (Q(lo) - Q(hi) when lo is
  above the median) for every link, so upper-tail cells never cancel to zero.
- **Probit Phi in JAX** is phi(x) t C(t), t = 1 / (1 + 0.2316419 |x|), returning the value,
  its exact derivative and the tail mass from one exp. f64: degree-11 C, ~1e-9 relative in
  both tails, native (`vprobit_triple`, pool-threaded) with `_probit_triple` as the JAX
  reference. f32: degree-7 C fitted on |x| < 13.5 (beyond it f32 cells underflow), pure JAX;
  measured faster than a native f32 entry. Stan and batch C++ use exact erfc.
- **f32 mode:** bivariate native kernels have f32 entries that widen to double, run the f64
  kernel, and round back (thread-local buffer, flush-to-zero; without those the entry was
  1.2x slower inside NUTS).
- **Open bounds** are +-Inf in Stan and C++ and +-1e10 in JAX (so z * phi(z) terms
  stay 0, not NaN). In Stan, standardise a possibly-infinite bound with `bvn_std()`:
  arithmetic on an infinite var poisons the gradient with 0 * Inf.

## Numerical landmines (don't relearn these the hard way)

- **Probability floors sit at the float underflow limit** (1e-300 in double,
  1e-37 in JAX float32 via `_pfloor()`), not 1e-20. A floor above the smallest
  representable cell makes the log-likelihood flat with zero gradient below it; a
  chain started in that region (e.g. Stan's random inits) can settle in a spurious
  mode that ignores the observation. Where a JVP divides by a clipped probability,
  zero the gradient where the clip is active so value and tangent stay consistent.
- **float32 potentials are summed relative to an offset.** A float32 sum of N log-likelihood
  terms resolves only ~|U| * 6e-8, which is 0.03 at |U| = 5e5 and swamps energy differences.
  In f32 both samplers subtract a per-observation offset (the log-likelihood at a warmed-up
  position; `drghmc.ll_offset`, `_nuts_ll_offset`) inside the sum. The offset travels with
  warm-restart state so rounds continue exactly; f64 passes zeros, which is exact.
- **Stan's `normal_lcdf` has an approximate hand-written derivative** (~1e-5
  relative); use `bvn_log_interval()` instead.
- **Pure-JAX bivariate code must be compiled** (`jit`): the |rho|-tier `lax.switch`
  under eager autodiff builds every branch op by op and can take many GB.
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
| `inst/python/drghmc.py` | DR-G-HMC sampler (`sampler = "drghmc"`), exact warm continuation |
| `inst/include/bayesroc/bvn_simd.h`, `bvn_tail.h`, `inst/ffi/bvn_ffi.cc`, `R/ffi_build.R` | native bivariate-normal kernels, built on the user's machine and cached in `tools::R_user_dir()` |
| `R/stan_bvn.R` | Stan-language mirror of the bivariate-normal kernel |
| `R/loo.R`, `R/npy.R` | streamed PSIS-LOO over the on-disk log-likelihood (`.npy`) |
| `R/autostop.R`, `R/compact.R` | ESS-targeted fitting rounds; compact storage of random-effect draws |
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
