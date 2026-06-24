# Set a Prior on an SDT Model Parameter

Specify a prior for one class of parameters in a
[`broc()`](https://jed709.github.io/bayesroc/reference/broc.md) model: a
fixed effect, a random-effect standard deviation, or a random-effect
correlation. Pass one or more `broc_prior()` calls to
`broc(..., priors = ...)`; combine several with
[`c()`](https://rdrr.io/r/base/c.html).

## Usage

``` r
broc_prior(prior, class, dpar = NULL, coef = NULL, group = NULL)
```

## Arguments

- prior:

  The prior as a distribution call, e.g. `"normal(0, 1)"`,
  `"student_t(3, 0, 2.5)"`, or `"lkj(2)"`. Supported on both backends:
  `normal`, `std_normal`, `student_t`, `cauchy`, `logistic`, `gumbel`,
  `double_exponential` (= `laplace`), `uniform`, `beta`, `exponential`,
  `gamma`, `inv_gamma`, `weibull`, `lognormal`, the half forms
  `half_normal`, `half_cauchy`, `half_student_t`, and `lkj`
  (`class = "cor"` only). For SD priors (`class = "sd"`/`"sds"`) a
  symmetric distribution is automatically restricted to positive values.
  The Stan backend additionally accepts any distribution in the Stan
  language.

- class:

  Parameter class: `"b"` (fixed effects), `"sd"` (random-effect SDs),
  `"cor"` (random-effect correlations), or `"sds"` (smooth penalty SDs,
  only for models with `s()`/`t2()` smooth terms).

- dpar:

  Distributional parameter the prior applies to, e.g. `"dprime"`,
  `"sigma"`, `"lambda"`. For criterion priors use `"thresh_mid"` /
  `"log_gaps"` (`"thresh_mid2"` / `"log_gaps2"` for the second response
  axis in bivariate and CDP families). Family-specific aliases (`mu`,
  `mu1`, `mu2`, `rec`, `fam`, `sigma_R`, `sigma_F`) are also accepted.
  `NULL` applies the prior to every parameter in `class`.

- coef:

  Optional specific coefficient name (e.g., `"conditionread"`)

- group:

  Optional grouping factor for random effects (e.g., `"participant"`)

## Value

A `broc_prior` object. Combine several with
[`c()`](https://rdrr.io/r/base/c.html) and pass to
[`broc()`](https://jed709.github.io/bayesroc/reference/broc.md).

## Details

Priors are matched from most to least specific: a `coef`-level prior
overrides a `group`-level one, which overrides a `dpar`-level one, which
overrides the global class default. Call
[`get_broc_prior()`](https://jed709.github.io/bayesroc/reference/get_broc_prior.md)
on a [`broc()`](https://jed709.github.io/bayesroc/reference/broc.md) or
[`brf()`](https://jed709.github.io/bayesroc/reference/brf.md) object
(with data) to see which coefficients exist and the current prior for
each.

Confidence thresholds are not set directly. They are reparameterized as
a central anchor (`thresh_mid`) plus strictly-positive, log-scale
spacings (`log_gaps`) between successive thresholds, which enforces
ordering. Set criterion priors via `dpar = "thresh_mid"` (overall
criterion location) and `dpar = "log_gaps"` (threshold spread).

## See also

[`get_broc_prior()`](https://jed709.github.io/bayesroc/reference/get_broc_prior.md)

## Examples

``` r
# Prior on all d' fixed effects
broc_prior("normal(1, 1)", class = "b", dpar = "dprime")
#> Prior: normal(1, 1) 
#>   class: b, dpar: dprime

# Prior on specific d' coefficient
broc_prior("normal(2, 0.5)", class = "b", dpar = "dprime", coef = "conditionread")
#> Prior: normal(2, 0.5) 
#>   class: b, dpar: dprime, coef: conditionread

# Prior on ALL random effect standard deviations
broc_prior("normal(0, 0.5)", class = "sd")
#> Prior: normal(0, 0.5) 
#>   class: sd

# Prior on random effect SDs for d' only
broc_prior("normal(0, 0.5)", class = "sd", dpar = "dprime")
#> Prior: normal(0, 0.5) 
#>   class: sd, dpar: dprime

# Prior on random effect SDs for d' for participant group only
broc_prior("normal(0, 0.3)", class = "sd", dpar = "dprime", group = "participant")
#> Prior: normal(0, 0.3) 
#>   class: sd, dpar: dprime, group: participant

# Prior on ALL correlations
broc_prior("lkj(2)", class = "cor")
#> Prior: lkj(2) 
#>   class: cor

# Prior on criterion correlations for participant
broc_prior("lkj(4)", class = "cor", dpar = "criterion", group = "participant")
#> Prior: lkj(4) 
#>   class: cor, dpar: criterion, group: participant
```
