# Plot Model Parameters

Create forest plots of parameter estimates with credible intervals from
fitted models, using ggdist.

## Usage

``` r
# S3 method for class 'broc_fit'
plot(
  x,
  class = "b",
  dpar = "dprime",
  group = NULL,
  type = c("interval", "halfeye", "gradient", "dots"),
  prob = 0.95,
  prob_inner = 0.5,
  ...
)
```

## Arguments

- x:

  A `broc_fit` object from
  [`fit_broc()`](https://jed709.github.io/bayesroc/reference/fit_broc.md).

- class:

  Parameter class to plot: "b" (fixed effects), "sd" (random effect
  SDs), or "cor" (correlations)

- dpar:

  Character vector of distributional parameters to plot (default
  `"dprime"`). The valid set depends on the family (see the family
  constructor).

- group:

  Optional: filter to specific grouping factor(s) for sd/cor plots

- type:

  Plot type: "interval" (default, point + interval), "halfeye"
  (density + interval), "gradient" (gradient interval), "dots" (quantile
  dot plot)

- prob:

  Probability mass for outer credible interval (default 0.95)

- prob_inner:

  Probability mass for inner credible interval (default 0.5)

- ...:

  Ignored.

## Value

A ggplot object

## Examples

``` r
if (FALSE) { # \dontrun{
# Plot d' fixed effects with default interval style
plot(fit, class = "b", dpar = "dprime")

# Plot with halfeye (density + interval)
plot(fit, class = "b", dpar = "dprime", type = "halfeye")

# Plot multiple fixed effects, faceted by dpar
plot(fit, class = "b", dpar = c("dprime", "sigma"))

# Plot random effect SDs
plot(fit, class = "sd", dpar = "dprime")

# Plot correlations with gradient intervals
plot(fit, class = "cor", dpar = "dprime", type = "gradient")
} # }
```
