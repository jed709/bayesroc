# Preview the Priors for an SDT Model

Show every prior a model will use as a data frame. Accepts either a
built [`broc()`](https://jed709.github.io/bayesroc/reference/broc.md)
model or a [`brf()`](https://jed709.github.io/bayesroc/reference/brf.md)
object that specifies a formula, data, and a family.

## Usage

``` r
get_broc_prior(object, data = NULL, family = NULL, user_priors = NULL, ...)
```

## Arguments

- object:

  A [`broc()`](https://jed709.github.io/bayesroc/reference/broc.md)
  model, or a
  [`brf()`](https://jed709.github.io/bayesroc/reference/brf.md)
  specification / bare formula. When a formula or `brf`, the model is
  built internally from `data`, `family`, and any `...`.

- data:

  Data frame. Required when `object` is a formula or `brf`, because
  coefficient names and the number of thresholds are read from it.
  Ignored when `object` is already a model.

- family:

  Optional family (e.g.
  [`evsd()`](https://jed709.github.io/bayesroc/reference/evsd.md),
  [`uvsd()`](https://jed709.github.io/bayesroc/reference/uvsd.md)) for
  the formula/`brf` path; defaults to the family set in the `brf`, or
  [`evsd()`](https://jed709.github.io/bayesroc/reference/evsd.md).
  Ignored when `object` is already a model.

- user_priors:

  Optional priors from
  [`broc_prior()`](https://jed709.github.io/bayesroc/reference/broc_prior.md)
  to override defaults.

- ...:

  Further arguments forwarded to
  [`broc()`](https://jed709.github.io/bayesroc/reference/broc.md) on the
  formula/`brf` path (e.g. `encoding_vars`, `counts`, `cor_threshold`,
  `gap_link`).

## Value

A data frame with columns `class`, `dpar`, `coef`, `group`, `prior`, and
`source`, where `source` flags whether each row is a package default
(`"default"`) or was overridden by `user_priors` (`"user"`).

## See also

[`broc_prior()`](https://jed709.github.io/bayesroc/reference/broc_prior.md)

## Examples

``` r
if (FALSE) { # \dontrun{
# Straight from a formula -- no need to build the model first
get_broc_prior(conf | old ~ condition + (1 | subject),
               data = my_data, family = evsd())

# From a brf() specification (family carried on the brf)
spec <- brf(conf | old ~ condition + (1 | subject),
            criterion ~ 1 + (1 | subject), family = uvsd())
get_broc_prior(spec, data = my_data)

# From an already-built model, with custom overrides
model <- broc(spec, data = my_data)
my_priors <- c(
  broc_prior("normal(2, 0.5)", class = "b", dpar = "dprime"),
  broc_prior("normal(0, 1)", class = "sd")
)
get_broc_prior(model, user_priors = my_priors)
} # }
```
