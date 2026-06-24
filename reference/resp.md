# resp() – Multi-response marker for bivariate/CDP models

Used inside
[`brf()`](https://jed709.github.io/bayesroc/reference/brf.md) formulas
to specify multiple response variables. The arguments are positional:
the specified family determines whether the second variable is a source
response (bivariate families) or R/K variable (CDP family).

## Usage

``` r
resp(response1, response2)
```

## Arguments

- response1:

  The primary response variable (e.g., detection confidence)

- response2:

  The second response variable (source confidence for bivariate
  families, R/K for CDP families)

## Value

Not called directly: this is a marker parsed from the (unevaluated)
[`brf()`](https://jed709.github.io/bayesroc/reference/brf.md) formula.
Invoking it directly raises an error.

## Examples

``` r
# Bivariate: resp(det_conf, src_conf) | type ~ ...
# CDP:       resp(conf, rk_var) | old ~ ...
```
