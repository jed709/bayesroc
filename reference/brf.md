# Compose a Multi-Parameter SDT Model Formula

Formula specification for bayesroc models. The first argument is the
formula for the primary parameter (d' / mu); each additional formula's
left-hand side names the parameter it applies to (e.g. `criterion`,
`sigma`, `lambda`).

## Usage

``` r
brf(
  formula,
  ...,
  family = NULL,
  encoding_vars = NULL,
  counts = NULL,
  cor_threshold = NULL,
  threads = NULL,
  gap_link = NULL
)
```

## Arguments

- formula:

  Two-sided formula for the primary parameter (d' / mu),
  `response | is_old ~ predictors`. `is_old` may be coded 0/1 (new/old)
  or centered (-0.5/0.5); centered coding is not supported with lure
  mixtures or multi-response families. For multi-response families, use
  [`resp()`](https://jed709.github.io/bayesroc/reference/resp.md) on the
  left-hand side.

- ...:

  Additional one-sided formulas, one per parameter, with the parameter
  name on the left. E.g. `criterion ~ 1 + (1 | subj)`, `sigma ~ cond`,
  `lambda ~ 1`.

- family:

  The SDT model family: one of
  [`evsd()`](https://jed709.github.io/bayesroc/reference/evsd.md),
  [`uvsd()`](https://jed709.github.io/bayesroc/reference/uvsd.md),
  [`dpsd()`](https://jed709.github.io/bayesroc/reference/dpsd.md),
  [`mixture()`](https://jed709.github.io/bayesroc/reference/mixture.md),
  [`source_mixture()`](https://jed709.github.io/bayesroc/reference/source_mixture.md),
  [`bivariate_gaussian()`](https://jed709.github.io/bayesroc/reference/bivariate_gaussian.md),
  [`bivariate_dp()`](https://jed709.github.io/bayesroc/reference/bivariate_dp.md),
  [`vrdp2d()`](https://jed709.github.io/bayesroc/reference/vrdp2d.md),
  [`cdp()`](https://jed709.github.io/bayesroc/reference/cdp.md),
  [`cumulative()`](https://jed709.github.io/bayesroc/reference/cumulative.md),
  or
  [`bivariate_cumulative()`](https://jed709.github.io/bayesroc/reference/bivariate_cumulative.md).
  May also be set in `brf()`; a value in broc() takes precedence.

- encoding_vars:

  Character vector of column names to treat as encoding-phase (study)
  manipulations for which lure items do not have a meaningful value.
  Listing a column here auto-wraps it in `encoding()`. A value passed to
  [`broc()`](https://jed709.github.io/bayesroc/reference/broc.md) takes
  precedence over one set here.

- counts:

  Name of a column giving per-row trial counts, for aggregated (count)
  data rather than one row per trial. May also be set in `brf()`.

- cor_threshold:

  If `TRUE` (default), random effects on the K-1 thresholds are modeled
  jointly (correlated) via the threshold parameterization; `FALSE`
  treats them as independent.

- threads:

  If `TRUE`, enable within-chain `reduce_sum` threading for the Stan
  backend (set the thread count per chain via
  `fit_broc(threads_per_chain=)`). Default `FALSE`.

- gap_link:

  Link for threshold gaps: `"log"` (default) or `"softplus"`. `NULL`
  uses the family default.

## Value

A `brf` object: the model specification to pass to
[`broc()`](https://jed709.github.io/bayesroc/reference/broc.md).

## Examples

``` r
# Standard SDT
brf(conf | old ~ cond + (1|subj),
     criterion ~ 1 + (1|subj),
     sigma ~ 1)
#> brf() formula composition
#> ==========================
#> Primary:  conf | old ~ cond + (1 | subj) 
#>    criterion :  ~1 + (1 | subj) 
#>    sigma :  ~1 

# Bivariate with resp()
brf(resp(det_conf, src_conf) | type ~ 1 + (1|subj),
     discrim ~ 1, criterion ~ 1, criterion2 ~ 1, rho ~ 1)
#> brf() formula composition
#> ==========================
#> Primary:  resp(det_conf, src_conf) | type ~ 1 + (1 | subj) 
#>    discrim :  ~1 
#>    criterion :  ~1 
#>    criterion2 :  ~1 
#>    rho :  ~1 

# CDP with resp()
brf(resp(conf, rk_var) | old ~ 1 + (1|subj),
     fam ~ 1, criterion ~ 1, rec_crit ~ 1)
#> brf() formula composition
#> ==========================
#> Primary:  resp(conf, rk_var) | old ~ 1 + (1 | subj) 
#>    fam :  ~1 
#>    criterion :  ~1 
#>    rec_crit :  ~1 

# Cumulative
brf(rating ~ cond, cutpoints ~ 1 + (1|subj))
#> brf() formula composition
#> ==========================
#> Primary:  rating ~ cond 
#>    cutpoints :  ~1 + (1 | subj) 
```
