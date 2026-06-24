# Save a fitted broc_fit to disk

Writes a portable .rds containing the draws, the broc_model, and the
metadata needed to reconstruct the fit. Pass `file` to
[`fit_broc()`](https://jed709.github.io/bayesroc/reference/fit_broc.md)
to save automatically, or call this on an existing fit. Always save with
this (or `file =` in
[`fit_broc()`](https://jed709.github.io/bayesroc/reference/fit_broc.md)),
not [`saveRDS()`](https://rdrr.io/r/base/readRDS.html), which can lose
Stan draws.

## Usage

``` r
save_broc_fit(
  fit,
  file,
  compress = FALSE,
  verbose = TRUE,
  include_loglik = NULL
)
```

## Arguments

- fit:

  A `broc_fit` object from
  [`fit_broc()`](https://jed709.github.io/bayesroc/reference/fit_broc.md).

- file:

  Path to write to (use `.rds` extension by convention).

- compress:

  Compression for `saveRDS`. Default `FALSE` (fast; posterior draws
  barely compress). Set `"gzip"`/`"xz"` for a smaller, slower file.

- verbose:

  If `TRUE` (default), print a brief progress message.

- include_loglik:

  Whether to write `log_lik` (needed for
  [`loo()`](https://mc-stan.org/loo/reference/loo.html)). `NULL`
  (default) follows the fit's `fit_broc(save_loglik = )` setting;
  `TRUE`/`FALSE` overrides it for this save.

## Value

Invisibly returns `file`.

## See also

[`load_broc_fit()`](https://jed709.github.io/bayesroc/reference/load_broc_fit.md).

## Examples

``` r
if (FALSE) { # \dontrun{
fit <- fit_broc(model)
save_broc_fit(fit, "my_fit.rds")
fit2 <- load_broc_fit("my_fit.rds")
summary(fit2)

# Smaller file (slower) when disk space matters:
save_broc_fit(fit, "tmp.rds", compress = "gzip")
} # }
```
