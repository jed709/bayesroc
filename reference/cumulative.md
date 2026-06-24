# Cumulative Ordinal Regression Model

Ordinal regression with cumulative-link cutpoints and no SDT structure
(Agresti, 2010).

## Usage

``` r
cumulative(link = link_probit())
```

## Arguments

- link:

  Link function (default:
  [`link_probit()`](https://jed709.github.io/bayesroc/reference/link_probit.md)).
  [`link_logit()`](https://jed709.github.io/bayesroc/reference/link_logit.md)
  also supported; GEV link not implemented for this family.

## Value

A `broc_family` object.

## Details

### Likelihood

For an ordinal response Y on a K-point scale with cutpoints c_1 \< ...
\< c_K-1: \$\$P(Y \le k \mid x) = F(c_k - \mu(x))\$\$ with \\c_0 =
-\infty\\ and \\c_K = +\infty\\, where F is the link CDF (probit or
logit) and \\\mu(x)\\ is the linear predictor. Cell probabilities by
differencing: \\P(Y = k \mid x) = F(c_k - \mu(x)) - F(c\_{k-1} -
\mu(x))\\.

### Parameters

- mu:

  (required) Linear predictor of covariate effects; the intercept is
  absorbed by the cutpoints, so `rating ~ 1` has no `mu` coefficients.
  Identity link.

- cutpoints:

  (required) K-1 ordered cutpoints (mid-anchor + K-2 ordered gaps).
  Identity link. Set priors via `thresh_mid` and `log_gaps`.

### Formula syntax

    broc(brf(rating ~ condition + (condition | subj),
             cutpoints ~ 1 + (1 | subj),
             family = cumulative()),
        data = dat)

## References

Agresti, A. (2010). *Analysis of ordinal categorical data* (2nd ed.).
Wiley.

## See also

[`evsd()`](https://jed709.github.io/bayesroc/reference/evsd.md) for the
simplest SDT model;
[`bivariate_cumulative()`](https://jed709.github.io/bayesroc/reference/bivariate_cumulative.md)
for the bivariate extension.
