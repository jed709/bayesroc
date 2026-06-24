# Equal Variance Signal Detection Model

Canonical signal detection model. Signal and noise distributions are
assumed normal with equal variance fixed at 1 (Macmillan & Creelman,
2005).

## Usage

``` r
evsd(link = link_probit())
```

## Arguments

- link:

  Link function (default:
  [`link_probit()`](https://jed709.github.io/bayesroc/reference/link_probit.md)).
  Also supports
  [`link_logit()`](https://jed709.github.io/bayesroc/reference/link_logit.md)
  for logistic CDF.

## Value

A `broc_family` object.

## Details

### Likelihood

For an old/new judgment Y on a K-point confidence scale (`old` in 0, 1)
with criteria c_1 \< c_2 \< ... \< c_K-1: \$\$P(Y \le k \mid
\mathrm{old}) = \Phi(c_k - \mathrm{old} \cdot d')\$\$ for k = 1, ...,
K-1, where \\\Phi\\ is the link CDF (probit by default). Cell
probabilities by differencing: \\P(Y = k \mid \mathrm{old}) = P(Y \le k
\mid \mathrm{old}) - P(Y \le k - 1 \mid \mathrm{old})\\ with \\P(Y \le
0) = 0\\ and \\P(Y \le K) = 1\\. Both new- and old-item distributions
have unit SD.

### Parameters

- dprime:

  (required) Sensitivity. Identity link.

- criterion:

  (required) K-1 confidence thresholds, parameterized internally as a
  mid-anchor `thresh_mid` plus K-2 ordered gaps. Identity link on the
  gap parameterization.

### Formula syntax

    broc(brf(conf | old ~ condition + (1 + condition | subj),
             criterion ~ 1 + (1 | subj),
             family = evsd()),
        data = dat)

## References

Macmillan, N. A., & Creelman, C. D. (2005). *Detection theory: A user's
guide* (2nd ed.). Lawrence Erlbaum.

## See also

[`uvsd()`](https://jed709.github.io/bayesroc/reference/uvsd.md) for
unequal-variance extension,
[`dpsd()`](https://jed709.github.io/bayesroc/reference/dpsd.md) for
dual-process,
[`mixture()`](https://jed709.github.io/bayesroc/reference/mixture.md)
for mixture variants.
