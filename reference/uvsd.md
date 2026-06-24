# Unequal Variance Signal Detection Model

Extends [`evsd()`](https://jed709.github.io/bayesroc/reference/evsd.md)
by allowing the signal (old-item) distribution to have a different
variance than the noise (new-item) distribution (Macmillan & Creelman,
2005).

## Usage

``` r
uvsd(link = link_probit())
```

## Arguments

- link:

  Link function (default:
  [`link_probit()`](https://jed709.github.io/bayesroc/reference/link_probit.md)).

## Value

A `broc_family` object.

## Details

### Likelihood

Noise distribution N(0, 1); signal distribution N(d', sigma). For an
old/new judgment Y on a K-point scale (`old` in 0, 1) with criteria c_1
\< ... \< c_K-1: \$\$P(Y \le k \mid \mathrm{old} = 0) = \Phi(c_k)\$\$
\$\$P(Y \le k \mid \mathrm{old} = 1) = \Phi\\\Big(\frac{c_k -
d'}{\sigma}\Big)\$\$ for k = 1, ..., K-1.

### Parameters

- dprime:

  (required) Sensitivity. Identity link.

- criterion:

  (required) K-1 confidence thresholds (mid-anchor + K-2 ordered gaps).
  Identity link.

- sigma:

  (required) SD of the signal distribution. Log link. To fix sigma at 1,
  use [`evsd()`](https://jed709.github.io/bayesroc/reference/evsd.md)
  instead.

### Formula syntax

    broc(brf(conf | old ~ condition + (1 | subj),
             criterion ~ 1 + (1 | subj),
             sigma ~ 1 + (1 | subj),
             family = uvsd()),
        data = dat)

## References

Macmillan, N. A., & Creelman, C. D. (2005). *Detection theory: A user's
guide* (2nd ed.). Lawrence Erlbaum.

## See also

[`evsd()`](https://jed709.github.io/bayesroc/reference/evsd.md),
[`dpsd()`](https://jed709.github.io/bayesroc/reference/dpsd.md),
[`mixture()`](https://jed709.github.io/bayesroc/reference/mixture.md).
