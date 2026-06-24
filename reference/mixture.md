# Mixture Signal Detection Model

Two-state mixture model for old items, combining a high- and a
low-strength Gaussian component (DeCarlo, 2002).

## Usage

``` r
mixture(link = link_probit())
```

## Arguments

- link:

  Link function (default:
  [`link_probit()`](https://jed709.github.io/bayesroc/reference/link_probit.md)).

## Value

A `broc_family` object.

## Details

### Likelihood

Mixture of two latent classes (DeCarlo, 2002), expressed cumulatively:
\$\$P(Y \le k \mid \mathrm{old}) = \lambda\\\Phi\\\Big(\tfrac{c_k -
d'\_1 \cdot \mathrm{old}}{\sigma_1}\Big) + (1 -
\lambda)\\\Phi\\\Big(\tfrac{c_k - d'\_2 \cdot
\mathrm{old}}{\sigma_2}\Big)\$\$ where lambda is the mixing weight on
state 1 (`lambda`), \\(d'\_1, \sigma_1)\\ = (`dprime`, `sigma`), and
\\(d'\_2, \sigma_2)\\ = (`dprime2`, `sigma2`). At `old = 0` both
Gaussians collapse to the noise N(0, 1) baseline, so new items reduce to
standard EVSD.

If a lure mixture is specified (both `dprime_L` and `lambda_L`), new
items get their own mixture: \$\$P(Y \le k \mid \mathrm{old} = 0) = (1 -
\lambda_L)\\\Phi(c_k) + \lambda_L\\\Phi\\\Big(\tfrac{c_k -
d'\_L}{\sigma_L}\Big)\$\$

When both `dprime` and `dprime2` are free, an ordered constraint
(`dprime > dprime2`) is imposed for identifiability.

### Parameters

The `_L` suffix marks the optional lure-item mixture component (applied
to new items); the unsuffixed parameters describe the old-item mixture.

- dprime:

  (required) \\d'\_1\\ for the high (state 1) old-item component.
  Identity link.

- criterion:

  (required) K-1 thresholds. Identity link.

- lambda:

  (required) Mixing weight on state 1. Logit link.

- dprime2:

  (optional) \\d'\_2\\ for state 2. If omitted, fixed at 0. Identity
  link.

- sigma:

  (optional) SD \\\sigma_1\\ of state 1. If omitted, fixed at 1. Log
  link.

- sigma2:

  (optional) SD \\\sigma_2\\ of state 2. If omitted, fixed at 1. Log
  link.

- dprime_L:

  (optional) d' for an additional lure-item mixture component (new items
  only). Identity link. If omitted, no lure mixture (new items are
  standard EVSD N(0, 1)).

- lambda_L:

  (optional) Mixing weight on the lure-mixture component (new items
  only). Logit link. Required (with `dprime_L`) to enable the lure
  mixture.

- sigma_L:

  (optional) SD of the lure-mixture component. If omitted, fixed at 1.
  Log link.

### Formula syntax

    broc(brf(conf | old ~ 1 + (1 | subj),
             criterion ~ 1 + (1 | subj),
             lambda ~ condition + (1 | subj),
             dprime2 ~ 1,
             family = mixture()),
        data = dat)

## References

DeCarlo, L. T. (2002). Signal detection theory with finite mixture
distributions: Theoretical developments with applications to recognition
memory. *Psychological Review*, *109*(4), 710-721.
https://doi.org/10.1037/0033-295X.109.4.710

## See also

[`dpsd()`](https://jed709.github.io/bayesroc/reference/dpsd.md),
[`uvsd()`](https://jed709.github.io/bayesroc/reference/uvsd.md),
[`evsd()`](https://jed709.github.io/bayesroc/reference/evsd.md).
