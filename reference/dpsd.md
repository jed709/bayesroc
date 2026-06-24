# Dual Process Signal Detection Model

Recognition-memory model combining discrete recollection and continuous
(UVSD) familiarity (Yonelinas, 1994).

## Usage

``` r
dpsd(link = link_probit())
```

## Arguments

- link:

  Link function (default:
  [`link_probit()`](https://jed709.github.io/bayesroc/reference/link_probit.md)).

## Value

A `broc_family` object.

## Details

### Likelihood

For new items, responses follow standard EVSD (no recollection): \$\$P(Y
\le k \mid \mathrm{old} = 0) = \Phi(c_k)\$\$ For old items, recollection
succeeds with probability lambda, producing the highest-confidence "old"
response Y = K. Otherwise, the familiarity-only distribution is UVSD
with mean d' and SD sigma: \$\$P(Y \le k \mid \mathrm{old} = 1) = (1 -
\lambda)\\\Phi\\\Big(\frac{c_k - d'}{\sigma}\Big) \quad (k \< K)\$\$
\$\$P(Y \le K \mid \mathrm{old} = 1) = 1\$\$

### Parameters

- dprime:

  (required) Familiarity sensitivity. Identity link.

- criterion:

  (required) K-1 thresholds. Identity link.

- lambda:

  (required) Recollection probability. Logit link.

- sigma:

  (optional) SD of the familiarity signal distribution. If omitted,
  fixed at 1 (= EVSD familiarity). Log link.

### Formula syntax

    broc(brf(conf | old ~ 1 + (1 | subj),
             criterion ~ 1 + (1 | subj),
             lambda ~ condition + (1 | subj),
             family = dpsd()),
        data = dat)

## References

Yonelinas, A. P. (1994). Receiver-operating characteristics in
recognition memory: Evidence for a dual-process model. *Journal of
Experimental Psychology: Learning, Memory, and Cognition*, *20*(6),
1341-1354. https://doi.org/10.1037/0278-7393.20.6.1341

## See also

[`evsd()`](https://jed709.github.io/bayesroc/reference/evsd.md),
[`uvsd()`](https://jed709.github.io/bayesroc/reference/uvsd.md),
[`mixture()`](https://jed709.github.io/bayesroc/reference/mixture.md),
[`bivariate_dp()`](https://jed709.github.io/bayesroc/reference/bivariate_dp.md).
