# Source Mixture Signal Detection Model

Finite-mixture signal detection model for source-discrimination tasks
(DeCarlo, 2003).

## Usage

``` r
source_mixture(link = link_probit())
```

## Arguments

- link:

  Link function (default:
  [`link_probit()`](https://jed709.github.io/bayesroc/reference/link_probit.md)).

## Value

A `broc_family` object.

## Details

### Likelihood

Three latent Gaussian components, all with unit variance: nonattended at
0 (shared), attended Source A at \\d'\_A\\ (typically negative), and
attended Source B at \\d'\_B\\ (typically positive). For each source,
the source-rating cumulative is a mixture of attended and nonattended:
\$\$P(Y \le k \mid \mathrm{source} = A) = \lambda_A\\\Phi(c_k - d'\_A) +
(1 - \lambda_A)\\\Phi(c_k)\$\$ \$\$P(Y \le k \mid \mathrm{source} = B) =
\lambda_B\\\Phi(c_k - d'\_B) + (1 - \lambda_B)\\\Phi(c_k)\$\$ where
lambda_A, lambda_B are the source-conditional attention probabilities
(`lambda`, `lambda_B`).

The overall source discriminability is \\d\_{AB} = d'\_B - d'\_A\\.

### Parameters

- dprime:

  (required) \\d'\_A\\: location of the attended Source A distribution.
  Identity link.

- criterion:

  (required) K-1 ordered confidence thresholds (mid-anchor + ordered
  gaps). Identity link.

- lambda:

  (required) Attention probability for Source A. Logit link.

- dprime_B:

  (optional) \\d'\_B\\: location of the attended Source B distribution.
  Identity link. If omitted, constrained to `dprime_B = -dprime`.

- lambda_B:

  (optional) Attention probability for Source B. Logit link. If omitted,
  constrained to `lambda_B = lambda`.

### Formula syntax

The conditioning variable after `|` is the source variable (2-level
factor or 0/1 indicator for Source A vs Source B).

    broc(brf(conf | source ~ 1 + (1 | subj),
             criterion ~ 1 + (1 | subj),
             lambda ~ 1 + (1 | subj),
             family = source_mixture()),
        data = dat)

## References

DeCarlo, L. T. (2003a). An application of signal detection theory with
finite mixture distributions to source discrimination. *Journal of
Experimental Psychology: Learning, Memory, and Cognition*, *29*(5),
767-778.

## See also

[`bivariate_gaussian()`](https://jed709.github.io/bayesroc/reference/bivariate_gaussian.md)
for the bivariate-normal approach to source monitoring (separate
detection + source ratings).
