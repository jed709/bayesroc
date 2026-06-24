# Two-Dimensional Variable Recollection Dual Process Model

Two-dimensional variable-recollection dual-process model for conjoint
item-source recognition, with recollection-gated source discrimination
(Onyper, Zhang & Howard, 2010).

## Usage

``` r
vrdp2d(varying_source_criteria = FALSE, varying_re = "shared")
```

## Arguments

- varying_source_criteria:

  If `TRUE`, estimate separate source criteria for each item confidence
  level (K1 x (K2-1) source thresholds); the full model of Onyper et al.
  (2010). If `FALSE` (default), a single set of source criteria (K2-1
  thresholds) shared across detection levels.

- varying_re:

  Random-effect structure for varying source criteria. `"shared"`
  (default): one source-criterion RE shift shared across all detection
  bins. `"per_bin"`: a separate RE shift per detection bin. `"full"`: a
  separate RE for every source threshold within every detection bin.
  Requires `varying_source_criteria = TRUE` unless `"shared"`.

## Value

A `broc_family` object.

## Details

### Likelihood

Both states have diagonal covariance (zero detection-source
correlation), so the joint cumulative factors across dimensions. For new
items: \$\$P(Y_1 \le j,\\ Y_2 \le k \mid X = N) =
\Phi(c1_j)\\\Phi(c2_k)\$\$ For old items, recollection succeeds with
probability lambda. The cumulative is a 2-component mixture over
recollection: \$\$P(Y_1 \le j,\\ Y_2 \le k \mid X = \mathrm{old}) = (1 -
\lambda)\\\Phi\\\Big(\tfrac{c1_j - d'\_F}{\sigma_1}\Big)\Phi(c2_k) +
\lambda\\\Phi\\\Big(\tfrac{c1_j - d'\_F - d'\_R}{\sigma_1}\Big)
\Phi\\\Big(\tfrac{c2_k - \psi_X}{\sigma_2}\Big)\$\$ where \\\psi_A =
+d'\_S, \psi_B = -d'\_S\\ (the recollection-conditional source location
for A vs B), and lambda is the recollection probability (`lambda`). The
non-recollected component has zero source mean (no source info); the
recollected component carries source discriminability.

### Parameters

- dprime:

  (required) \\d'\_F\\: familiarity strength on the item dimension.
  Identity link.

- dprime2:

  (required) \\d'\_R\\: recollection boost on the item dimension.
  Identity link.

- discrim:

  (required) \\d'\_S\\: source discriminability for recollected items
  only (non-recollected items center at 0 on source). Identity link.

- lambda:

  (required) Recollection probability. Logit link.

- criterion:

  (required) K1-1 item-dimension thresholds. Identity link.

- criterion2:

  (required) K2-1 source-dimension thresholds. Identity link.

- sigma:

  (optional) SD on the item dimension. If omitted, fixed at 1. Log link.

- sigma2:

  (optional) SD on the source dimension for recollected items. If
  omitted, fixed at 1. Log link.

- discrim_B:

  (optional) Source discriminability for B items. If omitted,
  constrained to mirror A: `discrim_B = -discrim`.

### Formula syntax

Uses `resp(det_conf, src_conf)` with a 3-level item type conditioning
variable (new, A, B):

    broc(brf(resp(det_conf, src_conf) | type ~ 1 + (1 | subj),
             dprime2 ~ 1,
             discrim ~ 1 + (1 | subj),
             criterion ~ 1 + (1 | subj),
             criterion2 ~ 1 + (1 | subj),
             lambda ~ 1 + (1 | subj),
             family = vrdp2d()),
        data = dat)

## References

Onyper, S. V., Zhang, Y. X., & Howard, M. W. (2010). Some-or-none
recollection: Evidence from item and source memory. *Journal of
Experimental Psychology: General*, *139*(2), 341-364.
https://doi.org/10.1037/a0018926

## See also

[`bivariate_gaussian()`](https://jed709.github.io/bayesroc/reference/bivariate_gaussian.md),
[`bivariate_dp()`](https://jed709.github.io/bayesroc/reference/bivariate_dp.md).
