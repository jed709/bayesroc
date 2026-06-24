# Bivariate Gaussian Signal Detection Model

Bivariate Gaussian signal detection model for source monitoring, with
unbounded and bounded (BBG) variants (DeCarlo, 2003; Starns et al.,
2014).

## Usage

``` r
bivariate_gaussian(
  varying_source_criteria = FALSE,
  bounded = FALSE,
  new_source_criteria = c("full", "shared"),
  old_levels = NULL,
  varying_re = "shared"
)
```

## Arguments

- varying_source_criteria:

  If `TRUE`, estimate separate source criteria for each detection
  confidence level.

- bounded:

  If `TRUE`, fit the Bounded Bivariate Gaussian (BBG; Starns et
  al. 2014) variant: dprime/discrim use log links (positive magnitudes),
  rho uses logistic link (in (0,1)), and the model anchors A on the
  negative source axis / B on the positive axis. Default `FALSE` fits
  the standard unbounded BG model (DeCarlo, 2003).

- new_source_criteria:

  With `varying_source_criteria = TRUE`, how source criteria are
  structured across detection responses (ignored otherwise). `"full"`
  (default): every detection response level has its own source criteria
  (K1 x (K2-1) in total). `"shared"`: responses that judge an item "new"
  share a single set of source criteria; requires `old_levels`.

- old_levels:

  Integer vector of detection response levels (on the 1..K1 scale, e.g.
  `old_levels = 4:6`) that count as "old" judgements; the rest are
  "new". Required when `new_source_criteria = "shared"`; ignored
  otherwise.

- varying_re:

  Random-effect structure for varying source criteria (distinct from
  `new_source_criteria`, which governs the fixed criteria). `"shared"`
  (default): one source-criterion RE shift shared across all detection
  bins. `"per_bin"`: a separate RE shift per detection bin. `"full"`: a
  separate RE for every source threshold within every detection bin.
  Requires `varying_source_criteria = TRUE` unless `"shared"`.

## Value

A `broc_family` object.

## Details

### Source labeling convention

Source A is the source associated with **low source ratings** (left end
of the source confidence scale); Source B with **high source ratings**.
In bounded mode, A items are anchored at the negative source-axis end
and B items at the positive end.

### Likelihood

For each item type \\X \in \\N, A, B\\\\, the joint cumulative on the
(detection, source) ratings is the bivariate normal CDF: \$\$P(Y_1 \le
j,\\ Y_2 \le k \mid X) = \Phi_2\\\Big(\tfrac{c1_j -
\mu_1(X)}{\sigma_1(X)},\\ \tfrac{c2_k - \mu_2(X)}{\sigma_2(X)};\\
\rho(X)\Big)\$\$ where the per-source means/SDs/correlation are:

- X = N::

  \\\mu_1 = 0,\\ \mu_2 = 0,\\ \sigma_1 = \sigma_2 = 1,\\ \rho = \rho_N\\

- X = A::

  \\\mu_1 = d'\_A,\\ \mu_2 = \psi_A,\\ \sigma_1 = \sigma\_{1A},\\
  \sigma_2 = \sigma\_{2A},\\ \rho = \rho_A\\

- X = B::

  \\\mu_1 = d'\_B,\\ \mu_2 = \psi_B,\\ \sigma_1 = \sigma\_{1B},\\
  \sigma_2 = \sigma\_{2B},\\ \rho = \rho_B\\

with \\d'\_A\\ = `dprime`, \\d'\_B\\ = `dprime_B`, \\\psi_A\\ =
`discrim`, \\\psi_B\\ = `discrim_B`. Cell probabilities by 4-corner
differencing: \$\$P(Y_1 = j,\\ Y_2 = k \mid X) = P(\le j, \le k) - P(\le
j-1, \le k) - P(\le j, \le k-1) + P(\le j-1, \le k-1)\$\$ In **bounded**
mode, the conditional source mean is clamped at 0 below the per-source
crossover point, so source items never have below-chance source
discrimination at low item evidence (Starns et al., 2014).

### Parameters

- dprime:

  (required) \\d'\_A\\ on the detection axis. Identity link (unbounded)
  or log link (bounded).

- discrim:

  (required) Signed source-axis position of A items. Identity link
  (unbounded) or log link (bounded – positive magnitude, placed on the
  negative source axis via internal negation).

- criterion:

  (required) K1-1 detection thresholds. Identity link.

- criterion2:

  (required) K2-1 source thresholds. Identity link.

- rho:

  (required) Detection-source correlation for A items. Fisher z link
  (unbounded) or logistic (bounded).

- dprime_B:

  (optional) \\d'\_B\\ for B items. If omitted, constrained
  `dprime_B = dprime` (equal detection sensitivity).

- discrim_B:

  (optional) Signed source-axis position of B items. If omitted,
  constrained mirror-symmetric (unbounded: `discrim_B = -discrim`;
  bounded: same magnitude as `discrim`, placed on the positive source
  axis).

- sigma:

  (optional) SD on detection axis for A items. If omitted, fixed at 1.
  Log link.

- sigma_B:

  (optional) SD on detection axis for B items. If omitted, constrained
  `sigma_B = sigma`.

- sigma2:

  (optional) SD on source axis for A items. If omitted, fixed at 1. Log
  link.

- sigma2_B:

  (optional) SD on source axis for B items. If omitted, constrained
  `sigma2_B = sigma2`.

- rho_B:

  (optional) Detection-source correlation for B items. If omitted,
  constrained mirror-symmetric to `rho`.

- rho_N:

  (optional) Detection-source correlation for new items. Fisher z link.
  If omitted, fixed at 0.

### Formula syntax

Uses `resp(det_conf, src_conf)` with a 3-level item type variable (new =
reference baseline, A, B) as the conditioning variable:

    broc(brf(resp(det_conf, src_conf) | type ~ 1 + (1 | subj),
             discrim ~ 1 + (1 | subj),
             criterion ~ 1 + (1 | subj),
             criterion2 ~ 1 + (1 | subj),
             rho ~ 1,
             family = bivariate_gaussian()),
        data = dat)

## References

DeCarlo, L. T. (2003b). Source monitoring and multivariate signal
detection theory, with a model for selection. *Journal of Mathematical
Psychology*, *47*(3), 292-303.

Starns, J. J., Rotello, C. M., & Hautus, M. J. (2014). Recognition
memory zROC slopes for items with correct versus incorrect source
decisions discriminate the dual process and unequal variance signal
detection models. *Journal of Experimental Psychology: Learning, Memory,
and Cognition*, *40*(5), 1205-1225.

## See also

[`bivariate_dp()`](https://jed709.github.io/bayesroc/reference/bivariate_dp.md)
for dual-process extension,
[`vrdp2d()`](https://jed709.github.io/bayesroc/reference/vrdp2d.md) for
variable recollection,
[`source_mixture()`](https://jed709.github.io/bayesroc/reference/source_mixture.md)
for the single-response mixture approach.
