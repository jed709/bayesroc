# Bivariate Dual Process Signal Detection Model

Bivariate dual-process model: Gaussian familiarity plus discrete item
and source recollection (Starns et al., 2014).

## Usage

``` r
bivariate_dp(
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

  If `TRUE`, fit the BBDP variant (conditional source mean clamped at
  the new-item baseline). If `FALSE` (default), fit the BDP variant with
  standard bivariate Gaussian familiarity.

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

Same source labeling convention as
[`bivariate_gaussian()`](https://jed709.github.io/bayesroc/reference/bivariate_gaussian.md):
source A = low source ratings, source B = high.

### Likelihood

For each old item, the cell probability is a 3-component mixture over
recollection states: \$\$P(Y_1 = j,\\ Y_2 = k \mid X) = (1 -
\lambda)\\P\_{\mathrm{F}}(j, k \mid X) + \lambda(1 -
\lambda_2)\\\mathbf{1}\[j = K_1\]\\P\_{\mathrm{S}}(k \mid X) + \lambda
\lambda_2\\\mathbf{1}\[j = K_1, k = k_X^\*\]\$\$ where:

- lambda is the item-recollection probability, lambda_2 is the
  source-recollection probability (`lambda`, `lambda2`).

- \\P\_{\mathrm{F}}(j, k \mid X)\\ is the bivariate-Gaussian familiarity
  cell probability, defined cumulatively as \\P\_{\mathrm{F}}(Y_1 \le j,
  Y_2 \le k \mid X) = \Phi_2\big((c1_j - \mu_1)/\sigma_1,\\ (c2_k -
  \mu_2)/\sigma_2;\\ \rho\big)\\ with 4-corner differencing, taking
  per-source \\(\mu_1, \mu_2, \sigma_1, \sigma_2, \rho)\\ from
  [`bivariate_gaussian()`](https://jed709.github.io/bayesroc/reference/bivariate_gaussian.md).

- \\P\_{\mathrm{S}}(k \mid X)\\ is the marginal source cell probability
  (familiarity integrated out): \\P\_{\mathrm{S}}(Y_2 \le k \mid X) =
  \Phi((c2_k - \mu_2)/\sigma_2)\\ with 1D differencing.

- \\k_X^\*\\ is the recollection-success source response: \\k_X^\* = 1\\
  for X = A, \\k_X^\* = K_2\\ for X = B. Both signal a correctly
  identified source under recollection.

- For new items: lambda = lambda_2 = 0, so the likelihood reduces to
  \\P\_{\mathrm{F}}\\ alone with new-item parameters.

In the bounded (BBDP) variant, the bivariate piece uses the source-mean
clamping of Starns et al. (2014); otherwise (BDP) the standard bivariate
normal applies.

### Parameters

All parameters of
[`bivariate_gaussian()`](https://jed709.github.io/bayesroc/reference/bivariate_gaussian.md)
(`dprime`, `dprime_B`, `discrim`, `discrim_B`, `sigma`, `sigma_B`,
`sigma2`, `sigma2_B`, `rho`, `rho_B`, `rho_N`, `criterion`,
`criterion2`) plus:

- lambda:

  (required) Item-recollection probability for Source A. Logit link.

- lambda_B:

  (optional) Item-recollection probability for Source B. If omitted,
  constrained `lambda_B = lambda`.

- lambda2:

  (required) Source-recollection probability for Source A. Logit link.

- lambda2_B:

  (optional) Source-recollection probability for Source B. If omitted,
  constrained `lambda2_B = lambda2`.

Defaults for the bivariate parameters match
[`bivariate_gaussian()`](https://jed709.github.io/bayesroc/reference/bivariate_gaussian.md)
– e.g. `dprime_B` defaults to `dprime`, `sigma` defaults to fixed at 1,
`discrim_B` defaults to mirror-symmetric.

### Formula syntax

    broc(brf(resp(det_conf, src_conf) | type ~ 1 + (1 | subj),
             discrim ~ 1 + (1 | subj),
             criterion ~ 1 + (1 | subj),
             criterion2 ~ 1 + (1 | subj),
             rho ~ 1,
             lambda ~ 1 + (1 | subj),
             lambda2 ~ 1 + (1 | subj),
             family = bivariate_dp()),
        data = dat)

## References

Starns, J. J., Rotello, C. M., & Hautus, M. J. (2014). Recognition
memory zROC slopes for items with correct versus incorrect source
decisions discriminate the dual process and unequal variance signal
detection models. *Journal of Experimental Psychology: Learning, Memory,
and Cognition*, *40*(5), 1205-1225.

## See also

[`bivariate_gaussian()`](https://jed709.github.io/bayesroc/reference/bivariate_gaussian.md),
[`vrdp2d()`](https://jed709.github.io/bayesroc/reference/vrdp2d.md).
