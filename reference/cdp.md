# Continuous Dual Process Signal Detection Model

Continuous dual-process model for Remember/Know paradigms, with
independent recollection and familiarity strengths (Wixted & Mickes,
2010).

## Usage

``` r
cdp(old_levels = NULL)
```

## Arguments

- old_levels:

  Integer vector specifying which confidence levels correspond to "old"
  responses (where R/K judgments apply). If `NULL` (default), inferred
  at fit time from the data as the bins where `rk` is non-NA. Pass
  explicitly only to override that inference (e.g. when some "old" bin
  happens to have no observed responses).

## Value

A `broc_family` object.

## Details

### Likelihood

Recollection R and familiarity F are independent Gaussians: \\R \sim
\mathcal{N}(\mu_R, \sigma_R)\\, \\F \sim \mathcal{N}(\mu_F, \sigma_F)\\
for targets, with \\(\mu_R, \mu_F) = (0, 0)\\ and \\(\sigma_R, \sigma_F)
= (1, 1)\\ for lures.

On the F (familiarity) dimension the cumulative is: \$\$P(Y \le k \mid
\mathrm{old} = 0) = \Phi(c_k), \quad P(Y \le k \mid \mathrm{old} = 1) =
\Phi\\\Big(\tfrac{c_k - \mu_F}{\sigma_F}\Big)\$\$ For ratings k not in
`old_levels` (new-side responses), the F cell (by differencing) is the
full likelihood. For ratings k in `old_levels` (old-side responses), the
cell factors across F (bin) and R (Remember vs Know): \$\$P(Y = k,\\
\mathrm{rk} \mid \mathrm{old}) = \big\[P(Y \le k) - P(Y \le k - 1)\big\]
\cdot P(\mathrm{rk} \mid \mathrm{old})\$\$ with the rk factor from the
cumulative on the R dimension. For targets: \$\$P(\mathrm{rk} = K \mid
\mathrm{old} = 1) = \Phi\\\Big(\tfrac{c_R - \mu_R}{\sigma_R}\Big), \quad
P(\mathrm{rk} = R \mid \mathrm{old} = 1) = 1 - \Phi\\\Big(\tfrac{c_R -
\mu_R}{\sigma_R}\Big)\$\$ (lures use the same form with \\\mu_R = 0,
\sigma_R = 1\\). For 3-level R/K/G with Know criterion \\c_K\\, the K
row becomes \\\Phi((c_R - \mu_R)/\sigma_R) - \Phi((c_K -
\mu_R)/\sigma_R)\\ and \\P(\mathrm{rk} = G) = \Phi((c_K -
\mu_R)/\sigma_R)\\.

### Parameters

- rec:

  (required) Recollection mean \\\mu_R\\ for targets. Identity link.

- fam:

  (required) Familiarity mean \\\mu_F\\ for targets. Identity link.

- criterion:

  (required) Confidence thresholds on the F dimension. Identity link.

- rec_crit:

  (required) Recollection criterion \\c_R\\ separating Remember vs Know
  responses on the R dimension. Identity link.

- sigma_R:

  (optional) SD of recollection for targets. If omitted, fixed at 1. Log
  link.

- sigma_F:

  (optional) SD of familiarity for targets. If omitted, fixed at 1. Log
  link.

- know_crit:

  (optional) Know criterion \\c_K\\ on the R dimension for the 3-level
  R/K/G case (when `rk` has values 1=R, 2=K, 3=G). Identity link. If
  omitted, only R/K is modeled.

### Response coding

`rk` values: 1 = Remember, 2 = Know, (3 = Guess for R/K/G mode).

### Formula syntax

    broc(brf(resp(conf, rk_resp) | old ~ 1 + (1 | subj),
             fam ~ 1 + (1 | subj),
             criterion ~ 1 + (1 | subj),
             rec_crit ~ 1 + (1 | subj),
             family = cdp(old_levels = 4:6)),
        data = dat)

## References

Wixted, J. T., & Mickes, L. (2010). A continuous dual-process model of
remember/know judgments. *Psychological Review*, *117*(4), 1025-1054.
