# Bivariate Cumulative Ordinal Model

Two correlated ordinal responses modelled jointly via the bivariate
normal CDF, with no SDT structure (Agresti, 2010).

## Usage

``` r
bivariate_cumulative()
```

## Value

A `broc_family` object.

## Details

### Likelihood

For two ordinal responses (Y_1, Y_2) with cutpoints c1_1 \< ... \<
c1_K_1-1 and c2_1 \< ... \< c2_K_2-1, the joint cumulative is the
bivariate normal CDF: \$\$P(Y_1 \le j,\\ Y_2 \le k \mid x) =
\Phi_2\big(c1_j - \mu_1(x),\\ c2_k - \mu_2(x);\\ \rho\big)\$\$ with
\\c_0 = -\infty\\, \\c_K = +\infty\\. Variance fixed at 1 on both
dimensions; \\\mu_1\\ and \\\mu_2\\ have no intercept (absorbed into
cutpoints) for identifiability. Cell probabilities by 4-corner
differencing: \$\$P(Y_1 = j,\\ Y_2 = k \mid x) = P(\le j, \le k) - P(\le
j-1, \le k) - P(\le j, \le k-1) + P(\le j-1, \le k-1)\$\$

### Parameters

- mu1:

  (required) Linear predictor for dimension 1; the intercept is absorbed
  by the cutpoints. Identity link.

- mu2:

  (required) Linear predictor for dimension 2. Identity link.

- cutpoints1:

  (required) K1-1 ordered cutpoints for dimension 1 (mid-anchor +
  ordered gaps). Identity link. Set priors via `thresh_mid` and
  `log_gaps`.

- cutpoints2:

  (required) K2-1 ordered cutpoints for dimension 2. Identity link. Set
  priors via `thresh_mid2` and `log_gaps2`.

- rho:

  (required) Correlation between dimensions; effective \\\rho =
  \tanh(\rho\_{\text{param}})\\. Fisher z link.

### Formula syntax

    broc(brf(resp(y1, y2) ~ x1,
             mu2 ~ x2,
             cutpoints1 ~ 1,
             cutpoints2 ~ 1,
             rho ~ 1,
             family = bivariate_cumulative()),
        data = dat)

## See also

[`cumulative()`](https://jed709.github.io/bayesroc/reference/cumulative.md)
for univariate ordinal,
[`bivariate_gaussian()`](https://jed709.github.io/bayesroc/reference/bivariate_gaussian.md)
for the SDT version with new-item baseline.
