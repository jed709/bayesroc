# Test fixtures

Reference values that the kernel tests check against. Do not edit them by hand.
No generator script is in this repository. The provenance below comes from the
commit that added each file, its test, and the working scripts that sit outside
the repository. "Not determinable" means none of these records it.

| File | Added | Contents | Reference method | Generator | Used by |
|---|---|---|---|---|---|
| `pcq_reference.txt` | 0923185, 2026-10-24 | 600 Gaussian product-of-CDFs integrals. Each line is `m; slopes; offsets; log value` | 30 digits (commit message and test title). The method is not determinable | Not determinable | test-kernel-consistency.R (python/pcq_check.py) |
| `bvn_tail_ref.csv` | 236279f, 2026-09-29 | 80 far-tail bivariate normal rectangles `a1,b1,a2,b2,rho,log_p`. Some bounds are `Inf` | Not determinable. The commit gives only the kernel's own accuracy (about 1e-9 relative) | Not determinable | test-kernel-consistency.R (python/bvn_check.py) |
| `bvn_tail_hirho_ref.csv` | 3a7458e, 2026-10-23 | 240 far-tail rectangles at rho = ±0.9, ±0.99 and ±0.999, same columns | A "40-digit reference" (commit message). The method is not determinable | Not determinable | test-kernel-consistency.R (python/bvn_check.py) |
| `cdp_cell_ref.csv` | c9eaec9, 2026-09-30 | 189 CDP R/K/G cells, 63 each of kinds 1 (Remember), 2 (Know) and 3 (Guess): `kind, mu_R, sigma_R, mu_F, sigma_F, c_R, c_K, tau_lo, tau_hi, log_p`. A non-finite `log_p` marks an empty cell | mpmath (the commit says the kernel matches "mpmath to ~1e-12"). The precision is not determinable | Not determinable | test-cdp-rkg.R (R cells), test-kernel-consistency.R (native, via python/bvn_check.py) |
| `lnoise_reference.csv` | b0cbcb9, 2026-10-24 | 1300 lognormal meta-noise cells with their gradients: `a, Dn, Df, s, P, da, dln, dlf, ds` | mpmath `quad` at 28 digits, printed to 20 significant digits | Byte-identical (after stripping CR) to `Downloads/bayesroc-scope/logn_ref/refcell.csv` (1000 rows, seed 11) followed by `refcell_k.csv` (300 rows, the heavy-tail draw, seed 12), both from `logn_ref/refcell.py`. That script was modified 3 minutes after `refcell.csv` was written, so its current version may not be the one that wrote the first 1000 rows | test-lognormal-noise.R (R and native, via python/lnoise_check.py) |
| `metad_reference.csv` | b52693d, 2026-10-24 | 2392 meta-d' cell probabilities (equal-variance sets) over both anchors and global or response-specific M | Cross-checked against bmm 1.4.1 (R and Stan), hmetad 0.2.0 (R) and numerical integration. Every row agrees with at least one of them to better than 1e-9 relative (max 8e-14 against bmm R) | Identical in values to the `variant == "ev"` rows of `Downloads/bayesroc-scope/metad_ref/metad_reference_fixture.csv` (2026-10-23, from `metad_ref/build_fixture.R`), with the cross-check columns dropped. See `metad_ref/README_fixture.md` there | test-metad.R (R cells; native via python/metad_check.py) |

The paper's Kernel Checks (`bayesroc-paper-scope/paper-code/Kernel Checks`) build
their own mpmath reference tables (`ref_*.csv`). They did not produce these files.
