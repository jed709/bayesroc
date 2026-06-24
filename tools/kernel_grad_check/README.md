# Kernel gradient checks

Standalone finite-difference verification of every hand-coded analytic gradient
in `inst/stan/batch_sdt_core.hpp` (the batch-likelihood C++ kernels). This is
config-independent and needs no Stan/MCMC — it directly compares each kernel's
analytic gradient to a central finite difference of its own value/lp. This is
the cheapest, most complete check for the analytic-gradient class (where the
bivariate_dp / bounded gradient bugs were found and fixed).

`test_kernels.cpp` covers: cell_log_prob(+logit), cell_prob, uni_cell,
binormal_cdf_grad (Owen's T), bivariate_cell, bivariate_sdt_cell,
bounded_marginal_source, dpsdt/mixture/vrdp2d cells, cdp_strip,
bivariate_dp_cell (fast/slow/corner, item types 2/3/new), and
bounded_bivariate_sdt_cell / bounded_bivariate_dp_cell. (239 checks.)

`stan/math.hpp` is a stub that pulls in only Eigen + boost owens_t so the header
compiles without the full Stan math toolchain.

## Run (from the package root)
    CMDSTAN=~/.cmdstan/cmdstan-2.38.0 GPP=/path/to/g++ bash tools/kernel_grad_check/run.sh

Prints `N checks, 0 BAD` on success (nonzero exit on any mismatch).
