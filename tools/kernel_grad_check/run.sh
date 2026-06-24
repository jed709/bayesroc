#!/usr/bin/env bash
# Finite-difference verification of every hand-coded kernel gradient in
# inst/stan/batch_sdt_core.hpp. Config-independent; no Stan/MCMC. This is the
# cheapest, most complete check for the analytic-gradient class (where the
# bivariate_dp / bounded gradient bugs lived). Run from the package root.
set -e
GPP="${GPP:-g++}"                 # set GPP to your Rtools/ucrt64 g++ if needed
CM="${CMDSTAN:-$HOME/.cmdstan/cmdstan-2.38.0}/stan/lib/stan_math"
HERE="tools/kernel_grad_check"
INC="-I $HERE -I $CM/lib/eigen_3.4.0 -I $CM/lib/boost_1.87.0 -I inst/stan"
for t in test_kernels; do
  echo "=== $t ==="
  "$GPP" -O2 -std=c++17 $INC "$HERE/$t.cpp" -o "$HERE/$t.exe"
  "./$HERE/$t.exe"
done
