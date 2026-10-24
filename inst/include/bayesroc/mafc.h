#pragma once
// =============================================================================
// m-alternative forced choice cell (DeCarlo 2012, unequal variance optional): log P(choice k | target s)
// =============================================================================

#include <cmath>

#include "pcq.h"

namespace mafc {

constexpr int kMaxAlt = pcq::kMaxTerms + 1;

// positions 0-based; b[0..m-1] the row's position biases; sig the target's noise SD (lures 1).
// Returns log P and d/dd, d/dsig and d/db[j] (db has m entries).
inline double cell(int m, int s, int k, double d, double sig, const double* b, double& dd, double& ds, double* db) {
  for (int j = 0; j < m; ++j) db[j] = 0.0;
  if (m == 2) {
    const int o = 1 - s;
    const double tau = std::sqrt(1.0 + sig * sig), sg = k == s ? 1.0 : -1.0;
    const double z = sg * (d + b[s] - b[o]) / tau;
    const double lp = bvn_tail::log_Phi(z), l = sg * pcq::lambda(z) / tau;
    dd = l;
    ds = -l * (d + b[s] - b[o]) * sig / (tau * tau);
    db[s] = l;
    db[o] = -l;
    return lp;
  }
  double a[pcq::kMaxTerms], c[pcq::kMaxTerms], dc[pcq::kMaxTerms], da[pcq::kMaxTerms];
  int idx[pcq::kMaxTerms];
  int n = 0;
  if (k == s) {
    for (int j = 0; j < m; ++j) {
      if (j == s) continue;
      a[n] = sig;
      c[n] = d + b[s] - b[j];
      idx[n++] = j;
    }
    const double lp = pcq::log_int(n, a, c, dc, da);
    dd = 0.0;
    ds = 0.0;
    for (int t = 0; t < n; ++t) {
      dd += dc[t];
      ds += da[t];
      db[s] += dc[t];
      db[idx[t]] -= dc[t];
    }
    return lp;
  }
  const double is = 1.0 / sig;
  a[0] = is;
  c[0] = (b[k] - b[s] - d) * is;
  idx[0] = s;
  n = 1;
  for (int j = 0; j < m; ++j) {
    if (j == s || j == k) continue;
    a[n] = 1.0;
    c[n] = b[k] - b[j];
    idx[n++] = j;
  }
  const double lp = pcq::log_int(n, a, c, dc, da);
  dd = -dc[0] * is;
  ds = -(dc[0] * c[0] + da[0] * is) * is;
  db[k] += dc[0] * is;
  db[s] -= dc[0] * is;
  for (int t = 1; t < n; ++t) {
    db[k] += dc[t];
    db[idx[t]] -= dc[t];
  }
  return lp;
}

}  // namespace mafc
