// Per-group linear algebra of the transport map, four groups per SIMD register set.
// Group g: P_g = Lam_g + Q (lower triangles), L_g = chol(P_g), y_g = L_g^-1 r_g,
// u_g = L_g^-T (y_g + zeta_g), ld_g = sum_i log L_g,ii; and the adjoint for fixed Lam, as the
// reverse of that recurrence, with the factor recomputed rather than stored.
// Lam is tiled [n/4, D, D, 4] (n padded to a multiple of 4); r, zeta, u, ubar are [D, n].
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

namespace tmap {

constexpr int kMaxD = 64;  // widest block instantiated; tmap.py _NATIVE_MAX_DIM must match
typedef double v4d __attribute__((vector_size(32)));

inline v4d splat(double x) { return v4d{x, x, x, x}; }
inline v4d ld(const double* p) { v4d v; std::memcpy(&v, p, 32); return v; }
inline void st(double* p, v4d v) { std::memcpy(p, &v, 32); }
inline v4d vsqrt(v4d x) { return (v4d)_mm256_sqrt_pd((__m256d)x); }
inline v4d vlog(v4d x) { v4d r; for (int l = 0; l < 4; ++l) r[l] = __builtin_log(x[l]); return r; }

// a [D, n] row at groups g..g+3; the tail past n reads as `fill`
inline v4d ldn(const double* p, int64_t n, int64_t g, double fill) {
  if (g + 4 <= n) return ld(p + g);
  v4d v = splat(fill);
  for (int64_t l = 0; g + l < n; ++l) v[l] = p[g + l];
  return v;
}

inline void stn(double* p, int64_t n, int64_t g, v4d v) {
  if (g + 4 <= n) { st(p + g, v); return; }
  for (int64_t l = 0; g + l < n; ++l) p[g + l] = v[l];
}

template <int D>
struct Core {
  static void chol(const double* lam, const double* Q, v4d (&l)[D][D], v4d (&inv)[D], v4d& prod) {
    prod = splat(1.0);
    for (int j = 0; j < D; ++j) {
      v4d d = ld(lam + (j * D + j) * 4) + splat(Q[j * D + j]);
      for (int k = 0; k < j; ++k) d -= l[j][k] * l[j][k];
      v4d ljj = vsqrt(d);
      l[j][j] = ljj;
      prod *= ljj;
      inv[j] = splat(1.0) / ljj;
      for (int i = j + 1; i < D; ++i) {
        v4d v = ld(lam + (i * D + j) * 4) + splat(Q[i * D + j]);
        for (int k = 0; k < j; ++k) v -= l[i][k] * l[j][k];
        l[i][j] = v * inv[j];
      }
    }
  }

  static void forward(int64_t g0, int64_t g1, int64_t n, const double* Lam, const double* Q,
                      const double* r, const double* zeta, double* u, double* ldet) {
    for (int64_t g = g0; g < g1; g += 4) {
      v4d l[D][D], y[D], inv[D], prod;
      chol(Lam + g * D * D, Q, l, inv, prod);
      for (int i = 0; i < D; ++i) {
        v4d v = ldn(r + i * n, n, g, 0.0);
        for (int k = 0; k < i; ++k) v -= l[i][k] * y[k];
        y[i] = v * inv[i];
      }
      for (int i = D - 1; i >= 0; --i) {
        v4d v = y[i] + ldn(zeta + i * n, n, g, 0.0);
        for (int k = i + 1; k < D; ++k) v -= l[k][i] * y[k];
        y[i] = v * inv[i];
      }
      for (int i = 0; i < D; ++i) stn(u + i * n, n, g, y[i]);
      stn(ldet, n, g, vlog(prod));
    }
  }

  // Qbar (lower triangle, summed over the groups of this range) goes to qsum [D * D]
  static void backward(int64_t g0, int64_t g1, int64_t n, const double* Lam, const double* Q,
                       const double* r, const double* zeta, const double* ubar, const double* ldbar,
                       double* zetabar, double* rbar, double* qsum) {
    v4d qb[D][D];
    for (int i = 0; i < D; ++i) for (int j = 0; j < D; ++j) qb[i][j] = splat(0.0);
    for (int64_t g = g0; g < g1; g += 4) {
      v4d l[D][D], lb[D][D], y[D], yb[D], ug[D], ub[D], inv[D], prod;
      chol(Lam + g * D * D, Q, l, inv, prod);
      for (int i = 0; i < D; ++i) {
        for (int j = 0; j <= i; ++j) lb[i][j] = splat(0.0);
        v4d v = ldn(r + i * n, n, g, 0.0);
        for (int k = 0; k < i; ++k) v -= l[i][k] * y[k];
        y[i] = v * inv[i];
        ub[i] = ldn(ubar + i * n, n, g, 0.0);
      }
      for (int i = D - 1; i >= 0; --i) {
        v4d v = y[i] + ldn(zeta + i * n, n, g, 0.0);
        for (int k = i + 1; k < D; ++k) v -= l[k][i] * ug[k];
        ug[i] = v * inv[i];
      }
      // reverse of u_i = (y_i + zeta_i - sum_{k>i} L_ki u_k) / L_ii
      for (int i = 0; i < D; ++i) {
        v4d t = ub[i] * inv[i];
        lb[i][i] -= t * ug[i];
        stn(zetabar + i * n, n, g, t);
        yb[i] = t;
        for (int k = i + 1; k < D; ++k) { lb[k][i] -= t * ug[k]; ub[k] -= t * l[k][i]; }
      }
      // reverse of y_i = (r_i - sum_{k<i} L_ik y_k) / L_ii
      for (int i = D - 1; i >= 0; --i) {
        v4d t = yb[i] * inv[i];
        lb[i][i] -= t * y[i];
        stn(rbar + i * n, n, g, t);
        for (int k = 0; k < i; ++k) { lb[i][k] -= t * y[k]; yb[k] -= t * l[i][k]; }
      }
      v4d lbar = ldn(ldbar, n, g, 0.0);
      for (int j = 0; j < D; ++j) lb[j][j] += lbar * inv[j];
      // reverse of the Cholesky recurrence (P read in its lower triangle)
      for (int j = D - 1; j >= 0; --j) {
        for (int i = D - 1; i > j; --i) {
          v4d t = lb[i][j] * inv[j];
          lb[j][j] -= t * l[i][j];
          qb[i][j] += t;
          for (int k = 0; k < j; ++k) { lb[i][k] -= t * l[j][k]; lb[j][k] -= t * l[i][k]; }
        }
        v4d db = lb[j][j] * inv[j] * splat(0.5);
        qb[j][j] += db;
        for (int k = 0; k < j; ++k) lb[j][k] -= splat(2.0) * db * l[j][k];
      }
    }
    for (int i = 0; i < D; ++i)
      for (int j = 0; j < D; ++j) qsum[i * D + j] = qb[i][j][0] + qb[i][j][1] + qb[i][j][2] + qb[i][j][3];
  }
};

using FwdFn = void (*)(int64_t, int64_t, int64_t, const double*, const double*, const double*,
                       const double*, double*, double*);
using BwdFn = void (*)(int64_t, int64_t, int64_t, const double*, const double*, const double*,
                       const double*, const double*, const double*, double*, double*, double*);

template <int D>
FwdFn fwd_of() { return &Core<D>::forward; }
template <int D>
BwdFn bwd_of() { return &Core<D>::backward; }

template <int D>
FwdFn forward_from(int d) {
  if (d == D) return fwd_of<D>();
  if constexpr (D < kMaxD) return forward_from<D + 1>(d);
  return nullptr;
}
template <int D>
BwdFn backward_from(int d) {
  if (d == D) return bwd_of<D>();
  if constexpr (D < kMaxD) return backward_from<D + 1>(d);
  return nullptr;
}
inline FwdFn forward_for(int D) { return D >= 1 ? forward_from<1>(D) : nullptr; }
inline BwdFn backward_for(int D) { return D >= 1 ? backward_from<1>(D) : nullptr; }

// r = s - Lam delta for groups g0..g1 (r holds s on entry); Lam full symmetric per group
inline void coupling(int D, int64_t n, int64_t g0, int64_t g1, const double* Lam, const double* delta,
                     double* r) {
  for (int64_t g = g0; g < g1; g += 4) {
    const double* lam = Lam + g * D * D;
    for (int a = 0; a < D; ++a) {
      v4d acc = ldn(r + a * n, n, g, 0.0);
      for (int b = 0; b < D; ++b) acc -= ld(lam + (a * D + b) * 4) * ldn(delta + b * n, n, g, 0.0);
      stn(r + a * n, n, g, acc);
    }
  }
}

// deltabar = -Lam rbar, in place (buffer holds rbar on entry)
inline void coupling_adjoint(int D, int64_t n, int64_t g0, int64_t g1, const double* Lam, double* rb) {
  v4d t[kMaxD];
  for (int64_t g = g0; g < g1; g += 4) {
    const double* lam = Lam + g * D * D;
    for (int b = 0; b < D; ++b) t[b] = ldn(rb + b * n, n, g, 0.0);
    for (int a = 0; a < D; ++a) {
      v4d acc = splat(0.0);
      for (int b = 0; b < D; ++b) acc -= ld(lam + (a * D + b) * 4) * t[b];
      stn(rb + a * n, n, g, acc);
    }
  }
}

}  // namespace tmap
