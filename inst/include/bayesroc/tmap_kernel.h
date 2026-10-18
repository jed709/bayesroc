// Per-group linear algebra of the transport map, four groups per SIMD register set.
// Group g: P_g = Lam_g + Q (lower triangles), L_g = chol(P_g), y_g = L_g^-1 r_g,
// u_g = L_g^-T (y_g + zeta_g), ld_g = sum_i log L_g,ii; and the adjoint for fixed Lam, as the
// reverse of that recurrence. Narrow blocks (Core) recompute the factor in the adjoint; wider
// ones (tri) keep it packed, rows of the lower triangle in turn, from the forward pass.
// Lam is tiled [n/4, D, D, 4] (n padded to a multiple of 4); r, zeta, u, ubar are [D, n].
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <vector>

namespace tmap {

constexpr int kMaxD = 15;  // widest block with its own unrolled instance
constexpr int kTriMin = kMaxD + 1;  // from here on: packed factors (namespace tri), stored for the adjoint
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
  std::vector<v4d> tb(D);
  v4d* t = tb.data();
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

namespace tri {

inline size_t off(int i) { return size_t(i) * (i + 1) / 2; }
inline size_t tri_size(int D) { return off(D); }

inline v4d P(const double* lam, const double* Q, int D, int i, int j) {
  return ld(lam + (size_t(i) * D + j) * 4) + splat(Q[i * D + j]);
}

inline v4d dot(const v4d* a, const v4d* b, int n) {
  v4d s0 = splat(0.0), s1 = s0, s2 = s0, s3 = s0;
  int k = 0;
  for (; k + 3 < n; k += 4) { s0 += a[k] * b[k]; s1 += a[k + 1] * b[k + 1]; s2 += a[k + 2] * b[k + 2]; s3 += a[k + 3] * b[k + 3]; }
  for (; k < n; ++k) s0 += a[k] * b[k];
  return (s0 + s1) + (s2 + s3);
}

// packed lower factor l (row i at off(i)), inv = 1 / diag; rows in blocks of four, two columns at a time
inline void chol_rows(int D, const double* lam, const double* Q, v4d* l, v4d* inv, v4d& prod, int i0, int R) {
  v4d* rw[4];
  for (int a = 0; a < R; ++a) rw[a] = l + off(i0 + a);
  int j = 0;
  if (R == 4) {
    v4d *r0 = rw[0], *r1 = rw[1], *r2 = rw[2], *r3 = rw[3];
    for (; j + 1 < i0; j += 2) {
      const v4d *l0 = l + off(j), *l1 = l + off(j + 1);
      v4d a0 = splat(0.0), a1 = a0, b0 = a0, b1 = a0, c0 = a0, c1 = a0, d0 = a0, d1 = a0;
      for (int k = 0; k < j; ++k) {
        v4d p = l0[k], q = l1[k], x;
        x = r0[k]; a0 += x * p; a1 += x * q;
        x = r1[k]; b0 += x * p; b1 += x * q;
        x = r2[k]; c0 += x * p; c1 += x * q;
        x = r3[k]; d0 += x * p; d1 += x * q;
      }
      v4d iv0 = inv[j], iv1 = inv[j + 1], q1j = l1[j];
      r0[j] = (P(lam, Q, D, i0, j) - a0) * iv0; a1 += r0[j] * q1j; r0[j + 1] = (P(lam, Q, D, i0, j + 1) - a1) * iv1;
      r1[j] = (P(lam, Q, D, i0 + 1, j) - b0) * iv0; b1 += r1[j] * q1j; r1[j + 1] = (P(lam, Q, D, i0 + 1, j + 1) - b1) * iv1;
      r2[j] = (P(lam, Q, D, i0 + 2, j) - c0) * iv0; c1 += r2[j] * q1j; r2[j + 1] = (P(lam, Q, D, i0 + 2, j + 1) - c1) * iv1;
      r3[j] = (P(lam, Q, D, i0 + 3, j) - d0) * iv0; d1 += r3[j] * q1j; r3[j + 1] = (P(lam, Q, D, i0 + 3, j + 1) - d1) * iv1;
    }
  }
  for (; j < i0; ++j) {
    const v4d* l0 = l + off(j);
    for (int a = 0; a < R; ++a) rw[a][j] = (P(lam, Q, D, i0 + a, j) - dot(rw[a], l0, j)) * inv[j];
  }
  for (int a = 0; a < R; ++a) {
    int i = i0 + a;
    for (int jj = i0; jj < i; ++jj) rw[a][jj] = (P(lam, Q, D, i, jj) - dot(rw[a], l + off(jj), jj)) * inv[jj];
    v4d d = vsqrt(P(lam, Q, D, i, i) - dot(rw[a], rw[a], i));
    rw[a][i] = d; inv[i] = splat(1.0) / d; prod *= d;
  }
}

inline void chol(int D, const double* lam, const double* Q, v4d* l, v4d* inv, v4d& prod) {
  prod = splat(1.0);
  int i = 0;
  for (; i + 3 < D; i += 4) chol_rows(D, lam, Q, l, inv, prod, i, 4);
  if (i < D) chol_rows(D, lam, Q, l, inv, prod, i, D - i);
}

// One pass over Lam per tile: r = s - Lam delta (delta may be null), the factor, the solves.
// lfac: packed factors per 4-group tile [n/4][tri_size(D)][4]; yres [D, n]: y = L^-1 r, both for
// the adjoint
inline void forward(int D, int64_t g0, int64_t g1, int64_t n, const double* Lam, const double* Q, const double* s,
                    const double* delta, const double* zeta, double* u, double* ldet, double* lfac, double* yres) {
  const size_t T = tri_size(D);
  std::vector<v4d> buf(5 * size_t(D));
  v4d *y = buf.data(), *uu = y + D, *inv = uu + D, *r = inv + D, *dl = r + D, prod;
  for (int64_t g = g0; g < g1; g += 4) {
    const double* lam = Lam + g * D * D;
    v4d* l = reinterpret_cast<v4d*>(lfac + (g / 4) * T * 4);
    for (int i = 0; i < D; ++i) r[i] = ldn(s + i * n, n, g, 0.0);
    if (delta) {
      for (int b = 0; b < D; ++b) dl[b] = ldn(delta + b * n, n, g, 0.0);
      for (int a = 0; a < D; ++a) {
        const double* row = lam + size_t(a) * D * 4;
        v4d acc0 = splat(0.0), acc1 = acc0;
        int b = 0;
        for (; b + 1 < D; b += 2) { acc0 += ld(row + b * 4) * dl[b]; acc1 += ld(row + (b + 1) * 4) * dl[b + 1]; }
        for (; b < D; ++b) acc0 += ld(row + b * 4) * dl[b];
        r[a] -= acc0 + acc1;
      }
    }
    chol(D, lam, Q, l, inv, prod);
    for (int i = 0; i < D; ++i) y[i] = (r[i] - dot(l + off(i), y, i)) * inv[i];
    for (int i = 0; i < D; ++i) { stn(yres + i * n, n, g, y[i]); uu[i] = y[i] + ldn(zeta + i * n, n, g, 0.0); }
    for (int i = D - 1; i >= 0; --i) {
      v4d ui = uu[i] * inv[i];
      uu[i] = ui;
      const v4d* li = l + off(i);
      for (int k = 0; k < i; ++k) uu[k] -= li[k] * ui;
    }
    for (int i = 0; i < D; ++i) stn(u + i * n, n, g, uu[i]);
    stn(ldet, n, g, vlog(prod));
  }
}

// From the forward pass's lfac, yres and u: zetabar, deltabar (when non-null; -Lam rbar, with
// Lam = L L^T - Q so that Lam is not read again) and qsum (lower triangle of Qbar, summed over
// this range)
inline void backward(int D, int64_t g0, int64_t g1, int64_t n, const double* lfac, const double* Q,
                     const double* yres, const double* ures, const double* ubar, const double* ldbar, double* zetabar,
                     double* deltabar, double* qsum) {
  const size_t T = tri_size(D);
  std::vector<v4d> buf(2 * T + 9 * size_t(D));
  v4d *qb = buf.data(), *lb = qb + T, *y = lb + T, *ug = y + D, *t = ug + D, *s = t + D, *inv = s + D, *yb = inv + D,
      *TT = yb + D, *UU = TT + D, *w = UU + D;
  for (size_t i = 0; i < T; ++i) qb[i] = splat(0.0);
  for (int64_t g = g0; g < g1; g += 4) {
    const v4d* l = reinterpret_cast<const v4d*>(lfac + (g / 4) * T * 4);
    for (int i = 0; i < D; ++i) {
      inv[i] = splat(1.0) / l[off(i) + i];
      y[i] = ldn(yres + i * n, n, g, 0.0);
      ug[i] = ldn(ures + i * n, n, g, 0.0);
    }
    // reverse of the back solve: t_i = (ubar_i - sum_{k<i} L_ik t_k) / L_ii
    for (int i = 0; i < D; ++i) {
      t[i] = (ldn(ubar + i * n, n, g, 0.0) - dot(l + off(i), t, i)) * inv[i];
      stn(zetabar + i * n, n, g, t[i]);
    }
    // reverse of the forward solve: s_i = (t_i - sum_{k>i} L_ki s_k) / L_ii
    for (int i = 0; i < D; ++i) yb[i] = t[i];
    for (int i = D - 1; i >= 0; --i) {
      v4d si = yb[i] * inv[i];
      s[i] = si;
      const v4d* li = l + off(i);
      for (int k = 0; k < i; ++k) yb[k] -= li[k] * si;
    }
    if (deltabar) {
      for (int k = 0; k < D; ++k) w[k] = splat(0.0);
      for (int i = 0; i < D; ++i) {
        const v4d* li = l + off(i);
        for (int k = 0; k <= i; ++k) w[k] += li[k] * s[i];
      }
      for (int i = 0; i < D; ++i) {
        v4d acc = -dot(l + off(i), w, i + 1);
        for (int k = 0; k < D; ++k) acc += splat(Q[i >= k ? i * D + k : k * D + i]) * s[k];
        stn(deltabar + i * n, n, g, acc);
      }
    }
    v4d lbar = ldn(ldbar, n, g, 0.0);
    for (int i = 0; i < D; ++i) {
      v4d* lbi = lb + off(i);
      v4d ti = t[i], si = s[i];
      for (int k = 0; k < i; ++k) lbi[k] = -(ug[i] * t[k] + si * y[k]);
      lbi[i] = lbar * inv[i] - ug[i] * ti - si * y[i];
    }
    // reverse of the Cholesky recurrence, two columns (j, j - 1) per sweep over the rows below
    int j = D - 1;
    for (; j >= 1; j -= 2) {
      const int m = j - 1;
      v4d *lbj = lb + off(j), *lbm = lb + off(m);
      const v4d *lj = l + off(j), *lm = l + off(m);
      v4d ivj = inv[j], ivm = inv[m], dj = lbj[j];
      for (int i = j + 1; i < D; ++i) {
        v4d ti = lb[off(i) + j] * ivj;
        TT[i] = ti;
        dj -= ti * l[off(i) + j];
        qb[off(i) + j] += ti;
      }
      v4d db = dj * ivj * splat(0.5);
      qb[off(j) + j] += db;
      v4d cj = lbj[m] - splat(2.0) * db * lj[m];
      for (int i = j + 1; i < D; ++i) { cj -= TT[i] * l[off(i) + m]; lb[off(i) + m] -= TT[i] * lj[m]; }
      lbj[m] = cj;
      v4d tj = cj * ivm, dm = lbm[m] - tj * lj[m];
      qb[off(j) + m] += tj;
      for (int i = j + 1; i < D; ++i) {
        v4d ui = lb[off(i) + m] * ivm;
        UU[i] = ui;
        dm -= ui * l[off(i) + m];
        qb[off(i) + m] += ui;
      }
      v4d dbm = dm * ivm * splat(0.5);
      qb[off(m) + m] += dbm;
      v4d db2 = splat(2.0) * db, dbm2 = splat(2.0) * dbm;
      for (int k = 0; k < m; ++k) {
        v4d c = lj[k], cm = lm[k];
        lbj[k] -= db2 * c + tj * cm;
        lbm[k] -= tj * c + dbm2 * cm;
      }
      int i = j + 1;
      for (; i + 3 < D; i += 4) {
        v4d *b0 = lb + off(i), *b1 = lb + off(i + 1), *b2 = lb + off(i + 2), *b3 = lb + off(i + 3);
        const v4d *x0 = l + off(i), *x1 = l + off(i + 1), *x2 = l + off(i + 2), *x3 = l + off(i + 3);
        v4d t0 = TT[i], t1 = TT[i + 1], t2 = TT[i + 2], t3 = TT[i + 3];
        v4d u0 = UU[i], u1 = UU[i + 1], u2 = UU[i + 2], u3 = UU[i + 3];
        for (int k = 0; k < m; ++k) {
          v4d c = lj[k], cm = lm[k], y0 = x0[k], y1 = x1[k], y2 = x2[k], y3 = x3[k];
          lbj[k] -= (t0 * y0 + t1 * y1) + (t2 * y2 + t3 * y3);
          lbm[k] -= (u0 * y0 + u1 * y1) + (u2 * y2 + u3 * y3);
          b0[k] -= t0 * c + u0 * cm; b1[k] -= t1 * c + u1 * cm;
          b2[k] -= t2 * c + u2 * cm; b3[k] -= t3 * c + u3 * cm;
        }
      }
      for (; i < D; ++i) {
        v4d* bi = lb + off(i);
        const v4d* xi = l + off(i);
        v4d ti = TT[i], ui = UU[i];
        for (int k = 0; k < m; ++k) {
          v4d y = xi[k];
          lbj[k] -= ti * y; lbm[k] -= ui * y; bi[k] -= ti * lj[k] + ui * lm[k];
        }
      }
    }
    if (j == 0) {
      v4d iv0 = inv[0], d0 = lb[0];
      for (int i = 1; i < D; ++i) {
        v4d ti = lb[off(i)] * iv0;
        d0 -= ti * l[off(i)];
        qb[off(i)] += ti;
      }
      qb[0] += d0 * iv0 * splat(0.5);
    }
  }
  for (int i = 0; i < D; ++i)
    for (int j = 0; j < D; ++j) {
      v4d q = j <= i ? qb[off(i) + j] : splat(0.0);
      qsum[i * D + j] = q[0] + q[1] + q[2] + q[3];
    }
}

}  // namespace tri

}  // namespace tmap
