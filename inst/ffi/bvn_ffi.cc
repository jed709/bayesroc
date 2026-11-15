#include <algorithm>
#include <limits>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <xmmintrin.h>
#endif
#include <atomic>
#include <memory>
#include <thread>
#include <vector>
#include <type_traits>

#include "xla/ffi/api/ffi.h"
#include "bayesroc/bvn_simd.h"
#include "bayesroc/pcq.h"
#include "bayesroc/lnoise.h"
#include "bayesroc/mafc.h"

static inline bvn::v4d vphi_p(bvn::v4d x);
#include "bayesroc/tmap_kernel.h"

namespace ffi = xla::ffi;

namespace {

template <typename F>
void pool_for(ffi::ThreadPool& pool, int64_t n, int64_t grain, F body) {
  int64_t threads = pool.num_threads();
  int64_t nchunks = grain > 0 ? n / grain : 0;
  if (threads <= 1 || nchunks <= 1) {
    body(int64_t(0), n);
    return;
  }
  struct State {
    std::atomic<int64_t> next{0}, done{0};
    int64_t n, chunk, nchunks;
    F body;
    State(int64_t n_, int64_t c_, int64_t k_, F b) : n(n_), chunk(c_), nchunks(k_), body(std::move(b)) {}
    void work() {
      for (int64_t c; (c = next.fetch_add(1, std::memory_order_relaxed)) < nchunks;) {
        int64_t i0 = c * chunk;
        body(i0, std::min(n, i0 + chunk));
        done.fetch_add(1, std::memory_order_release);
      }
    }
  };
  int64_t chunk = ((n + nchunks - 1) / nchunks + 3) & ~int64_t(3);
  nchunks = (n + chunk - 1) / chunk;
  auto st = std::make_shared<State>(n, chunk, nchunks, std::move(body));
  for (int64_t k = 1; k < std::min(threads, nchunks); ++k) pool.Schedule([st] { st->work(); });
  st->work();
  while (st->done.load(std::memory_order_acquire) < nchunks) std::this_thread::yield();
}

// ---- transport map: fused per-group factor / solves and their adjoint ----

template <typename T>
struct DBuf {
  std::vector<double> own;
  const double* ptr;
  int64_t n;
  explicit DBuf(const ffi::AnyBuffer& b) : n(b.element_count()) {
    if constexpr (std::is_same<T, double>::value) {
      ptr = static_cast<const double*>(b.untyped_data());
    } else {
      own.resize(n);
      const T* src = static_cast<const T*>(b.untyped_data());
      for (int64_t k = 0; k < n; ++k) own[k] = src[k];
      ptr = own.data();
    }
  }
};

template <typename T>
struct DOut {
  std::vector<double> own;
  double* ptr;
  T* dst;
  int64_t n;
  explicit DOut(ffi::Result<ffi::AnyBuffer>& b) : n(b->element_count()) {
    dst = static_cast<T*>(b->untyped_data());
    if constexpr (std::is_same<T, double>::value) {
      ptr = reinterpret_cast<double*>(dst);
    } else {
      own.resize(n);
      ptr = own.data();
    }
  }
  ~DOut() {
    if constexpr (!std::is_same<T, double>::value)
      for (int64_t k = 0; k < n; ++k) dst[k] = static_cast<T>(own[k]);
  }
};

constexpr int kTmapChunks = 8;

// groups are split into at most kTmapChunks ranges of whole 4-group tiles, each at least
// kTmapGrain / D^3 tiles (a tile's work grows as D^3)
constexpr int64_t kTmapGrain = 64 * 16 * 16 * 16;

template <typename F>
void tmap_for(ffi::ThreadPool& pool, int64_t n, int D, F body) {
  int64_t tiles = (n + 3) / 4, d3 = int64_t(D) * D * D;
  int64_t per = std::max<int64_t>((kTmapGrain + d3 - 1) / d3, (tiles + kTmapChunks - 1) / kTmapChunks);
  int64_t nchunks = (tiles + per - 1) / per;
  int64_t threads = std::min<int64_t>(pool.num_threads(), nchunks);
  if (threads <= 1) {
    for (int64_t c = 0; c < nchunks; ++c) body(c, c * per * 4, std::min(n, (c + 1) * per * 4));
    return;
  }
  struct State {
    std::atomic<int64_t> next{0}, done{0};
    int64_t n, per, nchunks;
    F body;
    State(int64_t n_, int64_t p_, int64_t k_, F b) : n(n_), per(p_), nchunks(k_), body(std::move(b)) {}
    void work() {
      for (int64_t c; (c = next.fetch_add(1, std::memory_order_relaxed)) < nchunks;) {
        body(c, c * per * 4, std::min(n, (c + 1) * per * 4));
        done.fetch_add(1, std::memory_order_release);
      }
    }
  };
  auto st = std::make_shared<State>(n, per, nchunks, std::move(body));
  for (int64_t k = 1; k < threads; ++k) pool.Schedule([st] { st->work(); });
  st->work();
  while (st->done.load(std::memory_order_acquire) < nchunks) std::this_thread::yield();
}

// Lam [n4, D, D, 4], Q [D, D], s, delta, zeta [D, n] (delta may be empty: no shift) -> u [D, n],
// ld [n], and for D >= kTriMin what the adjoint reads instead of Lam: the packed factors
// lfac [n4 / 4, D (D + 1) / 2 * 4] and yres = L^-1 r [D, n] (unwritten below kTriMin)
template <typename T>
ffi::Error TmapFwdImpl(ffi::ThreadPool pool, ffi::AnyBuffer Lam, ffi::AnyBuffer Q, ffi::AnyBuffer s,
                       ffi::AnyBuffer delta, ffi::AnyBuffer zeta, ffi::Result<ffi::AnyBuffer> u,
                       ffi::Result<ffi::AnyBuffer> ld, ffi::Result<ffi::AnyBuffer> lfac,
                       ffi::Result<ffi::AnyBuffer> yres) {
  const int D = static_cast<int>(Lam.dimensions()[1]);
  const int64_t n = s.dimensions()[1];
  const bool shift = delta.element_count() > 0;
  DBuf<T> lam(Lam), q(Q), ss(s), dd(delta), z(zeta);
  DOut<T> uo(u), ldo(ld), lo(lfac), yo(yres);
  const double *pl = lam.ptr, *pq = q.ptr, *ps = ss.ptr, *pd = shift ? dd.ptr : nullptr, *pz = z.ptr;
  double *pu = uo.ptr, *pld = ldo.ptr, *plf = lo.ptr, *py = yo.ptr;
  if (D >= tmap::kTriMin) {
    tmap_for(pool, n, D, [=](int64_t, int64_t g0, int64_t g1) {
      tmap::tri::forward(D, g0, g1, n, pl, pq, ps, pd, pz, pu, pld, plf, py);
    });
    return ffi::Error::Success();
  }
  tmap::FwdFn f = tmap::forward_for(D);
  std::vector<double> r(ps, ps + size_t(D) * n);
  double* pr = r.data();
  tmap_for(pool, n, D, [=](int64_t, int64_t g0, int64_t g1) {
    if (pd) tmap::coupling(D, n, g0, g1, pl, pd, pr);
    f(g0, g1, n, pl, pq, pr, pz, pu, pld);
  });
  return ffi::Error::Success();
}

// Lam, Q, s, delta, zeta as above, ubar [D, n], ldbar [n], and the forward pass's lfac, yres, u ->
// zetabar [D, n], deltabar [D, n] (unwritten when delta is empty), Qbar [kTmapChunks, D, D]
// (lower-triangle cotangent of Q, per chunk; sum over chunks)
template <typename T>
ffi::Error TmapBwdImpl(ffi::ThreadPool pool, ffi::AnyBuffer Lam, ffi::AnyBuffer Q, ffi::AnyBuffer s,
                       ffi::AnyBuffer delta, ffi::AnyBuffer zeta, ffi::AnyBuffer ubar,
                       ffi::AnyBuffer ldbar, ffi::AnyBuffer lfac, ffi::AnyBuffer yres, ffi::AnyBuffer ures,
                       ffi::Result<ffi::AnyBuffer> zetabar, ffi::Result<ffi::AnyBuffer> deltabar,
                       ffi::Result<ffi::AnyBuffer> Qbar) {
  const int D = static_cast<int>(Lam.dimensions()[1]);
  const int64_t n = s.dimensions()[1];
  const bool shift = delta.element_count() > 0;
  DBuf<T> q(Q), ub(ubar), lb(ldbar);
  DOut<T> zo(zetabar), dob(deltabar), qo(Qbar);
  for (int64_t k = 0; k < int64_t(kTmapChunks) * D * D; ++k) qo.ptr[k] = 0.0;
  const double *pq = q.ptr, *pub = ub.ptr, *plb = lb.ptr;
  double *pzb = zo.ptr, *pdb = shift ? dob.ptr : nullptr, *pqb = qo.ptr;
  if (D >= tmap::kTriMin) {
    DBuf<T> lf(lfac), yr(yres), ur(ures);
    const double *plf = lf.ptr, *py = yr.ptr, *pu = ur.ptr;
    tmap_for(pool, n, D, [=](int64_t c, int64_t g0, int64_t g1) {
      tmap::tri::backward(D, g0, g1, n, plf, pq, py, pu, pub, plb, pzb, pdb, pqb + c * D * D);
    });
    return ffi::Error::Success();
  }
  tmap::BwdFn f = tmap::backward_for(D);
  DBuf<T> lam(Lam), ss(s), dd(delta), z(zeta);
  const double *pl = lam.ptr, *pd = shift ? dd.ptr : nullptr, *pz = z.ptr;
  std::vector<double> r(ss.ptr, ss.ptr + size_t(D) * n), rb(size_t(D) * n);
  double *pr = r.data(), *prb = rb.data();
  tmap_for(pool, n, D, [=](int64_t c, int64_t g0, int64_t g1) {
    if (pd) tmap::coupling(D, n, g0, g1, pl, pd, pr);
    f(g0, g1, n, pl, pq, pr, pz, pub, plb, pzb, prb, pqb + c * D * D);
    if (pdb) {
      tmap::coupling_adjoint(D, n, g0, g1, pl, prb);
      for (int a = 0; a < D; ++a)
        for (int64_t g = g0; g < g1; ++g) pdb[a * n + g] = prb[a * n + g];
    }
  });
  return ffi::Error::Success();
}



// One wide block's whole map step, for D >= kTriMin. Natural order in and out: z [D, n] the
// sampler's coordinates, A [D, D] the prior factor (lower triangular), perm [D] (int32) the
// kernel's order (row i of the kernel is natural row perm[i]); Lam, s, delta as in TmapFwdImpl, in
// the kernel's order. -> th = A^-1 u [D, n] and u [D, n] (natural), ld [n], and for the adjoint
// lfac, yres, uk (u in the kernel's order), Qk = (A^-T A^-1) in the kernel's order [D, D] and
// Ainv = A^-1 [D, D].
template <typename T>
ffi::Error TmapBlkFwdImpl(ffi::ThreadPool pool, ffi::AnyBuffer Lam, ffi::AnyBuffer s, ffi::AnyBuffer delta,
                          ffi::AnyBuffer z, ffi::AnyBuffer Afac, ffi::AnyBuffer perm,
                          ffi::Result<ffi::AnyBuffer> th, ffi::Result<ffi::AnyBuffer> ld,
                          ffi::Result<ffi::AnyBuffer> un, ffi::Result<ffi::AnyBuffer> lfac,
                          ffi::Result<ffi::AnyBuffer> yres, ffi::Result<ffi::AnyBuffer> uk,
                          ffi::Result<ffi::AnyBuffer> Qk, ffi::Result<ffi::AnyBuffer> Ainv) {
  const int D = static_cast<int>(Lam.dimensions()[1]);
  const int64_t n = s.dimensions()[1];
  const bool shift = delta.element_count() > 0;
  const int32_t* pm = static_cast<const int32_t*>(perm.untyped_data());
  DBuf<T> lam(Lam), ss(s), dd(delta), zz(z), af(Afac);
  DOut<T> tho(th), ldo(ld), uno(un), lo(lfac), yo(yres), uko(uk), qo(Qk), aio(Ainv);
  double* Ai = aio.ptr;
  const double* F = af.ptr;
  for (int i = 0; i < D; ++i) {
    double* ri = Ai + size_t(i) * D;
    for (int c = 0; c < D; ++c) ri[c] = c == i ? 1.0 : 0.0;
    for (int k = 0; k < i; ++k) {
      const double f = F[i * D + k];
      const double* rk = Ai + size_t(k) * D;
      for (int c = 0; c <= k; ++c) ri[c] -= f * rk[c];
    }
    const double inv = 1.0 / F[i * D + i];
    for (int c = 0; c <= i; ++c) ri[c] *= inv;
  }
  const double* A = Ai;
  double* q = qo.ptr;
  for (int i = 0; i < D; ++i)
    for (int j = 0; j <= i; ++j) {
      const int b = pm[i], c = pm[j];
      double acc = 0.0;
      for (int a = std::max(b, c); a < D; ++a) acc += A[a * D + b] * A[a * D + c];
      q[i * D + j] = acc;
      q[j * D + i] = acc;
    }
  std::vector<double> zk(size_t(D) * n);
  for (int i = 0; i < D; ++i) std::copy(zz.ptr + size_t(pm[i]) * n, zz.ptr + size_t(pm[i] + 1) * n, zk.data() + size_t(i) * n);
  const double *pl = lam.ptr, *ps = ss.ptr, *pd = shift ? dd.ptr : nullptr, *pz = zk.data(), *pq = q;
  double *pth = tho.ptr, *pld = ldo.ptr, *pun = uno.ptr, *plf = lo.ptr, *py = yo.ptr, *puk = uko.ptr;
  tmap_for(pool, n, D, [=](int64_t, int64_t g0, int64_t g1) {
    tmap::tri::forward(D, g0, g1, n, pl, pq, ps, pd, pz, puk, pld, plf, py);
    std::vector<tmap::v4d> u(D);
    for (int64_t g = g0; g < g1; g += 4) {
      for (int i = 0; i < D; ++i) {
        u[pm[i]] = tmap::ldn(puk + size_t(i) * n, n, g, 0.0);
        tmap::stn(pun + size_t(pm[i]) * n, n, g, u[pm[i]]);
      }
      for (int a = 0; a < D; ++a) {
        tmap::v4d acc = tmap::splat(0.0);
        for (int b = 0; b <= a; ++b) acc += tmap::splat(A[a * D + b]) * u[b];
        tmap::stn(pth + size_t(a) * n, n, g, acc);
      }
    }
  });
  return ffi::Error::Success();
}

// The adjoint of TmapBlkFwdImpl: Ainv, perm, Qk, lfac, yres, uk, u from the forward pass, the
// cotangents thbar, ldbar, ubar, and delta (only its size is read) -> zbar [D, n] (natural),
// deltabar [D, n] (kernel order; unwritten when delta is empty), Abar [D, D] (lower triangle).
template <typename T>
ffi::Error TmapBlkBwdImpl(ffi::ThreadPool pool, ffi::AnyBuffer Ainv, ffi::AnyBuffer perm, ffi::AnyBuffer Qk,
                          ffi::AnyBuffer lfac, ffi::AnyBuffer yres, ffi::AnyBuffer uk, ffi::AnyBuffer un,
                          ffi::AnyBuffer thbar, ffi::AnyBuffer ldbar, ffi::AnyBuffer unbar, ffi::AnyBuffer delta,
                          ffi::Result<ffi::AnyBuffer> zbar, ffi::Result<ffi::AnyBuffer> deltabar,
                          ffi::Result<ffi::AnyBuffer> Abar) {
  const int D = static_cast<int>(Qk.dimensions()[0]);
  const int64_t n = thbar.dimensions()[1];
  const bool shift = delta.element_count() > 0;
  const int32_t* pm = static_cast<const int32_t*>(perm.untyped_data());
  DBuf<T> ai(Ainv), q(Qk), lf(lfac), yr(yres), ukb(uk), unb(un), tb(thbar), lb(ldbar), ubn(unbar);
  DOut<T> zo(zbar), dob(deltabar), ao(Abar);
  const double *A = ai.ptr, *pq = q.ptr, *plf = lf.ptr, *py = yr.ptr, *puk = ukb.ptr, *pun = unb.ptr,
               *ptb = tb.ptr, *plb = lb.ptr, *pubn = ubn.ptr;
  const size_t DD = size_t(D) * D;
  std::vector<double> ubk(size_t(D) * n), zbk(size_t(D) * n), qb(kTmapChunks * DD, 0.0), ab(kTmapChunks * DD, 0.0);
  double *pubk = ubk.data(), *pzbk = zbk.data(), *pqb = qb.data(), *pab = ab.data(), *pdb = shift ? dob.ptr : nullptr;
  tmap_for(pool, n, D, [=](int64_t c, int64_t g0, int64_t g1) {
    std::vector<tmap::v4d> t(D), u(D), abt(DD, tmap::splat(0.0));
    for (int64_t g = g0; g < g1; g += 4) {
      for (int a = 0; a < D; ++a) {
        t[a] = tmap::ldn(ptb + size_t(a) * n, n, g, 0.0);
        u[a] = tmap::ldn(pun + size_t(a) * n, n, g, 0.0);
      }
      for (int a = 0; a < D; ++a)
        for (int b = 0; b <= a; ++b) abt[size_t(a) * D + b] += t[a] * u[b];
      for (int i = 0; i < D; ++i) {
        const int b = pm[i];
        tmap::v4d acc = tmap::ldn(pubn + size_t(b) * n, n, g, 0.0);
        for (int a = b; a < D; ++a) acc += tmap::splat(A[a * D + b]) * t[a];
        tmap::stn(pubk + size_t(i) * n, n, g, acc);
      }
    }
    double* abc = pab + c * DD;
    for (size_t k = 0; k < DD; ++k) abc[k] = abt[k][0] + abt[k][1] + abt[k][2] + abt[k][3];
    tmap::tri::backward(D, g0, g1, n, plf, pq, py, puk, pubk, plb, pzbk, pdb, pqb + c * DD);
  });
  double* pz = zo.ptr;
  for (int i = 0; i < D; ++i) std::copy(pzbk + size_t(i) * n, pzbk + size_t(i + 1) * n, pz + size_t(pm[i]) * n);
  std::vector<double> G(DD, 0.0);
  for (int c = 1; c < kTmapChunks; ++c)
    for (size_t k = 0; k < DD; ++k) { pqb[k] += pqb[c * DD + k]; pab[k] += pab[c * DD + k]; }
  for (int i = 0; i < D; ++i)
    for (int j = 0; j <= i; ++j) {
      G[size_t(pm[i]) * D + pm[j]] += pqb[size_t(i) * D + j];
      G[size_t(pm[j]) * D + pm[i]] += pqb[size_t(i) * D + j];
    }
  std::vector<double> Gi(DD, 0.0), W(DD, 0.0);
  for (int a = 0; a < D; ++a)
    for (int b = 0; b <= a; ++b) {
      double acc = pab[size_t(a) * D + b];
      for (int k = 0; k <= a; ++k) acc += A[a * D + k] * G[size_t(k) * D + b];
      Gi[size_t(a) * D + b] = acc;
    }
  for (int a = 0; a < D; ++a)
    for (int c = 0; c < D; ++c) {
      const int m = std::min(a, c);
      double acc = 0.0;
      for (int b = 0; b <= m; ++b) acc += Gi[size_t(a) * D + b] * A[size_t(c) * D + b];
      W[size_t(a) * D + c] = acc;
    }
  double* pa = ao.ptr;
  for (size_t k = 0; k < DD; ++k) pa[k] = 0.0;
  for (int a = 0; a < D; ++a)
    for (int r = 0; r <= a; ++r) {
      const double f = A[a * D + r];
      double* pr = pa + size_t(r) * D;
      const double* wa = W.data() + size_t(a) * D;
      for (int c = 0; c <= r; ++c) pr[c] -= f * wa[c];
    }
  return ffi::Error::Success();
}


ffi::Error ProbitCellImpl(ffi::ThreadPool pool, ffi::Buffer<ffi::F64> zlo, ffi::Buffer<ffi::F64> zhi,
                          ffi::ResultBuffer<ffi::F64> lp, ffi::ResultBuffer<ffi::F64> glo,
                          ffi::ResultBuffer<ffi::F64> ghi) {
  const double *l = zlo.typed_data(), *h = zhi.typed_data();
  double *a = lp->typed_data(), *b = glo->typed_data(), *c = ghi->typed_data();
  pool_for(pool, zlo.element_count(), 8192,
           [=](int64_t i0, int64_t i1) { bvn::probit_cell_range(i0, i1, l, h, a, b, c); });
  return ffi::Error::Success();
}

void probit_cell_k(int64_t n, const double* const* i, double* const* o) {
  bvn::probit_cell_range(0, n, i[0], i[1], o[0], o[1], o[2]);
}

void probit_diff_k(int64_t n, const double* const* i, double* const* o) {
  bvn::probit_diff_range(0, n, i[0], i[1], o[0], o[1], o[2]);
}

void rect_k(int64_t n, const double* const* i, double* const* o) {
  bvn::rect_array(n, i[0], i[1], i[2], i[3], i[4], o[0], o[1], o[2], o[3], o[4], o[5]);
}

// z1_lo, z1_hi, w1, z2_lo, z2_hi, w2, zs_lo, zs_hi, ws, rho, z_cp, is_new [N] -> narrow, P, p_det, D_s, A_s, B_s,
// As, Bs, gA1, gB1, g_zcp, Ar2, Br2, drho [N]: bounded bivariate cell at unit weight
inline double strip_scale(double t_lo, double v, double a_lo, double a_hi, bool hl, bool hh, double rho) {
  const double r = std::fmin(std::fmax(rho, -0.999999), 0.999999), s = std::sqrt((1.0 - r) * (1.0 + r));
  const double m = t_lo + 0.5 * v;
  const double zl = hl ? std::fabs(a_lo - r * m) / s : 1e300, zh = hh ? std::fabs(a_hi - r * m) / s : 1e300;
  const double zmin = (hl || hh) ? std::fmin(zl, zh) : 0.0;
  return v * (std::fmax(1.0, std::fabs(m)) + std::fabs(r) / s * std::fmax(1.0, zmin));
}

inline bool probit_narrow(double z_lo, double w, double u) { return w * std::fmax(1.0, std::fabs(z_lo + 0.5 * w)) < u; }

template <int NARROW_EXP>
void bbv_k(int64_t n, const double* const* i, double* const* o) {
  const double narrow_u = NARROW_EXP == 5 ? 1e-5 : 0.05, strip_u = NARROW_EXP == 5 ? 1e-3 : 0.05;
  const double H = 5e9;
  thread_local std::vector<double> buf;
  if (buf.size() < size_t(13 * n)) buf.resize(13 * n);
  double *ze = buf.data(), *Dd = ze + n, *pdl = Dd + n, *pdh = pdl + n, *Ds = pdh + n, *psl = Ds + n, *psh = psl + n;
  double *rp = psh + n, *r1l = rp + n, *r1h = r1l + n, *r2l = r1h + n, *r2h = r2l + n, *rr = r2h + n;
  for (int64_t k = 0; k < n; ++k) {
    const double zc = i[10][k];
    ze[k] = i[11][k] != 0.0 ? i[0][k] : std::fmin(std::fmax(zc, i[0][k]), i[1][k]);
  }
  bvn::probit_diff_range(0, n, i[0], ze, Dd, pdl, pdh);
  bvn::probit_diff_range(0, n, i[6], i[7], Ds, psl, psh);
  bvn::rect_array(n, ze, i[1], i[3], i[4], i[9], rp, r1l, r1h, r2l, r2h, rr);
  for (int64_t k = 0; k < n; ++k) {
    const double z1l = i[0][k], z1h = i[1][k], w1 = i[2][k], z2l = i[3][k], z2h = i[4][k], w2 = i[5][k];
    const double zsl = i[6][k], ws = i[8][k], rho = i[9][k], zc = i[10][k], zce = ze[k];
    const bool is_new = i[11][k] != 0.0;
    const bool h1l = z1l > -H, h1h = z1h < H, h2l = z2l > -H, h2h = z2h < H, hsl = h2l, hsh = h2h;
    const bool split = !is_new && zc > z1l && zc < z1h, above_only = is_new || zc <= z1l;
    const bool below_only = !is_new && !above_only && zc >= z1h;
    const bool det_hi = zce < H, in_d = h1l && det_hi, in_s = hsl && hsh, in2 = h2l && h2h;
    const bool ehl = zce > -H, in_ab = ehl && h1h;
    const bool sc_floor = !(std::sqrt(std::fmax(1.0 - rho * rho, 1e-20)) > 1e-6);
    const double Ad0 = in_d ? pdh[k] - pdl[k] : (h1l ? -pdl[k] : pdh[k]), Bd0 = in_d ? pdh[k] : 0.0;
    const double As0 = in_s ? psh[k] - psl[k] : (hsl ? -psl[k] : psh[k]), Bs0 = in_s ? psh[k] : 0.0;
    const double p_det = above_only ? 0.0 : std::fmax(Dd[k], 0.0), p_src = std::fmax(Ds[k], 0.0);
    const bool above_on = rp[k] > 0.0 && !below_only;
    const double p_above = above_on ? rp[k] : 0.0;
    const double det_w = (Dd[k] > 0.0 && !above_only) ? p_src : 0.0;
    const double src_w = Ds[k] > 0.0 ? p_det : 0.0;
    const double Ad = det_w != 0.0 ? Ad0 * det_w : 0.0, Bd = det_w != 0.0 ? Bd0 * det_w : 0.0;
    const double Ar1 = above_on ? r1l[k] + r1h[k] : 0.0, Br1 = above_on && in_ab ? r1h[k] : 0.0;
    const double dd_lo = in_d ? Ad - Bd : (h1l ? Ad : 0.0), dd_hi = in_d ? Bd : (h1l ? 0.0 : Ad);
    const double dr_lo = in_ab ? Ar1 - Br1 : (ehl ? Ar1 : 0.0), dr_hi = in_ab ? Br1 : (ehl ? 0.0 : Ar1);
    const bool cont = split && h1l && in_ab && !sc_floor;
    double nar = 0.0;
    if (!above_only && in_d) {
      const double wd = below_only ? w1 : zce - z1l;
      if (probit_narrow(z1l, wd, narrow_u)) nar = 1.0;
    }
    if (in_s && probit_narrow(zsl, ws, narrow_u)) nar = 1.0;
    if (!below_only) {
      const double wab = above_only ? w1 : z1h - zce;
      const double v1 = in_ab ? wab : 1e-3, v2 = in2 ? w2 : 1e-3;
      const bool n2 = in2 && strip_scale(in2 ? z2l : 0.0, v2, zce, z1h, ehl, h1h, rho) < strip_u;
      const bool n1 = in_ab && !n2 && strip_scale(in_ab ? zce : 0.0, v1, z2l, z2h, h2l, h2h, rho) < strip_u;
      if (n1 || n2) nar = 1.0;
    }
    o[0][k] = nar;
    o[1][k] = p_det * p_src + p_above;
    o[2][k] = p_det;
    o[3][k] = Ds[k];
    o[4][k] = As0;
    o[5][k] = Bs0;
    o[6][k] = src_w != 0.0 ? As0 * src_w : 0.0;
    o[7][k] = src_w != 0.0 ? Bs0 * src_w : 0.0;
    o[8][k] = cont ? Ad + Ar1 : (split ? dd_lo + dr_hi : (below_only ? Ad : Ar1));
    o[9][k] = split ? dr_hi : (below_only ? Bd : Br1);
    o[10][k] = (split && sc_floor) ? dd_hi + dr_lo : 0.0;
    o[11][k] = above_on ? r2l[k] + r2h[k] : 0.0;
    o[12][k] = above_on && in2 ? r2h[k] : 0.0;
    o[13][k] = above_on ? rr[k] : 0.0;
  }
}

ffi::Error ProbitDiffImpl(ffi::ThreadPool pool, ffi::Buffer<ffi::F64> zlo, ffi::Buffer<ffi::F64> zhi,
                          ffi::ResultBuffer<ffi::F64> D, ffi::ResultBuffer<ffi::F64> pl,
                          ffi::ResultBuffer<ffi::F64> ph) {
  const double *l = zlo.typed_data(), *h = zhi.typed_data();
  double *a = D->typed_data(), *b = pl->typed_data(), *c = ph->typed_data();
  pool_for(pool, zlo.element_count(), 8192,
           [=](int64_t i0, int64_t i1) { bvn::probit_diff_range(i0, i1, l, h, a, b, c); });
  return ffi::Error::Success();
}

void cdp_k(int64_t n, const double* const* i, double* const* o) {
  for (int64_t k = 0; k < n; ++k) {
    if (i[0][k] == 0.0) {
      for (int j = 0; j < 10; ++j) o[j][k] = 0.0;
      continue;
    }
    bvn_tail::CdpCell c = bvn_tail::cdp_cell(static_cast<int>(i[1][k]), i[2][k], i[3][k], i[4][k], i[5][k],
                                             i[6][k], i[7][k], i[8][k], i[9][k], i[10][k]);
    o[0][k] = c.lp;
    for (int j = 0; j < 9; ++j) o[1 + j][k] = c.d[j];
  }
}

struct FlushDenormals {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  unsigned saved;
  FlushDenormals() : saved(_mm_getcsr()) { _mm_setcsr(saved | 0x8040); }
  ~FlushDenormals() { _mm_setcsr(saved); }
#elif defined(__aarch64__)
  uint64_t saved;
  FlushDenormals() {
    __asm__ volatile("mrs %0, fpcr" : "=r"(saved));
    uint64_t flushed = saved | (uint64_t(1) << 24);
    __asm__ volatile("msr fpcr, %0" : : "r"(flushed));
  }
  ~FlushDenormals() { __asm__ volatile("msr fpcr, %0" : : "r"(saved)); }
#endif
};

template <int NI, int NO, void (*K)(int64_t, const double* const*, double* const*)>
ffi::Error F64Impl(ffi::ThreadPool pool, ffi::RemainingArgs args, ffi::RemainingRets rets) {
  if (args.size() != NI || rets.size() != NO) return ffi::Error::InvalidArgument("bayesroc: f64 arity");
  const double* in[NI];
  double* out[NO];
  int64_t n = 0;
  for (int a = 0; a < NI; ++a) {
    auto b = args.get<ffi::AnyBuffer>(a);
    if (b.has_error()) return b.error();
    in[a] = static_cast<const double*>(b.value().untyped_data());
    n = b.value().element_count();
  }
  for (int a = 0; a < NO; ++a) {
    auto b = rets.get<ffi::AnyBuffer>(a);
    if (b.has_error()) return b.error();
    out[a] = static_cast<double*>(b.value()->untyped_data());
  }
  pool_for(pool, n, 256, [&](int64_t i0, int64_t i1) {
    FlushDenormals ftz;
    const double* ci[NI];
    double* co[NO];
    for (int a = 0; a < NI; ++a) ci[a] = in[a] + i0;
    for (int a = 0; a < NO; ++a) co[a] = out[a] + i0;
    K(i1 - i0, ci, co);
  });
  return ffi::Error::Success();
}

template <int NI, int NO, void (*K)(int64_t, const double* const*, double* const*)>
ffi::Error F32Impl(ffi::ThreadPool pool, ffi::RemainingArgs args, ffi::RemainingRets rets) {
  if (args.size() != NI || rets.size() != NO) return ffi::Error::InvalidArgument("bayesroc: f32 arity");
  const float* fin[NI];
  float* fout[NO];
  int64_t n = 0;
  for (int a = 0; a < NI; ++a) {
    auto b = args.get<ffi::AnyBuffer>(a);
    if (b.has_error()) return b.error();
    fin[a] = static_cast<const float*>(b.value().untyped_data());
    n = b.value().element_count();
  }
  for (int a = 0; a < NO; ++a) {
    auto b = rets.get<ffi::AnyBuffer>(a);
    if (b.has_error()) return b.error();
    fout[a] = static_cast<float*>(b.value()->untyped_data());
  }
  pool_for(pool, n, 256, [&](int64_t i0, int64_t i1) {
    const int64_t m = i1 - i0;
    thread_local std::vector<double> buf;
    if (buf.size() < size_t((NI + NO) * m)) buf.resize((NI + NO) * m);
    FlushDenormals ftz;
    const double* din[NI];
    double* dout[NO];
    for (int a = 0; a < NI; ++a) {
      double* d = buf.data() + a * m;
      for (int64_t k = 0; k < m; ++k) d[k] = fin[a][i0 + k];
      din[a] = d;
    }
    for (int a = 0; a < NO; ++a) dout[a] = buf.data() + (NI + a) * m;
    K(m, din, dout);
    for (int a = 0; a < NO; ++a)
      for (int64_t k = 0; k < m; ++k) fout[a][i0 + k] = static_cast<float>(dout[a][k]);
  });
  return ffi::Error::Success();
}

template <bool SP>
void ordinal_chunk(int64_t n, int G, const double* const* in, double* const* out) {
  const double *y = in[0], *mu = in[1], *sg = in[2], *mid = in[3];
  const int MI = G / 2, NU = G - MI, NV = SP ? 2 * G : G;
  thread_local std::vector<double> buf;
  if (buf.size() < size_t((4 + NV) * n + G + 1)) buf.resize((4 + NV) * n + G + 1);
  double *zlo = buf.data(), *zhi = zlo + n, *glo = zhi + n, *ghi = glo + n, *ev = ghi + n, *dv = SP ? ev + size_t(G) * n : ev;
  double* t = ev + size_t(NV) * n;
  for (int g = 0; g < G; ++g) {
    const double* x = in[4 + g];
    double *e = ev + size_t(g) * n, *d = dv + size_t(g) * n;
    if (SP) {
      for (int64_t k = 0; k < n; ++k) {
        const double a = std::exp(-std::fabs(x[k]));
        e[k] = std::max(x[k], 0.0) + std::log1p(a);
        d[k] = x[k] >= 0.0 ? 1.0 / (1.0 + a) : a / (1.0 + a);
      }
    } else {
      int64_t k = 0;
      bvn::v4d v;
      for (; k + 4 <= n; k += 4) { std::memcpy(&v, x + k, 32); v = bvn::vexp(v); std::memcpy(e + k, &v, 32); }
      for (; k < n; ++k) e[k] = std::exp(x[k]);
    }
  }
  for (int64_t k = 0; k < n; ++k) {
    const int yy = int(y[k]) - 1;
    t[MI] = mid[k];
    double c = 0.0;
    for (int j = 0; j < NU; ++j) { c += ev[size_t(j) * n + k]; t[MI + 1 + j] = mid[k] + c; }
    c = 0.0;
    for (int j = 0; j < MI; ++j) { c += ev[size_t(NU + j) * n + k]; t[MI - 1 - j] = mid[k] - c; }
    zlo[k] = yy > 0 ? (t[yy - 1] - mu[k]) / sg[k] : -1e10;
    zhi[k] = yy < G + 1 ? (t[yy] - mu[k]) / sg[k] : 1e10;
  }
  bvn::probit_cell_range(0, n, zlo, zhi, out[0], glo, ghi);
  const double lfl = std::log(1e-300);
  for (int64_t k = 0; k < n; ++k) {
    const int yy = int(y[k]) - 1;
    const bool lo = yy > 0, hi = yy < G + 1, inner = lo && hi;
    const int own = inner ? (yy - 1 >= MI ? yy - 1 - MI : NU + MI - yy) : 0;
    const double s = sg[k];
    const double w = inner ? ev[size_t(own) * n + k] / s : 1.0;
    double gA = (hi ? ghi[k] : 0.0) + (lo ? glo[k] : 0.0), gB = inner ? ghi[k] : 0.0;
    if (inner && w * std::max(1.0, std::fabs(zlo[k] + 0.5 * w)) < 0.05) {
      const double m = zlo[k] + 0.5 * w, m2 = m * m, w2 = w * w, u2 = (w * m) * (w * m);
      const double ser = (u2 - w2) / 24.0 + (u2 * (u2 - 6.0 * w2) + 3.0 * w2 * w2) / 1920.0 +
                         (u2 * (u2 * (u2 - 15.0 * w2) + 45.0 * w2 * w2) - 15.0 * w2 * w2 * w2) / 322560.0;
      const double lpn = std::log(w) - 0.5 * m2 - 0.9189385332046727 + std::log1p(ser);
      const double kk = std::exp(-0.125 * w2) / (w * (1.0 + ser));
      const bool kept = lpn > lfl;
      out[0][k] = kept ? lpn : lfl;
      gA = kept ? -2.0 * kk * std::sinh(0.5 * m * w) : 0.0;
      gB = kept ? kk * std::exp(-0.5 * m * w) : 0.0;
    }
    const double za = lo ? zlo[k] : zhi[k];
    const int ka = lo ? yy - 1 : yy;
    const bool wide = inner && w > 0.5 * std::max(1.0, std::fabs(za));
    const double sc = wide ? (gA - gB) * za + gB * zhi[k] : gA * za + (inner ? gB * w : 0.0);
    out[1][k] = -gA / s;
    out[2][k] = -sc / s;
    out[3][k] = gA / s;
    for (int g = 0; g < G; ++g) {
      const double mk = g < NU ? (ka >= MI + 1 + g ? 1.0 : 0.0) : (ka < MI - (g - NU) ? -1.0 : 0.0);
      const double dg = SP ? dv[size_t(g) * n + k] : ev[size_t(g) * n + k];
      out[4 + g][k] = (mk * gA + (inner && g == own ? gB : 0.0)) / s * dg;
    }
  }
}

template <bool SP, bool F32>
ffi::Error OrdinalImpl(ffi::ThreadPool pool, ffi::RemainingArgs args, ffi::RemainingRets rets) {
  const int NI = int(args.size());
  if (NI < 4 || int(rets.size()) != NI) return ffi::Error::InvalidArgument("bayesroc: ordinal arity");
  const int G = NI - 4;
  std::vector<const void*> in(NI);
  std::vector<void*> out(NI);
  int64_t n = 0;
  for (int a = 0; a < NI; ++a) {
    auto b = args.get<ffi::AnyBuffer>(a);
    if (b.has_error()) return b.error();
    in[a] = b.value().untyped_data();
    n = b.value().element_count();
    auto r = rets.get<ffi::AnyBuffer>(a);
    if (r.has_error()) return r.error();
    out[a] = r.value()->untyped_data();
  }
  pool_for(pool, n, 256, [&](int64_t i0, int64_t i1) {
    FlushDenormals ftz;
    const int64_t m = i1 - i0;
    std::vector<const double*> ci(NI);
    std::vector<double*> co(NI);
    if (!F32) {
      for (int a = 0; a < NI; ++a) {
        ci[a] = static_cast<const double*>(in[a]) + i0;
        co[a] = static_cast<double*>(out[a]) + i0;
      }
      ordinal_chunk<SP>(m, G, ci.data(), co.data());
      return;
    }
    thread_local std::vector<double> cv;
    if (cv.size() < size_t(2 * NI) * m) cv.resize(size_t(2 * NI) * m);
    for (int a = 0; a < NI; ++a) {
      double* d = cv.data() + size_t(a) * m;
      const float* f = static_cast<const float*>(in[a]) + i0;
      for (int64_t k = 0; k < m; ++k) d[k] = f[k];
      ci[a] = d;
      co[a] = cv.data() + size_t(NI + a) * m;
    }
    ordinal_chunk<SP>(m, G, ci.data(), co.data());
    for (int a = 0; a < NI; ++a) {
      float* f = static_cast<float*>(out[a]) + i0;
      for (int64_t k = 0; k < m; ++k) f[k] = static_cast<float>(co[a][k]);
    }
  });
  return ffi::Error::Success();
}

template <bool SP>
void metad_chunk(int64_t n, int G, const double* const* in, double* const* out) {
  const double *y = in[0], *d = in[1], *mo = in[2], *mn = in[3], *a = in[4], *mid = in[5], *q = in[6], *rel = in[7];
  const int MI = G / 2, NU = G - MI, NV = SP ? 2 * G : G, half = (G + 2) / 2;
  const double OPEN = 1e10;
  thread_local std::vector<double> buf;
  const size_t need = size_t(19 + NV) * n + G + 1;
  if (buf.size() < need) buf.resize(need);
  double* b = buf.data();
  double *za = b, *zl = za + n, *zh = zl + n, *zc = zh + n, *side = zc + n, *tail = side + n, *ee = tail + n, *wf = ee + n;
  double *lp = wf + n, *ga = lp + n, *gl = ga + n, *gh = gl + n, *gc = gh + n, *gs = gc + n;
  double *lb = gs + n, *rA = lb + n, *rB = rA + n, *flag = rB + n, *wv = flag + n;
  double *ev = wv + n, *dv = SP ? ev + size_t(G) * n : ev;
  double* fc = ev + size_t(NV) * n;
  for (int g = 0; g < G; ++g) {
    const double* x = in[8 + g];
    double *e = ev + size_t(g) * n, *dd = dv + size_t(g) * n;
    if (SP) {
      for (int64_t k = 0; k < n; ++k) {
        const double t = std::exp(-std::fabs(x[k]));
        e[k] = std::max(x[k], 0.0) + std::log1p(t);
        dd[k] = x[k] >= 0.0 ? 1.0 / (1.0 + t) : t / (1.0 + t);
      }
    } else {
      int64_t k = 0;
      bvn::v4d v;
      for (; k + 4 <= n; k += 4) { std::memcpy(&v, x + k, 32); v = bvn::vexp(v); std::memcpy(e + k, &v, 32); }
      for (; k < n; ++k) e[k] = std::exp(x[k]);
    }
  }
  for (int64_t k = 0; k < n; ++k) {
    const int yy = int(y[k]) - 1;
    fc[MI] = 0.0;
    double c = 0.0;
    for (int j = 0; j < NU; ++j) { c += ev[size_t(j) * n + k]; fc[MI + 1 + j] = c; }
    c = 0.0;
    for (int j = 0; j < MI; ++j) { c += ev[size_t(NU + j) * n + k]; fc[MI - 1 - j] = -c; }
    const bool old = yy + 1 > half, lo = yy > 0, hi = yy < G + 1, inner = lo && hi;
    const double M = old ? mo[k] : mn[k], ac = a[k] - q[k];
    const double mu_m = q[k] * d[k] + M * d[k] * ac, c_mid = mid[k] - q[k] * d[k];
    const double off = (rel[k] != 0.0 ? (M - 1.0) * c_mid : 0.0) - mu_m;
    const int k_hi = std::min(std::max(yy, 0), G), k_lo = std::min(std::max(yy - 1, 0), G);
    const double z_lo = lo ? (mid[k] + fc[k_lo]) + off : -OPEN, z_hi = hi ? (mid[k] + fc[k_hi]) + off : OPEN;
    const int own = inner ? (yy - 1 >= MI ? yy - 1 - MI : NU + MI - yy) : 0;
    const double w = inner ? ev[size_t(own) * n + k] : 1.0;
    const double ls = inner ? z_lo : 0.0;
    const double e = old ? fc[k_lo] : -fc[k_hi];
    const double z_c = mid[k] + off;
    const double m0 = ls + 0.5 * w;
    const bool narrow = inner && w * std::max(1.0, std::fabs(m0)) < 0.05;
    const bool tl = (old ? z_c : -z_c) > 8.0 && e >= 0.0;
    const bool sel = narrow && !tl;
    if (narrow) {
      const double m2 = m0 * m0, w2 = w * w, u2 = (w * m0) * (w * m0);
      const double s = (u2 - w2) / 24.0 + (u2 * (u2 - 6.0 * w2) + 3.0 * w2 * w2) / 1920.0 +
                       (u2 * (u2 * (u2 - 15.0 * w2) + 45.0 * w2 * w2) - 15.0 * w2 * w2 * w2) / 322560.0;
      lb[k] = std::log(w) - 0.5 * m2 - 0.9189385332046727 + std::log1p(s);
      const double kk = std::exp(-0.125 * w2) / (w * (1.0 + s));
      rA[k] = -2.0 * kk * std::sinh(0.5 * m0 * w);
      rB[k] = kk * std::exp(-0.5 * m0 * w);
    } else {
      lb[k] = 0.0; rA[k] = 0.0; rB[k] = 0.0;
    }
    za[k] = mid[k] - a[k] * d[k];
    zl[k] = sel ? -OPEN : z_lo;
    zh[k] = sel ? OPEN : z_hi;
    zc[k] = z_c;
    side[k] = old ? 1.0 : 0.0;
    tail[k] = tl ? 1.0 : 0.0;
    ee[k] = e;
    wf[k] = inner ? w : OPEN;
    wv[k] = w;
    flag[k] = (narrow ? 1.0 : 0.0) + (sel ? 2.0 : 0.0);
  }
  const double* ci[8] = {za, zl, zh, zc, side, tail, ee, wf};
  double* co[6] = {lp, ga, gl, gh, gc, gs};
  bvn::metad_array(n, ci, co);
  for (int64_t k = 0; k < n; ++k) {
    const int yy = int(y[k]) - 1;
    const bool old = yy + 1 > half, lo = yy > 0, hi = yy < G + 1, inner = lo && hi;
    const bool narrow = flag[k] == 1.0 || flag[k] == 3.0, sel = flag[k] >= 2.0;
    const double M = old ? mo[k] : mn[k], ac = a[k] - q[k], c_mid = mid[k] - q[k] * d[k];
    const double g_lo = lo ? gl[k] : 0.0, g_hi = hi ? gh[k] : 0.0;
    const double gA = narrow ? rA[k] : g_lo + g_hi, gB = narrow ? rB[k] : (inner ? g_hi : 0.0);
    const double Gs = sel ? rA[k] + gs[k] : gs[k];
    out[0][k] = sel ? lp[k] + lb[k] : lp[k];
    double d_mid, d_d, d_M;
    if (rel[k] != 0.0) {
      d_mid = ga[k] + Gs * M;
      d_d = -a[k] * ga[k] - Gs * M * a[k];
      d_M = Gs * (c_mid - d[k] * ac);
    } else {
      d_mid = ga[k] + Gs;
      d_d = -a[k] * ga[k] - Gs * (q[k] + M * ac);
      d_M = -Gs * d[k] * ac;
    }
    out[1][k] = d_d;
    out[2][k] = old ? d_M : 0.0;
    out[3][k] = old ? 0.0 : d_M;
    out[4][k] = d_mid;
    const int k_hi = std::min(std::max(yy, 0), G), k_lo = std::min(std::max(yy - 1, 0), G);
    const int ka = lo ? k_lo : k_hi;
    const int own = inner ? (yy - 1 >= MI ? yy - 1 - MI : NU + MI - yy) : -1;
    for (int g = 0; g < G; ++g) {
      const double mk = g < NU ? (ka >= MI + 1 + g ? 1.0 : 0.0) : (ka < MI - (g - NU) ? -1.0 : 0.0);
      const double dg = SP ? dv[size_t(g) * n + k] : ev[size_t(g) * n + k];
      out[5 + g][k] = (mk * gA + (g == own ? gB : 0.0)) * dg;
    }
  }
}

template <bool SP>
inline void gap_values(int64_t n, const double* x, double* e, double* d) {
  if (SP) {
    for (int64_t k = 0; k < n; ++k) {
      const double t = std::exp(-std::fabs(x[k]));
      e[k] = std::max(x[k], 0.0) + std::log1p(t);
      d[k] = x[k] >= 0.0 ? 1.0 / (1.0 + t) : t / (1.0 + t);
    }
    return;
  }
  int64_t k = 0;
  bvn::v4d v;
  for (; k + 4 <= n; k += 4) { std::memcpy(&v, x + k, 32); v = bvn::vexp(v); std::memcpy(e + k, &v, 32); }
  for (; k < n; ++k) e[k] = std::exp(x[k]);
}

inline void rect_strip(double t_lo, double v, double a_lo, double a_hi, double wa, bool hl, bool hh, double rho,
                       double* o) {
  using bvn::v4d;
  using bvn::splat;
  const double OPEN = 1e10;
  const double r = std::min(std::max(rho, -0.999999), 0.999999), s = std::sqrt((1.0 - r) * (1.0 + r));
  const v4d x = {0.5 * (1.0 - 0.8611363115940526), 0.5 * (1.0 - 0.3399810435848563),
                 0.5 * (1.0 + 0.3399810435848563), 0.5 * (1.0 + 0.8611363115940526)};
  const v4d W = {0.5 * 0.3478548451374538, 0.5 * 0.6521451548625461, 0.5 * 0.6521451548625461,
                 0.5 * 0.3478548451374538};
  const v4d t = splat(t_lo) + splat(v) * x;
  const v4d pt = bvn::vexp(splat(-0.5) * t * t - splat(0.9189385332046727));
  const v4d zlo = hl ? (splat(a_lo) - splat(r) * t) / splat(s) : splat(-OPEN);
  const v4d zhi = hh ? (splat(a_hi) - splat(r) * t) / splat(s) : splat(OPEN);
  const double om = wa / s;
  const bool inner = hl && hh;
  v4d Pl, dl, ql, Ph, dh, qh;
  bvn::vprobit_triple(zhi, Ph, dh, qh);
  bvn::vprobit_triple(zlo, Pl, dl, ql);
  v4d C = bvn::vprobit_band(zlo, zhi, Pl, Ph, ql, qh, dl, dh);
  v4d Ac, Bc;
  for (int l = 0; l < 4; ++l) {
    double A = inner ? dh[l] - dl[l] : (hl ? -dl[l] : dh[l]), B = inner ? dh[l] : 0.0;
    if (inner) {
      const double m0 = zlo[l] + 0.5 * om;
      if (om * std::max(1.0, std::fabs(m0)) < 0.05) {
        const double m2 = m0 * m0, w2 = om * om, u2 = (om * m0) * (om * m0);
        const double ser = (u2 - w2) / 24.0 + (u2 * (u2 - 6.0 * w2) + 3.0 * w2 * w2) / 1920.0 +
                           (u2 * (u2 * (u2 - 15.0 * w2) + 45.0 * w2 * w2) - 15.0 * w2 * w2 * w2) / 322560.0;
        const double lpn = std::log(om) - 0.5 * m2 - 0.9189385332046727 + std::log1p(ser);
        const double kk = std::exp(-0.125 * w2) / (om * (1.0 + ser));
        const double Dn = std::exp(lpn);
        C[l] = Dn;
        A = -2.0 * kk * std::sinh(0.5 * m0 * om) * Dn;
        B = kk * std::exp(-0.5 * m0 * om) * Dn;
      }
    }
    Ac[l] = A;
    Bc[l] = B;
  }
  const v4d za = hl ? zlo : zhi;
  const v4d g = pt * C, gp = pt * (-t * C - splat(r / s) * Ac);
  const v4d dza = (-t + za * splat(r / s)) / splat(s);
  const double dom = om * (r / (s * s));
  const v4d q0 = W * g, q1 = W * gp, q2 = W * x * gp, q3 = W * pt * Ac, q4 = W * pt * Bc,
            q5 = W * pt * (Ac * dza + Bc * splat(dom));
  auto sum4 = [](v4d q) { return ((q[0] + q[1]) + q[2]) + q[3]; };
  const double Sg = sum4(q0);
  o[0] = v * Sg;
  o[1] = v * sum4(q1);
  o[2] = Sg + v * sum4(q2);
  o[3] = v * sum4(q3) / s;
  o[4] = v * sum4(q4) / s;
  o[5] = std::fabs(rho) < 0.999999 ? v * sum4(q5) : 0.0;
}

struct DimEdges {
  double z_lo, z_hi, w, z_a;
  int k_a, own;
  bool lo, hi, inner;
};

inline DimEdges dim_edges(int yy, int G, double mid, double mu, double sg, const double* ev, int64_t n, int64_t k,
                          double* t) {
  const int MI = G / 2, NU = G - MI;
  t[MI] = mid;
  double c = 0.0;
  for (int j = 0; j < NU; ++j) { c += ev[size_t(j) * n + k]; t[MI + 1 + j] = mid + c; }
  c = 0.0;
  for (int j = 0; j < MI; ++j) { c += ev[size_t(NU + j) * n + k]; t[MI - 1 - j] = mid - c; }
  DimEdges e;
  e.lo = yy > 0;
  e.hi = yy < G + 1;
  e.inner = e.lo && e.hi;
  const int k_hi = std::min(std::max(yy, 0), G), k_lo = std::min(std::max(yy - 1, 0), G);
  e.z_hi = e.hi ? (t[k_hi] - mu) / sg : 1e10;
  e.z_lo = e.lo ? (t[k_lo] - mu) / sg : -1e10;
  e.own = e.inner ? (yy - 1 >= MI ? yy - 1 - MI : NU + MI - yy) : -1;
  e.w = G > 0 ? ev[size_t(e.inner ? e.own : 0) * n + k] / sg : 1.0;
  e.z_a = e.lo ? e.z_lo : e.z_hi;
  e.k_a = e.lo ? k_lo : k_hi;
  return e;
}

inline double lw_scale(double gA, double gB, double z_a, double w, bool inner, double z_b) {
  const double b = inner ? gB : 0.0, wi = inner ? w : 0.0;
  const bool wide = inner && wi > 0.5 * std::max(1.0, std::fabs(z_a));
  return wide ? (gA - b) * z_a + b * z_b : gA * z_a + b * wi;
}

inline double gap_sign(int g, int G, int ka) {
  const int MI = G / 2, NU = G - MI;
  return g < NU ? (ka >= MI + 1 + g ? 1.0 : 0.0) : (ka < MI - (g - NU) ? -1.0 : 0.0);
}

template <bool SP>
void bg_chunk(int64_t n, int GT, const double* const* in, double* const* out) {
  const double *y1 = in[0], *y2 = in[1], *mu1 = in[2], *mu2 = in[3], *s1 = in[4], *s2 = in[5], *rho = in[6];
  const double *mid1 = in[7], *mid2 = in[8];
  const int G1 = n > 0 ? int(in[9][0]) : 0, G2 = GT - G1, NV = SP ? 2 : 1;
  thread_local std::vector<double> buf;
  const size_t need = size_t(12 + NV * GT) * n + GT + 4;
  if (buf.size() < need) buf.resize(need);
  double* b = buf.data();
  double *a1 = b, *b1 = a1 + n, *a2 = b1 + n, *b2 = a2 + n, *w1 = b2 + n, *w2 = w1 + n;
  double *p = w2 + n, *d1l = p + n, *d1h = d1l + n, *d2l = d1h + n, *d2h = d2l + n, *dr = d2h + n;
  double *ev1 = dr + n, *ev2 = ev1 + size_t(G1) * n;
  double *dv1 = SP ? ev2 + size_t(G2) * n : ev1, *dv2 = SP ? dv1 + size_t(G1) * n : ev2;
  double *t1 = ev1 + size_t(NV * GT) * n, *t2 = t1 + G1 + 2;
  for (int g = 0; g < G1; ++g) gap_values<SP>(n, in[10 + g], ev1 + size_t(g) * n, dv1 + size_t(g) * n);
  for (int g = 0; g < G2; ++g) gap_values<SP>(n, in[10 + G1 + g], ev2 + size_t(g) * n, dv2 + size_t(g) * n);
  for (int64_t k = 0; k < n; ++k) {
    const DimEdges e1 = dim_edges(int(y1[k]) - 1, G1, mid1[k], mu1[k], s1[k], ev1, n, k, t1);
    const DimEdges e2 = dim_edges(int(y2[k]) - 1, G2, mid2[k], mu2[k], s2[k], ev2, n, k, t2);
    a1[k] = e1.z_lo; b1[k] = e1.z_hi; w1[k] = e1.w;
    a2[k] = e2.z_lo; b2[k] = e2.z_hi; w2[k] = e2.w;
  }
  bvn::rect_array(n, a1, b1, a2, b2, rho, p, d1l, d1h, d2l, d2h, dr);
  for (int64_t k = 0; k < n; ++k) {
    const int yy1 = int(y1[k]) - 1, yy2 = int(y2[k]) - 1;
    const bool lo1 = yy1 > 0, hi1 = yy1 < G1 + 1, in1 = lo1 && hi1, lo2 = yy2 > 0, hi2 = yy2 < G2 + 1, in2 = lo2 && hi2;
    const bool f1l = a1[k] > -5e9, f1h = b1[k] < 5e9, f2l = a2[k] > -5e9, f2h = b2[k] < 5e9;
    const bool r1 = f1l && f1h, r2 = f2l && f2h;
    double P = p[k], A1 = d1l[k] + d1h[k], B1 = r1 ? d1h[k] : 0.0, A2 = d2l[k] + d2h[k], B2 = r2 ? d2h[k] : 0.0;
    double R = dr[k];
    const double v1 = r1 ? w1[k] : 1e-3, v2 = r2 ? w2[k] : 1e-3, l1 = r1 ? a1[k] : 0.0, l2 = r2 ? a2[k] : 0.0;
    bool n2 = false, n1 = false;
    if (r1 || r2) {
      const double rr = std::min(std::max(rho[k], -0.999999), 0.999999), is = 1.0 / std::sqrt((1.0 - rr) * (1.0 + rr));
      const double ars = std::fabs(rr) * is;
      auto scale = [&](double t_lo, double v, double a_lo, double a_hi, bool hl, bool hh) {
        const double m = t_lo + 0.5 * v;
        const double zl = hl ? std::fabs(a_lo - rr * m) * is : 1e300, zh = hh ? std::fabs(a_hi - rr * m) * is : 1e300;
        const double zmin = (hl || hh) ? std::min(zl, zh) : 0.0;
        return v * (std::max(1.0, std::fabs(m)) + ars * std::max(1.0, zmin));
      };
      n2 = r2 && scale(l2, v2, a1[k], b1[k], f1l, f1h) < 0.05;
      n1 = r1 && !n2 && scale(l1, v1, a2[k], b2[k], f2l, f2h) < 0.05;
    }
    if (n2) {
      double o[6];
      rect_strip(l2, v2, a1[k], b1[k], r1 ? w1[k] : 1.0, f1l, f1h, rho[k], o);
      P = o[0]; A1 = o[3]; B1 = o[4]; A2 = o[1]; B2 = o[2]; R = o[5];
    } else if (n1) {
      double o[6];
      rect_strip(l1, v1, a2[k], b2[k], r2 ? w2[k] : 1.0, f2l, f2h, rho[k], o);
      P = o[0]; A1 = o[1]; B1 = o[2]; A2 = o[3]; B2 = o[4]; R = o[5];
    }
    const double prob = std::max(P, 1e-300), inv_P = P > 1e-300 ? 1.0 / prob : 0.0;
    const double is1 = 1.0 / s1[k], is2 = 1.0 / s2[k];
    const double gA1 = A1 * inv_P, gB1 = B1 * inv_P, gA2 = A2 * inv_P, gB2 = B2 * inv_P;
    const double za1 = lo1 ? a1[k] : b1[k], za2 = lo2 ? a2[k] : b2[k];
    out[0][k] = prob;
    out[1][k] = -gA1 * is1;
    out[2][k] = -gA2 * is2;
    out[3][k] = -lw_scale(gA1, gB1, za1, w1[k], in1, b1[k]) * is1;
    out[4][k] = -lw_scale(gA2, gB2, za2, w2[k], in2, b2[k]) * is2;
    out[5][k] = R * inv_P;
    out[6][k] = gA1 * is1;
    out[7][k] = gA2 * is2;
    const int MI1 = G1 / 2, NU1 = G1 - MI1, MI2 = G2 / 2, NU2 = G2 - MI2;
    const int ka1 = lo1 ? std::min(std::max(yy1 - 1, 0), G1) : std::min(std::max(yy1, 0), G1);
    const int ka2 = lo2 ? std::min(std::max(yy2 - 1, 0), G2) : std::min(std::max(yy2, 0), G2);
    const int own1 = in1 ? (yy1 - 1 >= MI1 ? yy1 - 1 - MI1 : NU1 + MI1 - yy1) : -1;
    const int own2 = in2 ? (yy2 - 1 >= MI2 ? yy2 - 1 - MI2 : NU2 + MI2 - yy2) : -1;
    for (int g = 0; g < G1; ++g)
      out[8 + g][k] = (gap_sign(g, G1, ka1) * (gA1 * is1) + (g == own1 ? gB1 * is1 : 0.0)) * dv1[size_t(g) * n + k];
    for (int g = 0; g < G2; ++g)
      out[8 + G1 + g][k] = (gap_sign(g, G2, ka2) * (gA2 * is2) + (g == own2 ? gB2 * is2 : 0.0)) * dv2[size_t(g) * n + k];
  }
  {
    int64_t k = 0;
    bvn::v4d v;
    for (; k + 4 <= n; k += 4) { std::memcpy(&v, out[0] + k, 32); v = bvn::vlog(v); std::memcpy(out[0] + k, &v, 32); }
    for (; k < n; ++k) out[0][k] = std::log(out[0][k]);
  }
}

inline void build_thresh(int G, double mid, const double* ev, int64_t n, int64_t k, double* t) {
  const int MI = G / 2, NU = G - MI;
  t[MI] = mid;
  double c = 0.0;
  for (int j = 0; j < NU; ++j) { c += ev[size_t(j) * n + k]; t[MI + 1 + j] = mid + c; }
  c = 0.0;
  for (int j = 0; j < MI; ++j) { c += ev[size_t(NU + j) * n + k]; t[MI - 1 - j] = mid - c; }
}

inline DimEdges edges_at(const double* t, int yy, int G, double mu, double sg, const double* ev, int64_t n,
                         int64_t k) {
  const int MI = G / 2, NU = G - MI;
  DimEdges e;
  e.lo = yy > 0;
  e.hi = yy < G + 1;
  e.inner = e.lo && e.hi;
  const int k_hi = std::min(std::max(yy, 0), G), k_lo = std::min(std::max(yy - 1, 0), G);
  e.z_hi = e.hi ? (t[k_hi] - mu) / sg : 1e10;
  e.z_lo = e.lo ? (t[k_lo] - mu) / sg : -1e10;
  e.own = e.inner ? (yy - 1 >= MI ? yy - 1 - MI : NU + MI - yy) : -1;
  e.w = G > 0 ? ev[size_t(e.inner ? e.own : 0) * n + k] / sg : 1.0;
  e.z_a = e.lo ? e.z_lo : e.z_hi;
  e.k_a = e.lo ? k_lo : k_hi;
  return e;
}

inline bool cell_narrow(const DimEdges& e) {
  return e.inner && e.w * std::max(1.0, std::fabs(e.z_lo + 0.5 * e.w)) < 0.05;
}

template <bool SP>
void vrdp_chunk(int64_t n, int GT, const double* const* in, double* const* out) {
  const double *y1 = in[0], *y2 = in[1], *dF = in[2], *dR = in[3], *sd = in[4], *lam = in[5], *si = in[6], *sS = in[7];
  const double *mid1 = in[8], *mid2 = in[9];
  const int G1 = n > 0 ? int(in[10][0]) : 0, G2 = GT - G1, NV = SP ? 2 : 1;
  thread_local std::vector<double> buf;
  const size_t need = size_t(19 + NV * GT) * n + GT + 4;
  if (buf.size() < need) buf.resize(need);
  double* b = buf.data();
  double *c1h = b, *c1l = c1h + n, *c2h = c1l + n, *c2l = c2h + n, *h1h = c2l + n, *h1l = h1h + n, *h2h = h1l + n;
  double *h2l = h2h + n, *o7 = h2l + n, *o8 = o7 + n, *o9 = o8 + n, *o10 = o9 + n, *T1B = o10 + n, *T2B = T1B + n;
  double *T1A = T2B + n, *T2A = T1A + n, *ka1 = T2A + n, *ka2 = ka1 + n, *nar = ka2 + n;
  double *ev1 = nar + n, *ev2 = ev1 + size_t(G1) * n;
  double *dv1 = SP ? ev2 + size_t(G2) * n : ev1, *dv2 = SP ? dv1 + size_t(G1) * n : ev2;
  double *t1 = ev1 + size_t(NV * GT) * n, *t2 = t1 + G1 + 2;
  for (int g = 0; g < G1; ++g) gap_values<SP>(n, in[11 + g], ev1 + size_t(g) * n, dv1 + size_t(g) * n);
  for (int g = 0; g < G2; ++g) gap_values<SP>(n, in[11 + G1 + g], ev2 + size_t(g) * n, dv2 + size_t(g) * n);
  for (int64_t k = 0; k < n; ++k) {
    const int yy1 = int(y1[k]) - 1, yy2 = int(y2[k]) - 1;
    build_thresh(G1, mid1[k], ev1, n, k, t1);
    build_thresh(G2, mid2[k], ev2, n, k, t2);
    c1h[k] = t1[std::min(std::max(yy1, 0), G1)];
    c1l[k] = t1[std::min(std::max(yy1 - 1, 0), G1)];
    c2h[k] = t2[std::min(std::max(yy2, 0), G2)];
    c2l[k] = t2[std::min(std::max(yy2 - 1, 0), G2)];
    h1h[k] = yy1 < G1 + 1 ? 1.0 : 0.0;
    h1l[k] = yy1 > 0 ? 1.0 : 0.0;
    h2h[k] = yy2 < G2 + 1 ? 1.0 : 0.0;
    h2l[k] = yy2 > 0 ? 1.0 : 0.0;
  }
  const double* ci[14] = {c1h, c1l, c2h, c2l, h1h, h1l, h2h, h2l, dF, dR, sd, lam, si, sS};
  double* co[11] = {out[0], out[1], out[2], out[3], out[4], out[5], out[6], o7, o8, o9, o10};
  bvn::vrdp_array(n, ci, co);
  for (int64_t k = 0; k < n; ++k) {
    const int yy1 = int(y1[k]) - 1, yy2 = int(y2[k]) - 1;
    const bool lo1 = yy1 > 0, in1 = lo1 && yy1 < G1 + 1, lo2 = yy2 > 0, in2 = lo2 && yy2 < G2 + 1;
    ka1[k] = lo1 ? std::min(std::max(yy1 - 1, 0), G1) : std::min(std::max(yy1, 0), G1);
    ka2[k] = lo2 ? std::min(std::max(yy2 - 1, 0), G2) : std::min(std::max(yy2, 0), G2);
    const int MI1 = G1 / 2, NU1 = G1 - MI1, MI2 = G2 / 2, NU2 = G2 - MI2;
    const double g1w = in1 ? ev1[size_t(yy1 - 1 >= MI1 ? yy1 - 1 - MI1 : NU1 + MI1 - yy1) * n + k] : 1e300;
    const double g2w = in2 ? ev2[size_t(yy2 - 1 >= MI2 ? yy2 - 1 - MI2 : NU2 + MI2 - yy2) * n + k] : 1e300;
    if (!(g1w < 0.05 * si[k] || g2w < 0.05 || g2w < 0.05 * sS[k])) {
      T1A[k] = o7[k] + o8[k];
      T1B[k] = in1 ? o7[k] : 0.0;
      T2A[k] = o9[k] + o10[k];
      T2B[k] = in2 ? o9[k] : 0.0;
      continue;
    }
    build_thresh(G1, mid1[k], ev1, n, k, t1);
    build_thresh(G2, mid2[k], ev2, n, k, t2);
    const DimEdges f1 = edges_at(t1, yy1, G1, dF[k], si[k], ev1, n, k);
    const DimEdges r1 = edges_at(t1, yy1, G1, dF[k] + dR[k], si[k], ev1, n, k);
    const DimEdges n2 = edges_at(t2, yy2, G2, 0.0, 1.0, ev2, n, k);
    const DimEdges r2 = edges_at(t2, yy2, G2, sd[k], sS[k], ev2, n, k);
    if (!(cell_narrow(f1) || cell_narrow(r1) || cell_narrow(n2) || cell_narrow(r2))) {
      T1A[k] = o7[k] + o8[k];
      T1B[k] = f1.inner ? o7[k] : 0.0;
      T2A[k] = o9[k] + o10[k];
      T2B[k] = n2.inner ? o9[k] : 0.0;
      continue;
    }
    const DimEdges* es[4] = {&f1, &r1, &n2, &r2};
    bvn::v4d zl, zh, D, pl, ph;
    for (int c = 0; c < 4; ++c) { zl[c] = es[c]->z_lo; zh[c] = es[c]->z_hi; }
    bvn::probit_diff4(zl, zh, D, pl, ph);
    double Dc[4], A[4], B[4];
    for (int c = 0; c < 4; ++c) {
      const DimEdges& e = *es[c];
      double d = D[c], Ain = ph[c] - pl[c], Bin = ph[c];
      if (cell_narrow(e)) {
        const double w = e.w, m0 = e.z_lo + 0.5 * w, m2 = m0 * m0, w2 = w * w, u2 = (w * m0) * (w * m0);
        const double ser = (u2 - w2) / 24.0 + (u2 * (u2 - 6.0 * w2) + 3.0 * w2 * w2) / 1920.0 +
                           (u2 * (u2 * (u2 - 15.0 * w2) + 45.0 * w2 * w2) - 15.0 * w2 * w2 * w2) / 322560.0;
        const double lpn = std::log(w) - 0.5 * m2 - 0.9189385332046727 + std::log1p(ser);
        const double kk = std::exp(-0.125 * w2) / (w * (1.0 + ser));
        d = std::exp(lpn);
        Ain = -2.0 * kk * std::sinh(0.5 * m0 * w) * d;
        Bin = kk * std::exp(-0.5 * m0 * w) * d;
      }
      Dc[c] = d;
      A[c] = e.inner ? Ain : (e.lo ? -pl[c] : ph[c]);
      B[c] = e.inner ? Bin : 0.0;
    }
    const double fl = 1e-300, L = lam[k];
    const double p1f = std::max(Dc[0], fl), p1r = std::max(Dc[1], fl), p2n = std::max(Dc[2], fl), p2r = std::max(Dc[3], fl);
    const double p_fam = p1f * p2n, p_rec = p1r * p2r, raw = (1.0 - L) * p_fam + L * p_rec;
    const double prob = std::max(raw, fl), inv_P = raw > fl ? 1.0 / prob : 0.0;
    const double cf1 = Dc[0] > fl ? (1.0 - L) * p2n * inv_P : 0.0, cr1 = Dc[1] > fl ? L * p2r * inv_P : 0.0;
    const double cn2 = Dc[2] > fl ? (1.0 - L) * p1f * inv_P : 0.0, cr2 = Dc[3] > fl ? L * p1r * inv_P : 0.0;
    const double isi = 1.0 / si[k], isS = 1.0 / sS[k];
    const double gfA = cf1 * A[0], gfB = cf1 * B[0], grA = cr1 * A[1], grB = cr1 * B[1], gsA = cr2 * A[3], gsB = cr2 * B[3];
    const double gf0 = -gfA * isi, gf1 = -lw_scale(gfA, gfB, f1.z_a, f1.w, f1.inner, f1.z_hi) * isi, gf2 = gfA * isi;
    const double gr0 = -grA * isi, gr1 = -lw_scale(grA, grB, r1.z_a, r1.w, r1.inner, r1.z_hi) * isi, gr2 = grA * isi;
    const double gs0 = -gsA * isS, gs1 = -lw_scale(gsA, gsB, r2.z_a, r2.w, r2.inner, r2.z_hi) * isS, gs2 = gsA * isS;
    out[0][k] = std::log(prob);
    out[1][k] = gf0 + gr0;
    out[2][k] = gr0;
    out[3][k] = gs0;
    out[4][k] = (p_rec - p_fam) * inv_P;
    out[5][k] = gf1 + gr1;
    out[6][k] = gs1;
    T1A[k] = gf2 + gr2;
    T1B[k] = f1.inner ? (cf1 * B[0] + cr1 * B[1]) * isi : 0.0;
    T2A[k] = cn2 * A[2] + gs2;
    T2B[k] = n2.inner ? cn2 * B[2] + cr2 * B[3] * isS : 0.0;
  }
  for (int64_t k = 0; k < n; ++k) {
    const int yy1 = int(y1[k]) - 1, yy2 = int(y2[k]) - 1;
    const int MI1 = G1 / 2, NU1 = G1 - MI1, MI2 = G2 / 2, NU2 = G2 - MI2;
    const bool in1 = yy1 > 0 && yy1 < G1 + 1, in2 = yy2 > 0 && yy2 < G2 + 1;
    const int own1 = in1 ? (yy1 - 1 >= MI1 ? yy1 - 1 - MI1 : NU1 + MI1 - yy1) : -1;
    const int own2 = in2 ? (yy2 - 1 >= MI2 ? yy2 - 1 - MI2 : NU2 + MI2 - yy2) : -1;
    out[7][k] = T1A[k];
    out[8][k] = T2A[k];
    for (int g = 0; g < G1; ++g)
      out[9 + g][k] = (gap_sign(g, G1, int(ka1[k])) * T1A[k] + (g == own1 ? T1B[k] : 0.0)) * dv1[size_t(g) * n + k];
    for (int g = 0; g < G2; ++g)
      out[9 + G1 + g][k] = (gap_sign(g, G2, int(ka2[k])) * T2A[k] + (g == own2 ? T2B[k] : 0.0)) * dv2[size_t(g) * n + k];
  }
}

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize("fp-contract=off")
#endif

struct DD {
  double hi, lo;
};

constexpr double kDdLn2[2] = {0.6931471805599453, 2.3190468138462996e-17};
constexpr double kDdLog1p[8][2] = {{1.0, 0.0}, {-0.5, 0.0}, {0.3333333333333333, 1.850371707708594e-17},
                                   {-0.25, 0.0}, {0.2, -1.1102230246251566e-17},
                                   {-0.16666666666666666, -9.25185853854297e-18},
                                   {0.14285714285714285, 7.93016446160826e-18}, {-0.125, 0.0}};
constexpr double kDdInvFact[13][2] = {{1.0, 0.0},
                                      {1.0, 0.0},
                                      {0.5, 0.0},
                                      {0.16666666666666666, 9.25185853854297e-18},
                                      {0.041666666666666664, 2.3129646346357427e-18},
                                      {0.008333333333333333, 1.1564823173178714e-19},
                                      {0.001388888888888889, -5.300543954373577e-20},
                                      {0.0001984126984126984, 1.7209558293420705e-22},
                                      {2.48015873015873e-05, 2.1511947866775882e-23},
                                      {2.7557319223985893e-06, -1.858393274046472e-22},
                                      {2.755731922398589e-07, 2.3767714622250297e-23},
                                      {2.505210838544172e-08, -1.448814070935912e-24},
                                      {2.08767569878681e-09, -1.20734505911326e-25}};

inline DD dd_two_sum(double a, double b) {
  const double s = a + b, bb = s - a;
  return {s, (a - (s - bb)) + (b - bb)};
}

inline DD dd_two_prod(double a, double b) {
  const double p = a * b;
  return {p, std::fma(a, b, -p)};
}

inline DD dd_add(DD a, DD b) {
  const DD s = dd_two_sum(a.hi, b.hi);
  return dd_two_sum(s.hi, s.lo + a.lo + b.lo);
}

inline DD dd_mul(DD a, DD b) {
  const DD p = dd_two_prod(a.hi, b.hi);
  return dd_two_sum(p.hi, p.lo + a.hi * b.lo + a.lo * b.hi);
}

inline DD dd_div(DD a, DD b) {
  const double q = a.hi / b.hi;
  const DD p = dd_two_prod(q, b.hi);
  return {q, (((a.hi - p.hi) - p.lo) + a.lo - q * b.lo) / b.hi};
}

inline DD dd_sum3(double a, double b, double c) {
  DD s = dd_two_sum(a, b);
  const DD t = dd_two_sum(s.hi, c);
  return {t.hi, s.lo + t.lo};
}

inline double dd_offset(DD a, DD b) {
  const DD s = dd_two_sum(a.hi, -b.hi);
  return s.hi + (s.lo + (a.lo - b.lo));
}

inline DD dd_exp(double x) {
  const double k = std::nearbyint(x / kDdLn2[0]);
  const DD p = dd_two_prod(k, kDdLn2[0]);
  DD r = dd_add(dd_add({x, 0.0}, {-p.hi, -p.lo}), {-k * kDdLn2[1], 0.0});
  r = {r.hi / 256.0, r.lo / 256.0};
  DD e = {kDdInvFact[12][0], kDdInvFact[12][1]};
  for (int n = 11; n >= 0; --n) e = dd_add(dd_mul(e, r), {kDdInvFact[n][0], kDdInvFact[n][1]});
  for (int s = 0; s < 8; ++s) e = dd_mul(e, e);
  const int ki = int(k);
  return {std::ldexp(e.hi, ki), std::ldexp(e.lo, ki)};
}

inline DD dd_gap(double g, bool softplus) {
  if (!softplus) return dd_exp(g);
  if (g < -10.0) {
    const DD v = dd_exp(g);
    DD c = {kDdLog1p[7][0], kDdLog1p[7][1]};
    for (int t = 6; t >= 0; --t) c = dd_add(dd_mul(c, v), {kDdLog1p[t][0], kDdLog1p[t][1]});
    return dd_mul(c, v);
  }
  const double y = std::log1p(std::exp(-std::fabs(g))) + std::fmax(g, 0.0);
  const DD ey = dd_exp(y);
  const DD rhs = dd_add(dd_exp(g), {1.0, 0.0});
  const DD num = dd_add(ey, {-rhs.hi, -rhs.lo});
  return dd_two_sum(y, -(num.hi + num.lo) / ey.hi);
}

void dd_offset_k(int64_t n, int G, const double* const* in, double* const* out) {
  for (int64_t k = 0; k < n; ++k) {
    const bool crossing = in[0][k] == 0.0, softplus = in[1][k] != 0.0;
    DD c = {in[7][k], 0.0};
    for (int g = 0; g < G; ++g) {
      const double sg = in[8 + G + g][k];
      if (sg == 0.0) continue;
      const DD v = dd_gap(in[8 + g][k], softplus);
      c = dd_add(c, {sg * v.hi, sg * v.lo});
    }
    if (crossing) {
      const double mu1 = in[2][k], s1 = in[3][k], mu2 = in[4][k], s2 = in[5][k], rho = in[6][k];
      const DD num = dd_sum3(c.hi, -mu1, c.lo);
      const DD z1 = dd_div(num, {s1, 0.0});
      const DD d0 = dd_two_prod(rho, s2);
      const DD d = dd_sum3(d0.hi, 1e-30, d0.lo);
      const DD zc = dd_div({-mu2, 0.0}, d);
      out[0][k] = dd_offset(zc, z1);
    } else {
      out[0][k] = dd_offset(dd_two_sum(in[2][k], in[3][k]), c);
    }
  }
}

#if defined(__clang__)
#pragma clang fp contract(on)
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

template <bool F32>
ffi::Error DdOffsetImpl(ffi::ThreadPool pool, ffi::RemainingArgs args, ffi::RemainingRets rets) {
  const int NI = int(args.size()), G = (NI - 8) / 2;
  if (NI < 8 || (NI - 8) % 2 != 0 || rets.size() != 1) return ffi::Error::InvalidArgument("bayesroc: dd_offset arity");
  std::vector<const void*> in(NI);
  int64_t n = 0;
  for (int a = 0; a < NI; ++a) {
    auto b = args.get<ffi::AnyBuffer>(a);
    if (b.has_error()) return b.error();
    in[a] = b.value().untyped_data();
    n = b.value().element_count();
  }
  auto r = rets.get<ffi::AnyBuffer>(0);
  if (r.has_error()) return r.error();
  void* out = r.value()->untyped_data();
  pool_for(pool, n, 256, [&](int64_t i0, int64_t i1) {
    const int64_t m = i1 - i0;
    std::vector<const double*> ci(NI);
    double* co[1];
    if (!F32) {
      for (int a = 0; a < NI; ++a) ci[a] = static_cast<const double*>(in[a]) + i0;
      co[0] = static_cast<double*>(out) + i0;
      dd_offset_k(m, G, ci.data(), co);
      return;
    }
    std::vector<double> cv(size_t(NI + 1) * m);
    for (int a = 0; a < NI; ++a) {
      double* d = cv.data() + size_t(a) * m;
      const float* f = static_cast<const float*>(in[a]) + i0;
      for (int64_t k = 0; k < m; ++k) d[k] = f[k];
      ci[a] = d;
    }
    co[0] = cv.data() + size_t(NI) * m;
    dd_offset_k(m, G, ci.data(), co);
    float* f = static_cast<float*>(out) + i0;
    for (int64_t k = 0; k < m; ++k) f[k] = static_cast<float>(co[0][k]);
  });
  return ffi::Error::Success();
}

template <void (*FN)(int64_t, int, const double* const*, double* const*), int NIF, int NOF, bool F32>
ffi::Error TrialImpl(ffi::ThreadPool pool, ffi::RemainingArgs args, ffi::RemainingRets rets) {
  const int NI = int(args.size()), G = NI - NIF, NO = NOF + G;
  if (G < 0 || int(rets.size()) != NO) return ffi::Error::InvalidArgument("bayesroc: trial kernel arity");
  std::vector<const void*> in(NI);
  std::vector<void*> out(NO);
  int64_t n = 0;
  for (int a = 0; a < NI; ++a) {
    auto b = args.get<ffi::AnyBuffer>(a);
    if (b.has_error()) return b.error();
    in[a] = b.value().untyped_data();
    n = b.value().element_count();
  }
  for (int a = 0; a < NO; ++a) {
    auto r = rets.get<ffi::AnyBuffer>(a);
    if (r.has_error()) return r.error();
    out[a] = r.value()->untyped_data();
  }
  pool_for(pool, n, 256, [&](int64_t i0, int64_t i1) {
    FlushDenormals ftz;
    const int64_t m = i1 - i0;
    std::vector<const double*> ci(NI);
    std::vector<double*> co(NO);
    if (!F32) {
      for (int a = 0; a < NI; ++a) ci[a] = static_cast<const double*>(in[a]) + i0;
      for (int a = 0; a < NO; ++a) co[a] = static_cast<double*>(out[a]) + i0;
      FN(m, G, ci.data(), co.data());
      return;
    }
    thread_local std::vector<double> cv;
    if (cv.size() < size_t(NI + NO) * m) cv.resize(size_t(NI + NO) * m);
    for (int a = 0; a < NI; ++a) {
      double* dd = cv.data() + size_t(a) * m;
      const float* f = static_cast<const float*>(in[a]) + i0;
      for (int64_t k = 0; k < m; ++k) dd[k] = f[k];
      ci[a] = dd;
    }
    for (int a = 0; a < NO; ++a) co[a] = cv.data() + size_t(NI + a) * m;
    FN(m, G, ci.data(), co.data());
    for (int a = 0; a < NO; ++a) {
      float* f = static_cast<float*>(out[a]) + i0;
      for (int64_t k = 0; k < m; ++k) f[k] = static_cast<float>(co[a][k]);
    }
  });
  return ffi::Error::Success();
}

}

#define BAYESROC_TRIAL(NAME, FN, NIF, NOF, F32)                                     \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(NAME, (TrialImpl<FN, NIF, NOF, F32>),                 \
                                ffi::Ffi::Bind().Ctx<ffi::ThreadPool>().RemainingArgs().RemainingRets())

XLA_FFI_DEFINE_HANDLER_SYMBOL(BayesrocDdOffset, DdOffsetImpl<false>,
                              ffi::Ffi::Bind().Ctx<ffi::ThreadPool>().RemainingArgs().RemainingRets());
XLA_FFI_DEFINE_HANDLER_SYMBOL(BayesrocDdOffsetF32, DdOffsetImpl<true>,
                              ffi::Ffi::Bind().Ctx<ffi::ThreadPool>().RemainingArgs().RemainingRets());

BAYESROC_TRIAL(BayesrocMetadTrial, metad_chunk<false>, 8, 5, false);
BAYESROC_TRIAL(BayesrocMetadTrialSp, metad_chunk<true>, 8, 5, false);
BAYESROC_TRIAL(BayesrocMetadTrialF32, metad_chunk<false>, 8, 5, true);
BAYESROC_TRIAL(BayesrocMetadTrialSpF32, metad_chunk<true>, 8, 5, true);
BAYESROC_TRIAL(BayesrocBgTrial, bg_chunk<false>, 10, 8, false);
BAYESROC_TRIAL(BayesrocBgTrialSp, bg_chunk<true>, 10, 8, false);
BAYESROC_TRIAL(BayesrocBgTrialF32, bg_chunk<false>, 10, 8, true);
BAYESROC_TRIAL(BayesrocBgTrialSpF32, bg_chunk<true>, 10, 8, true);
BAYESROC_TRIAL(BayesrocVrdpTrial, vrdp_chunk<false>, 11, 9, false);
BAYESROC_TRIAL(BayesrocVrdpTrialSp, vrdp_chunk<true>, 11, 9, false);
BAYESROC_TRIAL(BayesrocVrdpTrialF32, vrdp_chunk<false>, 11, 9, true);
BAYESROC_TRIAL(BayesrocVrdpTrialSpF32, vrdp_chunk<true>, 11, 9, true);

#define BAYESROC_ORDINAL(NAME, SP, F32)                                             \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(NAME, (OrdinalImpl<SP, F32>),                         \
                                ffi::Ffi::Bind().Ctx<ffi::ThreadPool>().RemainingArgs().RemainingRets())

BAYESROC_ORDINAL(BayesrocOrdinal, false, false);
BAYESROC_ORDINAL(BayesrocOrdinalSp, true, false);
BAYESROC_ORDINAL(BayesrocOrdinalF32, false, true);
BAYESROC_ORDINAL(BayesrocOrdinalSpF32, true, true);

#define BAYESROC_F32(NAME, NI, NO, K)                                               \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(NAME, (F32Impl<NI, NO, K>),                           \
                                ffi::Ffi::Bind().Ctx<ffi::ThreadPool>().RemainingArgs().RemainingRets())

BAYESROC_F32(BayesrocBvnRectF32, 5, 6, rect_k);
BAYESROC_F32(BayesrocVrdp2dF32, 14, 11, bvn::vrdp_array);
BAYESROC_F32(BayesrocBvnCdf2F32, 6, 18, (bvn::cdf_multi_array<2, 1>));
BAYESROC_F32(BayesrocBvnCdf4F32, 11, 36, (bvn::cdf_multi_array<4, 2>));
BAYESROC_F32(BayesrocCdpCellF32, 11, 10, cdp_k);
BAYESROC_F32(BayesrocMetadCellF32, 8, 6, bvn::metad_array);
BAYESROC_F32(BayesrocProbitCellF32, 2, 3, probit_cell_k);
BAYESROC_F32(BayesrocProbitDiffF32, 2, 3, probit_diff_k);
BAYESROC_F32(BayesrocBbvCellF32, 12, 14, bbv_k<2>);

#define BAYESROC_F64(NAME, NI, NO, K)                                               \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(NAME, (F64Impl<NI, NO, K>),                           \
                                ffi::Ffi::Bind().Ctx<ffi::ThreadPool>().RemainingArgs().RemainingRets())

BAYESROC_F64(BayesrocCdpCell, 11, 10, cdp_k);
BAYESROC_F64(BayesrocMetadCell, 8, 6, bvn::metad_array);
BAYESROC_F64(BayesrocBvnRect, 5, 6, rect_k);
BAYESROC_F64(BayesrocVrdp2d, 14, 11, bvn::vrdp_array);
BAYESROC_F64(BayesrocBvnCdf2, 6, 18, (bvn::cdf_multi_array<2, 1>));
BAYESROC_F64(BayesrocBvnCdf4, 11, 36, (bvn::cdf_multi_array<4, 2>));
BAYESROC_F64(BayesrocBbvCell, 12, 14, bbv_k<5>);

XLA_FFI_DEFINE_HANDLER_SYMBOL(BayesrocProbitDiff, ProbitDiffImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::ThreadPool>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(BayesrocProbitCell, ProbitCellImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::ThreadPool>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>());






// a, c [N, n] -> lp [N] = log int phi(z) prod_j Phi(a_j z + c_j) dz, dc, da [N, n]
template <typename T>
ffi::Error PcqImpl(ffi::ThreadPool pool, ffi::AnyBuffer a, ffi::AnyBuffer c, ffi::Result<ffi::AnyBuffer> lp,
                   ffi::Result<ffi::AnyBuffer> dc, ffi::Result<ffi::AnyBuffer> da) {
  const auto dims = a.dimensions();
  const int n = dims.size() > 1 ? static_cast<int>(dims[dims.size() - 1]) : 1;
  if (n > pcq::kMaxTerms) return ffi::Error::InvalidArgument("bayesroc: pcq term count");
  const int64_t N = a.element_count() / n;
  DBuf<T> pa(a), pc(c);
  DOut<T> ol(lp), oc(dc), oa(da);
  const double *A = pa.ptr, *C = pc.ptr;
  double *L = ol.ptr, *DC = oc.ptr, *DA = oa.ptr;
  pool_for(pool, N, 64, [=](int64_t i0, int64_t i1) {
    for (int64_t i = i0; i < i1; ++i) L[i] = pcq::log_int(n, A + i * n, C + i * n, DC + i * n, DA + i * n);
  });
  return ffi::Error::Success();
}

// a, ln, lf, s [N] -> lp, da, dln, dlf, ds [N]: lognormal meta-noise cell (ln = -inf / lf = +inf: open edge)
template <typename T>
ffi::Error LnoiseCellImpl(ffi::ThreadPool pool, ffi::AnyBuffer a, ffi::AnyBuffer ln, ffi::AnyBuffer lf,
                          ffi::AnyBuffer s, ffi::Result<ffi::AnyBuffer> lp, ffi::Result<ffi::AnyBuffer> da,
                          ffi::Result<ffi::AnyBuffer> dln, ffi::Result<ffi::AnyBuffer> dlf,
                          ffi::Result<ffi::AnyBuffer> ds) {
  const int64_t N = a.element_count();
  DBuf<T> pa(a), pn(ln), pf(lf), ps(s);
  DOut<T> o0(lp), o1(da), o2(dln), o3(dlf), o4(ds);
  const double *A = pa.ptr, *LN = pn.ptr, *LF = pf.ptr, *S = ps.ptr;
  double *P = o0.ptr, *GA = o1.ptr, *GN = o2.ptr, *GF = o3.ptr, *GS = o4.ptr;
  pool_for(pool, N, 512, [=](int64_t i0, int64_t i1) {
    for (int64_t i = i0; i < i1; ++i) {
      lnoise::Cell c = lnoise::cell(A[i], LN[i], LF[i], S[i]);
      P[i] = c.lp;
      GA[i] = c.da;
      GN[i] = c.dln;
      GF[i] = c.dlf;
      GS[i] = c.ds;
    }
  });
  return ffi::Error::Success();
}

#define BAYESROC_TMAP(NAME, IMPL, NARG, NRET)                                         XLA_FFI_DEFINE_HANDLER_SYMBOL(NAME, IMPL, ffi::Ffi::Bind().Ctx<ffi::ThreadPool>() NARG NRET)
#define A_ .Arg<ffi::AnyBuffer>()
#define R_ .Ret<ffi::AnyBuffer>()
BAYESROC_TMAP(BayesrocPcq, PcqImpl<double>, A_ A_, R_ R_ R_);
BAYESROC_TMAP(BayesrocPcqF32, PcqImpl<float>, A_ A_, R_ R_ R_);
BAYESROC_TMAP(BayesrocLnoiseCell, LnoiseCellImpl<double>, A_ A_ A_ A_, R_ R_ R_ R_ R_);
BAYESROC_TMAP(BayesrocLnoiseCellF32, LnoiseCellImpl<float>, A_ A_ A_ A_, R_ R_ R_ R_ R_);
BAYESROC_TMAP(BayesrocTmapFwd, TmapFwdImpl<double>, A_ A_ A_ A_ A_, R_ R_ R_ R_);
BAYESROC_TMAP(BayesrocTmapFwdF32, TmapFwdImpl<float>, A_ A_ A_ A_ A_, R_ R_ R_ R_);
BAYESROC_TMAP(BayesrocTmapBwd, TmapBwdImpl<double>, A_ A_ A_ A_ A_ A_ A_ A_ A_ A_, R_ R_ R_);
BAYESROC_TMAP(BayesrocTmapBwdF32, TmapBwdImpl<float>, A_ A_ A_ A_ A_ A_ A_ A_ A_ A_, R_ R_ R_);
BAYESROC_TMAP(BayesrocTmapBlkFwd, TmapBlkFwdImpl<double>, A_ A_ A_ A_ A_ A_, R_ R_ R_ R_ R_ R_ R_ R_);
BAYESROC_TMAP(BayesrocTmapBlkFwdF32, TmapBlkFwdImpl<float>, A_ A_ A_ A_ A_ A_, R_ R_ R_ R_ R_ R_ R_ R_);
BAYESROC_TMAP(BayesrocTmapBlkBwd, TmapBlkBwdImpl<double>, A_ A_ A_ A_ A_ A_ A_ A_ A_ A_ A_, R_ R_ R_);
BAYESROC_TMAP(BayesrocTmapBlkBwdF32, TmapBlkBwdImpl<float>, A_ A_ A_ A_ A_ A_ A_ A_ A_ A_ A_, R_ R_ R_);

// Plain C entry for R's .C(): Phi2(h, k; rho) at n finite points, split over threads.
extern "C" void bayesroc_r_bvn_cdf(const double* h, const double* k, const double* rho, const int* n,
                                   const int* threads, double* p) {
  const int64_t N = *n;
  if (N <= 0) return;
  const int64_t grain = 4096;
  int64_t T = std::max<int64_t>(1, std::min<int64_t>(*threads, (N + grain - 1) / grain));
  auto work = [=](int64_t i0, int64_t i1) {
    std::vector<double> d(size_t(3) * size_t(i1 - i0));
    const int64_t m = i1 - i0;
    bvn::cdf_array(m, h + i0, k + i0, rho + i0, p + i0, d.data(), d.data() + m, d.data() + 2 * m);
  };
  if (T == 1) { work(0, N); return; }
  const int64_t per = ((N + T - 1) / T + 3) / 4 * 4;
  std::vector<std::thread> pool;
  for (int64_t t = 1; t < T; ++t) {
    const int64_t i0 = t * per;
    if (i0 >= N) break;
    pool.emplace_back(work, i0, std::min(N, i0 + per));
  }
  work(0, std::min(N, per));
  for (auto& th : pool) th.join();
}

// Plain C entry for R's .C(): the standard normal CDF at n points, split over threads.
extern "C" void bayesroc_r_pnorm(const double* x, const int* n, const int* threads, double* p) {
  const int64_t N = *n;
  if (N <= 0) return;
  const int64_t grain = 16384;
  int64_t T = std::max<int64_t>(1, std::min<int64_t>(*threads, (N + grain - 1) / grain));
  auto work = [=](int64_t i0, int64_t i1) {
    int64_t i = i0;
    for (; i + 4 <= i1; i += 4) {
      bvn::v4d v;
      std::memcpy(&v, x + i, 32);
      bvn::v4d r = vphi_p(v);
      std::memcpy(p + i, &r, 32);
    }
    for (; i < i1; ++i) p[i] = 0.5 * std::erfc(-x[i] * 0.7071067811865476);
  };
  if (T == 1) { work(0, N); return; }
  const int64_t per = (N + T - 1) / T;
  std::vector<std::thread> pool;
  for (int64_t t = 1; t < T; ++t) {
    const int64_t i0 = t * per;
    if (i0 >= N) break;
    pool.emplace_back(work, i0, std::min(N, i0 + per));
  }
  work(0, std::min(N, per));
  for (auto& th : pool) th.join();
}

// Plain C entry for R's .C(): the continuous dual-process model's confidence-band cells for prediction.
// Per (draw, trial) muR, sR, muF, sF, cR, cK [S, N] (new items already carry means 0 and SDs 1); th [S, N, J]
// confidence thresholds; dims = {S, N, J, K, n_rkg, threads, n_new}; old_levels [J] and new_levels [n_new]
// (1-based response levels). Band b (0 = below the first threshold = the new responses, b >= 1 = old level b)
// splits into Remember / Know (/ Guess) by the closed forms of compute_rkg_probs_r (normal and bivariate-normal
// CDFs, the band's Phi2 shared with its neighbour), and any share below 1e-4 is replaced by the likelihood's
// exact one-dimensional integral (bvn_tail::cdp_cell, or the strips about the recollection criterion when
// n_rkg = 2). probs, prem, pknow [S, N, K]: the band's probability spread over its levels, and the judgement
// shares.
extern "C" void bayesroc_r_cdp_probs(const double* muR, const double* sR, const double* muF, const double* sF,
                                     const double* cR, const double* cK, const double* rf, const double* th,
                                     const int* dims,
                                     const int* old_levels, const int* new_levels, double* probs, double* prem,
                                     double* pknow) {
  const int64_t S = dims[0], N = dims[1], J = dims[2], K = dims[3];
  const int n_rkg = dims[4], threads = dims[5], n_new = dims[6];
  const int64_t SN = S * N;
  const double inf = std::numeric_limits<double>::infinity();
  auto Phi = [](double z) { return bvn_tail::Qd(-z); };
  int64_t T = std::max<int64_t>(1, std::min<int64_t>(threads, N));
  auto work = [&](int64_t n0, int64_t n1) {
    const int64_t nb = J + 2;
    std::vector<double> tau(nb), zc(nb), Pz(nb), A(nb), zs(nb), Ps(nb), Bv(nb), G(nb), Pk(nb);
    std::vector<double> h2(nb), k2(nb), r2(nb), p2(nb), e1(nb), e2(nb), e3(nb);
    std::vector<double> hh, kk, rr, pp, d1, d2, d3;
    std::vector<int> where;
    hh.reserve(2 * nb); kk.reserve(2 * nb); rr.reserve(2 * nb); pp.resize(2 * nb); d1.resize(2 * nb);
    d2.resize(2 * nb); d3.resize(2 * nb); where.reserve(2 * nb);
    for (int64_t n = n0; n < n1; ++n) {
      for (int64_t s = 0; s < S; ++s) {
        const int64_t sn = s + S * n;
        const double mR = muR[sn], dR = sR[sn], mF = muF[sn], dF = sF[sn], c_R = cR[sn], c_K = cK[sn], cr = rf[sn];
        const double mM = mR + mF, sM = std::sqrt(dR * dR + dF * dF + 2.0 * cr * dR * dF), rho = (dR + cr * dF) / sM;
        const double zcR = (c_R - mR) / dR, zcK = (c_K - mF) / dF, PcR = Phi(zcR), PcK = Phi(zcK);
        tau[0] = -inf;
        for (int64_t j = 0; j < J; ++j) tau[j + 1] = th[sn + SN * j];
        tau[J + 1] = inf;
        hh.clear(); kk.clear(); rr.clear(); where.clear();
        for (int64_t t = 0; t < nb; ++t) {
          zc[t] = std::isfinite(tau[t]) ? (tau[t] - mM) / sM : tau[t];
          Pz[t] = std::isfinite(zc[t]) ? Phi(zc[t]) : (zc[t] > 0 ? 1.0 : 0.0);
          // A = Phi2(zcR, zc): the recollection criterion against the band edge
          if (!std::isfinite(zc[t])) A[t] = zc[t] > 0 ? PcR : 0.0;
          else { hh.push_back(zcR); kk.push_back(zc[t]); rr.push_back(rho); where.push_back(static_cast<int>(t)); }
          if (n_rkg == 3) {
            const double sv = std::isfinite(tau[t]) ? std::fmin(c_R, tau[t] - c_K) : (tau[t] > 0 ? c_R : -inf);
            zs[t] = std::isfinite(sv) ? (sv - mR) / dR : -inf;
            Ps[t] = std::isfinite(zs[t]) ? Phi(zs[t]) : 0.0;
            if (!std::isfinite(zs[t]) || !std::isfinite(zc[t])) Bv[t] = (std::isfinite(zs[t]) && zc[t] > 0) ? Ps[t] : 0.0;
            else { hh.push_back(zs[t]); kk.push_back(zc[t]); rr.push_back(rho); where.push_back(static_cast<int>(-t - 1)); }
          }
        }
        const int64_t m = static_cast<int64_t>(hh.size());
        if (m) bvn::cdf_array(m, hh.data(), kk.data(), rr.data(), pp.data(), d1.data(), d2.data(), d3.data());
        for (int64_t q = 0; q < m; ++q) {
          if (where[q] >= 0) A[where[q]] = pp[q]; else Bv[-where[q] - 1] = pp[q];
        }
        if (n_rkg == 3) {
          // P(R <= s, F <= c_K): a product when R and F are independent, else Phi2(zs, zcK; corr)
          int64_t m2 = 0;
          for (int64_t t = 0; t < nb; ++t) {
            Pk[t] = Ps[t] * PcK;
            if (cr != 0.0 && std::isfinite(zs[t])) { h2[m2] = zs[t]; k2[m2] = zcK; r2[m2] = cr; ++m2; }
          }
          if (m2) {
            bvn::cdf_array(m2, h2.data(), k2.data(), r2.data(), p2.data(), e1.data(), e2.data(), e3.data());
            int64_t q = 0;
            for (int64_t t = 0; t < nb; ++t) if (std::isfinite(zs[t])) Pk[t] = p2[q++];
          }
          for (int64_t t = 0; t < nb; ++t) G[t] = Pk[t] + A[t] - Bv[t];
        }
        for (int64_t b = 0; b <= J; ++b) {
          const double tlo = tau[b], thi = tau[b + 1];
          const double pband = Pz[b + 1] - Pz[b];
          double c1 = (Pz[b + 1] - A[b + 1]) - (Pz[b] - A[b]), c2, c3 = 0.0;
          if (n_rkg == 3) {
            c3 = G[b + 1] - G[b];
            c2 = pband - c1 - c3;
            if (c1 < bvn_tail::kTailSwitch) c1 = std::exp(bvn_tail::cdp_cell(1, mR, dR, mF, dF, c_R, c_K, tlo, thi, cr).lp);
            if (c2 < bvn_tail::kTailSwitch) c2 = std::exp(bvn_tail::cdp_cell(2, mR, dR, mF, dF, c_R, c_K, tlo, thi, cr).lp);
            if (c3 < bvn_tail::kTailSwitch) c3 = std::exp(bvn_tail::cdp_cell(3, mR, dR, mF, dF, c_R, c_K, tlo, thi, cr).lp);
          } else {
            c2 = pband - c1;
            if (c1 < bvn_tail::kTailSwitch || c2 < bvn_tail::kTailSwitch) {
              const double sc = dF * std::sqrt((1.0 - cr) * (1.0 + cr)), B = -(dR + cr * dF) / sc;
              const double Alo = std::isfinite(tlo) ? (tlo - mR - mF) / sc : -inf;
              const double Ahi = std::isfinite(thi) ? (thi - mR - mF) / sc : inf;
              bvn_tail::LinCond c{Alo, std::isfinite(tlo) ? B : 0.0, Ahi, std::isfinite(thi) ? B : 0.0};
              if (c1 < bvn_tail::kTailSwitch) c1 = std::exp(bvn_tail::log_integral(c, zcR, inf));
              if (c2 < bvn_tail::kTailSwitch) c2 = std::exp(bvn_tail::log_integral(c, -inf, zcR));
            }
          }
          const double total = c1 + c2 + c3;
          const double pr = total > 0.0 ? c1 / total : 0.0, pk = total > 0.0 ? c2 / total : 0.0;
          if (b == 0) {
            for (int q = 0; q < n_new; ++q) {
              const int64_t k = new_levels[q] - 1, at = sn + SN * k;
              probs[at] = total / n_new;
              prem[at] = pr;
              pknow[at] = pk;
            }
          } else {
            const int64_t k = old_levels[b - 1] - 1, at = sn + SN * k;
            probs[at] = total;
            prem[at] = pr;
            pknow[at] = pk;
          }
        }
      }
    }
  };
  if (T == 1) {
    work(0, N);
  } else {
    const int64_t per = (N + T - 1) / T;
    std::vector<std::thread> pool;
    for (int64_t t = 1; t < T; ++t)
      if (t * per < N) pool.emplace_back(work, t * per, std::min(N, (t + 1) * per));
    work(0, std::min(N, per));
    for (auto& th_ : pool) th_.join();
  }
}

// Plain C entry for R's .C(): exp over a long vector on up to `threads` threads (the threshold gaps).
extern "C" void bayesroc_r_exp(const double* x, const int* n, const int* threads, double* p) {
  const int64_t N = *n;
  if (N <= 0) return;
  const int64_t grain = 16384;
  int64_t T = std::max<int64_t>(1, std::min<int64_t>(*threads, (N + grain - 1) / grain));
  auto work = [=](int64_t i0, int64_t i1) {
    int64_t i = i0;
    for (; i + 4 <= i1; i += 4) {
      bvn::v4d v;
      std::memcpy(&v, x + i, 32);
      bvn::v4d r = bvn::vexp(v);
      std::memcpy(p + i, &r, 32);
    }
    for (; i < i1; ++i) p[i] = std::exp(x[i]);
  };
  if (T == 1) { work(0, N); return; }
  const int64_t per = (N + T - 1) / T;
  std::vector<std::thread> pool;
  for (int64_t t = 1; t < T; ++t) {
    const int64_t i0 = t * per;
    if (i0 >= N) break;
    pool.emplace_back(work, i0, std::min(N, i0 + per));
  }
  work(0, std::min(N, per));
  for (auto& th : pool) th.join();
}

namespace {
// run body(i0, i1) over [0, N) on up to `threads` threads, in blocks of at least `grain`
template <typename F>
void r_threads(int64_t N, int threads, int64_t grain, F body) {
  int64_t T = std::max<int64_t>(1, std::min<int64_t>(threads, (N + grain - 1) / grain));
  if (T == 1) { body(0, N); return; }
  const int64_t per = ((N + T - 1) / T + 3) / 4 * 4;
  std::vector<std::thread> pool;
  for (int64_t t = 1; t < T; ++t) {
    const int64_t i0 = t * per;
    if (i0 >= N) break;
    pool.emplace_back(body, i0, std::min(N, i0 + per));
  }
  body(0, std::min(N, per));
  for (auto& th : pool) th.join();
}
}  // namespace

// Plain C entry for R's .C(): P(a1 < Z1 <= b1, a2 < Z2 <= b2) at n points; edges beyond
// +-1e9 are infinite.
extern "C" void bayesroc_r_bvn_rect(const double* a1, const double* b1, const double* a2, const double* b2,
                                    const double* rho, const int* n, const int* threads, double* p) {
  const int64_t N = *n;
  if (N <= 0) return;
  r_threads(N, *threads, 4096, [=](int64_t i0, int64_t i1) {
    const int64_t m = i1 - i0;
    std::vector<double> d(size_t(5) * size_t(m));
    bvn::rect_array(m, a1 + i0, b1 + i0, a2 + i0, b2 + i0, rho + i0, p + i0, d.data(), d.data() + m,
                    d.data() + 2 * m, d.data() + 3 * m, d.data() + 4 * m);
  });
}

// Plain C entry for R's .C(): log P of n lognormal meta-noise cells (ln = -inf / lf = +inf: open edge).
extern "C" void bayesroc_r_lnoise_cell(const double* a, const double* ln, const double* lf, const double* s,
                                       const int* n, const int* threads, double* lp) {
  const int64_t N = *n;
  if (N <= 0) return;
  r_threads(N, *threads, 256, [=](int64_t i0, int64_t i1) {
    for (int64_t i = i0; i < i1; ++i) lp[i] = lnoise::cell(a[i], ln[i], lf[i], s[i]).lp;
  });
}

// Plain C entry for R's .C(): m-AFC choice probabilities. d, sg, target (1-based), m [n]; b [n, M]
// column-major; out [n, M] column-major, P(choice k | target) for k < m and 0 beyond.
extern "C" void bayesroc_r_mafc_probs(const double* d, const double* sg, const double* b, const int* m,
                                      const int* target, const int* M, const int* n, const int* threads, double* out) {
  const int64_t N = *n;
  const int MM = *M;
  if (N <= 0) return;
  r_threads(N, *threads, 64, [=](int64_t i0, int64_t i1) {
    double bb[mafc::kMaxAlt], db[mafc::kMaxAlt], dd, ds;
    for (int64_t i = i0; i < i1; ++i) {
      for (int k = 0; k < MM; ++k) bb[k] = b[i + N * k];
      for (int k = 0; k < MM; ++k)
        out[i + N * k] = k < m[i] ? std::exp(mafc::cell(m[i], target[i] - 1, k, d[i], sg[i], bb, dd, ds, db)) : 0.0;
    }
  });
}

// Plain C entry for R's .C(): weighted survival curves of the detection response.
// th [S, R, J] thresholds, mu, sig, lam, mu2 [S, R], u [U], w [R]; dims = {S, R, J, U, kind,
// path, threads}. kind 0: normal(mu, sig); 1: (1 - lam) normal + lam at the top response;
// 2: (1 - lam) normal(mu, sig) + lam normal(mu2, sig); 4: (1 - lam) normal + lam at the top
// response where mu2 = 1, at the bottom one where mu2 = 0. path 1: criterion j lies at u[j] along
// the row's own thresholds (extended by ext at both ends, linear between knots); path 0: at
// u[j] itself. acc [S, U] receives sum over rows of w * max(1 - F(criterion), 0).
static inline bvn::v4d vlink_surv(bvn::v4d z, int link) {
  using bvn::splat;
  if (link == 1) return splat(1.0) / (splat(1.0) + bvn::vexp(z));
  if (link == 2) return bvn::vexp(-bvn::vexp(bvn::vmin(z, splat(700.0))));
  return vphi_p(-z);
}

static inline void accumulate_surv(double* out, int64_t s, int64_t S, int64_t j, int64_t U, double wr, bvn::v4d sv) {
  const int64_t n = std::min<int64_t>(4, U - j);
  for (int64_t q = 0; q < n; ++q)
    if (sv[q] > 0.0) out[s + S * (j + q)] += wr * sv[q];
}

extern "C" void bayesroc_r_det_survival(const double* th, const double* mu, const double* sig,
                                        const double* lam, const double* mu2, const double* u,
                                        const double* w, const int* dims, const double* ext,
                                        double* acc) {
  const int64_t S = dims[0], R = dims[1], J = dims[2], U = dims[3];
  const int kind = dims[4], path = dims[5], threads = dims[6], link = dims[7];
  const double e = *ext;
  std::vector<int> lo(U);
  std::vector<double> fr(U);
  for (int64_t j = 0; j < U; ++j) {
    double f = std::floor(u[j]);
    int l = static_cast<int>(f < J ? f : J);
    lo[j] = l;
    fr[j] = u[j] - l;
  }
  int64_t T = std::max<int64_t>(1, std::min<int64_t>(threads, R));
  std::vector<std::vector<double>> part(T, std::vector<double>(size_t(S) * size_t(U), 0.0));
  const int64_t Up = (U + 3) & ~int64_t(3);
  auto work = [&](int64_t t, int64_t r0, int64_t r1) {
    using bvn::splat;
    double* out = part[t].data();
    std::vector<double> knots(J + 2), cu(Up, 0.0);
    for (int64_t r = r0; r < r1; ++r) {
      const double wr = w[r];
      if (wr == 0.0) continue;
      for (int64_t s = 0; s < S; ++s) {
        const int64_t sr = s + S * r;
        const bvn::v4d vm = splat(mu[sr]), vis = splat(1.0 / sig[sr]);
        const bvn::v4d vl = splat(kind ? lam[sr] : 0.0), vm2 = splat(kind == 2 || kind == 4 ? mu2[sr] : 0.0);
        if (path) {
          for (int64_t j = 0; j < J; ++j) knots[j + 1] = th[sr + S * R * j];
          knots[0] = knots[1] - e;
          knots[J + 1] = knots[J] + e;
        }
        for (int64_t j = 0; j < U; ++j) cu[j] = path ? knots[lo[j]] * (1.0 - fr[j]) + knots[lo[j] + 1] * fr[j] : u[j];
        for (int64_t j = 0; j < Up; j += 4) {
          bvn::v4d c;
          std::memcpy(&c, &cu[j], 32);
          bvn::v4d sv = vlink_surv((c - vm) * vis, link);
          if (kind == 1) sv = vl + (splat(1.0) - vl) * sv;
          else if (kind == 2) sv = (splat(1.0) - vl) * sv + vl * vlink_surv((c - vm2) * vis, link);
          else if (kind == 4) sv = (splat(1.0) - vl) * sv + vl * vm2;
          accumulate_surv(out, s, S, j, U, wr, sv);
        }
      }
    }
  };
  if (T == 1) {
    work(0, 0, R);
  } else {
    const int64_t per = (R + T - 1) / T;
    std::vector<std::thread> pool;
    for (int64_t t = 1; t < T; ++t)
      if (t * per < R) pool.emplace_back(work, t, t * per, std::min(R, (t + 1) * per));
    work(0, 0, std::min(R, per));
    for (auto& th_ : pool) th_.join();
  }
  for (int64_t t = 0; t < T; ++t)
    for (int64_t k = 0; k < S * U; ++k) acc[k] += part[t][k];
}

static inline bvn::v4d vlink_cdf(bvn::v4d z, int link) {
  using bvn::splat;
  if (link == 1) return splat(1.0) / (splat(1.0) + bvn::vexp(-z));
  if (link == 2) return splat(1.0) - bvn::vexp(-bvn::vexp(bvn::vmin(z, splat(700.0))));
  return vphi_p(z);
}

// Plain C entry for R's .C(): weighted survival curves of the confidence response of the meta models.
// th [S, R, J] thresholds (the middle one the decision criterion c0), d, p1, p2 [S, R], a [R] (is_old),
// u [U], w [R]; dims = {S, R, J, U, path, threads, link, kind, rel}; q (meta-d' midpoint weight), ext.
// kind 0 meta-d' (p1, p2 the M-ratios of "old" and "new" responses), 1 Gaussian meta-noise (p1 = sigma_meta),
// 2 lognormal meta-noise (p1 = sigma_meta). acc [S, U] += w_r * P(rating above the criterion at u).
extern "C" void bayesroc_r_meta_survival(const double* th, const double* d, const double* p1, const double* p2,
                                         const double* a, const double* u, const double* w, const int* dims,
                                         const double* q, const double* ext, double* acc) {
  const int64_t S = dims[0], R = dims[1], J = dims[2], U = dims[3];
  const int path = dims[4], threads = dims[5], link = dims[6], kind = dims[7], rel = dims[8];
  const double e = *ext, qq = *q;
  const int64_t mid = (J + 1) / 2 - 1;
  std::vector<int> lo(U);
  std::vector<double> fr(U);
  for (int64_t j = 0; j < U; ++j) {
    double f = std::floor(u[j]);
    int l = static_cast<int>(f < J ? f : J);
    lo[j] = l;
    fr[j] = u[j] - l;
  }
  int64_t T = std::max<int64_t>(1, std::min<int64_t>(threads, R));
  std::vector<std::vector<double>> part(T, std::vector<double>(size_t(S) * size_t(U), 0.0));
  const int64_t Up = (U + 3) & ~int64_t(3);
  auto work = [&](int64_t t, int64_t r0, int64_t r1) {
    using bvn::splat;
    using bvn::v4d;
    double* out = part[t].data();
    std::vector<double> knots(J + 2), cu(Up, 0.0), gh_(Up), gk_(Up), gr_(Up), gp_(Up), gd_(3 * Up);
    for (int64_t r = r0; r < r1; ++r) {
      const double wr = w[r];
      if (wr == 0.0) continue;
      for (int64_t s = 0; s < S; ++s) {
        const int64_t sr = s + S * r;
        const double c0 = th[sr + S * R * mid], dd = d[sr], mu = a[r] * dd;
        if (path) {
          for (int64_t j = 0; j < J; ++j) knots[j + 1] = th[sr + S * R * j];
          knots[0] = knots[1] - e;
          knots[J + 1] = knots[J] + e;
        }
        for (int64_t j = 0; j < U; ++j) cu[j] = path ? knots[lo[j]] * (1.0 - fr[j]) + knots[lo[j] + 1] * fr[j] : u[j];
        if (kind == 0) {
          const double m0 = qq * dd;
          const double mo = p1[sr], mn = p2[sr];
          const double muo = m0 + mo * dd * (a[r] - qq), mun = m0 + mn * dd * (a[r] - qq);
          const double so = rel ? (mo - 1.0) * (c0 - m0) : 0.0, sn = rel ? (mn - 1.0) * (c0 - m0) : 0.0;
          const v4d head = v4d{c0 - mu, c0 + so - muo, c0 + sn - mun, 0.0};
          const v4d sv0 = vlink_surv(head, link), cd0 = vlink_cdf(head, link);
          const double pold = sv0[0], pnew = cd0[0], deno = sv0[1], denn = cd0[2];
          const double fo = deno > 0.0 ? pold / deno : 0.0, fn = denn > 0.0 ? pnew / denn : 0.0;
          for (int64_t j = 0; j < Up; j += 4) {
            v4d c;
            std::memcpy(&c, &cu[j], 32);
            const v4d upv = splat(fo) * vlink_surv(c + splat(so - muo), link);
            const v4d dnv = splat(1.0) - splat(fn) * vlink_cdf(c + splat(sn - mun), link);
            accumulate_surv(out, s, S, j, U, wr, bvn::vsel(c >= splat(c0), upv, dnv));
          }
        } else if (kind == 1) {
          const double tau = std::sqrt(1.0 + p1[sr] * p1[sr]), rho = 1.0 / tau, zc = c0 - mu;
          for (int64_t j = 0; j < U; ++j) {
            const double z = (cu[j] - mu) / tau, sg = cu[j] > c0 ? -1.0 : 1.0;
            gh_[j] = sg * zc;
            gk_[j] = sg * z;
            gr_[j] = rho;
          }
          bvn::cdf_array(U, gh_.data(), gk_.data(), gr_.data(), gp_.data(), gd_.data(), gd_.data() + U, gd_.data() + 2 * U);
          const double pold = vphi_p(splat(-zc))[0];
          for (int64_t j = 0; j < U; ++j) {
            const double v = cu[j] == c0 ? pold : cu[j] > c0 ? gp_[j] : 1.0 - gp_[j];
            if (v > 0.0) out[s + S * j] += wr * v;
          }
        } else {
          const double sm = p1[sr];
          for (int64_t j = 0; j < U; ++j) {
            const bool up = cu[j] >= c0;
            const double D = std::fabs(cu[j] - c0);
            const double pc = std::exp(lnoise::cell(up ? mu - c0 : c0 - mu, std::log(D), bvn_tail::kInf, sm).lp);
            const double v = up ? pc : 1.0 - pc;
            if (v > 0.0) out[s + S * j] += wr * v;
          }
        }
      }
    }
  };
  if (T == 1) {
    work(0, 0, R);
  } else {
    const int64_t per = (R + T - 1) / T;
    std::vector<std::thread> pool;
    for (int64_t t = 1; t < T; ++t)
      if (t * per < R) pool.emplace_back(work, t, t * per, std::min(R, (t + 1) * per));
    work(0, 0, std::min(R, per));
    for (auto& th_ : pool) th_.join();
  }
  for (int64_t t = 0; t < T; ++t)
    for (int64_t k = 0; k < S * U; ++k) acc[k] += part[t][k];
}

// Plain C entry for R's .C(): weighted survival curves of a two-normal mixture with its own mean and SD
// per component (the mixture and source-mixture families). th [S, R, J]; mu, sig, lam, mu2, sig2 [S, R];
// u [U]; w [R]; dims = {S, R, J, U, path, threads, link}; ext as for the detection kernel.
// acc [S, U] += w_r * (1 - (1 - lam) Phi((c(u) - mu) / sig) - lam Phi((c(u) - mu2) / sig2)).
extern "C" void bayesroc_r_mix_survival(const double* th, const double* mu, const double* sig,
                                        const double* lam, const double* mu2, const double* sig2, const double* u,
                                        const double* w, const int* dims, const double* ext,
                                        double* acc) {
  const int64_t S = dims[0], R = dims[1], J = dims[2], U = dims[3];
  const int path = dims[4], threads = dims[5], link = dims[6];
  const double e = *ext;
  std::vector<int> lo(U);
  std::vector<double> fr(U);
  for (int64_t j = 0; j < U; ++j) {
    double f = std::floor(u[j]);
    int l = static_cast<int>(f < J ? f : J);
    lo[j] = l;
    fr[j] = u[j] - l;
  }
  int64_t T = std::max<int64_t>(1, std::min<int64_t>(threads, R));
  std::vector<std::vector<double>> part(T, std::vector<double>(size_t(S) * size_t(U), 0.0));
  const int64_t Up = (U + 3) & ~int64_t(3);
  auto work = [&](int64_t t, int64_t r0, int64_t r1) {
    using bvn::splat;
    double* out = part[t].data();
    std::vector<double> knots(J + 2), cu(Up, 0.0);
    for (int64_t r = r0; r < r1; ++r) {
      const double wr = w[r];
      if (wr == 0.0) continue;
      for (int64_t s = 0; s < S; ++s) {
        const int64_t sr = s + S * r;
        const bvn::v4d vm = splat(mu[sr]), vis = splat(1.0 / sig[sr]);
        const bvn::v4d vl = splat(lam[sr]), vm2 = splat(mu2[sr]), vis2 = splat(1.0 / sig2[sr]);
        if (path) {
          for (int64_t j = 0; j < J; ++j) knots[j + 1] = th[sr + S * R * j];
          knots[0] = knots[1] - e;
          knots[J + 1] = knots[J] + e;
        }
        for (int64_t j = 0; j < U; ++j) cu[j] = path ? knots[lo[j]] * (1.0 - fr[j]) + knots[lo[j] + 1] * fr[j] : u[j];
        for (int64_t j = 0; j < Up; j += 4) {
          bvn::v4d c;
          std::memcpy(&c, &cu[j], 32);
          bvn::v4d sv = (splat(1.0) - vl) * vlink_surv((c - vm) * vis, link) + vl * vlink_surv((c - vm2) * vis2, link);
          accumulate_surv(out, s, S, j, U, wr, sv);
        }
      }
    }
  };
  if (T == 1) {
    work(0, 0, R);
  } else {
    const int64_t per = (R + T - 1) / T;
    std::vector<std::thread> pool;
    for (int64_t t = 1; t < T; ++t)
      if (t * per < R) pool.emplace_back(work, t, t * per, std::min(R, (t + 1) * per));
    work(0, 0, std::min(R, per));
    for (auto& th_ : pool) th_.join();
  }
  for (int64_t t = 0; t < T; ++t)
    for (int64_t k = 0; k < S * U; ++k) acc[k] += part[t][k];
}

// Plain C entry for R's .C(): weighted survival curves of the SOURCE response of the vrdp2d
// family (both components univariate given the recollection state). th1 [S, R, J1] detection
// thresholds; th2 [S, R, K1, J2] source thresholds per detection response when `varying`, else
// [S, R, J2]; muF, sig1, muR, lam, mu2, sig2 [S, R] (new items already carry mu 0, sig 1, lam 0);
// u [U]; w [R]; dims = {S, R, J1, J2, U, varying, path, threads, per_trial}; ext as for the detection
// kernel. per_trial: acc is [S, R, U] and holds each trial's own survival (w still applied).
// acc [S, U] += w_r * sum_j [ (1 - lam) P_F(j) (1 - Phi(c_j(u))) + lam P_R(j) (1 - Phi((c_j(u) - mu2) / sig2)) ].
extern "C" void bayesroc_r_src_survival(const double* th1, const double* th2, const double* muF,
                                        const double* sig1, const double* muR, const double* lam,
                                        const double* mu2, const double* sig2, const double* u,
                                        const double* w, const int* dims, const double* ext,
                                        double* acc) {
  const int64_t S = dims[0], R = dims[1], J1 = dims[2], J2 = dims[3], U = dims[4];
  const int varying = dims[5], path = dims[6], threads = dims[7], per_trial = dims[8];
  const int64_t K1 = J1 + 1;
  const double e = *ext, rs2 = 0.7071067811865476;
  std::vector<int> lo(U);
  std::vector<double> fr(U);
  for (int64_t j = 0; j < U; ++j) {
    double f = std::floor(u[j]);
    int l = static_cast<int>(f < J2 ? f : J2);
    lo[j] = l;
    fr[j] = u[j] - l;
  }
  const int64_t SR = S * R;
  int64_t T = std::max<int64_t>(1, std::min<int64_t>(threads, R));
  const int64_t oq = per_trial ? S * R : S;
  std::vector<std::vector<double>> part(T, std::vector<double>(per_trial ? 0 : size_t(S) * size_t(U), 0.0));
  auto Phi = [rs2](double z) { return 0.5 * std::erfc(-z * rs2); };
  auto work = [&](int64_t t, int64_t r0, int64_t r1) {
    double* out = per_trial ? acc : part[t].data();
    std::vector<double> knots(J2 + 2), pF(K1), pR(K1);
    for (int64_t r = r0; r < r1; ++r) {
      const double wr = w[r];
      if (wr == 0.0) continue;
      for (int64_t s = 0; s < S; ++s) {
        const int64_t sr = s + S * r, ob = per_trial ? sr : s;
        const double mF = muF[sr], mR = muR[sr], is1 = 1.0 / sig1[sr], l = lam[sr];
        const double m2 = mu2[sr], is2 = 1.0 / sig2[sr];
        // detection-bin weights of the two components
        double FprevF = 0.0, FprevR = 0.0;
        for (int64_t j = 0; j < K1; ++j) {
          double FF = 1.0, FR = 1.0;
          if (j < J1) {
            const double y = th1[sr + SR * j];
            FF = Phi((y - mF) * is1);
            FR = Phi((y - mR) * is1);
          }
          pF[j] = (1.0 - l) * (FF - FprevF);
          pR[j] = l * (FR - FprevR);
          FprevF = FF; FprevR = FR;
        }
        if (!varying || !path) {
          // the same source criteria for every detection response: one pass with the summed weights
          double sumF = 0.0, sumR = 0.0;
          for (int64_t j = 0; j < K1; ++j) { sumF += pF[j]; sumR += pR[j]; }
          if (path) {
            for (int64_t k = 0; k < J2; ++k) knots[k + 1] = th2[sr + SR * k];
            knots[0] = knots[1] - e;
            knots[J2 + 1] = knots[J2] + e;
          }
          for (int64_t q = 0; q < U; ++q) {
            const double c = path ? knots[lo[q]] * (1.0 - fr[q]) + knots[lo[q] + 1] * fr[q] : u[q];
            const double sv = sumF * (1.0 - Phi(c)) + sumR * (1.0 - Phi((c - m2) * is2));
            if (sv > 0.0) out[ob + oq * q] += wr * sv;
          }
        } else {
          for (int64_t j = 0; j < K1; ++j) {
            if (pF[j] == 0.0 && pR[j] == 0.0) continue;
            if (path) {
              for (int64_t k = 0; k < J2; ++k) knots[k + 1] = th2[sr + SR * (j + K1 * k)];
              knots[0] = knots[1] - e;
              knots[J2 + 1] = knots[J2] + e;
            }
            for (int64_t q = 0; q < U; ++q) {
              const double c = path ? knots[lo[q]] * (1.0 - fr[q]) + knots[lo[q] + 1] * fr[q] : u[q];
              const double sv = pF[j] * (1.0 - Phi(c)) + pR[j] * (1.0 - Phi((c - m2) * is2));
              if (sv > 0.0) out[ob + oq * q] += wr * sv;
            }
          }
        }
      }
    }
  };
  if (T == 1) {
    work(0, 0, R);
  } else {
    const int64_t per = (R + T - 1) / T;
    std::vector<std::thread> pool;
    for (int64_t t = 1; t < T; ++t)
      if (t * per < R) pool.emplace_back(work, t, t * per, std::min(R, (t + 1) * per));
    work(0, 0, std::min(R, per));
    for (auto& th_ : pool) th_.join();
  }
  if (!per_trial)
    for (int64_t t = 0; t < T; ++t)
      for (int64_t k = 0; k < S * U; ++k) acc[k] += part[t][k];
}

// Plain C entry for R's .C(): weighted survival curves of the SOURCE response of the Gaussian bivariate
// families (bivariate_sdt, bivariate_dp; bounded or not). Per (draw, trial): mu1, mu2, sig1, sig2, rho [S, R]
// with the A/B signs and the bounded flips applied (as .biv_trial_setup), old [S, R] (0 new, 1 source A, 2
// source B; the bounded model applies to studied items only), lam, lam2 [S, R] (dp only). th1 [S, R, J1]
// detection thresholds; th2 [S, R, K1, J2] when `varying`, else [S, R, J2]; u [U]; w [R]; dims = {S, R, J1, J2,
// U, varying, path, threads, bounded, dp, per_trial}; per_trial: acc is [S, R, U], each trial's own survival (w
// still applied). Otherwise acc [S, U] += w_r * survival.
//
// For each detection bin j and criterion c_j(u) the cell P(y in bin j, src > c) is the bounded cell with an
// infinite upper source edge. Its bivariate part P(x in [xlo, xhi], Z2 <= z2) (standardised item axis x,
// source axis Z2) is the cumulative integral over the SOURCE axis
//   C(z2) = int_{-6.5}^{z2} phi(y) [Phi((xhi - rho y) / sc) - Phi((xlo - rho y) / sc)] dy,   sc = sqrt(1 - rho^2),
// whose integrand is sampled once per (draw, trial) at the 8 Gauss-Legendre nodes of coarse segments shared by
// every bin and every criterion (segment width <= min(0.5, sc / (2 |rho|)) on |y| < 4, <= min(1, sc / (2 |rho|))
// beyond): the Phi of each finite edge is one SIMD pass over the nodes. Per segment the integrand's degree-7
// interpolant is integrated exactly, so C at any z2 is a prefix sum plus a Horner evaluation. Absolute error
// <= 1e-10 for |rho| <= 0.98 (the 6.5 clip costs 4e-11). With criteria shared by the detection responses (not
// varying, or the absolute grid of the distribution plots) the sum over the bins is the marginal: one pass.
// dp adds the recollection terms at the top detection bin.
static const double kGL8x[8] = {-0.9602898564975363, -0.7966664774136267, -0.525532409916329, -0.18343464249564984, 0.18343464249564984, 0.525532409916329, 0.7966664774136267, 0.9602898564975363};
static const double kGL8w[8] = {0.10122853629037562, 0.22238103445337473, 0.3137066458778876, 0.36268378337836205, 0.36268378337836205, 0.3137066458778876, 0.22238103445337473, 0.10122853629037562};
// c_k = sum_i kVinv8[8 k + i] f_i: monomial coefficients on [-1, 1] of the degree-7 interpolant through the nodes
static const double kVinv8[64] = {-0.017873231832895572, 0.0691757109831199, -0.17531514186002267, 0.6240126627098022, 0.6240126627098035, -0.17531514186002367, 0.0691757109831177, -0.01787323183289502,
                                  0.01861233013340499, -0.08683145700781747, 0.33359529983684305, -3.4018255996797278, 3.4018255996797238, -0.3335952998368486, 0.08683145700781977, -0.01861233013340649,
                                  0.6240543541747046, -2.3813321088208705, 5.676569223414572, -3.919291468768398, -3.9192914687683933, 5.67656922341455, -2.3813321088208625, 0.6240543541747018,
                                  -0.649860404077151, 2.9891205119509783, -10.801558793145155, 21.36614663100684, -21.366146631006853, 10.80155879314519, -2.989120511950997, 0.6498604040771583,
                                  -2.8621663107795268, 9.944755767918583, -14.158844036566009, 7.076254579426943, 7.076254579426943, -14.158844036565982, 9.944755767918567, -2.862166310779522,
                                  2.980523319509643, -12.482959996263471, 26.941904570300636, -38.576435089651675, 38.5764350896517, -26.941904570300686, 12.482959996263503, -2.9805233195096528,
                                  3.0303185085302786, -8.072112844132327, 8.902232716817863, -3.8604383812158085, -3.860438381215816, 8.902232716817856, -8.07211284413232, 3.0303185085302764,
                                  -3.1556289885042896, 10.132361625580524, -16.939455205503158, 21.045307084278463, -21.045307084278473, 16.93945520550318, -10.132361625580538, 3.1556289885042936};
// SIMD normal CDF without the derivative of bvn::vacc_phi (the same Chebyshev tail fit)
static inline bvn::v4d vphi_p(bvn::v4d x) {
  using bvn::splat;
  bvn::v4d ax = bvn::vabs(x);
  bvn::v4d t = splat(1.0) / (splat(1.0) + splat(0.2316419) * ax);
  bvn::v4d u = (splat(2.0) * t - splat(bvn::kPhiA + bvn::kPhiB)) * splat(1.0 / (bvn::kPhiB - bvn::kPhiA));
  bvn::v4d q = splat(0.3989422804014327) * bvn::vexp(splat(-0.5) * x * x) * t * bvn::clenshaw<19>(bvn::kPhiC, u);
  return bvn::vsel(x >= splat(0.0), splat(1.0) - q, q);
}
// Phi over a padded array (n a multiple of 4): out[i] = Phi((shift - scale_y * y[i]) * inv)
static inline void vphi_affine(int64_t n, const double* y, double shift, double scale_y, double inv, double* out) {
  const bvn::v4d vs = bvn::splat(shift), vr = bvn::splat(scale_y), vi = bvn::splat(inv);
  for (int64_t i = 0; i < n; i += 4) {
    bvn::v4d yv;
    std::memcpy(&yv, y + i, 32);
    bvn::v4d p = vphi_p((vs - vr * yv) * vi);
    std::memcpy(out + i, &p, 32);
  }
}
extern "C" void bayesroc_r_src_survival_gauss(const double* th1, const double* th2, const double* mu1,
                                              const double* mu2, const double* sig1, const double* sig2,
                                              const double* rho, const double* old, const double* lam,
                                              const double* lam2, const double* u, const double* w,
                                              const int* dims, const double* ext, double* acc) {
  const int64_t S = dims[0], R = dims[1], J1 = dims[2], J2 = dims[3], U = dims[4];
  const int varying = dims[5], path = dims[6], threads = dims[7], bounded = dims[8], dp = dims[9];
  const int per_trial = dims[10];
  const bool shared = !(varying && path);
  const int64_t K1 = J1 + 1, SR = S * R, NE = J1 + 1;
  const double e = *ext, rs2 = 0.7071067811865476, kInvSqrt2Pi = 0.3989422804014327, YCLIP = 6.5, YCORE = 4.0;
  const int64_t Up = (U + 3) & ~int64_t(3);
  std::vector<int> lo(U);
  std::vector<double> fr(U);
  for (int64_t j = 0; j < U; ++j) {
    double f = std::floor(u[j]);
    int l = static_cast<int>(f < J2 ? f : J2);
    lo[j] = l;
    fr[j] = u[j] - l;
  }
  auto Phi = [rs2](double z) { return 0.5 * std::erfc(-z * rs2); };
  int64_t T = std::max<int64_t>(1, std::min<int64_t>(threads, R));
  const int64_t oq = per_trial ? S * R : S;
  std::vector<std::vector<double>> part(T, std::vector<double>(per_trial ? 0 : size_t(S) * size_t(U), 0.0));
  auto work = [&](int64_t t, int64_t r0, int64_t r1) {
    double* out = per_trial ? acc : part[t].data();
    std::vector<double> knots(J2 + 2), cu(Up, 0.0), z2(Up, 0.0), Pz2(Up, 0.0), Psb(Up, 0.0), I(Up, 0.0), Imarg(Up, 0.0);
    std::vector<double> br, ys, phis, E, coef, A0, prefix;
    double Fv[8];
    int64_t NN = 0;
    int nseg = 0;
    // C(z2[q]) for the bracket [edge lo_e, edge hi_e] (-1: the infinite edge) into I, for every q
    auto cumulative = [&](int lo_e, int hi_e, std::vector<double>& I) {
      const double* Ehi = hi_e >= 0 ? &E[size_t(hi_e) * NN] : nullptr;
      const double* Elo = lo_e >= 0 ? &E[size_t(lo_e) * NN] : nullptr;
      for (int g = 0; g < nseg; ++g) {
        const double hh = 0.5 * (br[g + 1] - br[g]);
        double tot = 0.0;
        for (int i = 0; i < 8; ++i) {
          const int64_t n = 8 * g + i;
          const double f = phis[n] * ((Ehi ? Ehi[n] : 1.0) - (Elo ? Elo[n] : 0.0));
          Fv[i] = f;
          tot += kGL8w[i] * f;
        }
        prefix[g + 1] = prefix[g] + hh * tot;
        double am1 = 0.0;
        for (int k = 0; k < 8; ++k) {
          double c = 0.0;
          for (int i = 0; i < 8; ++i) c += kVinv8[8 * k + i] * Fv[i];
          const double d = hh * c / (k + 1);
          coef[8 * g + k] = d;
          am1 += (k & 1) ? d : -d;
        }
        A0[g] = am1;
      }
      int g = 0;
      for (int64_t q = 0; q < U; ++q) {
        const double z = z2[q];
        if (z <= br[0]) { I[q] = 0.0; continue; }
        if (z >= br[nseg]) { I[q] = prefix[nseg]; continue; }
        while (g > 0 && z < br[g]) --g;
        while (z >= br[g + 1]) ++g;
        const double tt = 2.0 * (z - br[g]) / (br[g + 1] - br[g]) - 1.0;
        const double* d = &coef[8 * g];
        double a = d[7];
        for (int k = 6; k >= 0; --k) a = a * tt + d[k];
        I[q] = prefix[g] + a * tt - A0[g];
      }
    };
    for (int64_t r = r0; r < r1; ++r) {
      const double wr = w[r];
      if (wr == 0.0) continue;
      for (int64_t s = 0; s < S; ++s) {
        const int64_t sr = s + S * r, ob = per_trial ? sr : s;
        const double m1 = mu1[sr], m2 = mu2[sr], s1 = sig1[sr], s2 = sig2[sr], rh = rho[sr];
        const bool isold = old[sr] > 0.5, dep = std::fabs(rh) >= 1e-10, bnd = bounded && isold && dep;
        const double l = dp ? lam[sr] : 0.0, l2 = dp ? lam2[sr] : 0.0;
        const double sc = std::sqrt(std::max(1.0 - rh * rh, 1e-300));
        double cp = 0.0, zcp = 0.0, scond = 1.0, Pzcp = 0.0;
        if (bnd) {
          cp = m1 - s1 * m2 / (rh * s2);
          zcp = (cp - m1) / s1;
          scond = s2 * sc;
          Pzcp = Phi(zcp);
        }
        const bool need_nodes = dep && (!shared || bnd);
        if (need_nodes) {
          // coarse segments of the source axis, finer where the conditional CDF is steep along y
          const double L = std::max(sc / std::fabs(rh), 0.05);
          const double hI = std::min(0.5, 0.5 * L), hT = std::min(1.0, 0.5 * L);
          br.clear();
          br.push_back(-YCLIP);
          double y = -YCLIP;
          while (y < YCLIP - 1e-12) {
            double h;
            if (y < -YCORE - 1e-12) h = std::min(hT, -YCORE - y);
            else if (y < YCORE - 1e-12) h = std::min(hI, YCORE - y);
            else h = std::min(hT, YCLIP - y);
            y += h;
            br.push_back(y);
          }
          nseg = static_cast<int>(br.size()) - 1;
          NN = 8 * nseg;
          if (static_cast<int64_t>(ys.size()) < NN) {
            ys.resize(NN); phis.resize(NN); E.resize(size_t(NE) * NN); coef.resize(NN);
          }
          if (static_cast<int>(A0.size()) < nseg) { A0.resize(nseg); prefix.resize(nseg + 1); }
          prefix[0] = 0.0;
          for (int g = 0; g < nseg; ++g) {
            const double hh = 0.5 * (br[g + 1] - br[g]), m = br[g] + hh;
            for (int i = 0; i < 8; ++i) ys[8 * g + i] = m + hh * kGL8x[i];
          }
          for (int64_t n = 0; n < NN; n += 4) {
            bvn::v4d yv;
            std::memcpy(&yv, &ys[n], 32);
            bvn::v4d p = bvn::splat(kInvSqrt2Pi) * bvn::vexp(bvn::splat(-0.5) * yv * yv);
            std::memcpy(&phis[n], &p, 32);
          }
          // Phi((edge - rho y) / sc) at the nodes for every finite edge: the detection thresholds and the crossing
          const double isc = 1.0 / sc;
          if (!shared)
            for (int64_t ed = 0; ed < J1; ++ed)
              vphi_affine(NN, ys.data(), (th1[sr + SR * ed] - m1) / s1, rh, isc, &E[size_t(ed) * NN]);
          if (bnd) vphi_affine(NN, ys.data(), zcp, rh, isc, &E[size_t(J1) * NN]);
        }
        const int64_t nb = shared ? 1 : K1, jtop = nb - 1;
        for (int64_t j = 0; j < nb; ++j) {
          const double ylo = (!shared && j > 0) ? th1[sr + SR * (j - 1)] : -INFINITY;
          const double yhi = (!shared && j < J1) ? th1[sr + SR * j] : INFINITY;
          const double a1 = (ylo - m1) / s1, b1 = (yhi - m1) / s1;
          const double Pa1 = std::isfinite(ylo) ? Phi(a1) : 0.0, Pb1 = std::isfinite(yhi) ? Phi(b1) : 1.0, pbin = Pb1 - Pa1;
          const double wfam = dp ? (isold ? 1.0 - l : 1.0) : 1.0;
          if (path) {
            if (varying) for (int64_t k = 0; k < J2; ++k) knots[k + 1] = th2[sr + SR * (j + K1 * k)];
            else for (int64_t k = 0; k < J2; ++k) knots[k + 1] = th2[sr + SR * k];
            knots[0] = knots[1] - e;
            knots[J2 + 1] = knots[J2] + e;
            for (int64_t q = 0; q < U; ++q) cu[q] = knots[lo[q]] * (1.0 - fr[q]) + knots[lo[q] + 1] * fr[q];
          } else {
            for (int64_t q = 0; q < U; ++q) cu[q] = u[q];
          }
          for (int64_t q = 0; q < U; ++q) z2[q] = (cu[q] - m2) / s2;
          vphi_affine(Up, z2.data(), 0.0, -1.0, 1.0, Pz2.data());
          if (bnd) vphi_affine(Up, cu.data(), 0.0, -1.0, 1.0 / scond, Psb.data());
          // bivariate part of the familiarity cell: P(x in [xlo, xhi], Z2 <= z2) for every criterion
          const bool need = dep && (bnd ? yhi > cp : !shared);
          if (need) {
            const int lo_e = (bnd && ylo < cp) ? static_cast<int>(J1) : (std::isfinite(ylo) ? static_cast<int>(j - 1) : -1);
            const int hi_e = std::isfinite(yhi) ? static_cast<int>(j) : -1;
            cumulative(lo_e, hi_e, I);
          }
          if (dp && bnd && j == jtop) cumulative(static_cast<int>(J1), -1, Imarg);
          for (int64_t q = 0; q < U; ++q) {
            double sv;
            if (!bnd) {
              sv = (dep && !shared) ? pbin - I[q] : pbin * (1.0 - Pz2[q]);
            } else {
              const double src_below = 1.0 - Psb[q];
              if (yhi <= cp) sv = pbin * src_below;
              else if (ylo >= cp) sv = pbin - I[q];
              else {
                const double zlo_b = std::isfinite(ylo) ? a1 : -8.0;
                sv = (Pzcp - Phi(zlo_b)) * src_below + ((Pb1 - Pzcp) - I[q]);
              }
            }
            sv *= wfam;
            if (dp && isold && j == jtop) {
              double smarg;
              if (bnd) smarg = Pzcp * (1.0 - Psb[q]) + (1.0 - Pzcp) - Imarg[q];
              else smarg = 1.0 - Pz2[q];
              sv += l * (1.0 - l2) * smarg;
            }
            if (sv > 0.0) out[ob + oq * q] += wr * sv;
          }
        }
        if (dp && isold && old[sr] > 1.5) {
          const double corner = l * l2;
          if (corner > 0.0) for (int64_t q = 0; q < U; ++q) out[ob + oq * q] += wr * corner;
        }
      }
    }
  };
  if (T == 1) {
    work(0, 0, R);
  } else {
    const int64_t per = (R + T - 1) / T;
    std::vector<std::thread> pool;
    for (int64_t t = 1; t < T; ++t)
      if (t * per < R) pool.emplace_back(work, t, t * per, std::min(R, (t + 1) * per));
    work(0, 0, std::min(R, per));
    for (auto& th_ : pool) th_.join();
  }
  if (!per_trial)
    for (int64_t t = 0; t < T; ++t)
      for (int64_t k = 0; k < S * U; ++k) acc[k] += part[t][k];
}
