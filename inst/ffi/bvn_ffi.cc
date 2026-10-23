#include <algorithm>
#include <limits>
#include <xmmintrin.h>
#include <atomic>
#include <memory>
#include <thread>
#include <vector>
#include <type_traits>

#include "xla/ffi/api/ffi.h"
#include "bayesroc/bvn_simd.h"

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

template <typename F>
void chunk_for(ffi::ThreadPool& pool, int64_t n, int64_t G, F body) {
  const int64_t nch = (n + G - 1) / G;
  pool_for(pool, nch, 1, [&](int64_t c0, int64_t c1) {
    for (int64_t c = c0; c < c1; ++c) body(c * G, std::min(n, (c + 1) * G));
  });
}

constexpr int64_t kCellChunk = 2048;

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

void rect_k(int64_t n, const double* const* i, double* const* o) {
  bvn::rect_array(n, i[0], i[1], i[2], i[3], i[4], o[0], o[1], o[2], o[3], o[4], o[5]);
}

void cdf_k(int64_t n, const double* const* i, double* const* o) {
  bvn::cdf_array(n, i[0], i[1], i[2], o[0], o[1], o[2], o[3]);
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
  chunk_for(pool, n, kCellChunk, [&](int64_t i0, int64_t i1) {
    const double* ci[NI];
    double* co[NO];
    for (int a = 0; a < NI; ++a) ci[a] = in[a] + i0;
    for (int a = 0; a < NO; ++a) co[a] = out[a] + i0;
    K(i1 - i0, ci, co);
  });
  return ffi::Error::Success();
}

struct FlushDenormals {
  unsigned saved;
  FlushDenormals() : saved(_mm_getcsr()) { _mm_setcsr(saved | 0x8040); }
  ~FlushDenormals() { _mm_setcsr(saved); }
};

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
  chunk_for(pool, n, kCellChunk, [&](int64_t i0, int64_t i1) {
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

}

#define BAYESROC_F32(NAME, NI, NO, K)                                               \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(NAME, (F32Impl<NI, NO, K>),                           \
                                ffi::Ffi::Bind().Ctx<ffi::ThreadPool>().RemainingArgs().RemainingRets())

BAYESROC_F32(BayesrocBvnRectF32, 5, 6, rect_k);
BAYESROC_F32(BayesrocBvnCdfF32, 3, 4, cdf_k);
BAYESROC_F32(BayesrocVrdp2dF32, 14, 11, bvn::vrdp_array);
BAYESROC_F32(BayesrocBvnCdf2F32, 6, 18, (bvn::cdf_multi_array<2, 1>));
BAYESROC_F32(BayesrocBvnCdf4F32, 11, 36, (bvn::cdf_multi_array<4, 2>));
BAYESROC_F32(BayesrocCdpCellF32, 11, 10, cdp_k);

#define BAYESROC_F64(NAME, NI, NO, K)                                               \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(NAME, (F64Impl<NI, NO, K>),                           \
                                ffi::Ffi::Bind().Ctx<ffi::ThreadPool>().RemainingArgs().RemainingRets())

BAYESROC_F64(BayesrocCdpCell, 11, 10, cdp_k);
BAYESROC_F64(BayesrocBvnRect, 5, 6, rect_k);
BAYESROC_F64(BayesrocBvnCdf, 3, 4, cdf_k);
BAYESROC_F64(BayesrocVrdp2d, 14, 11, bvn::vrdp_array);
BAYESROC_F64(BayesrocBvnCdf2, 6, 18, (bvn::cdf_multi_array<2, 1>));
BAYESROC_F64(BayesrocBvnCdf4, 11, 36, (bvn::cdf_multi_array<4, 2>));

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






#define BAYESROC_TMAP(NAME, IMPL, NARG, NRET)                                         XLA_FFI_DEFINE_HANDLER_SYMBOL(NAME, IMPL, ffi::Ffi::Bind().Ctx<ffi::ThreadPool>() NARG NRET)
#define A_ .Arg<ffi::AnyBuffer>()
#define R_ .Ret<ffi::AnyBuffer>()
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
// CDFs, the band's Phi2 shared with its neighbour), and any share below 1e-6 is replaced by the likelihood's
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
            if (c1 < 1e-6) c1 = std::exp(bvn_tail::cdp_cell(1, mR, dR, mF, dF, c_R, c_K, tlo, thi, cr).lp);
            if (c2 < 1e-6) c2 = std::exp(bvn_tail::cdp_cell(2, mR, dR, mF, dF, c_R, c_K, tlo, thi, cr).lp);
            if (c3 < 1e-6) c3 = std::exp(bvn_tail::cdp_cell(3, mR, dR, mF, dF, c_R, c_K, tlo, thi, cr).lp);
          } else {
            c2 = pband - c1;
            if (c1 < 1e-6 || c2 < 1e-6) {
              const double sc = dF * std::sqrt((1.0 - cr) * (1.0 + cr)), B = -(dR + cr * dF) / sc;
              const double Alo = std::isfinite(tlo) ? (tlo - mR - mF) / sc : -inf;
              const double Ahi = std::isfinite(thi) ? (thi - mR - mF) / sc : inf;
              bvn_tail::LinCond c{Alo, std::isfinite(tlo) ? B : 0.0, Ahi, std::isfinite(thi) ? B : 0.0};
              if (c1 < 1e-6) c1 = std::exp(bvn_tail::log_integral(c, zcR, inf));
              if (c2 < 1e-6) c2 = std::exp(bvn_tail::log_integral(c, -inf, zcR));
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

// Plain C entry for R's .C(): weighted survival curves of the detection response.
// th [S, R, J] thresholds, mu, sig, lam, mu2 [S, R], u [U], w [R]; dims = {S, R, J, U, kind,
// path, threads}. kind 0: normal(mu, sig); 1: (1 - lam) normal + lam at the top response;
// 2: (1 - lam) normal(mu, sig) + lam normal(mu2, sig). path 1: criterion j lies at u[j] along
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
        const bvn::v4d vl = splat(kind ? lam[sr] : 0.0), vm2 = splat(kind == 2 ? mu2[sr] : 0.0);
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
