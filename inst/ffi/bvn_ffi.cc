#include <algorithm>
#include <xmmintrin.h>
#include <atomic>
#include <memory>
#include <thread>
#include <vector>
#include <type_traits>

#include "xla/ffi/api/ffi.h"
#include "bayesroc/bvn_simd.h"
#include "bayesroc/tmap_kernel.h"

namespace ffi = xla::ffi;

namespace {

ffi::Error BvnRectImpl(ffi::Buffer<ffi::F64> z1lo, ffi::Buffer<ffi::F64> z1hi,
                       ffi::Buffer<ffi::F64> z2lo, ffi::Buffer<ffi::F64> z2hi,
                       ffi::Buffer<ffi::F64> rho, ffi::ResultBuffer<ffi::F64> p,
                       ffi::ResultBuffer<ffi::F64> d1lo, ffi::ResultBuffer<ffi::F64> d1hi,
                       ffi::ResultBuffer<ffi::F64> d2lo, ffi::ResultBuffer<ffi::F64> d2hi,
                       ffi::ResultBuffer<ffi::F64> drho) {
  bvn::rect_array(z1lo.element_count(), z1lo.typed_data(), z1hi.typed_data(), z2lo.typed_data(),
                  z2hi.typed_data(), rho.typed_data(), p->typed_data(), d1lo->typed_data(),
                  d1hi->typed_data(), d2lo->typed_data(), d2hi->typed_data(), drho->typed_data());
  return ffi::Error::Success();
}

ffi::Error BvnCdfImpl(ffi::Buffer<ffi::F64> h, ffi::Buffer<ffi::F64> k, ffi::Buffer<ffi::F64> rho,
                      ffi::ResultBuffer<ffi::F64> p, ffi::ResultBuffer<ffi::F64> dh,
                      ffi::ResultBuffer<ffi::F64> dk, ffi::ResultBuffer<ffi::F64> drho) {
  bvn::cdf_array(h.element_count(), h.typed_data(), k.typed_data(), rho.typed_data(),
                 p->typed_data(), dh->typed_data(), dk->typed_data(), drho->typed_data());
  return ffi::Error::Success();
}

ffi::Error VrdpImpl(ffi::Buffer<ffi::F64> i0, ffi::Buffer<ffi::F64> i1, ffi::Buffer<ffi::F64> i2, ffi::Buffer<ffi::F64> i3, ffi::Buffer<ffi::F64> i4, ffi::Buffer<ffi::F64> i5, ffi::Buffer<ffi::F64> i6, ffi::Buffer<ffi::F64> i7, ffi::Buffer<ffi::F64> i8, ffi::Buffer<ffi::F64> i9, ffi::Buffer<ffi::F64> i10, ffi::Buffer<ffi::F64> i11, ffi::Buffer<ffi::F64> i12, ffi::Buffer<ffi::F64> i13,
                    ffi::ResultBuffer<ffi::F64> o0, ffi::ResultBuffer<ffi::F64> o1, ffi::ResultBuffer<ffi::F64> o2, ffi::ResultBuffer<ffi::F64> o3, ffi::ResultBuffer<ffi::F64> o4, ffi::ResultBuffer<ffi::F64> o5, ffi::ResultBuffer<ffi::F64> o6, ffi::ResultBuffer<ffi::F64> o7, ffi::ResultBuffer<ffi::F64> o8, ffi::ResultBuffer<ffi::F64> o9, ffi::ResultBuffer<ffi::F64> o10) {
  const double* in[14] = {i0.typed_data(), i1.typed_data(), i2.typed_data(), i3.typed_data(), i4.typed_data(), i5.typed_data(), i6.typed_data(), i7.typed_data(), i8.typed_data(), i9.typed_data(), i10.typed_data(), i11.typed_data(), i12.typed_data(), i13.typed_data()};
  double* out[11] = {o0->typed_data(), o1->typed_data(), o2->typed_data(), o3->typed_data(), o4->typed_data(), o5->typed_data(), o6->typed_data(), o7->typed_data(), o8->typed_data(), o9->typed_data(), o10->typed_data()};
  bvn::vrdp_array(i0.element_count(), in, out);
  return ffi::Error::Success();
}

ffi::Error BvnCdf2Impl(ffi::Buffer<ffi::F64> i0, ffi::Buffer<ffi::F64> i1, ffi::Buffer<ffi::F64> i2, ffi::Buffer<ffi::F64> i3, ffi::Buffer<ffi::F64> i4, ffi::Buffer<ffi::F64> i5,
                        ffi::ResultBuffer<ffi::F64> o0, ffi::ResultBuffer<ffi::F64> o1, ffi::ResultBuffer<ffi::F64> o2, ffi::ResultBuffer<ffi::F64> o3, ffi::ResultBuffer<ffi::F64> o4, ffi::ResultBuffer<ffi::F64> o5, ffi::ResultBuffer<ffi::F64> o6, ffi::ResultBuffer<ffi::F64> o7, ffi::ResultBuffer<ffi::F64> o8, ffi::ResultBuffer<ffi::F64> o9, ffi::ResultBuffer<ffi::F64> o10, ffi::ResultBuffer<ffi::F64> o11, ffi::ResultBuffer<ffi::F64> o12, ffi::ResultBuffer<ffi::F64> o13, ffi::ResultBuffer<ffi::F64> o14, ffi::ResultBuffer<ffi::F64> o15, ffi::ResultBuffer<ffi::F64> o16, ffi::ResultBuffer<ffi::F64> o17) {
  const double* in[6] = {i0.typed_data(), i1.typed_data(), i2.typed_data(), i3.typed_data(), i4.typed_data(), i5.typed_data()};
  double* out[18] = {o0->typed_data(), o1->typed_data(), o2->typed_data(), o3->typed_data(), o4->typed_data(), o5->typed_data(), o6->typed_data(), o7->typed_data(), o8->typed_data(), o9->typed_data(), o10->typed_data(), o11->typed_data(), o12->typed_data(), o13->typed_data(), o14->typed_data(), o15->typed_data(), o16->typed_data(), o17->typed_data()};
  bvn::cdf_multi_array<2, 1>(i0.element_count(), in, out);
  return ffi::Error::Success();
}

ffi::Error BvnCdf4Impl(ffi::Buffer<ffi::F64> i0, ffi::Buffer<ffi::F64> i1, ffi::Buffer<ffi::F64> i2, ffi::Buffer<ffi::F64> i3, ffi::Buffer<ffi::F64> i4, ffi::Buffer<ffi::F64> i5, ffi::Buffer<ffi::F64> i6, ffi::Buffer<ffi::F64> i7, ffi::Buffer<ffi::F64> i8, ffi::Buffer<ffi::F64> i9, ffi::Buffer<ffi::F64> i10,
                        ffi::ResultBuffer<ffi::F64> o0, ffi::ResultBuffer<ffi::F64> o1, ffi::ResultBuffer<ffi::F64> o2, ffi::ResultBuffer<ffi::F64> o3, ffi::ResultBuffer<ffi::F64> o4, ffi::ResultBuffer<ffi::F64> o5, ffi::ResultBuffer<ffi::F64> o6, ffi::ResultBuffer<ffi::F64> o7, ffi::ResultBuffer<ffi::F64> o8, ffi::ResultBuffer<ffi::F64> o9, ffi::ResultBuffer<ffi::F64> o10, ffi::ResultBuffer<ffi::F64> o11, ffi::ResultBuffer<ffi::F64> o12, ffi::ResultBuffer<ffi::F64> o13, ffi::ResultBuffer<ffi::F64> o14, ffi::ResultBuffer<ffi::F64> o15, ffi::ResultBuffer<ffi::F64> o16, ffi::ResultBuffer<ffi::F64> o17, ffi::ResultBuffer<ffi::F64> o18, ffi::ResultBuffer<ffi::F64> o19, ffi::ResultBuffer<ffi::F64> o20, ffi::ResultBuffer<ffi::F64> o21, ffi::ResultBuffer<ffi::F64> o22, ffi::ResultBuffer<ffi::F64> o23, ffi::ResultBuffer<ffi::F64> o24, ffi::ResultBuffer<ffi::F64> o25, ffi::ResultBuffer<ffi::F64> o26, ffi::ResultBuffer<ffi::F64> o27, ffi::ResultBuffer<ffi::F64> o28, ffi::ResultBuffer<ffi::F64> o29, ffi::ResultBuffer<ffi::F64> o30, ffi::ResultBuffer<ffi::F64> o31, ffi::ResultBuffer<ffi::F64> o32, ffi::ResultBuffer<ffi::F64> o33, ffi::ResultBuffer<ffi::F64> o34, ffi::ResultBuffer<ffi::F64> o35) {
  const double* in[11] = {i0.typed_data(), i1.typed_data(), i2.typed_data(), i3.typed_data(), i4.typed_data(), i5.typed_data(), i6.typed_data(), i7.typed_data(), i8.typed_data(), i9.typed_data(), i10.typed_data()};
  double* out[36] = {o0->typed_data(), o1->typed_data(), o2->typed_data(), o3->typed_data(), o4->typed_data(), o5->typed_data(), o6->typed_data(), o7->typed_data(), o8->typed_data(), o9->typed_data(), o10->typed_data(), o11->typed_data(), o12->typed_data(), o13->typed_data(), o14->typed_data(), o15->typed_data(), o16->typed_data(), o17->typed_data(), o18->typed_data(), o19->typed_data(), o20->typed_data(), o21->typed_data(), o22->typed_data(), o23->typed_data(), o24->typed_data(), o25->typed_data(), o26->typed_data(), o27->typed_data(), o28->typed_data(), o29->typed_data(), o30->typed_data(), o31->typed_data(), o32->typed_data(), o33->typed_data(), o34->typed_data(), o35->typed_data()};
  bvn::cdf_multi_array<4, 2>(i0.element_count(), in, out);
  return ffi::Error::Success();
}

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

// groups are split into at most kTmapChunks ranges of whole 4-group tiles
template <typename F>
void tmap_for(ffi::ThreadPool& pool, int64_t n, F body) {
  int64_t tiles = (n + 3) / 4;
  int64_t per = std::max<int64_t>(64, (tiles + kTmapChunks - 1) / kTmapChunks);
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

// Lam [n4, D, D, 4], Q [D, D], s, delta, zeta [D, n] -> u [D, n], ld [n]
template <typename T>
ffi::Error TmapFwdImpl(ffi::ThreadPool pool, ffi::AnyBuffer Lam, ffi::AnyBuffer Q, ffi::AnyBuffer s,
                       ffi::AnyBuffer delta, ffi::AnyBuffer zeta, ffi::Result<ffi::AnyBuffer> u,
                       ffi::Result<ffi::AnyBuffer> ld) {
  const int D = static_cast<int>(Lam.dimensions()[1]);
  const int64_t n = s.dimensions()[1];
  tmap::FwdFn f = tmap::forward_for(D);
  if (!f) return ffi::Error::InvalidArgument("bayesroc: transport block too wide for the native kernel");
  DBuf<T> lam(Lam), q(Q), ss(s), dd(delta), z(zeta);
  DOut<T> uo(u), ldo(ld);
  std::vector<double> r(size_t(D) * n);
  for (int64_t k = 0; k < int64_t(D) * n; ++k) r[k] = ss.ptr[k];
  const double *pl = lam.ptr, *pq = q.ptr, *pd = dd.ptr, *pz = z.ptr;
  double *pr = r.data(), *pu = uo.ptr, *pld = ldo.ptr;
  tmap_for(pool, n, [=](int64_t, int64_t g0, int64_t g1) {
    tmap::coupling(D, n, g0, g1, pl, pd, pr);
    f(g0, g1, n, pl, pq, pr, pz, pu, pld);
  });
  return ffi::Error::Success();
}

// Lam, Q, s, delta, zeta as above, ubar [D, n], ldbar [n] -> zetabar [D, n], deltabar [D, n],
// Qbar [kTmapChunks, D, D] (lower-triangle cotangent of Q, per chunk; sum over chunks)
template <typename T>
ffi::Error TmapBwdImpl(ffi::ThreadPool pool, ffi::AnyBuffer Lam, ffi::AnyBuffer Q, ffi::AnyBuffer s,
                       ffi::AnyBuffer delta, ffi::AnyBuffer zeta, ffi::AnyBuffer ubar,
                       ffi::AnyBuffer ldbar, ffi::Result<ffi::AnyBuffer> zetabar,
                       ffi::Result<ffi::AnyBuffer> deltabar, ffi::Result<ffi::AnyBuffer> Qbar) {
  const int D = static_cast<int>(Lam.dimensions()[1]);
  const int64_t n = s.dimensions()[1];
  tmap::BwdFn f = tmap::backward_for(D);
  if (!f) return ffi::Error::InvalidArgument("bayesroc: transport block too wide for the native kernel");
  DBuf<T> lam(Lam), q(Q), ss(s), dd(delta), z(zeta), ub(ubar), lb(ldbar);
  DOut<T> zo(zetabar), dob(deltabar), qo(Qbar);
  std::vector<double> r(size_t(D) * n);
  for (int64_t k = 0; k < int64_t(D) * n; ++k) r[k] = ss.ptr[k];
  for (int64_t k = 0; k < int64_t(kTmapChunks) * D * D; ++k) qo.ptr[k] = 0.0;
  const double *pl = lam.ptr, *pq = q.ptr, *pd = dd.ptr, *pz = z.ptr, *pub = ub.ptr, *plb = lb.ptr;
  double *pr = r.data(), *pzb = zo.ptr, *pdb = dob.ptr, *pqb = qo.ptr;
  tmap_for(pool, n, [=](int64_t c, int64_t g0, int64_t g1) {
    tmap::coupling(D, n, g0, g1, pl, pd, pr);
    f(g0, g1, n, pl, pq, pr, pz, pub, plb, pzb, pdb, pqb + c * D * D);
    tmap::coupling_adjoint(D, n, g0, g1, pl, pdb);
  });
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
      for (int j = 0; j < 9; ++j) o[j][k] = 0.0;
      continue;
    }
    bvn_tail::CdpCell c = bvn_tail::cdp_cell(static_cast<int>(i[1][k]), i[2][k], i[3][k], i[4][k], i[5][k],
                                             i[6][k], i[7][k], i[8][k], i[9][k]);
    o[0][k] = c.lp;
    for (int j = 0; j < 8; ++j) o[1 + j][k] = c.d[j];
  }
}

template <int NI, int NO, void (*K)(int64_t, const double* const*, double* const*)>
ffi::Error F64Impl(ffi::RemainingArgs args, ffi::RemainingRets rets) {
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
  K(n, in, out);
  return ffi::Error::Success();
}

struct FlushDenormals {
  unsigned saved;
  FlushDenormals() : saved(_mm_getcsr()) { _mm_setcsr(saved | 0x8040); }
  ~FlushDenormals() { _mm_setcsr(saved); }
};

template <int NI, int NO, void (*K)(int64_t, const double* const*, double* const*)>
ffi::Error F32Impl(ffi::RemainingArgs args, ffi::RemainingRets rets) {
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
  thread_local std::vector<double> buf;
  if (buf.size() < size_t((NI + NO) * n)) buf.resize((NI + NO) * n);
  FlushDenormals ftz;
  const double* din[NI];
  double* dout[NO];
  for (int a = 0; a < NI; ++a) {
    double* d = buf.data() + a * n;
    for (int64_t k = 0; k < n; ++k) d[k] = fin[a][k];
    din[a] = d;
  }
  for (int a = 0; a < NO; ++a) dout[a] = buf.data() + (NI + a) * n;
  K(n, din, dout);
  for (int a = 0; a < NO; ++a)
    for (int64_t k = 0; k < n; ++k) fout[a][k] = static_cast<float>(dout[a][k]);
  return ffi::Error::Success();
}

}

#define BAYESROC_F32(NAME, NI, NO, K)                                               \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(NAME, (F32Impl<NI, NO, K>),                           \
                                ffi::Ffi::Bind().RemainingArgs().RemainingRets())

BAYESROC_F32(BayesrocBvnRectF32, 5, 6, rect_k);
BAYESROC_F32(BayesrocBvnCdfF32, 3, 4, cdf_k);
BAYESROC_F32(BayesrocVrdp2dF32, 14, 11, bvn::vrdp_array);
BAYESROC_F32(BayesrocBvnCdf2F32, 6, 18, (bvn::cdf_multi_array<2, 1>));
BAYESROC_F32(BayesrocBvnCdf4F32, 11, 36, (bvn::cdf_multi_array<4, 2>));
BAYESROC_F32(BayesrocCdpCellF32, 10, 9, cdp_k);

XLA_FFI_DEFINE_HANDLER_SYMBOL(BayesrocCdpCell, (F64Impl<10, 9, cdp_k>),
                              ffi::Ffi::Bind().RemainingArgs().RemainingRets());

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

XLA_FFI_DEFINE_HANDLER_SYMBOL(BayesrocBvnRect, BvnRectImpl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(BayesrocBvnCdf, BvnCdfImpl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(BayesrocVrdp2d, VrdpImpl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(BayesrocBvnCdf2, BvnCdf2Impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(BayesrocBvnCdf4, BvnCdf4Impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>());

#define BAYESROC_TMAP(NAME, IMPL, NARG, NRET)                                         XLA_FFI_DEFINE_HANDLER_SYMBOL(NAME, IMPL, ffi::Ffi::Bind().Ctx<ffi::ThreadPool>() NARG NRET)
#define A_ .Arg<ffi::AnyBuffer>()
#define R_ .Ret<ffi::AnyBuffer>()
BAYESROC_TMAP(BayesrocTmapFwd, TmapFwdImpl<double>, A_ A_ A_ A_ A_, R_ R_);
BAYESROC_TMAP(BayesrocTmapFwdF32, TmapFwdImpl<float>, A_ A_ A_ A_ A_, R_ R_);
BAYESROC_TMAP(BayesrocTmapBwd, TmapBwdImpl<double>, A_ A_ A_ A_ A_ A_ A_, R_ R_ R_);
BAYESROC_TMAP(BayesrocTmapBwdF32, TmapBwdImpl<float>, A_ A_ A_ A_ A_ A_ A_, R_ R_ R_);

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
    for (int64_t i = i0; i < i1; ++i) p[i] = 0.5 * std::erfc(-x[i] * 0.7071067811865476);
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
extern "C" void bayesroc_r_det_survival(const double* th, const double* mu, const double* sig,
                                        const double* lam, const double* mu2, const double* u,
                                        const double* w, const int* dims, const double* ext,
                                        double* acc) {
  const int64_t S = dims[0], R = dims[1], J = dims[2], U = dims[3];
  const int kind = dims[4], path = dims[5], threads = dims[6];
  const double e = *ext, rs2 = 0.7071067811865476;
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
  auto work = [&](int64_t t, int64_t r0, int64_t r1) {
    double* out = part[t].data();
    std::vector<double> knots(J + 2);
    for (int64_t r = r0; r < r1; ++r) {
      const double wr = w[r];
      if (wr == 0.0) continue;
      for (int64_t s = 0; s < S; ++s) {
        const int64_t sr = s + S * r;
        const double m = mu[sr], is = 1.0 / sig[sr];
        const double l = kind ? lam[sr] : 0.0, m2 = kind == 2 ? mu2[sr] : 0.0;
        if (path) {
          for (int64_t j = 0; j < J; ++j) knots[j + 1] = th[sr + S * R * j];
          knots[0] = knots[1] - e;
          knots[J + 1] = knots[J] + e;
        }
        for (int64_t j = 0; j < U; ++j) {
          const double c = path ? knots[lo[j]] * (1.0 - fr[j]) + knots[lo[j] + 1] * fr[j] : u[j];
          double F = 0.5 * std::erfc(-(c - m) * is * rs2);
          if (kind == 1) F = (1.0 - l) * F;
          else if (kind == 2) F = (1.0 - l) * F + l * 0.5 * std::erfc(-(c - m2) * is * rs2);
          const double sv = 1.0 - F;
          if (sv > 0.0) out[s + S * j] += wr * sv;
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
