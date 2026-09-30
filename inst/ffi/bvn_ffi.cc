#include <algorithm>
#include <xmmintrin.h>
#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#include "xla/ffi/api/ffi.h"
#include "bayesroc/bvn_simd.h"

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
