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

}

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
