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
