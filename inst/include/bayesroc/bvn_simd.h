#ifndef BAYESROC_BVN_SIMD_H
#define BAYESROC_BVN_SIMD_H

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "bvn_tail.h"

namespace bvn {

typedef double v4d __attribute__((vector_size(32)));
typedef int64_t v4i __attribute__((vector_size(32)));

inline v4d splat(double x) { return v4d{x, x, x, x}; }

inline v4d vsel(v4i m, v4d a, v4d b) {
  v4i ai, bi;
  std::memcpy(&ai, &a, 32);
  std::memcpy(&bi, &b, 32);
  v4i r = (m & ai) | (~m & bi);
  v4d out;
  std::memcpy(&out, &r, 32);
  return out;
}

inline v4d vabs(v4d x) {
  v4i xi;
  std::memcpy(&xi, &x, 32);
  xi &= (v4i){INT64_MAX, INT64_MAX, INT64_MAX, INT64_MAX};
  v4d out;
  std::memcpy(&out, &xi, 32);
  return out;
}

inline v4d vmin(v4d a, v4d b) { return vsel(a < b, a, b); }
inline v4d vmax(v4d a, v4d b) { return vsel(a > b, a, b); }

inline v4d vsqrt(v4d x) {
  v4d r;
  for (int l = 0; l < 4; ++l) r[l] = __builtin_sqrt(x[l]);
  return r;
}

inline v4d vexp(v4d x) {
  const v4d magic = splat(6755399441055744.0);
  x = vmin(vmax(x, splat(-708.0)), splat(709.0));
  v4d kf = (x * splat(1.4426950408889634) + magic) - magic;
  v4d r = (x - kf * splat(0.6931471805598903)) - kf * splat(5.497923018708371e-14);
  v4d p = splat(1.0 / 6227020800.0);
  p = p * r + splat(1.0 / 479001600.0);
  p = p * r + splat(1.0 / 39916800.0);
  p = p * r + splat(1.0 / 3628800.0);
  p = p * r + splat(1.0 / 362880.0);
  p = p * r + splat(1.0 / 40320.0);
  p = p * r + splat(1.0 / 5040.0);
  p = p * r + splat(1.0 / 720.0);
  p = p * r + splat(1.0 / 120.0);
  p = p * r + splat(1.0 / 24.0);
  p = p * r + splat(1.0 / 6.0);
  p = p * r + splat(0.5);
  p = p * r + splat(1.0);
  p = p * r + splat(1.0);
  v4d biased = kf + splat(1023.0 + 4503599627370496.0);
  v4i bits;
  std::memcpy(&bits, &biased, 32);
  bits <<= 52;
  v4d two_k;
  std::memcpy(&two_k, &bits, 32);
  return p * two_k;
}

inline v4d vlog(v4d x) {
  v4i bits;
  std::memcpy(&bits, &x, 32);
  v4i ebits = (bits >> 52) | (v4i){0x4330000000000000LL, 0x4330000000000000LL, 0x4330000000000000LL,
                                    0x4330000000000000LL};
  v4d e;
  std::memcpy(&e, &ebits, 32);
  e = e - splat(4503599627370496.0 + 1023.0);
  v4i mbits = (bits & (v4i){0x000FFFFFFFFFFFFFLL, 0x000FFFFFFFFFFFFFLL, 0x000FFFFFFFFFFFFFLL,
                            0x000FFFFFFFFFFFFFLL}) |
              (v4i){0x3FF0000000000000LL, 0x3FF0000000000000LL, 0x3FF0000000000000LL, 0x3FF0000000000000LL};
  v4d m;
  std::memcpy(&m, &mbits, 32);
  v4i big = m > splat(1.4142135623730951);
  m = vsel(big, m * splat(0.5), m);
  e = vsel(big, e + splat(1.0), e);
  v4d s = (m - splat(1.0)) / (m + splat(1.0));
  v4d s2 = s * s;
  v4d acc = splat(1.0 / 21.0);
  acc = acc * s2 + splat(1.0 / 19.0);
  acc = acc * s2 + splat(1.0 / 17.0);
  acc = acc * s2 + splat(1.0 / 15.0);
  acc = acc * s2 + splat(1.0 / 13.0);
  acc = acc * s2 + splat(1.0 / 11.0);
  acc = acc * s2 + splat(1.0 / 9.0);
  acc = acc * s2 + splat(1.0 / 7.0);
  acc = acc * s2 + splat(1.0 / 5.0);
  acc = acc * s2 + splat(1.0 / 3.0);
  acc = acc * s2 + splat(1.0);
  return e * splat(0.6931471805599453) + splat(2.0) * s * acc;
}

inline v4d vsin_halfpi(v4d x) {
  v4d x2 = x * x;
  v4d acc = splat(1.0 / 51090942171709440000.0);
  acc = acc * x2 - splat(1.0 / 121645100408832000.0);
  acc = acc * x2 + splat(1.0 / 355687428096000.0);
  acc = acc * x2 - splat(1.0 / 1307674368000.0);
  acc = acc * x2 + splat(1.0 / 6227020800.0);
  acc = acc * x2 - splat(1.0 / 39916800.0);
  acc = acc * x2 + splat(1.0 / 362880.0);
  acc = acc * x2 - splat(1.0 / 5040.0);
  acc = acc * x2 + splat(1.0 / 120.0);
  acc = acc * x2 - splat(1.0 / 6.0);
  acc = acc * x2 + splat(1.0);
  return x * acc;
}

inline v4d vasin(v4d x) {
  v4d a = vabs(x);
  v4d z = a * a;
  v4d pn = ((((splat(4.253011369004428248960E-3) * z + splat(-6.019598008014123785661E-1)) * z +
              splat(5.444622390564711410273E0)) * z + splat(-1.626247967210700244449E1)) * z +
            splat(1.956261983317594739197E1)) * z + splat(-8.198089802484824371615E0);
  v4d pd = ((((z + splat(-1.474091372988853791896E1)) * z + splat(7.049610280856842141659E1)) * z +
             splat(-1.471791292232726029859E2)) * z + splat(1.395105614657485689735E2)) * z +
           splat(-4.918853881490881290097E1);
  v4d small = a + a * z * pn / pd;
  v4d zz = splat(1.0) - a;
  v4d rn = (((splat(2.967721961301243206100E-3) * zz + splat(-5.634242780008963776856E-1)) * zz +
             splat(6.968710824104713396794E0)) * zz + splat(-2.556901049652824852289E1)) * zz +
           splat(2.853665548261061424989E1);
  v4d rd = (((zz + splat(-2.194779531642920639778E1)) * zz + splat(1.470656354026814941758E2)) * zz +
            splat(-3.838770957603691357202E2)) * zz + splat(3.424398657913078477438E2);
  v4d p = zz * rn / rd;
  v4d zs = vsqrt(zz + zz);
  v4d big = (splat(0.785398163397448309616) - zs) - (zs * p - splat(6.123233995736765886130E-17)) +
            splat(0.785398163397448309616);
  v4d v = vsel(a > splat(0.625), big, small);
  return vsel(x < splat(0.0), -v, v);
}

struct PhiV {
  v4d P, dP, Q;
};

inline void vpoly_triple(v4d x, v4d& P, v4d& dP, v4d& q) {
  v4d t = splat(1.0) / (splat(1.0) + splat(0.2316419) * vabs(x));
  v4d e = splat(0.3989422804014327) * vexp(splat(-0.5) * x * x);
  q = e * t * (splat(0.319381530) + t * (splat(-0.356563782) + t * (splat(1.781477937) +
               t * (splat(-1.821255978) + t * splat(1.330274429)))));
  v4d G = splat(1.3787727090824242) + t * (splat(-2.9180615078705534) + t * (splat(9.303925945620334) +
          t * (splat(-15.71822236097206) + t * (splat(14.843177660082445) +
          t * (splat(-7.430322503492646) + t * splat(1.5407364812748756))))));
  P = vsel(x >= splat(0.0), splat(1.0) - q, q);
  dP = e * G;
}

inline void vpoly_pair(v4d x, v4d& P, v4d& dP) {
  v4d t = splat(1.0) / (splat(1.0) + splat(0.2316419) * vabs(x));
  v4d e = splat(0.3989422804014327) * vexp(splat(-0.5) * x * x);
  v4d p = e * t * (splat(0.319381530) + t * (splat(-0.356563782) + t * (splat(1.781477937) +
               t * (splat(-1.821255978) + t * splat(1.330274429)))));
  v4d G = splat(1.3787727090824242) + t * (splat(-2.9180615078705534) + t * (splat(9.303925945620334) +
          t * (splat(-15.71822236097206) + t * (splat(14.843177660082445) +
          t * (splat(-7.430322503492646) + t * splat(1.5407364812748756))))));
  P = vsel(x >= splat(0.0), splat(1.0) - p, p);
  dP = e * G;
}

using bvn_tail::kPhiC;
constexpr double kPhiD[18] = {1.1854995356145523, 1.3097573132537195, 0.45298641056440525, 0.10328147437489583, 0.01295638227198467, -0.0003253821430972744, -0.0004287832995995369, -4.434221633584315e-05, 9.551609080045374e-06, 2.278034508995305e-06, -2.1182703839817368e-07, -9.437945361118253e-08, 6.090771522883504e-09, 3.89287566636132e-09, -2.7866990704357154e-10, -1.638034930910627e-10, 1.8902019919991792e-11, 7.084052091904388e-12};
using bvn_tail::kPhiA;
using bvn_tail::kPhiB;

template <int K>
inline v4d clenshaw(const double* co, v4d u) {
  v4d b1 = splat(0.0), b2 = splat(0.0);
  v4d u2 = splat(2.0) * u;
#pragma GCC unroll 20
  for (int k = K - 1; k > 0; --k) {
    v4d t = u2 * b1 - b2 + splat(co[k]);
    b2 = b1;
    b1 = t;
  }
  return u * b1 - b2 + splat(co[0]);
}

inline PhiV vacc_phi(v4d x) {
  v4d ax = vabs(x);
  v4d t = splat(1.0) / (splat(1.0) + splat(0.2316419) * ax);
  v4d u = (splat(2.0) * t - splat(kPhiA + kPhiB)) * splat(1.0 / (kPhiB - kPhiA));
  v4d Cu = clenshaw<19>(kPhiC, u);
  v4d Du = clenshaw<18>(kPhiD, u);
  v4d e = splat(0.3989422804014327) * vexp(splat(-0.5) * x * x);
  v4d q = e * t * Cu;
  v4d G = (splat(1.0) - t) * splat(1.0 / 0.2316419) * Cu + splat(0.2316419) * t * t * (Cu + t * Du);
  return {vsel(x >= splat(0.0), splat(1.0) - q, q), e * G, q};
}

constexpr double kX8[8] = {-0.9602898564975362, -0.7966664774136267, -0.525532409916329, -0.18343464249564978,
                           0.18343464249564978, 0.525532409916329, 0.7966664774136267, 0.9602898564975362};
constexpr double kW8[8] = {0.10122853629037706, 0.22238103445337443, 0.3137066458778869, 0.36268378337836166,
                           0.36268378337836166, 0.3137066458778869, 0.22238103445337443, 0.10122853629037706};
constexpr double kX20[20] = {-0.993128599185095, -0.9639719272779138, -0.912234428251326, -0.8391169718222188, -0.7463319064601508, -0.636053680726515, -0.5108670019508271, -0.37370608871541955, -0.22778585114164507, -0.07652652113349734, 0.07652652113349734, 0.22778585114164507, 0.37370608871541955, 0.5108670019508271, 0.636053680726515, 0.7463319064601508, 0.8391169718222188, 0.912234428251326, 0.9639719272779138, 0.993128599185095};
constexpr double kW20[20] = {0.017614007139150893, 0.040601429800386446, 0.06267204833410879, 0.08327674157670471, 0.1019301198172407, 0.1181945319615186, 0.1316886384491769, 0.1420961093183824, 0.14917298647260424, 0.15275338713072628, 0.15275338713072628, 0.14917298647260424, 0.1420961093183824, 0.1316886384491769, 0.1181945319615186, 0.1019301198172407, 0.08327674157670471, 0.06267204833410879, 0.040601429800386446, 0.017614007139150893};
constexpr double kX12[12] = {-0.9815606342467192, -0.9041172563704748, -0.7699026741943047, -0.5873179542866175, -0.3678314989981802, -0.1252334085114689, 0.1252334085114689, 0.3678314989981802, 0.5873179542866175, 0.7699026741943047, 0.9041172563704748, 0.9815606342467192};
constexpr double kW12[12] = {0.04717533638651141, 0.10693932599531907, 0.16007832854334642, 0.20316742672306573, 0.2334925365383546, 0.2491470458134027, 0.2491470458134027, 0.2334925365383546, 0.20316742672306573, 0.16007832854334642, 0.10693932599531907, 0.04717533638651141};
constexpr double kX10[10] = {-0.9739065285171717, -0.8650633666889845, -0.6794095682990244, -0.4333953941292472, -0.14887433898163122, 0.14887433898163122, 0.4333953941292472, 0.6794095682990244, 0.8650633666889845, 0.9739065285171717};
constexpr double kW10[10] = {0.06667134430868814, 0.1494513491505804, 0.219086362515982, 0.2692667193099965, 0.2955242247147528, 0.2955242247147528, 0.2692667193099965, 0.219086362515982, 0.1494513491505804, 0.06667134430868814};
constexpr double kX14[14] = {-0.9862838086968124, -0.9284348836635735, -0.827201315069765, -0.6872929048116855, -0.5152486363581541, -0.31911236892788974, -0.10805494870734367, 0.10805494870734367, 0.31911236892788974, 0.5152486363581541, 0.6872929048116855, 0.827201315069765, 0.9284348836635735, 0.9862838086968124};
constexpr double kW14[14] = {0.03511946033175175, 0.08015808715975999, 0.12151857068790331, 0.15720316715819363, 0.18553839747793788, 0.20519846372129572, 0.21526385346315793, 0.21526385346315793, 0.20519846372129572, 0.18553839747793788, 0.15720316715819363, 0.12151857068790331, 0.08015808715975999, 0.03511946033175175};
constexpr double kX16[16] = {-0.9894009349916499, -0.9445750230732326, -0.8656312023878318, -0.755404408355003, -0.6178762444026438, -0.45801677765722737, -0.2816035507792589, -0.09501250983763744, 0.09501250983763744, 0.2816035507792589, 0.45801677765722737, 0.6178762444026438, 0.755404408355003, 0.8656312023878318, 0.9445750230732326, 0.9894009349916499};
constexpr double kW16[16] = {0.027152459411754176, 0.062253523938647456, 0.0951585116824926, 0.12462897125553407, 0.1495959888165767, 0.16915651939500265, 0.18260341504492364, 0.18945061045506864, 0.18945061045506864, 0.18260341504492364, 0.16915651939500265, 0.1495959888165767, 0.12462897125553407, 0.0951585116824926, 0.062253523938647456, 0.027152459411754176};
constexpr double kInv4Pi = 0.07957747154594767;
constexpr double kRhoMax = 0.999999;
constexpr double kRhoTier[5] = {0.75, 0.85, 0.9, 0.925, 0.97};

struct CornerV {
  v4d I, dh, dk, dA;
};

template <int M>
struct Nodes {
  v4d s[M], ic2[M], sic[M], wdq[M];
};

template <int M>
inline void vnodes(v4d A, const double* X, const double* W, Nodes<M>& nd) {
#pragma GCC unroll 20
  for (int j = 0; j < M; ++j) {
    double xp = 0.5 * (X[j] + 1.0);
    v4d s = vsin_halfpi(A * splat(xp));
    v4d ic2 = splat(1.0) / (splat(1.0) - s * s);
    nd.s[j] = s;
    nd.ic2[j] = ic2;
    nd.sic[j] = s * ic2;
    nd.wdq[j] = splat(W[j] * xp) * vsqrt(ic2);
  }
}

template <int M>
inline CornerV vcorner(v4d h, v4d k, v4d A, const double* W, const Nodes<M>& nd) {
  v4d a = splat(0.5) * (h * h + k * k), b = h * k;
  v4d S0 = splat(0), S1 = splat(0), S2 = splat(0), T = splat(0);
#pragma GCC unroll 20
  for (int j = 0; j < M; ++j) {
    v4d e = vexp(b * nd.sic[j] - a * nd.ic2[j]);
    v4d we = splat(W[j]) * e;
    S0 += we;
    S1 += we * nd.ic2[j];
    S2 += we * nd.sic[j];
    T += e * nd.wdq[j] * (b - splat(2.0) * a * nd.sic[j] + splat(2.0) * b * nd.s[j] * nd.sic[j]);
  }
  v4d c = A * splat(kInv4Pi);
  return {c * S0, -c * (h * S1 - k * S2), -c * (k * S1 - h * S2), (S0 + A * T) * splat(kInv4Pi)};
}

inline CornerV csel(v4i m, const CornerV& a, const CornerV& b) {
  return {vsel(m, a.I, b.I), vsel(m, a.dh, b.dh), vsel(m, a.dk, b.dk), vsel(m, a.dA, b.dA)};
}

inline bool any_lane(v4i m) { return (m[0] | m[1] | m[2] | m[3]) != 0; }
inline bool all_lanes(v4i m) { return (m[0] & m[1] & m[2] & m[3]) != 0; }

template <int NC, int M>
inline void corners_n(const v4d* H, const v4d* K, v4d A, const double* X, const double* W, CornerV* out) {
  Nodes<M> nd;
  vnodes<M>(A, X, W, nd);
  for (int c = 0; c < NC; ++c) out[c] = vcorner<M>(H[c], K[c], A, W, nd);
}

template <int NC>
inline void corners(const v4d* H, const v4d* K, v4d rho, v4d& r, v4d& dAdr, CornerV* out) {
  r = vmin(vmax(rho, splat(-kRhoMax)), splat(kRhoMax));
  v4d A = vasin(r);
  dAdr = vsel(vabs(rho) < splat(kRhoMax), splat(1.0) / vsqrt(splat(1.0) - r * r), splat(0.0));
  v4d ar = vabs(r);
  double m = ar[0];
  for (int l = 1; l < 4; ++l) m = ar[l] > m ? ar[l] : m;
  if (!(m >= kRhoTier[0])) corners_n<NC, 8>(H, K, A, kX8, kW8, out);
  else if (m < kRhoTier[1]) corners_n<NC, 10>(H, K, A, kX10, kW10, out);
  else if (m < kRhoTier[2]) corners_n<NC, 12>(H, K, A, kX12, kW12, out);
  else if (m < kRhoTier[3]) corners_n<NC, 14>(H, K, A, kX14, kW14, out);
  else if (m < kRhoTier[4]) corners_n<NC, 16>(H, K, A, kX16, kW16, out);
  else corners_n<NC, 20>(H, K, A, kX20, kW20, out);
}

inline void vphi_diff(v4d lo, v4d hi, v4d& D, v4d& dlo, v4d& dhi) {
  PhiV H = vacc_phi(hi), L = vacc_phi(lo);
  D = vsel(lo >= splat(0.0), L.Q - H.Q, H.P - L.P);
  dlo = -L.dP;
  dhi = H.dP;
}

template <int M>
inline void rect4m(v4d a1, v4d b1, v4d a2, v4d b2, v4d rho, v4d& p, v4d& d1l, v4d& d1h,
                   v4d& d2l, v4d& d2h, v4d& drho) {
  constexpr int NC = ((M >> 0) & 1) + ((M >> 1) & 1) + ((M >> 2) & 1) + ((M >> 3) & 1);
  const v4d zero = splat(0.0);
  CornerV c[4] = {{zero, zero, zero, zero}, {zero, zero, zero, zero},
                  {zero, zero, zero, zero}, {zero, zero, zero, zero}};
  v4d r, dAdr = zero;
  if constexpr (NC > 0) {
    const v4d Hall[4] = {b1, a1, b1, a1}, Kall[4] = {b2, b2, a2, a2};
    v4d H[NC], K[NC];
    CornerV cc[NC];
    int q = 0;
    for (int j = 0; j < 4; ++j)
      if ((M >> j) & 1) { H[q] = Hall[j]; K[q] = Kall[j]; ++q; }
    corners<NC>(H, K, rho, r, dAdr, cc);
    q = 0;
    for (int j = 0; j < 4; ++j)
      if ((M >> j) & 1) c[j] = cc[q++];
  }
  v4d D1, D1l, D1h, D2, D2l, D2h;
  vphi_diff(a1, b1, D1, D1l, D1h);
  vphi_diff(a2, b2, D2, D2l, D2h);
  p = D1 * D2 + c[0].I - c[1].I - c[2].I + c[3].I;
  d1l = D1l * D2 - c[1].dh + c[3].dh;
  d1h = D1h * D2 + c[0].dh - c[2].dh;
  d2l = D1 * D2l - c[2].dk + c[3].dk;
  d2h = D1 * D2h + c[0].dk - c[1].dk;
  drho = (c[0].dA - c[1].dA - c[2].dA + c[3].dA) * dAdr;
}

inline void rect4(v4d a1, v4d b1, v4d a2, v4d b2, v4d rho, v4d& p, v4d& d1l, v4d& d1h,
                  v4d& d2l, v4d& d2h, v4d& drho) {
  rect4m<15>(a1, b1, a2, b2, rho, p, d1l, d1h, d2l, d2h, drho);
}

using Rect4Fn = void (*)(v4d, v4d, v4d, v4d, v4d, v4d&, v4d&, v4d&, v4d&, v4d&, v4d&);
constexpr Rect4Fn kRect4[16] = {rect4m<0>, rect4m<1>, rect4m<2>, rect4m<3>, rect4m<4>, rect4m<5>,
                                rect4m<6>, rect4m<7>, rect4m<8>, rect4m<9>, rect4m<10>, rect4m<11>,
                                rect4m<12>, rect4m<13>, rect4m<14>, rect4m<15>};

inline bool finite_bound(double x) { return std::fabs(x) < 1e9; }

inline void rect_tail_fix(int64_t j, const double* a1, const double* b1, const double* a2,
                          const double* b2, const double* rho, double* p, double* d1l, double* d1h,
                          double* d2l, double* d2h, double* drho) {
  if (p[j] >= bvn_tail::kTailSwitch) return;
  if (std::isnan(a1[j]) || std::isnan(b1[j]) || std::isnan(a2[j]) || std::isnan(b2[j]) || std::isnan(rho[j])) return;
  bvn_tail::Rect r = bvn_tail::rect(a1[j], b1[j], a2[j], b2[j], rho[j]);
  p[j] = r.p;
  d1l[j] = r.d1l;
  d1h[j] = r.d1h;
  d2l[j] = r.d2l;
  d2h[j] = r.d2h;
  drho[j] = r.drho;
}

inline void rect_array(int64_t n, const double* a1, const double* b1, const double* a2,
                       const double* b2, const double* rho, double* p, double* d1l, double* d1h,
                       double* d2l, double* d2h, double* drho) {
  std::vector<int32_t> perm(n);
  std::vector<uint8_t> pat(n);
  int64_t cnt[17] = {0};
  for (int64_t i = 0; i < n; ++i) {
    bool fa1 = finite_bound(a1[i]), fb1 = finite_bound(b1[i]);
    bool fa2 = finite_bound(a2[i]), fb2 = finite_bound(b2[i]);
    int m = (fb1 && fb2) | ((fa1 && fb2) << 1) | ((fb1 && fa2) << 2) | ((fa1 && fa2) << 3);
    pat[i] = static_cast<uint8_t>(m);
    ++cnt[m + 1];
  }
  for (int m = 0; m < 16; ++m) cnt[m + 1] += cnt[m];
  int64_t pos[16];
  for (int m = 0; m < 16; ++m) pos[m] = cnt[m];
  for (int64_t i = 0; i < n; ++i) perm[pos[pat[i]]++] = static_cast<int32_t>(i);
  for (int m = 0; m < 16; ++m) {
    for (int64_t s0 = cnt[m]; s0 < cnt[m + 1]; s0 += 4) {
      int64_t lanes = cnt[m + 1] - s0 < 4 ? cnt[m + 1] - s0 : 4;
      v4d A1, B1, A2, B2, R, P, O1, O2, O3, O4, O5;
      for (int l = 0; l < 4; ++l) {
        int64_t j = perm[s0 + (l < lanes ? l : 0)];
        A1[l] = a1[j]; B1[l] = b1[j]; A2[l] = a2[j]; B2[l] = b2[j]; R[l] = rho[j];
      }
      kRect4[m](A1, B1, A2, B2, R, P, O1, O2, O3, O4, O5);
      for (int l = 0; l < lanes; ++l) {
        int64_t j = perm[s0 + l];
        p[j] = P[l]; d1l[j] = O1[l]; d1h[j] = O2[l]; d2l[j] = O3[l]; d2h[j] = O4[l]; drho[j] = O5[l];
      }
    }
  }
  for (int64_t j = 0; j < n; ++j) rect_tail_fix(j, a1, b1, a2, b2, rho, p, d1l, d1h, d2l, d2h, drho);
}

inline void cdf4(v4d h, v4d k, v4d rho, v4d& p, v4d& dh, v4d& dk, v4d& drho) {
  CornerV c[1];
  v4d r, dAdr;
  corners<1>(&h, &k, rho, r, dAdr, c);
  PhiV H = vacc_phi(h), K = vacc_phi(k);
  p = H.P * K.P + c[0].I;
  dh = H.dP * K.P + c[0].dh;
  dk = H.P * K.dP + c[0].dk;
  drho = c[0].dA * dAdr;
}

template <int NP, int NM>
inline void cdf_multi4(const v4d* H, const v4d* K, const v4d* X, v4d rho, v4d* out) {
  CornerV c[NP];
  v4d r, dAdr;
  corners<NP>(H, K, rho, r, dAdr, c);
  for (int q = 0; q < NP; ++q) {
    PhiV Ph = vacc_phi(H[q]), Pk = vacc_phi(K[q]);
    v4d* o = out + 8 * q;
    o[0] = Ph.P * Pk.P + c[q].I;
    o[1] = Ph.dP * Pk.P + c[q].dh;
    o[2] = Ph.P * Pk.dP + c[q].dk;
    o[3] = c[q].dA * dAdr;
    o[4] = Ph.Q;
    o[5] = Ph.dP;
    o[6] = Pk.Q;
    o[7] = Pk.dP;
  }
  for (int m = 0; m < NM; ++m) {
    PhiV Px = vacc_phi(X[m]);
    out[8 * NP + 2 * m] = Px.Q;
    out[8 * NP + 2 * m + 1] = Px.dP;
  }
}

template <int NP, int NM>
inline void cdf_multi_array(int64_t n, const double* const* in, double* const* out) {
  constexpr int NI = 2 * NP + NM + 1, NO = 8 * NP + 2 * NM;
  v4d V[NI], O[NO];
  int64_t i = 0;
  for (; i + 4 <= n; i += 4) {
    for (int a = 0; a < NI; ++a) std::memcpy(&V[a], in[a] + i, 32);
    cdf_multi4<NP, NM>(V, V + NP, V + 2 * NP, V[NI - 1], O);
    for (int a = 0; a < NO; ++a) std::memcpy(out[a] + i, &O[a], 32);
  }
  if (i < n) {
    for (int a = 0; a < NI; ++a) {
      V[a] = splat(0.0);
      for (int64_t l = 0; i + l < n; ++l) V[a][l] = in[a][i + l];
    }
    cdf_multi4<NP, NM>(V, V + NP, V + 2 * NP, V[NI - 1], O);
    for (int a = 0; a < NO; ++a)
      for (int64_t l = 0; i + l < n; ++l) out[a][i + l] = O[a][l];
  }
  const double* r = in[NI - 1];
  for (int q = 0; q < NP; ++q) {
    const double* h = in[q];
    const double* k = in[NP + q];
    for (int64_t j = 0; j < n; ++j) {
      if (out[8 * q][j] >= bvn_tail::kTailSwitch || std::isnan(h[j]) || std::isnan(k[j]) || std::isnan(r[j])) continue;
      bvn_tail::Rect t = bvn_tail::rect(-bvn_tail::kInf, h[j], -bvn_tail::kInf, k[j], r[j]);
      out[8 * q][j] = t.p;
      out[8 * q + 1][j] = t.d1h;
      out[8 * q + 2][j] = t.d2h;
      out[8 * q + 3][j] = t.drho;
    }
  }
}

inline void cdf_array(int64_t n, const double* h, const double* k, const double* rho, double* p,
                      double* dh, double* dk, double* drho) {
  int64_t i = 0;
  for (; i + 4 <= n; i += 4) {
    v4d H, K, R, P, O1, O2, O3;
    std::memcpy(&H, h + i, 32);
    std::memcpy(&K, k + i, 32);
    std::memcpy(&R, rho + i, 32);
    cdf4(H, K, R, P, O1, O2, O3);
    std::memcpy(p + i, &P, 32);
    std::memcpy(dh + i, &O1, 32);
    std::memcpy(dk + i, &O2, 32);
    std::memcpy(drho + i, &O3, 32);
  }
  if (i < n) {
    v4d H = splat(0), K = splat(0), R = splat(0), P, O1, O2, O3;
    for (int64_t l = 0; i + l < n; ++l) {
      H[l] = h[i + l];
      K[l] = k[i + l];
      R[l] = rho[i + l];
    }
    cdf4(H, K, R, P, O1, O2, O3);
    for (int64_t l = 0; i + l < n; ++l) {
      p[i + l] = P[l];
      dh[i + l] = O1[l];
      dk[i + l] = O2[l];
      drho[i + l] = O3[l];
    }
  }
  for (int64_t j = 0; j < n; ++j) {
    if (p[j] >= bvn_tail::kTailSwitch || std::isnan(h[j]) || std::isnan(k[j]) || std::isnan(rho[j])) continue;
    bvn_tail::Rect r = bvn_tail::rect(-bvn_tail::kInf, h[j], -bvn_tail::kInf, k[j], rho[j]);
    p[j] = r.p;
    dh[j] = r.d1h;
    dk[j] = r.d2h;
    drho[j] = r.drho;
  }
}

inline void vrdp4(const v4d* in, v4d* out) {
  const v4d c1h = in[0], c1l = in[1], c2h = in[2], c2l = in[3];
  const v4i h1h = in[4] > splat(0.5), h1l = in[5] > splat(0.5), h2h = in[6] > splat(0.5), h2l = in[7] > splat(0.5);
  const v4d dpF = in[8], dpR = in[9], sd = in[10], lam = in[11], si = in[12], sS = in[13];
  const v4d INF = splat(1e10), NINF = splat(-1e10), zero = splat(0.0), floor20 = splat(1e-300);
  v4d isi = splat(1.0) / si, isS = splat(1.0) / sS, murec = dpF + dpR;
  v4d z1fh = vsel(h1h, (c1h - dpF) * isi, INF), z1fl = vsel(h1l, (c1l - dpF) * isi, NINF);
  v4d z2h = vsel(h2h, c2h, INF), z2l = vsel(h2l, c2l, NINF);
  v4d z1rh = vsel(h1h, (c1h - murec) * isi, INF), z1rl = vsel(h1l, (c1l - murec) * isi, NINF);
  v4d z2rh = vsel(h2h, (c2h - sd) * isS, INF), z2rl = vsel(h2l, (c2l - sd) * isS, NINF);
  v4d P1fh, d1fh, P1fl, d1fl, P2nh, d2nh, P2nl, d2nl, P1rh, d1rh, P1rl, d1rl, P2rh, d2rh, P2rl, d2rl;
  v4d q1fh, q1fl, q2nh, q2nl, q1rh, q1rl, q2rh, q2rl;
  vpoly_triple(z1fh, P1fh, d1fh, q1fh);
  vpoly_triple(z1fl, P1fl, d1fl, q1fl);
  vpoly_triple(z2h, P2nh, d2nh, q2nh);
  vpoly_triple(z2l, P2nl, d2nl, q2nl);
  vpoly_triple(z1rh, P1rh, d1rh, q1rh);
  vpoly_triple(z1rl, P1rl, d1rl, q1rl);
  vpoly_triple(z2rh, P2rh, d2rh, q2rh);
  vpoly_triple(z2rl, P2rl, d2rl, q2rl);
  v4d D1f = vsel(z1fl >= zero, q1fl - q1fh, P1fh - P1fl), D2n = vsel(z2l >= zero, q2nl - q2nh, P2nh - P2nl);
  v4d D1r = vsel(z1rl >= zero, q1rl - q1rh, P1rh - P1rl), D2r = vsel(z2rl >= zero, q2rl - q2rh, P2rh - P2rl);
  v4d p1f = vmax(D1f, floor20), p2n = vmax(D2n, floor20), p1r = vmax(D1r, floor20), p2r = vmax(D2r, floor20);
  v4d pfam = p1f * p2n, prec = p1r * p2r, oml = splat(1.0) - lam;
  v4d praw = oml * pfam + lam * prec;
  v4d prob = vmax(praw, floor20);
  v4d invP = vsel(praw > floor20, splat(1.0) / prob, zero);
  v4d val = vlog(prob);
  v4d wf1 = vsel(D1f > floor20, oml * p2n * isi * invP, zero);
  v4d wr1 = vsel(D1r > floor20, lam * p2r * isi * invP, zero);
  v4d wf2 = vsel(D2n > floor20, oml * p1f * invP, zero);
  v4d wr2 = vsel(D2r > floor20, lam * p1r * isS * invP, zero);
  v4d f1h = vsel(h1h, d1fh, zero), f1l = vsel(h1l, d1fl, zero), n2h = vsel(h2h, d2nh, zero), n2l = vsel(h2l, d2nl, zero);
  v4d r1h = vsel(h1h, d1rh, zero), r1l = vsel(h1l, d1rl, zero), r2h = vsel(h2h, d2rh, zero), r2l = vsel(h2l, d2rl, zero);
  out[0] = val;
  out[1] = -(wf1 * (f1h - f1l) + wr1 * (r1h - r1l));
  out[2] = -wr1 * (r1h - r1l);
  out[3] = -wr2 * (r2h - r2l);
  out[4] = (prec - pfam) * invP;
  out[5] = -(wf1 * (f1h * z1fh - f1l * z1fl) + wr1 * (r1h * z1rh - r1l * z1rl));
  out[6] = -wr2 * (r2h * z2rh - r2l * z2rl);
  out[7] = wf1 * f1h + wr1 * r1h;
  out[8] = -(wf1 * f1l + wr1 * r1l);
  out[9] = wf2 * n2h + wr2 * r2h;
  out[10] = -(wf2 * n2l + wr2 * r2l);
}

inline void vrdp_array(int64_t n, const double* const* in, double* const* out) {
  int64_t i = 0;
  v4d vin[14], vout[11];
  for (; i + 4 <= n; i += 4) {
    for (int a = 0; a < 14; ++a) std::memcpy(&vin[a], in[a] + i, 32);
    vrdp4(vin, vout);
    for (int a = 0; a < 11; ++a) std::memcpy(out[a] + i, &vout[a], 32);
  }
  if (i < n) {
    for (int a = 0; a < 14; ++a) {
      vin[a] = splat(a == 12 || a == 13 ? 1.0 : 0.0);
      for (int64_t l = 0; i + l < n; ++l) vin[a][l] = in[a][i + l];
    }
    vrdp4(vin, vout);
    for (int a = 0; a < 11; ++a)
      for (int64_t l = 0; i + l < n; ++l) out[a][i + l] = vout[a][l];
  }
}

}

#endif
