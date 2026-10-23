#pragma once
// =============================================================================
// Tail-accurate normal and bivariate-normal cell probabilities (scalar)
// =============================================================================

#include <cmath>
#include <limits>

namespace bvn_tail {

constexpr double kPhiC[19] = {0.5893276044905619, 0.47231061768868227, 0.1485473376619552, 0.03611913148900286, 0.006378297104388951, 0.0006592194044005224, -1.1534346994167076e-05, -1.541999589462102e-05, -1.4350295963176838e-06, 2.671384604629693e-07, 5.842069382918476e-08, -4.878387335732758e-09, -2.0166300903829314e-09, 1.2065191939381096e-10, 7.135408878679637e-11, -4.885139134318498e-12, -2.630066125081603e-12, 2.7380131795870467e-13, 9.691376820174753e-14};
constexpr double kPhiA = 0.015;
constexpr double kPhiB = 1.0;
constexpr double kLogSqrt2Pi = 0.91893853320467274;
constexpr double kInvSqrt2Pi = 0.3989422804014327;
constexpr double kLog2Pi = 1.8378770664093455;
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kTailSwitch = 1e-6;
constexpr double kDomain = 40.0;
constexpr double kSpan = 25.0;
constexpr int kPanels = 2;
constexpr int kNodes = 20;

constexpr double kGX[kNodes] = {-0.993128599185095, -0.9639719272779138, -0.912234428251326, -0.8391169718222188, -0.7463319064601508, -0.636053680726515, -0.5108670019508271, -0.37370608871541955, -0.22778585114164507, -0.07652652113349734, 0.07652652113349734, 0.22778585114164507, 0.37370608871541955, 0.5108670019508271, 0.636053680726515, 0.7463319064601508, 0.8391169718222188, 0.912234428251326, 0.9639719272779138, 0.993128599185095};
constexpr double kGW[kNodes] = {0.017614007139150893, 0.040601429800386446, 0.06267204833410879, 0.08327674157670471, 0.1019301198172407, 0.1181945319615186, 0.1316886384491769, 0.1420961093183824, 0.14917298647260424, 0.15275338713072628, 0.15275338713072628, 0.14917298647260424, 0.1420961093183824, 0.1316886384491769, 0.1181945319615186, 0.1019301198172407, 0.08327674157670471, 0.06267204833410879, 0.040601429800386446, 0.017614007139150893};

inline double log_phi(double x) { return -0.5 * x * x - kLogSqrt2Pi; }

inline double mills(double x) {
  if (x > 40.0) {
    double r = 1.0 / (x * x);
    return (1.0 + r * (-1.0 + r * (3.0 + r * (-15.0 + r * (105.0 - 945.0 * r))))) / x;
  }
  double t = 1.0 / (1.0 + 0.2316419 * x);
  double u = (2.0 * t - (kPhiA + kPhiB)) / (kPhiB - kPhiA);
  double b1 = 0.0, b2 = 0.0, u2 = 2.0 * u;
  for (int k = 18; k > 0; --k) {
    double tt = u2 * b1 - b2 + kPhiC[k];
    b2 = b1;
    b1 = tt;
  }
  return t * (u * b1 - b2 + kPhiC[0]);
}

inline double log_Q(double x) {
  if (x == kInf) return -kInf;
  if (x == -kInf) return 0.0;
  if (x >= 0.0) return log_phi(x) + std::log(mills(x));
  return std::log1p(-std::exp(log_phi(-x)) * mills(-x));
}

inline double log_Phi(double x) { return log_Q(-x); }

struct Band {
  double ld, Al, Au;
};

inline Band band(double lo, double hi) {
  if (!(hi > lo)) return {-kInf, 0.0, 0.0};
  if (lo >= 0.0) {
    double E = std::isfinite(hi) ? std::exp(0.5 * (lo - hi) * (lo + hi)) : 0.0;
    double W = mills(lo) - (E > 0.0 ? E * mills(hi) : 0.0);
    return {log_phi(lo) + std::log(W), 1.0 / W, E / W};
  }
  if (hi <= 0.0) {
    double E = std::isfinite(lo) ? std::exp(0.5 * (hi - lo) * (hi + lo)) : 0.0;
    double W = mills(-hi) - (E > 0.0 ? E * mills(-lo) : 0.0);
    return {log_phi(hi) + std::log(W), E / W, 1.0 / W};
  }
  double D = 0.5 * (std::erf(hi * 0.7071067811865476) - std::erf(lo * 0.7071067811865476));
  double pl = std::isfinite(lo) ? kInvSqrt2Pi * std::exp(-0.5 * lo * lo) : 0.0;
  double pu = std::isfinite(hi) ? kInvSqrt2Pi * std::exp(-0.5 * hi * hi) : 0.0;
  return {std::log(D), pl / D, pu / D};
}

inline double log_diff_Phi(double lo, double hi) { return band(lo, hi).ld; }

struct Rect {
  double lp, p, d1l, d1h, d2l, d2h, drho;
};

struct G3 {
  double g, d1, d2;
};

struct Cond {
  double a2, b2, rho, s, k;
  G3 eval(double x) const {
    double l = (a2 - rho * x) / s, u = (b2 - rho * x) / s;
    Band b = band(l, u);
    double R = b.Al - b.Au;
    double lA = std::isfinite(l) ? l * b.Al : 0.0, uA = std::isfinite(u) ? u * b.Au : 0.0;
    return {log_phi(x) + b.ld, -x + k * R, -1.0 + k * k * (lA - uA - R * R)};
  }
  double g(double x) const { return log_phi(x) + band((a2 - rho * x) / s, (b2 - rho * x) / s).ld; }
};

inline double lphi2(double x, double y, double rho, double s) {
  if (!std::isfinite(x) || !std::isfinite(y)) return -kInf;
  return -(x * x - 2.0 * rho * x * y + y * y) / (2.0 * s * s) - std::log(s) - kLog2Pi;
}

template <class C>
inline double log_integral(const C& c, double a, double b) {
  double lo = std::fmax(a, -kDomain), hi = std::fmin(b, kDomain);
  if (!(hi > lo)) return -kInf;
  G3 e = c.eval(lo);
  if (!std::isfinite(e.g)) e.d1 = kInf;
  double m = lo;
  if (e.d1 > 0.0) {
    G3 eh = c.eval(hi);
    if (!std::isfinite(eh.g)) eh.d1 = -kInf;
    if (eh.d1 >= 0.0) {
      m = hi;
      e = eh;
    } else {
      double L = lo, H = hi, x = 0.5 * (lo + hi);
      e = c.eval(x);
      for (int it = 0; it < 60; ++it) {
        if (e.d1 > 0.0) L = x; else H = x;
        double nx = x - e.d1 / e.d2;
        if (!(nx > L && nx < H)) nx = 0.5 * (L + H);
        bool done = std::fabs(nx - x) < 1e-12 * (1.0 + std::fabs(x)) || H - L < 1e-12 * (1.0 + std::fabs(x));
        x = nx;
        e = c.eval(x);
        if (done) break;
      }
      m = x;
    }
  }
  double gm = e.g;
  if (!std::isfinite(gm)) return -kInf;
  double rs = (m == lo || m == hi) ? std::fabs(e.d1) : 0.0;
  double cv = std::fmax(-e.d2, 1.0);
  double ends[2];
  for (int side = 0; side < 2; ++side) {
    double dir = side == 0 ? -1.0 : 1.0, edge = side == 0 ? lo : hi;
    if ((edge - m) * dir <= 0.0) {
      ends[side] = m;
      continue;
    }
    double Tg = -rs + std::sqrt(rs * rs + 2.0 * kSpan);
    double T = (-rs + std::sqrt(rs * rs + 2.0 * cv * kSpan)) / cv;
    for (int it = 0; it < 8 && T < Tg; ++it) {
      double x = m + dir * T;
      if ((x - edge) * dir >= 0.0 || gm - c.g(x) >= kSpan) break;
      T = std::fmin(2.0 * T, Tg);
    }
    double x = m + dir * T;
    ends[side] = (x - edge) * dir >= 0.0 ? edge : x;
  }
  double vals[2 * kPanels * kNodes];
  int nv = 0;
  double vmax = -kInf;
  double segA[2] = {ends[0], m}, segB[2] = {m, ends[1]};
  for (int sgi = 0; sgi < 2; ++sgi) {
    double A0 = segA[sgi], B0 = segB[sgi];
    if (!(B0 > A0)) continue;
    for (int j = 0; j < kPanels; ++j) {
      double A = A0 + (B0 - A0) * j / kPanels, B = A0 + (B0 - A0) * (j + 1) / kPanels;
      double h = 0.5 * (B - A), c0 = 0.5 * (A + B), lh = std::log(h);
      for (int q = 0; q < kNodes; ++q) {
        double v = lh + std::log(kGW[q]) + c.g(c0 + h * kGX[q]);
        vals[nv++] = v;
        if (v > vmax) vmax = v;
      }
    }
  }
  if (nv == 0 || !std::isfinite(vmax)) return -kInf;
  double acc = 0.0;
  for (int i = 0; i < nv; ++i) acc += std::exp(vals[i] - vmax);
  return vmax + std::log(acc);
}

inline Rect rect(double a1, double b1, double a2, double b2, double rho) {
  Rect r{-kInf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  if (rho > 0.999999) rho = 0.999999;
  if (rho < -0.999999) rho = -0.999999;
  double s = std::sqrt((1.0 - rho) * (1.0 + rho));
  Cond c{a2, b2, rho, s, rho / s};
  r.lp = log_integral(c, a1, b1);
  if (!std::isfinite(r.lp)) return r;
  r.p = std::exp(r.lp);
  if (std::isfinite(b1)) r.d1h = std::exp(log_phi(b1) + band((a2 - rho * b1) / s, (b2 - rho * b1) / s).ld);
  if (std::isfinite(a1)) r.d1l = -std::exp(log_phi(a1) + band((a2 - rho * a1) / s, (b2 - rho * a1) / s).ld);
  if (std::isfinite(b2)) r.d2h = std::exp(log_phi(b2) + band((a1 - rho * b2) / s, (b1 - rho * b2) / s).ld);
  if (std::isfinite(a2)) r.d2l = -std::exp(log_phi(a2) + band((a1 - rho * a2) / s, (b1 - rho * a2) / s).ld);
  r.drho = std::exp(lphi2(b1, b2, rho, s)) - std::exp(lphi2(a1, b2, rho, s))
         - std::exp(lphi2(b1, a2, rho, s)) + std::exp(lphi2(a1, a2, rho, s));
  return r;
}

constexpr double kDX8[8] = {-0.9602898564975362, -0.7966664774136267, -0.525532409916329, -0.18343464249564978, 0.18343464249564978, 0.525532409916329, 0.7966664774136267, 0.9602898564975362};
constexpr double kDW8[8] = {0.10122853629037706, 0.22238103445337443, 0.3137066458778869, 0.36268378337836166, 0.36268378337836166, 0.3137066458778869, 0.22238103445337443, 0.10122853629037706};
constexpr double kDX10[10] = {-0.9739065285171717, -0.8650633666889845, -0.6794095682990244, -0.4333953941292472, -0.14887433898163122, 0.14887433898163122, 0.4333953941292472, 0.6794095682990244, 0.8650633666889845, 0.9739065285171717};
constexpr double kDW10[10] = {0.06667134430868814, 0.1494513491505804, 0.219086362515982, 0.2692667193099965, 0.2955242247147528, 0.2955242247147528, 0.2692667193099965, 0.219086362515982, 0.1494513491505804, 0.06667134430868814};
constexpr double kDX12[12] = {-0.9815606342467192, -0.9041172563704748, -0.7699026741943047, -0.5873179542866175, -0.3678314989981802, -0.1252334085114689, 0.1252334085114689, 0.3678314989981802, 0.5873179542866175, 0.7699026741943047, 0.9041172563704748, 0.9815606342467192};
constexpr double kDW12[12] = {0.04717533638651141, 0.10693932599531907, 0.16007832854334642, 0.20316742672306573, 0.2334925365383546, 0.2491470458134027, 0.2491470458134027, 0.2334925365383546, 0.20316742672306573, 0.16007832854334642, 0.10693932599531907, 0.04717533638651141};
constexpr double kDX14[14] = {-0.9862838086968124, -0.9284348836635735, -0.827201315069765, -0.6872929048116855, -0.5152486363581541, -0.31911236892788974, -0.10805494870734367, 0.10805494870734367, 0.31911236892788974, 0.5152486363581541, 0.6872929048116855, 0.827201315069765, 0.9284348836635735, 0.9862838086968124};
constexpr double kDW14[14] = {0.03511946033175175, 0.08015808715975999, 0.12151857068790331, 0.15720316715819363, 0.18553839747793788, 0.20519846372129572, 0.21526385346315793, 0.21526385346315793, 0.20519846372129572, 0.18553839747793788, 0.15720316715819363, 0.12151857068790331, 0.08015808715975999, 0.03511946033175175};
constexpr double kDX16[16] = {-0.9894009349916499, -0.9445750230732326, -0.8656312023878318, -0.755404408355003, -0.6178762444026438, -0.45801677765722737, -0.2816035507792589, -0.09501250983763744, 0.09501250983763744, 0.2816035507792589, 0.45801677765722737, 0.6178762444026438, 0.755404408355003, 0.8656312023878318, 0.9445750230732326, 0.9894009349916499};
constexpr double kDW16[16] = {0.027152459411754176, 0.062253523938647456, 0.0951585116824926, 0.12462897125553407, 0.1495959888165767, 0.16915651939500265, 0.18260341504492364, 0.18945061045506864, 0.18945061045506864, 0.18260341504492364, 0.16915651939500265, 0.1495959888165767, 0.12462897125553407, 0.0951585116824926, 0.062253523938647456, 0.027152459411754176};
constexpr double kDTier[5] = {0.75, 0.85, 0.9, 0.925, 0.97};

struct DwNodes {
  int m = 0;
  double rho = 2.0, A = 0.0, dAdr = 0.0;
  double w[20], s[20], ic2[20], sic[20], wdq[20];
};

inline void dw_nodes(double rho, DwNodes& nd) {
  if (rho == nd.rho) return;
  double r = std::fmin(std::fmax(rho, -0.999999), 0.999999);
  double ar = std::fabs(r);
  const double *X, *W;
  int m;
  if (!(ar >= kDTier[0])) { m = 8; X = kDX8; W = kDW8; }
  else if (ar < kDTier[1]) { m = 10; X = kDX10; W = kDW10; }
  else if (ar < kDTier[2]) { m = 12; X = kDX12; W = kDW12; }
  else if (ar < kDTier[3]) { m = 14; X = kDX14; W = kDW14; }
  else if (ar < kDTier[4]) { m = 16; X = kDX16; W = kDW16; }
  else { m = 20; X = kGX; W = kGW; }
  nd.rho = rho;
  nd.m = m;
  nd.A = std::asin(r);
  nd.dAdr = std::fabs(rho) < 0.999999 ? 1.0 / std::sqrt(1.0 - r * r) : 0.0;
  for (int j = 0; j < m; ++j) {
    double xp = 0.5 * (X[j] + 1.0);
    double sj = std::sin(nd.A * xp);
    double ic = 1.0 / (1.0 - sj * sj);
    nd.w[j] = W[j];
    nd.s[j] = sj;
    nd.ic2[j] = ic;
    nd.sic[j] = sj * ic;
    nd.wdq[j] = W[j] * xp * std::sqrt(ic);
  }
}

struct Corner {
  double I, dh, dk, dA;
};

inline Corner dw_corner(double h, double k, const DwNodes& nd) {
  double a = 0.5 * (h * h + k * k), b = h * k;
  double S0 = 0.0, S1 = 0.0, S2 = 0.0, T = 0.0;
  for (int j = 0; j < nd.m; ++j) {
    double e = std::exp(b * nd.sic[j] - a * nd.ic2[j]);
    double we = nd.w[j] * e;
    S0 += we;
    S1 += we * nd.ic2[j];
    S2 += we * nd.sic[j];
    T += e * nd.wdq[j] * (b - 2.0 * a * nd.sic[j] + 2.0 * b * nd.s[j] * nd.sic[j]);
  }
  double c = nd.A * 0.07957747154594767;
  return {c * S0, -c * (h * S1 - k * S2), -c * (k * S1 - h * S2), (S0 + nd.A * T) * 0.07957747154594767};
}

inline double Qd(double x) { return 0.5 * std::erfc(x * 0.7071067811865476); }
inline double pdf(double x) { return std::isfinite(x) ? kInvSqrt2Pi * std::exp(-0.5 * x * x) : 0.0; }

inline double interval(double lo, double hi) {
  return lo >= 0.0 ? Qd(lo) - Qd(hi) : Qd(-hi) - Qd(-lo);
}

inline Rect rect_full(double a1, double b1, double a2, double b2, double rho, DwNodes& nd) {
  dw_nodes(rho, nd);
  bool fa1 = std::isfinite(a1), fb1 = std::isfinite(b1), fa2 = std::isfinite(a2), fb2 = std::isfinite(b2);
  double D1 = interval(a1, b1), D2 = interval(a2, b2);
  Corner c[4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
  if (fb1 && fb2) c[0] = dw_corner(b1, b2, nd);
  if (fa1 && fb2) c[1] = dw_corner(a1, b2, nd);
  if (fb1 && fa2) c[2] = dw_corner(b1, a2, nd);
  if (fa1 && fa2) c[3] = dw_corner(a1, a2, nd);
  Rect r;
  r.p = D1 * D2 + c[0].I - c[1].I - c[2].I + c[3].I;
  if (!(r.p >= kTailSwitch)) return rect(a1, b1, a2, b2, rho);
  r.lp = std::log(r.p);
  r.d1l = -pdf(a1) * D2 - c[1].dh + c[3].dh;
  r.d1h = pdf(b1) * D2 + c[0].dh - c[2].dh;
  r.d2l = -D1 * pdf(a2) - c[2].dk + c[3].dk;
  r.d2h = D1 * pdf(b2) + c[0].dk - c[1].dk;
  r.drho = (c[0].dA - c[1].dA - c[2].dA + c[3].dA) * nd.dAdr;
  return r;
}

struct LinCond {
  double Al, Bl, Au, Bu;
  double lo_at(double x) const { return std::isfinite(Al) ? Al + Bl * x : Al; }
  double hi_at(double x) const { return std::isfinite(Au) ? Au + Bu * x : Au; }
  G3 eval(double x) const {
    double l = lo_at(x), u = hi_at(x);
    Band b = band(l, u);
    double fl = std::isfinite(l) ? b.Al : 0.0, fu = std::isfinite(u) ? b.Au : 0.0;
    double lA = std::isfinite(l) ? l * b.Al : 0.0, uA = std::isfinite(u) ? u * b.Au : 0.0;
    double q = Bu * fu - Bl * fl;
    return {log_phi(x) + b.ld, -x + q, -1.0 - Bu * Bu * uA + Bl * Bl * lA - q * q};
  }
  double g(double x) const { return log_phi(x) + band(lo_at(x), hi_at(x)).ld; }
};

struct GaussLine {
  double lJ0, r1;
};

inline GaussLine gauss_line(double A, double B, double xa, double xb) {
  double s2 = 1.0 + B * B, s = std::sqrt(s2), m = A * B / s2;
  double tl = std::isfinite(xa) ? s * (xa + m) : xa, th = std::isfinite(xb) ? s * (xb + m) : xb;
  Band b = band(tl, th);
  double fl = std::isfinite(tl) ? b.Al : 0.0, fh = std::isfinite(th) ? b.Au : 0.0;
  return {log_phi(A / s) - std::log(s) + b.ld, (fl - fh) / s - m};
}

struct CdpCell {
  double lp;
  double d[9];
};

struct CdpPiece {
  double xa, xb, Al, Bl, Au, Bu;
  const double *dxa, *dxb, *dAl, *dBl, *dAu, *dBu;
};

// One CDP cell (kind 1 Remember, 2 Know, 3 Guess) within tlo < R + F < thi, as an integral over the
// standardised recollection value x: F given x is normal with mean muF + corr sF x and SD sF sqrt(1 - corr^2),
// so every bound on it is linear in x. d: muR, sR, muF, sF, cR, cK, tlo, thi, corr.
inline CdpCell cdp_cell(int kind, double muR, double sR, double muF, double sF, double cR, double cK,
                        double tlo, double thi, double corr = 0.0) {
  static const double Z[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  CdpCell out{-kInf, {0, 0, 0, 0, 0, 0, 0, 0, 0}};
  double s2 = (1.0 - corr) * (1.0 + corr), s = std::sqrt(s2), sc = sF * s, cs2 = corr / s2;
  double B = -(sR + corr * sF) / sc;
  double xcR = (cR - muR) / sR;
  double xkh = std::isfinite(thi) ? (thi - cK - muR) / sR : kInf;
  double xkl = std::isfinite(tlo) ? (tlo - cK - muR) / sR : -kInf;
  double Alo = std::isfinite(tlo) ? (tlo - muR - muF) / sc : -kInf;
  double Ahi = std::isfinite(thi) ? (thi - muR - muF) / sc : kInf;
  double AK = (cK - muF) / sc;
  double BK = -corr / s;
  double dxcR[9] = {-1 / sR, -xcR / sR, 0, 0, 1 / sR, 0, 0, 0, 0};
  double dxkh[9] = {-1 / sR, -xkh / sR, 0, 0, 0, -1 / sR, 0, 1 / sR, 0};
  double dxkl[9] = {-1 / sR, -xkl / sR, 0, 0, 0, -1 / sR, 1 / sR, 0, 0};
  double dAlo[9] = {-1 / sc, 0, -1 / sc, -Alo / sF, 0, 0, 1 / sc, 0, Alo * cs2};
  double dAhi[9] = {-1 / sc, 0, -1 / sc, -Ahi / sF, 0, 0, 0, 1 / sc, Ahi * cs2};
  double dB[9] = {0, -1 / sc, 0, sR / (sF * sc), 0, 0, 0, 0, -sR * corr / (sc * s2) - 1 / (s * s2)};
  double dAK[9] = {0, 0, -1 / sc, -AK / sF, 0, 1 / sc, 0, 0, AK * cs2};
  double dBK[9] = {0, 0, 0, 0, 0, 0, 0, 0, -1 / (s * s2)};
  const double* pdxkh = std::isfinite(xkh) ? dxkh : Z;
  const double* pdxkl = std::isfinite(xkl) ? dxkl : Z;
  const double* pdAlo = std::isfinite(Alo) ? dAlo : Z;
  const double* pdAhi = std::isfinite(Ahi) ? dAhi : Z;
  const double* pdBlo = std::isfinite(Alo) ? dB : Z;
  const double* pdBhi = std::isfinite(Ahi) ? dB : Z;
  double Blo = std::isfinite(Alo) ? B : 0.0, Bhi = std::isfinite(Ahi) ? B : 0.0;
  CdpPiece pc[2];
  int np = 0;
  auto lesser = [](double a, const double* da, double b, const double* db, double& v, const double*& dv) {
    if (a <= b) { v = a; dv = da; } else { v = b; dv = db; }
  };
  auto greater = [](double a, const double* da, double b, const double* db, double& v, const double*& dv) {
    if (a >= b) { v = a; dv = da; } else { v = b; dv = db; }
  };
  if (kind == 1) {
    pc[np++] = {xcR, kInf, Alo, Blo, Ahi, Bhi, dxcR, Z, pdAlo, pdBlo, pdAhi, pdBhi};
  } else if (kind == 3) {
    double b1; const double* db1;
    lesser(xkh, pdxkh, xcR, dxcR, b1, db1);
    pc[np++] = {xkl, b1, Alo, Blo, AK, BK, pdxkl, db1, pdAlo, pdBlo, dAK, dBK};
    double a2; const double* da2;
    greater(xkh, pdxkh, xkl, pdxkl, a2, da2);
    pc[np++] = {a2, xcR, Alo, Blo, Ahi, Bhi, da2, dxcR, pdAlo, pdBlo, pdAhi, pdBhi};
  } else {
    double ub; const double* dub;
    lesser(xcR, dxcR, xkh, pdxkh, ub, dub);
    double b1; const double* db1;
    lesser(xkl, pdxkl, ub, dub, b1, db1);
    pc[np++] = {-kInf, b1, Alo, Blo, Ahi, Bhi, Z, db1, pdAlo, pdBlo, pdAhi, pdBhi};
    pc[np++] = {xkl, ub, AK, BK, Ahi, Bhi, pdxkl, dub, dAK, dBK, pdAhi, pdBhi};
  }
  double lps[2] = {-kInf, -kInf};
  double lmax = -kInf;
  for (int k = 0; k < np; ++k) {
    LinCond c{pc[k].Al, pc[k].Bl, pc[k].Au, pc[k].Bu};
    lps[k] = log_integral(c, pc[k].xa, pc[k].xb);
    if (lps[k] > lmax) lmax = lps[k];
  }
  if (!std::isfinite(lmax)) return out;
  double acc = 0.0;
  for (int k = 0; k < np; ++k) acc += std::exp(lps[k] - lmax);
  out.lp = lmax + std::log(acc);
  for (int k = 0; k < np; ++k) {
    const CdpPiece& q = pc[k];
    if (!std::isfinite(lps[k])) continue;
    LinCond c{q.Al, q.Bl, q.Au, q.Bu};
    double xa = std::fmax(q.xa, -kDomain), xb = std::fmin(q.xb, kDomain);
    double w[6] = {0, 0, 0, 0, 0, 0};
    if (q.xb < kDomain) w[1] = std::exp(c.g(xb) - out.lp);
    if (q.xa > -kDomain) w[0] = -std::exp(c.g(xa) - out.lp);
    if (std::isfinite(q.Au)) {
      GaussLine gl = gauss_line(q.Au, q.Bu, xa, xb);
      double e = std::exp(gl.lJ0 - out.lp);
      w[4] = e;
      w[5] = e * gl.r1;
    }
    if (std::isfinite(q.Al)) {
      GaussLine gl = gauss_line(q.Al, q.Bl, xa, xb);
      double e = std::exp(gl.lJ0 - out.lp);
      w[2] = -e;
      w[3] = -e * gl.r1;
    }
    const double* dv[6] = {q.dxa, q.dxb, q.dAl, q.dBl, q.dAu, q.dBu};
    for (int t = 0; t < 6; ++t)
      if (w[t] != 0.0)
        for (int j = 0; j < 9; ++j) out.d[j] += w[t] * dv[t][j];
  }
  return out;
}

}  // namespace bvn_tail
