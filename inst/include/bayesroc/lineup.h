#pragma once
// =============================================================================
// Independent Observations (MAX) lineup cell: log P that one face wins with its value in a criterion
// interval, the faces independent normals plus an optional shared lineup offset b ~ N(0, sb^2)
// =============================================================================

#include <cmath>

#include "pcq.h"

namespace lineup {

using bvn::splat;
using bvn::v4d;
using bvn::v4i;

constexpr double kDrop = 40.0;
constexpr int kMaxPieces = 16;
constexpr double kGrade = 6.0;

enum Kind { kSuspect = 0, kFiller = 1, kAny = 2 };

struct Cell {
  double lp, dm, ds, dlo, dhi, dsb;
};

struct Integrand {
  double ma, sa, m, s, lo, hi, sb;
  int n1, n2;
  bool lo_open, hi_open, filler;

  // log h, d/dw and d2/dw2 at one w
  void eval(double w, double& h, double& g, double& h2) const {
    const double u = (w - ma) / sa, v = (w - m) / s;
    double zl = lo_open ? 0.0 : (lo - w) / sb, zh = hi_open ? 0.0 : (hi - w) / sb;
    if (sb <= 0.0) zl = zh = 0.0;
    const int side = lo_open ? 3 : hi_open ? 4 : zl > 0.0 ? 0 : zh < 0.0 ? 1 : 2;
    const double x1 = side == 0 || side == 4 ? -zl : side == 1 || side == 3 ? zh : zl;
    const double x2 = side == 0 ? -zh : side == 1 ? zl : -zh;
    v4d lp, l;
    pcq::vlogphi(v4d{w, v, x1, x2}, lp, l);
    h = -0.5 * u * u - std::log(sa) - bvn_tail::kLogSqrt2Pi;
    g = -u / sa;
    h2 = -1.0 / (sa * sa);
    if (n1) {
      h += n1 * lp[0];
      g += n1 * l[0];
      h2 -= n1 * l[0] * (w + l[0]);
    }
    if (n2) {
      h += lp[1];
      g += l[1] / s;
      h2 -= l[1] * (v + l[1]) / (s * s);
    }
    if (sb > 0.0) {
      double lg;
      if (lo_open && hi_open) {
        lg = 0.0;
      } else if (side >= 3) {
        lg = lp[2];
      } else if (side < 2) {
        lg = lp[2] + std::log1p(-std::exp(lp[3] - lp[2]));
      } else {
        lg = std::log1p(-std::exp(lp[2]) - std::exp(lp[3]));
      }
      const double ph = hi_open ? 0.0 : std::exp(bvn_tail::log_phi(zh) - lg);
      const double pl = lo_open ? 0.0 : std::exp(bvn_tail::log_phi(zl) - lg);
      const double d1 = (pl - ph) / sb;
      h += lg;
      g += d1;
      h2 += (pl * zl - ph * zh) / (sb * sb) - d1 * d1;
    }
  }
};

inline double mode(const Integrand& f, double a, double b) {
  double z = std::fmin(std::fmax(f.ma, a), b), h, g, h2;
  f.eval(z, h, g, h2);
  double zl, zr;
  if (g > 0.0) {
    zl = z;
    double step = f.sa;
    zr = z;
    for (int it = 0; it < 200; ++it) {
      const double zn = std::fmin(zr + step, b);
      f.eval(zn, h, g, h2);
      if (g <= 0.0) {
        zr = zn;
        break;
      }
      zl = zn;
      zr = zn;
      if (zn >= b) return b;
      step *= 2.0;
    }
  } else {
    zr = z;
    double step = f.sa;
    zl = z;
    for (int it = 0; it < 200; ++it) {
      const double zn = std::fmax(zl - step, a);
      f.eval(zn, h, g, h2);
      if (g >= 0.0) {
        zl = zn;
        break;
      }
      zr = zn;
      zl = zn;
      if (zn <= a) return a;
      step *= 2.0;
    }
  }
  z = 0.5 * (zl + zr);
  for (int it = 0; it < 100; ++it) {
    f.eval(z, h, g, h2);
    if (g > 0.0) zl = z; else zr = z;
    double zn = z - g / h2;
    if (!(zn > zl && zn < zr)) zn = 0.5 * (zl + zr);
    const double tol = 1e-3 / std::sqrt(-h2);
    if (std::fabs(zn - z) < tol || zr - zl < tol) return zn;
    z = zn;
  }
  return z;
}

inline double edge(const Integrand& f, double zm, double target, double side, double bound) {
  double zo = zm + side * std::sqrt(2.0 * kDrop) * f.sa;
  zo = side > 0.0 ? std::fmin(zo, bound) : std::fmax(zo, bound);
  double h, g, h2;
  f.eval(zo, h, g, h2);
  if (h >= target) return zo;
  double zi = zm, z = zo;
  for (int it = 0; it < 80; ++it) {
    f.eval(z, h, g, h2);
    if (std::fabs(h - target) < 0.5) break;
    if (h < target) zo = z; else zi = z;
    double zn = g != 0.0 ? z - (h - target) / g : 0.5 * (zi + zo);
    if (!((zn - zi) * (zn - zo) < 0.0)) zn = 0.5 * (zi + zo);
    z = zn;
  }
  return z;
}

// log of the interval probability Phi(zh) - Phi(zl) at four nodes, with the two pdf ratios: each lane
// takes the two tail probabilities on its own side of the interval
inline void vinterval(v4d zl, v4d zh, bool lo_open, bool hi_open, v4d& lg, v4d& pl, v4d& ph) {
  const v4d one = splat(1.0), zero = splat(0.0);
  v4d a, la;
  if (lo_open && hi_open) {
    lg = zero;
  } else if (lo_open) {
    pcq::vlogphi(zh, lg, la);
  } else if (hi_open) {
    pcq::vlogphi(-zl, lg, la);
  } else {
    const v4i up = zl > zero, dn = zh < zero;
    const v4d x1 = bvn::vsel(up, -zl, bvn::vsel(dn, zh, zl));
    const v4d x2 = bvn::vsel(up, -zh, bvn::vsel(dn, zl, -zh));
    v4d b, lb;
    pcq::vlogphi(x1, a, la);
    pcq::vlogphi(x2, b, lb);
    const v4d tail = a + bvn::vlog(one - bvn::vexp(b - a));
    const v4d mid = bvn::vlog(one - bvn::vexp(a) - bvn::vexp(b));
    lg = bvn::vsel(up, tail, bvn::vsel(dn, tail, mid));
  }
  const v4d nh = splat(-0.5) * zh * zh - splat(bvn_tail::kLogSqrt2Pi);
  const v4d nl = splat(-0.5) * zl * zl - splat(bvn_tail::kLogSqrt2Pi);
  ph = hi_open ? splat(0.0) : bvn::vexp(nh - lg);
  pl = lo_open ? splat(0.0) : bvn::vexp(nl - lg);
}

// log integral of h over the window, with d/dm, d/ds, d/dlo, d/dhi, d/dsb of the log
inline Cell quad(const Integrand& f) {
  const bool interval = f.sb > 0.0;
  const double a = (interval || f.lo_open) ? -bvn_tail::kInf : f.lo;
  const double b = (interval || f.hi_open) ? bvn_tail::kInf : f.hi;
  const double zm = mode(f, a, b);
  double hm, g, h2;
  f.eval(zm, hm, g, h2);
  const double target = hm - kDrop;
  const double zl = zm > a ? edge(f, zm, target, -1.0, a) : zm;
  const double zr = zm < b ? edge(f, zm, target, 1.0, b) : zm;
  double br[kMaxPieces + 1];
  int nb = 0;
  br[nb++] = zl;
  br[nb++] = zr;
  if (zl < zm && zm < zr) br[nb++] = zm;
  if (interval) {
    const bool steep = 4.0 * kGrade * f.sb < zr - zl;
    for (int t = 0; t < 2; ++t) {
      if (t == 0 ? f.lo_open : f.hi_open) continue;
      const double c = t == 0 ? f.lo : f.hi;
      const double offs[2] = {-kGrade * f.sb, kGrade * f.sb};
      for (int o = 0; o < (steep ? 2 : 1); ++o) {
        const double x = c + (steep ? offs[o] : 0.0);
        if (zl < x && x < zr) br[nb++] = x;
      }
    }
  }
  for (int i = 1; i < nb; ++i)
    for (int j = i; j > 0 && br[j - 1] > br[j]; --j) std::swap(br[j - 1], br[j]);
  const double* X = pcq::kGX32;
  const double* W = pcq::kGW32;
  const v4d vsa = splat(f.sa), vma = splat(f.ma), vs = splat(f.s), vm = splat(f.m), vsb = splat(f.sb);
  const double cA = -std::log(f.sa) - bvn_tail::kLogSqrt2Pi - hm;
  v4d S = splat(0.0), Dm = S, Ds = S, Dlo = S, Dhi = S, Dsb = S;
  for (int p = 0; p + 1 < nb; ++p) {
    const double hh = 0.5 * (br[p + 1] - br[p]);
    if (!(hh > 0.0)) continue;
    for (int i = 0; i < 32; i += 4) {
      v4d x0, w0;
      for (int k = 0; k < 4; ++k) {
        x0[k] = X[i + k];
        w0[k] = W[i + k];
      }
      const v4d w = splat(br[p] + hh) + splat(hh) * x0;
      const v4d u = (w - vma) / vsa;
      v4d H = splat(-0.5) * u * u + splat(cA);
      v4d lp, l;
      if (f.n1) {
        pcq::vlogphi(w, lp, l);
        H += splat(double(f.n1)) * lp;
      }
      v4d dm, ds;
      if (f.filler) {
        const v4d v = (w - vm) / vs;
        pcq::vlogphi(v, lp, l);
        H += lp;
        dm = -l / vs;
        ds = -l * v / vs;
      } else {
        dm = u / vsa;
        ds = (u * u - splat(1.0)) / vsa;
      }
      v4d dlo = splat(0.0), dhi = dlo, dsb = dlo;
      if (interval) {
        const v4d zlv = f.lo_open ? splat(0.0) : (splat(f.lo) - w) / vsb;
        const v4d zhv = f.hi_open ? splat(0.0) : (splat(f.hi) - w) / vsb;
        v4d lg, pl, ph;
        vinterval(zlv, zhv, f.lo_open, f.hi_open, lg, pl, ph);
        H += lg;
        dlo = -pl / vsb;
        dhi = ph / vsb;
        dsb = (pl * zlv - ph * zhv) / vsb;
      }
      const v4d e = w0 * splat(hh) * bvn::vexp(H);
      S += e;
      Dm += e * dm;
      Ds += e * ds;
      if (interval) {
        Dlo += e * dlo;
        Dhi += e * dhi;
        Dsb += e * dsb;
      }
    }
  }
  const double sum = pcq::hsum(S), is = 1.0 / sum;
  Cell c;
  c.lp = hm + std::log(sum);
  c.dm = pcq::hsum(Dm) * is;
  c.ds = pcq::hsum(Ds) * is;
  c.dsb = interval ? pcq::hsum(Dsb) * is : 0.0;
  if (interval) {
    c.dlo = pcq::hsum(Dlo) * is;
    c.dhi = pcq::hsum(Dhi) * is;
  } else {
    double h0, g0, h20;
    if (f.lo_open) {
      c.dlo = 0.0;
    } else {
      f.eval(f.lo, h0, g0, h20);
      c.dlo = -std::exp(h0 - c.lp);
    }
    if (f.hi_open) {
      c.dhi = 0.0;
    } else {
      f.eval(f.hi, h0, g0, h20);
      c.dhi = std::exp(h0 - c.lp);
    }
  }
  return c;
}

// log H(c) = log P(every face below c) and its derivatives in c, m, s (sb = 0)
inline void log_H(int k, double m, double s, double c, double& lh, double& dc, double& dm, double& ds) {
  const double u = (c - m) / s;
  v4d lp, l;
  pcq::vlogphi(v4d{u, c, -u, -c}, lp, l);
  const double pu = u > 0.0 ? std::log1p(-std::exp(lp[2])) : lp[0];
  const double pc = c > 0.0 ? std::log1p(-std::exp(lp[3])) : lp[1];
  lh = pu + (k - 1) * pc;
  dc = l[0] / s + (k - 1) * l[1];
  dm = -l[0] / s;
  ds = -l[0] * u / s;
}

inline Cell any_closed(int k, double m, double s, double lo, double hi, bool lo_open, bool hi_open) {
  double lh = 0.0, dch = 0.0, dmh = 0.0, dsh = 0.0, ll = 0.0, dcl = 0.0, dml = 0.0, dsl = 0.0;
  if (!hi_open) log_H(k, m, s, hi, lh, dch, dmh, dsh);
  Cell c{};
  if (lo_open) {
    c.lp = lh;
    c.dhi = dch;
    c.dm = dmh;
    c.ds = dsh;
    return c;
  }
  log_H(k, m, s, lo, ll, dcl, dml, dsl);
  const double d = ll - lh;
  const double om = -std::expm1(d);
  c.lp = lh + std::log(om);
  const double wh = 1.0 / om, wl = std::exp(d) / om;
  c.dhi = hi_open ? 0.0 : wh * dch;
  c.dlo = -wl * dcl;
  c.dm = wh * dmh - wl * dml;
  c.ds = wh * dsh - wl * dsl;
  return c;
}

// k = 1: the face plus the offset is N(m, s^2 + sb^2)
inline Cell single(double m, double s, double lo, double hi, double sb, bool lo_open, bool hi_open) {
  const double t = std::sqrt(s * s + sb * sb);
  const double zl = lo_open ? -bvn_tail::kInf : (lo - m) / t, zh = hi_open ? bvn_tail::kInf : (hi - m) / t;
  Cell c{};
  double pl = 0.0, ph = 0.0;
  if (lo_open && hi_open) return c;
  if (lo_open) {
    c.lp = bvn_tail::log_Phi(zh);
    ph = std::exp(bvn_tail::log_phi(zh) - c.lp);
  } else if (hi_open) {
    c.lp = bvn_tail::log_Phi(-zl);
    pl = std::exp(bvn_tail::log_phi(zl) - c.lp);
  } else {
    c.lp = bvn_tail::log_diff_Phi(zl, zh);
    ph = std::exp(bvn_tail::log_phi(zh) - c.lp);
    pl = std::exp(bvn_tail::log_phi(zl) - c.lp);
  }
  c.dhi = ph / t;
  c.dlo = -pl / t;
  c.dm = (pl - ph) / t;
  const double dt = (pl * (lo_open ? 0.0 : zl) - ph * (hi_open ? 0.0 : zh)) / t;
  c.ds = dt * s / t;
  c.dsb = dt * sb / t;
  return c;
}

inline Integrand make(Kind kind, int k, double m, double s, double lo, double hi, double sb, bool lo_open, bool hi_open) {
  Integrand f;
  f.m = m;
  f.s = s;
  f.lo = lo;
  f.hi = hi;
  f.sb = sb;
  f.lo_open = lo_open;
  f.hi_open = hi_open;
  f.filler = kind == kFiller;
  if (f.filler) {
    f.ma = 0.0;
    f.sa = 1.0;
    f.n1 = k - 2;
    f.n2 = 1;
  } else {
    f.ma = m;
    f.sa = s;
    f.n1 = k - 1;
    f.n2 = 0;
  }
  return f;
}

// kind: the suspect wins, a filler wins (any of the k - 1), or any face wins; the suspect N(m, s^2),
// fillers N(0, 1); lo / hi the criterion interval on the winner's value plus the offset
inline Cell cell(Kind kind, int k, double m, double s, double lo, double hi, double sb, bool lo_open, bool hi_open) {
  if (k == 1) return single(m, s, lo, hi, sb, lo_open, hi_open);
  if (kind == kAny && sb <= 0.0) return any_closed(k, m, s, lo, hi, lo_open, hi_open);
  if (kind != kAny) {
    Cell c = quad(make(kind, k, m, s, lo, hi, sb, lo_open, hi_open));
    if (kind == kFiller) c.lp += std::log(double(k - 1));
    return c;
  }
  const Cell a = quad(make(kSuspect, k, m, s, lo, hi, sb, lo_open, hi_open));
  Cell b = quad(make(kFiller, k, m, s, lo, hi, sb, lo_open, hi_open));
  b.lp += std::log(double(k - 1));
  const double mx = std::fmax(a.lp, b.lp);
  const double ea = std::exp(a.lp - mx), eb = std::exp(b.lp - mx), t = ea + eb;
  const double wa = ea / t, wb = eb / t;
  return {mx + std::log(t), wa * a.dm + wb * b.dm, wa * a.ds + wb * b.ds, wa * a.dlo + wb * b.dlo,
          wa * a.dhi + wb * b.dhi, wa * a.dsb + wb * b.dsb};
}

}  // namespace lineup
