// batch_sdt_core.hpp — Shared helpers for batch SDT likelihood computation
// Used by dynamically generated model-specific batch functions.

#ifndef BATCH_SDT_CORE_HPP
#define BATCH_SDT_CORE_HPP

#include <stan/math.hpp>
#include <cmath>
#include <limits>

#include "../include/bayesroc/bvn_tail.h"

namespace batch_sdt {

constexpr double kPFloor = 1e-300;
constexpr double kEmptySlope = 10.0;

// ============================================================
// Normal PDF / CDF (probit link)
// ============================================================

inline double phi(double x) {
  return std::exp(-0.5 * x * x) * 0.3989422804014327;
}

inline double Phi(double x) {
  return 0.5 * std::erfc(-x * 0.7071067811865476);
}

inline double Qn(double x) {
  return 0.5 * std::erfc(x * 0.7071067811865476);
}

inline double Phi_diff(double lo, double hi) {
  return lo >= 0.0 ? Qn(lo) - Qn(hi) : Phi(hi) - Phi(lo);
}

// ============================================================
// Cell log-probability and gradients (probit link)
// ============================================================

struct CellResult {
  double lp;
  double d_z_lo;
  double d_z_hi;
};

// Interior cell: log(Phi(z_hi) - Phi(z_lo))
inline CellResult cell_log_prob(double z_lo, double z_hi) {
  double diff = Phi_diff(z_lo, z_hi);
  if (diff < kPFloor) return {std::log(kPFloor), 0.0, 0.0};
  double inv_diff = 1.0 / diff;
  return {std::log(diff), -phi(z_lo) * inv_diff, phi(z_hi) * inv_diff};
}

// Lowest cell: log(Phi(z))
inline CellResult edge_lo(double z) {
  double p = Phi(z);
  if (p < kPFloor) return {std::log(kPFloor), 0.0, 0.0};
  return {std::log(p), 0.0, phi(z) / p};
}

// Highest cell: log(1 - Phi(z))
inline CellResult edge_hi(double z) {
  double q = Qn(z);
  if (q < kPFloor) return {std::log(kPFloor), 0.0, 0.0};
  return {std::log(q), -phi(z) / q, 0.0};
}

// ============================================================
// Threshold gradient propagation
// ============================================================

// Propagate d_lp/d_thresh[k] through mid-anchor + gap parameterization.
// Population-only version (no criterion RE).
inline void propagate_thresh_grad(
    int k, double d,
    int mid, int n_thresh, int n_upper,
    const Eigen::VectorXd& exp_gaps,
    double& grad_thresh_mid,
    Eigen::VectorXd& grad_log_gaps) {

  grad_thresh_mid += d;

  if (k > mid) {
    for (int j = 0; j <= k - mid - 1; ++j) {
      grad_log_gaps(j) += d * exp_gaps(j);
    }
  } else if (k < mid) {
    for (int j = n_upper; j <= n_upper + (mid - k - 1); ++j) {
      grad_log_gaps(j) += d * (-exp_gaps(j));
    }
  }
}

// With criterion RE: accumulates into both population and per-subject RE accumulators.
// exp_eff_gaps are per-observation (include RE contribution).
inline void propagate_thresh_grad_with_re(
    int k, double d,
    int mid, int n_thresh, int n_upper,
    const Eigen::VectorXd& exp_eff_gaps,
    double& grad_thresh_mid_pop,
    Eigen::VectorXd& grad_log_gaps_pop,
    double& grad_u_crit_mid,
    Eigen::VectorXd& grad_u_crit_gaps) {

  grad_thresh_mid_pop += d;
  grad_u_crit_mid += d;

  if (k > mid) {
    for (int j = 0; j <= k - mid - 1; ++j) {
      double g = d * exp_eff_gaps(j);
      grad_log_gaps_pop(j) += g;
      grad_u_crit_gaps(j) += g;
    }
  } else if (k < mid) {
    for (int j = n_upper; j <= n_upper + (mid - k - 1); ++j) {
      double g = d * (-exp_eff_gaps(j));
      grad_log_gaps_pop(j) += g;
      grad_u_crit_gaps(j) += g;
    }
  }
}

// ============================================================
// Cell probability (not log) and raw gradients for mixture models
// ============================================================

// Returns {p, d_p/d_z_lo, d_p/d_z_hi} for a single cell.
// Used by mixture and dpsdt to combine component probabilities
// before taking the log.
struct CellProbResult {
  double p;
  double d_z_lo;   // d_p / d_z_lo
  double d_z_hi;   // d_p / d_z_hi
};

// Interior cell: Phi(z_hi) - Phi(z_lo)
inline CellProbResult cell_prob(double z_lo, double z_hi) {
  double diff = Phi_diff(z_lo, z_hi);
  if (diff < kPFloor) return {kPFloor, 0.0, 0.0};
  return {diff, -phi(z_lo), phi(z_hi)};
}

// Lowest cell: Phi(z)
inline CellProbResult edge_lo_prob(double z) {
  double p = Phi(z);
  return {p, 0.0, phi(z)};
}

// Highest cell: 1 - Phi(z)
inline CellProbResult edge_hi_prob(double z) {
  double q = Qn(z);
  return {q, -phi(z), 0.0};
}

// ============================================================
// DPSDT helpers
// ============================================================

// DPSDT result for old items, y == K (recollection possible):
//   P = lambda + (1-lambda) * (1 - Phi(z_K))
struct DpsdtTopResult {
  double lp;
  double d_z;       // d_logP / d_z_K
  double d_lambda;  // d_logP / d_lambda
};

// y == K: P = lambda + (1-lambda)*(1-Phi(z))
inline DpsdtTopResult dpsdt_top_cell(double z, double lambda) {
  double q = Qn(z);
  double Phi_z = 1.0 - q;
  double P = lambda + (1.0 - lambda) * q;
  if (P < kPFloor) return {std::log(kPFloor), 0.0, 0.0};
  double inv_P = 1.0 / P;
  double d_z = -(1.0 - lambda) * phi(z) * inv_P;
  double d_lambda = Phi_z * inv_P;
  return {std::log(P), d_z, d_lambda};
}

// DPSDT result for old items, y < K (familiarity only):
//   P = (1-lambda) * cell_prob
//   log P = log(1-lambda) + log(cell_prob)
struct DpsdtNonTopResult {
  double lp;
  double d_z_lo;    // d_logP / d_z_lo (from cell_log_prob)
  double d_z_hi;    // d_logP / d_z_hi
  double d_lambda;  // d_logP / d_lambda = -1/(1-lambda)
};

inline DpsdtNonTopResult dpsdt_non_top_cell(double z_lo, double z_hi,
                                             double lambda, int yn, int K) {
  CellResult cr;
  if (yn == 1) {
    cr = edge_lo(z_hi);
  } else {
    cr = cell_log_prob(z_lo, z_hi);
  }
  double d_lambda = -1.0 / (1.0 - lambda);
  // Guard extreme lambda
  if (1.0 - lambda < 1e-15) d_lambda = 0.0;
  return {std::log(std::fmax(1.0 - lambda, kPFloor)) + cr.lp,
          cr.d_z_lo, cr.d_z_hi, d_lambda};
}

// ============================================================
// Mixture SDT helpers
// ============================================================

struct MixtureCellResult {
  double lp;
  double d_z1_lo;    // d_logP / d_z1_lo  (component 1)
  double d_z1_hi;    // d_logP / d_z1_hi
  double d_z2_lo;    // d_logP / d_z2_lo  (component 2)
  double d_z2_hi;    // d_logP / d_z2_hi
  double d_lambda;   // d_logP / d_lambda
};

// Mixture: P = lambda * p1 + (1-lambda) * p2
// Both components share the same cell (yn), but different mu/sigma.
inline MixtureCellResult mixture_cell(
    double z1_lo, double z1_hi,
    double z2_lo, double z2_hi,
    double lambda, int yn, int K) {

  CellProbResult cp1, cp2;
  if (yn == 1) {
    cp1 = edge_lo_prob(z1_hi);
    cp2 = edge_lo_prob(z2_hi);
  } else if (yn == K) {
    cp1 = edge_hi_prob(z1_lo);
    cp2 = edge_hi_prob(z2_lo);
  } else {
    cp1 = cell_prob(z1_lo, z1_hi);
    cp2 = cell_prob(z2_lo, z2_hi);
  }

  double P = lambda * cp1.p + (1.0 - lambda) * cp2.p;
  if (P < kPFloor) return {std::log(kPFloor), 0.0, 0.0, 0.0, 0.0, 0.0};
  double inv_P = 1.0 / P;

  return {std::log(P),
          lambda * cp1.d_z_lo * inv_P,
          lambda * cp1.d_z_hi * inv_P,
          (1.0 - lambda) * cp2.d_z_lo * inv_P,
          (1.0 - lambda) * cp2.d_z_hi * inv_P,
          (cp1.p - cp2.p) * inv_P};
}

// ============================================================
// Non-probit links: CDF, survival, pdf and tail-side difference
// ============================================================

struct LogitLink {
  static double cdf(double x) { return 1.0 / (1.0 + std::exp(-x)); }
  static double ccdf(double x) { return cdf(-x); }
  static double pdf(double x) {
    double e = std::exp(-std::fabs(x));
    double d = 1.0 + e;
    return e / (d * d);
  }
  static double diff(double lo, double hi) { return lo >= 0.0 ? ccdf(lo) - ccdf(hi) : cdf(hi) - cdf(lo); }
};

struct CloglogLink {
  static constexpr double kHi = 700.0;
  static constexpr double kMedian = -0.36651292058166435;
  static double cdf(double x) { return -std::expm1(-std::exp(std::fmin(x, kHi))); }
  static double ccdf(double x) { return std::exp(-std::exp(std::fmin(x, kHi))); }
  static double pdf(double x) {
    double xc = std::fmin(x, kHi);
    return std::exp(xc - std::exp(xc));
  }
  static double diff(double lo, double hi) {
    return lo >= kMedian ? ccdf(lo) - ccdf(hi) : cdf(hi) - cdf(lo);
  }
};

template <class L>
inline CellResult cell_log_prob_t(double z_lo, double z_hi) {
  double diff = L::diff(z_lo, z_hi);
  if (diff < kPFloor) return {std::log(kPFloor), 0.0, 0.0};
  double inv_diff = 1.0 / diff;
  return {std::log(diff), -L::pdf(z_lo) * inv_diff, L::pdf(z_hi) * inv_diff};
}

template <class L>
inline CellResult edge_lo_t(double z) {
  double p = L::cdf(z);
  if (p < kPFloor) return {std::log(kPFloor), 0.0, 0.0};
  return {std::log(p), 0.0, L::pdf(z) / p};
}

template <class L>
inline CellResult edge_hi_t(double z) {
  double q = L::ccdf(z);
  if (q < kPFloor) return {std::log(kPFloor), 0.0, 0.0};
  return {std::log(q), -L::pdf(z) / q, 0.0};
}

template <class L>
inline CellProbResult cell_prob_t(double z_lo, double z_hi) {
  double diff = L::diff(z_lo, z_hi);
  if (diff < kPFloor) return {kPFloor, 0.0, 0.0};
  return {diff, -L::pdf(z_lo), L::pdf(z_hi)};
}

template <class L>
inline CellProbResult edge_lo_prob_t(double z) {
  return {L::cdf(z), 0.0, L::pdf(z)};
}

template <class L>
inline CellProbResult edge_hi_prob_t(double z) {
  return {L::ccdf(z), -L::pdf(z), 0.0};
}

template <class L>
inline DpsdtTopResult dpsdt_top_cell_t(double z, double lambda) {
  double q = L::ccdf(z);
  double P = lambda + (1.0 - lambda) * q;
  if (P < kPFloor) return {std::log(kPFloor), 0.0, 0.0};
  double inv_P = 1.0 / P;
  return {std::log(P), -(1.0 - lambda) * L::pdf(z) * inv_P, L::cdf(z) * inv_P};
}

template <class L>
inline DpsdtNonTopResult dpsdt_non_top_cell_t(double z_lo, double z_hi, double lambda, int yn, int K) {
  CellResult cr = yn == 1 ? edge_lo_t<L>(z_hi) : cell_log_prob_t<L>(z_lo, z_hi);
  double d_lambda = 1.0 - lambda < 1e-15 ? 0.0 : -1.0 / (1.0 - lambda);
  return {std::log(std::fmax(1.0 - lambda, kPFloor)) + cr.lp, cr.d_z_lo, cr.d_z_hi, d_lambda};
}

template <class L>
inline MixtureCellResult mixture_cell_t(double z1_lo, double z1_hi, double z2_lo, double z2_hi,
                                        double lambda, int yn, int K) {
  CellProbResult cp1, cp2;
  if (yn == 1) {
    cp1 = edge_lo_prob_t<L>(z1_hi);
    cp2 = edge_lo_prob_t<L>(z2_hi);
  } else if (yn == K) {
    cp1 = edge_hi_prob_t<L>(z1_lo);
    cp2 = edge_hi_prob_t<L>(z2_lo);
  } else {
    cp1 = cell_prob_t<L>(z1_lo, z1_hi);
    cp2 = cell_prob_t<L>(z2_lo, z2_hi);
  }
  double P = lambda * cp1.p + (1.0 - lambda) * cp2.p;
  if (P < kPFloor) return {std::log(kPFloor), 0.0, 0.0, 0.0, 0.0, 0.0};
  double inv_P = 1.0 / P;
  return {std::log(P),
          lambda * cp1.d_z_lo * inv_P,
          lambda * cp1.d_z_hi * inv_P,
          (1.0 - lambda) * cp2.d_z_lo * inv_P,
          (1.0 - lambda) * cp2.d_z_hi * inv_P,
          (cp1.p - cp2.p) * inv_P};
}

inline CellResult cell_log_prob_logit(double a, double b) { return cell_log_prob_t<LogitLink>(a, b); }
inline CellResult edge_lo_logit(double z) { return edge_lo_t<LogitLink>(z); }
inline CellResult edge_hi_logit(double z) { return edge_hi_t<LogitLink>(z); }
inline DpsdtTopResult dpsdt_top_cell_logit(double z, double l) { return dpsdt_top_cell_t<LogitLink>(z, l); }
inline DpsdtNonTopResult dpsdt_non_top_cell_logit(double a, double b, double l, int yn, int K) {
  return dpsdt_non_top_cell_t<LogitLink>(a, b, l, yn, K);
}
inline MixtureCellResult mixture_cell_logit(double a, double b, double c, double d, double l, int yn, int K) {
  return mixture_cell_t<LogitLink>(a, b, c, d, l, yn, K);
}
inline MixtureCellResult source_mixture_cell_logit(double a, double b, double c, double d, double l, int yn,
                                                   int K) {
  return mixture_cell_t<LogitLink>(a, b, c, d, l, yn, K);
}

inline CellResult cell_log_prob_cloglog(double a, double b) { return cell_log_prob_t<CloglogLink>(a, b); }
inline CellResult edge_lo_cloglog(double z) { return edge_lo_t<CloglogLink>(z); }
inline CellResult edge_hi_cloglog(double z) { return edge_hi_t<CloglogLink>(z); }
inline DpsdtTopResult dpsdt_top_cell_cloglog(double z, double l) { return dpsdt_top_cell_t<CloglogLink>(z, l); }
inline DpsdtNonTopResult dpsdt_non_top_cell_cloglog(double a, double b, double l, int yn, int K) {
  return dpsdt_non_top_cell_t<CloglogLink>(a, b, l, yn, K);
}
inline MixtureCellResult mixture_cell_cloglog(double a, double b, double c, double d, double l, int yn, int K) {
  return mixture_cell_t<CloglogLink>(a, b, c, d, l, yn, K);
}
inline MixtureCellResult source_mixture_cell_cloglog(double a, double b, double c, double d, double l, int yn,
                                                     int K) {
  return mixture_cell_t<CloglogLink>(a, b, c, d, l, yn, K);
}

// ============================================================
// Source mixture helpers (probit link)
// ============================================================

// Source mixture: P = lambda * cell_prob(z_signal) + (1-lambda) * cell_prob(z_noise)
// Returns same struct as MixtureCellResult (z1 = signal, z2 = noise)
inline MixtureCellResult source_mixture_cell(
    double z_sig_lo, double z_sig_hi,
    double z_ref_lo, double z_ref_hi,
    double lambda, int yn, int K) {

  CellProbResult cp_sig, cp_ref;
  if (yn == 1) {
    cp_sig = edge_lo_prob(z_sig_hi);
    cp_ref = edge_lo_prob(z_ref_hi);
  } else if (yn == K) {
    cp_sig = edge_hi_prob(z_sig_lo);
    cp_ref = edge_hi_prob(z_ref_lo);
  } else {
    cp_sig = cell_prob(z_sig_lo, z_sig_hi);
    cp_ref = cell_prob(z_ref_lo, z_ref_hi);
  }

  double P = lambda * cp_sig.p + (1.0 - lambda) * cp_ref.p;
  if (P < kPFloor) return {std::log(kPFloor), 0.0, 0.0, 0.0, 0.0, 0.0};
  double inv_P = 1.0 / P;

  return {std::log(P),
          lambda * cp_sig.d_z_lo * inv_P,
          lambda * cp_sig.d_z_hi * inv_P,
          (1.0 - lambda) * cp_ref.d_z_lo * inv_P,
          (1.0 - lambda) * cp_ref.d_z_hi * inv_P,
          (cp_sig.p - cp_ref.p) * inv_P};
}

// ============================================================
// VRDP2D helpers (probit link only — vrdp2d requires probit)
// ============================================================

// Get univariate cell probability P(Y=y) and gradients d_p/d_mu, d_p/d_sigma
// for a single response category given thresholds, mu, and sigma.
// Returns {p, d_p/d_thresh[y-2] (lo), d_p/d_thresh[y-1] (hi), d_p/d_mu, d_p/d_sigma}.
struct UniCellResult {
  double p;
  double d_thresh_lo;  // d_p / d_thresh[k_lo], 0 if edge
  double d_thresh_hi;  // d_p / d_thresh[k_hi], 0 if edge
  double d_mu;         // d_p / d_mu
  double d_sigma;      // d_p / d_sigma (w.r.t. sigma_val, not log-sigma)
};

// Compute univariate cell probability for response y in 1..K with K-1 thresholds
// z = (thresh - mu) / sigma
inline UniCellResult uni_cell(int y, int K, double mu, double sigma,
                               const double* thresh, int n_thresh) {
  double inv_s = 1.0 / sigma;
  if (y == 1) {
    double z = (thresh[0] - mu) * inv_s;
    double p = Phi(z);
    if (p < kPFloor) return {kPFloor, 0.0, 0.0, 0.0, 0.0};
    double pdf_z = phi(z);
    return {p, 0.0, pdf_z * inv_s, -pdf_z * inv_s, -pdf_z * z * inv_s};
  } else if (y == K) {
    double z = (thresh[n_thresh - 1] - mu) * inv_s;
    double q = Qn(z);
    if (q < kPFloor) return {kPFloor, 0.0, 0.0, 0.0, 0.0};
    double pdf_z = phi(z);
    return {q, -pdf_z * inv_s, 0.0, pdf_z * inv_s, pdf_z * z * inv_s};
  } else {
    double z_lo = (thresh[y - 2] - mu) * inv_s;
    double z_hi = (thresh[y - 1] - mu) * inv_s;
    double p = Phi_diff(z_lo, z_hi);
    if (p < kPFloor) return {kPFloor, 0.0, 0.0, 0.0, 0.0};
    double pdf_lo = phi(z_lo);
    double pdf_hi = phi(z_hi);
    return {p,
            -pdf_lo * inv_s,           // d_p / d_thresh_lo
            pdf_hi * inv_s,            // d_p / d_thresh_hi
            (pdf_lo - pdf_hi) * inv_s, // d_p / d_mu
            (pdf_lo * z_lo - pdf_hi * z_hi) * inv_s};  // d_p / d_sigma
  }
}

// VRDP2D result for a single observation
struct Vrdp2dResult {
  double lp;
  // Gradients w.r.t. the parameters (already through log and mixture):
  double d_dprime_F;    // d_logP / d_dprime_F (on linear predictor scale)
  double d_dprime_R;    // d_logP / d_dprime_R (on linear predictor scale)
  double d_source_d;    // d_logP / d_source_d (discrim for this source)
  double d_lambda;      // d_logP / d_lambda (on inv_logit output R)
  double d_sigma_item;  // d_logP / d_sigma_item (on exp output)
  double d_sigma_S;     // d_logP / d_sigma_S (on exp output)
  // Threshold gradients (d_logP / d_thresh[k]):
  // Dimension 1 (item): lo/hi index and derivatives
  int k1_lo, k1_hi;
  double d_thresh1_lo, d_thresh1_hi;
  // Dimension 2 (source): lo/hi index and derivatives
  int k2_lo, k2_hi;
  double d_thresh2_lo, d_thresh2_hi;
};

// Compute VRDP2D log-likelihood and gradients for one observation.
// thresh1: n_thresh1 item thresholds, thresh2: n_thresh2 source thresholds.
// item_type: 1=new, 2=source A, 3=source B.
// dprime_F, dprime_R, source_d, R, sigma_item, sigma_S are all on transformed scale.
inline Vrdp2dResult vrdp2d_cell(
    int y1, int y2, int item_type, int K1, int K2,
    double dprime_F, double dprime_R, double source_d,
    double R, double sigma_item, double sigma_S,
    const double* thresh1, int n_thresh1,
    const double* thresh2, int n_thresh2) {

  Vrdp2dResult res = {};
  res.k1_lo = -1; res.k1_hi = -1;
  res.k2_lo = -1; res.k2_hi = -1;

  // Set threshold index tracking
  if (y1 == 1) { res.k1_hi = 0; }
  else if (y1 == K1) { res.k1_lo = n_thresh1 - 1; }
  else { res.k1_lo = y1 - 2; res.k1_hi = y1 - 1; }

  if (y2 == 1) { res.k2_hi = 0; }
  else if (y2 == K2) { res.k2_lo = n_thresh2 - 1; }
  else { res.k2_lo = y2 - 2; res.k2_hi = y2 - 1; }

  if (item_type == 1) {
    // New items: P = P1(y1|0,1) * P2(y2|0,1)
    auto c1 = uni_cell(y1, K1, 0.0, 1.0, thresh1, n_thresh1);
    auto c2 = uni_cell(y2, K2, 0.0, 1.0, thresh2, n_thresh2);
    double P = c1.p * c2.p;
    if (P < kPFloor) P = kPFloor;
    res.lp = std::log(P);
    double inv_P = 1.0 / P;
    // d_logP/d_thresh1 = (d_p1/d_thresh1) * p2 / P
    res.d_thresh1_lo = c1.d_thresh_lo * c2.p * inv_P;
    res.d_thresh1_hi = c1.d_thresh_hi * c2.p * inv_P;
    res.d_thresh2_lo = c2.d_thresh_lo * c1.p * inv_P;
    res.d_thresh2_hi = c2.d_thresh_hi * c1.p * inv_P;
    // No gradients for dprime_F, dprime_R, source_d, lambda, sigma for new items
  } else {
    // Old items: P = (1-R) * p1_fam * p2_fam + R * p1_rec * p2_rec
    // Non-recollected: dim1 at (dprime_F, sigma_item), dim2 at (0, 1)
    auto c1_fam = uni_cell(y1, K1, dprime_F, sigma_item, thresh1, n_thresh1);
    auto c2_fam = uni_cell(y2, K2, 0.0, 1.0, thresh2, n_thresh2);
    double p_fam = c1_fam.p * c2_fam.p;

    // Recollected: dim1 at (dprime_F + dprime_R, sigma_item), dim2 at (source_d, sigma_S)
    auto c1_rec = uni_cell(y1, K1, dprime_F + dprime_R, sigma_item, thresh1, n_thresh1);
    auto c2_rec = uni_cell(y2, K2, source_d, sigma_S, thresh2, n_thresh2);
    double p_rec = c1_rec.p * c2_rec.p;

    double P = (1.0 - R) * p_fam + R * p_rec;
    if (P < kPFloor) P = kPFloor;
    res.lp = std::log(P);
    double inv_P = 1.0 / P;

    // d_logP / d_R = (p_rec - p_fam) / P
    res.d_lambda = (p_rec - p_fam) * inv_P;

    // d_logP / d_dprime_F:
    // Through fam component: (1-R) * (d_p1_fam/d_mu * p2_fam) / P
    // Through rec component: R * (d_p1_rec/d_mu * p2_rec) / P
    double d_p_fam_d_dpF = c1_fam.d_mu * c2_fam.p;
    double d_p_rec_d_dpF = c1_rec.d_mu * c2_rec.p;  // d_mu1_rec / d_dprime_F = 1
    res.d_dprime_F = ((1.0 - R) * d_p_fam_d_dpF + R * d_p_rec_d_dpF) * inv_P;

    // d_logP / d_dprime_R:
    // Only through rec component: R * (d_p1_rec/d_mu * p2_rec) / P
    res.d_dprime_R = R * d_p_rec_d_dpF * inv_P;

    // d_logP / d_source_d:
    // Only through rec component dim2: R * (p1_rec * d_p2_rec/d_mu) / P
    res.d_source_d = R * (c1_rec.p * c2_rec.d_mu) * inv_P;

    // d_logP / d_sigma_item:
    // Through fam dim1: (1-R) * (d_p1_fam/d_sigma * p2_fam) / P
    // Through rec dim1: R * (d_p1_rec/d_sigma * p2_rec) / P
    res.d_sigma_item = ((1.0 - R) * c1_fam.d_sigma * c2_fam.p +
                         R * c1_rec.d_sigma * c2_rec.p) * inv_P;

    // d_logP / d_sigma_S:
    // Only through rec dim2: R * (p1_rec * d_p2_rec/d_sigma) / P
    res.d_sigma_S = R * (c1_rec.p * c2_rec.d_sigma) * inv_P;

    // Threshold gradients for dimension 1:
    // d_logP / d_thresh1[k] = ((1-R) * d_p1_fam/d_thresh1[k] * p2_fam +
    //                           R * d_p1_rec/d_thresh1[k] * p2_rec) / P
    res.d_thresh1_lo = ((1.0 - R) * c1_fam.d_thresh_lo * c2_fam.p +
                         R * c1_rec.d_thresh_lo * c2_rec.p) * inv_P;
    res.d_thresh1_hi = ((1.0 - R) * c1_fam.d_thresh_hi * c2_fam.p +
                         R * c1_rec.d_thresh_hi * c2_rec.p) * inv_P;

    // Threshold gradients for dimension 2:
    // d_logP / d_thresh2[k] = ((1-R) * p1_fam * d_p2_fam/d_thresh2[k] +
    //                           R * p1_rec * d_p2_rec/d_thresh2[k]) / P
    res.d_thresh2_lo = ((1.0 - R) * c1_fam.p * c2_fam.d_thresh_lo +
                         R * c1_rec.p * c2_rec.d_thresh_lo) * inv_P;
    res.d_thresh2_hi = ((1.0 - R) * c1_fam.p * c2_fam.d_thresh_hi +
                         R * c1_rec.p * c2_rec.d_thresh_hi) * inv_P;
  }

  return res;
}

// ============================================================
// Bivariate normal CDF and gradients
// ============================================================

// Binormal CDF result with analytic gradients
struct BinormalCdfResult {
  double val;    // Phi_2(z1, z2, rho)
  double d_z1;   // d Phi_2 / d z1
  double d_z2;   // d Phi_2 / d z2
  double d_rho;  // d Phi_2 / d rho
};

// Bivariate normal CDF: P(Z1 <= z1, Z2 <= z2) with correlation rho
// Drezner-Wesolowsky rectangle (bvn_tail::rect_full) with exact derivatives.
inline bvn_tail::DwNodes& dw_cache() {
  static thread_local bvn_tail::DwNodes nd;
  return nd;
}

inline BinormalCdfResult binormal_cdf_grad(double z1, double z2, double rho) {
  const double ninf = -std::numeric_limits<double>::infinity();
  bvn_tail::Rect r = bvn_tail::rect_full(ninf, z1, ninf, z2, rho, dw_cache());
  return {r.p, r.d1h, r.d2h, r.drho};
}


// ============================================================
// Bivariate cell probability with gradients
// ============================================================

// Result for bivariate cell P(Y1=y1, Y2=y2) and its gradients
struct BivariateCellResult {
  double lp;           // log P(y1, y2)
  double d_z1_lo;      // d_logP / d_z1_lo (detection dimension lower z)
  double d_z1_hi;      // d_logP / d_z1_hi (detection dimension upper z)
  double d_z2_lo;      // d_logP / d_z2_lo (discrimination dimension lower z)
  double d_z2_hi;      // d_logP / d_z2_hi (discrimination dimension upper z)
  double d_rho;        // d_logP / d_rho
};

// Compute bivariate cell log-probability and gradients.
// z1_lo, z1_hi: standardized thresholds for dimension 1 (detection)
// z2_lo, z2_hi: standardized thresholds for dimension 2 (discrimination)
// is_lo1/hi1/lo2/hi2: whether this is an edge cell (true = no threshold on that side)
inline BivariateCellResult bivariate_cell(
    double z1_lo, double z1_hi, bool is_lo1, bool is_hi1,
    double z2_lo, double z2_hi, bool is_lo2, bool is_hi2,
    double rho) {
  const double inf = std::numeric_limits<double>::infinity();
  bvn_tail::Rect r = bvn_tail::rect_full(is_lo1 ? -inf : z1_lo, is_hi1 ? inf : z1_hi,
                                         is_lo2 ? -inf : z2_lo, is_hi2 ? inf : z2_hi, rho, dw_cache());
  if (!(r.p > kPFloor)) return {std::log(kPFloor), 0.0, 0.0, 0.0, 0.0, 0.0};
  double inv_P = 1.0 / r.p;
  return {r.lp, r.d1l * inv_P, r.d1h * inv_P, r.d2l * inv_P, r.d2h * inv_P, r.drho * inv_P};
}


// ============================================================
// Bivariate SDT: full observation likelihood with gradients
// ============================================================

// Result for a single bivariate SDT observation
struct BivariateResult {
  double lp;
  // Gradients w.r.t. SDT parameters (on linear predictor / transformed scale):
  double d_dprime;       // d_logP / d_dprime (detection mean)
  double d_discrim;      // d_logP / d_discrim (discrimination mean, pre-negation for source B)
  double d_sigma1;       // d_logP / d_sigma1 (detection SD, on exp output)
  double d_sigma2;       // d_logP / d_sigma2 (discrimination SD, on exp output)
  double d_rho;          // d_logP / d_rho (on tanh output, correlation)
  // Threshold gradients (d_logP / d_thresh[k]):
  int k1_lo, k1_hi;
  double d_thresh1_lo, d_thresh1_hi;
  int k2_lo, k2_hi;
  double d_thresh2_lo, d_thresh2_hi;
};

// Compute bivariate SDT log-likelihood and gradients for one observation.
// item_type: 1=new, 2=source A, 3=source B
// mu1, mu2: means for this item type (already resolved including source negation)
// sigma1, sigma2: SDs for this item type
// rho: correlation for this item type
inline BivariateResult bivariate_sdt_cell(
    int y1, int y2, int K1, int K2,
    double mu1, double mu2, double sigma1, double sigma2, double rho,
    const double* thresh1, int n_thresh1,
    const double* thresh2, int n_thresh2) {

  BivariateResult res = {};
  res.k1_lo = -1; res.k1_hi = -1;
  res.k2_lo = -1; res.k2_hi = -1;

  // Set threshold index tracking
  if (y1 == 1) { res.k1_hi = 0; }
  else if (y1 == K1) { res.k1_lo = n_thresh1 - 1; }
  else { res.k1_lo = y1 - 2; res.k1_hi = y1 - 1; }

  if (y2 == 1) { res.k2_hi = 0; }
  else if (y2 == K2) { res.k2_lo = n_thresh2 - 1; }
  else { res.k2_lo = y2 - 2; res.k2_hi = y2 - 1; }

  double inv_s1 = 1.0 / sigma1;
  double inv_s2 = 1.0 / sigma2;

  // Compute standardized thresholds for this cell
  bool is_lo1 = (y1 == 1);
  bool is_hi1 = (y1 == K1);
  bool is_lo2 = (y2 == 1);
  bool is_hi2 = (y2 == K2);

  double z1_lo = is_lo1 ? -8.0 : (thresh1[y1 - 2] - mu1) * inv_s1;
  double z1_hi = is_hi1 ?  8.0 : (thresh1[y1 - 1] - mu1) * inv_s1;
  double z2_lo = is_lo2 ? -8.0 : (thresh2[y2 - 2] - mu2) * inv_s2;
  double z2_hi = is_hi2 ?  8.0 : (thresh2[y2 - 1] - mu2) * inv_s2;

  auto bc = bivariate_cell(z1_lo, z1_hi, is_lo1, is_hi1,
                            z2_lo, z2_hi, is_lo2, is_hi2, rho);

  res.lp = bc.lp;
  res.d_rho = bc.d_rho;

  // Chain rule: d_logP/d_thresh1[k] = d_logP/d_z * d_z/d_thresh = d_logP/d_z * (1/sigma1)
  res.d_thresh1_lo = bc.d_z1_lo * inv_s1;
  res.d_thresh1_hi = bc.d_z1_hi * inv_s1;
  res.d_thresh2_lo = bc.d_z2_lo * inv_s2;
  res.d_thresh2_hi = bc.d_z2_hi * inv_s2;

  // d_logP/d_mu1 = d_logP/d_z1_lo * (-1/sigma1) + d_logP/d_z1_hi * (-1/sigma1)
  res.d_dprime = -(bc.d_z1_lo + bc.d_z1_hi) * inv_s1;

  // d_logP/d_mu2 = -(d_z2_lo + d_z2_hi) / sigma2
  res.d_discrim = -(bc.d_z2_lo + bc.d_z2_hi) * inv_s2;

  // d_logP/d_sigma1 = d_logP/d_z1 * d_z1/d_sigma1
  // d_z/d_sigma1 = -(thresh - mu1)/sigma1^2 = -z/sigma1
  double d_sigma1 = 0.0;
  if (!is_lo1) d_sigma1 += bc.d_z1_lo * (-z1_lo * inv_s1);
  if (!is_hi1) d_sigma1 += bc.d_z1_hi * (-z1_hi * inv_s1);
  res.d_sigma1 = d_sigma1;

  double d_sigma2 = 0.0;
  if (!is_lo2) d_sigma2 += bc.d_z2_lo * (-z2_lo * inv_s2);
  if (!is_hi2) d_sigma2 += bc.d_z2_hi * (-z2_hi * inv_s2);
  res.d_sigma2 = d_sigma2;

  return res;
}


// ============================================================
// Bivariate DP: full observation likelihood with gradients
// ============================================================

// Result for a single bivariate DP observation
struct BivariateDpResult {
  double lp;
  // Same gradients as BivariateResult, plus lambda and lambda2
  double d_dprime;
  double d_discrim;
  double d_sigma1;
  double d_sigma2;
  double d_rho;
  double d_lambda;     // d_logP / d_R_I    (Source A, on inv_logit output)
  double d_lambda2;    // d_logP / d_R_S    (Source A)
  double d_lambda_B;   // d_logP / d_R_I_B  (Source B)
  double d_lambda2_B;  // d_logP / d_R_S_B  (Source B)
  // Threshold gradients
  int k1_lo, k1_hi;
  double d_thresh1_lo, d_thresh1_hi;
  int k2_lo, k2_hi;
  double d_thresh2_lo, d_thresh2_hi;
};

// Compute bivariate DP log-likelihood for one observation.
// BDP model: P = (1-R_I)*biv_prob + R_I*(1-R_S)*marg_item*delta_src + R_I*R_S*delta_both
// For old items:
//   p1: both from familiarity = (1-R_I) * bivariate_cell_prob
//   p2: item recollected (y1==K1), source from familiarity = R_I*(1-R_S)*marg_source
//   p3: both recollected (y1==K1, y2==1 for A or y2==K2 for B) = R_I*R_S
inline BivariateDpResult bivariate_dp_cell(
    int y1, int y2, int item_type, int K1, int K2,
    double mu1, double mu2, double sigma1, double sigma2, double rho,
    double R_I, double R_S, double R_I_B, double R_S_B,
    const double* thresh1, int n_thresh1,
    const double* thresh2, int n_thresh2) {

  BivariateDpResult res = {};
  res.k1_lo = -1; res.k1_hi = -1;
  res.k2_lo = -1; res.k2_hi = -1;

  // Set threshold index tracking
  if (y1 == 1) { res.k1_hi = 0; }
  else if (y1 == K1) { res.k1_lo = n_thresh1 - 1; }
  else { res.k1_lo = y1 - 2; res.k1_hi = y1 - 1; }

  if (y2 == 1) { res.k2_hi = 0; }
  else if (y2 == K2) { res.k2_lo = n_thresh2 - 1; }
  else { res.k2_lo = y2 - 2; res.k2_hi = y2 - 1; }

  if (item_type == 1) {
    // New items: standard bivariate, no recollection
    // rho is rho_N for new items (passed in as rho)
    auto bv = bivariate_sdt_cell(y1, y2, K1, K2, 0.0, 0.0, 1.0, 1.0, rho,
                                  thresh1, n_thresh1, thresh2, n_thresh2);
    res.lp = bv.lp;
    res.d_rho = bv.d_rho;
    res.d_thresh1_lo = bv.d_thresh1_lo;
    res.d_thresh1_hi = bv.d_thresh1_hi;
    res.d_thresh2_lo = bv.d_thresh2_lo;
    res.d_thresh2_hi = bv.d_thresh2_hi;
    return res;
  }

  // Per-source effective recollection probabilities. With R_I_B = R_I and
  // R_S_B = R_S (supplied by the caller) both sources share one pair of
  // recollection parameters. When item_type == 2 the
  // gradient flows back into R_I/R_S (lambda); when item_type == 3 it
  // flows into R_I_B/R_S_B (lambda_B). The result struct keeps the four
  // gradient slots separate so the caller can route them to the correct
  // beta/RE accumulators without needing to know which source produced
  // this observation.
  double R_I_eff = (item_type == 2) ? R_I : R_I_B;
  double R_S_eff = (item_type == 2) ? R_S : R_S_B;

  // Old items
  double inv_s1 = 1.0 / sigma1;
  double inv_s2 = 1.0 / sigma2;

  bool is_lo1 = (y1 == 1), is_hi1 = (y1 == K1);
  bool is_lo2 = (y2 == 1), is_hi2 = (y2 == K2);
  double z1_lo = is_lo1 ? -8.0 : (thresh1[y1 - 2] - mu1) * inv_s1;
  double z1_hi = is_hi1 ?  8.0 : (thresh1[y1 - 1] - mu1) * inv_s1;
  double z2_lo = is_lo2 ? -8.0 : (thresh2[y2 - 2] - mu2) * inv_s2;
  double z2_hi = is_hi2 ?  8.0 : (thresh2[y2 - 1] - mu2) * inv_s2;

  auto bc = bivariate_cell(z1_lo, z1_hi, is_lo1, is_hi1,
                            z2_lo, z2_hi, is_lo2, is_hi2, rho);

  // d_log(biv_prob) / d_param_biv (chain rule into SDT params)
  double d_biv_dmu1 = -(bc.d_z1_lo + bc.d_z1_hi) * inv_s1;
  double d_biv_dmu2 = -(bc.d_z2_lo + bc.d_z2_hi) * inv_s2;
  double d_biv_dsigma1 = 0.0;
  if (!is_lo1) d_biv_dsigma1 += bc.d_z1_lo * (-z1_lo * inv_s1);
  if (!is_hi1) d_biv_dsigma1 += bc.d_z1_hi * (-z1_hi * inv_s1);
  double d_biv_dsigma2 = 0.0;
  if (!is_lo2) d_biv_dsigma2 += bc.d_z2_lo * (-z2_lo * inv_s2);
  if (!is_hi2) d_biv_dsigma2 += bc.d_z2_hi * (-z2_hi * inv_s2);
  double d_biv_dthresh1_lo = bc.d_z1_lo * inv_s1;
  double d_biv_dthresh1_hi = bc.d_z1_hi * inv_s1;
  double d_biv_dthresh2_lo = bc.d_z2_lo * inv_s2;
  double d_biv_dthresh2_hi = bc.d_z2_hi * inv_s2;

  // Fast path: y1 != K1 -> p2 == p3 == 0, mixture collapses to
  // (1 - R_I_eff) * biv_prob. log P = log(1 - R_I_eff) + bc.lp; SDT-param
  // gradients equal bivariate gradients exactly.
  if (y1 != K1) {
    res.lp = std::log(1.0 - R_I_eff) + bc.lp;
    double d_RI_eff = -1.0 / (1.0 - R_I_eff);
    if (item_type == 2) {
      res.d_lambda   = d_RI_eff;
      res.d_lambda_B = 0.0;
    } else {
      res.d_lambda   = 0.0;
      res.d_lambda_B = d_RI_eff;
    }
    res.d_lambda2   = 0.0;
    res.d_lambda2_B = 0.0;
    res.d_dprime = d_biv_dmu1;
    res.d_discrim = d_biv_dmu2;
    res.d_sigma1 = d_biv_dsigma1;
    res.d_sigma2 = d_biv_dsigma2;
    res.d_rho = bc.d_rho;
    res.d_thresh1_lo = d_biv_dthresh1_lo;
    res.d_thresh1_hi = d_biv_dthresh1_hi;
    res.d_thresh2_lo = d_biv_dthresh2_lo;
    res.d_thresh2_hi = d_biv_dthresh2_hi;
    return res;
  }

  // Slow path: y1 == K1 — full mixture; uni_cell is evaluated once.
  double biv_prob = std::exp(bc.lp);
  double p1 = (1.0 - R_I_eff) * biv_prob;

  auto uc = uni_cell(y2, K2, mu2, sigma2, thresh2, n_thresh2);
  double p2 = R_I_eff * (1.0 - R_S_eff) * uc.p;

  // Recollection success cells: Source A items at the low
  // end of the source axis (y2 = 1, "sure A"); Source B items at the high
  // end (y2 = K2, "sure B").
  bool is_corner = (item_type == 2 && y2 == 1) || (item_type == 3 && y2 == K2);
  double p3 = is_corner ? R_I_eff * R_S_eff : 0.0;

  double P = p1 + p2 + p3;
  if (P < kPFloor) {
    res.lp = std::log(kPFloor);
    return res;
  }

  res.lp = std::log(P);
  double inv_P = 1.0 / P;

  double dp_dRI_eff = -biv_prob + (1.0 - R_S_eff) * uc.p;
  if (is_corner) dp_dRI_eff += R_S_eff;
  double d_RI_eff = dp_dRI_eff * inv_P;

  double dp_dRS_eff = -R_I_eff * uc.p;
  if (is_corner) dp_dRS_eff += R_I_eff;
  double d_RS_eff = dp_dRS_eff * inv_P;

  // Route per-source gradients into the matching slots. Source-A items
  // (item_type == 2) feed lambda/lambda2; Source-B items (item_type == 3)
  // feed lambda_B/lambda2_B. Default constrained-equal callers just see
  // the gradient routed identically since R_I_B == R_I etc.
  if (item_type == 2) {
    res.d_lambda    = d_RI_eff;
    res.d_lambda_B  = 0.0;
    res.d_lambda2   = d_RS_eff;
    res.d_lambda2_B = 0.0;
  } else {
    res.d_lambda    = 0.0;
    res.d_lambda_B  = d_RI_eff;
    res.d_lambda2   = 0.0;
    res.d_lambda2_B = d_RS_eff;
  }

  double w1 = (1.0 - R_I_eff) * biv_prob * inv_P;
  double w2 = R_I_eff * (1.0 - R_S_eff) * inv_P;

  res.d_dprime = w1 * d_biv_dmu1;
  res.d_discrim = w1 * d_biv_dmu2 + w2 * uc.d_mu;
  res.d_sigma1 = w1 * d_biv_dsigma1;
  res.d_sigma2 = w1 * d_biv_dsigma2 + w2 * uc.d_sigma;
  res.d_rho = w1 * bc.d_rho;

  res.d_thresh1_lo = w1 * d_biv_dthresh1_lo;
  res.d_thresh1_hi = w1 * d_biv_dthresh1_hi;
  res.d_thresh2_lo = w1 * d_biv_dthresh2_lo + w2 * uc.d_thresh_lo;
  res.d_thresh2_hi = w1 * d_biv_dthresh2_hi + w2 * uc.d_thresh_hi;

  return res;
}

// ============================================================
// CDP (Continuous Dual-Process) helpers
// ============================================================

// CDP strip probability: P(R > c_R, tau_lo < M < tau_hi) = "Remember" probability
// Uses binormal_cdf_grad for forward + gradient computation.
// z_cR = (c_R - mu_R) / sigma_R (standardized recollection criterion)
// z_lo, z_hi = (tau - mu_M) / sigma_M (standardized memory thresholds)
// rho = sigma_R / sigma_M (bivariate correlation)
//
// strip_upper = [Phi(z_hi) - Phi2(z_cR, z_hi, rho)] - [Phi(z_lo) - Phi2(z_cR, z_lo, rho)]
//
// Returns {prob, d_prob/d_z_cR, d_prob/d_z_lo, d_prob/d_z_hi, d_prob/d_rho}
struct CdpStripResult {
  double p;
  double d_z_cR;
  double d_z_lo;
  double d_z_hi;
  double d_rho;
};

// Interior strip: both z_lo and z_hi are finite
inline CdpStripResult cdp_strip_upper(double z_cR, double z_lo, double z_hi, double rho) {
  const double inf = std::numeric_limits<double>::infinity();
  bvn_tail::Rect r = bvn_tail::rect_full(z_cR, inf, z_lo, z_hi, rho, dw_cache());
  return {std::fmax(r.p, kPFloor), r.d1l, r.d2l, r.d2h, r.drho};
}

inline CdpStripResult cdp_strip_lower(double z_cR, double z_lo, double z_hi, double rho) {
  const double inf = std::numeric_limits<double>::infinity();
  bvn_tail::Rect r = bvn_tail::rect_full(-inf, z_cR, z_lo, z_hi, rho, dw_cache());
  return {std::fmax(r.p, kPFloor), r.d1h, r.d2l, r.d2h, r.drho};
}

inline CdpStripResult cdp_strip_upper_lo_edge(double z_cR, double z_hi, double rho) {
  return cdp_strip_upper(z_cR, -std::numeric_limits<double>::infinity(), z_hi, rho);
}

inline CdpStripResult cdp_strip_upper_hi_edge(double z_cR, double z_lo, double rho) {
  return cdp_strip_upper(z_cR, z_lo, std::numeric_limits<double>::infinity(), rho);
}


// CDP result for a single observation (R/K model, n_rkg == 2)
struct CdpResult {
  double lp;
  double d_dprime;      // d_logP / d_dprime_n (recollection linear predictor)
  double d_dprime2;     // d_logP / d_dprime2_n (familiarity linear predictor)
  double d_sigma;       // d_logP / d_sigma_n (log-sigma_R linear predictor)
  double d_sigma2;      // d_logP / d_sigma2_n (log-sigma_F linear predictor)
  double d_rec_crit;    // d_logP / d_rec_crit_n
  double d_corr;        // d_logP / d corr (the item type's own recollection-familiarity correlation)
  // Threshold gradients:
  int k_lo, k_hi;       // threshold indices touched (-1 if edge)
  double d_thresh_lo;   // d_logP / d_tau[k_lo]
  double d_thresh_hi;   // d_logP / d_tau[k_hi]
};

// Compute CDP R/K log-likelihood and gradients for one observation.
// Parameters on their natural/transformed scale:
//   mu_R, mu_F: recollection/familiarity means
//   sigma_R, sigma_F: recollection/familiarity SDs (already exp'd)
//   c_R: recollection criterion
//   tau: J thresholds for old confidence levels
//   old_level_map: maps 0..J-1 to actual confidence levels (0-indexed, i.e., old_level_map[j]-1 == actual 0-based level)
//   rk_n: 1=Remember, 2=Know
//   is_old: 1 for target, 0 for lure
inline CdpResult cdp_rk_cell(
    int yn, int rk_n, int is_old,
    double mu_R, double sigma_R, double mu_F, double sigma_F,
    double c_R, double corr,
    const double* tau, int J,
    const int* old_level_map) {

  CdpResult res = {};
  res.k_lo = -1;
  res.k_hi = -1;

  int old_idx = -1;
  for (int j = 0; j < J; ++j) {
    if (yn == old_level_map[j]) {
      old_idx = j;
      break;
    }
  }
  bool is_old_resp = (old_idx >= 0);

  // Lures: mu_R = mu_F = 0, sigma_R = sigma_F = 1. corr is the item type's own R-F correlation.
  double sR = is_old ? sigma_R : 1.0, sF = is_old ? sigma_F : 1.0;
  double mu_R_eff = is_old ? mu_R : 0.0;
  double mu_M = is_old ? mu_R + mu_F : 0.0;
  double sigma_M = std::sqrt(sR * sR + sF * sF + 2.0 * corr * sR * sF);
  double rho = (sR + corr * sF) / sigma_M;

  double inv_sR = 1.0 / sR;
  double inv_sM = 1.0 / sigma_M;
  double z_cR = (c_R - mu_R_eff) * inv_sR;

  bool judged = rk_n > 0;
  bool is_lo_edge = !is_old_resp;
  bool is_hi_edge = is_old_resp && (old_idx == J - 1);
  const double inf = std::numeric_limits<double>::infinity();
  int k_hi_idx = is_lo_edge ? 0 : old_idx + 1;
  double z_lo = is_lo_edge ? -inf : (tau[old_idx] - mu_M) * inv_sM;
  double z_hi = is_hi_edge ? inf : (tau[k_hi_idx] - mu_M) * inv_sM;

  if (!is_lo_edge) res.k_lo = old_idx;
  if (!is_hi_edge) res.k_hi = k_hi_idx;

  double p_band = is_lo_edge ? Phi(z_hi) : (is_hi_edge ? Qn(z_lo) : Phi_diff(z_lo, z_hi));
  double phi_lo = is_lo_edge ? 0.0 : phi(z_lo);
  double phi_hi = is_hi_edge ? 0.0 : phi(z_hi);

  double prob;
  double d_prob_dzcR = 0.0, d_prob_dzlo, d_prob_dzhi, d_prob_drho = 0.0;

  if (!judged) {
    prob = p_band;
    d_prob_dzlo = -phi_lo;
    d_prob_dzhi = phi_hi;
  } else {
    CdpStripResult strip = rk_n == 1 ? cdp_strip_upper(z_cR, z_lo, z_hi, rho) : cdp_strip_lower(z_cR, z_lo, z_hi, rho);
    prob = strip.p;
    d_prob_dzcR = strip.d_z_cR;
    d_prob_dzlo = strip.d_z_lo;
    d_prob_dzhi = strip.d_z_hi;
    d_prob_drho = strip.d_rho;
  }
  if (is_lo_edge) d_prob_dzlo = 0.0;
  if (is_hi_edge) d_prob_dzhi = 0.0;
  if (prob < kPFloor) prob = kPFloor;

  res.lp = std::log(prob);
  double inv_prob = 1.0 / prob;

  double dlp_dzcR = d_prob_dzcR * inv_prob;
  double dlp_dzlo = d_prob_dzlo * inv_prob;
  double dlp_dzhi = d_prob_dzhi * inv_prob;
  double dlp_drho = d_prob_drho * inv_prob;

  res.d_rec_crit = dlp_dzcR * inv_sR;
  res.d_thresh_lo = dlp_dzlo * inv_sM;
  res.d_thresh_hi = dlp_dzhi * inv_sM;

  // through sigma_M (z_lo, z_hi) and the R-M correlation rho
  double zl_s = is_lo_edge ? 0.0 : dlp_dzlo * (-z_lo * inv_sM);
  double zh_s = is_hi_edge ? 0.0 : dlp_dzhi * (-z_hi * inv_sM);
  double dlp_dsM = zl_s + zh_s;
  double sM3 = sigma_M * sigma_M * sigma_M;
  res.d_corr = dlp_dsM * (sR * sF * inv_sM) + dlp_drho * (sF * inv_sM - (sR + corr * sF) * sR * sF / sM3);

  if (is_old) {
    res.d_dprime = dlp_dzcR * (-inv_sR) + (dlp_dzlo + dlp_dzhi) * (-inv_sM);
    res.d_dprime2 = (dlp_dzlo + dlp_dzhi) * (-inv_sM);

    double drho_dsR = sF * sF * (1.0 - corr * corr) / sM3;
    double dsR = dlp_dzcR * (-z_cR * inv_sR) + dlp_dsM * ((sR + corr * sF) * inv_sM) + dlp_drho * drho_dsR;
    res.d_sigma = dsR * sigma_R;

    double drho_dsF = corr * inv_sM - (sR + corr * sF) * (sF + corr * sR) / sM3;
    double dsF = dlp_dsM * ((sF + corr * sR) * inv_sM) + dlp_drho * drho_dsF;
    res.d_sigma2 = dsF * sigma_F;
  }

  return res;
}


// Forward declarations for G_cdp infrastructure used by cdp_rkg_cell below.
// Full definitions live further down in the file (struct GCdpResult + the
// G_cdp_grad / G_cdp_grad_reuse functions implementing the Wixted/Mickes
// R/K/G trapezoid decomposition).
struct GCdpResult {
  double val;
  double d_c;
  double d_mu_R;
  double d_sigma_R;
  double d_mu_F;
  double d_sigma_F;
  double d_c_R;
  double d_c_K;
  double d_corr;
};
inline GCdpResult G_cdp_grad_reuse(
    double c, double mu_R, double sigma_R, double mu_F, double sigma_F,
    double c_R, double c_K, double corr, const BinormalCdfResult& bv_cR);


// CDP R/K/G result (n_rkg == 3) — adds know_crit gradient
struct CdpRkgResult {
  double lp;
  double d_dprime;
  double d_dprime2;
  double d_sigma;
  double d_sigma2;
  double d_rec_crit;
  double d_know_crit;
  double d_corr;
  int k_lo, k_hi;
  double d_thresh_lo;
  double d_thresh_hi;
};

// Compute CDP R/K/G log-likelihood for one observation (Wixted & Mickes 2010).
//
// Decision rule (matches the R/K/G model implemented by cdp_loglik in JAX's
// fused path and compute_rkg_probs_r in R):
//   Remember: R > c_R
//   Know:     R <= c_R  AND  F > c_K
//   Guess:    R <= c_R  AND  F <= c_K
// where c_K is a criterion on F (not a second criterion on R).
//
// Probabilities within a confidence band (tau_lo < M < tau_hi):
//   p_R = P(R > c_R, tau_lo < M < tau_hi)  -- binormal strip on (R, M)
//   p_G = G(tau_hi) - G(tau_lo)            -- trapezoid via G_cdp_grad_reuse
//   p_K = band(tau_lo, tau_hi) - p_R - p_G
// G(c) = P(R <= c_R, F <= c_K, M <= c) has a closed form case-split on
// whether c_R is active, implemented in G_cdp_grad / G_cdp_grad_reuse below.
inline CdpRkgResult cdp_rkg_cell(
    int yn, int rk_n, int is_old,
    double mu_R, double sigma_R, double mu_F, double sigma_F,
    double c_R, double c_K, double corr,
    const double* tau, int J,
    const int* old_level_map) {
  CdpRkgResult res = {};
  res.k_lo = -1;
  res.k_hi = -1;
  int old_idx = -1;
  for (int j = 0; j < J; ++j) {
    if (yn == old_level_map[j]) {
      old_idx = j;
      break;
    }
  }
  bool is_old_resp = (old_idx >= 0);
  // Lures: mu_R = mu_F = 0, sigma_R = sigma_F = 1. corr is the item type's own R-F correlation.
  double mu_R_eff, sigma_R_eff, mu_F_eff, sigma_F_eff;
  if (is_old) {
    mu_R_eff = mu_R; sigma_R_eff = sigma_R;
    mu_F_eff = mu_F; sigma_F_eff = sigma_F;
  } else {
    mu_R_eff = 0.0; sigma_R_eff = 1.0;
    mu_F_eff = 0.0; sigma_F_eff = 1.0;
  }
  double sR = sigma_R_eff, sF = sigma_F_eff;
  double mu_M = mu_R_eff + mu_F_eff;
  double sigma_M = std::sqrt(sR * sR + sF * sF + 2.0 * corr * sR * sF);
  double rho_val = (sR + corr * sF) / sigma_M;
  double inv_sR = 1.0 / sR;
  double inv_sM = 1.0 / sigma_M;
  double z_cR = (c_R - mu_R_eff) * inv_sR;
  bool judged = rk_n > 0;
  bool is_lo_edge = !is_old_resp;
  bool is_hi_edge = is_old_resp && (old_idx == J - 1);
  int k_hi_idx = is_lo_edge ? 0 : old_idx + 1;
  double tau_lo_val = is_lo_edge ? -bvn_tail::kInf : tau[old_idx];
  double tau_hi_val = is_hi_edge ? bvn_tail::kInf : tau[k_hi_idx];
  double z_lo = is_lo_edge ? -bvn_tail::kInf : (tau_lo_val - mu_M) * inv_sM;
  double z_hi = is_hi_edge ? bvn_tail::kInf : (tau_hi_val - mu_M) * inv_sM;
  if (!is_lo_edge) res.k_lo = old_idx;
  if (!is_hi_edge) res.k_hi = k_hi_idx;
  double p_band = is_lo_edge ? Phi(z_hi) : (is_hi_edge ? Qn(z_lo) : Phi_diff(z_lo, z_hi));
  double phi_lo = is_lo_edge ? 0.0 : phi(z_lo);
  double phi_hi = is_hi_edge ? 0.0 : phi(z_hi);
  if (judged && rk_n == 3 && !is_lo_edge && c_R + c_K <= tau_lo_val) {
    res.lp = std::log(kPFloor) - kEmptySlope * (tau_lo_val - c_R - c_K);
    res.d_rec_crit = kEmptySlope;
    res.d_know_crit = kEmptySlope;
    res.d_thresh_lo = -kEmptySlope;
    return res;
  }
  BinormalCdfResult bv_hi = {};
  BinormalCdfResult bv_lo = {};
  GCdpResult Ghi = {};
  GCdpResult Glo = {};
  double prob = p_band;
  if (judged) {
    bv_hi = binormal_cdf_grad(z_cR, z_hi, rho_val);
    if (!is_lo_edge) bv_lo = binormal_cdf_grad(z_cR, z_lo, rho_val);
    double p_R_raw = (is_hi_edge ? 1.0 : Phi(z_hi)) - bv_hi.val - ((is_lo_edge ? 0.0 : Phi(z_lo)) - bv_lo.val);
    Ghi = G_cdp_grad_reuse(tau_hi_val, mu_R_eff, sR, mu_F_eff, sF, c_R, c_K, corr, bv_hi);
    if (!is_lo_edge) Glo = G_cdp_grad_reuse(tau_lo_val, mu_R_eff, sR, mu_F_eff, sF, c_R, c_K, corr, bv_lo);
    double p_G_raw = Ghi.val - Glo.val;
    double p_K_raw = p_band - p_R_raw - p_G_raw;
    prob = rk_n == 1 ? p_R_raw : (rk_n == 2 ? p_K_raw : p_G_raw);
  }
  if (judged && !(prob >= bvn_tail::kTailSwitch)) {
    bvn_tail::CdpCell t = bvn_tail::cdp_cell(rk_n, mu_R_eff, sR, mu_F_eff, sF,
                                             c_R, c_K, tau_lo_val, tau_hi_val, corr);
    double lfloor = std::log(kPFloor);
    if (!(t.lp > lfloor)) {
      res.lp = lfloor;
      return res;
    }
    res.lp = t.lp;
    if (is_old) {
      res.d_dprime = t.d[0];
      res.d_sigma = t.d[1] * sigma_R_eff;
      res.d_dprime2 = t.d[2];
      res.d_sigma2 = t.d[3] * sigma_F_eff;
    }
    res.d_rec_crit = t.d[4];
    res.d_know_crit = t.d[5];
    res.d_thresh_lo = is_lo_edge ? 0.0 : t.d[6];
    res.d_thresh_hi = is_hi_edge ? 0.0 : t.d[7];
    res.d_corr = t.d[8];
    return res;
  }
  if (prob < kPFloor) prob = kPFloor;
  res.lp = std::log(prob);
  double inv_prob = 1.0 / prob;
  double dlp_dpR = 0.0, dlp_dpK = 0.0, dlp_dpG = 0.0, dlp_dpB = 0.0;
  if (!judged) dlp_dpB = inv_prob;
  else if (rk_n == 1) dlp_dpR = inv_prob;
  else if (rk_n == 2) dlp_dpK = inv_prob;
  else dlp_dpG = inv_prob;
  double eff_dpR = dlp_dpR - dlp_dpK;
  double eff_dpG = dlp_dpG - dlp_dpK;
  double eff_dpBand = dlp_dpK + dlp_dpB;
  double dlp_dzcR = eff_dpR * (-bv_hi.d_z1 + bv_lo.d_z1);
  double dlp_dzhi = is_hi_edge ? 0.0 : eff_dpR * (phi_hi - bv_hi.d_z2) + eff_dpBand * phi_hi;
  double dlp_dzlo = is_lo_edge ? 0.0 : eff_dpR * (-(phi_lo - bv_lo.d_z2)) + eff_dpBand * (-phi_lo);
  double dlp_drho = eff_dpR * (-bv_hi.d_rho + bv_lo.d_rho);
  double dG_dmuR  = Ghi.d_mu_R  - Glo.d_mu_R;
  double dG_dsigR = Ghi.d_sigma_R - Glo.d_sigma_R;
  double dG_dmuF  = Ghi.d_mu_F  - Glo.d_mu_F;
  double dG_dsigF = Ghi.d_sigma_F - Glo.d_sigma_F;
  double dG_dcR   = Ghi.d_c_R  - Glo.d_c_R;
  double dG_dcK   = Ghi.d_c_K  - Glo.d_c_K;
  double dG_dcorr = Ghi.d_corr - Glo.d_corr;
  double dG_dtau_hi = Ghi.d_c;
  double dG_dtau_lo = Glo.d_c;
  double zh_s = is_hi_edge ? 0.0 : dlp_dzhi * (-z_hi * inv_sM);
  double zl_s = is_lo_edge ? 0.0 : dlp_dzlo * (-z_lo * inv_sM);
  double dlp_dsM = zh_s + zl_s;
  double sM3 = sigma_M * sigma_M * sigma_M;
  res.d_corr = dlp_dsM * (sR * sF * inv_sM)
             + dlp_drho * (sF * inv_sM - (sR + corr * sF) * sR * sF / sM3)
             + eff_dpG * dG_dcorr;
  if (is_old) {
    res.d_dprime  = dlp_dzcR * (-inv_sR)
                  + (dlp_dzlo + dlp_dzhi) * (-inv_sM)
                  + eff_dpG * dG_dmuR;
    res.d_dprime2 = (dlp_dzlo + dlp_dzhi) * (-inv_sM)
                  + eff_dpG * dG_dmuF;
    double drho_dsigR = sF * sF * (1.0 - corr * corr) / sM3;
    double d_sigR_raw = dlp_dzcR * (-z_cR * inv_sR)
                      + dlp_dsM * ((sR + corr * sF) * inv_sM)
                      + dlp_drho * drho_dsigR
                      + eff_dpG * dG_dsigR;
    res.d_sigma = d_sigR_raw * sigma_R_eff;
    double drho_dsigF = corr * inv_sM - (sR + corr * sF) * (sF + corr * sR) / sM3;
    double d_sigF_raw = dlp_dsM * ((sF + corr * sR) * inv_sM)
                      + dlp_drho * drho_dsigF
                      + eff_dpG * dG_dsigF;
    res.d_sigma2 = d_sigF_raw * sigma_F_eff;
  }
  res.d_rec_crit  = dlp_dzcR * inv_sR + eff_dpG * dG_dcR;
  res.d_know_crit = eff_dpG * dG_dcK;
  res.d_thresh_lo = is_lo_edge ? 0.0 : (dlp_dzlo * inv_sM + eff_dpG * (-dG_dtau_lo));
  res.d_thresh_hi = is_hi_edge ? 0.0 : (dlp_dzhi * inv_sM + eff_dpG * dG_dtau_hi);
  return res;
}


// G_cdp gradient computation reusing pre-computed Phi2(z_cR, z_c, rho).
// In cdp_rkg_cell, bv_cR (= binormal_cdf_grad(z_cR, z_c, rho)) is already
// computed for the Remember term. Passing it here saves up to 2 bivariate-normal
// CDF evaluations per threshold boundary (4 total for hi+lo).
inline GCdpResult G_cdp_grad_reuse(
    double c, double mu_R, double sigma_R, double mu_F, double sigma_F,
    double c_R, double c_K, double corr,
    const BinormalCdfResult& bv_cR_pre) {
  double inv_sR = 1.0 / sigma_R;
  double inv_sF = 1.0 / sigma_F;
  double z_cR = (c_R - mu_R) * inv_sR;
  double z_cK_F = (c_K - mu_F) * inv_sF;
  double mu_M = mu_R + mu_F;
  double sigma_M = std::sqrt(sigma_R * sigma_R + sigma_F * sigma_F + 2.0 * corr * sigma_R * sigma_F);
  double inv_sM = 1.0 / sigma_M;
  double z_c = (c - mu_M) * inv_sM;
  double rho = (sigma_R + corr * sigma_F) * inv_sM;
  double sM3 = sigma_M * sigma_M * sigma_M;
  double dsM_dsR = (sigma_R + corr * sigma_F) * inv_sM;
  double dsM_dsF = (sigma_F + corr * sigma_R) * inv_sM;
  double dsM_dc = sigma_R * sigma_F * inv_sM;
  double drho_dsR = sigma_F * sigma_F * (1.0 - corr * corr) / sM3;
  double drho_dsF = corr * inv_sM - (sigma_R + corr * sigma_F) * (sigma_F + corr * sigma_R) / sM3;
  double drho_dc = sigma_F * inv_sM - (sigma_R + corr * sigma_F) * sigma_R * sigma_F / sM3;
  // P(R <= R value at za, F <= c_K): a product of normal CDFs for independent R and F, else Phi2 with corr
  auto lower_quadrant = [&](double za, double& val, double& a1, double& a2, double& ac) {
    if (corr == 0.0) {
      double Pa = Phi(za), Pb = Phi(z_cK_F);
      val = Pa * Pb; a1 = phi(za) * Pb; a2 = Pa * phi(z_cK_F); ac = phi(za) * phi(z_cK_F);
    } else {
      auto q = binormal_cdf_grad(za, z_cK_F, corr);
      val = q.val; a1 = q.d_z1; a2 = q.d_z2; ac = q.d_rho;
    }
  };
  bool case1 = (c_R <= c - c_K);
  GCdpResult res = {};
  double P, A1, A2, Ac;
  if (case1) {
    lower_quadrant(z_cR, P, A1, A2, Ac);
    res.val = P;
    res.d_c = 0.0;
    res.d_c_R = A1 * inv_sR;
    res.d_c_K = A2 * inv_sF;
    res.d_mu_R = -A1 * inv_sR;
    res.d_mu_F = -A2 * inv_sF;
    res.d_sigma_R = A1 * (-z_cR * inv_sR);
    res.d_sigma_F = A2 * (-z_cK_F * inv_sF);
    res.d_corr = Ac;
  } else {
    double s = c - c_K;
    double z_s = (s - mu_R) * inv_sR;
    lower_quadrant(z_s, P, A1, A2, Ac);
    const auto& bv1 = bv_cR_pre;
    auto bv2 = binormal_cdf_grad(z_s, z_c, rho);
    res.val = P + bv1.val - bv2.val;
    res.d_c = A1 * inv_sR
            + bv1.d_z2 * inv_sM
            - bv2.d_z1 * inv_sR - bv2.d_z2 * inv_sM;
    res.d_c_R = bv1.d_z1 * inv_sR;
    res.d_c_K = A1 * (-inv_sR)
              + A2 * inv_sF
              - bv2.d_z1 * (-inv_sR);
    res.d_mu_R = A1 * (-inv_sR)
               + bv1.d_z1 * (-inv_sR) + bv1.d_z2 * (-inv_sM)
               - bv2.d_z1 * (-inv_sR) - bv2.d_z2 * (-inv_sM);
    res.d_mu_F = A2 * (-inv_sF)
               + bv1.d_z2 * (-inv_sM) - bv2.d_z2 * (-inv_sM);
    double zc_s = -z_c * inv_sM;
    res.d_sigma_R = A1 * (-z_s * inv_sR)
                  + bv1.d_z1 * (-z_cR * inv_sR)
                  + bv1.d_z2 * zc_s * dsM_dsR
                  + bv1.d_rho * drho_dsR
                  - bv2.d_z1 * (-z_s * inv_sR)
                  - bv2.d_z2 * zc_s * dsM_dsR
                  - bv2.d_rho * drho_dsR;
    res.d_sigma_F = A2 * (-z_cK_F * inv_sF)
                  + bv1.d_z2 * zc_s * dsM_dsF
                  + bv1.d_rho * drho_dsF
                  - bv2.d_z2 * zc_s * dsM_dsF
                  - bv2.d_rho * drho_dsF;
    res.d_corr = Ac + (bv1.d_z2 - bv2.d_z2) * zc_s * dsM_dc + (bv1.d_rho - bv2.d_rho) * drho_dc;
  }
  return res;
}


// ============================================================
// Bounded bivariate SDT: cell probability with analytic gradients
// ============================================================

// Bounded bivariate SDT probability computation.
// When the detection boundary falls within the cell, the source dimension
// uses a conditional normal: below the crossover point cp, source ~ N(0, sigma_cond);
// above cp, standard bivariate with correlation rho.
//
// cp = mu_I - sigma_I * mu_S / (rho * sigma_S)
//
// Returns BivariateResult with gradients w.r.t. dprime, discrim, sigma1, sigma2, rho,
// and threshold indices.
inline BivariateResult bounded_bivariate_sdt_cell(
    int y1, int y2, int K1, int K2,
    double mu1, double mu2, double sigma1, double sigma2, double rho,
    const double* thresh1, int n_thresh1,
    const double* thresh2, int n_thresh2) {

  BivariateResult res = {};
  res.k1_lo = -1; res.k1_hi = -1;
  res.k2_lo = -1; res.k2_hi = -1;

  // Set threshold index tracking
  if (y1 == 1) { res.k1_hi = 0; }
  else if (y1 == K1) { res.k1_lo = n_thresh1 - 1; }
  else { res.k1_lo = y1 - 2; res.k1_hi = y1 - 1; }

  if (y2 == 1) { res.k2_hi = 0; }
  else if (y2 == K2) { res.k2_lo = n_thresh2 - 1; }
  else { res.k2_lo = y2 - 2; res.k2_hi = y2 - 1; }

  // Small rho: fall back to standard (independent) bivariate
  if (std::fabs(rho) < 1e-10) {
    return bivariate_sdt_cell(y1, y2, K1, K2, mu1, mu2, sigma1, sigma2, 0.0,
                               thresh1, n_thresh1, thresh2, n_thresh2);
  }

  double inv_s1 = 1.0 / sigma1;
  double inv_s2 = 1.0 / sigma2;
  double sigma_cond = sigma2 * std::sqrt(1.0 - rho * rho);
  double inv_sc = 1.0 / sigma_cond;

  // Crossover point: below cp, source conditional mean is 0
  double cp = mu1 - sigma1 * mu2 / (rho * sigma2);

  // Detection cell boundaries
  bool is_lo1 = (y1 == 1), is_hi1 = (y1 == K1);
  bool is_lo2 = (y2 == 1), is_hi2 = (y2 == K2);
  double y_lo = is_lo1 ? -1e8 : thresh1[y1 - 2];
  double y_hi = is_hi1 ?  1e8 : thresh1[y1 - 1];

  bool below_only = (y_hi <= cp);
  bool above_only = (y_lo >= cp);

  double P = 0.0;
  // We accumulate gradients w.r.t. the SDT parameters
  double d_mu1 = 0.0, d_mu2 = 0.0, d_s1 = 0.0, d_s2 = 0.0, d_rho_acc = 0.0;
  double d_t1_lo = 0.0, d_t1_hi = 0.0, d_t2_lo = 0.0, d_t2_hi = 0.0;

  if (below_only) {
    // Entire detection cell is below cp: source ~ N(0, sigma_cond), independent
    auto bv = bivariate_sdt_cell(y1, y2, K1, K2, mu1, 0.0, sigma1, sigma_cond, 0.0,
                                  thresh1, n_thresh1, thresh2, n_thresh2);
    res.lp = bv.lp;
    // bv.d_sigma2 is d_logP/d_sigma_cond. Chain rule for actual sigma2 and rho:
    // sigma_cond = sigma2 * sqrt(1 - rho^2)
    // d_sigma_cond/d_sigma2 = sqrt(1 - rho^2)
    // d_sigma_cond/d_rho = -rho * sigma2 / sqrt(1 - rho^2)
    double sqrt_1mrr = std::sqrt(1.0 - rho * rho);
    res.d_dprime = bv.d_dprime;
    res.d_discrim = 0.0;  // mu2 doesn't affect prob in below-only case
    res.d_sigma1 = bv.d_sigma1;
    res.d_sigma2 = bv.d_sigma2 * sqrt_1mrr;
    res.d_rho = bv.d_sigma2 * (-rho * sigma2 / (sqrt_1mrr + 1e-30));
    res.d_thresh1_lo = bv.d_thresh1_lo;
    res.d_thresh1_hi = bv.d_thresh1_hi;
    res.d_thresh2_lo = bv.d_thresh2_lo;
    res.d_thresh2_hi = bv.d_thresh2_hi;
    return res;
  }
  else if (above_only) {
    // Entire detection cell is above cp: standard bivariate
    return bivariate_sdt_cell(y1, y2, K1, K2, mu1, mu2, sigma1, sigma2, rho,
                               thresh1, n_thresh1, thresh2, n_thresh2);
  }

  // Mixed case: cell straddles cp
  // Below-cp region: detection in [y_lo, cp], source ~ N(0, sigma_cond), independent
  double z_cp = (cp - mu1) * inv_s1;
  double z_ylo_b = is_lo1 ? -8.0 : (thresh1[y1 - 2] - mu1) * inv_s1;

  double p_det_below = Phi_diff(z_ylo_b, z_cp);

  // Source probability in below-cp region
  double p_src_below = 0.0;
  double zs_lo_b = 0.0, zs_hi_b = 0.0;
  if (is_lo2) {
    zs_hi_b = thresh2[0] * inv_sc;
    p_src_below = Phi(zs_hi_b);
  } else if (is_hi2) {
    zs_lo_b = thresh2[n_thresh2 - 1] * inv_sc;
    p_src_below = Qn(zs_lo_b);
  } else {
    zs_lo_b = thresh2[y2 - 2] * inv_sc;
    zs_hi_b = thresh2[y2 - 1] * inv_sc;
    p_src_below = Phi_diff(zs_lo_b, zs_hi_b);
  }

  double prob_below = p_det_below * p_src_below;
  P += prob_below;

  // Above-cp region: detection in [cp, y_hi], standard bivariate
  double z_yhi_a = is_hi1 ? 8.0 : (thresh1[y1 - 1] - mu1) * inv_s1;

  // Standardized source thresholds for above-cp region
  double z2_lo_a = is_lo2 ? -8.0 : (thresh2[y2 - 2] - mu2) * inv_s2;
  double z2_hi_a = is_hi2 ?  8.0 : (thresh2[y2 - 1] - mu2) * inv_s2;

  // Compute above-cp bivariate probability using the decomposition
  // P_above = Phi2(z_yhi, z2_hi, rho) - Phi2(z_cp, z2_hi, rho) - ...
  auto bc_above = bivariate_cell(z_cp, z_yhi_a, false, is_hi1,
                                  z2_lo_a, z2_hi_a, is_lo2, is_hi2, rho);
  double prob_above = std::exp(bc_above.lp);
  P += prob_above;

  if (P < kPFloor) {
    res.lp = std::log(kPFloor);
    return res;
  }

  res.lp = std::log(P);
  double inv_P = 1.0 / P;

  // --- Gradient computation ---
  // d_logP/d_thresh1 (detection thresholds)
  // From below-cp: d_P/d_z_ylo_b = -phi(z_ylo_b) * p_src_below
  // d_z_ylo_b / d_thresh1_lo = inv_s1
  if (!is_lo1) {
    // Above-cp: z1_lo = z_cp (not a function of thresh1_lo for the above region)
    // Only below-cp contributes to thresh1_lo gradient
    d_t1_lo = -phi(z_ylo_b) * p_src_below * inv_s1 * inv_P;
  }
  if (!is_hi1) {
    // Only above-cp contributes to thresh1_hi gradient (z_yhi_a = (thresh1_hi - mu1)/sigma1)
    d_t1_hi = prob_above * bc_above.d_z1_hi * inv_s1 * inv_P;
  }

  // d_logP/d_thresh2 (source thresholds)
  // From below-cp: through p_src_below
  double d_below_t2_lo = 0.0, d_below_t2_hi = 0.0;
  if (!is_lo2) {
    d_below_t2_lo = -p_det_below * phi(zs_lo_b) * inv_sc;
  }
  if (!is_hi2) {
    d_below_t2_hi = p_det_below * phi(zs_hi_b) * inv_sc;
  }
  // From above-cp: through bivariate_cell
  double d_above_t2_lo = 0.0, d_above_t2_hi = 0.0;
  if (!is_lo2) d_above_t2_lo = prob_above * bc_above.d_z2_lo * inv_s2;
  if (!is_hi2) d_above_t2_hi = prob_above * bc_above.d_z2_hi * inv_s2;

  d_t2_lo = (d_below_t2_lo + d_above_t2_lo) * inv_P;
  d_t2_hi = (d_below_t2_hi + d_above_t2_hi) * inv_P;

  // d_logP/d_mu1 (dprime): from below and above regions
  // z_cp = (cp - mu1)/sigma1 = -mu2/(rho*sigma2) does not depend on mu1.
  // Below: p_det_below = Phi(z_cp) - Phi(z_ylo_b), z_ylo_b = (thresh1_lo - mu1)/sigma1, so
  //   d(p_det_below)/d(mu1) = phi(z_ylo_b)*inv_s1 (0 at the lo edge); p_src_below does not
  //   depend on mu1.
  double d_Pbelow_dmu1 = 0.0;
  if (!is_lo1) {
    d_Pbelow_dmu1 = phi(z_ylo_b) * inv_s1 * p_src_below;
  }
  // Above: a bivariate cell with z1_lo = z_cp (constant in mu1) and
  //   z1_hi = (thresh1_hi - mu1)/sigma1; the z2 edges depend on mu2 only, so
  //   d_prob_above/d_mu1 = prob_above * bc_above.d_z1_hi * (-inv_s1) (0 at the hi edge).
  double d_Pabove_dmu1 = 0.0;
  if (!is_hi1) {
    d_Pabove_dmu1 = prob_above * bc_above.d_z1_hi * (-inv_s1);
  }
  d_mu1 = (d_Pbelow_dmu1 + d_Pabove_dmu1) * inv_P;

  // d_logP/d_mu2 (discrim)
  // Below: p_src_below doesn't depend on mu2 (source ~ N(0, sigma_cond))
  //   but p_det_below depends on mu2 through z_cp = -mu2/(rho*sigma2):
  //   cp = mu1 - sigma1*mu2/(rho*sigma2)
  //   d_cp/d_mu2 = -sigma1/(rho*sigma2)
  //   d_z_cp/d_mu2 = d_cp/d_mu2 * inv_s1 = -1/(rho*sigma2)
  double dz_cp_dmu2 = -1.0 / (rho * sigma2);
  double dPbelow_dmu2 = phi(z_cp) * dz_cp_dmu2 * p_src_below;

  // Above: z2_lo/z2_hi depend on mu2
  double dPabove_dmu2 = prob_above * (-(bc_above.d_z2_lo + bc_above.d_z2_hi) * inv_s2);
  // Also z_cp depends on mu2 (z1_lo of above region):
  dPabove_dmu2 += prob_above * bc_above.d_z1_lo * dz_cp_dmu2;

  d_mu2 = (dPbelow_dmu2 + dPabove_dmu2) * inv_P;

  // d_logP/d_sigma1:
  // z_cp = -mu2/(rho*sigma2) does NOT depend on sigma1.
  // z_ylo_b = (thresh1_lo - mu1)/sigma1 => dz_ylo/dsigma1 = -z_ylo_b/sigma1
  // z_yhi_a = (thresh1_hi - mu1)/sigma1 => dz_yhi/dsigma1 = -z_yhi_a/sigma1
  // Below: dP_below/dsigma1 = phi(z_ylo_b) * (z_ylo_b/sigma1) * p_src_below [if not lo edge]
  //   (from d/dsigma1 of -Phi(z_ylo_b): -phi(z_ylo_b)*(-z_ylo_b/sigma1))
  // Above: dP_above/dsigma1 = prob_above * bc_above.d_z1_hi * (-z_yhi_a/sigma1) [if not hi edge]
  //   (z1_lo = z_cp doesn't depend on sigma1)
  double dPbelow_ds1 = 0.0;
  if (!is_lo1) {
    dPbelow_ds1 = phi(z_ylo_b) * (z_ylo_b * inv_s1) * p_src_below;
  }
  double dPabove_ds1 = 0.0;
  if (!is_hi1) {
    dPabove_ds1 = prob_above * bc_above.d_z1_hi * (-z_yhi_a * inv_s1);
  }

  // d_logP/d_sigma2:
  // z_cp = -mu2/(rho*sigma2) => dz_cp/dsigma2 = mu2/(rho*sigma2^2)
  // sigma_cond = sigma2*sqrt(1-rho^2) => zs = thresh2/sigma_cond => dzs/dsigma2 = -zs/sigma2
  // z2 = (thresh2-mu2)/sigma2 => dz2/dsigma2 = -z2/sigma2
  double dz_cp_ds2 = mu2 / (rho * sigma2 * sigma2);

  // Below: through p_det_below (via z_cp) and p_src_below (via sigma_cond)
  double dPdet_ds2 = phi(z_cp) * dz_cp_ds2;
  double dPsrc_ds2 = 0.0;
  if (is_lo2) {
    dPsrc_ds2 = phi(zs_hi_b) * (-zs_hi_b * inv_s2);
  } else if (is_hi2) {
    dPsrc_ds2 = phi(zs_lo_b) * (zs_lo_b * inv_s2);
  } else {
    dPsrc_ds2 = phi(zs_hi_b) * (-zs_hi_b * inv_s2)
              - phi(zs_lo_b) * (-zs_lo_b * inv_s2);
  }
  double dPbelow_ds2 = dPdet_ds2 * p_src_below + p_det_below * dPsrc_ds2;

  // Above: through z_cp (z1_lo) and z2 thresholds
  double dPabove_ds2 = prob_above * bc_above.d_z1_lo * dz_cp_ds2;
  if (!is_lo2) dPabove_ds2 += prob_above * bc_above.d_z2_lo * (-z2_lo_a * inv_s2);
  if (!is_hi2) dPabove_ds2 += prob_above * bc_above.d_z2_hi * (-z2_hi_a * inv_s2);

  // d_logP/d_rho:
  // z_cp = -mu2/(rho*sigma2) => dz_cp/drho = mu2/(rho^2 * sigma2)
  // sigma_cond = sigma2*sqrt(1-rho^2) => dzs/drho = zs * rho / (1-rho^2)
  double dz_cp_drho = mu2 / (rho * rho * sigma2);
  double rho_factor = rho / (1.0 - rho * rho);  // for source z-score derivatives

  // Below: through p_det_below (via z_cp) and p_src_below (via sigma_cond)
  double dPdet_drho = phi(z_cp) * dz_cp_drho;
  double dPsrc_drho = 0.0;
  if (is_lo2) {
    dPsrc_drho = phi(zs_hi_b) * (zs_hi_b * rho_factor);
  } else if (is_hi2) {
    dPsrc_drho = -phi(zs_lo_b) * (zs_lo_b * rho_factor);
  } else {
    dPsrc_drho = phi(zs_hi_b) * (zs_hi_b * rho_factor)
               - phi(zs_lo_b) * (zs_lo_b * rho_factor);
  }
  double dPbelow_drho = dPdet_drho * p_src_below + p_det_below * dPsrc_drho;

  // Above: through z_cp (z1_lo) and rho directly (z2 thresholds don't depend on rho)
  double dPabove_drho = prob_above * (bc_above.d_z1_lo * dz_cp_drho + bc_above.d_rho);

  res.d_dprime = d_mu1;
  res.d_discrim = d_mu2;
  res.d_sigma1 = (dPbelow_ds1 + dPabove_ds1) * inv_P;
  res.d_sigma2 = (dPbelow_ds2 + dPabove_ds2) * inv_P;
  res.d_rho = (dPbelow_drho + dPabove_drho) * inv_P;
  res.d_thresh1_lo = d_t1_lo;
  res.d_thresh1_hi = d_t1_hi;
  res.d_thresh2_lo = d_t2_lo;
  res.d_thresh2_hi = d_t2_hi;

  return res;
}

// Bounded marginal source probability (for use in BDP p2 component)
// Returns probability and gradients w.r.t. mu2, sigma2, source thresholds
struct BoundedMarginalResult {
  double p;
  double d_mu2;
  double d_sigma2;
  double d_rho;
  double d_thresh2_lo;
  double d_thresh2_hi;
};

inline BoundedMarginalResult bounded_marginal_source(
    int y2, int K2,
    double mu1, double mu2, double sigma1, double sigma2, double rho,
    const double* thresh2, int n_thresh2) {

  BoundedMarginalResult res = {};

  if (std::fabs(rho) < 1e-10) {
    // Independent: marginal source is N(mu2, sigma2^2)
    auto uc = uni_cell(y2, K2, mu2, sigma2, thresh2, n_thresh2);
    res.p = uc.p;
    res.d_mu2 = uc.d_mu;
    res.d_sigma2 = uc.d_sigma;
    res.d_thresh2_lo = uc.d_thresh_lo;
    res.d_thresh2_hi = uc.d_thresh_hi;
    return res;
  }

  double inv_s2 = 1.0 / sigma2;
  double sigma_cond = sigma2 * std::sqrt(1.0 - rho * rho);
  double inv_sc = 1.0 / sigma_cond;

  // z_cp = -mu2/(rho*sigma2), p_below_det = Phi(z_cp)
  double z_cp = -mu2 / (rho * sigma2);
  double p_below_det = Phi(z_cp);
  double phi_zcp = phi(z_cp);
  double dz_cp_dmu2 = -1.0 / (rho * sigma2);
  double dz_cp_ds2 = mu2 / (rho * sigma2 * sigma2);

  // Below-cp source probability: N(0, sigma_cond)
  double p_src_below = 0.0;
  double zs_lo_v = 0.0, zs_hi_v = 0.0;  // store for gradient use
  double phi_zs_lo = 0.0, phi_zs_hi = 0.0;
  bool is_lo2 = (y2 == 1), is_hi2 = (y2 == K2);
  if (is_lo2) {
    zs_hi_v = thresh2[0] * inv_sc;
    p_src_below = Phi(zs_hi_v);
    phi_zs_hi = phi(zs_hi_v);
  } else if (is_hi2) {
    zs_lo_v = thresh2[n_thresh2 - 1] * inv_sc;
    p_src_below = Qn(zs_lo_v);
    phi_zs_lo = phi(zs_lo_v);
  } else {
    zs_lo_v = thresh2[y2 - 2] * inv_sc;
    zs_hi_v = thresh2[y2 - 1] * inv_sc;
    p_src_below = Phi_diff(zs_lo_v, zs_hi_v);
    phi_zs_lo = phi(zs_lo_v);
    phi_zs_hi = phi(zs_hi_v);
  }

  double p_below = p_below_det * p_src_below;

  // Above-cp: standard BVN marginal
  // prob_above = Phi(z2_hi) - Phi(z2_lo) - Phi2(z_cp, z2_hi, rho) + Phi2(z_cp, z2_lo, rho)
  const double inf = std::numeric_limits<double>::infinity();
  double z2_lo_a = is_lo2 ? -inf : (thresh2[y2 - 2] - mu2) * inv_s2;
  double z2_hi_a = is_hi2 ? inf : (thresh2[y2 - 1] - mu2) * inv_s2;
  bvn_tail::Rect above = bvn_tail::rect_full(z_cp, inf, z2_lo_a, z2_hi_a, rho, dw_cache());
  double prob_above = above.p;
  double A_cp = above.d1l, A_lo = above.d2l, A_hi = above.d2h;
  double zlo_A = is_lo2 ? 0.0 : z2_lo_a * A_lo, zhi_A = is_hi2 ? 0.0 : z2_hi_a * A_hi;

  res.p = p_below + prob_above;

  // --- Gradient computation ---
  // d_mu2: through z_cp (below det + above z1_lo) and z2 (above)
  // dz2/dmu2 = -inv_s2
  double dp_below_dmu2 = phi_zcp * dz_cp_dmu2 * p_src_below;
  double dp_above_dmu2 = A_cp * dz_cp_dmu2 - (A_lo + A_hi) * inv_s2;
  res.d_mu2 = dp_below_dmu2 + dp_above_dmu2;

  // d_sigma2: through z_cp, sigma_cond (below src zs), z2 (above)
  // dzs/dsigma2 = -zs/sigma2, dz2/dsigma2 = -z2/sigma2
  double dp_below_ds2_det = phi_zcp * dz_cp_ds2 * p_src_below;
  double dp_below_ds2_src = 0.0;
  if (is_lo2) {
    dp_below_ds2_src = p_below_det * phi_zs_hi * (-zs_hi_v * inv_s2);
  } else if (is_hi2) {
    dp_below_ds2_src = p_below_det * phi_zs_lo * (zs_lo_v * inv_s2);
  } else {
    dp_below_ds2_src = p_below_det * (phi_zs_hi * (-zs_hi_v * inv_s2)
                                    - phi_zs_lo * (-zs_lo_v * inv_s2));
  }
  double dp_above_ds2 = A_cp * dz_cp_ds2 - (zlo_A + zhi_A) * inv_s2;
  res.d_sigma2 = dp_below_ds2_det + dp_below_ds2_src + dp_above_ds2;

  // d_thresh2_lo (when not lo edge): through zs_lo (below) and z2_lo (above)
  // dzs_lo/dthresh2_lo = inv_sc, dz2_lo/dthresh2_lo = inv_s2
  if (!is_lo2) {
    double dp_below_dt2lo = p_below_det * (-phi_zs_lo * inv_sc);
    double dp_above_dt2lo = A_lo * inv_s2;
    res.d_thresh2_lo = dp_below_dt2lo + dp_above_dt2lo;
  }

  // d_thresh2_hi (when not hi edge): through zs_hi (below) and z2_hi (above)
  // dzs_hi/dthresh2_hi = inv_sc, dz2_hi/dthresh2_hi = inv_s2
  if (!is_hi2) {
    double dp_below_dt2hi = p_below_det * phi_zs_hi * inv_sc;
    double dp_above_dt2hi = A_hi * inv_s2;
    res.d_thresh2_hi = dp_below_dt2hi + dp_above_dt2hi;
  }

  // d_rho: the marginal source depends on rho through z_cp = -mu2/(rho*sigma2),
  // sigma_cond = sigma2*sqrt(1-rho^2) (so dzs/drho = zs * rho/(1-rho^2)), and the
  // binormal_cdf terms (direct r.d_rho + chain through z_cp).
  double dz_cp_drho = mu2 / (rho * rho * sigma2);
  double zs_rho_factor = rho / (1.0 - rho * rho);
  double dp_below_det_drho = phi_zcp * dz_cp_drho;
  double dp_src_below_drho = 0.0;
  if (is_lo2) {
    dp_src_below_drho = phi_zs_hi * zs_hi_v * zs_rho_factor;
  } else if (is_hi2) {
    dp_src_below_drho = -phi_zs_lo * zs_lo_v * zs_rho_factor;
  } else {
    dp_src_below_drho = (phi_zs_hi * zs_hi_v - phi_zs_lo * zs_lo_v) * zs_rho_factor;
  }
  double dp_below_drho = dp_below_det_drho * p_src_below + p_below_det * dp_src_below_drho;
  double dp_above_drho = A_cp * dz_cp_drho + above.drho;
  res.d_rho = dp_below_drho + dp_above_drho;

  return res;
}


// ============================================================
// Bounded Bivariate DP: full observation likelihood with gradients
// ============================================================
// Same as bivariate_dp_cell but uses bounded_bivariate_sdt_cell for p1
// and bounded_marginal_source for p2 (old items only).

inline BivariateDpResult bounded_bivariate_dp_cell(
    int y1, int y2, int item_type, int K1, int K2,
    double mu1, double mu2, double sigma1, double sigma2, double rho,
    double R_I, double R_S, double R_I_B, double R_S_B,
    const double* thresh1, int n_thresh1,
    const double* thresh2, int n_thresh2) {

  BivariateDpResult res = {};
  res.k1_lo = -1; res.k1_hi = -1;
  res.k2_lo = -1; res.k2_hi = -1;

  // Set threshold index tracking
  if (y1 == 1) { res.k1_hi = 0; }
  else if (y1 == K1) { res.k1_lo = n_thresh1 - 1; }
  else { res.k1_lo = y1 - 2; res.k1_hi = y1 - 1; }

  if (y2 == 1) { res.k2_hi = 0; }
  else if (y2 == K2) { res.k2_lo = n_thresh2 - 1; }
  else { res.k2_lo = y2 - 2; res.k2_hi = y2 - 1; }

  if (item_type == 1) {
    // New items: standard bivariate, no recollection
    auto bv = bivariate_sdt_cell(y1, y2, K1, K2, 0.0, 0.0, 1.0, 1.0, rho,
                                  thresh1, n_thresh1, thresh2, n_thresh2);
    res.lp = bv.lp;
    res.d_rho = bv.d_rho;
    res.d_thresh1_lo = bv.d_thresh1_lo;
    res.d_thresh1_hi = bv.d_thresh1_hi;
    res.d_thresh2_lo = bv.d_thresh2_lo;
    res.d_thresh2_hi = bv.d_thresh2_hi;
    return res;
  }

  // Per-source effective recollection (see bivariate_dp_cell for details).
  double R_I_eff = (item_type == 2) ? R_I : R_I_B;
  double R_S_eff = (item_type == 2) ? R_S : R_S_B;

  // Old items — use bounded bivariate for p1
  auto bv = bounded_bivariate_sdt_cell(y1, y2, K1, K2,
                                        mu1, mu2, sigma1, sigma2, rho,
                                        thresh1, n_thresh1, thresh2, n_thresh2);

  // Fast path: y1 != K1 means p2 == p3 == 0, mixture collapses to
  // (1 - R_I_eff) * biv_prob. log P = log(1 - R_I_eff) + bv.lp; gradients
  // w.r.t. SDT params equal bv gradients exactly. Avoids std::exp(bv.lp),
  // bm = {}, weight multiplications, std::log of P.
  if (y1 != K1) {
    res.lp = std::log(1.0 - R_I_eff) + bv.lp;
    double d_RI_eff = -1.0 / (1.0 - R_I_eff);
    if (item_type == 2) {
      res.d_lambda   = d_RI_eff;
      res.d_lambda_B = 0.0;
    } else {
      res.d_lambda   = 0.0;
      res.d_lambda_B = d_RI_eff;
    }
    res.d_lambda2   = 0.0;
    res.d_lambda2_B = 0.0;
    res.d_dprime = bv.d_dprime;
    res.d_discrim = bv.d_discrim;
    res.d_sigma1 = bv.d_sigma1;
    res.d_sigma2 = bv.d_sigma2;
    res.d_rho = bv.d_rho;
    res.d_thresh1_lo = bv.d_thresh1_lo;
    res.d_thresh1_hi = bv.d_thresh1_hi;
    res.d_thresh2_lo = bv.d_thresh2_lo;
    res.d_thresh2_hi = bv.d_thresh2_hi;
    return res;
  }

  // Slow path: y1 == K1 — both p2 and (possibly) p3 contribute.
  double biv_prob = std::exp(bv.lp);
  double p1 = (1.0 - R_I_eff) * biv_prob;

  BoundedMarginalResult bm = bounded_marginal_source(
      y2, K2, mu1, mu2, sigma1, sigma2, rho, thresh2, n_thresh2);
  double p2 = R_I_eff * (1.0 - R_S_eff) * bm.p;

  double p3 = 0.0;
  // Recollection success cells: Source A items at the low
  // end of the source axis (y2 = 1, "sure A"); Source B items at the high
  // end (y2 = K2, "sure B").
  bool is_corner = (item_type == 2 && y2 == 1) || (item_type == 3 && y2 == K2);
  if (is_corner) p3 = R_I_eff * R_S_eff;

  double P = p1 + p2 + p3;
  if (P < kPFloor) {
    res.lp = std::log(kPFloor);
    return res;
  }

  res.lp = std::log(P);
  double inv_P = 1.0 / P;

  // d_logP / d_R_I_eff
  double dp_dRI_eff = -biv_prob + (1.0 - R_S_eff) * bm.p;
  if (is_corner) dp_dRI_eff += R_S_eff;
  double d_RI_eff = dp_dRI_eff * inv_P;

  // d_logP / d_R_S_eff
  double dp_dRS_eff = -R_I_eff * bm.p;
  if (is_corner) dp_dRS_eff += R_I_eff;
  double d_RS_eff = dp_dRS_eff * inv_P;

  if (item_type == 2) {
    res.d_lambda    = d_RI_eff;
    res.d_lambda_B  = 0.0;
    res.d_lambda2   = d_RS_eff;
    res.d_lambda2_B = 0.0;
  } else {
    res.d_lambda    = 0.0;
    res.d_lambda_B  = d_RI_eff;
    res.d_lambda2   = 0.0;
    res.d_lambda2_B = d_RS_eff;
  }

  // Weighted SDT-param gradients
  double w1 = (1.0 - R_I_eff) * biv_prob * inv_P;
  double w2_inv_P = R_I_eff * (1.0 - R_S_eff) * inv_P;

  res.d_dprime = w1 * bv.d_dprime;
  res.d_discrim = w1 * bv.d_discrim + w2_inv_P * bm.d_mu2;
  res.d_sigma1 = w1 * bv.d_sigma1;
  res.d_sigma2 = w1 * bv.d_sigma2 + w2_inv_P * bm.d_sigma2;
  res.d_rho = w1 * bv.d_rho + w2_inv_P * bm.d_rho;

  res.d_thresh1_lo = w1 * bv.d_thresh1_lo;
  res.d_thresh1_hi = w1 * bv.d_thresh1_hi;
  res.d_thresh2_lo = w1 * bv.d_thresh2_lo + w2_inv_P * bm.d_thresh2_lo;
  res.d_thresh2_hi = w1 * bv.d_thresh2_hi + w2_inv_P * bm.d_thresh2_hi;

  return res;
}


}  // namespace batch_sdt

#endif  // BATCH_SDT_CORE_HPP
