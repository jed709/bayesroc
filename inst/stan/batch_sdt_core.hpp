// batch_sdt_core.hpp — Shared helpers for batch SDT likelihood computation
// Used by dynamically generated model-specific batch functions.

#ifndef BATCH_SDT_CORE_HPP
#define BATCH_SDT_CORE_HPP

#include <stan/math.hpp>
#include <cmath>

namespace batch_sdt {

// ============================================================
// Normal PDF / CDF (probit link)
// ============================================================

inline double phi(double x) {
  return std::exp(-0.5 * x * x) * 0.3989422804014327;
}

inline double Phi(double x) {
  return 0.5 * std::erfc(-x * 0.7071067811865476);
}

// ============================================================
// Cell log-probability and gradients (probit link)
// ============================================================

// Numerically stable softplus: log1p(exp(x)) ≈ x for large x
inline double stable_softplus(double x) {
  if (x > 20.0) return x;      // exp(x) >> 1, so log1p(exp(x)) ≈ x
  if (x < -20.0) return 0.0;   // exp(x) ≈ 0, so log1p(exp(x)) ≈ 0
  return std::log1p(std::exp(x));
}

struct CellResult {
  double lp;
  double d_z_lo;
  double d_z_hi;
};

// Interior cell: log(Phi(z_hi) - Phi(z_lo))
inline CellResult cell_log_prob(double z_lo, double z_hi) {
  double p_hi = Phi(z_hi);
  double p_lo = Phi(z_lo);
  double diff = p_hi - p_lo;
  if (diff < 1e-20) return {std::log(1e-20), 0.0, 0.0};
  double inv_diff = 1.0 / diff;
  return {std::log(diff), -phi(z_lo) * inv_diff, phi(z_hi) * inv_diff};
}

// Lowest cell: log(Phi(z))
inline CellResult edge_lo(double z) {
  double p = Phi(z);
  if (p < 1e-20) return {std::log(1e-20), 0.0, 0.0};
  return {std::log(p), 0.0, phi(z) / p};
}

// Highest cell: log(1 - Phi(z))
inline CellResult edge_hi(double z) {
  double q = 1.0 - Phi(z);
  if (q < 1e-20) return {std::log(1e-20), 0.0, 0.0};
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
  double p_hi = Phi(z_hi);
  double p_lo = Phi(z_lo);
  double diff = p_hi - p_lo;
  if (diff < 1e-30) diff = 1e-30;
  return {diff, -phi(z_lo), phi(z_hi)};
}

// Lowest cell: Phi(z)
inline CellProbResult edge_lo_prob(double z) {
  double p = Phi(z);
  return {p, 0.0, phi(z)};
}

// Highest cell: 1 - Phi(z)
inline CellProbResult edge_hi_prob(double z) {
  double q = 1.0 - Phi(z);
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
  double Phi_z = Phi(z);
  double q = 1.0 - Phi_z;
  double P = lambda + (1.0 - lambda) * q;
  if (P < 1e-20) return {std::log(1e-20), 0.0, 0.0};
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
  return {std::log(std::fmax(1.0 - lambda, 1e-20)) + cr.lp,
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
  if (P < 1e-20) return {std::log(1e-20), 0.0, 0.0, 0.0, 0.0, 0.0};
  double inv_P = 1.0 / P;

  return {std::log(P),
          lambda * cp1.d_z_lo * inv_P,
          lambda * cp1.d_z_hi * inv_P,
          (1.0 - lambda) * cp2.d_z_lo * inv_P,
          (1.0 - lambda) * cp2.d_z_hi * inv_P,
          (cp1.p - cp2.p) * inv_P};
}

// ============================================================
// Logistic PDF / CDF (logit link)
// ============================================================

inline double logistic_cdf(double x) {
  return 1.0 / (1.0 + std::exp(-x));
}

inline double logistic_pdf(double x) {
  double e = std::exp(-std::fabs(x));
  double d = 1.0 + e;
  return e / (d * d);
}

// ============================================================
// Cell log-probability and gradients (logit link)
// ============================================================

// Interior cell: log(F(z_hi) - F(z_lo)) with logistic CDF
inline CellResult cell_log_prob_logit(double z_lo, double z_hi) {
  double p_hi = logistic_cdf(z_hi);
  double p_lo = logistic_cdf(z_lo);
  double diff = p_hi - p_lo;
  if (diff < 1e-20) return {std::log(1e-20), 0.0, 0.0};
  double inv_diff = 1.0 / diff;
  return {std::log(diff), -logistic_pdf(z_lo) * inv_diff, logistic_pdf(z_hi) * inv_diff};
}

// Lowest cell: log(F(z)) with logistic CDF
inline CellResult edge_lo_logit(double z) {
  double p = logistic_cdf(z);
  if (p < 1e-20) return {std::log(1e-20), 0.0, 0.0};
  return {std::log(p), 0.0, logistic_pdf(z) / p};
}

// Highest cell: log(1 - F(z)) with logistic CDF
inline CellResult edge_hi_logit(double z) {
  double q = 1.0 - logistic_cdf(z);
  if (q < 1e-20) return {std::log(1e-20), 0.0, 0.0};
  return {std::log(q), -logistic_pdf(z) / q, 0.0};
}

// ============================================================
// Cell probability (not log) for logit link (mixture models)
// ============================================================

inline CellProbResult cell_prob_logit(double z_lo, double z_hi) {
  double p_hi = logistic_cdf(z_hi);
  double p_lo = logistic_cdf(z_lo);
  double diff = p_hi - p_lo;
  if (diff < 1e-30) diff = 1e-30;
  return {diff, -logistic_pdf(z_lo), logistic_pdf(z_hi)};
}

inline CellProbResult edge_lo_prob_logit(double z) {
  double p = logistic_cdf(z);
  return {p, 0.0, logistic_pdf(z)};
}

inline CellProbResult edge_hi_prob_logit(double z) {
  double q = 1.0 - logistic_cdf(z);
  return {q, -logistic_pdf(z), 0.0};
}

// ============================================================
// DPSDT helpers (logit link)
// ============================================================

inline DpsdtTopResult dpsdt_top_cell_logit(double z, double lambda) {
  double F_z = logistic_cdf(z);
  double q = 1.0 - F_z;
  double P = lambda + (1.0 - lambda) * q;
  if (P < 1e-20) return {std::log(1e-20), 0.0, 0.0};
  double inv_P = 1.0 / P;
  double d_z = -(1.0 - lambda) * logistic_pdf(z) * inv_P;
  double d_lambda = F_z * inv_P;
  return {std::log(P), d_z, d_lambda};
}

inline DpsdtNonTopResult dpsdt_non_top_cell_logit(double z_lo, double z_hi,
                                                    double lambda, int yn, int K) {
  CellResult cr;
  if (yn == 1) {
    cr = edge_lo_logit(z_hi);
  } else {
    cr = cell_log_prob_logit(z_lo, z_hi);
  }
  double d_lambda = -1.0 / (1.0 - lambda);
  if (1.0 - lambda < 1e-15) d_lambda = 0.0;
  return {std::log(std::fmax(1.0 - lambda, 1e-20)) + cr.lp,
          cr.d_z_lo, cr.d_z_hi, d_lambda};
}

// ============================================================
// Mixture SDT helpers (logit link)
// ============================================================

inline MixtureCellResult mixture_cell_logit(
    double z1_lo, double z1_hi,
    double z2_lo, double z2_hi,
    double lambda, int yn, int K) {

  CellProbResult cp1, cp2;
  if (yn == 1) {
    cp1 = edge_lo_prob_logit(z1_hi);
    cp2 = edge_lo_prob_logit(z2_hi);
  } else if (yn == K) {
    cp1 = edge_hi_prob_logit(z1_lo);
    cp2 = edge_hi_prob_logit(z2_lo);
  } else {
    cp1 = cell_prob_logit(z1_lo, z1_hi);
    cp2 = cell_prob_logit(z2_lo, z2_hi);
  }

  double P = lambda * cp1.p + (1.0 - lambda) * cp2.p;
  if (P < 1e-20) return {std::log(1e-20), 0.0, 0.0, 0.0, 0.0, 0.0};
  double inv_P = 1.0 / P;

  return {std::log(P),
          lambda * cp1.d_z_lo * inv_P,
          lambda * cp1.d_z_hi * inv_P,
          (1.0 - lambda) * cp2.d_z_lo * inv_P,
          (1.0 - lambda) * cp2.d_z_hi * inv_P,
          (cp1.p - cp2.p) * inv_P};
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
  if (P < 1e-20) return {std::log(1e-20), 0.0, 0.0, 0.0, 0.0, 0.0};
  double inv_P = 1.0 / P;

  return {std::log(P),
          lambda * cp_sig.d_z_lo * inv_P,
          lambda * cp_sig.d_z_hi * inv_P,
          (1.0 - lambda) * cp_ref.d_z_lo * inv_P,
          (1.0 - lambda) * cp_ref.d_z_hi * inv_P,
          (cp_sig.p - cp_ref.p) * inv_P};
}

// ============================================================
// Source mixture helpers (logit link)
// ============================================================

inline MixtureCellResult source_mixture_cell_logit(
    double z_sig_lo, double z_sig_hi,
    double z_ref_lo, double z_ref_hi,
    double lambda, int yn, int K) {

  CellProbResult cp_sig, cp_ref;
  if (yn == 1) {
    cp_sig = edge_lo_prob_logit(z_sig_hi);
    cp_ref = edge_lo_prob_logit(z_ref_hi);
  } else if (yn == K) {
    cp_sig = edge_hi_prob_logit(z_sig_lo);
    cp_ref = edge_hi_prob_logit(z_ref_lo);
  } else {
    cp_sig = cell_prob_logit(z_sig_lo, z_sig_hi);
    cp_ref = cell_prob_logit(z_ref_lo, z_ref_hi);
  }

  double P = lambda * cp_sig.p + (1.0 - lambda) * cp_ref.p;
  if (P < 1e-20) return {std::log(1e-20), 0.0, 0.0, 0.0, 0.0, 0.0};
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
    if (p < 1e-30) p = 1e-30;
    double pdf_z = phi(z);
    return {p, 0.0, pdf_z * inv_s, -pdf_z * inv_s, -pdf_z * z * inv_s};
  } else if (y == K) {
    double z = (thresh[n_thresh - 1] - mu) * inv_s;
    double q = 1.0 - Phi(z);
    if (q < 1e-30) q = 1e-30;
    double pdf_z = phi(z);
    return {q, -pdf_z * inv_s, 0.0, pdf_z * inv_s, pdf_z * z * inv_s};
  } else {
    double z_lo = (thresh[y - 2] - mu) * inv_s;
    double z_hi = (thresh[y - 1] - mu) * inv_s;
    double p = Phi(z_hi) - Phi(z_lo);
    if (p < 1e-30) p = 1e-30;
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
    if (P < 1e-20) P = 1e-20;
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
    if (P < 1e-20) P = 1e-20;
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
// Uses Owen's T function (via boost) for forward value,
// direct analytic formulas for gradients.
inline BinormalCdfResult binormal_cdf_grad(double z1, double z2, double rho) {
  // Edge case: rho ~= 0 => product of univariates
  if (std::fabs(rho) < 1e-10) {
    double p1 = Phi(z1);
    double p2 = Phi(z2);
    double phi1 = phi(z1);
    double phi2 = phi(z2);
    return {p1 * p2, phi1 * p2, phi2 * p1, phi1 * phi2};
  }

  // Forward value via Owen's T
  double z1_safe = (std::fabs(z1) < 1e-10) ? 1e-10 * (z1 >= 0 ? 1.0 : -1.0) : z1;
  double z2_safe = (std::fabs(z2) < 1e-10) ? 1e-10 * (z2 >= 0 ? 1.0 : -1.0) : z2;

  double denom = std::sqrt((1.0 + rho) * (1.0 - rho));
  double a1 = (z2_safe / z1_safe - rho) / denom;
  double a2 = (z1_safe / z2_safe - rho) / denom;

  double product = z1 * z2;
  double delta = (product < 0.0 || (product == 0.0 && (z1 + z2) < 0.0)) ? 1.0 : 0.0;

  double T1 = boost::math::owens_t(z1_safe, a1);
  double T2 = boost::math::owens_t(z2_safe, a2);
  double val = 0.5 * (Phi(z1) + Phi(z2) - delta) - T1 - T2;

  // Clamp to [0, 1]
  if (val < 0.0) val = 0.0;
  if (val > 1.0) val = 1.0;

  // Analytic gradients (known identities for bivariate normal CDF):
  // d Phi_2 / d z1 = phi(z1) * Phi((z2 - rho*z1) / sqrt(1 - rho^2))
  // d Phi_2 / d z2 = phi(z2) * Phi((z1 - rho*z2) / sqrt(1 - rho^2))
  // d Phi_2 / d rho = phi_2(z1, z2, rho) = bivariate normal PDF
  double inv_denom = 1.0 / denom;
  double d_z1 = phi(z1) * Phi((z2 - rho * z1) * inv_denom);
  double d_z2 = phi(z2) * Phi((z1 - rho * z2) * inv_denom);

  // Bivariate normal PDF
  double rho2 = rho * rho;
  double exponent = -(z1 * z1 - 2.0 * rho * z1 * z2 + z2 * z2) / (2.0 * (1.0 - rho2));
  // 1/(2*pi) = 0.15915494309189535
  double d_rho = std::exp(exponent) * 0.15915494309189535 / denom;

  return {val, d_z1, d_z2, d_rho};
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

  // Compute up to 4 binormal CDF values depending on cell type
  // P = Phi2(z1_hi, z2_hi) - Phi2(z1_lo, z2_hi) - Phi2(z1_hi, z2_lo) + Phi2(z1_lo, z2_lo)
  // With edge simplifications:
  //   Phi2(-inf, z) = 0
  //   Phi2(+inf, z) = Phi(z)

  double P = 0.0;
  double d_z1_lo = 0.0, d_z1_hi = 0.0;
  double d_z2_lo = 0.0, d_z2_hi = 0.0;
  double d_rho = 0.0;

  // Corner: bottom-left (y1==1, y2==1)
  if (is_lo1 && is_lo2) {
    auto r = binormal_cdf_grad(z1_hi, z2_hi, rho);
    P = r.val;
    d_z1_hi = r.d_z1;
    d_z2_hi = r.d_z2;
    d_rho = r.d_rho;
  }
  // Corner: bottom-right (y1==1, y2==K2)
  else if (is_lo1 && is_hi2) {
    auto r = binormal_cdf_grad(z1_hi, z2_lo, rho);
    P = Phi(z1_hi) - r.val;
    double phi1 = phi(z1_hi);
    d_z1_hi = phi1 - r.d_z1;
    d_z2_lo = -r.d_z2;
    d_rho = -r.d_rho;
  }
  // Corner: top-left (y1==K1, y2==1)
  else if (is_hi1 && is_lo2) {
    auto r = binormal_cdf_grad(z1_lo, z2_hi, rho);
    P = Phi(z2_hi) - r.val;
    d_z1_lo = -r.d_z1;
    double phi2 = phi(z2_hi);
    d_z2_hi = phi2 - r.d_z2;
    d_rho = -r.d_rho;
  }
  // Corner: top-right (y1==K1, y2==K2)
  else if (is_hi1 && is_hi2) {
    auto r = binormal_cdf_grad(z1_lo, z2_lo, rho);
    P = 1.0 - Phi(z1_lo) - Phi(z2_lo) + r.val;
    d_z1_lo = -phi(z1_lo) + r.d_z1;
    d_z2_lo = -phi(z2_lo) + r.d_z2;
    d_rho = r.d_rho;
  }
  // Edge: bottom row (y1==1, interior y2)
  else if (is_lo1) {
    auto r_hi = binormal_cdf_grad(z1_hi, z2_hi, rho);
    auto r_lo = binormal_cdf_grad(z1_hi, z2_lo, rho);
    P = r_hi.val - r_lo.val;
    d_z1_hi = r_hi.d_z1 - r_lo.d_z1;
    d_z2_hi = r_hi.d_z2;
    d_z2_lo = -r_lo.d_z2;
    d_rho = r_hi.d_rho - r_lo.d_rho;
  }
  // Edge: left column (interior y1, y2==1)
  else if (is_lo2) {
    auto r_hi = binormal_cdf_grad(z1_hi, z2_hi, rho);
    auto r_lo = binormal_cdf_grad(z1_lo, z2_hi, rho);
    P = r_hi.val - r_lo.val;
    d_z1_hi = r_hi.d_z1;
    d_z1_lo = -r_lo.d_z1;
    d_z2_hi = r_hi.d_z2 - r_lo.d_z2;
    d_rho = r_hi.d_rho - r_lo.d_rho;
  }
  // Edge: top row (y1==K1, interior y2)
  else if (is_hi1) {
    auto r_hi = binormal_cdf_grad(z1_lo, z2_hi, rho);
    auto r_lo = binormal_cdf_grad(z1_lo, z2_lo, rho);
    P = Phi(z2_hi) - Phi(z2_lo) - r_hi.val + r_lo.val;
    d_z1_lo = -r_hi.d_z1 + r_lo.d_z1;
    d_z2_hi = phi(z2_hi) - r_hi.d_z2;
    d_z2_lo = -phi(z2_lo) + r_lo.d_z2;
    d_rho = -r_hi.d_rho + r_lo.d_rho;
  }
  // Edge: right column (interior y1, y2==K2)
  else if (is_hi2) {
    auto r_hi = binormal_cdf_grad(z1_hi, z2_lo, rho);
    auto r_lo = binormal_cdf_grad(z1_lo, z2_lo, rho);
    P = Phi(z1_hi) - Phi(z1_lo) - r_hi.val + r_lo.val;
    d_z1_hi = phi(z1_hi) - r_hi.d_z1;
    d_z1_lo = -phi(z1_lo) + r_lo.d_z1;
    d_z2_lo = -r_hi.d_z2 + r_lo.d_z2;
    d_rho = -r_hi.d_rho + r_lo.d_rho;
  }
  // Interior cell
  else {
    auto r11 = binormal_cdf_grad(z1_hi, z2_hi, rho);
    auto r10 = binormal_cdf_grad(z1_hi, z2_lo, rho);
    auto r01 = binormal_cdf_grad(z1_lo, z2_hi, rho);
    auto r00 = binormal_cdf_grad(z1_lo, z2_lo, rho);
    P = r11.val - r10.val - r01.val + r00.val;
    d_z1_hi = r11.d_z1 - r10.d_z1;
    d_z1_lo = -r01.d_z1 + r00.d_z1;
    d_z2_hi = r11.d_z2 - r01.d_z2;
    d_z2_lo = -r10.d_z2 + r00.d_z2;
    d_rho = r11.d_rho - r10.d_rho - r01.d_rho + r00.d_rho;
  }

  // Clamp probability and zero out gradients if clamped
  if (P < 1e-20) {
    return {std::log(1e-20), 0.0, 0.0, 0.0, 0.0, 0.0};
  }

  double inv_P = 1.0 / P;
  return {std::log(P),
          d_z1_lo * inv_P,
          d_z1_hi * inv_P,
          d_z2_lo * inv_P,
          d_z2_hi * inv_P,
          d_rho * inv_P};
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

  // Per-source effective recollection probabilities. Default behavior
  // (R_I_B = R_I, R_S_B = R_S, supplied by the caller) reproduces the
  // pre-asymmetric-lambda likelihood exactly. When item_type == 2 the
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

  // Slow path: y1 == K1 — full mixture. Compute uni_cell ONCE (was called 3x).
  double biv_prob = std::exp(bc.lp);
  double p1 = (1.0 - R_I_eff) * biv_prob;

  auto uc = uni_cell(y2, K2, mu2, sigma2, thresh2, n_thresh2);
  double p2 = R_I_eff * (1.0 - R_S_eff) * uc.p;

  // Recollection success cells (new convention): Source A items at the low
  // end of the source axis (y2 = 1, "sure A"); Source B items at the high
  // end (y2 = K2, "sure B").
  bool is_corner = (item_type == 2 && y2 == 1) || (item_type == 3 && y2 == K2);
  double p3 = is_corner ? R_I_eff * R_S_eff : 0.0;

  double P = p1 + p2 + p3;
  if (P < 1e-20) {
    res.lp = std::log(1e-20);
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
  // Upper part at z_hi: Phi(z_hi) - Phi2(z_cR, z_hi, rho)
  auto bv_hi = binormal_cdf_grad(z_cR, z_hi, rho);
  double u_hi = Phi(z_hi) - bv_hi.val;
  double du_hi_dzcR = -bv_hi.d_z1;
  double du_hi_dzhi = phi(z_hi) - bv_hi.d_z2;
  double du_hi_drho = -bv_hi.d_rho;

  // Upper part at z_lo: Phi(z_lo) - Phi2(z_cR, z_lo, rho)
  auto bv_lo = binormal_cdf_grad(z_cR, z_lo, rho);
  double u_lo = Phi(z_lo) - bv_lo.val;
  double du_lo_dzcR = -bv_lo.d_z1;
  double du_lo_dzlo = phi(z_lo) - bv_lo.d_z2;
  double du_lo_drho = -bv_lo.d_rho;

  double p = u_hi - u_lo;
  if (p < 1e-20) p = 1e-20;

  return {p,
          du_hi_dzcR - du_lo_dzcR,  // d_p / d_z_cR
          -du_lo_dzlo,               // d_p / d_z_lo (negative because u_lo is subtracted)
          du_hi_dzhi,                // d_p / d_z_hi
          du_hi_drho - du_lo_drho};  // d_p / d_rho
}

// Edge strip for y==1 (z_lo = -inf): strip = Phi(z_hi) - Phi2(z_cR, z_hi, rho)
inline CdpStripResult cdp_strip_upper_lo_edge(double z_cR, double z_hi, double rho) {
  auto bv_hi = binormal_cdf_grad(z_cR, z_hi, rho);
  double p = Phi(z_hi) - bv_hi.val;
  if (p < 1e-20) p = 1e-20;

  return {p,
          -bv_hi.d_z1,           // d_p / d_z_cR
          0.0,                    // d_p / d_z_lo (no lower threshold)
          phi(z_hi) - bv_hi.d_z2, // d_p / d_z_hi
          -bv_hi.d_rho};          // d_p / d_rho
}

// Edge strip for y==K (z_hi = +inf): strip = Phi(-z_cR) - [Phi(z_lo) - Phi2(z_cR, z_lo, rho)]
// = Phi(-z_cR) - Phi(z_lo) + Phi2(z_cR, z_lo, rho)
inline CdpStripResult cdp_strip_upper_hi_edge(double z_cR, double z_lo, double rho) {
  auto bv_lo = binormal_cdf_grad(z_cR, z_lo, rho);
  double p = Phi(-z_cR) - Phi(z_lo) + bv_lo.val;
  if (p < 1e-20) p = 1e-20;

  return {p,
          -phi(-z_cR) + bv_lo.d_z1,    // d_p / d_z_cR: d(Phi(-z_cR))/d(z_cR) = -phi(-z_cR)
          -(phi(z_lo) - bv_lo.d_z2),    // d_p / d_z_lo = -phi(z_lo) + d(Phi2)/d(z_lo)
          0.0,                            // d_p / d_z_hi (no upper threshold)
          bv_lo.d_rho};                   // d_p / d_rho
}


// CDP result for a single observation (R/K model, n_rkg == 2)
struct CdpResult {
  double lp;
  double d_dprime;      // d_logP / d_dprime_n (recollection linear predictor)
  double d_dprime2;     // d_logP / d_dprime2_n (familiarity linear predictor)
  double d_sigma;       // d_logP / d_sigma_n (log-sigma_R linear predictor)
  double d_sigma2;      // d_logP / d_sigma2_n (log-sigma_F linear predictor)
  double d_rec_crit;    // d_logP / d_rec_crit_n
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
    double c_R,
    const double* tau, int J,
    const int* old_level_map) {

  CdpResult res = {};
  res.k_lo = -1;
  res.k_hi = -1;

  // Determine if this is an "old" response
  int old_idx = -1;
  for (int j = 0; j < J; ++j) {
    if (yn == old_level_map[j]) {
      old_idx = j;
      break;
    }
  }
  bool is_old_resp = (old_idx >= 0);

  // Derived bivariate parameters
  double mu_M, sigma_M, rho;
  double mu_R_eff, sigma_R_eff;
  if (is_old) {
    mu_R_eff = mu_R;
    sigma_R_eff = sigma_R;
    mu_M = mu_R + mu_F;
    sigma_M = std::sqrt(sigma_R * sigma_R + sigma_F * sigma_F);
    rho = sigma_R / sigma_M;
  } else {
    // Lure: mu_R=0, sigma_R=1, mu_F=0, sigma_F=1
    mu_R_eff = 0.0;
    sigma_R_eff = 1.0;
    mu_M = 0.0;
    sigma_M = std::sqrt(2.0);
    rho = 1.0 / std::sqrt(2.0);
  }

  double inv_sR = 1.0 / sigma_R_eff;
  double inv_sM = 1.0 / sigma_M;
  double z_cR = (c_R - mu_R_eff) * inv_sR;

  if (!is_old_resp) {
    // "New" response: P = Phi((tau[0] - mu_M) / sigma_M)
    double z_tau1 = (tau[0] - mu_M) * inv_sM;
    double p = Phi(z_tau1);
    if (p < 1e-20) p = 1e-20;
    res.lp = std::log(p);
    double d_lp_dz = phi(z_tau1) / p;

    // d_logP / d_tau[0]:
    res.k_hi = 0;
    res.d_thresh_hi = d_lp_dz * inv_sM;

    if (is_old) {
      // d_logP / d_mu_M = d_lp_dz * (-1/sigma_M)
      double d_muM = d_lp_dz * (-inv_sM);
      // mu_M = mu_R + mu_F, so d_dprime = d_muM, d_dprime2 = d_muM
      res.d_dprime = d_muM;
      res.d_dprime2 = d_muM;
      // d_logP / d_sigma_M = d_lp_dz * (-(tau[0]-mu_M)/sigma_M^2)
      double d_sigM = d_lp_dz * (-z_tau1 * inv_sM);
      // sigma_M = sqrt(sR^2 + sF^2), dsigM/dsR = sR/sigM, dsigM/dsF = sF/sigM
      // Chain through exp: d/d(log_sigma_n) = d_sigM * (dsigM/dsR) * sR
      res.d_sigma = d_sigM * (sigma_R / sigma_M) * sigma_R;  // * sigma_R for exp chain
      res.d_sigma2 = d_sigM * (sigma_F / sigma_M) * sigma_F;
    }
    return res;
  }

  // "Old" response: use R/K bivariate strip
  double tau_lo_val = tau[old_idx];
  double tau_hi_val = (old_idx < J - 1) ? tau[old_idx + 1] : 20.0;
  double z_lo = (tau_lo_val - mu_M) * inv_sM;
  double z_hi = (tau_hi_val - mu_M) * inv_sM;

  // Set threshold indices
  res.k_lo = old_idx;
  if (old_idx < J - 1) res.k_hi = old_idx + 1;

  // Compute strip probability and raw gradients
  CdpStripResult strip;
  bool is_lo_edge = false;  // old_idx == 0 means tau_lo is first threshold, not -inf
  bool is_hi_edge = (old_idx == J - 1);  // last old level -> tau_hi = 20 (effectively +inf)

  if (is_hi_edge) {
    strip = cdp_strip_upper_hi_edge(z_cR, z_lo, rho);
  } else {
    strip = cdp_strip_upper(z_cR, z_lo, z_hi, rho);
  }

  double prob;
  double d_prob_dzcR, d_prob_dzlo, d_prob_dzhi, d_prob_drho;

  if (rk_n == 1) {
    // Remember: P = strip_upper
    prob = strip.p;
    d_prob_dzcR = strip.d_z_cR;
    d_prob_dzlo = strip.d_z_lo;
    d_prob_dzhi = strip.d_z_hi;
    d_prob_drho = strip.d_rho;
  } else {
    // Know: P = band - strip_upper
    double p_band = Phi(z_hi) - Phi(z_lo);
    prob = p_band - strip.p;
    if (prob < 1e-20) prob = 1e-20;
    // d_prob/d_z_lo = -phi(z_lo) - strip.d_z_lo
    d_prob_dzlo = -phi(z_lo) - strip.d_z_lo;
    // d_prob/d_z_hi = phi(z_hi) - strip.d_z_hi
    d_prob_dzhi = is_hi_edge ? 0.0 : (phi(z_hi) - strip.d_z_hi);
    d_prob_dzcR = -strip.d_z_cR;
    d_prob_drho = -strip.d_rho;
  }

  res.lp = std::log(prob);
  double inv_prob = 1.0 / prob;

  // d_logP / d_z_foo = (d_prob / d_z_foo) / prob
  double dlp_dzcR = d_prob_dzcR * inv_prob;
  double dlp_dzlo = d_prob_dzlo * inv_prob;
  double dlp_dzhi = d_prob_dzhi * inv_prob;
  double dlp_drho = d_prob_drho * inv_prob;

  // Chain rule: z_cR = (c_R - mu_R) / sigma_R
  // d_z_cR/d_c_R = 1/sigma_R
  // d_z_cR/d_mu_R = -1/sigma_R
  // d_z_cR/d_sigma_R = -(c_R - mu_R)/sigma_R^2 = -z_cR/sigma_R
  res.d_rec_crit = dlp_dzcR * inv_sR;

  // Chain rule: z = (tau - mu_M) / sigma_M
  // d_z/d_mu_M = -1/sigma_M
  // d_z/d_sigma_M = -(tau - mu_M)/sigma_M^2 = -z/sigma_M
  // d_z/d_tau = 1/sigma_M

  // Threshold gradients: d_logP / d_tau[k]
  res.d_thresh_lo = dlp_dzlo * inv_sM;
  res.d_thresh_hi = is_hi_edge ? 0.0 : dlp_dzhi * inv_sM;

  if (is_old) {
    // d_logP / d_mu_R:
    // Through z_cR: dlp_dzcR * (-1/sigma_R)
    // Through z_lo, z_hi (via mu_M): (dlp_dzlo + dlp_dzhi) * (-1/sigma_M)
    // Through rho: dlp_drho * d_rho/d_sigma_R * d_sigma_R/d_mu_R = 0 (mu_R doesn't affect sigma_R directly)
    // Wait — mu_R doesn't affect sigma_R or rho. So:
    double d_muR = dlp_dzcR * (-inv_sR);
    // mu_M = mu_R + mu_F, so d_mu_M/d_mu_R = 1
    d_muR += (dlp_dzlo + (is_hi_edge ? 0.0 : dlp_dzhi)) * (-inv_sM);
    res.d_dprime = d_muR;

    // d_logP / d_mu_F:
    // Only through mu_M: (dlp_dzlo + dlp_dzhi) * (-1/sigma_M)
    res.d_dprime2 = (dlp_dzlo + (is_hi_edge ? 0.0 : dlp_dzhi)) * (-inv_sM);

    // d_logP / d_sigma_R (on exp output scale):
    // Through z_cR: dlp_dzcR * (-z_cR/sigma_R)
    // Through z_lo, z_hi (via sigma_M): dlp_dz * (-z/sigma_M) * (sigma_R/sigma_M)
    // Through rho: dlp_drho * d_rho/d_sigma_R
    //   rho = sigma_R / sigma_M, d_rho/d_sigma_R = 1/sigma_M - sigma_R * (sigma_R/sigma_M) / sigma_M^2
    //   = 1/sigma_M - sigma_R^2 / sigma_M^3 = (sigma_M^2 - sigma_R^2) / sigma_M^3
    //   = sigma_F^2 / sigma_M^3
    double dsR = dlp_dzcR * (-z_cR * inv_sR);
    double sR_over_sM = sigma_R * inv_sM;
    dsR += dlp_dzlo * (-z_lo * inv_sM) * sR_over_sM;
    if (!is_hi_edge) dsR += dlp_dzhi * (-z_hi * inv_sM) * sR_over_sM;
    double drho_dsR = sigma_F * sigma_F / (sigma_M * sigma_M * sigma_M);
    dsR += dlp_drho * drho_dsR;
    // Chain through exp: d/d(sigma_n) = dsR * sigma_R
    res.d_sigma = dsR * sigma_R;

    // d_logP / d_sigma_F (on exp output scale):
    // Through sigma_M: dlp_dz * (-z/sigma_M) * (sigma_F/sigma_M)
    // Through rho: dlp_drho * d_rho/d_sigma_F
    //   d_rho/d_sigma_F = -sigma_R * sigma_F / sigma_M^3
    double dsF = 0.0;
    double sF_over_sM = sigma_F * inv_sM;
    dsF += dlp_dzlo * (-z_lo * inv_sM) * sF_over_sM;
    if (!is_hi_edge) dsF += dlp_dzhi * (-z_hi * inv_sM) * sF_over_sM;
    double drho_dsF = -sigma_R * sigma_F / (sigma_M * sigma_M * sigma_M);
    dsF += dlp_drho * drho_dsF;
    res.d_sigma2 = dsF * sigma_F;
  } else {
    // Lure: no gradients for dprime, dprime2, sigma, sigma2 (parameters are fixed)
    // But d_rec_crit still propagates (c_R is shared)
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
};
inline GCdpResult G_cdp_grad_reuse(
    double c, double mu_R, double sigma_R, double mu_F, double sigma_F,
    double c_R, double c_K, const BinormalCdfResult& bv_cR);


// CDP R/K/G result (n_rkg == 3) — adds know_crit gradient
struct CdpRkgResult {
  double lp;
  double d_dprime;
  double d_dprime2;
  double d_sigma;
  double d_sigma2;
  double d_rec_crit;
  double d_know_crit;
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
// where c_K is a criterion on F (not a second criterion on R). Previously
// this function used c_K as a lower criterion on R, giving a materially
// different model and disagreeing with JAX on n_rkg=3 fits.
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
    double c_R, double c_K,
    const double* tau, int J,
    const int* old_level_map) {

  CdpRkgResult res = {};
  res.k_lo = -1;
  res.k_hi = -1;

  // Determine if "old" response
  int old_idx = -1;
  for (int j = 0; j < J; ++j) {
    if (yn == old_level_map[j]) {
      old_idx = j;
      break;
    }
  }
  bool is_old_resp = (old_idx >= 0);

  // Effective parameters. CDP fixes lure reference to N(0, 1) on both
  // dimensions — mu_R=0, sigma_R=1, mu_F=0, sigma_F=1 when is_old == 0.
  double mu_R_eff, sigma_R_eff, mu_F_eff, sigma_F_eff;
  if (is_old) {
    mu_R_eff = mu_R; sigma_R_eff = sigma_R;
    mu_F_eff = mu_F; sigma_F_eff = sigma_F;
  } else {
    mu_R_eff = 0.0; sigma_R_eff = 1.0;
    mu_F_eff = 0.0; sigma_F_eff = 1.0;
  }

  double mu_M = mu_R_eff + mu_F_eff;
  double sigma_M = std::sqrt(sigma_R_eff * sigma_R_eff + sigma_F_eff * sigma_F_eff);
  double rho_val = sigma_R_eff / sigma_M;
  double inv_sR = 1.0 / sigma_R_eff;
  double inv_sM = 1.0 / sigma_M;
  double z_cR = (c_R - mu_R_eff) * inv_sR;

  if (!is_old_resp) {
    // "New" response: P = Phi((tau[0] - mu_M) / sigma_M). Unchanged from
    // the previous implementation — this path was already correct.
    double z_tau1 = (tau[0] - mu_M) * inv_sM;
    double p = Phi(z_tau1);
    if (p < 1e-20) p = 1e-20;
    res.lp = std::log(p);
    double d_lp_dz = phi(z_tau1) / p;
    res.k_hi = 0;
    res.d_thresh_hi = d_lp_dz * inv_sM;

    if (is_old) {
      double d_muM = d_lp_dz * (-inv_sM);
      res.d_dprime = d_muM;
      res.d_dprime2 = d_muM;
      double d_sigM = d_lp_dz * (-z_tau1 * inv_sM);
      res.d_sigma = d_sigM * (sigma_R / sigma_M) * sigma_R;
      res.d_sigma2 = d_sigM * (sigma_F / sigma_M) * sigma_F;
    }
    return res;
  }

  // "Old" response: use the same Remember + G(c) decomposition.
  double tau_lo_val = tau[old_idx];
  double tau_hi_val = (old_idx < J - 1) ? tau[old_idx + 1] : 20.0;
  double z_lo = (tau_lo_val - mu_M) * inv_sM;
  double z_hi = (tau_hi_val - mu_M) * inv_sM;
  bool is_hi_edge = (old_idx == J - 1);

  res.k_lo = old_idx;
  if (!is_hi_edge) res.k_hi = old_idx + 1;

  // REMEMBER: shared binormal CDF values (reused by G_cdp_grad_reuse)
  BinormalCdfResult bv_hi = binormal_cdf_grad(z_cR, z_hi, rho_val);
  BinormalCdfResult bv_lo = binormal_cdf_grad(z_cR, z_lo, rho_val);
  double p_band = Phi(z_hi) - Phi(z_lo);
  double p_R_raw = (Phi(z_hi) - bv_hi.val) - (Phi(z_lo) - bv_lo.val);

  // GUESS: G(tau_hi) - G(tau_lo) with c_K acting on F (Wixted/Mickes)
  GCdpResult Ghi = G_cdp_grad_reuse(tau_hi_val, mu_R_eff, sigma_R_eff,
                                      mu_F_eff, sigma_F_eff, c_R, c_K, bv_hi);
  GCdpResult Glo = G_cdp_grad_reuse(tau_lo_val, mu_R_eff, sigma_R_eff,
                                      mu_F_eff, sigma_F_eff, c_R, c_K, bv_lo);
  double p_G_raw = Ghi.val - Glo.val;

  // KNOW: remainder
  double p_K_raw = p_band - p_R_raw - p_G_raw;

  // Soft clamp (same as fused JAX path)
  double kk = 10000.0;
  double p_R = stable_softplus(kk * p_R_raw) / kk;
  double p_K = stable_softplus(kk * p_K_raw) / kk;
  double p_G = stable_softplus(kk * p_G_raw) / kk;

  double prob;
  if (rk_n == 1) prob = p_R;
  else if (rk_n == 2) prob = p_K;
  else prob = p_G;

  if (prob < 1e-20) prob = 1e-20;
  res.lp = std::log(prob);

  double inv_prob = 1.0 / prob;
  double sig_R = 1.0 / (1.0 + std::exp(-kk * p_R_raw));
  double sig_K = 1.0 / (1.0 + std::exp(-kk * p_K_raw));
  double sig_G = 1.0 / (1.0 + std::exp(-kk * p_G_raw));

  double dlp_dpR = 0.0, dlp_dpK = 0.0, dlp_dpG = 0.0;
  if (rk_n == 1) dlp_dpR = sig_R * inv_prob;
  else if (rk_n == 2) dlp_dpK = sig_K * inv_prob;
  else dlp_dpG = sig_G * inv_prob;

  // K = band - R - G, so every param's K-contribution is -(R + G) contribution,
  // plus the band derivative where R/G don't reach (e.g. through phi(z_hi/lo)).
  double eff_dpR = dlp_dpR - dlp_dpK;
  double eff_dpG = dlp_dpG - dlp_dpK;
  double eff_dpBand = dlp_dpK;

  // Remember + band gradients (no c_K dependence through this path)
  double dlp_dzcR = eff_dpR * (-bv_hi.d_z1 + bv_lo.d_z1);
  double dlp_dzhi = eff_dpR * (phi(z_hi) - bv_hi.d_z2) + eff_dpBand * phi(z_hi);
  double dlp_dzlo = eff_dpR * (-(phi(z_lo) - bv_lo.d_z2)) + eff_dpBand * (-phi(z_lo));
  double dlp_drho = eff_dpR * (-bv_hi.d_rho + bv_lo.d_rho);

  // Guess gradients (c_K enters here, as a criterion on F)
  double dG_dmuR  = Ghi.d_mu_R  - Glo.d_mu_R;
  double dG_dsigR = Ghi.d_sigma_R - Glo.d_sigma_R;
  double dG_dmuF  = Ghi.d_mu_F  - Glo.d_mu_F;
  double dG_dsigF = Ghi.d_sigma_F - Glo.d_sigma_F;
  double dG_dcR   = Ghi.d_c_R  - Glo.d_c_R;
  double dG_dcK   = Ghi.d_c_K  - Glo.d_c_K;
  double dG_dtau_hi = Ghi.d_c;
  double dG_dtau_lo = Glo.d_c;

  // Chain to primitive parameters. dprime/dprime2/sigma/sigma2 are only
  // free params for old items; for lures mu/sigma are fixed constants so
  // their gradients must be zero (guarded explicitly below).
  if (is_old) {
    double sR_over_sM = sigma_R_eff * inv_sM;

    // dprime (mu_R): through z_cR, through mu_M (z_lo/z_hi), and through G
    res.d_dprime  = dlp_dzcR * (-inv_sR)
                  + (dlp_dzlo + dlp_dzhi) * (-inv_sM)
                  + eff_dpG * dG_dmuR;

    // dprime2 (mu_F): through mu_M and through G
    res.d_dprime2 = (dlp_dzlo + dlp_dzhi) * (-inv_sM)
                  + eff_dpG * dG_dmuF;

    // sigma_R: through z_cR, through sigma_M (z_lo/z_hi and rho), and G
    double drho_dsigR = (sigma_F_eff * sigma_F_eff) / (sigma_M * sigma_M * sigma_M);
    double d_sigR_raw = dlp_dzcR * (-z_cR * inv_sR)
                      + (dlp_dzhi * (-z_hi * inv_sM) + dlp_dzlo * (-z_lo * inv_sM))
                        * sR_over_sM
                      + dlp_drho * drho_dsigR
                      + eff_dpG * dG_dsigR;
    // Map to exp-link parameter (sigma = exp(beta_sigma))
    res.d_sigma = d_sigR_raw * sigma_R_eff;

    // sigma_F: same shape, through sigma_M and G
    double sF_over_sM = sigma_F_eff * inv_sM;
    double drho_dsigF = -sigma_R_eff * sigma_F_eff / (sigma_M * sigma_M * sigma_M);
    double d_sigF_raw = (dlp_dzhi * (-z_hi * inv_sM) + dlp_dzlo * (-z_lo * inv_sM))
                          * sF_over_sM
                      + dlp_drho * drho_dsigF
                      + eff_dpG * dG_dsigF;
    res.d_sigma2 = d_sigF_raw * sigma_F_eff;
  }
  // rec_crit and know_crit affect ALL items (both old and lure) because the
  // criteria themselves don't depend on is_old.
  res.d_rec_crit  = dlp_dzcR * inv_sR + eff_dpG * dG_dcR;
  res.d_know_crit = eff_dpG * dG_dcK;

  // Threshold gradients
  res.d_thresh_lo = dlp_dzlo * inv_sM + eff_dpG * (-dG_dtau_lo);
  res.d_thresh_hi = is_hi_edge ? 0.0 :
                    (dlp_dzhi * inv_sM + eff_dpG * dG_dtau_hi);

  return res;
}


// G_cdp gradient computation reusing pre-computed Phi2(z_cR, z_c, rho).
// In cdp_rkg_cell, bv_cR (= binormal_cdf_grad(z_cR, z_c, rho)) is already
// computed for the Remember term. Passing it here eliminates up to 2 redundant
// Owen's-T evaluations per threshold boundary (4 total for hi+lo).
inline GCdpResult G_cdp_grad_reuse(
    double c, double mu_R, double sigma_R, double mu_F, double sigma_F,
    double c_R, double c_K,
    const BinormalCdfResult& bv_cR_pre) {

  double inv_sR = 1.0 / sigma_R;
  double inv_sF = 1.0 / sigma_F;
  double z_cR = (c_R - mu_R) * inv_sR;
  double z_cK_F = (c_K - mu_F) * inv_sF;
  double mu_M = mu_R + mu_F;
  double sigma_M = std::sqrt(sigma_R * sigma_R + sigma_F * sigma_F);
  double inv_sM = 1.0 / sigma_M;
  double z_c = (c - mu_M) * inv_sM;
  double rho = sigma_R * inv_sM;

  bool case1 = (c_R <= c - c_K);
  GCdpResult res = {};

  if (case1) {
    double Phi_zcR = Phi(z_cR);
    double Phi_zcKF = Phi(z_cK_F);
    double phi_zcR = phi(z_cR);
    double phi_zcKF = phi(z_cK_F);
    res.val = Phi_zcR * Phi_zcKF;
    res.d_c = 0.0;
    res.d_c_R = phi_zcR * inv_sR * Phi_zcKF;
    res.d_c_K = Phi_zcR * phi_zcKF * inv_sF;
    res.d_mu_R = -phi_zcR * inv_sR * Phi_zcKF;
    res.d_mu_F = -Phi_zcR * phi_zcKF * inv_sF;
    res.d_sigma_R = phi_zcR * (-z_cR * inv_sR) * Phi_zcKF;
    res.d_sigma_F = Phi_zcR * phi_zcKF * (-z_cK_F * inv_sF);
  } else {
    double s = c - c_K;
    double z_s = (s - mu_R) * inv_sR;
    double Phi_zs = Phi(z_s);
    double phi_zs = phi(z_s);
    double Phi_zcKF = Phi(z_cK_F);
    double phi_zcKF = phi(z_cK_F);

    // Reuse bv1 = bv_cR_pre (avoids redundant binormal_cdf_grad call)
    const auto& bv1 = bv_cR_pre;
    auto bv2 = binormal_cdf_grad(z_s, z_c, rho);

    res.val = Phi_zs * Phi_zcKF + bv1.val - bv2.val;

    res.d_c = phi_zs * inv_sR * Phi_zcKF
            + bv1.d_z2 * inv_sM
            - bv2.d_z1 * inv_sR - bv2.d_z2 * inv_sM;

    res.d_c_R = bv1.d_z1 * inv_sR;

    res.d_c_K = phi_zs * (-inv_sR) * Phi_zcKF
              + Phi_zs * phi_zcKF * inv_sF
              - bv2.d_z1 * (-inv_sR);

    res.d_mu_R = phi_zs * (-inv_sR) * Phi_zcKF
               + bv1.d_z1 * (-inv_sR) + bv1.d_z2 * (-inv_sM)
               - bv2.d_z1 * (-inv_sR) - bv2.d_z2 * (-inv_sM);

    res.d_mu_F = Phi_zs * phi_zcKF * (-inv_sF)
               + bv1.d_z2 * (-inv_sM) - bv2.d_z2 * (-inv_sM);

    double sR_over_sM = sigma_R * inv_sM;
    double drho_dsR = (sigma_F * sigma_F) / (sigma_M * sigma_M * sigma_M);
    res.d_sigma_R = phi_zs * (-z_s * inv_sR) * Phi_zcKF
                  + bv1.d_z1 * (-z_cR * inv_sR)
                  + bv1.d_z2 * (-z_c * inv_sM) * sR_over_sM
                  + bv1.d_rho * drho_dsR
                  - bv2.d_z1 * (-z_s * inv_sR)
                  - bv2.d_z2 * (-z_c * inv_sM) * sR_over_sM
                  - bv2.d_rho * drho_dsR;

    double sF_over_sM = sigma_F * inv_sM;
    double drho_dsF = -sigma_R * sigma_F / (sigma_M * sigma_M * sigma_M);
    res.d_sigma_F = Phi_zs * phi_zcKF * (-z_cK_F * inv_sF)
                  + bv1.d_z2 * (-z_c * inv_sM) * sF_over_sM
                  + bv1.d_rho * drho_dsF
                  - bv2.d_z2 * (-z_c * inv_sM) * sF_over_sM
                  - bv2.d_rho * drho_dsF;
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

  double p_det_below = Phi(z_cp) - Phi(z_ylo_b);

  // Source probability in below-cp region
  double p_src_below = 0.0;
  double zs_lo_b = 0.0, zs_hi_b = 0.0;
  if (is_lo2) {
    zs_hi_b = thresh2[0] * inv_sc;
    p_src_below = Phi(zs_hi_b);
  } else if (is_hi2) {
    zs_lo_b = thresh2[n_thresh2 - 1] * inv_sc;
    p_src_below = 1.0 - Phi(zs_lo_b);
  } else {
    zs_lo_b = thresh2[y2 - 2] * inv_sc;
    zs_hi_b = thresh2[y2 - 1] * inv_sc;
    p_src_below = Phi(zs_hi_b) - Phi(zs_lo_b);
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

  if (P < 1e-20) {
    res.lp = std::log(1e-20);
    return res;
  }

  res.lp = std::log(P);
  double inv_P = 1.0 / P;

  // --- Gradient computation ---
  // d_logP/d_thresh1 (detection thresholds)
  // From below-cp: d_P/d_z_ylo_b = -phi(z_ylo_b) * p_src_below
  // d_z_ylo_b / d_thresh1_lo = inv_s1
  if (!is_lo1) {
    d_t1_lo = (-phi(z_ylo_b) * p_src_below * inv_s1 + prob_above * bc_above.d_z1_lo * 0.0) * inv_P;
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
  // Below: d/d_mu1 of p_det_below * p_src_below
  //   = (-phi(z_cp)*(-inv_s1) + phi(z_ylo_b)*inv_s1) * p_src_below  [through z's]
  //   BUT z_cp depends on mu1 through cp = mu1 - sigma1*mu2/(rho*sigma2)
  //   d_z_cp/d_mu1 = (d_cp/d_mu1 - 1) * inv_s1 = (1 - 1)*inv_s1 = 0
  //   Wait: d_cp/d_mu1 = 1, so d_z_cp/d_mu1 = (1 - 1) / sigma1 = 0
  //   Actually z_cp = (cp - mu1)/sigma1 = (mu1 - sigma1*mu2/(rho*sigma2) - mu1)/sigma1
  //                 = -mu2/(rho*sigma2), which does NOT depend on mu1!
  //   So d(p_det_below)/d(mu1) = phi(z_ylo_b) * inv_s1 * p_src_below  (only through z_ylo_b)
  //   BUT z_ylo_b = (thresh1_lo - mu1)/sigma1, so d_z_ylo_b/d_mu1 = -inv_s1
  //   => d(p_det_below)/d(mu1) = phi(z_ylo_b) * inv_s1  (from -d/d(mu1) of Phi(z_ylo_b))
  //   Hmm, p_det_below = Phi(z_cp) - Phi(z_ylo_b)
  //   d(p_det_below)/d(mu1) = 0 - phi(z_ylo_b)*(-inv_s1) = phi(z_ylo_b)*inv_s1 (if not lo edge)
  double d_Pbelow_dmu1 = 0.0;
  if (!is_lo1) {
    d_Pbelow_dmu1 = phi(z_ylo_b) * inv_s1 * p_src_below;
  }
  // Above: d_prob_above/d_mu1 through z_yhi_a (=-(bc_above.d_z1_hi)*inv_s1 if not hi edge)
  //   and through z_cp (d_z_cp/d_mu1 = 0 as shown above, so z_cp contribution = 0 for mu1)
  //   Wait: z_cp = -mu2/(rho*sigma2) does NOT depend on mu1, but the upper bound z_yhi depends on mu1
  //   d_prob_above/d_mu1 = prob_above * (bc_above.d_z1_lo * 0 + bc_above.d_z1_hi * (-inv_s1))
  //                       + prob_above * (bc_above.d_z2_lo * (-inv_s2) * 0 + ...)
  //   Hmm, this is getting complex. Let me use a simpler approach.
  //
  // Actually, the clean approach is: for above-cp, we have a bivariate cell with
  // z1_lo = z_cp (constant w.r.t. mu1!), z1_hi = (thresh1_hi - mu1)/sigma1
  // So d_prob_above/d_mu1 = prob_above * bc_above.d_z1_hi * (-inv_s1)  [from z1_hi]
  //                        + prob_above * (bc_above.d_z2_lo + bc_above.d_z2_hi) * (-inv_s2) * 0
  //   (z2 depends on mu2, not mu1)
  // For the above region z2 thresholds don't depend on mu1, so:
  double d_Pabove_dmu1 = 0.0;
  if (!is_hi1) {
    d_Pabove_dmu1 = prob_above * bc_above.d_z1_hi * (-inv_s1);
  }
  d_mu1 = (d_Pbelow_dmu1 + d_Pabove_dmu1) * inv_P;

  // d_logP/d_mu2 (discrim)
  // Below: p_src_below doesn't depend on mu2 (source ~ N(0, sigma_cond))
  //   BUT z_cp = -mu2/(rho*sigma2), so p_det_below depends on mu2 through z_cp
  //   d_z_cp/d_mu2 = -1/(rho*sigma2) * 1/sigma1... wait
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
    p_src_below = 1.0 - Phi(zs_lo_v);
    phi_zs_lo = phi(zs_lo_v);
  } else {
    zs_lo_v = thresh2[y2 - 2] * inv_sc;
    zs_hi_v = thresh2[y2 - 1] * inv_sc;
    p_src_below = Phi(zs_hi_v) - Phi(zs_lo_v);
    phi_zs_lo = phi(zs_lo_v);
    phi_zs_hi = phi(zs_hi_v);
  }

  double p_below = p_below_det * p_src_below;

  // Above-cp: standard BVN marginal
  // prob_above = Phi(z2_hi) - Phi(z2_lo) - Phi2(z_cp, z2_hi, rho) + Phi2(z_cp, z2_lo, rho)
  double z2_lo_a = is_lo2 ? -8.0 : (thresh2[y2 - 2] - mu2) * inv_s2;
  double z2_hi_a = is_hi2 ?  8.0 : (thresh2[y2 - 1] - mu2) * inv_s2;

  // Compute binormal_cdf with gradients for all cases
  // r_lo: Phi2(z_cp, z2_lo_a, rho), r_hi: Phi2(z_cp, z2_hi_a, rho)
  BinormalCdfResult r_lo = {}, r_hi = {};
  double prob_above = 0.0;
  double phi_z2_lo = is_lo2 ? 0.0 : phi(z2_lo_a);
  double phi_z2_hi = is_hi2 ? 0.0 : phi(z2_hi_a);

  if (is_lo2) {
    r_hi = binormal_cdf_grad(z_cp, z2_hi_a, rho);
    prob_above = Phi(z2_hi_a) - r_hi.val;
  } else if (is_hi2) {
    r_lo = binormal_cdf_grad(z_cp, z2_lo_a, rho);
    prob_above = 1.0 - Phi(z2_lo_a) - Phi(z_cp) + r_lo.val;
  } else {
    r_hi = binormal_cdf_grad(z_cp, z2_hi_a, rho);
    r_lo = binormal_cdf_grad(z_cp, z2_lo_a, rho);
    prob_above = Phi(z2_hi_a) - Phi(z2_lo_a) - r_hi.val + r_lo.val;
  }

  res.p = p_below + prob_above;

  // --- Gradient computation ---
  // d_mu2: through z_cp (below det + above z1_lo) and z2 (above)
  // dz2/dmu2 = -inv_s2
  double dp_below_dmu2 = phi_zcp * dz_cp_dmu2 * p_src_below;
  double dp_above_dmu2 = 0.0;
  if (is_lo2) {
    // prob_above = Phi(z2_hi) - Phi2(z_cp, z2_hi, rho)
    dp_above_dmu2 = phi_z2_hi * (-inv_s2)
                  - (r_hi.d_z1 * dz_cp_dmu2 + r_hi.d_z2 * (-inv_s2));
  } else if (is_hi2) {
    // prob_above = 1 - Phi(z2_lo) - Phi(z_cp) + Phi2(z_cp, z2_lo, rho)
    dp_above_dmu2 = -phi_z2_lo * (-inv_s2) - phi_zcp * dz_cp_dmu2
                  + (r_lo.d_z1 * dz_cp_dmu2 + r_lo.d_z2 * (-inv_s2));
  } else {
    // prob_above = Phi(z2_hi) - Phi(z2_lo) - Phi2(z_cp, z2_hi, rho) + Phi2(z_cp, z2_lo, rho)
    dp_above_dmu2 = phi_z2_hi * (-inv_s2) - phi_z2_lo * (-inv_s2)
                  - (r_hi.d_z1 * dz_cp_dmu2 + r_hi.d_z2 * (-inv_s2))
                  + (r_lo.d_z1 * dz_cp_dmu2 + r_lo.d_z2 * (-inv_s2));
  }
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
  double dp_above_ds2 = 0.0;
  if (is_lo2) {
    dp_above_ds2 = phi_z2_hi * (-z2_hi_a * inv_s2)
                 - (r_hi.d_z1 * dz_cp_ds2 + r_hi.d_z2 * (-z2_hi_a * inv_s2));
  } else if (is_hi2) {
    dp_above_ds2 = -phi_z2_lo * (-z2_lo_a * inv_s2) - phi_zcp * dz_cp_ds2
                 + (r_lo.d_z1 * dz_cp_ds2 + r_lo.d_z2 * (-z2_lo_a * inv_s2));
  } else {
    dp_above_ds2 = phi_z2_hi * (-z2_hi_a * inv_s2) - phi_z2_lo * (-z2_lo_a * inv_s2)
                 - (r_hi.d_z1 * dz_cp_ds2 + r_hi.d_z2 * (-z2_hi_a * inv_s2))
                 + (r_lo.d_z1 * dz_cp_ds2 + r_lo.d_z2 * (-z2_lo_a * inv_s2));
  }
  res.d_sigma2 = dp_below_ds2_det + dp_below_ds2_src + dp_above_ds2;

  // d_thresh2_lo (when not lo edge): through zs_lo (below) and z2_lo (above)
  // dzs_lo/dthresh2_lo = inv_sc, dz2_lo/dthresh2_lo = inv_s2
  if (!is_lo2) {
    double dp_below_dt2lo = p_below_det * (-phi_zs_lo * inv_sc);
    double dp_above_dt2lo = 0.0;
    if (is_hi2) {
      dp_above_dt2lo = -phi_z2_lo * inv_s2 + r_lo.d_z2 * inv_s2;
    } else {
      dp_above_dt2lo = -phi_z2_lo * inv_s2 + r_lo.d_z2 * inv_s2;
    }
    res.d_thresh2_lo = dp_below_dt2lo + dp_above_dt2lo;
  }

  // d_thresh2_hi (when not hi edge): through zs_hi (below) and z2_hi (above)
  // dzs_hi/dthresh2_hi = inv_sc, dz2_hi/dthresh2_hi = inv_s2
  if (!is_hi2) {
    double dp_below_dt2hi = p_below_det * phi_zs_hi * inv_sc;
    double dp_above_dt2hi = 0.0;
    if (is_lo2) {
      dp_above_dt2hi = phi_z2_hi * inv_s2 - r_hi.d_z2 * inv_s2;
    } else {
      dp_above_dt2hi = phi_z2_hi * inv_s2 - r_hi.d_z2 * inv_s2;
    }
    res.d_thresh2_hi = dp_below_dt2hi + dp_above_dt2hi;
  }

  // d_rho: the marginal source depends on rho through z_cp = -mu2/(rho*sigma2),
  // sigma_cond = sigma2*sqrt(1-rho^2) (so dzs/drho = zs * rho/(1-rho^2)), and the
  // binormal_cdf terms (direct r.d_rho + chain through z_cp). Dropping this term
  // halves/biases the rho gradient for bounded DP (item-recollected component).
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
  double dp_above_drho = 0.0;
  if (is_lo2) {
    // prob_above = Phi(z2_hi) - Phi2(z_cp, z2_hi, rho); z2_hi indep of rho
    dp_above_drho = -(r_hi.d_z1 * dz_cp_drho + r_hi.d_rho);
  } else if (is_hi2) {
    // prob_above = 1 - Phi(z2_lo) - Phi(z_cp) + Phi2(z_cp, z2_lo, rho)
    dp_above_drho = -phi_zcp * dz_cp_drho + (r_lo.d_z1 * dz_cp_drho + r_lo.d_rho);
  } else {
    dp_above_drho = -(r_hi.d_z1 * dz_cp_drho + r_hi.d_rho)
                  + (r_lo.d_z1 * dz_cp_drho + r_lo.d_rho);
  }
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
  // Recollection success cells (new convention): Source A items at the low
  // end of the source axis (y2 = 1, "sure A"); Source B items at the high
  // end (y2 = K2, "sure B").
  bool is_corner = (item_type == 2 && y2 == 1) || (item_type == 3 && y2 == K2);
  if (is_corner) p3 = R_I_eff * R_S_eff;

  double P = p1 + p2 + p3;
  if (P < 1e-20) {
    res.lp = std::log(1e-20);
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
