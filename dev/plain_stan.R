# =============================================================================
# Stan-language likelihood: development-only verification path (not shipped)
# =============================================================================
# env <- new.env(parent = asNamespace("bayesroc")); sys.source("dev/plain_stan.R", env)
# options(bayesroc.dev_plain_stan = env) then builds models whose Stan program evaluates
# the likelihood in the Stan language instead of the batch C++ kernel
# (tests do this via .broc_plain() in helper-parity.R).


#' Stan functions: tail-accurate bivariate-normal rectangles (mirrors bvn_tail.h)
#' @noRd
.stan_bvn_functions <- function() {
  '
  // Tail-accurate bivariate normal rectangle probabilities (mirrors bvn_tail.h)
  array[] real bvn_gl_x(int m) {
    if (m == 8) return {-0.9602898564975362, -0.7966664774136267, -0.525532409916329, -0.18343464249564978, 0.18343464249564978, 0.525532409916329, 0.7966664774136267, 0.9602898564975362};
    if (m == 10) return {-0.9739065285171717, -0.8650633666889845, -0.6794095682990244, -0.4333953941292472, -0.14887433898163122, 0.14887433898163122, 0.4333953941292472, 0.6794095682990244, 0.8650633666889845, 0.9739065285171717};
    if (m == 12) return {-0.9815606342467192, -0.9041172563704748, -0.7699026741943047, -0.5873179542866175, -0.3678314989981802, -0.1252334085114689, 0.1252334085114689, 0.3678314989981802, 0.5873179542866175, 0.7699026741943047, 0.9041172563704748, 0.9815606342467192};
    if (m == 14) return {-0.9862838086968124, -0.9284348836635735, -0.827201315069765, -0.6872929048116855, -0.5152486363581541, -0.31911236892788974, -0.10805494870734367, 0.10805494870734367, 0.31911236892788974, 0.5152486363581541, 0.6872929048116855, 0.827201315069765, 0.9284348836635735, 0.9862838086968124};
    if (m == 16) return {-0.9894009349916499, -0.9445750230732326, -0.8656312023878318, -0.755404408355003, -0.6178762444026438, -0.45801677765722737, -0.2816035507792589, -0.09501250983763744, 0.09501250983763744, 0.2816035507792589, 0.45801677765722737, 0.6178762444026438, 0.755404408355003, 0.8656312023878318, 0.9445750230732326, 0.9894009349916499};
    if (m == 20) return {-0.993128599185095, -0.9639719272779138, -0.912234428251326, -0.8391169718222188, -0.7463319064601508, -0.636053680726515, -0.5108670019508271, -0.37370608871541955, -0.22778585114164507, -0.07652652113349734, 0.07652652113349734, 0.22778585114164507, 0.37370608871541955, 0.5108670019508271, 0.636053680726515, 0.7463319064601508, 0.8391169718222188, 0.912234428251326, 0.9639719272779138, 0.993128599185095};
    return {0.0};
  }

  array[] real bvn_gl_w(int m) {
    if (m == 8) return {0.10122853629037706, 0.22238103445337443, 0.3137066458778869, 0.36268378337836166, 0.36268378337836166, 0.3137066458778869, 0.22238103445337443, 0.10122853629037706};
    if (m == 10) return {0.06667134430868814, 0.1494513491505804, 0.219086362515982, 0.2692667193099965, 0.2955242247147528, 0.2955242247147528, 0.2692667193099965, 0.219086362515982, 0.1494513491505804, 0.06667134430868814};
    if (m == 12) return {0.04717533638651141, 0.10693932599531907, 0.16007832854334642, 0.20316742672306573, 0.2334925365383546, 0.2491470458134027, 0.2491470458134027, 0.2334925365383546, 0.20316742672306573, 0.16007832854334642, 0.10693932599531907, 0.04717533638651141};
    if (m == 14) return {0.03511946033175175, 0.08015808715975999, 0.12151857068790331, 0.15720316715819363, 0.18553839747793788, 0.20519846372129572, 0.21526385346315793, 0.21526385346315793, 0.20519846372129572, 0.18553839747793788, 0.15720316715819363, 0.12151857068790331, 0.08015808715975999, 0.03511946033175175};
    if (m == 16) return {0.027152459411754176, 0.062253523938647456, 0.0951585116824926, 0.12462897125553407, 0.1495959888165767, 0.16915651939500265, 0.18260341504492364, 0.18945061045506864, 0.18945061045506864, 0.18260341504492364, 0.16915651939500265, 0.1495959888165767, 0.12462897125553407, 0.0951585116824926, 0.062253523938647456, 0.027152459411754176};
    if (m == 20) return {0.017614007139150893, 0.040601429800386446, 0.06267204833410879, 0.08327674157670471, 0.1019301198172407, 0.1181945319615186, 0.1316886384491769, 0.1420961093183824, 0.14917298647260424, 0.15275338713072628, 0.15275338713072628, 0.14917298647260424, 0.1420961093183824, 0.1316886384491769, 0.1181945319615186, 0.1019301198172407, 0.08327674157670471, 0.06267204833410879, 0.040601429800386446, 0.017614007139150893};
    return {0.0};
  }

  real bvn_std(real c, real mu, real sd) {
    return is_inf(c) ? c : (c - mu) / sd;
  }

  real bvn_log_mills(real x) {
    if (x > 40) {
      real r = 1 / (x * x);
      return log1p(r * (-1 + r * (3 + r * (-15 + r * (105 - 945 * r))))) - log(x);
    }
    array[19] real co = {0.5893276044905619, 0.47231061768868227, 0.1485473376619552, 0.03611913148900286, 0.006378297104388951, 0.0006592194044005224, -1.1534346994167076e-05, -1.541999589462102e-05, -1.4350295963176838e-06, 2.671384604629693e-07, 5.842069382918476e-08, -4.878387335732758e-09, -2.0166300903829314e-09, 1.2065191939381096e-10, 7.135408878679637e-11, -4.885139134318498e-12, -2.630066125081603e-12, 2.7380131795870467e-13, 9.691376820174753e-14};
    real t = 1 / (1 + 0.2316419 * x);
    real u = (2 * t - 1.015) / 0.985;
    real b1 = 0;
    real b2 = 0;
    for (j in 1:18) {
      real tt = 2 * u * b1 - b2 + co[20 - j];
      b2 = b1;
      b1 = tt;
    }
    return log(t * (u * b1 - b2 + co[1]));
  }

  real bvn_log_interval(real lo, real hi) {
    if (hi <= lo) return negative_infinity();
    if (lo >= 0) {
      real w = exp(bvn_log_mills(lo));
      if (!is_inf(hi)) w -= exp(0.5 * (lo - hi) * (lo + hi) + bvn_log_mills(hi));
      return -0.5 * lo * lo - 0.9189385332046727 + log(w);
    }
    if (hi <= 0) {
      real w = exp(bvn_log_mills(-hi));
      if (!is_inf(lo)) w -= exp(0.5 * (hi - lo) * (hi + lo) + bvn_log_mills(-lo));
      return -0.5 * hi * hi - 0.9189385332046727 + log(w);
    }
    return log(0.5 * (erf(hi * 0.7071067811865476) - erf(lo * 0.7071067811865476)));
  }

  real bvn_dw_corner(real h, real k, real A, array[] real xs, array[] real ws) {
    real a = 0.5 * (h * h + k * k);
    real b = h * k;
    real S = 0;
    for (j in 1:size(xs)) {
      real s = sin(A * 0.5 * (xs[j] + 1));
      S += ws[j] * exp((b * s - a) / (1 - s * s));
    }
    return A * 0.07957747154594767 * S;
  }

  vector bvn_tail_eval(real x, real a2, real b2, real rho, real s, real k) {
    real l = bvn_std(a2, rho * x, s);
    real u = bvn_std(b2, rho * x, s);
    real ld = bvn_log_interval(l, u);
    real Al = is_inf(l) ? 0 : exp(-0.5 * l * l - 0.9189385332046727 - ld);
    real Au = is_inf(u) ? 0 : exp(-0.5 * u * u - 0.9189385332046727 - ld);
    real R = Al - Au;
    real lA = is_inf(l) ? 0 : l * Al;
    real uA = is_inf(u) ? 0 : u * Au;
    return [-0.5 * x * x - 0.9189385332046727 + ld, -x + k * R, -1 + k * k * (lA - uA - R * R)]\';
  }

  real bvn_log_rect_tail(real a1, real b1, real a2, real b2, real rho_in) {
    real rho = fmin(fmax(rho_in, -0.999999), 0.999999);
    real s = sqrt((1 - rho) * (1 + rho));
    real k = rho / s;
    real lo = fmax(a1, -40);
    real hi = fmin(b1, 40);
    if (hi <= lo) return negative_infinity();
    vector[3] e = bvn_tail_eval(lo, a2, b2, rho, s, k);
    real m = lo;
    if (e[2] > 0) {
      vector[3] eh = bvn_tail_eval(hi, a2, b2, rho, s, k);
      if (eh[2] >= 0) {
        m = hi;
        e = eh;
      } else {
        real L = lo;
        real H = hi;
        real x = 0.5 * (lo + hi);
        e = bvn_tail_eval(x, a2, b2, rho, s, k);
        for (it in 1:60) {
          if (e[2] > 0) L = x; else H = x;
          real nx = x - e[2] / e[3];
          if (!(nx > L && nx < H)) nx = 0.5 * (L + H);
          int done = abs(nx - x) < 1e-12 * (1 + abs(x)) || H - L < 1e-12 * (1 + abs(x));
          x = nx;
          e = bvn_tail_eval(x, a2, b2, rho, s, k);
          if (done) break;
        }
        m = x;
      }
    }
    real gm = e[1];
    if (is_inf(gm) || is_nan(gm)) return negative_infinity();
    real rs = (m == lo || m == hi) ? abs(e[2]) : 0;
    real cv = fmax(-e[3], 1);
    array[2] real ends;
    for (side in 1:2) {
      real dir = side == 1 ? -1 : 1;
      real edge = side == 1 ? lo : hi;
      if ((edge - m) * dir <= 0) {
        ends[side] = m;
      } else {
        real Tg = -rs + sqrt(rs * rs + 50);
        real T = (-rs + sqrt(rs * rs + 50 * cv)) / cv;
        for (it in 1:8) {
          if (T >= Tg) break;
          real xx = m + dir * T;
          if ((xx - edge) * dir >= 0) break;
          if (gm - bvn_tail_eval(xx, a2, b2, rho, s, k)[1] >= 25) break;
          T = fmin(2 * T, Tg);
        }
        real xe = m + dir * T;
        ends[side] = (xe - edge) * dir >= 0 ? edge : xe;
      }
    }
    array[20] real gx = bvn_gl_x(20);
    array[20] real gw = bvn_gl_w(20);
    vector[80] vals = rep_vector(negative_infinity(), 80);
    int nv = 0;
    for (sg in 1:2) {
      real A0 = sg == 1 ? ends[1] : m;
      real B0 = sg == 1 ? m : ends[2];
      if (B0 > A0) {
        for (j in 0:1) {
          real A = A0 + (B0 - A0) * j / 2.0;
          real B = A0 + (B0 - A0) * (j + 1) / 2.0;
          real h = 0.5 * (B - A);
          real c0 = 0.5 * (A + B);
          for (q in 1:20) {
            nv += 1;
            vals[nv] = log(h * gw[q]) + bvn_tail_eval(c0 + h * gx[q], a2, b2, rho, s, k)[1];
          }
        }
      }
    }
    if (nv == 0) return negative_infinity();
    return log_sum_exp(vals[1:nv]);
  }

  vector bvn_lin_eval(real x, real Al, real Bl, real Au, real Bu) {
    real l = is_inf(Al) ? Al : Al + Bl * x;
    real u = is_inf(Au) ? Au : Au + Bu * x;
    real ld = bvn_log_interval(l, u);
    if (is_inf(ld)) return [negative_infinity(), 0, -1]\';
    real fl = is_inf(l) ? 0 : exp(-0.5 * l * l - 0.9189385332046727 - ld);
    real fu = is_inf(u) ? 0 : exp(-0.5 * u * u - 0.9189385332046727 - ld);
    real lA = is_inf(l) ? 0 : l * fl;
    real uA = is_inf(u) ? 0 : u * fu;
    real q = Bu * fu - Bl * fl;
    return [-0.5 * x * x - 0.9189385332046727 + ld, -x + q, -1 - Bu * Bu * uA + Bl * Bl * lA - q * q]\';
  }

  real bvn_log_lin_integral(real Al, real Bl, real Au, real Bu, real a, real b) {
    real lo = fmax(a, -40);
    real hi = fmin(b, 40);
    if (hi <= lo) return negative_infinity();
    vector[3] e = bvn_lin_eval(lo, Al, Bl, Au, Bu);
    if (is_inf(e[1])) e[2] = positive_infinity();
    real m = lo;
    if (e[2] > 0) {
      vector[3] eh = bvn_lin_eval(hi, Al, Bl, Au, Bu);
      if (is_inf(eh[1])) eh[2] = negative_infinity();
      if (eh[2] >= 0) {
        m = hi;
        e = eh;
      } else {
        real L = lo;
        real H = hi;
        real x = 0.5 * (lo + hi);
        e = bvn_lin_eval(x, Al, Bl, Au, Bu);
        for (it in 1:60) {
          if (e[2] > 0) L = x; else H = x;
          real nx = x - e[2] / e[3];
          if (!(nx > L && nx < H)) nx = 0.5 * (L + H);
          int done = abs(nx - x) < 1e-12 * (1 + abs(x)) || H - L < 1e-12 * (1 + abs(x));
          x = nx;
          e = bvn_lin_eval(x, Al, Bl, Au, Bu);
          if (done) break;
        }
        m = x;
      }
    }
    real gm = e[1];
    if (is_inf(gm) || is_nan(gm)) return negative_infinity();
    real rs = (m == lo || m == hi) ? abs(e[2]) : 0;
    real cv = fmax(-e[3], 1);
    array[2] real ends;
    for (side in 1:2) {
      real dir = side == 1 ? -1 : 1;
      real edge = side == 1 ? lo : hi;
      if ((edge - m) * dir <= 0) {
        ends[side] = m;
      } else {
        real Tg = -rs + sqrt(rs * rs + 50);
        real T = (-rs + sqrt(rs * rs + 50 * cv)) / cv;
        for (it in 1:8) {
          if (T >= Tg) break;
          real xx = m + dir * T;
          if ((xx - edge) * dir >= 0) break;
          if (gm - bvn_lin_eval(xx, Al, Bl, Au, Bu)[1] >= 25) break;
          T = fmin(2 * T, Tg);
        }
        real xe = m + dir * T;
        ends[side] = (xe - edge) * dir >= 0 ? edge : xe;
      }
    }
    array[20] real gx = bvn_gl_x(20);
    array[20] real gw = bvn_gl_w(20);
    vector[80] vals = rep_vector(negative_infinity(), 80);
    int nv = 0;
    for (sg in 1:2) {
      real A0 = sg == 1 ? ends[1] : m;
      real B0 = sg == 1 ? m : ends[2];
      if (B0 > A0) {
        for (j in 0:1) {
          real A = A0 + (B0 - A0) * j / 2.0;
          real B = A0 + (B0 - A0) * (j + 1) / 2.0;
          real h = 0.5 * (B - A);
          real c0 = 0.5 * (A + B);
          for (q in 1:20) {
            nv += 1;
            vals[nv] = log(h * gw[q]) + bvn_lin_eval(c0 + h * gx[q], Al, Bl, Au, Bu)[1];
          }
        }
      }
    }
    if (nv == 0) return negative_infinity();
    return log_sum_exp(vals[1:nv]);
  }

  real cdp_log_cell(int kind, real muR, real sR, real muF, real sF, real cR, real cK, real tlo, real thi) {
    real B = -sR / sF;
    real xcR = (cR - muR) / sR;
    real xkh = is_inf(thi) ? positive_infinity() : (thi - cK - muR) / sR;
    real xkl = is_inf(tlo) ? negative_infinity() : (tlo - cK - muR) / sR;
    real Alo = is_inf(tlo) ? negative_infinity() : (tlo - muR - muF) / sF;
    real Blo = is_inf(tlo) ? 0 : B;
    real Ahi = is_inf(thi) ? positive_infinity() : (thi - muR - muF) / sF;
    real Bhi = is_inf(thi) ? 0 : B;
    real AK = (cK - muF) / sF;
    if (kind == 1) return bvn_log_lin_integral(Alo, Blo, Ahi, Bhi, xcR, positive_infinity());
    if (kind == 3) return log_sum_exp(bvn_log_lin_integral(Alo, Blo, AK, 0, xkl, fmin(xkh, xcR)),
                                      bvn_log_lin_integral(Alo, Blo, Ahi, Bhi, fmax(xkh, xkl), xcR));
    real ub = fmin(xcR, xkh);
    return log_sum_exp(bvn_log_lin_integral(Alo, Blo, Ahi, Bhi, negative_infinity(), fmin(xkl, ub)),
                       bvn_log_lin_integral(AK, 0, Ahi, Bhi, xkl, ub));
  }

  real bvn_rect(real a1, real b1, real a2, real b2, real rho_in) {
    real rho = fmin(fmax(rho_in, -0.999999), 0.999999);
    real ar = abs(rho);
    int m = ar < 0.75 ? 8 : (ar < 0.85 ? 10 : (ar < 0.9 ? 12 : (ar < 0.925 ? 14 : (ar < 0.97 ? 16 : 20))));
    array[m] real xs = bvn_gl_x(m);
    array[m] real ws = bvn_gl_w(m);
    real A = asin(rho);
    real p = exp(bvn_log_interval(a1, b1) + bvn_log_interval(a2, b2));
    if (!is_inf(b1) && !is_inf(b2)) p += bvn_dw_corner(b1, b2, A, xs, ws);
    if (!is_inf(a1) && !is_inf(b2)) p -= bvn_dw_corner(a1, b2, A, xs, ws);
    if (!is_inf(b1) && !is_inf(a2)) p -= bvn_dw_corner(b1, a2, A, xs, ws);
    if (!is_inf(a1) && !is_inf(a2)) p += bvn_dw_corner(a1, a2, A, xs, ws);
    if (!(p >= 1e-6)) p = exp(bvn_log_rect_tail(a1, b1, a2, b2, rho));
    return p;
  }

  real binormal_cdf_custom(real z1, real z2, real rho) {
    return bvn_rect(negative_infinity(), z1, negative_infinity(), z2, rho);
  }
'
}

.plain_functions_block <- function(model_data, family) {
  batch_info <- NULL
  family_name <- if (is.list(family)) family$family else family
  link <- if (is.list(family) && !is.null(family$link)) family$link else link_probit()
  link_name <- link$name
  
  lines <- c("functions {")
  
  cdf_code <- generate_link_cdf_wrapper(link)
  lines <- c(lines, cdf_code)
  
  if (family_name == "cumulative") {
    lines <- c(lines, '
  real cumulative_lpmf(int y, real mu, int K, vector thresh) {
    if (y == 1) {
      return log(fmax(link_cdf(thresh[1] - mu), 1e-300));
    } else if (y == K) {
      return log(fmax(link_ccdf(thresh[K-1] - mu), 1e-300));
    } else {
      return log(fmax(link_diff(thresh[y-1] - mu, thresh[y] - mu), 1e-300));
    }
  }')
    if (isTRUE(model_data$threads) && is.null(batch_info)) {
      lines <- c(lines, "", generate_partial_log_lik(model_data, family))
    }
    if (!is.null(batch_info)) {
      lines <- c(lines, "", "  // Batch C++ likelihood (implemented in user header)", batch_info$stan_decl,
                   batch_info$stan_decl_vec)
      if (isTRUE(model_data$threads)) {
        lines <- c(lines, "", batch_info$stan_partial)
      }
    }
    lines <- c(lines, "}")
    return(paste(lines, collapse = "\n"))
  }
  
  lines <- c(lines, '
  real evsdt_lpmf(int y, real dprime, real is_old, int K, vector thresh) {
    real mu = is_old * dprime;
    
    if (y == 1) {
      return log(link_cdf(thresh[1] - mu));
    } else if (y == K) {
      return log(link_ccdf(thresh[K-1] - mu));
    } else {
      return log(fmax(link_diff(thresh[y-1] - mu, thresh[y] - mu), 1e-300));
    }
  }')
  
  if (family_name %in% c("uvsdt", "dpsdt", "mixture") && model_data$has_sigma) {
    lines <- c(lines, '
  real uvsdt_lpmf(int y, real dprime, real sigma, real is_old, int K, vector thresh) {
    real mu = is_old * dprime;
    real s = is_old > 0 ? sigma : 1.0;
    
    if (y == 1) {
      return log(link_cdf((thresh[1] - mu) / s));
    } else if (y == K) {
      return log(link_ccdf((thresh[K-1] - mu) / s));
    } else {
      return log(fmax(link_diff((thresh[y-1] - mu) / s, (thresh[y] - mu) / s), 1e-300));
    }
  }')
  }
  
  if (family_name == "dpsdt") {
    if (model_data$has_sigma) {
      lines <- c(lines, '
  real dpsdt_lpmf(int y, real dprime, real sigma, real lambda, real is_old, int K, vector thresh) {
    if (is_old > 0) {
      real mu = is_old * dprime;
      if (y == K) {
        // P(Y=K | old) = lambda + (1-lambda) * (link_ccdf((thresh[K-1] - mu)/sigma))
        real p_fam = link_ccdf((thresh[K-1] - mu) / sigma);
        return log(fmax(lambda + (1 - lambda) * p_fam, 1e-300));
      } else {
        // P(Y=y | old) = (1-lambda) * [link_diff((thresh[y-1]-mu)/sigma, (thresh[y]-mu)/sigma)]
        real log_prob;
        if (y == 1) {
          log_prob = log(link_cdf((thresh[1] - mu) / sigma));
        } else {
          log_prob = log(fmax(link_diff((thresh[y-1] - mu) / sigma, (thresh[y] - mu) / sigma), 1e-300));
        }
        return log1m(lambda) + log_prob;
      }
    } else {
      return evsdt_lpmf(y | dprime, is_old, K, thresh);
    }
  }')
    } else {
      lines <- c(lines, '
  real dpsdt_lpmf(int y, real dprime, real lambda, real is_old, int K, vector thresh) {
    if (is_old > 0) {
      real mu = is_old * dprime;
      if (y == K) {
        real p_fam = link_ccdf(thresh[K-1] - mu);
        return log(fmax(lambda + (1 - lambda) * p_fam, 1e-300));
      } else {
        real log_prob;
        if (y == 1) {
          log_prob = log(link_cdf(thresh[1] - mu));
        } else {
          log_prob = log(fmax(link_diff(thresh[y-1] - mu, thresh[y] - mu), 1e-300));
        }
        return log1m(lambda) + log_prob;
      }
    } else {
      return evsdt_lpmf(y | dprime, is_old, K, thresh);
    }
  }')
    }
  }
  
  if (family_name == "mixture") {
    sigma_fixed <- !model_data$has_sigma
    sigma2_fixed <- !model_data$has_sigma2
    
    if (sigma_fixed && sigma2_fixed) {
      lines <- c(lines, '
  real mixture_sdt_lpmf(int y, real dprime, real dprime2, real lambda, real is_old, int K, vector thresh) {
    if (is_old > 0) {
      real mu1 = is_old * dprime;
      real mu2 = dprime2 - (1.0 - is_old) * dprime;
      real p1, p2;
      if (y == 1) {
        p1 = link_cdf(thresh[1] - mu1);
        p2 = link_cdf(thresh[1] - mu2);
      } else if (y == K) {
        p1 = link_ccdf(thresh[K-1] - mu1);
        p2 = link_ccdf(thresh[K-1] - mu2);
      } else {
        p1 = link_diff(thresh[y-1] - mu1, thresh[y] - mu1);
        p2 = link_diff(thresh[y-1] - mu2, thresh[y] - mu2);
      }
      return log(fmax(lambda * p1 + (1 - lambda) * p2, 1e-300));
    } else {
      return evsdt_lpmf(y | dprime, is_old, K, thresh);
    }
  }')
    } else {
      lines <- c(lines, '
  real mixture_sdt_lpmf(int y, real dprime, real sigma, real dprime2, real sigma2, real lambda, real is_old, int K, vector thresh) {
    if (is_old > 0) {
      real mu1 = is_old * dprime;
      real mu2 = dprime2 - (1.0 - is_old) * dprime;
      real p1, p2;
      if (y == 1) {
        p1 = link_cdf((thresh[1] - mu1) / sigma);
        p2 = link_cdf((thresh[1] - mu2) / sigma2);
      } else if (y == K) {
        p1 = link_ccdf((thresh[K-1] - mu1) / sigma);
        p2 = link_ccdf((thresh[K-1] - mu2) / sigma2);
      } else {
        p1 = link_diff((thresh[y-1] - mu1) / sigma, (thresh[y] - mu1) / sigma);
        p2 = link_diff((thresh[y-1] - mu2) / sigma2, (thresh[y] - mu2) / sigma2);
      }
      return log(fmax(lambda * p1 + (1 - lambda) * p2, 1e-300));
    } else {
      return evsdt_lpmf(y | dprime, is_old, K, thresh);
    }
  }')
    }
    
    if (isTRUE(model_data$has_lure_mixture)) {
      if (model_data$has_sigma_L) {
        lines <- c(lines, '
  // Lure mixture: P(y|new) = lambda_L * P(y|-dprime_L, sigma_L) + (1-lambda_L) * P(y|0, 1)
  // dprime_L is lure detection strength; negative sign means lures fall below reference
  real lure_mixture_lpmf(int y, real dprime_L, real sigma_L, real lambda_L, int K, vector thresh) {
    real p_lure, p_ref;
    if (y == 1) {
      p_lure = link_cdf((thresh[1] + dprime_L) / sigma_L);
      p_ref = link_cdf(thresh[1]);
    } else if (y == K) {
      p_lure = link_ccdf((thresh[K-1] + dprime_L) / sigma_L);
      p_ref = link_ccdf(thresh[K-1]);
    } else {
      p_lure = link_diff((thresh[y-1] + dprime_L) / sigma_L, (thresh[y] + dprime_L) / sigma_L);
      p_ref = link_diff(thresh[y-1], thresh[y]);
    }
    return log(fmax(lambda_L * p_lure + (1 - lambda_L) * p_ref, 1e-300));
  }')
      } else {
        lines <- c(lines, '
  // Lure mixture: P(y|new) = lambda_L * P(y|-dprime_L, 1) + (1-lambda_L) * P(y|0, 1)
  // dprime_L is lure detection strength; negative sign means lures fall below reference
  real lure_mixture_lpmf(int y, real dprime_L, real lambda_L, int K, vector thresh) {
    real p_lure, p_ref;
    if (y == 1) {
      p_lure = link_cdf(thresh[1] + dprime_L);
      p_ref = link_cdf(thresh[1]);
    } else if (y == K) {
      p_lure = link_ccdf(thresh[K-1] + dprime_L);
      p_ref = link_ccdf(thresh[K-1]);
    } else {
      p_lure = link_diff(thresh[y-1] + dprime_L, thresh[y] + dprime_L);
      p_ref = link_diff(thresh[y-1], thresh[y]);
    }
    return log(fmax(lambda_L * p_lure + (1 - lambda_L) * p_ref, 1e-300));
  }')
      }
    }
  }
  
  if (family_name == "source_mixture") {
    has_dprime_B <- isTRUE(model_data$has_dprime_B)
    has_lambda_B <- isTRUE(model_data$has_lambda_B)
    
    if (has_dprime_B && has_lambda_B) {
      lines <- c(lines, '
  // Source mixture SDT - General model (separate d and lambda for each source)
  real source_mixture_lpmf(int y, int source, real dprime_A, real dprime_B, 
                           real lambda_A, real lambda_B, int K, vector thresh) {
    real p_attended;
    real p_nonattended;
    real lambda;
    real d;
    
    // Select parameters based on source (0 = A, 1 = B)
    if (source == 0) {
      lambda = lambda_A;
      d = dprime_A;
    } else {
      lambda = lambda_B;
      d = dprime_B;
    }
    
    // Probability is mixture of attended (at d) and nonattended (at 0) distributions
    if (y == 1) {
      p_attended = link_cdf(thresh[1] - d);
      p_nonattended = link_cdf(thresh[1]);
    } else if (y == K) {
      p_attended = link_ccdf(thresh[K-1] - d);
      p_nonattended = link_ccdf(thresh[K-1]);
    } else {
      p_attended = link_diff(thresh[y-1] - d, thresh[y] - d);
      p_nonattended = link_diff(thresh[y-1], thresh[y]);
    }
    
    return log(fmax(lambda * p_attended + (1 - lambda) * p_nonattended, 1e-300));
  }')
    } else if (has_dprime_B) {
      lines <- c(lines, '
  // Source mixture SDT - Equal attention model (lambda_B = lambda)
  real source_mixture_lpmf(int y, int source, real dprime_A, real dprime_B, 
                           real lambda, int K, vector thresh) {
    real p_attended;
    real p_nonattended;
    real d;
    
    d = source == 0 ? dprime_A : dprime_B;
    
    if (y == 1) {
      p_attended = link_cdf(thresh[1] - d);
      p_nonattended = link_cdf(thresh[1]);
    } else if (y == K) {
      p_attended = link_ccdf(thresh[K-1] - d);
      p_nonattended = link_ccdf(thresh[K-1]);
    } else {
      p_attended = link_diff(thresh[y-1] - d, thresh[y] - d);
      p_nonattended = link_diff(thresh[y-1], thresh[y]);
    }
    
    return log(fmax(lambda * p_attended + (1 - lambda) * p_nonattended, 1e-300));
  }')
    } else if (has_lambda_B) {
      lines <- c(lines, '
  // Source mixture SDT - Symmetric d model (dprime_B = -dprime)
  real source_mixture_lpmf(int y, int source, real dprime, real lambda_A, 
                           real lambda_B, int K, vector thresh) {
    real p_attended;
    real p_nonattended;
    real lambda;
    real d;
    
    // dprime_B = -dprime (symmetric)
    if (source == 0) {
      lambda = lambda_A;
      d = dprime;  // dprime_A
    } else {
      lambda = lambda_B;
      d = -dprime;  // dprime_B = -dprime_A
    }
    
    if (y == 1) {
      p_attended = link_cdf(thresh[1] - d);
      p_nonattended = link_cdf(thresh[1]);
    } else if (y == K) {
      p_attended = link_ccdf(thresh[K-1] - d);
      p_nonattended = link_ccdf(thresh[K-1]);
    } else {
      p_attended = link_diff(thresh[y-1] - d, thresh[y] - d);
      p_nonattended = link_diff(thresh[y-1], thresh[y]);
    }
    
    return log(fmax(lambda * p_attended + (1 - lambda) * p_nonattended, 1e-300));
  }')
    } else {
      lines <- c(lines, '
  // Source mixture SDT - Fully symmetric model (dprime_B = -dprime, lambda_B = lambda)
  real source_mixture_lpmf(int y, int source, real dprime, real lambda, int K, vector thresh) {
    real p_attended;
    real p_nonattended;
    real d;
    
    // Symmetric: d_A = dprime, d_B = -dprime
    d = source == 0 ? dprime : -dprime;
    
    if (y == 1) {
      p_attended = link_cdf(thresh[1] - d);
      p_nonattended = link_cdf(thresh[1]);
    } else if (y == K) {
      p_attended = link_ccdf(thresh[K-1] - d);
      p_nonattended = link_ccdf(thresh[K-1]);
    } else {
      p_attended = link_diff(thresh[y-1] - d, thresh[y] - d);
      p_nonattended = link_diff(thresh[y-1], thresh[y]);
    }
    
    return log(fmax(lambda * p_attended + (1 - lambda) * p_nonattended, 1e-300));
  }')
    }
  }
  
  if (family_name %in% c("bivariate_sdt", "bivariate_dp", "bivariate_cumulative")) {
    lines <- c(lines, .stan_bvn_functions(), '
  real compute_bivariate_prob(int resp1, int resp2, int K1, int K2,
                              real mu1, real mu2, real sigma1, real sigma2,
                              vector c1, vector c2, real rho) {
    real a1 = resp1 == 1 ? negative_infinity() : (c1[resp1 - 1] - mu1) / sigma1;
    real b1 = resp1 == K1 ? positive_infinity() : (c1[resp1] - mu1) / sigma1;
    real a2 = resp2 == 1 ? negative_infinity() : (c2[resp2 - 1] - mu2) / sigma2;
    real b2 = resp2 == K2 ? positive_infinity() : (c2[resp2] - mu2) / sigma2;
    return bvn_rect(a1, b1, a2, b2, rho);
  }')

    if (isTRUE(model_data$bounded)) {
      lines <- c(lines, '
  // Bounded bivariate probability: conditional source mean clamped at 0
  real compute_bounded_prob(int resp_det, int resp_src, int K1, int K2,
                           real mu_I, real mu_S, real sigma_I, real sigma_S,
                           vector c1, vector c2, real rho) {

    if (abs(rho) < 1e-10) {
      return compute_bivariate_prob(resp_det, resp_src, K1, K2,
                                    mu_I, mu_S, sigma_I, sigma_S,
                                    c1, c2, 0.0);
    }

    real cp = mu_I - sigma_I * mu_S / (rho * sigma_S);
    real y_lo = (resp_det == 1)  ? negative_infinity() : c1[resp_det - 1];
    real y_hi = (resp_det == K1) ? positive_infinity() : c1[resp_det];

    int below_only = (y_hi <= cp) ? 1 : 0;
    int above_only = (y_lo >= cp) ? 1 : 0;

    real sigma_cond = sigma_S * sqrt(1 - rho * rho);
    real prob = 0.0;

    if (below_only == 1) {
      prob = compute_bivariate_prob(resp_det, resp_src, K1, K2,
                                     mu_I, 0.0, sigma_I, sigma_cond,
                                     c1, c2, 0.0);
    }
    else if (above_only == 1) {
      prob = compute_bivariate_prob(resp_det, resp_src, K1, K2,
                                     mu_I, mu_S, sigma_I, sigma_S,
                                     c1, c2, rho);
    }
    else {
      real z_cp = (cp - mu_I) / sigma_I;
      real z_ylo = resp_det == 1 ? negative_infinity() : (c1[resp_det - 1] - mu_I) / sigma_I;
      real z_yhi = resp_det == K1 ? positive_infinity() : (c1[resp_det] - mu_I) / sigma_I;
      real s_lo = resp_src == 1 ? negative_infinity() : c2[resp_src - 1];
      real s_hi = resp_src == K2 ? positive_infinity() : c2[resp_src];
      prob = exp(bvn_log_interval(z_ylo, z_cp)
                 + bvn_log_interval(bvn_std(s_lo, 0, sigma_cond), bvn_std(s_hi, 0, sigma_cond)))
             + bvn_rect(z_cp, z_yhi, bvn_std(s_lo, mu_S, sigma_S), bvn_std(s_hi, mu_S, sigma_S), rho);
    }

    return prob;
  }')
    }

    lines <- c(lines, '

  // Bivariate SDT log-likelihood for a single observation
  real bivariate_sdt_lpmf_custom(int y1, int y2, int item_type, int K1, int K2,
                          real dprime_A, real dprime_B, real discrim_A, real discrim_B,
                          real sigma1_A, real sigma1_B, real sigma2_A, real sigma2_B,
                          real rho_A, real rho_B, real rho_N,
                          vector c1, vector c2) {
    real mu1;
    real mu2;
    real sigma1;
    real sigma2;
    real rho;
    
    // Select parameters based on item type (1=new, 2=A, 3=B)
    if (item_type == 1) {
      // New items: reference distribution
      mu1 = 0;
      mu2 = 0;
      sigma1 = 1;
      sigma2 = 1;
      rho = rho_N;
    } else if (item_type == 2) {
      // Source A items
      mu1 = dprime_A;
      mu2 = discrim_A;
      sigma1 = sigma1_A;
      sigma2 = sigma2_A;
      rho = rho_A;
    } else {
      // Source B items: parameters describe B items position on the shared
      // source axis (no negation). discrim_B is the signed source-axis
      // position of B items; rho_B is the bivariate-normal correlation for
      // B items, both in the same shared-axis frame as A items. Expect
      // discrim and discrim_B to come out with opposite signs in a fit
      // (matching A and B being on opposite ends of the source axis).
      mu1 = dprime_B;
      mu2 = discrim_B;
      sigma1 = sigma1_B;
      sigma2 = sigma2_B;
      rho = rho_B;
    }

    // Compute probability and return log
    real p = compute_bivariate_prob(y1, y2, K1, K2, mu1, mu2, sigma1, sigma2, c1, c2, rho);
    return log(fmax(p, 1e-300));
  }')

    if (isTRUE(model_data$varying_source_criteria)) {
      lines <- c(lines, '
  // Bivariate SDT with source criteria varying by item response level
  real bivariate_sdt_varying_lpmf_custom(int y1, int y2, int item_type, int K1, int K2,
                          real dprime_A, real dprime_B, real discrim_A, real discrim_B,
                          real sigma1_A, real sigma1_B, real sigma2_A, real sigma2_B,
                          real rho_A, real rho_B, real rho_N,
                          vector c1, matrix c2_varying) {
    real mu1;
    real mu2;
    real sigma1;
    real sigma2;
    real rho;

    if (item_type == 1) {
      mu1 = 0; mu2 = 0; sigma1 = 1; sigma2 = 1; rho = rho_N;
    } else if (item_type == 2) {
      mu1 = dprime_A; mu2 = discrim_A; sigma1 = sigma1_A; sigma2 = sigma2_A; rho = rho_A;
    } else {
      mu1 = dprime_B; mu2 = discrim_B; sigma1 = sigma1_B; sigma2 = sigma2_B; rho = rho_B;
    }

    // Use source thresholds for this item response level
    vector[K2-1] c2 = to_vector(c2_varying[y1, ]);
    real p = compute_bivariate_prob(y1, y2, K1, K2, mu1, mu2, sigma1, sigma2, c1, c2, rho);
    return log(fmax(p, 1e-300));
  }')
    }

    if (isTRUE(model_data$bounded)) {
      lines <- c(lines, '
  // Bounded bivariate SDT log-likelihood: new items use standard, old items use bounded
  real bivariate_sdt_bounded_lpmf_custom(int y1, int y2, int item_type, int K1, int K2,
                          real dprime_A, real dprime_B, real discrim_A, real discrim_B,
                          real sigma1_A, real sigma1_B, real sigma2_A, real sigma2_B,
                          real rho_A, real rho_B, real rho_N,
                          vector c1, vector c2) {
    real mu1; real mu2; real sigma1; real sigma2; real rho;
    if (item_type == 1) {
      mu1 = 0; mu2 = 0; sigma1 = 1; sigma2 = 1; rho = rho_N;
    } else if (item_type == 2) {
      // Source A on negative source axis (low source ratings); discrim_A and
      // rho_A are positive magnitudes (log/logistic links), negated here to
      // place A on the negative axis with the desired effective correlation.
      mu1 = dprime_A; mu2 = -discrim_A; sigma1 = sigma1_A; sigma2 = sigma2_A; rho = -rho_A;
    } else {
      // Source B on positive source axis (high source ratings); discrim_B and
      // rho_B used directly (positive magnitudes from log/logistic links).
      mu1 = dprime_B; mu2 = discrim_B; sigma1 = sigma1_B; sigma2 = sigma2_B; rho = rho_B;
    }
    real p;
    if (item_type == 1) {
      p = compute_bivariate_prob(y1, y2, K1, K2, mu1, mu2, sigma1, sigma2, c1, c2, rho);
    } else {
      p = compute_bounded_prob(y1, y2, K1, K2, mu1, mu2, sigma1, sigma2, c1, c2, rho);
    }
    return log(fmax(p, 1e-300));
  }')

      if (isTRUE(model_data$varying_source_criteria)) {
        lines <- c(lines, '
  // Bounded bivariate SDT with varying source criteria
  real bivariate_sdt_varying_bounded_lpmf_custom(int y1, int y2, int item_type, int K1, int K2,
                          real dprime_A, real dprime_B, real discrim_A, real discrim_B,
                          real sigma1_A, real sigma1_B, real sigma2_A, real sigma2_B,
                          real rho_A, real rho_B, real rho_N,
                          vector c1, matrix c2_varying) {
    real mu1; real mu2; real sigma1; real sigma2; real rho;
    if (item_type == 1) {
      mu1 = 0; mu2 = 0; sigma1 = 1; sigma2 = 1; rho = rho_N;
    } else if (item_type == 2) {
      // Source A on negative source axis; bounded convention: negate to place
      // A on the negative end of the shared source axis.
      mu1 = dprime_A; mu2 = -discrim_A; sigma1 = sigma1_A; sigma2 = sigma2_A; rho = -rho_A;
    } else {
      // Source B on positive source axis; used directly.
      mu1 = dprime_B; mu2 = discrim_B; sigma1 = sigma1_B; sigma2 = sigma2_B; rho = rho_B;
    }
    vector[K2-1] c2 = to_vector(c2_varying[y1, ]);
    real p;
    if (item_type == 1) {
      p = compute_bivariate_prob(y1, y2, K1, K2, mu1, mu2, sigma1, sigma2, c1, c2, rho);
    } else {
      p = compute_bounded_prob(y1, y2, K1, K2, mu1, mu2, sigma1, sigma2, c1, c2, rho);
    }
    return log(fmax(p, 1e-300));
  }')
      }
    }

    if (identical(model_data$new_source_criteria, "shared")) {
      if (!isTRUE(model_data$bounded)) {
        lines <- c(lines, '
  // Varying source criteria with shared new-response thresholds
  real bivariate_sdt_varying_new_shared_lpmf_custom(int y1, int y2, int item_type, int K1, int K2,
                          real dprime_A, real dprime_B, real discrim_A, real discrim_B,
                          real sigma1_A, real sigma1_B, real sigma2_A, real sigma2_B,
                          real rho_A, real rho_B, real rho_N,
                          vector c1, matrix c2_varying, vector c2_new,
                          array[] int is_new_response) {
    real mu1; real mu2; real sigma1; real sigma2; real rho;
    if (item_type == 1) {
      mu1 = 0; mu2 = 0; sigma1 = 1; sigma2 = 1; rho = rho_N;
    } else if (item_type == 2) {
      mu1 = dprime_A; mu2 = discrim_A; sigma1 = sigma1_A; sigma2 = sigma2_A; rho = rho_A;
    } else {
      mu1 = dprime_B; mu2 = discrim_B; sigma1 = sigma1_B; sigma2 = sigma2_B; rho = rho_B;
    }
    vector[K2-1] c2;
    if (is_new_response[y1] == 1) {
      c2 = c2_new;
    } else {
      c2 = to_vector(c2_varying[y1, ]);
    }
    real p = compute_bivariate_prob(y1, y2, K1, K2, mu1, mu2, sigma1, sigma2, c1, c2, rho);
    return log(fmax(p, 1e-300));
  }')
      } else {
        lines <- c(lines, '
  // Bounded varying source criteria with shared new-response thresholds
  real bivariate_sdt_varying_bounded_new_shared_lpmf_custom(int y1, int y2, int item_type, int K1, int K2,
                          real dprime_A, real dprime_B, real discrim_A, real discrim_B,
                          real sigma1_A, real sigma1_B, real sigma2_A, real sigma2_B,
                          real rho_A, real rho_B, real rho_N,
                          vector c1, matrix c2_varying, vector c2_new,
                          array[] int is_new_response) {
    real mu1; real mu2; real sigma1; real sigma2; real rho;
    if (item_type == 1) {
      mu1 = 0; mu2 = 0; sigma1 = 1; sigma2 = 1; rho = rho_N;
    } else if (item_type == 2) {
      // Source A on negative source axis; bounded convention.
      mu1 = dprime_A; mu2 = -discrim_A; sigma1 = sigma1_A; sigma2 = sigma2_A; rho = -rho_A;
    } else {
      // Source B on positive source axis.
      mu1 = dprime_B; mu2 = discrim_B; sigma1 = sigma1_B; sigma2 = sigma2_B; rho = rho_B;
    }
    vector[K2-1] c2;
    if (is_new_response[y1] == 1) {
      c2 = c2_new;
    } else {
      c2 = to_vector(c2_varying[y1, ]);
    }
    real p;
    if (item_type == 1) {
      p = compute_bivariate_prob(y1, y2, K1, K2, mu1, mu2, sigma1, sigma2, c1, c2, rho);
    } else {
      p = compute_bounded_prob(y1, y2, K1, K2, mu1, mu2, sigma1, sigma2, c1, c2, rho);
    }
    return log(fmax(p, 1e-300));
  }')
      }
    }
  }

  if (family_name == "bivariate_dp") {
    lines <- c(lines, '
  real univariate_cell_prob(int resp, int K, real mu, real sigma, vector thresh) {
    real lo = resp == 1 ? negative_infinity() : (thresh[resp - 1] - mu) / sigma;
    real hi = resp == K ? positive_infinity() : (thresh[resp] - mu) / sigma;
    return link_diff(lo, hi);
  }')

    if (isTRUE(model_data$bounded)) {
      lines <- c(lines, '
  // Bounded marginal source probability for p2 component (item recollected, source from familiarity)
  real compute_bounded_marginal_source(int resp_src, int K2,
                                       real mu_I, real mu_S,
                                       real sigma_I, real sigma_S,
                                       real rho, vector c2) {

    real s_lo = resp_src == 1 ? negative_infinity() : c2[resp_src - 1];
    real s_hi = resp_src == K2 ? positive_infinity() : c2[resp_src];
    if (abs(rho) < 1e-10) return exp(bvn_log_interval(bvn_std(s_lo, mu_S, sigma_S), bvn_std(s_hi, mu_S, sigma_S)));
    real cp = mu_I - sigma_I * mu_S / (rho * sigma_S);
    real sigma_cond = sigma_S * sqrt(1 - rho * rho);
    real z_cp = (cp - mu_I) / sigma_I;
    return exp(bvn_log_interval(negative_infinity(), z_cp)
               + bvn_log_interval(bvn_std(s_lo, 0, sigma_cond), bvn_std(s_hi, 0, sigma_cond)))
           + bvn_rect(z_cp, positive_infinity(), bvn_std(s_lo, mu_S, sigma_S), bvn_std(s_hi, mu_S, sigma_S), rho);
  }')
    }

    bounded <- isTRUE(model_data$bounded)
    biv_fn <- if (bounded) "compute_bounded_prob" else "compute_bivariate_prob"
    marg_fn <- if (bounded) "compute_bounded_marginal_source" else "univariate_cell_prob"
    dp_src_A <- if (bounded) {
      "mu1 = dprime_A; mu2 = -discrim_A; sigma1 = sigma1_A; sigma2 = sigma2_A; rho = -rho_A;"
    } else {
      "mu1 = dprime_A; mu2 = discrim_A; sigma1 = sigma1_A; sigma2 = sigma2_A; rho = rho_A;"
    }
    dp_src_B <- "mu1 = dprime_B; mu2 = discrim_B; sigma1 = sigma1_B; sigma2 = sigma2_B; rho = rho_B;"

    lines <- c(lines, sprintf('
  // Bivariate Dual-Process log-likelihood
  real bivariate_dp_lpmf_custom(int y1, int y2, int item_type, int K1, int K2,
                          real dprime_A, real dprime_B, real discrim_A, real discrim_B,
                          real sigma1_A, real sigma1_B, real sigma2_A, real sigma2_B,
                          real rho_A, real rho_B, real rho_N,
                          real R_I, real R_S, real R_I_B, real R_S_B,
                          vector c1, vector c2) {
    // New items: standard bivariate, no recollection
    if (item_type == 1) {
      real p = compute_bivariate_prob(y1, y2, K1, K2, 0.0, 0.0, 1.0, 1.0, c1, c2, rho_N);
      return log(fmax(p, 1e-300));
    }

    real mu1; real mu2; real sigma1; real sigma2; real rho;
    real R_I_eff; real R_S_eff;
    if (item_type == 2) {
      %s
      R_I_eff = R_I; R_S_eff = R_S;
    } else {
      %s
      R_I_eff = R_I_B; R_S_eff = R_S_B;
    }

    // p1: both from familiarity
    real p1 = (1.0 - R_I_eff) * %s(y1, y2, K1, K2, mu1, mu2, sigma1, sigma2, c1, c2, rho);

    // p2: item recollected (highest conf), source from familiarity
    real p2 = 0.0;
    if (y1 == K1) {%s
    }

    // p3: both recollected -> corner cell
    real p3 = 0.0;
    if (y1 == K1) {
      // Recollection success cells (new convention): Source A items at the
      // low end of the source axis (y2 = 1, "sure A"); Source B items at the
      // high end (y2 = K2, "sure B").
      if (item_type == 2 && y2 == 1) p3 = R_I_eff * R_S_eff;
      else if (item_type == 3 && y2 == K2) p3 = R_I_eff * R_S_eff;
    }

    return log(fmax(p1 + p2 + p3, 1e-300));
  }', dp_src_A, dp_src_B, biv_fn,
    if (bounded) {
      sprintf('\n      p2 = R_I_eff * (1.0 - R_S_eff) * compute_bounded_marginal_source(y2, K2, mu1, mu2, sigma1, sigma2, rho, c2);')
    } else {
      '\n      p2 = R_I_eff * (1.0 - R_S_eff) * univariate_cell_prob(y2, K2, mu2, sigma2, c2);'
    }))

    if (isTRUE(model_data$varying_source_criteria)) {
      lines <- c(lines, sprintf('
  // Bivariate DP with varying source criteria
  real bivariate_dp_varying_lpmf_custom(int y1, int y2, int item_type, int K1, int K2,
                          real dprime_A, real dprime_B, real discrim_A, real discrim_B,
                          real sigma1_A, real sigma1_B, real sigma2_A, real sigma2_B,
                          real rho_A, real rho_B, real rho_N,
                          real R_I, real R_S, real R_I_B, real R_S_B,
                          vector c1, matrix c2_varying) {
    if (item_type == 1) {
      vector[K2-1] c2 = to_vector(c2_varying[y1, ]);
      real p = compute_bivariate_prob(y1, y2, K1, K2, 0.0, 0.0, 1.0, 1.0, c1, c2, rho_N);
      return log(fmax(p, 1e-300));
    }

    real mu1; real mu2; real sigma1; real sigma2; real rho;
    real R_I_eff; real R_S_eff;
    if (item_type == 2) {
      %s
      R_I_eff = R_I; R_S_eff = R_S;
    } else {
      %s
      R_I_eff = R_I_B; R_S_eff = R_S_B;
    }

    vector[K2-1] c2 = to_vector(c2_varying[y1, ]);
    real p1 = (1.0 - R_I_eff) * %s(y1, y2, K1, K2, mu1, mu2, sigma1, sigma2, c1, c2, rho);

    real p2 = 0.0;
    if (y1 == K1) {%s
    }

    real p3 = 0.0;
    if (y1 == K1) {
      // Recollection success cells (new convention): Source A items at the
      // low end of the source axis (y2 = 1, "sure A"); Source B items at the
      // high end (y2 = K2, "sure B").
      if (item_type == 2 && y2 == 1) p3 = R_I_eff * R_S_eff;
      else if (item_type == 3 && y2 == K2) p3 = R_I_eff * R_S_eff;
    }

    return log(fmax(p1 + p2 + p3, 1e-300));
  }', dp_src_A, dp_src_B, biv_fn,
      if (bounded) {
        '\n      p2 = R_I_eff * (1.0 - R_S_eff) * compute_bounded_marginal_source(y2, K2, mu1, mu2, sigma1, sigma2, rho, c2);'
      } else {
        '\n      p2 = R_I_eff * (1.0 - R_S_eff) * univariate_cell_prob(y2, K2, mu2, sigma2, c2);'
      }))
    }

    if (identical(model_data$new_source_criteria, "shared")) {
      lines <- c(lines, sprintf('
  // Bivariate DP with varying source criteria + shared new-response thresholds
  real bivariate_dp_varying_new_shared_lpmf_custom(int y1, int y2, int item_type, int K1, int K2,
                          real dprime_A, real dprime_B, real discrim_A, real discrim_B,
                          real sigma1_A, real sigma1_B, real sigma2_A, real sigma2_B,
                          real rho_A, real rho_B, real rho_N,
                          real R_I, real R_S, real R_I_B, real R_S_B,
                          vector c1, matrix c2_varying, vector c2_new,
                          array[] int is_new_response) {
    // Select source thresholds based on detection RESPONSE (not item type)
    vector[K2-1] c2;
    if (is_new_response[y1] == 1) {
      c2 = c2_new;
    } else {
      c2 = to_vector(c2_varying[y1, ]);
    }

    if (item_type == 1) {
      real p = compute_bivariate_prob(y1, y2, K1, K2, 0.0, 0.0, 1.0, 1.0, c1, c2, rho_N);
      return log(fmax(p, 1e-300));
    }

    real mu1; real mu2; real sigma1; real sigma2; real rho;
    real R_I_eff; real R_S_eff;
    if (item_type == 2) {
      %s
      R_I_eff = R_I; R_S_eff = R_S;
    } else {
      %s
      R_I_eff = R_I_B; R_S_eff = R_S_B;
    }

    real p1 = (1.0 - R_I_eff) * %s(y1, y2, K1, K2, mu1, mu2, sigma1, sigma2, c1, c2, rho);

    real p2 = 0.0;
    if (y1 == K1) {%s
    }

    real p3 = 0.0;
    if (y1 == K1) {
      // Recollection success cells (new convention): Source A items at the
      // low end of the source axis (y2 = 1, "sure A"); Source B items at the
      // high end (y2 = K2, "sure B").
      if (item_type == 2 && y2 == 1) p3 = R_I_eff * R_S_eff;
      else if (item_type == 3 && y2 == K2) p3 = R_I_eff * R_S_eff;
    }

    return log(fmax(p1 + p2 + p3, 1e-300));
  }', dp_src_A, dp_src_B, biv_fn,
      if (bounded) {
        '\n      p2 = R_I_eff * (1.0 - R_S_eff) * compute_bounded_marginal_source(y2, K2, mu1, mu2, sigma1, sigma2, rho, c2);'
      } else {
        '\n      p2 = R_I_eff * (1.0 - R_S_eff) * univariate_cell_prob(y2, K2, mu2, sigma2, c2);'
      }))
    }
  }

  if (family_name == "bivariate_cumulative") {
    lines <- c(lines, '
  real bivariate_cumulative_lpmf_custom(int y1, int y2, int K1, int K2,
                                         real mu1, real mu2,
                                         vector c1, vector c2, real rho) {
    real p = compute_bivariate_prob(y1, y2, K1, K2, mu1, mu2, 1.0, 1.0, c1, c2, rho);
    return log(fmax(p, 1e-300));
  }')
  }

  if (family_name == "cdp") {
    lines <- c(lines, .stan_bvn_functions(), '
  real binormal_cdf_cdp(real z1, real z2, real rho) {
    return bvn_rect(negative_infinity(), z1, negative_infinity(), z2, rho);
  }

  real binormal_strip_upper(real z_cR, real z_tau_lo, real z_tau_hi, real rho) {
    return fmax(bvn_rect(z_cR, positive_infinity(), z_tau_lo, z_tau_hi, rho), 1e-300);
  }

  real binormal_strip_lower(real z_cR, real z_tau_lo, real z_tau_hi, real rho) {
    return fmax(bvn_rect(negative_infinity(), z_cR, z_tau_lo, z_tau_hi, rho), 1e-300);
  }')

    if (isTRUE(model_data$stan_data$n_rkg == 3)) {
      lines <- c(lines, '
  // G(c) = P(R <= c_R, F <= c_K, M < c) -- unified formula (Guess-first approach)
  // Uses fmin for exact min (no smooth approximation needed)
  real G_cdp(real c, real mu_R, real sigma_R, real mu_F, real sigma_F,
             real c_R, real c_K) {
    real s = fmin(c_R, c - c_K);
    real z_s = (s - mu_R) / sigma_R;
    real z_cR = (c_R - mu_R) / sigma_R;
    real z_cK = (c_K - mu_F) / sigma_F;
    real mu_M = mu_R + mu_F;
    real sigma_M = sqrt(square(sigma_R) + square(sigma_F));
    real z_c = bvn_std(c, mu_M, sigma_M);
    real rho = sigma_R / sigma_M;
    return Phi(z_s) * Phi(z_cK) + binormal_cdf_cdp(z_cR, z_c, rho)
           - binormal_cdf_cdp(z_s, z_c, rho);
  }

  // Compute R/K/G probabilities using G(c) (Guess-first approach)
  // Shares binormal CDF values between Remember and Guess to avoid
  // redundant Owen T evaluations.
  vector compute_rkg_probs(real mu_R, real sigma_R, real mu_F, real sigma_F,
                           real c_R, real c_K, real tau_lo, real tau_hi) {
    vector[3] probs;
    real mu_M = mu_R + mu_F;
    real sigma_M = sqrt(square(sigma_R) + square(sigma_F));
    real rho = sigma_R / sigma_M;
    real z_cR = (c_R - mu_R) / sigma_R;
    real z_cK_F = (c_K - mu_F) / sigma_F;

    // Standardized thresholds
    real z_lo = bvn_std(tau_lo, mu_M, sigma_M);
    real z_hi = bvn_std(tau_hi, mu_M, sigma_M);

    // SHARED: Phi2(z_cR, z_hi/z_lo, rho) -- used by both Remember and Guess
    real bv_cR_hi = binormal_cdf_cdp(z_cR, z_hi, rho);
    real bv_cR_lo = is_inf(tau_lo) ? 0 : binormal_cdf_cdp(z_cR, z_lo, rho);

    // REMEMBER: P(R > c_R, tau_lo < M < tau_hi)
    real p_Remember = bvn_rect(z_cR, positive_infinity(), z_lo, z_hi, rho);

    // GUESS: G(tau_hi) - G(tau_lo) -- inlined to reuse bv_cR
    // G(c) = Phi(z_s)*Phi(z_cK_F) + Phi2(z_cR, z_c, rho) - Phi2(z_s, z_c, rho)
    // where s = fmin(c_R, c-c_K), z_c = z_hi or z_lo (same as thresholds!)
    real s_hi = fmin(c_R, tau_hi - c_K);
    real z_s_hi = (s_hi - mu_R) / sigma_R;
    real G_hi;
    if (c_R <= tau_hi - c_K) {
      G_hi = Phi(z_cR) * Phi(z_cK_F);
    } else {
      G_hi = Phi(z_s_hi) * Phi(z_cK_F) + bv_cR_hi - binormal_cdf_cdp(z_s_hi, z_hi, rho);
    }

    real G_lo;
    if (is_inf(tau_lo)) {
      G_lo = 0;
    } else if (c_R <= tau_lo - c_K) {
      G_lo = Phi(z_cR) * Phi(z_cK_F);
    } else {
      real z_s_lo = (tau_lo - c_K - mu_R) / sigma_R;
      G_lo = Phi(z_s_lo) * Phi(z_cK_F) + bv_cR_lo - binormal_cdf_cdp(z_s_lo, z_lo, rho);
    }

    real p_Guess = G_hi - G_lo;

    // KNOW: remainder
    real p_Band = link_diff(z_lo, z_hi);
    real p_Know = p_Band - p_Remember - p_Guess;

    probs[1] = p_Remember;
    probs[2] = p_Know >= 1e-6 ? p_Know
      : exp(cdp_log_cell(2, mu_R, sigma_R, mu_F, sigma_F, c_R, c_K, tau_lo, tau_hi));
    probs[3] = p_Guess >= 1e-6 ? p_Guess
      : exp(cdp_log_cell(3, mu_R, sigma_R, mu_F, sigma_F, c_R, c_K, tau_lo, tau_hi));
    return probs;
  }')
    }
  }

  if (family_name == "vrdp2d") {
    lines <- c(lines, '
  // 2D-VRDP log-likelihood for a single observation
  // Since rho=0 for all distributions, bivariate normal factors into product of univariates:
  // P(X <= x, Y <= y | rho=0) = Phi(x) * Phi(y)
  //
  // This is MUCH more efficient than computing bivariate CDFs!
  //
  // item_type: 1=new, 2=source A, 3=source B
  // dprime_F: familiarity strength 
  // dprime_R: recollection boost
  // dprime_S: source discriminability (only for recollected items)
  // R: recollection probability
  // sigma_S: SD on source dimension for recollected items
  
  // Helper: compute univariate cell probability
  real univariate_cell_prob(int resp, int K, real mu, real sigma, vector thresh) {
    real lo = resp == 1 ? negative_infinity() : (thresh[resp - 1] - mu) / sigma;
    real hi = resp == K ? positive_infinity() : (thresh[resp] - mu) / sigma;
    return link_diff(lo, hi);
  }
  
  real vrdp2d_lpmf_custom(int y1, int y2, int item_type, int K1, int K2,
                          real dprime_F, real dprime_R, real dprime_S, real dprime_S_B,
                          real R, real sigma_item, real sigma_S,
                          vector c1, vector c2) {
    real p;
    
    if (item_type == 1) {
      // New items: at (0, 0) with SD=1 on both dimensions
      real p1 = univariate_cell_prob(y1, K1, 0.0, 1.0, c1);
      real p2 = univariate_cell_prob(y2, K2, 0.0, 1.0, c2);
      p = p1 * p2;
    } else {
      // Old items: mixture of non-recollected and recollected
      // Source discriminability: Source A gets dprime_S, Source B gets dprime_S_B
      real source_d = (item_type == 2) ? dprime_S : dprime_S_B;
      
      // Non-recollected: at (dprime_F, 0) - item memory but no source info
      real p1_fam = univariate_cell_prob(y1, K1, dprime_F, sigma_item, c1);
      real p2_fam = univariate_cell_prob(y2, K2, 0.0, 1.0, c2);
      real p_fam = p1_fam * p2_fam;
      
      // Recollected: at (dprime_F + dprime_R, source_d)
      real p1_rec = univariate_cell_prob(y1, K1, dprime_F + dprime_R, sigma_item, c1);
      real p2_rec = univariate_cell_prob(y2, K2, source_d, sigma_S, c2);
      real p_rec = p1_rec * p2_rec;
      
      // Mixture
      p = (1.0 - R) * p_fam + R * p_rec;
    }
    
    return log(fmax(p, 1e-300));
  }
  
  // Version with source criteria varying by item response
  // thresh2_varying is a K1 x (K2-1) matrix where each row is thresholds for that item response
  real vrdp2d_varying_lpmf_custom(int y1, int y2, int item_type, int K1, int K2,
                                   real dprime_F, real dprime_R, real dprime_S, real dprime_S_B,
                                   real R, real sigma_item, real sigma_S,
                                   vector c1, matrix c2_varying) {
    real p;
    
    // Get the source thresholds for this item response level
    vector[K2-1] c2 = to_vector(c2_varying[y1, ]);
    
    if (item_type == 1) {
      // New items: at (0, 0) with SD=1 on both dimensions
      real p1 = univariate_cell_prob(y1, K1, 0.0, 1.0, c1);
      real p2 = univariate_cell_prob(y2, K2, 0.0, 1.0, c2);
      p = p1 * p2;
    } else {
      // Old items: mixture of non-recollected and recollected
      real source_d = (item_type == 2) ? dprime_S : dprime_S_B;
      
      // Non-recollected: at (dprime_F, 0)
      real p1_fam = univariate_cell_prob(y1, K1, dprime_F, sigma_item, c1);
      real p2_fam = univariate_cell_prob(y2, K2, 0.0, 1.0, c2);
      real p_fam = p1_fam * p2_fam;
      
      // Recollected: at (dprime_F + dprime_R, source_d)
      real p1_rec = univariate_cell_prob(y1, K1, dprime_F + dprime_R, sigma_item, c1);
      real p2_rec = univariate_cell_prob(y2, K2, source_d, sigma_S, c2);
      real p_rec = p1_rec * p2_rec;
      
      // Mixture
      p = (1.0 - R) * p_fam + R * p_rec;
    }
    
    return log(fmax(p, 1e-300));
  }')
  }
  
  if (isTRUE(model_data$threads) && is.null(batch_info)) {
    lines <- c(lines, "", generate_partial_log_lik(model_data, family))
  }

  if (!is.null(batch_info)) {
    lines <- c(lines, "", "  // Batch C++ likelihood (implemented in user header)", batch_info$stan_decl,
                   batch_info$stan_decl_vec)
    if (isTRUE(model_data$threads)) {
      lines <- c(lines, "", batch_info$stan_partial)
    }
  }

  lines <- c(lines, "}")
  paste(lines, collapse = "\n")
}

.plain_model_likelihood <- function(model_data, family, family_name) {
  if (isTRUE(model_data$threads)) {
    args <- build_reduce_sum_args(model_data, family)
    arg_names <- paste(args$names, collapse = ",\n      ")
    return(c("",
             "  // Threaded likelihood via reduce_sum",
             "  target += reduce_sum(partial_log_lik_lpmf, seq_n, grainsize,",
             paste0("      ", arg_names, ");")))
  }
  c("", "  for (n in 1:N) {", generate_likelihood_code(model_data, family_name), "  }")
}

.plain_gq_log_lik <- function(model_data, family_name) {
  ll_str <- paste(generate_likelihood_code(model_data, family_name), collapse = "\n")
  ll_str <- gsub("target \\+= ([^;]+);", "log_lik[n] = \\1;", ll_str)
  ll_str <- gsub("log_lik\\[n\\] = counts\\[n\\] \\* ", "log_lik[n] = ", ll_str)
  c("", "  vector[N] log_lik;", "  for (n in 1:N) {", strsplit(ll_str, "\n")[[1]], "  }")
}

#' Generate link CDF wrapper function
#' @noRd
generate_link_cdf_wrapper <- function(link) {
  link_name <- link$name
  
  if (link_name == "probit") {
    return('
  // Probit link (standard normal CDF)
  real link_cdf(real x) {
    return Phi(x);
  }

  real link_ccdf(real x) {
    return 0.5 * erfc(x * 0.7071067811865476);
  }

  real link_diff(real lo, real hi) {
    if (lo >= 0) return link_ccdf(lo) - link_ccdf(hi);
    return link_ccdf(-hi) - link_ccdf(-lo);
  }')
  } else if (link_name == "logit") {
    return('
  // Logit link (logistic CDF)
  real link_cdf(real x) {
    return inv_logit(x);
  }

  real link_ccdf(real x) {
    return inv_logit(-x);
  }

  real link_diff(real lo, real hi) {
    if (lo >= 0) return link_ccdf(lo) - link_ccdf(hi);
    return link_ccdf(-hi) - link_ccdf(-lo);
  }')
  } else if (link_name == "cloglog") {
    return('
  // Complementary log-log link (Gumbel-min CDF)
  real link_cdf(real x) {
    real xc = fmin(x, 700.0);
    return -expm1(-exp(xc));
  }

  real link_ccdf(real x) {
    return exp(-exp(fmin(x, 700.0)));
  }

  real link_diff(real lo, real hi) {
    if (lo >= -0.36651292058166435) return link_ccdf(lo) - link_ccdf(hi);
    return -expm1(-exp(fmin(hi, 700.0))) + expm1(-exp(fmin(lo, 700.0)));
  }')
  } else {
    stop("Unknown link function: ", link_name)
  }
}

#' Generate the partial_log_lik function for reduce_sum threading
#'
#' Wraps the existing per-observation likelihood code in a Stan function
#' that can be called by reduce_sum. Slices y directly (the response array)
#' and uses start/end to index into shared observation-level data.
#' @noRd
generate_partial_log_lik <- function(model_data, family) {
  family_name <- if (is.list(family)) family$family else family

  args <- build_reduce_sum_args(model_data, family)

  sig_args <- paste(args$declarations, collapse = ",\n      ")
  lines <- c(
    "  // Partial log-likelihood for reduce_sum threading",
    "  real partial_log_lik_lpmf(array[] int seq_n_slice, int start, int end,",
    paste0("      ", sig_args, ") {"),
    "    real lp = 0;",
    "    for (i in 1:size(seq_n_slice)) {",
    "      int n = seq_n_slice[i];"
  )

  ll_lines <- generate_likelihood_code(model_data, family_name)

  ll_lines <- gsub("target \\+=", "lp +=", ll_lines)
  ll_lines <- paste0("  ", ll_lines)

  lines <- c(lines, ll_lines)
  lines <- c(lines, "    }",
             "    return lp;",
             "  }")

  lines
}

#' Build shared argument lists for reduce_sum partial_log_lik
#'
#' Inspects model_data and family to enumerate all data arrays and parameters
#' that the per-observation likelihood loop accesses. Returns declarations for
#' the function signature and names for the reduce_sum call.
#' @noRd
build_reduce_sum_args <- function(model_data, family) {
  family_name <- if (is.list(family)) family$family else family
  is_cumulative <- family_name == "cumulative"
  is_source_mixture <- family_name == "source_mixture"
  is_bivariate <- family_name %in% c("bivariate_sdt", "bivariate_dp", "bivariate_cumulative")
  is_vrdp2d <- family_name == "vrdp2d"
  is_cdp <- family_name == "cdp"

  decls <- character(0)
  nms <- character(0)

  add <- function(decl, nm) {
    decls <<- c(decls, decl)
    nms <<- c(nms, nm)
  }

  # --- Response data (data-qualified to avoid autodiff copying) ---
  add("data array[] int y", "y")

  if (is_bivariate || is_vrdp2d) {
    add("data array[] int y2", "y2")
    add("data array[] int item_type", "item_type")
  } else if (is_cdp) {
    add("data array[] int rk", "rk")
    add("data array[] real is_old", "is_old")
  } else if (is_source_mixture) {
    add("data array[] int source", "source")
  } else if (!is_cumulative) {
    add("data array[] real is_old", "is_old")
  }

  if (isTRUE(model_data$has_counts)) {
    add("data array[] int counts", "counts")
  }

  # --- Scalars (data) ---
  if (is_bivariate || is_vrdp2d) {
    add("data int K1", "K1")
    add("data int K2", "K2")
    add("data int mid_thresh1", "mid_thresh1")
    if (isTRUE(model_data$has_criterion2)) {
      add("data int mid_thresh2", "mid_thresh2")
    }
  } else if (is_cdp) {
    add("data int K", "K")
    add("data int J", "J")
    add("data int n_rkg", "n_rkg")
    add("data int n_new", "n_new")
    add("data array[] int old_level_map", "old_level_map")
    add("data array[] int new_level_map", "new_level_map")
    add("data int mid_thresh_cdp", "mid_thresh_cdp")
  } else {
    add("data int K", "K")
    add("data int mid_thresh", "mid_thresh")
  }

  # --- Design matrices (data) & beta vectors (parameters) ---
  has_dprime_fixed <- isTRUE(model_data$has_dprime_fixed) || is.null(model_data$has_dprime_fixed)
  if (has_dprime_fixed) {
    add("data matrix X_dprime", "X_dprime")
    if (model_data$needs_ordered_dprime) {
      add("vector beta_dprime", "beta_dprime")
      add("vector beta_dprime2", "beta_dprime2")
    } else {
      add("vector beta_dprime", "beta_dprime")
    }
  }
  if (model_data$has_dprime2 && !model_data$needs_ordered_dprime) {
    add("data matrix X_dprime2", "X_dprime2")
    add("vector beta_dprime2", "beta_dprime2")
  } else if (model_data$needs_ordered_dprime && model_data$has_dprime2) {
    add("data matrix X_dprime2", "X_dprime2")
  }

  add("data matrix X_criterion", "X_criterion")

  add("vector beta_thresh_mid", "beta_thresh_mid")
  if (model_data$K > 2) {
    add("matrix beta_log_gaps", "beta_log_gaps")
  }

  param_list <- c("sigma", "lambda", "sigma2", "dprime_B", "lambda_B",
                   "discrim", "discrim_B", "sigma_B", "sigma2_B",
                   "rho", "rho_B", "rho_N", "lambda2", "lambda2_B",
                   "rec_crit", "know_crit",
                   "dprime_L", "sigma_L", "lambda_L")
  for (pname in param_list) {
    if (isTRUE(model_data[[paste0("has_", pname)]])) {
      add(sprintf("data matrix X_%s", pname), sprintf("X_%s", pname))
      if (isTRUE(model_data[[paste0("per_bin_", pname)]])) {
        add(sprintf("matrix beta_%s", pname), sprintf("beta_%s", pname))
      } else {
        add(sprintf("vector beta_%s", pname), sprintf("beta_%s", pname))
      }
    }
  }

  if (isTRUE(model_data$has_criterion2)) {
    if (isTRUE(model_data$varying_source_criteria)) {
      add("vector beta_thresh_mid_2_varying", "beta_thresh_mid_2_varying")
      add("matrix beta_log_gaps_2_varying", "beta_log_gaps_2_varying")
      add("data array[] int vbin", "vbin")
      if (.has_crit2_cov(model_data)) {
        add("data matrix X_criterion2_cov", "X_criterion2_cov")
        add("vector beta_thresh_mid_2_cov", "beta_thresh_mid_2_cov")
        add("matrix beta_log_gaps_2_cov", "beta_log_gaps_2_cov")
      }
      if (identical(model_data$new_source_criteria, "shared")) {
        add("real beta_thresh_mid_2_new", "beta_thresh_mid_2_new")
        add("vector beta_log_gaps_2_new", "beta_log_gaps_2_new")
        add("data array[] int is_new_response", "is_new_response")
      }
    } else {
      add("data matrix X_criterion2", "X_criterion2")
      add("vector beta_thresh_mid_2", "beta_thresh_mid_2")
      add("matrix beta_log_gaps_2", "beta_log_gaps_2")
    }
  }

  # --- Random effects ---
  all_groups <- get_all_groups(model_data)
  for (group in all_groups) {
    add(sprintf("data array[] int %s", group), group)
  }

  all_re_params <- c("dprime", "sigma", "lambda", "dprime2", "sigma2", "dprime_B", "lambda_B",
                      "discrim", "discrim_B", "sigma_B", "sigma2_B", "rho", "rho_B", "rho_N",
                      "lambda2", "lambda2_B",
                      "rec_crit", "know_crit", "dprime_L", "sigma_L", "lambda_L")
  for (pname in all_re_params) {
    re_list <- model_data[[paste0(pname, "_random")]]
    if (is.null(re_list)) next
    for (group in names(re_list)) {
      re <- re_list[[group]]
      if (!is.null(re$cor_id)) {
        var_name <- sprintf("u_%s_%s_from_%s", pname, group, re$cor_id)
      } else {
        var_name <- sprintf("u_%s_%s", pname, group)
      }
      if (re$dim == 1) {
        add(sprintf("vector %s", var_name), var_name)
      } else {
        add(sprintf("matrix %s", var_name), var_name)
      }
    }
  }

  for (group in names(model_data$criterion$random)) {
    re <- model_data$criterion$random[[group]]
    if (!is.null(re$cor_id)) {
      var_name <- sprintf("u_criterion_%s_from_%s", group, re$cor_id)
    } else {
      var_name <- sprintf("u_criterion_%s", group)
    }
    add(sprintf("matrix %s", var_name), var_name)
  }

  if (isTRUE(model_data$has_criterion2) && !is.null(model_data$criterion2$random)) {
    for (group in names(model_data$criterion2$random)) {
      re <- model_data$criterion2$random[[group]]
      if (!is.null(re$cor_id)) {
        var_name <- sprintf("u_criterion2_%s_from_%s", group, re$cor_id)
      } else {
        var_name <- sprintf("u_criterion2_%s", group)
      }
      if (re$dim == 1) {
        add(sprintf("vector %s", var_name), var_name)
      } else {
        add(sprintf("matrix %s", var_name), var_name)
      }
    }
  }

  term_indices <- get_all_term_indices(model_data)
  for (ti in term_indices) {
    add(sprintf("data array[] int %s", ti$idx_name), ti$idx_name)
  }

  z_matrices <- get_all_z_matrices(model_data)
  for (zm in z_matrices) {
    add(sprintf("data matrix %s", zm$name), zm$name)
  }

  if (!is.null(model_data$smooth_data)) {
    for (pname in names(model_data$smooth_data)) {
      for (sm in model_data$smooth_data[[pname]]) {
        for (comp in sm$components) {
          for (k in seq_along(comp$Zs_list)) {
            zs_name <- paste0("Zs_", pname, "_", comp$san_label, "_", k)
            s_name <- paste0("s_", pname, "_", comp$san_label, "_", k)
            add(sprintf("data matrix %s", zs_name), zs_name)
            add(sprintf("vector %s", s_name), s_name)
          }
        }
      }
    }
  }

  list(declarations = decls, names = nms)
}

generate_likelihood_code <- function(model_data, family_name) {
  lines <- character(0)

  has_encoding_params <- isTRUE(model_data$dprime_fixed$has_encoding) ||
    isTRUE(model_data$lambda_fixed$has_encoding) ||
    isTRUE(model_data$sigma_fixed$has_encoding)

  if (family_name == "bivariate_cumulative") {
    lines <- c(lines, generate_bivariate_cumulative_likelihood_code(model_data))
    return(lines)
  }

  if (family_name == "bivariate_sdt") {
    lines <- c(lines, generate_bivariate_likelihood_code(model_data))
    return(lines)
  }

  if (family_name == "bivariate_dp") {
    lines <- c(lines, generate_bivariate_dp_likelihood_code(model_data))
    return(lines)
  }

  if (family_name == "vrdp2d") {
    lines <- c(lines, generate_vrdp2d_likelihood_code(model_data))
    return(lines)
  }
  
  if (family_name == "cdp") {
    lines <- c(lines, generate_cdp_likelihood_code(model_data))
    return(lines)
  }

  if ((has_encoding_params || isTRUE(model_data$has_lure_mixture)) && family_name %in% c("mixture", "dpsdt")) {
    lines <- c(lines, generate_likelihood_with_encoding_branch(model_data, family_name))
  } else {
    lines <- c(lines, generate_param_lp("dprime", model_data$dprime_fixed, model_data$dprime_random, model_data$smooth_data[["dprime"]]))
    
    if (model_data$has_sigma) {
      lines <- c(lines, generate_param_lp("sigma", model_data$sigma_fixed, model_data$sigma_random, model_data$smooth_data[["sigma"]]))
      lines <- c(lines, "    real sigma_val = exp(sigma_n);")
    }
    
    if (model_data$has_lambda) {
      lines <- c(lines, generate_param_lp("lambda", model_data$lambda_fixed, model_data$lambda_random, model_data$smooth_data[["lambda"]]))
      lines <- c(lines, "    real lambda_val = inv_logit(lambda_n);")
    }
    
    if (model_data$has_dprime2 || model_data$needs_ordered_dprime) {
      lines <- c(lines, generate_param_lp("dprime2", model_data$dprime2_fixed, model_data$dprime2_random, model_data$smooth_data[["dprime2"]]))
      lines <- c(lines, "    real dprime2_val = dprime2_n;")
    } else if (family_name == "mixture") {
      if (model_data$has_sigma || model_data$has_sigma2) {
        lines <- c(lines, "    real dprime2_val = 0;")
      }
    }
    
    if (model_data$has_sigma2) {
      lines <- c(lines, generate_param_lp("sigma2", model_data$sigma2_fixed, model_data$sigma2_random, model_data$smooth_data[["sigma2"]]))
      lines <- c(lines, "    real sigma2_val = exp(sigma2_n);")
    }
    
    if (isTRUE(model_data$has_dprime_B)) {
      lines <- c(lines, generate_param_lp("dprime_B", model_data$dprime_B_fixed, model_data$dprime_B_random, model_data$smooth_data[["dprime_B"]]))
      lines <- c(lines, "    real dprime_B_val = dprime_B_n;")
    }
    
    if (isTRUE(model_data$has_lambda_B)) {
      lines <- c(lines, generate_param_lp("lambda_B", model_data$lambda_B_fixed, model_data$lambda_B_random, model_data$smooth_data[["lambda_B"]]))
      lines <- c(lines, "    real lambda_B_val = inv_logit(lambda_B_n);")
    }
    
    lines <- c(lines, generate_thresh_code(model_data))
    lines <- c(lines, generate_likelihood_call(model_data, family_name))
  }
  
  lines
}

#' Generate likelihood code for bivariate cumulative ordinal model
#' @noRd
generate_bivariate_cumulative_likelihood_code <- function(model_data) {
  lines <- character(0)

  lines <- c(lines, generate_param_lp_bivariate("dprime", model_data$dprime_fixed, model_data$dprime_random))

  lines <- c(lines, generate_param_lp_bivariate("discrim", model_data$discrim_fixed, model_data$discrim_random))

  lines <- c(lines, generate_param_lp_bivariate("rho", model_data$rho_fixed, model_data$rho_random))
  lines <- c(lines, "    real rho_val = tanh(rho_n);  // Fisher z to correlation")

  lines <- c(lines, generate_thresh_code_bivariate(model_data, 1))
  lines <- c(lines, generate_thresh_code_bivariate(model_data, 2))

  has_counts <- isTRUE(model_data$has_counts)
  ll_expr <- "bivariate_cumulative_lpmf_custom(y[n], y2[n], K1, K2, dprime_n, discrim_n, thresh1, thresh2, rho_val)"

  if (has_counts) {
    lines <- c(lines, "", sprintf("    target += counts[n] * %s;", ll_expr))
  } else {
    lines <- c(lines, "", sprintf("    target += %s;", ll_expr))
  }

  lines
}

#' Generate likelihood code for bivariate SDT
#' @noRd
generate_bivariate_likelihood_code <- function(model_data) {
  lines <- character(0)
  
  lines <- c(lines, generate_param_lp_bivariate("dprime", model_data$dprime_fixed, model_data$dprime_random))
  
  if (isTRUE(model_data$has_dprime_B)) {
    lines <- c(lines, generate_param_lp_bivariate("dprime_B", model_data$dprime_B_fixed, model_data$dprime_B_random))
    lines <- c(lines, "    real dprime_B_val = dprime_B_n;")
  } else {
    lines <- c(lines, "    real dprime_B_val = dprime_n;  // constrained equal to dprime")
  }
  
  lines <- c(lines, generate_param_lp_bivariate("discrim", model_data$discrim_fixed, model_data$discrim_random))
  
  if (isTRUE(model_data$has_discrim_B)) {
    lines <- c(lines, generate_param_lp_bivariate("discrim_B", model_data$discrim_B_fixed, model_data$discrim_B_random))
    lines <- c(lines, "    real discrim_B_val = discrim_B_n;")
  } else if (isTRUE(model_data$bounded)) {
    lines <- c(lines, "    real discrim_B_val = discrim_n;  // bounded symmetric: same magnitude")
  } else {
    lines <- c(lines, "    real discrim_B_val = -discrim_n;  // unbounded symmetric: mirror across origin")
  }
  
  if (isTRUE(model_data$has_sigma)) {
    lines <- c(lines, generate_param_lp_bivariate("sigma", model_data$sigma_fixed, model_data$sigma_random))
    lines <- c(lines, "    real sigma1_A_val = exp(sigma_n);")
  } else {
    lines <- c(lines, "    real sigma1_A_val = 1;  // fixed")
  }
  
  if (isTRUE(model_data$has_sigma_B)) {
    lines <- c(lines, generate_param_lp_bivariate("sigma_B", model_data$sigma_B_fixed, model_data$sigma_B_random))
    lines <- c(lines, "    real sigma1_B_val = exp(sigma_B_n);")
  } else if (isTRUE(model_data$has_sigma)) {
    lines <- c(lines, "    real sigma1_B_val = sigma1_A_val;  // constrained equal to sigma")
  } else {
    lines <- c(lines, "    real sigma1_B_val = 1;  // fixed")
  }
  
  if (isTRUE(model_data$has_sigma2)) {
    lines <- c(lines, generate_param_lp_bivariate("sigma2", model_data$sigma2_fixed, model_data$sigma2_random))
    lines <- c(lines, "    real sigma2_A_val = exp(sigma2_n);")
  } else {
    lines <- c(lines, "    real sigma2_A_val = 1;  // fixed")
  }
  
  if (isTRUE(model_data$has_sigma2_B)) {
    lines <- c(lines, generate_param_lp_bivariate("sigma2_B", model_data$sigma2_B_fixed, model_data$sigma2_B_random))
    lines <- c(lines, "    real sigma2_B_val = exp(sigma2_B_n);")
  } else if (isTRUE(model_data$has_sigma2)) {
    lines <- c(lines, "    real sigma2_B_val = sigma2_A_val;  // constrained equal to sigma2")
  } else {
    lines <- c(lines, "    real sigma2_B_val = 1;  // fixed")
  }
  
  lines <- c(lines, generate_param_lp_bivariate("rho", model_data$rho_fixed, model_data$rho_random))
  lines <- c(lines, "    real rho_A_val = tanh(rho_n);  // Fisher z to correlation")
  
  if (isTRUE(model_data$has_rho_B)) {
    lines <- c(lines, generate_param_lp_bivariate("rho_B", model_data$rho_B_fixed, model_data$rho_B_random))
    lines <- c(lines, "    real rho_B_val = tanh(rho_B_n);")
  } else if (isTRUE(model_data$bounded)) {
    lines <- c(lines, "    real rho_B_val = rho_A_val;  // bounded symmetric: same magnitude")
  } else {
    lines <- c(lines, "    real rho_B_val = -rho_A_val;  // unbounded symmetric: opposite sign")
  }
  
  if (isTRUE(model_data$has_rho_N)) {
    lines <- c(lines, generate_param_lp_bivariate("rho_N", model_data$rho_N_fixed, model_data$rho_N_random))
    lines <- c(lines, "    real rho_N_val = tanh(rho_N_n);")
  } else {
    lines <- c(lines, "    real rho_N_val = 0;  // fixed at 0 for new items")
  }
  
  lines <- c(lines, generate_thresh_code_bivariate(model_data, 1))
  
  varying_source_criteria <- isTRUE(model_data$varying_source_criteria)

  if (varying_source_criteria) {
    lines <- c(lines, generate_thresh_code_vrdp2d_varying(model_data))
  } else {
    lines <- c(lines, generate_thresh_code_bivariate(model_data, 2))
  }

  new_shared <- identical(model_data$new_source_criteria, "shared")

  if (new_shared) {
    lines <- c(lines,
      "    // Shared source thresholds for new items (mid-anchor)",
      "    vector[K2-1] thresh2_new;",
      "    thresh2_new[mid_thresh2] = beta_thresh_mid_2_new;",
      "    for (k2 in (mid_thresh2+1):(K2-1)) {",
      "      thresh2_new[k2] = thresh2_new[k2-1] + exp(beta_log_gaps_2_new[k2 - mid_thresh2]);",
      "    }",
      "    for (k2_down in 1:(mid_thresh2-1)) {",
      "      int k2 = mid_thresh2 - k2_down;",
      "      thresh2_new[k2] = thresh2_new[k2+1] - exp(beta_log_gaps_2_new[K2 - 1 - mid_thresh2 + k2_down]);",
      "    }")
  }

  has_counts <- isTRUE(model_data$has_counts)
  bounded <- isTRUE(model_data$bounded)

  if (bounded) {
    dprime_idx <- grep("real dprime_n = ", lines, fixed = TRUE)
    if (length(dprime_idx) > 0) {
      lines[dprime_idx] <- gsub(
        "real dprime_n = (.+);",
        "real dprime_n = exp(\\1);  // log link: mu_I > 0 (bounded)",
        lines[dprime_idx]
      )
    }
    dprime_B_idx <- grep("real dprime_B_val = dprime_B_n;", lines, fixed = TRUE)
    if (length(dprime_B_idx) > 0) {
      lines[dprime_B_idx] <- "    real dprime_B_val = exp(dprime_B_n);  // log link: mu_I_B > 0 (bounded)"
    }

    discrim_idx <- grep("real discrim_n = ", lines, fixed = TRUE)
    if (length(discrim_idx) > 0) {
      lines[discrim_idx] <- gsub(
        "real discrim_n = (.+);",
        "real discrim_n = exp(\\1);  // log link: mu_S > 0 (bounded)",
        lines[discrim_idx]
      )
    }
    discrim_B_idx <- grep("real discrim_B_val = discrim_B_n;", lines, fixed = TRUE)
    if (length(discrim_B_idx) > 0) {
      lines[discrim_B_idx] <- "    real discrim_B_val = exp(discrim_B_n);  // log link: mu_S_B > 0 (bounded)"
    }

    rho_idx <- grep("real rho_A_val = tanh(rho_n)", lines, fixed = TRUE)
    if (length(rho_idx) > 0) {
      lines[rho_idx] <- "    real rho_A_val = inv_logit(rho_n);  // logistic link: rho_A in (0,1) (bounded)"
    }
    rho_B_idx <- grep("real rho_B_val = tanh(rho_B_n);", lines, fixed = TRUE)
    if (length(rho_B_idx) > 0) {
      lines[rho_B_idx] <- "    real rho_B_val = inv_logit(rho_B_n);  // logistic link: rho_B in (0,1) (bounded)"
    }
  }

  if (new_shared && bounded) {
    lpmf_name <- "bivariate_sdt_varying_bounded_new_shared_lpmf_custom"
    thresh_arg <- "thresh1, thresh2_varying, thresh2_new, is_new_response"
  } else if (new_shared) {
    lpmf_name <- "bivariate_sdt_varying_new_shared_lpmf_custom"
    thresh_arg <- "thresh1, thresh2_varying, thresh2_new, is_new_response"
  } else if (varying_source_criteria && bounded) {
    lpmf_name <- "bivariate_sdt_varying_bounded_lpmf_custom"
    thresh_arg <- "thresh1, thresh2_varying"
  } else if (varying_source_criteria) {
    lpmf_name <- "bivariate_sdt_varying_lpmf_custom"
    thresh_arg <- "thresh1, thresh2_varying"
  } else if (bounded) {
    lpmf_name <- "bivariate_sdt_bounded_lpmf_custom"
    thresh_arg <- "thresh1, thresh2"
  } else {
    lpmf_name <- "bivariate_sdt_lpmf_custom"
    thresh_arg <- "thresh1, thresh2"
  }

  ll_expr <- paste0(
    lpmf_name, "(y[n], y2[n], item_type[n], K1, K2,\n",
    "                                  dprime_n, dprime_B_val, discrim_n, discrim_B_val,\n",
    "                                  sigma1_A_val, sigma1_B_val, sigma2_A_val, sigma2_B_val,\n",
    "                                  rho_A_val, rho_B_val, rho_N_val,\n",
    "                                  ", thresh_arg, ")"
  )
  
  if (has_counts) {
    lines <- c(lines, "", sprintf("    target += counts[n] * %s;", ll_expr))
  } else {
    lines <- c(lines, "", sprintf("    target += %s;", ll_expr))
  }
  
  lines
}

#' Generate likelihood code for bivariate DP model
#' @noRd
generate_bivariate_dp_likelihood_code <- function(model_data) {
  lines <- generate_bivariate_likelihood_code(model_data)

  target_idx <- grep("target \\+=", lines)
  if (length(target_idx) > 0) {
    remove_start <- max(1, target_idx[length(target_idx)] - 1)
    lines <- lines[1:(remove_start - 1)]
  }

  lines <- c(lines, generate_param_lp_bivariate("lambda", model_data$lambda_fixed, model_data$lambda_random))
  lines <- c(lines, "    real R_I = inv_logit(lambda_n);")

  if (isTRUE(model_data$has_lambda_B)) {
    lines <- c(lines, generate_param_lp_bivariate("lambda_B", model_data$lambda_B_fixed, model_data$lambda_B_random))
    lines <- c(lines, "    real R_I_B = inv_logit(lambda_B_n);")
  } else {
    lines <- c(lines, "    real R_I_B = R_I;  // constrained equal to lambda")
  }

  lines <- c(lines, generate_param_lp_bivariate("lambda2", model_data$lambda2_fixed, model_data$lambda2_random))
  lines <- c(lines, "    real R_S = inv_logit(lambda2_n);")

  if (isTRUE(model_data$has_lambda2_B)) {
    lines <- c(lines, generate_param_lp_bivariate("lambda2_B", model_data$lambda2_B_fixed, model_data$lambda2_B_random))
    lines <- c(lines, "    real R_S_B = inv_logit(lambda2_B_n);")
  } else {
    lines <- c(lines, "    real R_S_B = R_S;  // constrained equal to lambda2")
  }

  has_counts <- isTRUE(model_data$has_counts)
  varying <- isTRUE(model_data$varying_source_criteria)
  new_shared <- identical(model_data$new_source_criteria, "shared")

  if (new_shared) {
    lpmf_name <- "bivariate_dp_varying_new_shared_lpmf_custom"
    thresh_arg <- "thresh1, thresh2_varying, thresh2_new, is_new_response"
  } else if (varying) {
    lpmf_name <- "bivariate_dp_varying_lpmf_custom"
    thresh_arg <- "thresh1, thresh2_varying"
  } else {
    lpmf_name <- "bivariate_dp_lpmf_custom"
    thresh_arg <- "thresh1, thresh2"
  }

  ll_expr <- paste0(
    lpmf_name, "(y[n], y2[n], item_type[n], K1, K2,\n",
    "                                  dprime_n, dprime_B_val, discrim_n, discrim_B_val,\n",
    "                                  sigma1_A_val, sigma1_B_val, sigma2_A_val, sigma2_B_val,\n",
    "                                  rho_A_val, rho_B_val, rho_N_val,\n",
    "                                  R_I, R_S, R_I_B, R_S_B,\n",
    "                                  ", thresh_arg, ")"
  )

  if (has_counts) {
    lines <- c(lines, "", sprintf("    target += counts[n] * %s;", ll_expr))
  } else {
    lines <- c(lines, "", sprintf("    target += %s;", ll_expr))
  }

  lines
}

#' Generate likelihood code for CDP model
#' @noRd
generate_cdp_likelihood_code <- function(model_data) {
  lines <- character(0)
  
  lines <- c(lines, generate_param_lp("dprime", model_data$dprime_fixed, model_data$dprime_random, model_data$smooth_data[["dprime"]]))
  lines <- c(lines, "    real mu_R = dprime_n;")
  
  if (model_data$has_dprime2) {
    lines <- c(lines, generate_param_lp("dprime2", model_data$dprime2_fixed, model_data$dprime2_random, model_data$smooth_data[["dprime2"]]))
    lines <- c(lines, "    real mu_F = dprime2_n;")
  } else {
    lines <- c(lines, "    real mu_F = 0;")
  }
  
  if (model_data$has_sigma) {
    lines <- c(lines, generate_param_lp("sigma", model_data$sigma_fixed, model_data$sigma_random, model_data$smooth_data[["sigma"]]))
    lines <- c(lines, "    real sigma_R = exp(sigma_n);")
  } else {
    lines <- c(lines, "    real sigma_R = 1;")
  }
  
  if (model_data$has_sigma2) {
    lines <- c(lines, generate_param_lp("sigma2", model_data$sigma2_fixed, model_data$sigma2_random, model_data$smooth_data[["sigma2"]]))
    lines <- c(lines, "    real sigma_F = exp(sigma2_n);")
  } else {
    lines <- c(lines, "    real sigma_F = 1;")
  }
  
  if (isTRUE(model_data$has_rec_crit)) {
    lines <- c(lines, generate_param_lp("rec_crit", model_data$rec_crit_fixed, model_data$rec_crit_random, model_data$smooth_data[["rec_crit"]]))
    lines <- c(lines, "    real c_R = rec_crit_n;")
  } else {
    lines <- c(lines, "    real c_R = 1.5;")
  }

  n_rkg <- model_data$stan_data$n_rkg
  if (isTRUE(model_data$has_know_crit)) {
    lines <- c(lines, generate_param_lp("know_crit", model_data$know_crit_fixed, model_data$know_crit_random, model_data$smooth_data[["know_crit"]]))
    lines <- c(lines, "    real c_K = know_crit_n;")
  }

  lines <- c(lines, "",
             "    // Derived CDP parameters",
             "    real mu_M = mu_R + mu_F;  // Memory strength mean for targets",
             "    real sigma_M = sqrt(square(sigma_R) + square(sigma_F));  // Memory strength SD",
             "    real rho = sigma_R / sigma_M;  // Correlation between R and M",
             "",
             "    // Lure parameters (fixed reference)",
             "    real sigma_M_l = sqrt(2.0);",
             "    real rho_l = 1.0 / sqrt(2.0);")

  lines <- c(lines, "", generate_thresh_code_cdp(model_data))

  has_counts <- isTRUE(model_data$has_counts)

  lines <- c(lines, "",
             "    // CDP likelihood",
             "    real ll;",
             "    int is_old_resp = 0;",
             "    int old_idx = 0;",
             "    for (j in 1:J) {",
             "      if (y[n] == old_level_map[j]) {",
             "        is_old_resp = 1;",
             "        old_idx = j;",
             "      }",
             "    }")

  lines <- c(lines, "",
             "    real tau_lo = (is_old_resp == 1) ? tau[old_idx] : negative_infinity();",
             "    real tau_hi = (is_old_resp == 1) ? ((old_idx == J) ? positive_infinity() : tau[old_idx + 1]) : tau[1];",
             "    real mR = (is_old[n] == 1) ? mu_R : 0;",
             "    real sR = (is_old[n] == 1) ? sigma_R : 1;",
             "    real mF = (is_old[n] == 1) ? mu_F : 0;",
             "    real sF = (is_old[n] == 1) ? sigma_F : 1;",
             "    real mM = mR + mF;",
             "    real sM = sqrt(square(sR) + square(sF));",
             "    real z_tau_lo = bvn_std(tau_lo, mM, sM);",
             "    real z_tau_hi = bvn_std(tau_hi, mM, sM);",
             "    if (rk[n] == 0) {",
             "      ll = bvn_log_interval(z_tau_lo, z_tau_hi);",
             if (isTRUE(n_rkg == 3)) c(
             "    } else {",
             "      vector[3] rkg = compute_rkg_probs(mR, sR, mF, sF, c_R, c_K, tau_lo, tau_hi);",
             "      ll = log(fmax(rkg[rk[n]], 1e-300));",
             "    }") else c(
             "    } else if (rk[n] == 1) {",
             "      ll = log(binormal_strip_upper((c_R - mR) / sR, z_tau_lo, z_tau_hi, sR / sM));",
             "    } else {",
             "      ll = log(binormal_strip_lower((c_R - mR) / sR, z_tau_lo, z_tau_hi, sR / sM));",
             "    }"))

  lines <- c(lines,
             if (has_counts) "    target += counts[n] * ll;" else "    target += ll;")

  lines
}

#' Generate likelihood code for 2D-VRDP model
#' @noRd
generate_vrdp2d_likelihood_code <- function(model_data) {
  lines <- character(0)
  
  varying_source_criteria <- isTRUE(model_data$varying_source_criteria)
  
  lines <- c(lines, generate_param_lp_bivariate("dprime", model_data$dprime_fixed, model_data$dprime_random))
  lines <- c(lines, "    real dprime_F = dprime_n;")
  
  lines <- c(lines, generate_param_lp_bivariate("dprime2", model_data$dprime2_fixed, model_data$dprime2_random))
  lines <- c(lines, "    real dprime_R = dprime2_n;")
  
  lines <- c(lines, generate_param_lp_bivariate("discrim", model_data$discrim_fixed, model_data$discrim_random))
  lines <- c(lines, "    real dprime_S = discrim_n;")
  
  if (isTRUE(model_data$has_discrim_B)) {
    lines <- c(lines, generate_param_lp_bivariate("discrim_B", model_data$discrim_B_fixed, model_data$discrim_B_random))
    lines <- c(lines, "    real dprime_S_B = discrim_B_n;")
  } else {
    lines <- c(lines, "    real dprime_S_B = -dprime_S;  // symmetric: d'_S_B = -d'_S")
  }
  
  lines <- c(lines, generate_param_lp_bivariate("lambda", model_data$lambda_fixed, model_data$lambda_random))
  lines <- c(lines, "    real R = inv_logit(lambda_n);")
  
  if (isTRUE(model_data$has_sigma)) {
    lines <- c(lines, generate_param_lp_bivariate("sigma", model_data$sigma_fixed, model_data$sigma_random))
    lines <- c(lines, "    real sigma_item = exp(sigma_n);")
  } else {
    lines <- c(lines, "    real sigma_item = 1;  // fixed")
  }
  
  if (isTRUE(model_data$has_sigma2)) {
    lines <- c(lines, generate_param_lp_bivariate("sigma2", model_data$sigma2_fixed, model_data$sigma2_random))
    lines <- c(lines, "    real sigma_S = exp(sigma2_n);")
  } else {
    lines <- c(lines, "    real sigma_S = 1;  // fixed")
  }
  
  lines <- c(lines, generate_thresh_code_bivariate(model_data, 1))
  
  if (varying_source_criteria) {
    lines <- c(lines, generate_thresh_code_vrdp2d_varying(model_data))
  } else {
    lines <- c(lines, generate_thresh_code_bivariate(model_data, 2))
  }
  
  has_counts <- isTRUE(model_data$has_counts)
  
  if (varying_source_criteria) {
    ll_expr <- paste0(
      "vrdp2d_varying_lpmf_custom(y[n], y2[n], item_type[n], K1, K2,\n",
      "                                  dprime_F, dprime_R, dprime_S, dprime_S_B,\n",
      "                                  R, sigma_item, sigma_S,\n",
      "                                  thresh1, thresh2_varying)"
    )
  } else {
    ll_expr <- paste0(
      "vrdp2d_lpmf_custom(y[n], y2[n], item_type[n], K1, K2,\n",
      "                                  dprime_F, dprime_R, dprime_S, dprime_S_B,\n",
      "                                  R, sigma_item, sigma_S,\n",
      "                                  thresh1, thresh2)"
    )
  }
  
  if (has_counts) {
    lines <- c(lines, "", sprintf("    target += counts[n] * %s;", ll_expr))
  } else {
    lines <- c(lines, "", sprintf("    target += %s;", ll_expr))
  }
  
  lines
}

#' Generate threshold code for vrdp2d with varying source criteria
#' @noRd
generate_thresh_code_vrdp2d_varying <- function(model_data) {

  varying_re_mode <- if (!is.null(model_data$varying_re)) model_data$varying_re else "shared"
  re_list <- if (!is.null(model_data$criterion2)) model_data$criterion2$random else NULL

  mid_terms <- character(0)
  gap_terms <- character(0)
  for (group in names(re_list)) {
    re <- re_list[[group]]
    v <- if (!is.null(re$cor_id)) {
      sprintf("u_criterion2_%s_from_%s", group, re$cor_id)
    } else {
      sprintf("u_criterion2_%s", group)
    }
    R <- re$n_re_terms %||% 1L
    at <- function(base0) {
      if (isTRUE(re$use_z_matrix)) {
        sprintf("Z_criterion2_%s[n, ] * %s[%s[n], ((%s)*%d+1):((%s)*%d+%d)]'", group, v, group, base0, R, base0, R, R)
      } else if (re$dim == 1) {
        sprintf("%s[%s[n]]", v, group)
      } else {
        sprintf("%s[%s[n], (%s)+1]", v, group, base0)
      }
    }
    if (varying_re_mode == "full") {
      mid_terms <- c(mid_terms, at("(vbin[k1]-1)*(K2-1)+mid_thresh2-1"))
      gap_terms <- c(gap_terms, at("(vbin[k1]-1)*(K2-1)+k2-1"))
    } else if (varying_re_mode == "per_bin" && re$dim > R) {
      mid_terms <- c(mid_terms, at("vbin[k1]-1"))
    } else {
      mid_terms <- c(mid_terms, at("0"))
    }
  }
  has_cov <- .has_crit2_cov(model_data)
  sm_terms <- function(row_expr) {
    out <- character(0)
    for (sm in model_data$smooth_data[["criterion2"]]) {
      for (comp in sm$components) {
        for (ki in seq_along(comp$Zs_list)) {
          out <- c(out, sprintf("Zs_criterion2_%s_%d[n, ] * s_criterion2_%s_%d[%s, ]'",
                                comp$san_label, ki, comp$san_label, ki, row_expr))
        }
      }
    }
    out
  }
  if (has_cov) mid_terms <- c(mid_terms, "X_criterion2_cov[n, ] * beta_thresh_mid_2_cov")
  mid_terms <- c(mid_terms, sm_terms("1"))
  up_terms <- c(gap_terms,
                if (has_cov) "X_criterion2_cov[n, ] * beta_log_gaps_2_cov[k2-mid_thresh2, ]'",
                sm_terms("1 + k2-mid_thresh2"))
  down_terms <- c(gap_terms,
                  if (has_cov) "X_criterion2_cov[n, ] * beta_log_gaps_2_cov[K2-1-mid_thresh2+k2_down, ]'",
                  sm_terms("1 + K2-1-mid_thresh2+k2_down"))
  plus <- function(x) if (length(x)) paste0(" + ", paste(x, collapse = " + ")) else ""

  c(
    "    // Source thresholds varying by item response (mid-anchor)",
    "    matrix[K1, K2-1] thresh2_varying;",
    "    for (k1 in 1:K1) {",
    sprintf("      thresh2_varying[k1, mid_thresh2] = beta_thresh_mid_2_varying[vbin[k1]]%s;",
            plus(mid_terms)),
    "      for (k2 in (mid_thresh2+1):(K2-1)) {",
    sprintf("        thresh2_varying[k1, k2] = thresh2_varying[k1, k2-1] + exp(beta_log_gaps_2_varying[vbin[k1], k2-mid_thresh2]%s);",
            plus(up_terms)),
    "      }",
    "      for (k2_down in 1:(mid_thresh2-1)) {",
    "        int k2 = mid_thresh2 - k2_down;",
    sprintf("        thresh2_varying[k1, k2] = thresh2_varying[k1, k2+1] - exp(beta_log_gaps_2_varying[vbin[k1], K2-1-mid_thresh2+k2_down]%s);",
            plus(down_terms)),
    "      }",
    "    }"
  )
}

#' Generate threshold code for bivariate SDT (two sets of thresholds)
#' @noRd
generate_thresh_code_bivariate <- function(model_data, dimension) {
  if (dimension == 1) {
    crit <- model_data$criterion
    K <- model_data$K
    K_var <- "K1"
    thresh_name <- "thresh1"
    beta_thresh <- "beta_thresh_mid"
    beta_log_gaps <- "beta_log_gaps"
    mid_var <- "mid_thresh1"
    crit_name <- "criterion"
  } else {
    crit <- model_data$criterion2
    K <- model_data$K2
    K_var <- "K2"
    thresh_name <- "thresh2"
    beta_thresh <- "beta_thresh_mid_2"
    beta_log_gaps <- "beta_log_gaps_2"
    mid_var <- "mid_thresh2"
    crit_name <- "criterion2"
  }

  if (is.null(crit)) {
    stop("Criterion structure not found for dimension ", dimension)
  }

  is_intercept_only <- isTRUE(crit$is_intercept_only) || crit$n_coef == 1

  gen_re_biv <- function(target, k_expr) {
    re_lines <- character(0)
    if (!is.null(crit$random)) {
      for (group in names(crit$random)) {
        re <- crit$random[[group]]
        var_name <- if (!is.null(re$cor_id)) {
          sprintf("u_%s_%s_from_%s", crit_name, group, re$cor_id)
        } else {
          sprintf("u_%s_%s", crit_name, group)
        }
        n_re_terms <- if (!is.null(re$n_re_terms)) re$n_re_terms else re$n_cond_levels

        if (isTRUE(re$use_z_matrix)) {
          z_name <- paste0("Z_", crit_name, "_", group)
          re_lines <- c(re_lines, sprintf("    %s += %s[n, ] * %s[%s[n], ((%s-1)*%d+1):(%s*%d)]';",
                                          target, z_name, var_name, group,
                                          k_expr, n_re_terms, k_expr, n_re_terms))
        } else if (re$dim == 1) {
          re_lines <- c(re_lines, sprintf("    %s += %s[%s[n]];", target, var_name, group))
        } else if (!is.null(re$term_idx)) {
          idx_var <- paste0("idx_", crit_name, "_", group)
          re_lines <- c(re_lines, sprintf("    if (%s[n] > 0) %s += %s[%s[n], (%s-1) * %d + %s[n]];",
                                          idx_var, target, var_name, group, k_expr, n_re_terms, idx_var))
        } else {
          re_lines <- c(re_lines, sprintf("    %s += %s[%s[n], %s];", target, var_name, group, k_expr))
        }
      }
    }
    re_lines
  }

  lines <- c(sprintf("    vector[%s-1] %s;", K_var, thresh_name))

  if (is_intercept_only) {
    lines <- c(lines, sprintf("    %s[%s] = %s[1];", thresh_name, mid_var, beta_thresh))
  } else {
    lines <- c(lines, sprintf("    %s[%s] = X_%s[n, ] * %s;", thresh_name, mid_var, crit_name, beta_thresh))
  }
  lines <- c(lines, gen_re_biv(sprintf("%s[%s]", thresh_name, mid_var), mid_var))

  smooth_key <- if (dimension == 1) "criterion" else "criterion2"
  if (!is.null(model_data$smooth_data[[smooth_key]])) {
    for (sm in model_data$smooth_data[[smooth_key]]) {
      for (comp in sm$components) {
        for (ki in seq_along(comp$Zs_list)) {
          zs_name <- paste0("Zs_", smooth_key, "_", comp$san_label, "_", ki)
          coef_name <- paste0("s_", smooth_key, "_", comp$san_label, "_", ki)
          lines <- c(lines, sprintf("    %s[%s] += %s[n, ] * %s[1, ]';", thresh_name, mid_var, zs_name, coef_name))
        }
      }
    }
  }

  lines <- c(lines, sprintf("    for (k in (%s+1):(%s-1)) {", mid_var, K_var))

  if (is_intercept_only) {
    lines <- c(lines, sprintf("      real log_gap = %s[k - %s, 1];", beta_log_gaps, mid_var))
  } else {
    lines <- c(lines, sprintf("      real log_gap = X_%s[n, ] * %s[k - %s, ]';", crit_name, beta_log_gaps, mid_var))
  }
  lines <- c(lines, gen_re_biv("log_gap", "k"))
  if (!is.null(model_data$smooth_data[[smooth_key]])) {
    for (sm in model_data$smooth_data[[smooth_key]]) {
      for (comp in sm$components) {
        for (ki in seq_along(comp$Zs_list)) {
          zs_name <- paste0("Zs_", smooth_key, "_", comp$san_label, "_", ki)
          coef_name <- paste0("s_", smooth_key, "_", comp$san_label, "_", ki)
          lines <- c(lines, sprintf("      log_gap += %s[n, ] * %s[1 + k - %s, ]';", zs_name, coef_name, mid_var))
        }
      }
    }
  }
  lines <- c(lines, sprintf("      %s[k] = %s[k-1] + exp(log_gap);", thresh_name, thresh_name), "    }")

  lines <- c(lines, sprintf("    for (k_down in 1:(%s-1)) {", mid_var),
             sprintf("      int k = %s - k_down;", mid_var))

  if (is_intercept_only) {
    lines <- c(lines, sprintf("      real log_gap = %s[%s - 1 - %s + k_down, 1];", beta_log_gaps, K_var, mid_var))
  } else {
    lines <- c(lines, sprintf("      real log_gap = X_%s[n, ] * %s[%s - 1 - %s + k_down, ]';", crit_name, beta_log_gaps, K_var, mid_var))
  }
  lines <- c(lines, gen_re_biv("log_gap", "k"))
  if (!is.null(model_data$smooth_data[[smooth_key]])) {
    for (sm in model_data$smooth_data[[smooth_key]]) {
      for (comp in sm$components) {
        for (ki in seq_along(comp$Zs_list)) {
          zs_name <- paste0("Zs_", smooth_key, "_", comp$san_label, "_", ki)
          coef_name <- paste0("s_", smooth_key, "_", comp$san_label, "_", ki)
          lines <- c(lines, sprintf("      log_gap += %s[n, ] * %s[1 + %s - 1 - %s + k_down, ]';", zs_name, coef_name, K_var, mid_var))
        }
      }
    }
  }
  lines <- c(lines, sprintf("      %s[k] = %s[k+1] - exp(log_gap);", thresh_name, thresh_name), "    }")

  lines
}

#' Generate parameter linear predictor for bivariate SDT (no is_old branching)
#' @noRd
generate_param_lp_bivariate <- function(param_name, fixed_info, random_info) {
  lines <- character(0)
  
  has_fixed <- !is.null(fixed_info) && fixed_info$n_coef > 0
  
  if (!has_fixed) {
    lines <- c(lines, sprintf("    real %s_n = 0;", param_name))
  } else {
    lines <- c(lines, sprintf("    real %s_n = X_%s[n, ] * beta_%s;", 
                              param_name, param_name, param_name))
  }
  
  if (!is.null(random_info)) {
    for (group in names(random_info)) {
      re <- random_info[[group]]
      var_name <- if (!is.null(re$cor_id)) {
        sprintf("u_%s_%s_from_%s", param_name, group, re$cor_id)
      } else {
        sprintf("u_%s_%s", param_name, group)
      }
      
      if (isTRUE(re$use_z_matrix)) {
        z_name <- paste0("Z_", param_name, "_", group)
        lines <- c(lines, sprintf("    %s_n += %s[n, ] * %s[%s[n], ]';", param_name, z_name, var_name, group))
      } else if (re$dim == 1) {
        if (!is.null(re$term_idx)) {
          idx_var <- paste0("idx_", param_name, "_", group)
          lines <- c(lines, sprintf("    if (%s[n] > 0) %s_n += %s[%s[n]];", idx_var, param_name, var_name, group))
        } else {
          lines <- c(lines, sprintf("    %s_n += %s[%s[n]];", param_name, var_name, group))
        }
      } else if (!is.null(re$term_idx)) {
        idx_var <- paste0("idx_", param_name, "_", group)
        lines <- c(lines, sprintf("    if (%s[n] > 0) %s_n += %s[%s[n], %s[n]];", idx_var, param_name, var_name, group, idx_var))
      }
    }
  }

  lines
}

#' Generate likelihood with branching for encoding variables
#' This avoids computing dprime/lambda for new items
#' @noRd
generate_likelihood_with_encoding_branch <- function(model_data, family_name) {
  lines <- character(0)
  has_counts <- isTRUE(model_data$has_counts)
  
  lines <- c(lines, generate_thresh_code(model_data))
  
  lines <- c(lines, "", "    if (is_old[n] == 1) {")
  
  lines <- c(lines, generate_param_lp_old_only("dprime", model_data$dprime_fixed, model_data$dprime_random, model_data$smooth_data[["dprime"]]))
  
  if (model_data$has_sigma) {
    lines <- c(lines, generate_param_lp_old_only("sigma", model_data$sigma_fixed, model_data$sigma_random, model_data$smooth_data[["sigma"]]))
    lines <- c(lines, "      real sigma_val = exp(sigma_n);")
  }
  
  if (model_data$has_lambda) {
    lines <- c(lines, generate_param_lp_old_only("lambda", model_data$lambda_fixed, model_data$lambda_random, model_data$smooth_data[["lambda"]]))
    lines <- c(lines, "      real lambda_val = inv_logit(lambda_n);")
  }
  
  if (model_data$has_dprime2 || model_data$needs_ordered_dprime) {
    lines <- c(lines, generate_param_lp_old_only("dprime2", model_data$dprime2_fixed, model_data$dprime2_random, model_data$smooth_data[["dprime2"]]))
  }
  
  if (model_data$has_sigma2) {
    lines <- c(lines, generate_param_lp_old_only("sigma2", model_data$sigma2_fixed, model_data$sigma2_random, model_data$smooth_data[["sigma2"]]))
    lines <- c(lines, "      real sigma2_val = exp(sigma2_n);")
  }
  
  count_mult <- if (has_counts) "counts[n] * " else ""
  
  if (family_name == "dpsdt") {
    if (model_data$has_sigma) {
      lines <- c(lines, sprintf("      target += %sdpsdt_lpmf(y[n] | dprime_n, sigma_val, lambda_val, 1, K, thresh);", count_mult))
    } else {
      lines <- c(lines, sprintf("      target += %sdpsdt_lpmf(y[n] | dprime_n, lambda_val, 1, K, thresh);", count_mult))
    }
  } else if (family_name == "mixture") {
    if (model_data$has_lambda) {
      sigma_fixed <- !model_data$has_sigma
      sigma2_fixed <- !model_data$has_sigma2

      if (sigma_fixed && sigma2_fixed) {
        dprime2_arg <- if (model_data$has_dprime2 || model_data$needs_ordered_dprime) "dprime2_n" else "0"
        lines <- c(lines, sprintf("      target += %smixture_sdt_lpmf(y[n] | dprime_n, %s, lambda_val, 1, K, thresh);", count_mult, dprime2_arg))
      } else {
        sigma_val <- if (model_data$has_sigma) "sigma_val" else "1"
        sigma2_val <- if (model_data$has_sigma2) "sigma2_val" else "1"
        dprime2_arg <- if (model_data$has_dprime2 || model_data$needs_ordered_dprime) "dprime2_n" else "0"
        lines <- c(lines, sprintf("      target += %smixture_sdt_lpmf(y[n] | dprime_n, %s, %s, %s, lambda_val, 1, K, thresh);",
                                  count_mult, sigma_val, dprime2_arg, sigma2_val))
      }
    } else {
      if (model_data$has_sigma) {
        lines <- c(lines, sprintf("      target += %suvsdt_lpmf(y[n] | dprime_n, sigma_val, 1, K, thresh);", count_mult))
      } else {
        lines <- c(lines, sprintf("      target += %sevsdt_lpmf(y[n] | dprime_n, 1, K, thresh);", count_mult))
      }
    }
  }
  
  if (isTRUE(model_data$has_lure_mixture)) {
    lines <- c(lines, "    } else {")
    lines <- c(lines, generate_param_lp_old_only("dprime_L", model_data$dprime_L_fixed, model_data$dprime_L_random, model_data$smooth_data[["dprime_L"]]))
    if (isTRUE(model_data$has_sigma_L)) {
      lines <- c(lines, generate_param_lp_old_only("sigma_L", model_data$sigma_L_fixed, model_data$sigma_L_random, model_data$smooth_data[["sigma_L"]]))
      lines <- c(lines, "      real sigma_L_val = exp(sigma_L_n);")
    }
    lines <- c(lines, generate_param_lp_old_only("lambda_L", model_data$lambda_L_fixed, model_data$lambda_L_random, model_data$smooth_data[["lambda_L"]]))
    lines <- c(lines, "      real lambda_L_val = inv_logit(lambda_L_n);")
    
    if (isTRUE(model_data$has_sigma_L)) {
      lines <- c(lines, sprintf("      target += %slure_mixture_lpmf(y[n] | dprime_L_n, sigma_L_val, lambda_L_val, K, thresh);", count_mult))
    } else {
      lines <- c(lines, sprintf("      target += %slure_mixture_lpmf(y[n] | dprime_L_n, lambda_L_val, K, thresh);", count_mult))
    }
    lines <- c(lines, "    }")
  } else {
    lines <- c(lines, "    } else {",
               sprintf("      target += %sevsdt_lpmf(y[n] | 0, 0, K, thresh);", count_mult),
               "    }")
  }
  
  lines
}

#' Generate parameter linear predictor for old items only (no is_old check)
#' Uses direct indexing when possible for faster sampling
#' @noRd
generate_param_lp_old_only <- function(param_name, fixed_info, random_info, smooth_info = NULL) {
  lines <- character(0)

  use_direct_indexing <- is_simple_encoding(fixed_info, random_info)

  if (use_direct_indexing) {
    first_group <- names(random_info)[1]
    idx_var <- paste0("idx_", param_name, "_", first_group)

    lines <- c(lines, sprintf("      int c_%s = %s[n];", param_name, idx_var))
    lines <- c(lines, sprintf("      real %s_n = beta_%s[c_%s];", param_name, param_name, param_name))

    for (group in names(random_info)) {
      re <- random_info[[group]]
      var_name <- if (!is.null(re$cor_id)) sprintf("u_%s_%s_from_%s", param_name, group, re$cor_id) else sprintf("u_%s_%s", param_name, group)
      lines <- c(lines, sprintf("      %s_n += %s[%s[n], c_%s];", param_name, var_name, group, param_name))
    }
  } else {
    lines <- c(lines, sprintf("      real %s_n = X_%s[n, ] * beta_%s;", param_name, param_name, param_name))

    if (!is.null(random_info)) {
      for (group in names(random_info)) {
        re <- random_info[[group]]
        var_name <- if (!is.null(re$cor_id)) sprintf("u_%s_%s_from_%s", param_name, group, re$cor_id) else sprintf("u_%s_%s", param_name, group)

        if (isTRUE(re$use_z_matrix)) {
          z_name <- paste0("Z_", param_name, "_", group)
          lines <- c(lines, sprintf("      %s_n += %s[n, ] * %s[%s[n], ]';", param_name, z_name, var_name, group))
        } else if (re$dim == 1) {
          if (!is.null(re$term_idx)) {
            idx_var <- paste0("idx_", param_name, "_", group)
            lines <- c(lines, sprintf("      if (%s[n] > 0) %s_n += %s[%s[n]];", idx_var, param_name, var_name, group))
          } else {
            lines <- c(lines, sprintf("      %s_n += %s[%s[n]];", param_name, var_name, group))
          }
        } else if (!is.null(re$term_idx)) {
          idx_var <- paste0("idx_", param_name, "_", group)
          lines <- c(lines, sprintf("      if (%s[n] > 0) %s_n += %s[%s[n], %s[n]];", idx_var, param_name, var_name, group, idx_var))
        }
      }
    }
  }

  if (!is.null(smooth_info)) {
    for (sm in smooth_info) {
      for (j in seq_along(sm$components)) {
        comp <- sm$components[[j]]
        comp_suffix <- comp$san_label
        for (k in seq_along(comp$Zs_list)) {
          zs_name <- paste0("Zs_", param_name, "_", comp_suffix, "_", k)
          coef_name <- paste0("s_", param_name, "_", comp_suffix, "_", k)
          lines <- c(lines, sprintf("      %s_n += %s[n, ] * %s;", param_name, zs_name, coef_name))
        }
      }
    }
  }

  lines
}

#' Generate the likelihood function call
#' @noRd
generate_likelihood_call <- function(model_data, family_name) {
  has_counts <- isTRUE(model_data$has_counts)
  
  ll_expr <- if (family_name == "cumulative") {
    "cumulative_lpmf(y[n] | dprime_n, K, thresh)"
  } else if (family_name == "source_mixture") {
    has_dprime_B <- isTRUE(model_data$has_dprime_B)
    has_lambda_B <- isTRUE(model_data$has_lambda_B)
    
    if (has_dprime_B && has_lambda_B) {
      "source_mixture_lpmf(y[n] | source[n], dprime_n, dprime_B_val, lambda_val, lambda_B_val, K, thresh)"
    } else if (has_dprime_B) {
      "source_mixture_lpmf(y[n] | source[n], dprime_n, dprime_B_val, lambda_val, K, thresh)"
    } else if (has_lambda_B) {
      "source_mixture_lpmf(y[n] | source[n], dprime_n, lambda_val, lambda_B_val, K, thresh)"
    } else {
      "source_mixture_lpmf(y[n] | source[n], dprime_n, lambda_val, K, thresh)"
    }
  } else if (family_name == "evsdt") {
    "evsdt_lpmf(y[n] | dprime_n, is_old[n], K, thresh)"
  } else if (family_name == "uvsdt") {
    "uvsdt_lpmf(y[n] | dprime_n, sigma_val, is_old[n], K, thresh)"
  } else if (family_name == "dpsdt") {
    if (model_data$has_sigma) {
      "dpsdt_lpmf(y[n] | dprime_n, sigma_val, lambda_val, is_old[n], K, thresh)"
    } else {
      "dpsdt_lpmf(y[n] | dprime_n, lambda_val, is_old[n], K, thresh)"
    }
  } else if (family_name == "mixture") {
    if (model_data$has_lambda) {
      sigma_fixed <- !model_data$has_sigma
      sigma2_fixed <- !model_data$has_sigma2

      if (sigma_fixed && sigma2_fixed) {
        dprime2_arg <- if (model_data$has_dprime2 || model_data$needs_ordered_dprime) "dprime2_val" else "0"
        sprintf("mixture_sdt_lpmf(y[n] | dprime_n, %s, lambda_val, is_old[n], K, thresh)", dprime2_arg)
      } else {
        sigma_val <- if (model_data$has_sigma) "sigma_val" else "1"
        sigma2_val <- if (model_data$has_sigma2) "sigma2_val" else "1"
        dprime2_arg <- if (model_data$has_dprime2 || model_data$needs_ordered_dprime) "dprime2_val" else "0"
        sprintf("mixture_sdt_lpmf(y[n] | dprime_n, %s, %s, %s, lambda_val, is_old[n], K, thresh)",
                sigma_val, dprime2_arg, sigma2_val)
      }
    } else {
      if (model_data$has_sigma) {
        "uvsdt_lpmf(y[n] | dprime_n, sigma_val, is_old[n], K, thresh)"
      } else {
        "evsdt_lpmf(y[n] | dprime_n, is_old[n], K, thresh)"
      }
    }
  } else {
    stop("Unknown family: ", family_name)
  }
  
  if (has_counts) {
    sprintf("    target += counts[n] * %s;", ll_expr)
  } else {
    sprintf("    target += %s;", ll_expr)
  }
}

generate_thresh_code <- function(model_data) {
  crit <- model_data$criterion
  K <- model_data$K

  is_intercept_only <- isTRUE(crit$is_intercept_only) || crit$n_coef == 1

  gen_re_lines <- function(target, thresh_k_expr, role = "mid") {
    re_lines <- character(0)
    for (group in names(crit$random)) {
      re <- crit$random[[group]]
      var_name <- if (!is.null(re$cor_id)) sprintf("u_criterion_%s_from_%s", group, re$cor_id) else sprintf("u_criterion_%s", group)
      n_re_terms <- if (!is.null(re$n_re_terms)) re$n_re_terms else re$n_cond_levels

      if (isTRUE(re$use_z_matrix)) {
        z_name <- paste0("Z_criterion_", group)
        re_lines <- c(re_lines, sprintf("    %s += %s[n, ] * %s[%s[n], ((%s-1)*%d + 1):(%s*%d)]';",
                                        target, z_name, var_name, group,
                                        thresh_k_expr, n_re_terms, thresh_k_expr, n_re_terms))
      } else if (n_re_terms == 1 && is.null(re$term_idx)) {
        re_lines <- c(re_lines, sprintf("    %s += %s[%s[n], %s];", target, var_name, group, thresh_k_expr))
      } else {
        idx_var <- paste0("idx_criterion_", group)
        re_lines <- c(re_lines, sprintf("    if (%s[n] > 0) %s += %s[%s[n], (%s-1) * %d + %s[n]];",
                                        idx_var, target, var_name, group, thresh_k_expr, n_re_terms, idx_var))
      }
    }
    re_lines
  }

  lines <- c("    vector[K-1] thresh;")

  if (is_intercept_only) {
    lines <- c(lines, "    thresh[mid_thresh] = beta_thresh_mid[1];")
  } else {
    lines <- c(lines, "    thresh[mid_thresh] = X_criterion[n, ] * beta_thresh_mid;")
  }
  lines <- c(lines, gen_re_lines("thresh[mid_thresh]", "mid_thresh"))

  if (!is.null(model_data$smooth_data[["criterion"]])) {
    for (sm in model_data$smooth_data[["criterion"]]) {
      for (comp in sm$components) {
        for (k in seq_along(comp$Zs_list)) {
          zs_name <- paste0("Zs_criterion_", comp$san_label, "_", k)
          coef_name <- paste0("s_criterion_", comp$san_label, "_", k)
          lines <- c(lines, sprintf("    thresh[mid_thresh] += %s[n, ] * %s[1, ]';", zs_name, coef_name))
        }
      }
    }
  }

  if (K > 2) {
    lines <- c(lines,
               "    for (k in (mid_thresh+1):(K-1)) {")

    if (is_intercept_only) {
      lines <- c(lines, "      real log_gap = beta_log_gaps[k - mid_thresh, 1];")
    } else {
      lines <- c(lines, "      real log_gap = X_criterion[n, ] * beta_log_gaps[k - mid_thresh, ]';")
    }
    lines <- c(lines, gen_re_lines("log_gap", "k"))
    if (!is.null(model_data$smooth_data[["criterion"]])) {
      for (sm in model_data$smooth_data[["criterion"]]) {
        for (comp in sm$components) {
          for (ki in seq_along(comp$Zs_list)) {
            zs_name <- paste0("Zs_criterion_", comp$san_label, "_", ki)
            coef_name <- paste0("s_criterion_", comp$san_label, "_", ki)
            lines <- c(lines, sprintf("      log_gap += %s[n, ] * %s[1 + k - mid_thresh, ]';", zs_name, coef_name))
          }
        }
      }
    }
    lines <- c(lines, "      thresh[k] = thresh[k-1] + exp(log_gap);", "    }")

    lines <- c(lines,
               "    for (k_down in 1:(mid_thresh-1)) {",
               "      int k = mid_thresh - k_down;")

    if (is_intercept_only) {
      lines <- c(lines, "      real log_gap = beta_log_gaps[K - 1 - mid_thresh + k_down, 1];")
    } else {
      lines <- c(lines, "      real log_gap = X_criterion[n, ] * beta_log_gaps[K - 1 - mid_thresh + k_down, ]';")
    }
    lines <- c(lines, gen_re_lines("log_gap", "k"))
    if (!is.null(model_data$smooth_data[["criterion"]])) {
      for (sm in model_data$smooth_data[["criterion"]]) {
        for (comp in sm$components) {
          for (ki in seq_along(comp$Zs_list)) {
            zs_name <- paste0("Zs_criterion_", comp$san_label, "_", ki)
            coef_name <- paste0("s_criterion_", comp$san_label, "_", ki)
            lines <- c(lines, sprintf("      log_gap += %s[n, ] * %s[1 + K - 1 - mid_thresh + k_down, ]';", zs_name, coef_name))
          }
        }
      }
    }
    lines <- c(lines, "      thresh[k] = thresh[k+1] - exp(log_gap);", "    }")
  }

  lines
}

#' Check if a parameter has a simple encoding pattern
#' 
#' A simple encoding pattern means:
#' - Fixed effects use encoding() wrapper
#' - All random effects for this parameter use encoding with term_idx (not Z matrix)
#' - The fixed effect dimension matches the random effect dimension
#' 
#' This allows using `beta[idx]` instead of `X[n,] * beta` for faster sampling.
#' @noRd
is_simple_encoding <- function(fixed_info, random_info) {
  if (!isTRUE(fixed_info$has_encoding)) return(FALSE)
  
  if (is.null(random_info) || length(random_info) == 0) return(FALSE)
  
  for (group in names(random_info)) {
    re <- random_info[[group]]
    if (!isTRUE(re$is_encoding)) return(FALSE)
    if (is.null(re$term_idx)) return(FALSE)
    if (isTRUE(re$use_z_matrix)) return(FALSE)
    if (re$dim != fixed_info$n_coef) return(FALSE)
  }
  
  TRUE
}
