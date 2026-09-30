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
