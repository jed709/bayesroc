"""Rank-normalised bulk/tail ESS and R-hat (Vehtari et al., 2021), vectorised over
parameters. A line-by-line port of the posterior R package (1.7.1): ess_bulk,
ess_tail, rhat and their helpers, so the stopping rule of iter_sampling = "auto"
gives the numbers summary() reports. Internally arrays are [parameters, chains,
iterations] so sorts and FFTs run over contiguous memory."""
import os
import numpy as np
from scipy.special import ndtri

_EPS = np.finfo(float).eps


def _split(x):
    n = x.shape[2]
    if n == 1:
        return x
    return np.concatenate([x[:, :, :n // 2], x[:, :, n - n // 2:]], axis=1)


def _degenerate(x):
    flat = x.reshape(x.shape[0], -1)
    finite = np.isfinite(flat).all(1)
    with np.errstate(invalid="ignore"):
        const = np.abs(np.nanmax(flat, 1) - np.nanmin(flat, 1)) < _EPS
    return ~finite | const


def _avg_ranks(flat):
    V, S = flat.shape
    order = np.argsort(flat, axis=1)
    sv = np.take_along_axis(flat, order, axis=1)
    new = np.ones((V, S), dtype=bool)
    new[:, 1:] = sv[:, 1:] != sv[:, :-1]
    gid = np.cumsum(new, axis=1) - 1
    pos = np.broadcast_to(np.arange(1, S + 1, dtype=np.float64), (V, S))
    off = (gid + np.arange(V)[:, None] * S).ravel()
    sums = np.bincount(off, weights=pos.ravel(), minlength=V * S)
    cnts = np.bincount(off, minlength=V * S)
    avg = (sums / np.maximum(cnts, 1))[off].reshape(V, S)
    r = np.empty((V, S))
    np.put_along_axis(r, order, avg, axis=1)
    return r


def _z_scale(x):
    V = x.shape[0]
    S = x.shape[1] * x.shape[2]
    r = _avg_ranks(x.reshape(V, S))
    return ndtri((r - 0.375) / (S + 0.25)).reshape(x.shape)


def _next_fast(n):
    m = n
    while True:
        k = m
        for f in (2, 3, 5):
            while k % f == 0:
                k //= f
        if k == 1:
            return m
        m += 1


def _ess(x):
    V, M, N = x.shape
    out = np.full(V, np.nan)
    if N < 3:
        return out
    bad = _degenerate(x)
    x = np.where(bad[:, None, None], 0.0, x)
    yc = x - x.mean(2, keepdims=True)
    nfft = 2 * _next_fast(N)
    f = np.fft.rfft(yc, n=nfft, axis=2)
    acov = (np.fft.irfft(f.real ** 2 + f.imag ** 2, n=nfft, axis=2)[:, :, :N] / N).mean(1).T
    mean_var = acov[0] * N / (N - 1)
    var_plus = mean_var * (N - 1) / N
    if M > 1:
        var_plus = var_plus + x.mean(2).var(1, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = lambda k: 1.0 - (mean_var - acov[k]) / var_plus
        R = np.zeros((N, V))
        R[0] = 1.0
        even = np.ones(V)
        odd = rho(1)
        R[1] = odd
        t = np.zeros(V, dtype=int)
        active = np.ones(V, dtype=bool)
        tt = 0
        while True:
            s = even + odd
            active &= (tt < N - 5) & ~np.isnan(s) & (s > 0)
            if not active.any():
                break
            tt += 2
            e, o = rho(tt), rho(tt + 1)
            even = np.where(active, e, even)
            odd = np.where(active, o, odd)
            store = active & (e + o >= 0)
            R[tt] = np.where(store, e, R[tt])
            R[tt + 1] = np.where(store, o, R[tt + 1])
            t = np.where(active, tt, t)
        cols = np.arange(V)
        pos = even > 0
        R[t[pos], cols[pos]] = even[pos]
        tt = 0
        while True:
            act = tt <= t - 4
            if not act.any():
                break
            tt += 2
            prev = R[tt - 2] + R[tt - 1]
            fix = act & (R[tt] + R[tt + 1] > prev)
            R[tt] = np.where(fix, prev / 2, R[tt])
            R[tt + 1] = np.where(fix, prev / 2, R[tt + 1])
        cs = np.cumsum(R, axis=0)
        head = np.where(t > 0, cs[np.maximum(t - 1, 0), cols], R[0])
        tau = -1.0 + 2.0 * head + R[t, cols]
        tau = np.maximum(tau, 1.0 / np.log10(M * N))
        ess = M * N / tau
    return np.where(bad, np.nan, ess)


def _rhat(x):
    N = x.shape[2]
    bad = _degenerate(x)
    with np.errstate(invalid="ignore", divide="ignore"):
        cm = x.mean(2)
        within = x.var(2, ddof=1).mean(1)
        between = N * cm.var(1, ddof=1)
        r = np.sqrt((between / within + N - 1) / N)
    return np.where(bad, np.nan, r)


def _ess_quantile(x, prob, raw_bad):
    q = np.quantile(x.reshape(x.shape[0], -1), prob, axis=1)
    ind = (x <= q[:, None, None]).astype(float)
    return np.where(raw_bad, np.nan, _ess(_split(ind)))


def ess_rhat(x, threads=None):
    """x [iterations, chains, parameters] -> (ess_bulk, ess_tail, rhat), each [parameters].
    Parameters are split across threads; every step is per parameter."""
    x = np.ascontiguousarray(np.transpose(np.asarray(x, dtype=np.float64), (2, 1, 0)))
    V = x.shape[0]
    n = min(threads or os.cpu_count() or 1, 8, max(1, V // 16))
    if n <= 1:
        return _ess_rhat(x)
    from concurrent.futures import ThreadPoolExecutor
    parts = np.array_split(np.arange(V), n)
    with ThreadPoolExecutor(n) as ex:
        res = list(ex.map(lambda idx: _ess_rhat(x[idx]), parts))
    return tuple(np.concatenate([r[k] for r in res]) for k in range(3))


def _ess_rhat(x):
    raw_bad = _degenerate(x)
    xs = _split(x)
    zs = _z_scale(xs)
    bulk = _ess(zs)
    tail = np.minimum(_ess_quantile(x, 0.05, raw_bad), _ess_quantile(x, 0.95, raw_bad))
    with np.errstate(invalid="ignore"):
        folded = np.abs(x - np.median(x.reshape(x.shape[0], -1), axis=1)[:, None, None])
    rb = _rhat(zs)
    rt = _rhat(_z_scale(_split(folded)))
    rhat = np.where(np.isnan(rb) | np.isnan(rt), np.nan, np.maximum(rb, rt))
    return bulk, tail, rhat
