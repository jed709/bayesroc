"""Rank-normalised bulk/tail ESS and R-hat, vectorised over parameters. R-hat is a port of
the posterior R package (1.7.1). ESS is by batch means: each chain's draws are cut into
batches of b = max(100, N/10) (N/2 for shorter runs), the batch means of all chains are
pooled about the grand mean, and ESS = M N var(x) / (b var(batch means)). Geyer's initial
positive sequence, which posterior uses, assumes a reversible chain; under generalised HMC
with persistent momentum the autocorrelation oscillates, so it stops at the first negative
pair and overstates tail ESS while understating it for means. The R side (R/ess.R)
computes the same estimator, so summary() reports the numbers the stopping rule of
iter_sampling = "auto" uses. Internally arrays are [parameters, chains, iterations] so
sorts run over contiguous memory."""
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


_BATCH = 100


def _batch(N):
    b = max(_BATCH, N // 10)
    return b if N >= 2 * b else N // 2


def _ess(x):
    V, M, N = x.shape
    out = np.full(V, np.nan)
    b = _batch(N)
    if b < 1:
        return out
    bad = _degenerate(x)
    x = np.where(bad[:, None, None], 0.0, x)
    nb = N // b
    bm = x[:, :, :nb * b].reshape(V, M, nb, b).mean(3).reshape(V, M * nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        vb = b * bm.var(1, ddof=1)
        vx = x.reshape(V, M * N).var(1, ddof=1)
        ess = np.minimum(M * N * vx / vb, M * N * np.log10(M * N))
    return np.where(bad | ~(vb > 0), np.nan, ess)


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
    return np.where(raw_bad, np.nan, _ess(ind))


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
    zs = _z_scale(_split(x))
    bulk = _ess(_z_scale(x))
    tail = np.minimum(_ess_quantile(x, 0.05, raw_bad), _ess_quantile(x, 0.95, raw_bad))
    with np.errstate(invalid="ignore"):
        folded = np.abs(x - np.median(x.reshape(x.shape[0], -1), axis=1)[:, None, None])
    rb = _rhat(zs)
    rt = _rhat(_z_scale(_split(folded)))
    rhat = np.where(np.isnan(rb) | np.isnan(rt), np.nan, np.maximum(rb, rt))
    return bulk, tail, rhat
