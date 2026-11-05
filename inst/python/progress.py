"""Progress bars. Under BAYESROC_PROGRESS=events (set when R launches the backend) a bar is
a stream of `@@bayesroc-progress {json}` lines on stdout that R draws as one line in place;
otherwise it is a tqdm bar."""
import itertools
import json
import os
import sys
import threading
import time
from contextlib import contextmanager

EVENTS = os.environ.get("BAYESROC_PROGRESS") == "events"
_MARK = "@@bayesroc-progress "
_RATE = 0.25
_lock = threading.Lock()
_ids = itertools.count()
_groups = []


def _emit(ev):
    with _lock:
        sys.stdout.write(_MARK + json.dumps(ev, separators=(",", ":")) + "\n")
        sys.stdout.flush()


class Bar:
    """n of total `unit`s done, counted from `start`; `leave` keeps the finished line; total None counts
    with no end."""

    def __init__(self, total, desc, start=0, unit="iterations", leave=True):
        self.id = f"{os.getpid()}-{next(_ids)}"
        self.open = total is None
        self.total, self.desc, self.start, self.n = 0 if self.open else int(total), desc, int(start), int(start)
        self.unit, self.leave = unit, bool(leave)
        self.t, self.closed = 0.0, False
        self._send(True)

    def _send(self, force, done=False, drop=False):
        now = time.monotonic()
        if not force and now - self.t < _RATE:
            return
        self.t = now
        _emit(dict(id=self.id, desc=self.desc, n=self.n, total=self.total, start=self.start,
                   unit=self.unit, leave=self.leave, done=done, drop=drop))

    def update(self, k=1):
        if self.closed:
            return
        self.n += int(k)
        if not self.open and self.n > self.total:
            self.total = self.n
        self._send(not self.open and self.n >= self.total)

    def close(self, drop=False):
        if not self.closed:
            self.closed = True
            self._send(True, done=True, drop=drop)


class _Stage:
    """A stage of a group: its iterations count toward the group's bar under its name."""

    def __init__(self, g, total, name):
        self.g = g
        g.desc = f"{g.title} ({name})"
        if g.n + int(total) > g.total:
            g.total = g.n + int(total)

    def update(self, k=1):
        self.g.update(k)

    def close(self, drop=False):
        pass


def bar(total, desc, start=0, unit="iterations", leave=True, stage=None):
    """A bar over [start, total]; inside group(), a bar with a `stage` name advances the
    group's bar instead of drawing its own. None when there is no way to show one."""
    if EVENTS:
        if stage is not None and _groups:
            return _Stage(_groups[-1], int(total) - int(start), stage)
        return Bar(total, desc if stage is None else f"{desc} ({stage})", start, unit, leave)
    try:
        from tqdm.auto import tqdm
        return tqdm(total=None if total is None else int(total) - int(start), desc=desc, unit=unit[:-1] if unit.endswith("s") else unit,
                    leave=leave)
    except Exception:
        return None


@contextmanager
def group(total, title):
    """One bar for several stages (warmup's), titled `title`."""
    if not EVENTS:
        yield None
        return
    g = Bar(total, title)
    g.title = title
    _groups.append(g)
    try:
        yield g
    finally:
        _groups.remove(g)
        g.close()


class _Chains:
    """NumPyro's per-chain bars as one bar per phase: chains in parallel share one that
    follows the slowest, labelled with the range over chains; chains in sequence
    each get their own."""

    def __init__(self, chains, parallel, desc, split):
        self.chains, self.parallel, self.desc, self.split = int(chains), bool(parallel), desc, split
        self.shims, self.bars, self.lock = [], {}, threading.Lock()

    def phases(self, per):
        if self.split is not None and 0 < self.split < per:
            return [("NUTS warmup", self.split), ("NUTS sampling", per - self.split)]
        return [(self.desc or "NUTS", per)]

    def refresh(self, shim):
        with self.lock:
            if self.parallel:
                counts = [s.n for s in self.shims]
                key, n = 0, min(counts)
            else:
                counts, key, n = None, self.shims.index(shim), shim.n
            lo = 0
            for i, (name, m) in enumerate(self.phases(shim.total)):
                hi = lo + m
                if i == 0 or n > lo:
                    if counts is None:
                        desc = name + (f" (chain {key + 1}/{self.chains})" if self.chains > 1 else "")
                    else:
                        c = [min(max(x - lo, 0), m) for x in counts]
                        desc = name + f" ({self.chains} chains: {min(c)}-{max(c)})"
                    b = self.bars.get((key, i))
                    if b is None:
                        b = self.bars[(key, i)] = Bar(m, desc)
                    moved, b.desc = b.desc != desc, desc
                    k = min(n, hi) - lo - b.n
                    if k > 0:
                        b.update(k)
                    elif moved and not b.closed:
                        b._send(False)
                    if n >= hi:
                        b.close()
                lo = hi

    def close(self):
        with self.lock:
            for b in self.bars.values():
                b.close(drop=b.n == 0)


class _Shim:
    def __init__(self, chains, it=None, total=None, **_):
        self.total = len(it) if it is not None else int(total or 0)
        self.n, self.agg = 0, chains
        with chains.lock:
            chains.shims.append(self)
        chains.refresh(self)

    def set_description(self, *a, **k):
        pass

    def set_postfix_str(self, *a, **k):
        pass

    def update(self, k=1):
        self.n += int(k)
        self.agg.refresh(self)

    def close(self):
        pass

    def __iter__(self):
        for i in range(self.total):
            yield i
            self.update(1)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@contextmanager
def nuts(chains, parallel, desc=None, split=None):
    """NumPyro's progress bars inside the block as events: `desc` names the one phase, or
    `split` warmup iterations start a run that goes on to sampling."""
    if not EVENTS:
        yield
        return
    import types
    import numpyro.util as nu
    agg = _Chains(chains, parallel, desc, split)
    old = nu.tqdm_auto, nu.tqdm
    nu.tqdm_auto = lambda it=None, *a, **k: _Shim(agg, it, **k)
    nu.tqdm = types.SimpleNamespace(trange=lambda n, *a, **k: _Shim(agg, range(int(n))))
    try:
        yield
        import jax
        jax.effects_barrier()
    finally:
        nu.tqdm_auto, nu.tqdm = old
        agg.close()
