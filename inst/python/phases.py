"""Wall-clock phase marks for profiling a fit (BAYESROC_PROFILE=1): seconds since the
process started, printed as "[profile] <seconds>s <phase>" lines."""
import os
import time

ON = os.environ.get("BAYESROC_PROFILE", "") not in ("", "0")


def _start():
    try:
        import psutil
        return psutil.Process().create_time()
    except Exception:
        return time.time()


T0 = _start() if ON else None


def reset():
    global T0
    if ON:
        T0 = time.time()


def mark(what):
    if ON:
        print(f"[profile] {time.time() - T0:8.3f}s {what}", flush=True)
