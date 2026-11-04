"""The set of logical CPUs a fit may run on (fit_broc(total_cores =)).

XLA sizes its CPU thread pool from the machine and offers no thread count, so the limit is the
process's CPU affinity, which every thread and child process inherits. CPUs are taken in order of
preference: the most performant cores first (P-cores before E-cores on hybrid processors), one
logical CPU per physical core before any second hyperthread. Windows and Linux only; elsewhere the
process keeps every CPU."""
import os
import sys

_INITIAL = None


def _windows_cores():
    """[(efficiency class, [logical CPUs])] for processor group 0, from
    GetLogicalProcessorInformationEx(RelationProcessorCore)."""
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    fn = k32.GetLogicalProcessorInformationEx
    fn.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
    fn.restype = wintypes.BOOL
    n = wintypes.DWORD(0)
    fn(0, None, ctypes.byref(n))
    buf = ctypes.create_string_buffer(n.value)
    if not fn(0, buf, ctypes.byref(n)):
        return []
    raw, off, cores = buf.raw, 0, []
    while off < n.value:
        size = int.from_bytes(raw[off + 4:off + 8], "little")
        eff = raw[off + 9]
        group_count = int.from_bytes(raw[off + 30:off + 32], "little")
        for g in range(group_count):
            base = off + 32 + 16 * g
            mask = int.from_bytes(raw[base:base + 8], "little")
            group = int.from_bytes(raw[base + 8:base + 10], "little")
            if group == 0 and mask:
                cores.append((eff, [b for b in range(64) if mask >> b & 1]))
        off += size
    return cores


def _linux_cores():
    """[(performance rank, [logical CPUs])] from sysfs: cpu_core/cpu_atom lists on Intel hybrid
    processors, else cpu_capacity, else every core equal."""
    base = "/sys/devices/system/cpu"
    online = sorted(os.sched_getaffinity(0))

    def read(path):
        try:
            with open(path) as fh:
                return fh.read().strip()
        except OSError:
            return None

    def cpulist(s):
        out = set()
        for part in (s or "").split(","):
            if "-" in part:
                a, b = part.split("-")
                out.update(range(int(a), int(b) + 1))
            elif part:
                out.add(int(part))
        return out

    p_cores = cpulist(read("/sys/devices/cpu_core/cpus"))
    groups = {}
    for c in online:
        key = (read(f"{base}/cpu{c}/topology/physical_package_id"), read(f"{base}/cpu{c}/topology/core_id"))
        groups.setdefault(key, []).append(c)
    cores = []
    for cpus in groups.values():
        cap = read(f"{base}/cpu{cpus[0]}/cpu_capacity")
        rank = 1 if (p_cores and cpus[0] in p_cores) else (int(cap) if cap and not p_cores else 0)
        cores.append((rank, sorted(cpus)))
    return cores


def cpu_order():
    """Every logical CPU of the process's current set, in order of preference."""
    if sys.platform == "win32":
        cores = _windows_cores()
    elif sys.platform.startswith("linux"):
        cores = _linux_cores()
    else:
        return []
    order = []
    for rank in sorted({c[0] for c in cores}, reverse=True):
        tier = sorted((c[1] for c in cores if c[0] == rank), key=lambda cpus: cpus[0])
        for level in range(max(len(cpus) for cpus in tier)):
            order += [cpus[level] for cpus in tier if level < len(cpus)]
    return order


def _windows_mask(mask=None):
    import ctypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    proc = k32.GetCurrentProcess()
    if mask is None:
        pm, sm = ctypes.c_size_t(), ctypes.c_size_t()
        k32.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                                               ctypes.POINTER(ctypes.c_size_t)]
        k32.GetProcessAffinityMask(proc, ctypes.byref(pm), ctypes.byref(sm))
        return pm.value
    k32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    if not k32.SetProcessAffinityMask(proc, mask):
        raise OSError(f"SetProcessAffinityMask failed (error {ctypes.get_last_error()})")
    return mask


def _windows_full_speed():
    """Opt the process out of Windows power throttling (EcoQoS), which otherwise lowers the clock
    of a process whose window is not in the foreground. True when the setting was applied."""
    import ctypes

    class _State(ctypes.Structure):
        _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.SetProcessInformation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
    st = _State(1, 0x1, 0x0)
    return bool(k32.SetProcessInformation(k32.GetCurrentProcess(), 4, ctypes.byref(st), ctypes.sizeof(st)))


def current():
    """The process's logical CPUs."""
    if sys.platform == "win32":
        m = _windows_mask()
        return [b for b in range(64) if m >> b & 1]
    if sys.platform.startswith("linux"):
        return sorted(os.sched_getaffinity(0))
    return []


def limit(n):
    """Restrict the process to its n preferred logical CPUs; None restores the set it started with.
    Returns the CPUs in use, or None where the platform has no affinity control."""
    global _INITIAL
    if not (sys.platform == "win32" or sys.platform.startswith("linux")):
        return None
    if _INITIAL is None:
        _INITIAL = current()
    if n is None:
        chosen = list(_INITIAL)
    else:
        allowed = set(_INITIAL)
        chosen = [c for c in cpu_order() if c in allowed][:int(n)] or list(_INITIAL)
    if sys.platform == "win32":
        _windows_mask(sum(1 << c for c in chosen))
        _windows_full_speed()
    else:
        os.sched_setaffinity(0, set(chosen))
    return sorted(chosen)
