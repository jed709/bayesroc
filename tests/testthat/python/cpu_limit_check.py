import json
import sys

sys.path.insert(0, sys.argv[1])
import cpus

out = {}
hybrid = [(1, [0, 1]), (1, [2, 3]), (1, [4, 5]), (0, [6]), (0, [7])]
plain = [(0, [0, 4]), (0, [1, 5]), (0, [2, 6]), (0, [3, 7])]
for name, cores in (("hybrid", hybrid), ("plain", plain)):
    cpus._windows_cores = lambda c=cores: c
    cpus._linux_cores = lambda c=cores: c
    out[name] = cpus.cpu_order()
del cpus._windows_cores, cpus._linux_cores
import importlib
importlib.reload(cpus)
start = cpus.current()
out["start_n"] = len(start)
if start:
    out["limited"] = cpus.limit(2)
    out["now"] = cpus.current()
    out["restored"] = cpus.limit(None)
print("RESULT " + json.dumps(out))
