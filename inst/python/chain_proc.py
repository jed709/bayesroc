import os
import pickle
import sys

argf, outf, cpulist = sys.argv[1:4]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpus

if sys.platform == "win32":
    cpus._windows_mask(sum(1 << int(c) for c in cpulist.split(",")))
    cpus._windows_full_speed()
elif sys.platform.startswith("linux"):
    os.sched_setaffinity(0, {int(c) for c in cpulist.split(",")})
import numpy as np
import jax
import numpyro_backend as nb

with open(argf, "rb") as fh:
    kw = pickle.load(fh)
res = nb.fit_model(**kw)
res = jax.tree.map(lambda x: np.asarray(x) if hasattr(x, "shape") else x, res)
with open(outf, "wb") as fh:
    pickle.dump(res, fh)
