import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
import autorun as ar

n = ["a", "b", "c", "d"]
cl = np.array(["pop", "sd", "cor", "cor"])
t = {"pop": 400, "sd": 400, "cor": 400}
one = np.ones(4)
cases = [
    (ar._blocker(np.array([500, 500, 300, 500.]), one, cl, n, t, 1.01), "c"),
    (ar._blocker(np.array([500, 500, 300, 300.]), one, cl, n, t, 1.01), None),
    (ar._blocker(np.array([500, 500, 300, 500.]), np.array([1, 1.02, 1, 1]), cl, n, t, 1.01), None),
    (ar._blocker(np.array([500, 500, 300, 500.]), np.array([1, 1, 1.02, 1]), cl, n, t, 1.01), "c"),
    (ar._blocker(np.array([500, 500, 500, np.nan]), np.array([1, 1, 1, np.nan]), cl, n, t, 1.01), None),
    (ar._blocker(np.array([500, 900, 500, 500.]), one, cl, n, {"pop": 400, "sd": 1000, "cor": 400}, 1.01), "b"),
]
print("OK" if all(g == w for g, w in cases) else "FAIL " + repr([g for g, _ in cases]))
