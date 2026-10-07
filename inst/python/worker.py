"""Fit driver shared by one-off fits and the persistent worker behind iter_sampling = "auto".

run_one() executes a fit described by the JSON written by R's fit_jax() and writes
the results next to it. serve() keeps one process alive across fits so imports,
the auto-sampling session (autorun.AutoSession) lives in `held` between requests."""
import glob
import json
import os
import pickle
import sys
import time
import traceback

import numpy as np

_MATRIX_KEYS = ("X", "Z", "X_crit")


def _convert_matrices(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in _MATRIX_KEYS and isinstance(v, list) and v and isinstance(v[0], list):
                obj[k] = [list(map(float, row)) for row in v]
            else:
                _convert_matrices(v)
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                _convert_matrices(item)


def _warnings(ef, sampler, max_treedepth):
    out = []
    if "diverging" in ef:
        n_div = int(ef["diverging"].sum())
        n_total = ef["diverging"].size
        if n_div > 0:
            out.append(f"WARNING: {n_div} of {n_total} ({100 * n_div / n_total:.1f}%) "
                       "transitions ended with a divergence.")
    if sampler != "drghmc" and "energy" in ef:
        low = []
        for c in range(ef["energy"].shape[0]):
            e = ef["energy"][c]
            bfmi = np.sum(np.diff(e) ** 2) / np.sum((e - np.mean(e)) ** 2)
            if bfmi < 0.3:
                low.append((c + 1, bfmi))
        if low:
            out.append("WARNING: Low E-BFMI for " +
                       ", ".join(f"chain {c} (E-BFMI={b:.3f})" for c, b in low) + ".")
    if "num_steps" in ef:
        n_hit = int((ef["num_steps"] >= 2 ** int(max_treedepth)).sum())
        n_total = ef["num_steps"].size
        if n_hit > 0:
            out.append(f"WARNING: {n_hit} of {n_total} ({100 * n_hit / n_total:.1f}%) "
                       f"transitions hit max treedepth ({max_treedepth}).")
    if "dr_stage" in ef:
        rej = float((ef["dr_stage"] < 0).mean())
        if rej > 0.30:
            out.append(f"WARNING: DR-G-HMC rejection fraction {100 * rej:.0f}%; "
                       "adaptation may be insufficient.")
    return out


def run_one(config_path, out_dir):
    from numpyro_backend import fit_model

    with open(config_path) as fh:
        args = json.load(fh)
    config = args["config"]
    _convert_matrices(config)
    kw = args["fit_kwargs"]
    result = fit_model(config=config, **kw)

    os.makedirs(out_dir, exist_ok=True)
    output = {"elapsed": result["elapsed"], "samples": {}, "extra_fields": {}}
    for group, prefix in (("samples", "sample_"), ("extra_fields", "extra_")):
        for k, v in result[group].items():
            fname = f"{prefix}{k}.npy"
            np.save(os.path.join(out_dir, fname), v)
            output[group][k] = fname
    if result.get("state") is not None:
        with open(os.path.join(out_dir, "state.pkl"), "wb") as fh:
            pickle.dump(result["state"], fh)
        output["state"] = "state.pkl"
    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        json.dump(output, fh)
    for w in _warnings(result["extra_fields"], kw.get("sampler", "nuts"),
                       kw.get("max_tree_depth", 10)):
        print(w, flush=True)


def run_auto(op, args_path, out_dir, held, token):
    """fit_broc(iter_sampling = "auto"): start a session (warmup + compiled block), take one
    round, or finish (constrain the kept draws once and write the usual results)."""
    import autorun
    with open(args_path) as fh:
        args = json.load(fh)
    os.makedirs(out_dir, exist_ok=True)
    if op == "auto_start":
        from numpyro_backend import sdt_model
        config = args["config"]
        _convert_matrices(config)
        old = held.pop(("auto", token), None)
        if old is not None:
            old.close()
        sess = autorun.AutoSession(sdt_model, config, args["kwargs"])
        held[("auto", token)] = sess
        with open(os.path.join(out_dir, "start.json"), "w") as fh:
            json.dump(sess.start_info(), fh)
    elif op == "auto_map":
        held[("auto", token)].set_map(args)
    elif op == "auto_step":
        held[("auto", token)].step(out_dir, args)
    elif op == "auto_finish":
        sess = held.pop(("auto", token))
        try:
            sess.finish(out_dir, args)
        finally:
            sess.close()
    else:
        raise ValueError(f"unknown worker op {op!r}")


def _watch_stop(control_dir):
    """Leave as soon as R asks, even mid-computation: the stop file is R's interrupt, and a
    request in flight cannot be resumed anyway."""
    import threading

    def watch():
        stop = os.path.join(control_dir, "stop")
        while True:
            if os.path.exists(stop):
                os._exit(0)
            time.sleep(0.1)
    threading.Thread(target=watch, daemon=True).start()


def serve(control_dir):
    held = {}
    _watch_stop(control_dir)
    print("[bayesroc worker] ready", flush=True)
    while True:
        if os.path.exists(os.path.join(control_dir, "stop")):
            return
        reqs = sorted(glob.glob(os.path.join(control_dir, "req_*.json")))
        if not reqs:
            time.sleep(0.02)
            continue
        req = reqs[0]
        try:
            with open(req) as fh:
                job = json.load(fh)
        except (PermissionError, OSError, ValueError):
            time.sleep(0.05)
            continue
        out_dir = job["out"]
        try:
            op = job.get("op") or "fit"
            if op == "fit":
                run_one(job["config"], out_dir)
            else:
                run_auto(op, job["config"], out_dir, held, job.get("token"))
            open(os.path.join(out_dir, "DONE"), "w").close()
        except BaseException:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "ERROR.tmp"), "w") as fh:
                fh.write(traceback.format_exc())
            os.replace(os.path.join(out_dir, "ERROR.tmp"), os.path.join(out_dir, "ERROR"))
        finally:
            try:
                os.remove(req)
            except OSError:
                pass


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--serve":
        serve(sys.argv[2])
    else:
        run_one(sys.argv[1], sys.argv[2])
