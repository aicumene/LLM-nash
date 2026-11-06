"""The readouts of every call's internal state into one npz per layer: s0 (after the last prompt token), mean (over the
reply), last (after the last token written), act (at the step that writes the first token after the action marker —
MOVE:, GUESS:, ROUTE: — the last marker in the reply), with the call's run, condition, seed, round, agent and the round's
exploitability (this round's outcome, and the previous round's, which the prompt already contains).
Usage: export_states.py RUNS_DIR OUT_DIR --markers markers.json [--dumps DIR] [--layers 6,18,30,47] [--stream mean]"""
from __future__ import annotations
import json, glob, os, sys, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tapio

ap = argparse.ArgumentParser(); ap.add_argument("runs"); ap.add_argument("out"); ap.add_argument("--markers", required=True)
ap.add_argument("--dumps", default=None); ap.add_argument("--layers", default="6,18,30,47"); ap.add_argument("--stream", default="mean")
A = ap.parse_args(); layers = [int(x) for x in A.layers.split(",")]; os.makedirs(A.out, exist_ok=True)
markers = json.load(open(A.markers))                                   # {"MOVE:": [[ids], [ids], ...], ...}
variants = [v for vs in markers.values() for v in vs if v]


def act_index(stepped):
    """The step whose state writes the token after the last marker occurrence: the index of the marker's last token."""
    best = None
    for v in variants:
        L = len(v)
        for i in range(len(stepped) - L, 0, -1):
            if stepped[i:i + L] == v:
                j = i + L - 1
                if best is None or j > best: best = j
                break
    return best


rows = []; X = {L: {k: [] for k in ("s0", "mean", "last", "act")} for L in layers}
for f in sorted(glob.glob(f"{A.runs}/*.json")):
    run = json.load(open(f)); name = os.path.basename(f)[:-5]; m = run["meta"]; R = run["rounds"]
    for c in run["calls"]:
        if A.dumps:
            d = f"{A.dumps}/{name}/{c['i']}"
            if not os.path.exists(f"{d}/index.tsv"): continue
            st = tapio.read_dump(d); stepped = st["tokens"][-st["kept"]:]
        else:
            t = c.get("tap")
            if not (t and t.get("dir") and os.path.exists(t["dir"])): continue
            st = tapio.read_tap(t["dir"]); stepped = st["stepped"]
        k = act_index(stepped); r = c["round"]
        e_now = R[r].get("normalised") if r < len(R) else None; e_prev = R[r - 1].get("normalised") if r > 0 else None
        rows.append(dict(run=name, game=m["game"], cond=m["cond"], seed=m["seed"], round=r, agent=c["agent"], i=c["i"], T=len(stepped), act_k=k,
                         e_now=np.nan if e_now is None else e_now, e_prev=np.nan if e_prev is None else e_prev))
        for L in layers:
            x = st["states"][L]; x = x.mean(1) if A.stream == "mean" else x.reshape(x.shape[0], -1)
            X[L]["s0"].append(x[0]); X[L]["mean"].append(x.mean(0)); X[L]["last"].append(x[-1])
            X[L]["act"].append(x[k] if k is not None and k < len(x) else np.full(x.shape[1], np.nan, np.float32))
meta = {k: np.array([r[k] for r in rows]) for k in rows[0]}
for L in layers:
    np.savez_compressed(f"{A.out}/states_L{L}.npz", **{k: np.stack(v).astype(np.float16) for k, v in X[L].items()}, **{f"meta_{k}": v for k, v in meta.items()})
found = sum(r["act_k"] is not None for r in rows)
print(f"{len(rows)} calls, act marker found in {found} ({100 * found / max(len(rows), 1):.0f} %); layers {layers}; stream {A.stream} -> {A.out}")
