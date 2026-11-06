"""The BOID signature (signature_lm.py: PR, r95, shuffle-thresholded d_osc, ESPRIT sustained modes and periods, MC-SSA, synchrony,
speed phases, one linear operator) of the reply trajectories of the Nash runs, per condition and layer: the first T steps of each
reply (T by game), all seeds and rounds together; then per trajectory against the round and its exploitability.
Usage: boid_runs.py RUNS_DIR OUT_DIR [--dumps DIR] [--layers 6,18,30,47] [--stream mean] [--procs 8] [--T pd=24,pbeauty=40,braess=48]"""
from __future__ import annotations
import json, glob, os, sys, argparse, subprocess, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tapio

ap = argparse.ArgumentParser(); ap.add_argument("runs"); ap.add_argument("out"); ap.add_argument("--dumps", default=None)
ap.add_argument("--layers", default="6,18,30,47"); ap.add_argument("--stream", default="mean"); ap.add_argument("--procs", type=int, default=8)
ap.add_argument("--T", default="pd=24,pbeauty=40,braess=48"); ap.add_argument("--only", default="")
A = ap.parse_args(); layers = [int(x) for x in A.layers.split(",")]; TG = {k: int(v) for k, v in (x.split("=") for x in A.T.split(","))}
os.makedirs(A.out, exist_ok=True); here = os.path.dirname(os.path.abspath(__file__))

groups = {}
for f in sorted(glob.glob(f"{A.runs}/*.json")):
    run = json.load(open(f)); name = os.path.basename(f)[:-5]; m = run["meta"]; R = run["rounds"]; T = TG[m["game"]]
    if A.only and A.only not in name: continue
    for c in run["calls"]:
        if A.dumps:
            d = f"{A.dumps}/{name}/{c['i']}"
            if not os.path.exists(f"{d}/index.tsv"): continue
            st = tapio.read_dump(d)
        else:
            t = c.get("tap")
            if not (t and t.get("dir") and os.path.exists(t["dir"])): continue
            st = tapio.read_tap(t["dir"])
        g = groups.setdefault(f"{m['game']}_{m['cond']}", {L: [] for L in layers} | {"meta": []})
        x0 = st["states"][layers[0]]
        if x0.shape[0] < T: continue
        r = c["round"]; e = R[r].get("normalised") if r < len(R) else None
        g["meta"].append(dict(run=name, seed=m["seed"], round=r, agent=c["agent"], i=c["i"], T_full=int(x0.shape[0]), e_now=np.nan if e is None else e))
        for L in layers:
            x = st["states"][L]; x = x.mean(1) if A.stream == "mean" else x.reshape(x.shape[0], -1); g[L].append(x[:T].astype(np.float16))
summary = {}
for cond, g in groups.items():
    N = len(g["meta"]); print(f"== {cond}: {N} replies of >= {g[layers[0]][0].shape[0]} steps", flush=True)
    if N < 8: continue
    json.dump(g["meta"], open(f"{A.out}/meta_{cond}.json", "w"))
    for L in layers:
        tag = f"{cond}_L{L}"; npz = f"{A.out}/Z_{tag}.npz"; np.savez_compressed(npz, Z=np.stack(g[L]))
        env = dict(os.environ, SIG_OUT=A.out, OMP_NUM_THREADS="1")
        r = subprocess.run([sys.executable, f"{here}/signature_lm.py", tag, npz, "Z", "--n", str(N), "--procs", str(A.procs)], env=env, capture_output=True, text=True)
        last = [l for l in r.stdout.strip().split("\n") if l][-1] if r.stdout.strip() else r.stderr[-300:]
        print("  ", tag, last[:200], flush=True)
        try:
            d = json.load(open(f"{A.out}/signature_{tag}.json")); summary[tag] = d["summary"]
        except Exception as e: print("   no signature:", e, r.stderr[-300:])
json.dump(summary, open(f"{A.out}/SUMMARY.json", "w"), indent=1)
lines = ["| condition | layer | N | T | PR | r95 | d_osc | ESPRIT modes / sustained | periods (median) | sync | MC-SSA n_osc | speed contrast | fast phases | operator share q=1 / q=8 | sustained rotating |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for tag, s in sorted(summary.items()):
    sp, op = s["speed"], s["operator"]
    lines.append(f"| {tag.rsplit('_L', 1)[0]} | {tag.rsplit('_L', 1)[1]} | {s['N']} | {s['T']} | {s['pr']:.1f} | {s['r95']:.0f} | {s['d_osc']:.1f} | {s['n_modes']:.0f} / {s['n_sust']:.0f} | "
                 f"{s['periods_median'] if s['periods_median'] is None else round(s['periods_median'], 1)} | {s['sync']:.3f} | {s['mc_n_osc']} | {sp['contrast']:.1f} | {sp['n_fast_phases_mean']:.1f} | "
                 f"{op['share_q1']:.2f} / {op['share_q8']:.2f} | {op['n_sustained_rotating']} |")
open(f"{A.out}/BOID.md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
