"""The BOID signature on long trajectories: (a) each agent's replies of a whole game concatenated in round order (the agent's
"life" over the game: 6–20 replies, hundreds of tokens; the first T_long steps kept, trajectories shorter than that dropped),
per condition and layer; (b) the long single replies of the Braess formula games (T_reply steps). Same estimators as
boid_runs.py (signature_lm.py). Usage: boid_long.py RUNS_DIR OUT_DIR [--dumps DIR] [--layers 18,30] [--T_long 240] [--T_reply 200] [--procs 8]"""
from __future__ import annotations
import json, glob, os, sys, argparse, subprocess, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tapio

ap = argparse.ArgumentParser(); ap.add_argument("runs"); ap.add_argument("out"); ap.add_argument("--dumps", default=None)
ap.add_argument("--layers", default="18,30"); ap.add_argument("--stream", default="mean"); ap.add_argument("--procs", type=int, default=8)
ap.add_argument("--T_long", type=int, default=240); ap.add_argument("--T_reply", type=int, default=200)
A = ap.parse_args(); layers = [int(x) for x in A.layers.split(",")]; os.makedirs(A.out, exist_ok=True); here = os.path.dirname(os.path.abspath(__file__))

def states_of(name, c):
    if A.dumps:
        d = f"{A.dumps}/{name}/{c['i']}"; return tapio.read_dump(d) if os.path.exists(f"{d}/index.tsv") else None
    t = c.get("tap"); return tapio.read_tap(t["dir"]) if t and t.get("dir") and os.path.exists(t["dir"]) else None

long_groups, reply_groups, meta_long = {}, {}, {}
for f in sorted(glob.glob(f"{A.runs}/*.json")):
    run = json.load(open(f)); name = os.path.basename(f)[:-5]; m = run["meta"]; cond = f"{m['game']}_{m['cond']}"
    per_agent = {}
    for c in run["calls"]:
        st = states_of(name, c)
        if st is None: continue
        for L in layers:
            x = st["states"][L]; x = x.mean(1) if A.stream == "mean" else x.reshape(x.shape[0], -1)
            per_agent.setdefault((c["agent"], L), []).append((c["round"], x[1:].astype(np.float16)))      # the reply's steps (step 0 = the prompt's last token)
            if m["game"] == "braess" and x.shape[0] - 1 >= A.T_reply:
                reply_groups.setdefault((cond, L), []).append(x[1:1 + A.T_reply].astype(np.float16))
    for (a, L), parts in per_agent.items():
        parts.sort(key=lambda p: p[0]); Z = np.concatenate([p[1] for p in parts], 0)
        if Z.shape[0] >= A.T_long:
            long_groups.setdefault((cond, L), []).append(Z[:A.T_long]); meta_long.setdefault(cond, []).append(dict(run=name, seed=m["seed"], agent=a, total=int(Z.shape[0])))
summary = {}
def run_sig(tag, Z):
    npz = f"{A.out}/Z_{tag}.npz"; np.savez_compressed(npz, Z=np.stack(Z))
    r = subprocess.run([sys.executable, f"{here}/signature_lm.py", tag, npz, "Z", "--n", str(len(Z)), "--procs", str(A.procs)],
                       env=dict(os.environ, SIG_OUT=A.out, OMP_NUM_THREADS="1"), capture_output=True, text=True)
    try: summary[tag] = json.load(open(f"{A.out}/signature_{tag}.json"))["summary"]; print(tag, len(Z), "ok", flush=True)
    except Exception as e: print(tag, "failed:", e, r.stderr[-200:], flush=True)
for (cond, L), Z in sorted(long_groups.items()):
    if len(Z) >= 6: run_sig(f"long_{cond}_L{L}", Z)
for (cond, L), Z in sorted(reply_groups.items()):
    if len(Z) >= 8: run_sig(f"reply{A.T_reply}_{cond}_L{L}", Z)
json.dump(dict(summary=summary, meta_long={k: v for k, v in meta_long.items()}), open(f"{A.out}/SUMMARY.json", "w"), indent=1)
lines = ["| trajectories | layer | N | T | PR | r95 | d_osc | ESPRIT modes / sustained | periods (median) | sync | MC-SSA n_osc | speed contrast | fast phases | operator q=1 / q=8 | sustained rotating |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for tag, s in sorted(summary.items()):
    sp, op = s["speed"], s["operator"]
    lines.append(f"| {tag.rsplit('_L', 1)[0]} | {tag.rsplit('_L', 1)[1]} | {s['N']} | {s['T']} | {s['pr']:.1f} | {s['r95']:.0f} | {s['d_osc']:.1f} | {s['n_modes']:.0f} / {s['n_sust']:.0f} | "
                 f"{s['periods_median'] if s['periods_median'] is None else round(s['periods_median'], 1)} | {s['sync']:.3f} | {s['mc_n_osc']} | {sp['contrast']:.1f} | {sp['n_fast_phases_mean']:.1f} | {op['share_q1']:.2f} / {op['share_q8']:.2f} | {op['n_sustained_rotating']} |")
open(f"{A.out}/BOID_LONG.md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
