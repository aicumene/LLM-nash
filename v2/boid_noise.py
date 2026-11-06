"""The signature's null levels on short windows: white Gaussian and AR(1) (phi = 0.9) trajectories of the same shape as the
reply arrays (N × T × d), per T used by game. Usage: boid_noise.py OUT_DIR [--N 200] [--d 2560] [--T 24,40,48] [--procs 8]"""
import os, sys, json, argparse, subprocess, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--N", type=int, default=200); ap.add_argument("--d", type=int, default=2560)
ap.add_argument("--T", default="24,40,48"); ap.add_argument("--procs", type=int, default=8); A = ap.parse_args()
os.makedirs(A.out, exist_ok=True); here = os.path.dirname(os.path.abspath(__file__)); rng = np.random.default_rng(7); summary = {}
for T in [int(x) for x in A.T.split(",")]:
    for kind in ("white", "ar09"):
        Z = rng.standard_normal((A.N, T, A.d)).astype(np.float32)
        if kind == "ar09":
            for t in range(1, T): Z[:, t] = 0.9 * Z[:, t - 1] + np.sqrt(1 - 0.81) * Z[:, t]
        tag = f"noise_{kind}_T{T}"; npz = f"{A.out}/Z_{tag}.npz"; np.savez_compressed(npz, Z=Z.astype(np.float16))
        r = subprocess.run([sys.executable, f"{here}/signature_lm.py", tag, npz, "Z", "--n", str(A.N), "--procs", str(A.procs)],
                           env=dict(os.environ, SIG_OUT=A.out, OMP_NUM_THREADS="1"), capture_output=True, text=True)
        try: summary[tag] = json.load(open(f"{A.out}/signature_{tag}.json"))["summary"]; print(tag, "done", flush=True)
        except Exception as e: print(tag, "failed:", e, r.stderr[-200:], flush=True)
json.dump(summary, open(f"{A.out}/NOISE.json", "w"), indent=1)
lines = ["| control | T | PR | r95 | d_osc | ESPRIT modes / sustained | sync | MC-SSA n_osc | speed contrast | operator q=1 / q=8 | sustained rotating |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for tag, s in sorted(summary.items()):
    sp, op = s["speed"], s["operator"]
    lines.append(f"| {tag} | {s['T']} | {s['pr']:.1f} | {s['r95']:.0f} | {s['d_osc']:.1f} | {s['n_modes']:.0f} / {s['n_sust']:.0f} | {s['sync']:.3f} | {s['mc_n_osc']} | {sp['contrast']:.1f} | {op['share_q1']:.2f} / {op['share_q8']:.2f} | {op['n_sustained_rotating']} |")
open(f"{A.out}/NOISE.md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
