"""measure.py over every run of a directory, with the states from the dumps (--dumps DIR/<run>/<call>) or the tap
(the calls' `tap.dir`), for several layers; then the aggregate per condition: rounds pooled over seeds, Spearman
correlations of the internal measures with exploitability, and per-condition means by round.
Usage: analyze_runs.py RUNS_DIR OUT_DIR [--dumps DIR] [--layers 6,18,30,47] [--stream mean] [--procs 8]"""
from __future__ import annotations
import json, glob, os, sys, argparse, numpy as np, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure, tapio

A = None
def one(f):
    run = json.load(open(f)); name = os.path.basename(f)[:-5]; out = {}
    def states_of(c):
        if A.dumps:
            d = f"{A.dumps}/{name}/{c['i']}"
            if not os.path.exists(f"{d}/index.tsv"): return None
            st = tapio.read_dump(d)
            ids = c.get("token_ids") or []
            if ids and st["tokens"][-len(ids):] != ids: st["token_mismatch"] = True
            return st
        t = c.get("tap"); return tapio.read_tap(t["dir"]) if t and t.get("dir") and os.path.exists(t["dir"]) else None
    for L in A.layers:
        try: out[L] = measure.run_measures(run, L, A.stream, states_of)
        except Exception as e: out[L] = dict(error=f"{type(e).__name__}: {e}"[:200])
    res = dict(meta=run["meta"], layers={str(L): v for L, v in out.items()})
    json.dump(res, open(f"{A.out}/{name}_measures.json", "w"), indent=1)
    return name, {L: len(v.get("rounds", [])) for L, v in out.items()}


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4: return float("nan")
    rx = np.argsort(np.argsort(x[m])); ry = np.argsort(np.argsort(y[m]))
    return float(np.corrcoef(rx, ry)[0, 1])


KEYS = ["reply_pr16", "reply_pr", "reply_T", "drift_s0", "drift_mean", "drift_last", "spread_s0", "spread_mean", "spread_last", "apr_mean"]


def aggregate(out_dir, layers):
    """Per condition and layer: Spearman of each measure with the normalised exploitability over the pooled rounds, and the
    measure's mean in the first and the last round (AGGREGATE.md); the per-round means of the key measures (ROUNDS.md)."""
    table, byround = {}, {}
    for f in sorted(glob.glob(f"{out_dir}/*_measures.json")):
        d = json.load(open(f)); m = d["meta"]; cond = f"{m['game']}_{m['cond']}"
        for L in layers:
            for r in d["layers"].get(str(L), {}).get("rounds", []):
                e = table.setdefault((cond, L), {"expl": [], "round": [], **{k: [] for k in KEYS}})
                e["expl"].append(r["normalised"] if r.get("normalised") is not None else np.nan); e["round"].append(r["round"])
                for k in KEYS: e[k].append(r.get(k, np.nan))
    lines = ["| condition | layer | n | " + " | ".join(f"ρ({k})" for k in KEYS) + " | " + " | ".join(f"{k} first→last" for k in KEYS) + " |",
             "|---|---|---|" + "---|" * (2 * len(KEYS))]
    for (cond, L), e in sorted(table.items()):
        rho = [spearman(e[k], e["expl"]) for k in KEYS]; R = np.array(e["round"]); fl = []
        for k in KEYS:
            v = np.array(e[k], float); a, b = v[R == R.min()], v[R == R.max()]
            fl.append(f"{np.nanmean(a):.3g}→{np.nanmean(b):.3g}" if np.isfinite(v).any() else "–")
        lines.append(f"| {cond} | {L} | {len(e['expl'])} | " + " | ".join(f"{x:+.2f}" if np.isfinite(x) else "–" for x in rho) + " | " + " | ".join(fl) + " |")
        rows = []
        for rr in sorted(set(R.tolist())):
            sel = R == rr; ex = np.array(e["expl"], float)[sel]
            rows.append(f"| {rr + 1} | {np.nanmean(ex):.2f} | " + " | ".join(f"{np.nanmean(np.array(e[k], float)[sel]):.3g}" if np.isfinite(np.array(e[k], float)[sel]).any() else "–" for k in KEYS) + " |")
        byround[(cond, L)] = rows
    text = "\n".join(lines); open(f"{out_dir}/AGGREGATE.md", "w").write(text + "\n")
    with open(f"{out_dir}/ROUNDS.md", "w") as f:
        for (cond, L), rows in sorted(byround.items()):
            f.write(f"\n### {cond}, layer {L}\n| round | expl | " + " | ".join(KEYS) + " |\n|---|---|" + "---|" * len(KEYS) + "\n" + "\n".join(rows) + "\n")
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("runs"); ap.add_argument("out"); ap.add_argument("--dumps", default=None)
    ap.add_argument("--layers", default="6,18,30,47"); ap.add_argument("--stream", default="mean"); ap.add_argument("--procs", type=int, default=8)
    A = ap.parse_args(); A.layers = [int(x) for x in A.layers.split(",")]; os.makedirs(A.out, exist_ok=True)
    files = sorted(glob.glob(f"{A.runs}/*.json"))
    with mp.get_context("fork").Pool(A.procs) as pool:
        for name, n in pool.imap_unordered(one, files): print(name, n, flush=True)
    print(aggregate(A.out, A.layers))
