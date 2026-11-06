"""Internal-state measures of the Nash runs (plan A). Per reply: the trajectory of the residual stream over the stepped
tokens (step 0 = the last prompt token, then the reply). Readouts of an agent's state in a round: `s0` (after the last
prompt token: the representation of the game before writing), `mean` (over the reply's tokens), `last` (after the last
token written). Per round, for each readout: the drift of each agent's state from its state in the previous round
(relative to the mean state norm) and, with ≥ 3 agents, their spread (mean distance to the centroid, relative) and
participation ratio — beside the round's exploitability. Per reply: PR of the whole trajectory, PR of its first 16
steps, the speed of the state per token and its contrast, the length."""
from __future__ import annotations
import json, sys, os, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tapio

READOUTS = ("s0", "mean", "last"); WIN = 16


def pr(X):
    X = X - X.mean(0); T = X.shape[0]
    if T < 3: return float("nan")
    s = np.linalg.svd(X, compute_uv=False) ** 2 / max(T - 1, 1)
    return float(s.sum() ** 2 / (s ** 2).sum()) if s.sum() > 0 else float("nan")


def reply_measures(X):
    T = X.shape[0]; out = dict(T=int(T))
    if T < 4: return out
    Z = (X - X.mean(0)) / (X.std(0) + 1e-6); sp = ((Z[1:] - Z[:-1]) ** 2).mean(1)
    q10, q90 = np.quantile(sp, 0.1), np.quantile(sp, 0.9)
    out.update(pr=pr(X), pr16=pr(X[1:1 + WIN]) if T >= WIN + 1 else float("nan"), speed_mean=float(sp.mean()),
               speed_contrast=float(q90 / max(q10, 1e-9)), drift=float(np.linalg.norm(X[-1] - X[0]) / (np.linalg.norm(X[1:] - X[:-1], axis=1).mean() + 1e-9)))
    return out


def flat(states, layer, stream):
    x = states[layer]
    return x.mean(1) if stream == "mean" else x.reshape(x.shape[0], -1)


def run_measures(run, layer, stream, states_of):
    calls = run["calls"]; rounds = run["rounds"]; per_call = {}; st = {k: {} for k in READOUTS}
    for c in calls:
        s = states_of(c)
        if s is None: continue
        X = flat(s["states"], layer, stream); m = reply_measures(X); m.update(agent=c["agent"], round=c["round"], token_mismatch=bool(s.get("token_mismatch")))
        per_call[c["i"]] = m; key = (c["round"], c["agent"])
        st["s0"][key] = X[0]; st["mean"][key] = X.mean(0); st["last"][key] = X[-1]
    norm = {k: float(np.mean([np.linalg.norm(v) for v in st[k].values()])) + 1e-9 for k in READOUTS if st[k]}
    out_rounds = []; R = sorted({r for r, _ in st["s0"]}); prev = None
    for r in R:
        agents = sorted(a for (rr, a) in st["s0"] if rr == r)
        row = dict(round=r, n_agents=len(agents), exploitability=rounds[r].get("exploitability") if r < len(rounds) else None,
                   normalised=rounds[r].get("normalised") if r < len(rounds) else None)
        pc = [per_call[i] for i in per_call if per_call[i]["round"] == r]
        for k in ("pr", "pr16", "speed_mean", "T"): row[f"reply_{k}"] = float(np.nanmean([m[k] for m in pc if k in m])) if pc else float("nan")
        for k in READOUTS:
            S = np.stack([st[k][(r, a)] for a in agents])
            if len(agents) >= 3:
                C = S - S.mean(0); row[f"spread_{k}"] = float(np.linalg.norm(C, axis=1).mean() / norm[k]); row[f"apr_{k}"] = pr(S)
            if prev is not None:
                common = [a for a in agents if (prev, a) in st[k]]
                if common: row[f"drift_{k}"] = float(np.mean([np.linalg.norm(st[k][(r, a)] - st[k][(prev, a)]) for a in common]) / norm[k])
        out_rounds.append(row); prev = r
    return dict(per_call=per_call, rounds=out_rounds, norms=norm)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--layer", type=int, default=30); ap.add_argument("--stream", default="mean")
    ap.add_argument("--dumps", default=None); A = ap.parse_args(); run = json.load(open(A.run))
    def states_of(c):
        if A.dumps:
            d = f"{A.dumps}/{c['i']}"; return tapio.read_dump(d) if os.path.exists(f"{d}/index.tsv") else None
        t = c.get("tap"); return tapio.read_tap(t["dir"]) if t and t.get("dir") and os.path.exists(t["dir"]) else None
    res = run_measures(run, A.layer, A.stream, states_of)
    for r in res["rounds"]: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
