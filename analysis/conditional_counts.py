"""The memory-one estimates of the preprint from the stored games, and cooperation by own last move x opponent's last move (Section 4.2).
A tit-for-tat player has gamma = 1 against any opponent, so the fall of gamma against a random defector has to come from something else;
these counts show what. Standard library only.
    python3 analysis/conditional_counts.py"""
import collections, glob, json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAY = {("C", "C"): 3, ("C", "D"): 0, ("D", "C"): 5, ("D", "D"): 1}


def report(name, games):
    """games: lists of (agent's move, opponent's move)."""
    last = collections.Counter(); n = own = opp = pay = 0
    for g in games:
        for t, (a, b) in enumerate(g):
            n += 1; own += a == "C"; opp += b == "C"; pay += PAY[(a, b)]
            if t: last[(g[t - 1][0], g[t - 1][1], a)] += 1
    k = lambda o, p: (last[(o, p, "C")], last[(o, p, "C")] + last[(o, p, "D")])
    gc, gn = k("C", "C")[0] + k("D", "C")[0], k("C", "C")[1] + k("D", "C")[1]
    ac, an = k("C", "D")[0] + k("D", "D")[0], k("C", "D")[1] + k("D", "D")[1]
    br = 5 * opp / n + (1 - opp / n)
    print(f"{name}: {len(games)} games, {n} rounds; own C {own / n:.3f}, opponent C {opp / n:.3f}")
    print(f"  gamma {gc}/{gn}" + (f" = {gc / gn:.3f}" if gn else "") + f"; alpha {ac}/{an}" + (f" = {ac / an:.3f}" if an else ""))
    print(f"  realized payoff {pay / n:.3f}, always-defect against this record {br:.3f}, regret {(br - pay / n) / br:.1%}")
    for o in "CD":
        for p in "CD":
            c, m = k(o, p); print(f"  own last {o}, opponent last {p}: cooperate {c}/{m}")


def v2(pattern):
    out = []
    for d in ("runs", "runs_tap"):
        for f in sorted(glob.glob(os.path.join(ROOT, "v2", "results", d, pattern))):
            out.append([tuple(r["moves"]) for r in json.load(open(f))["rounds"]])
    return out


v1 = json.load(open(os.path.join(ROOT, "v1", "nash_recheck2_out.json")))
report("Qwen 3.6, random defector (v1)", [[tuple(m) for m in g["history"]] for g in v1["random"]])
report("Qwen 3.6, single betrayal (v1)", [[tuple(m) for m in g["history"]] for g in v1["single"]])
report("Qwen 3.8, random defector", v2("pd_script_random35_s*.json"))
report("Qwen 3.8, always-C script", v2("pd_script_allC_s*.json"))
both = lambda gs: gs + [[(b, a) for a, b in g] for g in gs]                 # in self-play both players are agents
report("Qwen 3.8, self-play, loaded labels", both(v2("pd_loaded_s*.json")))
report("Qwen 3.8, self-play, neutral labels", both(v2("pd_neutral_s*.json")))
