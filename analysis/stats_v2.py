"""Numbers for the v2 text, in the paper's own definitions: normalized exploitability = relative regret of the most exploitable
agent against its best response (payoff games: (u(b) − u(a)) / u(b); cost games: (c(a) − c(b)) / c(b)); the memory-one estimator
(alpha, gamma) of Menshikov et al.; the reaction to a scripted betrayal. Over both Qwen 3.8 batches (5 + 3 seeds)."""
import json, glob, os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "v2")); import games as G
R = os.path.join(ROOT, "v2", "results")
def runs(pattern): return [json.load(open(f)) for d in ("runs", "runs_tap") for f in sorted(glob.glob(f"{R}/{d}/{pattern}"))]
out = {}
# ---- beauty contest
for cond in ("p0.667", "p1.2", "surrogate_p0.667"):
    rr = runs(f"pbeauty_{cond}_s*.json"); r1 = [r["rounds"][0]["mean"] for r in rr]; r6 = [r["rounds"][5]["mean"] for r in rr]
    if cond.startswith("surrogate"): r1 = [r["rounds"][0]["picks"][0] for r in rr]; r6 = [r["rounds"][5]["picks"][0] for r in rr]
    out[f"beauty_{cond}"] = dict(n=len(rr), round1=(min(r1), max(r1), float(np.mean(r1))), round6=(min(r6), max(r6), float(np.mean(r6))))
# ---- PD: outcomes, memory-one estimator, regret in the paper's normalisation
def pd_stats(cond):
    rr = runs(f"pd_{cond}_s*.json"); games_ = [[tuple(x["moves"]) for x in r["rounds"]] for r in rr]
    coop_all = sum(all(m == ("C", "C") for m in g) for g in games_); dd_from = []
    for g in games_:
        k = next((i for i in range(len(g)) if all(m == ("D", "D") for m in g[i:])), None); dd_from.append(None if k is None else k + 1)
    Ncoop = Nrecoop = Ndef = Ntol = 0; own_c = 0; n = 0; pay = 0.0; opp_c = 0
    for g in games_:
        for i in range(len(g)):
            a, b = g[i]; n += 1; own_c += a == "C"; opp_c += b == "C"; pay += G.PAY[(a, b)][0]
            if i == 0: continue
            if g[i - 1][1] == "C": Ncoop += 1; Nrecoop += a == "C"
            else: Ndef += 1; Ntol += a == "C"
    c_opp = opp_c / n; br = 5 * c_opp + 1 * (1 - c_opp)                       # always defect against a history-independent opponent
    return dict(games=len(games_), rounds=n, all_CC=coop_all, DD_from_round=dd_from, gamma=(Nrecoop, Ncoop, Nrecoop / Ncoop if Ncoop else None),
                alpha=(Ntol, Ndef, Ntol / Ndef if Ndef else None), own_coop=own_c / n, opp_coop=c_opp, payoff=pay / n, br_payoff=br, regret=(br - pay / n) / br)
for cond in ("loaded", "neutral", "script_allC", "script_random35"): out[f"pd_{cond}"] = pd_stats(cond)
# ---- the betrayal: reaction and recovery
AT = {"early": 3, "mid": 7, "late": 10}; react = []; recover = []; streak = []
for kind, at in AT.items():
    for r in runs(f"pd_script_single_{kind}_s*.json"):
        g = [tuple(x["moves"]) for x in r["rounds"]]
        if g[at - 1][0] == "D": continue                                     # already defecting before the betrayal: no reaction to measure
        react.append(g[at + 1][0] == "D")
        k = next((j for j in range(at + 1, len(g)) if g[j][0] == "C"), None); streak.append(len(g) - at - 1 if k is None else k - at - 1)
        recover.append(k is not None and k - at - 1 <= 2)
out["betrayal"] = dict(games_cooperating_before=len(react), retaliate_next_round=sum(react), back_to_C_within_2=sum(recover),
                       defection_streak_after=(sorted(streak), float(np.median(streak))))
# ---- Braess in the paper's normalisation
def eps_cost(routes, with_tg):
    worst = 0.0
    for i, r in enumerate(routes):
        others = routes[:i] + routes[i + 1:]; opt = G.option_times(others, with_tg); c = opt[r]; b = min(opt.values()); worst = max(worst, (c - b) / b)
    return worst
for cond in ("easy_noTG", "easy_TG", "hard_noTG", "hard_TG", "experience_TG"):
    rr = runs(f"braess_{cond}_s*.json"); with_tg = cond.endswith("_TG")
    finals = [eps_cost(r["rounds"][-1]["routes"], with_tg) for r in rr]; allp = [eps_cost(x["routes"], with_tg) for r in rr for x in r["rounds"]]
    ne = [sum(x["exploitability"] == 0 for x in r["rounds"]) for r in rr]
    out[f"braess_{cond}"] = dict(games=len(rr), eps_final=(min(finals), max(finals), float(np.mean(finals))), eps_all=(float(np.median(allp)), float(np.max(allp))), rounds_at_NE=ne,
                                 counts_final=[r["rounds"][-1]["counts"] for r in rr])
rr = runs("braess_surrogate_hard_TG_s*.json"); reg = [(x["options"][x["route"]] - min(x["options"].values())) / min(x["options"].values()) for r in rr for x in r["rounds"]]
out["braess_surrogate_hard_TG"] = dict(decisions=len(reg), argmin_rate=float(np.mean([x == 0 for x in reg])), mean_rel_regret=float(np.mean(reg)))
json.dump(out, open(os.path.join(ROOT, "analysis", "stats_v2.json"), "w"), indent=1, default=str)
for k, v in out.items(): print(k, json.dumps(v, default=str)[:300])
