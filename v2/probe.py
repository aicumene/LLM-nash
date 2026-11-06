"""A linear probe of the round's exploitability from the agent's internal state: PCA (fit on the training seeds) to k
components, ridge, leave-one-seed-out; the targets are this round's normalised exploitability (the outcome of the move the
state is about to write) and the previous round's (already in the prompt); a shuffle control. Per game and condition,
per layer and readout; also trained on one condition and tested on another of the same game (e.g. real -> surrogate).
Usage: probe.py STATES_DIR [--layers 6,18,30,47] [--k 8] [--alpha 10]"""
from __future__ import annotations
import glob, os, sys, argparse, json, numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("states"); ap.add_argument("--layers", default="6,18,30,47"); ap.add_argument("--k", type=int, default=8)
ap.add_argument("--alpha", type=float, default=10.0); A = ap.parse_args()


def fit_predict(Xtr, ytr, Xte, k, alpha):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    U, S, Vt = np.linalg.svd(Ztr - Ztr.mean(0), full_matrices=False); P = Vt[:k].T
    Ftr, Fte = (Ztr - Ztr.mean(0)) @ P, (Zte - Ztr.mean(0)) @ P
    G = Ftr.T @ Ftr + alpha * np.eye(k); w = np.linalg.solve(G, Ftr.T @ (ytr - ytr.mean()))
    return Fte @ w + ytr.mean()


def loso(X, y, seeds, k, alpha, rng=None):
    pred = np.full_like(y, np.nan)
    for s in np.unique(seeds):
        tr, te = seeds != s, seeds == s
        if tr.sum() < k + 2 or te.sum() == 0: continue
        ytr = y[tr].copy()
        if rng is not None: rng.shuffle(ytr)
        pred[te] = fit_predict(X[tr], ytr, X[te], k, alpha)
    m = np.isfinite(pred) & np.isfinite(y)
    if m.sum() < 4 or y[m].std() < 1e-9: return np.nan, np.nan                       # a constant target cannot be probed
    r2 = 1 - ((pred[m] - y[m]) ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum()
    rho = np.corrcoef(np.argsort(np.argsort(pred[m])), np.argsort(np.argsort(y[m])))[0, 1]
    return float(r2), float(rho)


lines = ["| game / condition | layer | readout | n | R² now | ρ now | R² prev | ρ prev | R² now, shuffled | cross: train→test ρ now |", "|---|---|---|---|---|---|---|---|---|---|"]
for L in [int(x) for x in A.layers.split(",")]:
    d = np.load(f"{A.states}/states_L{L}.npz"); game, cond, seed = d["meta_game"], d["meta_cond"], d["meta_seed"]
    e_now, e_prev = d["meta_e_now"].astype(float), d["meta_e_prev"].astype(float); rng = np.random.default_rng(0)
    for g in np.unique(game):
        conds = np.unique(cond[game == g])
        for c in conds:
            sel = (game == g) & (cond == c)
            for ro in ("s0", "act", "mean", "last"):
                X = d[ro][sel].astype(np.float32); ok = np.isfinite(X).all(1)
                res = []
                for y in (e_now[sel], e_prev[sel]):
                    m = ok & np.isfinite(y); res.append(loso(X[m], y[m], seed[sel][m], A.k, A.alpha) if m.sum() >= 8 else (np.nan, np.nan))
                m = ok & np.isfinite(e_now[sel]); sh = loso(X[m], e_now[sel][m], seed[sel][m], A.k, A.alpha, rng)[0] if m.sum() >= 8 else np.nan
                cross = []
                for c2 in conds:
                    if c2 == c: continue
                    sel2 = (game == g) & (cond == c2); X2 = d[ro][sel2].astype(np.float32); ok2 = np.isfinite(X2).all(1) & np.isfinite(e_now[sel2])
                    if m.sum() >= 8 and ok2.sum() >= 4 and e_now[sel][m].std() > 1e-9 and e_now[sel2][ok2].std() > 1e-9:
                        p = fit_predict(X[m], e_now[sel][m], X2[ok2], A.k, A.alpha); y2 = e_now[sel2][ok2]
                        rho2 = np.corrcoef(np.argsort(np.argsort(p)), np.argsort(np.argsort(y2)))[0, 1]
                        cross.append(f"{c2}: ρ {rho2:+.2f}")                                      # rank agreement: the scales differ between conditions
                lines.append(f"| {g}_{c} | {L} | {ro} | {int(m.sum())} | {res[0][0]:+.2f} | {res[0][1]:+.2f} | {res[1][0]:+.2f} | {res[1][1]:+.2f} | {sh:+.2f} | {'; '.join(cross) or '–'} |")
text = "\n".join(lines); open(f"{A.states}/PROBE.md", "w").write(text + "\n"); print(text)
