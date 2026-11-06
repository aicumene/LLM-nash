"""Figures of the v2 additions, from the run JSONs and the per-run measures. Output: paper/figs/*.pdf + *.png.
Colours: the validated categorical order (blue, orange, aqua); text in ink; one axis per panel; direct labels + legend."""
import json, glob, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R = os.path.join(ROOT, "v2", "results"); OUT = os.path.join(ROOT, "analysis", "figs"); os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"; INK, INK2, GRID = "#0b0b0b", "#52514e", "#dddcd8"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False, "pdf.fonttype": 42})

def runs(pattern):
    out = []
    for d in ("runs", "runs_tap"):
        for f in sorted(glob.glob(f"{R}/{d}/{pattern}")): out.append(json.load(open(f)))
    return out

def band(ax, x, Y, color, label, ls="-"):
    Y = np.array(Y, float); m = np.nanmean(Y, 0)
    ax.fill_between(x, np.nanmin(Y, 0), np.nanmax(Y, 0), color=color, alpha=0.12, linewidth=0)
    ax.plot(x, m, color=color, lw=2, ls=ls, label=label, marker="o", ms=4, mfc="white", mew=1.5); return m

# ------------------------------------------------------------------ Figure A: the beauty contest by round, Qwen 3.8 against the v1 Qwen 3.6
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
for ax, p, cond, ne, v1 in ((axes[0], "2/3", "p0.667", 0, (6.1, 0.0)), (axes[1], "1.2", "p1.2", 100, (44, 20))):
    rr = runs(f"pbeauty_{cond}_s*.json"); x = np.arange(1, 7)
    m = band(ax, x, [[r_["mean"] for r_ in run["rounds"]] for run in rr], BLUE, f"Qwen 3.8 ({len(rr)} seeds)")
    if cond == "p0.667":
        rs = runs("pbeauty_surrogate_p0.667_s*.json")
        ms = band(ax, x, [[r_["picks"][0] for r_ in run["rounds"]] for run in rs], ORANGE, f"Qwen 3.8, surrogate others ({len(rs)} seeds)")
    ax.plot([1, 6], v1, color=AQUA, lw=2, ls=":", marker="s", ms=4, mfc="white", mew=1.5, label="Qwen 3.6, v1 (rounds 1 and 6)")
    ax.axhline(ne, color=INK2, lw=0.8, ls="--"); ax.text(1.0 if ne == 100 else 6.05, ne + (3 if ne == 0 else -7), f"equilibrium {ne}", color=INK2, fontsize=8, ha="left" if ne == 100 else "right")
    ax.set_title(f"p = {p}", loc="left", fontsize=10, color=INK); ax.set_xlabel("round"); ax.set_xticks(x); ax.set_ylim(-4, 104)
axes[0].set_ylabel("mean guess"); axes[1].legend(loc="lower right", fontsize=7.5)
axes[0].legend(loc="center right", fontsize=7.5)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_beauty.pdf"); fig.savefig(f"{OUT}/fig_beauty.png", dpi=160); plt.close(fig)

# ------------------------------------------------------------------ Figure B: the game representation's drift and the reply states' spread, by layer
def measure_rows(cond, L, key, src="measures2_dump"):
    rows = []
    for f in sorted(glob.glob(f"{R}/{src}/{cond}_s*_measures.json")):
        d = json.load(open(f)); rr = d["layers"][str(L)]["rounds"]; rows.append([r_.get(key, np.nan) for r_ in rr])
    return rows
layers = [6, 18, 30, 47]
fig, axes = plt.subplots(2, 4, figsize=(7.5, 4.2), sharex=True)
for j, L in enumerate(layers):
    ax = axes[0, j]; x = np.arange(2, 7)
    for cond, col, lab in (("pbeauty_p0.667", BLUE, "real, p = 2/3"), ("pbeauty_surrogate_p0.667", ORANGE, "surrogate others"), ("pbeauty_p1.2", AQUA, "probe, p = 1.2")):
        Y = np.array(measure_rows(cond, L, "drift_s0"), float)[:, 1:6]; band(ax, x, Y, col, lab)
    ax.set_title(f"layer {L}", loc="left", fontsize=9, color=INK); ax.set_ylim(0, 0.6)
    if j == 0: ax.set_ylabel("drift of the pre-reply state\n(relative to its norm)", fontsize=8)
    ax = axes[1, j]; x = np.arange(1, 7)
    for cond, col, lab in (("pbeauty_p0.667", BLUE, "real, p = 2/3"), ("pbeauty_p1.2", AQUA, "probe, p = 1.2")):
        Y = np.array(measure_rows(cond, L, "spread_last"), float)[:, :6]; band(ax, x, Y, col, lab)
    ax.set_ylim(0, 0.8 if L != 47 else 0.5); ax.set_xlabel("round"); ax.set_xticks(x)
    if j == 0: ax.set_ylabel("spread of the agents'\nreply states (relative)", fontsize=8)
axes[0, 0].legend(fontsize=7, loc="upper right"); axes[1, 0].legend(fontsize=7, loc="upper right")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_internal.pdf"); fig.savefig(f"{OUT}/fig_internal.png", dpi=160); plt.close(fig)

# ------------------------------------------------------------------ Figure C: the scripted betrayal as an event in the state (layer 18)
fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.6))
AT = {"early": 3, "mid": 7, "late": 10}; cols = {"early": BLUE, "mid": ORANGE, "late": AQUA}
for ax, key, lab in ((axes[0], "drift_s0", "drift of the pre-reply state"), (axes[1], "reply_T", "reply length (tokens)"), (axes[2], "reply_pr16", "PR of the reply's first 16 steps")):
    for kind in ("early", "mid", "late"):
        rows = np.array(measure_rows(f"pd_script_single_{kind}", 18, key), float); off = np.arange(14) - AT[kind]
        sel = (off >= -3) & (off <= 4); m = np.nanmean(rows, 0)[sel]; lo, hi = np.nanmin(rows, 0)[sel], np.nanmax(rows, 0)[sel]
        ax.fill_between(off[sel], lo, hi, color=cols[kind], alpha=0.10, linewidth=0); ax.plot(off[sel], m, color=cols[kind], lw=2, marker="o", ms=3.5, mfc="white", mew=1.2, label=f"betrayal at round {AT[kind] + 1}")
    ax.axvline(0, color=INK2, lw=0.8, ls="--"); ax.set_title(lab, loc="left", fontsize=9, color=INK); ax.set_xlabel("rounds after the betrayal"); ax.set_xticks(range(-3, 5))
axes[1].legend(fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_betrayal.pdf"); fig.savefig(f"{OUT}/fig_betrayal.png", dpi=160); plt.close(fig)
print("figures written:", sorted(os.listdir(OUT)))
