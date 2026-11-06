#!/usr/bin/env python3
"""nash_recheck2.py — probe Qwen's ALPHA (tolerance to defection) against a SCRIPTED opponent.

nash_recheck.py found gamma=1.0 exactly and alpha UNOBSERVABLE (self-play/cross-family never
produced a single defection). This probes alpha directly: player 2's moves are a pre-defined
script, not an LLM — Qwen (player 1) gets the IDENTICAL prompt/history framing as before and
cannot tell the opponent is scripted, so this is a clean, exogenous measurement of Qwen's own
reactive policy, uncontaminated by an opponent that is itself trying to reach equilibrium.

Two conditions:
  RANDOM   - opponent defects each round iid with p=0.35 (seeded, recorded) -> the main pooled
             alpha,gamma estimator (Menshikov et al. 2017's Ncoop/Nrecoop/Ndefault/Ntolerant).
  SINGLE   - opponent cooperates every round except ONE scripted defection, at an early / mid /
             late round -> an event-study of immediate reaction (memory-1 alpha) AND whether
             cooperation recovers after 1 round (memory-1) or stays suppressed longer (grim-
             trigger / longer memory - outside the memory-1 Markov-strategy model).

Writes nash_recheck2_out.json (full sequences + both analyses) for audit.
"""
from __future__ import annotations

import json
import random
import time
import urllib.request

PAY = {("C", "C"): (3, 3), ("C", "D"): (0, 5), ("D", "C"): (5, 0), ("D", "D"): (1, 1)}
QWEN = {"endpoint": "http://localhost:8000/v1", "model": "Qwen3.6-27B-AWQ",
        "extra": {"chat_template_kwargs": {"enable_thinking": False}}}

N_RANDOM_GAMES, R_RANDOM, P_DEFECT = 4, 24, 0.35
SINGLE_CONFIGS = [("early", 14, 3), ("mid", 14, 7), ("late", 14, 10)]


def chat(agent, messages, max_tokens=160, temperature=0.7, retries=3):
    body = {"model": agent["model"], "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, **agent["extra"]}
    data = json.dumps(body).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(agent["endpoint"] + "/chat/completions", data=data,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=90) as r:
                d = json.load(r)
            return (d["choices"][0]["message"].get("content") or "").strip()
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                print(f"  !! chat failed after {retries} tries: {e}")
                return ""
            time.sleep(2)
    return ""


def parse_move(txt, cl="Cooperate", dl="Defect"):
    t = txt.upper()
    tail = t.split("MOVE:")[-1]
    if dl.upper() in tail and cl.upper() not in tail:
        return "D"
    if cl.upper() in tail and dl.upper() not in tail:
        return "C"
    return "D" if t.rfind(dl.upper()) > t.rfind(cl.upper()) else "C"


def qwen_move(history, rounds, cl="Cooperate", dl="Defect"):
    """Player 1 = Qwen. Identical prompt framing to pd.py/xgame_verbose.py; the opponent's
    scripted moves are fed back through the SAME 'opponent=...' history line, indistinguishable
    from a live opponent."""
    payline = (f"Both {cl}: you 3. You {cl}/they {dl}: you 0, they 5. "
               f"You {dl}/they {cl}: you 5, they 0. Both {dl}: you 1.")
    sysmsg = (f"You are Player 1 in a {rounds}-round game vs the SAME opponent. "
              f"Each round both pick {cl} or {dl}. Payoffs: {payline} Maximize YOUR total. "
              f"Think one short sentence, then end with exactly 'MOVE: {cl}' or 'MOVE: {dl}'.")
    msgs = [{"role": "system", "content": sysmsg}]
    if history:
        lines = [f"round {i + 1}: you={a}, opponent={b}" for i, (a, b) in enumerate(history)]
        msgs.append({"role": "user", "content": "History:\n" + "\n".join(lines) + "\nYour move."})
    else:
        msgs.append({"role": "user", "content": "Round 1. Your move."})
    return parse_move(chat(QWEN, msgs), cl, dl)


def play_vs_scripted(opp_pattern, rounds):
    history = []
    for i in range(rounds):
        m1 = qwen_move(history, rounds)
        m2 = opp_pattern[i]
        history.append((m1, m2))
    return history


def random_pattern(rounds, p, seed):
    rng = random.Random(seed)
    return ["D" if rng.random() < p else "C" for _ in range(rounds)]


def single_defection_pattern(rounds, defect_at):
    return ["D" if i == defect_at else "C" for i in range(rounds)]


def counts(games):
    """Qwen's (player 1) alpha,gamma from its OWN reaction stream vs the scripted opponent."""
    Ncoop = Nrecoop = Ndefault = Ntolerant = 0
    for h in games:
        for i in range(1, len(h)):
            own, opp_prev = h[i][0], h[i - 1][1]
            if opp_prev == "C":
                Ncoop += 1
                Nrecoop += (own == "C")
            else:
                Ndefault += 1
                Ntolerant += (own == "C")
    return {"Ncoop": Ncoop, "Nrecoop": Nrecoop, "Ndefault": Ndefault, "Ntolerant": Ntolerant,
            "alpha": (Ntolerant / Ndefault) if Ndefault else None,
            "gamma": (Nrecoop / Ncoop) if Ncoop else None}


def event_study(history, defect_at):
    """For a single-defection game: Qwen's move at defect_at+1 (immediate reaction) vs its
    cooperation rate over the rest of the game post-recovery-window (rounds >= defect_at+3,
    where the opponent has resumed C) - tests memory-1 (recovers immediately) vs longer memory."""
    immediate = history[defect_at + 1][0] if defect_at + 1 < len(history) else None
    later = [history[i][0] for i in range(defect_at + 3, len(history))]
    later_coop_rate = (sum(m == "C" for m in later) / len(later)) if later else None
    return {"immediate_reaction": immediate, "later_rounds_n": len(later),
            "later_coop_rate": later_coop_rate}


def main():
    out = {"random": [], "single": []}

    print(f"=== RANDOM-defection probe: {N_RANDOM_GAMES} games x {R_RANDOM} rounds, p={P_DEFECT} ===")
    random_games = []
    for g in range(N_RANDOM_GAMES):
        pat = random_pattern(R_RANDOM, P_DEFECT, seed=1000 + g)
        h = play_vs_scripted(pat, R_RANDOM)
        random_games.append(h)
        seq = " ".join(f"{a}{b}" for a, b in h)
        print(f"  game {g + 1} (seed={1000+g}): {seq}")
        out["random"].append({"seed": 1000 + g, "pattern": pat, "history": h})

    rstats = counts(random_games)
    print(f"\nRANDOM condition Qwen stats: {rstats}")

    print(f"\n=== SINGLE-betrayal probe: {len(SINGLE_CONFIGS)} configs ===")
    single_results = []
    for name, rounds, defect_at in SINGLE_CONFIGS:
        pat = single_defection_pattern(rounds, defect_at)
        h = play_vs_scripted(pat, rounds)
        seq = " ".join(f"{a}{b}" for a, b in h)
        ev = event_study(h, defect_at)
        print(f"  [{name}] defect@round{defect_at+1}/{rounds}: {seq}")
        print(f"    immediate reaction (round {defect_at+2}): {ev['immediate_reaction']}   "
              f"later coop-rate (rounds {defect_at+4}-{rounds}): {ev['later_coop_rate']}")
        single_results.append({"name": name, "rounds": rounds, "defect_at": defect_at,
                               "history": h, "event_study": ev})
    out["single"] = single_results

    single_all_games = [r["history"] for r in single_results]
    sstats = counts(single_all_games)
    print(f"\nSINGLE condition Qwen stats (pooled): {sstats}")

    out["stats"] = {"random": rstats, "single_pooled": sstats}
    with open("nash_recheck2_out.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote nash_recheck2_out.json")


if __name__ == "__main__":
    main()
