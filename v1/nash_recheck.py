#!/usr/bin/env python3
"""nash_recheck.py — fresh PD move data to test the Markov-strategy totally-mixed Nash
equilibrium (Menshikov/Shklover/Babkina/Myagkov 2017, PLOS ONE) against the LLM PD runs in
aicumene-llm-nash-article.md, using pd.py's OWN payoff matrix (T=5,R=3,P=1,S=0), not the
paper's (T=10,R=5,P=1,S=0) — the curve was re-derived for T=5,R=3,P=1,S=0 separately.

Self-play: N_GAMES_SELF x R_SELF rounds, Qwen vs Qwen (pd.py's exact prompt/labels).
Cross-family: N_GAMES_CROSS x R_CROSS rounds, Qwen vs Gemma4 (xgame_verbose.py's setup,
moves only, no reasoning capture - just need the C/D sequence).

For each transition round i>0: classify by OPPONENT's move at round i-1.
  opponent prev = C -> a "coop opportunity" (Ncoop++); own move C -> Nrecoop++
  opponent prev = D -> a "default opportunity" (Ndefault++); own move C -> Ntolerant++
alpha = Ntolerant/Ndefault (tolerance to defection), gamma = Nrecoop/Ncoop (reciprocal cooperation)
— exactly Table 6's estimator in the 2017 paper.

Writes raw JSON (full round sequences + counts) to nash_recheck_out.json for audit.
"""
from __future__ import annotations

import json
import time
import urllib.request

PAY = {("C", "C"): (3, 3), ("C", "D"): (0, 5), ("D", "C"): (5, 0), ("D", "D"): (1, 1)}
QWEN = {"endpoint": "http://localhost:8000/v1", "model": "Qwen3.6-27B-AWQ",
        "extra": {"chat_template_kwargs": {"enable_thinking": False}}}
import os; QWEN_URL = os.environ.get("QWEN_URL", "http://localhost:8000/v1"); OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/v1")
GEMMA = {"endpoint": OLLAMA_URL, "model": "gemma4:31b", "extra": {}}

N_GAMES_SELF, R_SELF = 4, 20
N_GAMES_CROSS, R_CROSS = 3, 12


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


def move(agent, history, me, rounds, cl="Cooperate", dl="Defect"):
    payline = (f"Both {cl}: you 3. You {cl}/they {dl}: you 0, they 5. "
               f"You {dl}/they {cl}: you 5, they 0. Both {dl}: you 1.")
    sysmsg = (f"You are Player {me} in a {rounds}-round game vs the SAME opponent. "
              f"Each round both pick {cl} or {dl}. Payoffs: {payline} Maximize YOUR total. "
              f"Think one short sentence, then end with exactly 'MOVE: {cl}' or 'MOVE: {dl}'.")
    msgs = [{"role": "system", "content": sysmsg}]
    if history:
        lines = [f"round {i + 1}: you={(a if me == '1' else b)}, opponent={(b if me == '1' else a)}"
                 for i, (a, b) in enumerate(history)]
        msgs.append({"role": "user", "content": "History:\n" + "\n".join(lines) + "\nYour move."})
    else:
        msgs.append({"role": "user", "content": "Round 1. Your move."})
    return parse_move(chat(agent, msgs), cl, dl)


def play(agent1, agent2, rounds):
    history = []
    for _ in range(rounds):
        m1 = move(agent1, history, "1", rounds)
        m2 = move(agent2, history, "2", rounds)
        history.append((m1, m2))
    return history


def counts_from_games(games, seat):
    """seat=0 -> track player-1's stream (reacting to player-2's prev move); seat=1 -> reverse.
    seat='both' -> pool both seats (self-play)."""
    Ncoop = Nrecoop = Ndefault = Ntolerant = 0
    seats = [0, 1] if seat == "both" else [seat]
    for h in games:
        for s in seats:
            for i in range(1, len(h)):
                own = h[i][s]
                opp_prev = h[i - 1][1 - s]
                if opp_prev == "C":
                    Ncoop += 1
                    Nrecoop += (own == "C")
                else:
                    Ndefault += 1
                    Ntolerant += (own == "C")
    return {"Ncoop": Ncoop, "Nrecoop": Nrecoop, "Ndefault": Ndefault, "Ntolerant": Ntolerant,
            "alpha": (Ntolerant / Ndefault) if Ndefault else None,
            "gamma": (Nrecoop / Ncoop) if Ncoop else None}


def main():
    out = {"self_play": [], "cross_family": []}

    print(f"=== self-play Qwen x Qwen: {N_GAMES_SELF} games x {R_SELF} rounds ===")
    self_games = []
    for g in range(N_GAMES_SELF):
        h = play(QWEN, QWEN, R_SELF)
        self_games.append(h)
        seq = " ".join(f"{a}{b}" for a, b in h)
        coop = sum(m == "C" for a, b in h for m in (a, b)) / (2 * R_SELF)
        print(f"  game {g + 1}: {seq}   coop={coop * 100:.0f}%")
        out["self_play"].append(h)

    self_stats = counts_from_games(self_games, "both")
    print(f"\nself-play pooled: {self_stats}")

    print(f"\n=== cross-family Qwen vs Gemma4: {N_GAMES_CROSS} games x {R_CROSS} rounds ===")
    cross_games = []
    for g in range(N_GAMES_CROSS):
        h = play(QWEN, GEMMA, R_CROSS)
        cross_games.append(h)
        seq = " ".join(f"{a}{b}" for a, b in h)
        pQ = sum(PAY[mv][0] for mv in h)
        pG = sum(PAY[mv][1] for mv in h)
        print(f"  game {g + 1}: {seq}   Qwen={pQ} Gemma={pG}")
        out["cross_family"].append(h)

    qwen_cross_stats = counts_from_games(cross_games, 0)
    gemma_cross_stats = counts_from_games(cross_games, 1)
    print(f"\nQwen (cross-family) stats:  {qwen_cross_stats}")
    print(f"Gemma (cross-family) stats: {gemma_cross_stats}")

    out["stats"] = {"self_play_pooled": self_stats, "qwen_cross": qwen_cross_stats,
                    "gemma_cross": gemma_cross_stats}
    with open("nash_recheck_out.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote nash_recheck_out.json")


if __name__ == "__main__":
    main()
