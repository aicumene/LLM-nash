"""
Кросс-машинные / кросс-семейные экономические игры: агенты РАЗНЫХ семей на РАЗНЫХ
машинах играют друг против друга.

  Agent A = Qwen3.6-27B-AWQ  на tesla (V100, vLLM @ :8000)
  Agent B = gemma4:31b       на Thor   (ollama @ :11434, OpenAI-compat)

Оркестратор — на Mac (чистый stdlib urllib, без зависимостей), дотягивается до обеих.
Игры: итерированная дилемма заключённого (гружёные vs нейтральные метки). Вопрос —
ломает ли гетерогенность семей 100%-кооперацию, что была в Qwen-self-play.
"""
from __future__ import annotations

import json
import re
import urllib.request

PAY = {("C", "C"): (3, 3), ("C", "D"): (0, 5), ("D", "C"): (5, 0), ("D", "D"): (1, 1)}


class Agent:
    def __init__(self, name: str, endpoint: str, model: str, extra: dict | None = None):
        self.name, self.endpoint, self.model, self.extra = name, endpoint, model, extra or {}

    def chat(self, messages, max_tokens=300, temperature=0.7) -> str:
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature, **self.extra}
        req = urllib.request.Request(self.endpoint + "/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=240) as r:
            d = json.load(r)
        return (d["choices"][0]["message"].get("content") or "").strip()


import os; QWEN_URL = os.environ.get("QWEN_URL", "http://localhost:8000/v1"); OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/v1")
QWEN = Agent("Qwen3.6 (tesla/V100)", QWEN_URL, "Qwen3.6-27B-AWQ",
             extra={"chat_template_kwargs": {"enable_thinking": False}})
GEMMA = Agent("Gemma-4-31B (Thor)", OLLAMA_URL, "gemma4:31b")


def parse_move(txt: str, cl: str, dl: str) -> str:
    t = txt.upper()
    tail = t.split("MOVE:")[-1]
    if dl.upper() in tail and cl.upper() not in tail:
        return "D"
    if cl.upper() in tail and dl.upper() not in tail:
        return "C"
    return "D" if t.rfind(dl.upper()) > t.rfind(cl.upper()) else "C"


def pd_move(agent: Agent, labels, history, me: str, rounds: int) -> str:
    cl, dl = labels
    payline = (f"Both choose {cl}: you get 3. You {cl} & they {dl}: you 0, they 5. "
               f"You {dl} & they {cl}: you 5, they 0. Both {dl}: you 1.")
    sysmsg = (f"You are Player {me}, playing {rounds} rounds against the SAME opponent. "
              f"Each round you BOTH simultaneously choose {cl} or {dl}. Per-round payoffs: {payline} "
              f"Goal: maximize YOUR OWN total points over all {rounds} rounds. "
              f"End your reply with exactly 'MOVE: {cl}' or 'MOVE: {dl}'.")
    msgs = [{"role": "system", "content": sysmsg}]
    if history:
        lines = [f"round {i + 1}: you={(a if me == '1' else b)}, opponent={(b if me == '1' else a)}"
                 for i, (a, b) in enumerate(history)]
        msgs.append({"role": "user", "content": "History:\n" + "\n".join(lines) + "\n\nYour move."})
    else:
        msgs.append({"role": "user", "content": "Round 1. Your move."})
    try:
        return parse_move(agent.chat(msgs), cl, dl)
    except Exception as e:
        print(f"  [warn {agent.name}: {type(e).__name__} {str(e)[:60]} -> default C]")
        return "C"


def play_pd(a: Agent, b: Agent, labels, rounds=6):
    history: list[tuple[str, str]] = []
    for _ in range(rounds):
        m1 = pd_move(a, labels, history, "1", rounds)
        m2 = pd_move(b, labels, history, "2", rounds)
        history.append((m1, m2))
    return history


if __name__ == "__main__":
    print(f"PD кросс-машина: P1 = {QWEN.name}  vs  P2 = {GEMMA.name}\n")
    for name, labels in [("гружёные (Cooperate/Defect)", ("Cooperate", "Defect")),
                         ("нейтральные (Option A/Option B)", ("Option A", "Option B"))]:
        h = play_pd(QWEN, GEMMA, labels, rounds=6)
        seq = " ".join(f"{a}{b}" for a, b in h)
        coop = sum(m == "C" for a, b in h for m in (a, b)) / (2 * len(h))
        p1 = sum(PAY[mv][0] for mv in h)
        p2 = sum(PAY[mv][1] for mv in h)
        print(f"=== {name} ===")
        print(f"раунды (Qwen Gemma): {seq}")
        print(f"кооперация {coop * 100:.0f}%  |  Qwen={p1} Gemma={p2}  |  "
              f"взаимный дефект(Нэш) {sum((a, b) == ('D', 'D') for a, b in h)}/{len(h)}\n")
