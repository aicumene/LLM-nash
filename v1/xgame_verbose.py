"""
Verbose кросс-машинный PD: захватываем РАССУЖДЕНИЯ агентов —
что одна модель (Qwen на tesla) думает про другую (Gemma-4 на Thor) и наоборот.
Печатаем для каждого раунда: ход + что агент думает об оппоненте/почему так ходит.
Gemma — thinking-модель: её внутренний монолог в reasoning_content.
"""
from __future__ import annotations

import json
import urllib.request

PAY = {("C", "C"): (3, 3), ("C", "D"): (0, 5), ("D", "C"): (5, 0), ("D", "D"): (1, 1)}


class Agent:
    def __init__(self, name, endpoint, model, extra=None):
        self.name, self.endpoint, self.model, self.extra = name, endpoint, model, extra or {}

    def chat(self, messages, max_tokens=500, temperature=0.7):
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature, **self.extra}
        req = urllib.request.Request(self.endpoint + "/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.load(r)
        m = d["choices"][0]["message"]
        return (m.get("content") or "").strip(), (m.get("reasoning_content") or m.get("reasoning") or "").strip()


import os; QWEN_URL = os.environ.get("QWEN_URL", "http://localhost:8000/v1"); OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/v1")
QWEN = Agent("Qwen3.6 (tesla)", QWEN_URL, "Qwen3.6-27B-AWQ",
             extra={"chat_template_kwargs": {"enable_thinking": False}})
GEMMA = Agent("Gemma-4-31B (Thor)", OLLAMA_URL, "gemma4:31b")


def parse_move(txt, cl, dl):
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
              f"In ONE sentence say what you think the OTHER player is like / will do, then "
              f"your reason, then end with exactly 'MOVE: {cl}' or 'MOVE: {dl}'.")
    msgs = [{"role": "system", "content": sysmsg}]
    if history:
        lines = [f"round {i + 1}: you={(a if me == '1' else b)}, opponent={(b if me == '1' else a)}"
                 for i, (a, b) in enumerate(history)]
        msgs.append({"role": "user", "content": "History:\n" + "\n".join(lines) + "\nYour move."})
    else:
        msgs.append({"role": "user", "content": "Round 1. Your move."})
    content, reasoning = agent.chat(msgs)
    return parse_move(content, cl, dl), content, reasoning


def clip(s, n=320):
    s = " ".join(s.split())
    return s[:n] + ("…" if len(s) > n else "")


if __name__ == "__main__":
    R = 6
    history = []
    print(f"PD verbose: Qwen (tesla) vs Gemma-4 (Thor), {R} раундов, метки Cooperate/Defect\n")
    for rd in range(R):
        m1, c1, r1 = move(QWEN, history, "1", R)
        m2, c2, r2 = move(GEMMA, history, "2", R)
        history.append((m1, m2))
        print(f"━━━ РАУНД {rd + 1}:  Qwen={m1}   Gemma={m2} ━━━")
        print(f"  QWEN  про оппонента/ход: {clip(c1)}")
        print(f"  GEMMA думает (reasoning): {clip(r2, 380)}")
        print(f"  GEMMA итог: {clip(c2, 160)}\n")
    coop = sum(m == "C" for a, b in history for m in (a, b)) / (2 * R)
    print(f"Итог: {' '.join(a + b for a, b in history)}  | кооп {coop * 100:.0f}%  "
          f"| Qwen={sum(PAY[mv][0] for mv in history)} Gemma={sum(PAY[mv][1] for mv in history)}")
