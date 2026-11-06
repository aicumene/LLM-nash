"""
p-beauty contest на Qwen3.6 (vLLM) — тест сходимости ИИ-агентов к равновесию Нэша.

N игроков выбирают число в [0,100]; цель = p·(среднее); победитель — ближайший к цели.
При p<1 игра dominance-solvable: ЕДИНСТВЕННОЕ равновесие Нэша = ВСЕ выбирают 0.
Итерируем R раундов (агенты видят прошлые выборы и пересматривают). Меряем:
  * среднее по раунду -> distance-to-NE (= |среднее - 0|): сходимость к равновесию;
  * глубину рассуждения (level-k): round-1 среднее (L0≈50, L1≈33, L2≈22, ..., NE=0).

Старт на одной модели (Qwen, N инстансов через сэмплинг). Дальше — Qwen vs Gemma-4
(гетерогенные приоры) + exploitability + суррогатный контроль.
vLLM: http://localhost:8000/v1, model Qwen3.6-27B-AWQ. Запуск на V100-машине.
"""
from __future__ import annotations

import re
import statistics

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
MODEL = "Qwen3.6-27B-AWQ"
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}


def guess(n_players: int, p: float, history: list[list[float]]) -> float:
    sys = (f"You are a contestant in a guessing game with {n_players} players. "
           f"Each player picks a number in [0,100]. The winner is the player whose pick "
           f"is closest to {p:.3f} times the AVERAGE of all picks. "
           f"Reason briefly (one or two sentences), then put your pick on the LAST line "
           f"exactly as 'GUESS: <number>'.")
    msgs = [{"role": "system", "content": sys}]
    if history:
        h = "\n".join(f"round {i + 1}: picks = {g}, avg = {statistics.mean(g):.1f}, "
                      f"target = {p * statistics.mean(g):.1f}" for i, g in enumerate(history))
        msgs.append({"role": "user", "content": "History so far:\n" + h + "\n\nGive your next pick."})
    else:
        msgs.append({"role": "user", "content": "Round 1. Give your pick."})
    r = client.chat.completions.create(model=MODEL, messages=msgs, max_tokens=220,
                                       temperature=0.7, top_p=0.95, extra_body=NOTHINK)
    txt = r.choices[0].message.content or ""
    m = re.findall(r"GUESS:\s*(-?[0-9]+(?:\.[0-9]+)?)", txt) or re.findall(r"(-?[0-9]+(?:\.[0-9]+)?)", txt)
    val = float(m[-1]) if m else 50.0
    return max(0.0, min(100.0, val))


if __name__ == "__main__":
    import sys
    N, R = 5, 6
    p = float(sys.argv[1]) if len(sys.argv) > 1 else 2 / 3
    ne = 0.0 if p < 1 else 100.0          # p<1 -> NE=0; p>1 -> NE=100 (контаминация-проба)
    print(f"p-beauty: N={N}, p={p:.3f}; равновесие Нэша = {ne:.0f}. Сходимость среднего к {ne:.0f} = к NE.\n")
    print(f"{'round':>5} {'picks':>34} {'mean':>6} {'target':>7} {'dist→NE':>8}")
    history: list[list[float]] = []
    for rd in range(R):
        picks = [round(guess(N, p, history), 1) for _ in range(N)]
        history.append(picks)
        mean = statistics.mean(picks)
        print(f"{rd + 1:>5} {str(picks):>34} {mean:>6.1f} {p * mean:>7.1f} {abs(mean - ne):>8.1f}")
    m1 = statistics.mean(history[0])
    print(f"\nГлубина (round-1 среднее = {m1:.1f}); NE={ne:.0f}. Если p>1 и среднее НЕ растёт к 100 -> контаминация.")
    print(f"Сходимость: {m1:.1f} → {statistics.mean(history[-1]):.1f} за {R} раундов (цель NE={ne:.0f})")
