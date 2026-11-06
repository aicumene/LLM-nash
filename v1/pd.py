"""
Итерированная дилемма заключённого на Qwen3.6 — к чему приходят ИИ-агенты:
к равновесию Нэша (взаимный дефект, 1/1) или к кооперации (3/3)?

+ КОНТАМИНАЦИЯ-проба: ОДНИ И ТЕ ЖЕ выплаты, но метки действий разные —
  гружёные «Cooperate/Defect» vs нейтральные «Option A/Option B».
  Если кооперируют только при гружёных метках -> заучили знаменитую PD,
  а не вычисляют стратегию по выплатам (как p=1.2 в p-beauty).

Выплаты (T>R>P>S = 5>3>1>0): CC=(3,3) награда; DD=(1,1) наказание=одноразовый Нэш;
CD: дефектор 5, кооператор 0. Конечный горизонт R раундов (backward induction -> all-D).
vLLM: localhost:8000, model Qwen3.6-27B-AWQ. Запуск на tesla (V100).
"""
from __future__ import annotations

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
MODEL = "Qwen3.6-27B-AWQ"
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}
PAY = {("C", "C"): (3, 3), ("C", "D"): (0, 5), ("D", "C"): (5, 0), ("D", "D"): (1, 1)}


def parse_move(txt: str, cl: str, dl: str) -> str:
    t = txt.upper()
    tail = t.split("MOVE:")[-1]
    if dl.upper() in tail and cl.upper() not in tail:
        return "D"
    if cl.upper() in tail and dl.upper() not in tail:
        return "C"
    return "D" if t.rfind(dl.upper()) > t.rfind(cl.upper()) else "C"


def move(labels: tuple[str, str], history: list[tuple[str, str]], me: str, rounds: int) -> str:
    cl, dl = labels
    payline = (f"Both choose {cl}: you get 3. You {cl} & they {dl}: you 0, they 5. "
               f"You {dl} & they {cl}: you 5, they 0. Both {dl}: you 1.")
    sysmsg = (f"You are Player {me}, playing {rounds} rounds against the SAME opponent. "
              f"Each round you BOTH simultaneously choose {cl} or {dl}. Per-round payoffs: {payline} "
              f"Goal: maximize YOUR OWN total points across all {rounds} rounds. "
              f"Think one short sentence, then end with exactly 'MOVE: {cl}' or 'MOVE: {dl}'.")
    msgs = [{"role": "system", "content": sysmsg}]
    if history:
        lines = []
        for i, (a, b) in enumerate(history):
            mine, opp = (a, b) if me == "1" else (b, a)
            lines.append(f"round {i + 1}: you={mine}, opponent={opp}")
        msgs.append({"role": "user", "content": "History:\n" + "\n".join(lines) + "\n\nYour move."})
    else:
        msgs.append({"role": "user", "content": "Round 1. Your move."})
    r = client.chat.completions.create(model=MODEL, messages=msgs, max_tokens=160,
                                       temperature=0.7, extra_body=NOTHINK)
    return parse_move(r.choices[0].message.content or "", cl, dl)


def play(labels: tuple[str, str], rounds: int = 8) -> list[tuple[str, str]]:
    history: list[tuple[str, str]] = []
    for _ in range(rounds):
        m1 = move(labels, history, "1", rounds)
        m2 = move(labels, history, "2", rounds)
        history.append((m1, m2))
    return history


if __name__ == "__main__":
    R = 8
    for name, labels in [("гружёные (Cooperate/Defect)", ("Cooperate", "Defect")),
                         ("нейтральные (Option A/Option B)", ("Option A", "Option B"))]:
        h = play(labels, rounds=R)
        seq = " ".join(f"{a}{b}" for a, b in h)
        coop = sum(m == "C" for a, b in h for m in (a, b)) / (2 * R)
        p1 = sum(PAY[mv][0] for mv in h)
        p2 = sum(PAY[mv][1] for mv in h)
        dd = sum((a, b) == ("D", "D") for a, b in h)
        print(f"\n=== {name} ===")
        print("раунды (P1P2):", seq)
        print(f"кооперация {coop * 100:.0f}%  |  очки P1={p1} P2={p2}  |  "
              f"взаимный дефект (Нэш) в {dd}/{R} раундах")
