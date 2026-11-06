"""
Парадокс Брайеса (Braess) на ИИ-агентах: воспроизводят ли селфиш-агенты плохое
равновесие, когда добавлена «лишняя» дорога Артем–Геймланд?

Сеть (Аэропорт A -> Владивосток V, 6 водителей), время дороги = f(числа машин y на ней):
  A->Геймланд(G): y+50   G->V: 10y   A->Артем(T): 10y   T->V: y+50   [T->G: y+10 — only with_TG]
Маршруты: R1 A-G-V ; R2 A-T-V ; R3 A-T-G-V (только когда T->G построена).
Теория: без T->G равновесие 3/3 -> 83 (систем. 498). С T->G равновесие 2/2/2 -> 92 (систем. 552)
— ВСЕМ ХУЖЕ, хотя добавили дорогу.

Динамика: ПОСЛЕДОВАТЕЛЬНЫЙ best-response (конгешн-игра потенциальная -> сходится к Нэшу).
Каждый шаг один агент (Qwen) видит точное время по каждому варианту при текущем положении
остальных и выбирает минимальное. Оркестр с Mac (stdlib urllib).
"""
from __future__ import annotations

import json
import re
import urllib.request

import os; QWEN_URL = os.environ.get("QWEN_URL", "http://localhost:8000/v1"); OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/v1")
ENDPOINT, MODEL = QWEN_URL, "Qwen3.6-27B-AWQ"
EXTRA = {"chat_template_kwargs": {"enable_thinking": False}}
N = 6


def chat(messages, max_tokens=120):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.4, **EXTRA}
    req = urllib.request.Request(ENDPOINT + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return (json.load(r)["choices"][0]["message"].get("content") or "").strip()


def loads_costs(choices, with_tg):
    n1, n2, n3 = choices.count("R1"), choices.count("R2"), choices.count("R3")
    y_AG, y_GV, y_AT, y_TV, y_TG = n1, n1 + n3, n2 + n3, n2, n3
    c = {"R1": (y_AG + 50) + 10 * y_GV, "R2": 10 * y_AT + (y_TV + 50)}
    if with_tg:
        c["R3"] = 10 * y_AT + (y_TG + 10) + 10 * y_GV
    return c


def option_times(others, with_tg):
    routes = ["R1", "R2", "R3"] if with_tg else ["R1", "R2"]
    return {r: loads_costs(others + [r], with_tg)[r] for r in routes}


def choose(others, with_tg):
    opt = option_times(others, with_tg)
    desc = "; ".join(f"{r}={t} min" for r, t in opt.items())
    sysmsg = ("You are a driver from Airport to Vladivostok choosing among routes. "
              "Pick the route with the LOWEST travel time FOR YOU. "
              "End with exactly 'ROUTE: R1' (or R2"
              + (" or R3)." if with_tg else ")."))
    user = (f"Given where the other {N - 1} drivers currently are, YOUR travel time on each route "
            f"would be: {desc}. Choose the fastest route for you.")
    txt = chat([{"role": "system", "content": sysmsg}, {"role": "user", "content": user}]).upper()
    m = re.findall(r"ROUTE:\s*(R[123])", txt) or re.findall(r"\b(R[123])\b", txt)
    r = m[-1] if m else min(opt, key=opt.get)
    return r if r in opt else min(opt, key=opt.get)


def run(with_tg, passes=4):
    routes = ["R2"] * N                       # стартовое распределение
    label = "С дорогой Артем–Геймланд (R3)" if with_tg else "Без дороги Артем–Геймланд"
    print(f"\n=== {label} ===")
    print(f"{'pass':>4} {'R1':>3} {'R2':>3} {'R3':>3} {'route times':>30} {'avg':>6} {'system':>7}")
    avg = sysc = 0.0
    for p in range(passes):
        for i in range(N):
            routes[i] = choose(routes[:i] + routes[i + 1:], with_tg)
        c = loads_costs(routes, with_tg)
        avg = sum(c[r] for r in routes) / N
        sysc = sum(c[r] for r in routes)
        n1, n2, n3 = routes.count("R1"), routes.count("R2"), routes.count("R3")
        print(f"{p + 1:>4} {n1:>3} {n2:>3} {n3:>3} {str(c):>30} {avg:>6.1f} {sysc:>7.0f}")
    return avg, sysc


if __name__ == "__main__":
    a_avg, a_sys = run(False)
    print("  теория (Нэш 3/3): время 83, систем. 498")
    b_avg, b_sys = run(True)
    print("  теория (Нэш 2/2/2): время 92, систем. 552")
    worse = b_avg > a_avg + 1
    print(f"\nПАРАДОКС: добавили дорогу -> среднее время {a_avg:.1f} -> {b_avg:.1f}, "
          f"систем. {a_sys:.0f} -> {b_sys:.0f}.  "
          f"{'ВСЕМ ХУЖЕ — парадокс Брайеса воспроизведён' if worse else 'не ухудшилось'}")
