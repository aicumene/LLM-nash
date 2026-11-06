"""
Парадокс Брайеса — продвинутые версии:
  (1) HARD: агентам дают ФОРМУЛЫ дорог + текущие загрузки, они САМИ считают своё время
      по каждому маршруту (тест congestion-reasoning, не argmin готовых чисел).
  (2) CROSS: агенты РАЗНЫХ семей на РАЗНЫХ машинах (Qwen tesla + Gemma & Nemotron Thor)
      в одной сети — маршрутизируют ли по-разному, кто получает меньшее время.

Сеть (A->V, 6 машин), время дороги = f(y): A->G: y+50, G->V: 10y, A->T: 10y, T->V: y+50,
T->G: y+10 (только with_TG). Маршруты R1 A-G-V, R2 A-T-V, R3 A-T-G-V.
Теория Нэша: без T->G 3/3 -> 83; с T->G 2/2/2 -> 92 (всем хуже). Sequential best-response.
Запуск: python braess2.py hard|cross   (оркестр с Mac, stdlib).
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request


class Agent:
    def __init__(self, family, endpoint, model, extra=None, max_tokens=220):
        self.family, self.endpoint, self.model = family, endpoint, model
        self.extra, self.max_tokens = extra or {}, max_tokens

    def chat(self, messages):
        body = {"model": self.model, "messages": messages, "max_tokens": self.max_tokens,
                "temperature": 0.4, **self.extra}
        req = urllib.request.Request(self.endpoint + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return (json.load(r)["choices"][0]["message"].get("content") or "").strip()


import os; QWEN_URL = os.environ.get("QWEN_URL", "http://localhost:8000/v1"); OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/v1")
QWEN = Agent("Qwen", QWEN_URL, "Qwen3.6-27B-AWQ",
             {"chat_template_kwargs": {"enable_thinking": False}}, max_tokens=220)
GEMMA = Agent("Gemma", OLLAMA_URL, "gemma4:31b", {}, max_tokens=700)
NEMO = Agent("Nemotron", OLLAMA_URL, "nemotron-3-nano:30b", {}, max_tokens=700)


def road_loads(routes):
    n1, n2, n3 = routes.count("R1"), routes.count("R2"), routes.count("R3")
    return {"A->G": n1, "G->V": n1 + n3, "A->T": n2 + n3, "T->V": n2, "T->G": n3}


def route_costs(routes, with_tg):
    L = road_loads(routes)
    c = {"R1": (L["A->G"] + 50) + 10 * L["G->V"], "R2": 10 * L["A->T"] + (L["T->V"] + 50)}
    if with_tg:
        c["R3"] = 10 * L["A->T"] + (L["T->G"] + 10) + 10 * L["G->V"]
    return c


def choose_hard(agent, others, with_tg):
    ol = road_loads(others)
    fm = ("A->Gameland = y+50, Gameland->Vladivostok = 10*y, Airport->Artem = 10*y, "
          "Artem->Vladivostok = y+50" + (", Artem->Gameland = y+10" if with_tg else ""))
    cur = (f"A->G:{ol['A->G']}, G->V:{ol['G->V']}, A->T:{ol['A->T']}, T->V:{ol['T->V']}"
           + (f", T->G:{ol['T->G']}" if with_tg else ""))
    routes = ("R1 = Airport->Gameland->Vladivostok; R2 = Airport->Artem->Vladivostok"
              + ("; R3 = Airport->Artem->Gameland->Vladivostok" if with_tg else ""))
    opts = "R1, R2 or R3" if with_tg else "R1 or R2"
    sysmsg = ("You drive Airport->Vladivostok and want the LOWEST travel time FOR YOU. "
              "Each road's time depends on the number of drivers y on it: " + fm + ". Routes: "
              + routes + ". When YOU pick a route you ADD yourself (+1) to each road on it. "
              "Compute your own total time for each route, then end with exactly 'ROUTE: R1' "
              f"({opts}).")
    user = (f"The OTHER {len(others)} drivers currently put this many drivers on each road: {cur}. "
            "Add yourself to the roads of your chosen route, compute your time per route, pick the fastest.")
    try:
        txt = agent.chat([{"role": "system", "content": sysmsg}, {"role": "user", "content": user}]).upper()
    except Exception as e:
        print(f"   [warn {agent.family}: {type(e).__name__} -> R2]")
        return "R2"
    m = re.findall(r"ROUTE:\s*(R[123])", txt) or re.findall(r"\b(R[123])\b", txt)
    valid = ["R1", "R2", "R3"] if with_tg else ["R1", "R2"]
    return next((x for x in reversed(m) if x in valid), "R2") if m else "R2"


def run(agents, with_tg, passes, title):
    routes = ["R2"] * len(agents)
    print(f"\n=== {title} ===")
    for p in range(passes):
        for i, ag in enumerate(agents):
            routes[i] = choose_hard(ag, routes[:i] + routes[i + 1:], with_tg)
        c = route_costs(routes, with_tg)
        n1, n2, n3 = routes.count("R1"), routes.count("R2"), routes.count("R3")
        avg = sum(c[r] for r in routes) / len(routes)
        print(f"  pass {p + 1}: R1={n1} R2={n2} R3={n3}  times={c}  avg={avg:.1f} system={sum(c[r] for r in routes):.0f}")
    c = route_costs(routes, with_tg)
    byfam = {}
    for ag, r in zip(agents, routes):
        byfam.setdefault(ag.family, []).append(c[r])
    if len(byfam) > 1:
        print("  по семьям (среднее время / маршруты):")
        for fam, times in byfam.items():
            rs = [r for ag, r in zip(agents, routes) if ag.family == fam]
            print(f"    {fam:>9}: avg {sum(times) / len(times):.1f}  маршруты {rs}")
    return routes, c


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "hard"
    if mode == "hard":
        run([QWEN] * 6, False, 3, "HARD, 6×Qwen, БЕЗ дороги (формулы, считают сами)")
        print("  теория: 3/3 -> 83")
        run([QWEN] * 6, True, 3, "HARD, 6×Qwen, С дорогой R3 (формулы, считают сами)")
        print("  теория: 2/2/2 -> 92 (всем хуже)")
    elif mode == "cross":
        agents = [QWEN, QWEN, GEMMA, GEMMA, NEMO, NEMO]
        run(agents, True, 2, "CROSS-семья, 2 Qwen + 2 Gemma + 2 Nemotron, С дорогой R3")
        print("  теория Нэша: 2/2/2 -> время 92 каждому; кто отклонится / получит меньше?")
