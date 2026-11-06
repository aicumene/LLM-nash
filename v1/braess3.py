"""
Строгий кросс-семья Braess: РАНДОМИЗИРОВАННЫЙ порядок хода (снимаем конфаунд «кто
первый хватает маршрут») + EXPLOITABILITY (max regret = насколько далеко от равновесия;
0 = Нэш). 2 Qwen (tesla) + 2 Gemma + 2 Nemotron (Thor), С дорогой R3, hard-промпт (считают сами).

Запуск: python braess3.py
"""
from __future__ import annotations

import random
import sys

from braess2 import GEMMA, NEMO, QWEN, choose_hard, route_costs


def exploitability(routes, with_tg):
    """Для каждого агента: на сколько он мог бы УЛУЧШИТЬ своё время, сменив маршрут
    (остальные фиксированы). max regret = расстояние до равновесия (0 = Нэш)."""
    cur = route_costs(routes, with_tg)
    regrets = []
    valid = ["R1", "R2", "R3"] if with_tg else ["R1", "R2"]
    for i, r in enumerate(routes):
        others = routes[:i] + routes[i + 1:]
        opt = {rr: route_costs(others + [rr], with_tg)[rr] for rr in valid}
        regrets.append(cur[r] - min(opt.values()))
    return max(regrets), sum(regrets) / len(regrets)


def run(agents, with_tg, passes, seed):
    rng = random.Random(seed)
    routes = ["R2"] * len(agents)
    idx = list(range(len(agents)))
    print(f"\n--- seed {seed}: рандомизированный порядок, {passes} проходов ---")
    for p in range(passes):
        rng.shuffle(idx)
        for i in idx:
            routes[i] = choose_hard(agents[i], routes[:i] + routes[i + 1:], with_tg)
        c = route_costs(routes, with_tg)
        mx, mn = exploitability(routes, with_tg)
        n1, n2, n3 = routes.count("R1"), routes.count("R2"), routes.count("R3")
        print(f"  pass {p + 1}: R1={n1} R2={n2} R3={n3}  times={c}  "
              f"exploitability(max regret)={mx}  avg-regret={mn:.1f}")
    c = route_costs(routes, with_tg)
    byfam = {}
    for ag, r in zip(agents, routes):
        byfam.setdefault(ag.family, []).append((r, c[r]))
    print("  итог по семьям:")
    for fam, lst in byfam.items():
        print(f"    {fam:>9}: avg time {sum(t for _, t in lst) / len(lst):.1f}  маршруты {[r for r, _ in lst]}")
    mx, _ = exploitability(routes, with_tg)
    print(f"  -> {'РАВНОВЕСИЕ (exploitability=0)' if mx == 0 else f'НЕ равновесие (кто-то выиграл бы {mx} сменой маршрута)'}")
    return routes


if __name__ == "__main__":
    agents = [QWEN, QWEN, GEMMA, GEMMA, NEMO, NEMO]
    seeds = [int(x) for x in sys.argv[1:]] or [1, 2]
    print("Строгий кросс-Braess (2 Qwen + 2 Gemma + 2 Nemotron, R3 есть, hard):")
    for s in seeds:
        run(agents, True, 5, s)
    print("\nНапоминание: Нэш = 2/2/2 -> всем по 92; exploitability там = 0.")
