"""
Braess cross — развести REASONING от STICKINESS: случайный СТАРТ (не весь R2) +
счётчик СМЕН маршрута по семьям. Если Gemma/Nemotron застревают на медленных маршрутах
при ЛЮБОМ старте и мало двигаются → слабость congestion-reasoning; если двигаются, но
всё равно медленно → тоже reasoning; если не двигаются только из дефолта → липкость.

2 Qwen + 2 Gemma + 2 Nemotron, R3 есть, hard-промпт, рандом-порядок, случайный init.
Запуск: python braess4.py [seed ...]
"""
from __future__ import annotations

import random
import sys

from braess2 import GEMMA, NEMO, QWEN, choose_hard, route_costs
from braess3 import exploitability


def run(agents, passes, seed):
    rng = random.Random(seed)
    valid = ["R1", "R2", "R3"]
    routes = [rng.choice(valid) for _ in agents]          # СЛУЧАЙНЫЙ старт
    init = routes[:]
    moves = [0] * len(agents)
    idx = list(range(len(agents)))
    for _ in range(passes):
        rng.shuffle(idx)
        for i in idx:
            new = choose_hard(agents[i], routes[:i] + routes[i + 1:], True)
            if new != routes[i]:
                moves[i] += 1
            routes[i] = new
    c = route_costs(routes, True)
    print(f"\n--- seed {seed}: init={init} -> final={routes}  exploit(max regret)={exploitability(routes, True)[0]} ---")
    byfam = {}
    for ag, r, m, ir in zip(agents, routes, moves, init):
        byfam.setdefault(ag.family, []).append((r, c[r], m, ir))
    for fam, lst in byfam.items():
        print(f"  {fam:>9}: avg time {sum(t for _, t, _, _ in lst) / len(lst):5.1f}  "
              f"смен маршрута {sum(m for _, _, m, _ in lst)}  "
              f"init {[ir for _, _, _, ir in lst]} -> final {[r for r, _, _, _ in lst]}")
    return [c[r] for r in routes], byfam


if __name__ == "__main__":
    agents = [QWEN, QWEN, GEMMA, GEMMA, NEMO, NEMO]
    seeds = [int(x) for x in sys.argv[1:]] or [7, 8]
    print("Braess cross со СЛУЧАЙНЫМ стартом (reasoning vs stickiness). Нэш=2/2/2->92.")
    agg = {}
    for s in seeds:
        _, byfam = run(agents, 4, s)
        for fam, lst in byfam.items():
            agg.setdefault(fam, []).extend(t for _, t, _, _ in lst)
    print("\n=== СВОДКА по семьям (среднее время по всем seed) ===")
    for fam, times in agg.items():
        print(f"  {fam:>9}: {sum(times) / len(times):.1f}")
