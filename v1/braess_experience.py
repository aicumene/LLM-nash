"""
Брайес с обучением на СОБСТВЕННОМ ОПЫТЕ (day-to-day route choice, как у Selten и др.
2007 «Commuters route choice behaviour»): агент НЕ видит формулы издержек и НЕ видит
текущую загрузку сети — только СВОЮ историю (какой маршрут выбирал, сколько реально
ехал). Follow-up к braess_qre.py: там Gemma/Nemotron ИГНОРИРОВАЛИ формулы (100% R2
независимо от чисел). Вопрос: смогут ли они хотя бы НАУЧИТЬСЯ на повторяющемся
собственном опыте, если формульная арифметика для них недоступна?

6 агентов (2 Qwen + 2 Gemma + 2 Nemotron), сеть С короткой дорогой, R раундов.
Каждый раунд: все агенты одновременно выбирают маршрут по своей истории; считаем
РЕАЛЬНЫЕ издержки (route_costs от совместного выбора); дописываем каждому его строку
истории. Смотрим: убывает ли среднее время; смещается ли распределение к 2/2/2 (Нэш);
уходят ли Gemma/Nemotron с R2, если он у них плохой.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

from braess2 import GEMMA, NEMO, QWEN, route_costs

ROUTES = ["R1", "R2", "R3"]
ROUTE_DESC = ("R1 = via Gameland (north bridge); R2 = via Artem (south road); "
              "R3 = via Artem then Gameland (the shortcut connector)")


def parse_move(txt: str) -> str | None:
    if txt.startswith("__ERROR__") or not txt:
        return None
    t = txt.upper()
    tail = t.split("ROUTE:")[-1]
    for r in ROUTES:
        if r in tail:
            return r
    for r in reversed(ROUTES):
        if r in t:
            return r
    return None


def choose_from_experience(agent, history: list[tuple[str, float]]) -> str | None:
    """history: список (маршрут, реальное время) СОБСТВЕННЫХ прошлых поездок этого
    агента. Формулы и состояние сети НЕ раскрываются — только опыт."""
    sysmsg = (f"You are a daily commuter Airport->Vladivostok. Three routes exist: "
              f"{ROUTE_DESC}. You do NOT know the traffic formulas or how many other "
              f"drivers are on each road today — you only know YOUR OWN past experience. "
              f"Pick the route you expect to be FASTEST for you today, based on your "
              f"history. End with exactly 'ROUTE: R1', 'ROUTE: R2', or 'ROUTE: R3'.")
    if history:
        lines = [f"day {i + 1}: you took {r}, your travel time was {t:.0f}"
                 for i, (r, t) in enumerate(history)]
        user = "Your commute history so far:\n" + "\n".join(lines) + "\n\nToday's choice?"
    else:
        user = "This is your first day commuting this route. No history yet. Pick a route."
    return parse_move(agent.chat([{"role": "system", "content": sysmsg},
                                  {"role": "user", "content": user}]))


def run(agents: list, rounds: int) -> list[dict]:
    histories: list[list[tuple[str, float]]] = [[] for _ in agents]
    log = []
    for rd in range(rounds):
        with ThreadPoolExecutor(max_workers=14) as ex:
            choices = list(ex.map(lambda i: choose_from_experience(agents[i], histories[i]),
                                  range(len(agents))))
        choices = [c if c in ROUTES else "R2" for c in choices]  # дефолт при парсе-фейле
        costs = route_costs(choices, True)
        for i, ag in enumerate(agents):
            t = costs[choices[i]]
            histories[i].append((choices[i], t))
        avg = sum(costs[c] for c in choices) / len(choices)
        dist = {r: choices.count(r) for r in ROUTES}
        fam_line = {ag.family + str(i): choices[i] for i, ag in enumerate(agents)}
        print(f"  round {rd + 1:>2}: {dist}  avg={avg:.1f}  choices={fam_line}")
        log.append({"round": rd + 1, "choices": choices, "costs": costs, "avg": avg, "dist": dist})
    return log


if __name__ == "__main__":
    agents = [QWEN] * 10 + [GEMMA, GEMMA, NEMO, NEMO]
    R = 20
    print(f"Брайес на опыте: 10 Qwen + 2 Gemma + 2 Nemotron ({len(agents)} агентов), {R} раундов, "
          f"БЕЗ формул/состояния сети — только своя история.")
    print("Равновесие Нэша = 2/2/2, время 92 каждому.\n")

    log = run(agents, R)

    by_family = {}
    for i, ag in enumerate(agents):
        key = ag.family
        by_family.setdefault(key, {"routes": [], "times": []})
        by_family[key]["routes"].append([entry["choices"][i] for entry in log])
        by_family[key]["times"].append([entry["costs"][entry["choices"][i]] for entry in log])

    print("\n=== по семьям: маршруты первых 5 / последних 5 раундов ===")
    for fam, d in by_family.items():
        for j, routes in enumerate(d["routes"]):
            print(f"  {fam}[{j}]: first5={routes[:5]}  last5={routes[-5:]}")

    early_avg = sum(e["avg"] for e in log[:5]) / 5
    late_avg = sum(e["avg"] for e in log[-5:]) / 5
    print(f"\nсреднее время: первые 5 раундов={early_avg:.1f}  последние 5={late_avg:.1f}  "
          f"(Нэш=92)")
    print(f"{'СХОДИТСЯ к Нэшу' if abs(late_avg - 92) < abs(early_avg - 92) else 'НЕ сходится / расходится'}")

    with open("braess_experience_out.json", "w") as f:
        json.dump({"log": log, "by_family": by_family}, f, indent=2, default=str)
    print("\nwrote braess_experience_out.json")
