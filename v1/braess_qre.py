"""
QRE/λ для Брайеса: логит-модель выбора маршрута (stochastic user equilibrium, та же
идея, что λ в qre_pd_sweep.py, только объект — маршрут, а не C/D).

Сеть С короткой дорогой (R1,R2,R3), hard-режим (агент сам считает время по формулам
из choose_hard в braess2.py — там же раньше нашли провал congestion-reasoning).
Для каждого независимого решения: рандомизируем маршруты 5 «остальных» водителей
(равномерно по {R1,R2,R3}) → это даёт РАЗБРОС истинных издержек по маршрутам, нужный
для устойчивого фита λ. Агент выбирает; параллельно вычисляем ИСТИННЫЕ издержки
route_costs(others+[r], True)[r] для каждого r — это и есть регрессор.

Логит-модель: P(choice=r) = exp(-λ·cost_r) / Σ_r' exp(-λ·cost_r').
λ→∞ = детерминированный argmin (истинная рациональность); λ→0 = выбор не зависит
от издержек (случайный/по умолчанию).
"""
from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from scipy.optimize import minimize_scalar

from braess2 import GEMMA, NEMO, QWEN, choose_hard, route_costs

FAMILIES = {"Qwen": QWEN, "Gemma": GEMMA, "Nemotron": NEMO}
N_DRAWS = 30
ROUTES = ["R1", "R2", "R3"]


def sample_decision(agent, seed: int):
    rng = random.Random(seed)
    others = [rng.choice(ROUTES) for _ in range(5)]
    costs = {r: route_costs(others + [r], True)[r] for r in ROUTES}
    choice = choose_hard(agent, others, True)
    return {"others": others, "costs": costs, "choice": choice}


def collect(family: str, agent) -> list[dict]:
    out = [None] * N_DRAWS
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(sample_decision, agent, 2000 + i): i for i in range(N_DRAWS)}
        for f in as_completed(futs):
            i = futs[f]
            out[i] = f.result()
    n_valid = sum(1 for d in out if d["choice"] in ROUTES)
    print(f"  [{family}] {n_valid}/{N_DRAWS} valid decisions")
    return out


def fit_lambda(decisions: list[dict]) -> tuple[float, float]:
    valid = [d for d in decisions if d["choice"] in ROUTES]

    def neg_log_lik(lam):
        ll = 0.0
        for d in valid:
            c = np.array([d["costs"][r] for r in ROUTES])
            # numerically stable softmax log-prob for the chosen route:
            shifted = -lam * (c - c.min())
            logZ = np.log(np.sum(np.exp(shifted)))
            idx = ROUTES.index(d["choice"])
            ll += shifted[idx] - logZ
        return -ll

    res = minimize_scalar(neg_log_lik, bounds=(0.0, 5.0), method="bounded")
    return float(res.x), float(res.fun)


def choice_vs_optimal(decisions: list[dict]) -> dict:
    """Доля решений, где агент выбрал ИСТИННО дешевейший маршрут (argmin), и средний
    региональный regret (издержка выбора минус издержка оптимума)."""
    valid = [d for d in decisions if d["choice"] in ROUTES]
    if not valid:
        return {"optimal_rate": None, "mean_regret": None}
    opt_hits, regrets = 0, []
    for d in valid:
        best_r = min(ROUTES, key=lambda r: d["costs"][r])
        opt_hits += d["choice"] == best_r
        regrets.append(d["costs"][d["choice"]] - d["costs"][best_r])
    return {"optimal_rate": opt_hits / len(valid), "mean_regret": float(np.mean(regrets))}


if __name__ == "__main__":
    results = {}
    for family, agent in FAMILIES.items():
        print(f"\n=== {family} ({agent.endpoint}) ===")
        decisions = collect(family, agent)
        lam, nll = fit_lambda(decisions)
        stats = choice_vs_optimal(decisions)
        results[family] = {"decisions": decisions, "lambda": lam, "neg_log_lik": nll, **stats}
        print(f"  λ = {lam:.3f}  (NLL={nll:.2f})  optimal-choice rate = "
              f"{stats['optimal_rate']:.1%}  mean regret = {stats['mean_regret']:.2f}")

    with open("braess_qre_out.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nwrote braess_qre_out.json")

    print("\n=== SUMMARY (route-choice λ, higher = more rational/argmin-consistent) ===")
    for family, r in results.items():
        print(f"  {family:>9}: λ={r['lambda']:.3f}  optimal-rate={r['optimal_rate']:.1%}  "
              f"mean-regret={r['mean_regret']:.2f}")
