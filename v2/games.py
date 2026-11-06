"""The three games of "When Do Language Models Reach Nash?" with the prompts of the original scripts
(pd.py, pbeauty.py, braess.py, braess2.py, braess_experience.py, nash_recheck2.py), plus
  - per-round exploitability (max over agents of the gain from a unilateral switch; 0 exactly at a Nash equilibrium),
  - surrogate (non-interacting) variants: the agent's prompt and history are identical in form, but the other players are
    fabricated and never react to it, so any structure in the agent's play is its own disposition.
Pure game logic; no model calls (play.py does those)."""
from __future__ import annotations
import re, random, statistics

# ======================================================================= Prisoner's Dilemma
PAY = {("C", "C"): (3, 3), ("C", "D"): (0, 5), ("D", "C"): (5, 0), ("D", "D"): (1, 1)}
PD_LABELS = {"loaded": ("Cooperate", "Defect"), "neutral": ("Option A", "Option B")}


def pd_messages(labels, history, me, rounds, history_labels=True):
    """pd.py's move(): the same system prompt; `history` holds ("C"/"D", "C"/"D") per round (player 1, player 2).
    history_labels=True writes the chosen labels into the history lines (pd.py wrote the internal letters C/D even with
    neutral labels; for the contamination probe the labels must stay neutral everywhere)."""
    cl, dl = labels
    payline = (f"Both choose {cl}: you get 3. You {cl} & they {dl}: you 0, they 5. "
               f"You {dl} & they {cl}: you 5, they 0. Both {dl}: you 1.")
    sysmsg = (f"You are Player {me}, playing {rounds} rounds against the SAME opponent. "
              f"Each round you BOTH simultaneously choose {cl} or {dl}. Per-round payoffs: {payline} "
              f"Goal: maximize YOUR OWN total points across all {rounds} rounds. "
              f"Think one short sentence, then end with exactly 'MOVE: {cl}' or 'MOVE: {dl}'.")
    msgs = [{"role": "system", "content": sysmsg}]
    if history:
        name = (lambda m: {"C": cl, "D": dl}[m]) if history_labels else (lambda m: m)
        lines = []
        for i, (a, b) in enumerate(history):
            mine, opp = (a, b) if me == "1" else (b, a)
            lines.append(f"round {i + 1}: you={name(mine)}, opponent={name(opp)}")
        msgs.append({"role": "user", "content": "History:\n" + "\n".join(lines) + "\n\nYour move."})
    else:
        msgs.append({"role": "user", "content": "Round 1. Your move."})
    return msgs


def pd_parse(txt, labels):
    cl, dl = labels; t = txt.upper(); tail = t.split("MOVE:")[-1]
    if dl.upper() in tail and cl.upper() not in tail: return "D"
    if cl.upper() in tail and dl.upper() not in tail: return "C"
    return "D" if t.rfind(dl.upper()) > t.rfind(cl.upper()) else "C"


def pd_exploitability(a, b):
    """Stage-game regret of each player given the other's actual move; the maximum over the two players, and normalised
    by the largest payoff (5)."""
    ra = max(PAY[(x, b)][0] for x in "CD") - PAY[(a, b)][0]
    rb = max(PAY[(a, y)][1] for y in "CD") - PAY[(a, b)][1]
    return dict(regret=[ra, rb], exploitability=max(ra, rb), normalised=max(ra, rb) / 5.0)


def pd_script(kind, rounds, seed=0):
    """Scripted (surrogate) opponent for player 2: allC; single_early/mid/late (one defection at round 3/7/10 of 14, as in
    nash_recheck2.py); random35 (iid defection with p = 0.35)."""
    if kind == "allC": return ["C"] * rounds
    if kind.startswith("single_"):
        at = {"early": 3, "mid": 7, "late": 10}[kind.split("_")[1]]
        return ["D" if i == at else "C" for i in range(rounds)]
    if kind == "random35":
        rng = random.Random(seed); return ["D" if rng.random() < 0.35 else "C" for _ in range(rounds)]
    raise ValueError(kind)


# ======================================================================= p-beauty contest
def pb_messages(n_players, p, history):
    """pbeauty.py's guess(): `history` = list of rounds, each a list of all picks."""
    sysm = (f"You are a contestant in a guessing game with {n_players} players. "
            f"Each player picks a number in [0,100]. The winner is the player whose pick "
            f"is closest to {p:.3f} times the AVERAGE of all picks. "
            f"Reason briefly (one or two sentences), then put your pick on the LAST line "
            f"exactly as 'GUESS: <number>'.")
    msgs = [{"role": "system", "content": sysm}]
    if history:
        h = "\n".join(f"round {i + 1}: picks = {g}, avg = {statistics.mean(g):.1f}, "
                      f"target = {p * statistics.mean(g):.1f}" for i, g in enumerate(history))
        msgs.append({"role": "user", "content": "History so far:\n" + h + "\n\nGive your next pick."})
    else:
        msgs.append({"role": "user", "content": "Round 1. Give your pick."})
    return msgs


def pb_parse(txt):
    m = re.findall(r"GUESS:\s*(-?[0-9]+(?:\.[0-9]+)?)", txt) or re.findall(r"(-?[0-9]+(?:\.[0-9]+)?)", txt)
    val = float(m[-1]) if m else 50.0
    return max(0.0, min(100.0, val))


def pb_exploitability(picks, p):
    """Best response of player i to the others' picks: x* = p·S_others / (N − p); the regret is |pick − x*|
    (distance from the winning pick); exploitability = max over players, normalised by 100."""
    N = len(picks); S = sum(picks); reg = []
    for x in picks:
        xs = p * (S - x) / (N - p); reg.append(abs(x - xs))
    ne = 0.0 if p < 1 else 100.0
    return dict(regret=reg, exploitability=max(reg), normalised=max(reg) / 100.0, mean=statistics.mean(picks),
                target=p * statistics.mean(picks), dist_to_ne=abs(statistics.mean(picks) - ne))


def pb_surrogate_others(n_others, rounds, seed):
    """Fabricated picks of the other players: uniform in [0, 100], fixed in advance (they never react)."""
    rng = random.Random(seed); return [[round(rng.uniform(0, 100), 1) for _ in range(n_others)] for _ in range(rounds)]


# ======================================================================= Braess routing
ROUTES3 = ["R1", "R2", "R3"]


def road_loads(routes):
    n1, n2, n3 = routes.count("R1"), routes.count("R2"), routes.count("R3")
    return {"A->G": n1, "G->V": n1 + n3, "A->T": n2 + n3, "T->V": n2, "T->G": n3}


def route_costs(routes, with_tg):
    L = road_loads(routes)
    c = {"R1": (L["A->G"] + 50) + 10 * L["G->V"], "R2": 10 * L["A->T"] + (L["T->V"] + 50)}
    if with_tg: c["R3"] = 10 * L["A->T"] + (L["T->G"] + 10) + 10 * L["G->V"]
    return c


def option_times(others, with_tg):
    routes = ROUTES3 if with_tg else ROUTES3[:2]
    return {r: route_costs(others + [r], with_tg)[r] for r in routes}


def braess_exploitability(routes, with_tg):
    """For each driver: its cost minus the cheapest alternative given the others; the maximum over drivers (0 at Nash)."""
    gains = []
    for i, r in enumerate(routes):
        others = routes[:i] + routes[i + 1:]; opt = option_times(others, with_tg)
        gains.append(opt[r] - min(opt.values()))
    costs = route_costs(routes, with_tg)
    return dict(gain=gains, exploitability=max(gains), normalised=max(gains) / 100.0,
                avg=sum(costs[r] for r in routes) / len(routes), system=sum(costs[r] for r in routes),
                counts={r: routes.count(r) for r in ROUTES3})


def braess_messages_easy(others, with_tg, n_total=6):
    """braess.py's choose(): the agent is shown its own travel time on every route."""
    opt = option_times(others, with_tg); desc = "; ".join(f"{r}={t} min" for r, t in opt.items())
    sysmsg = ("You are a driver from Airport to Vladivostok choosing among routes. "
              "Pick the route with the LOWEST travel time FOR YOU. "
              "End with exactly 'ROUTE: R1' (or R2" + (" or R3)." if with_tg else ")."))
    user = (f"Given where the other {n_total - 1} drivers currently are, YOUR travel time on each route "
            f"would be: {desc}. Choose the fastest route for you.")
    return [{"role": "system", "content": sysmsg}, {"role": "user", "content": user}], opt


def braess_messages_hard(others, with_tg):
    """braess2.py's choose_hard(): formulas and current loads; the agent computes its own times."""
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
    return [{"role": "system", "content": sysmsg}, {"role": "user", "content": user}], option_times(others, with_tg)


ROUTE_DESC = ("R1 = via Gameland (north bridge); R2 = via Artem (south road); "
              "R3 = via Artem then Gameland (the shortcut connector)")


def braess_messages_experience(history):
    """braess_experience.py's choose_from_experience(): only the agent's own (route, time) history."""
    sysmsg = (f"You are a daily commuter Airport->Vladivostok. Three routes exist: "
              f"{ROUTE_DESC}. You do NOT know the traffic formulas or how many other "
              f"drivers are on each road today — you only know YOUR OWN past experience. "
              f"Pick the route you expect to be FASTEST for you today, based on your "
              f"history. End with exactly 'ROUTE: R1', 'ROUTE: R2', or 'ROUTE: R3'.")
    if history:
        lines = [f"day {i + 1}: you took {r}, your travel time was {t:.0f}" for i, (r, t) in enumerate(history)]
        user = "Your commute history so far:\n" + "\n".join(lines) + "\n\nToday's choice?"
    else:
        user = "This is your first day commuting this route. No history yet. Pick a route."
    return [{"role": "system", "content": sysmsg}, {"role": "user", "content": user}]


def braess_parse(txt, with_tg, default="R2"):
    t = txt.upper(); m = re.findall(r"ROUTE:\s*(R[123])", t) or re.findall(r"\b(R[123])\b", t)
    valid = ROUTES3 if with_tg else ROUTES3[:2]
    return next((x for x in reversed(m) if x in valid), default) if m else default


def braess_surrogate_others(n_others, with_tg, seed):
    """Fabricated positions of the other drivers for one decision: uniform over the routes (the design of braess_qre.py)."""
    rng = random.Random(seed); valid = ROUTES3 if with_tg else ROUTES3[:2]
    return [rng.choice(valid) for _ in range(n_others)]
