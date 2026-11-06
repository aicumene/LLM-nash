"""Symmetric equilibria of the reactive (memory-one) iterated PD, Menshikov et al. 2017, by direct best-response search.
Player 1's payoff depends on (a1, g1) only through its stationary p1; constant strategies reach every p1 in [0, 1], so
the best reply to (a, g) is max over q of u(q, a + (g - a) q). A sampled search over (a1, g1) checks that reduction."""
import numpy as np
GAMES = {'human 10,5,1,0': (10, 5, 1, 0), 'ours 5,3,1,0': (5, 3, 1, 0)}
Q = np.linspace(0, 1, 2001)


def u(p1, p2, T, R, P, S):
    return R*p1*p2 + S*p1*(1-p2) + T*(1-p1)*p2 + P*(1-p1)*(1-p2)


def regret(a, g, game, start=1.0):
    """Best-reply gain against (a, g) when both play it; the TFT pair uses the p fixed by its opening (default C)."""
    T, R, P, S = game
    b = g - a
    p = start if abs(1 - b) < 1e-12 else a / (1 - b)
    best = u(Q, a + b*Q, T, R, P, S).max()
    return best - u(p, p, T, R, P, S), p


def poly(a, g, game):
    """The FOC with the denominator (1 - b) cleared."""
    T, R, P, S = game
    b = g - a
    return -(T-R)*a - (P-S)*(1-b-a) + b*((R-S)*a + (T-P)*(1-b-a))


rng = np.random.default_rng(0)
for name, game in GAMES.items():
    T, R, P, S = game
    print(f'== {name}: 2R={2*R}, T+S={T+S}')
    print('  poly at (0,1):', poly(0, 1, game), ' regret at TFT opening C:', '%.6f' % regret(0, 1, game)[0],
          ' opening D:', '%.6f' % regret(0, 1, game, start=0.0)[0])
    # sampled check of the reduction, opponent TFT and a few others
    for a2, g2 in [(0, 1), (0.1, 0.9), (0.2, 0.7), (1/3, 1), (1, 1)]:
        a1 = rng.random(200000); g1 = rng.random(200000)
        r1 = g1 - a1; r2 = g2 - a2
        p1 = (a1 + r1*a2) / (1 - r1*r2); p2 = a2 + r2*p1
        sampled = u(p1, p2, T, R, P, S).max(); reduced = u(Q, a2 + r2*Q, T, R, P, S).max()
        print(f'  reply to ({a2:.2f},{g2:.2f}): sampled {sampled:.4f}  via q {reduced:.4f}')
    # equilibrium set on a grid
    G = np.linspace(0, 1, 401)
    eq = []
    for a in G:
        for g in G:
            r, p = regret(a, g, game)
            if r < 2e-4:
                eq.append((a, g, p))
    eq = np.array(eq)
    print(f'  grid equilibria: {len(eq)}')
    # the part of the equilibrium set with g = 1, and how close equilibria with g < 1 come to (0, 1)
    top = eq[np.isclose(eq[:, 1], 1)]
    print('  on g=1: alpha from %.3f to %.3f' % (top[:, 0].min(), top[:, 0].max()))
    inner = eq[(eq[:, 1] < 0.999) & (eq[:, 1] > eq[:, 0])]
    d = np.hypot(inner[:, 0], inner[:, 1] - 1)
    print('  nearest equilibrium with g<1 to (0,1): (%.3f, %.3f), distance %.3f' % (*inner[d.argmin(), :2], d.min()))
    print('  max alpha in the inner set: %.3f at g=%.3f' % (inner[:, 0].max(), inner[inner[:, 0].argmax(), 1]))
    print('  inner set at alpha=0:', [round(float(x), 3) for x in inner[np.isclose(inner[:, 0], 0), 1][:6]])
    # always-cooperate
    print('  ALLC (1,1) regret %.3f' % regret(1, 1, game)[0])
    # cross-check by simulation against the analytic gain for two opponents
    sim = np.random.default_rng(1)
    for (a2, g2), q in [((0.0, 1.0), 0.8), ((1/3, 1.0), 0.0), ((0.5, 1.0), 0.0)]:
        n = 400000
        m1 = sim.random(n) < q
        m2 = np.zeros(n, bool); m2[0] = True
        draws = sim.random(n)
        for t in range(1, n):
            m2[t] = draws[t] < (g2 if m1[t-1] else a2)
        pay = np.where(m1 & m2, R, np.where(m1 & ~m2, S, np.where(~m1 & m2, T, P)))
        print(f'  sim constant q={q} vs ({a2:.2f},{g2:.2f}): {pay.mean():.3f}  analytic {u(q, a2+(g2-a2)*q, T,R,P,S):.3f}')
