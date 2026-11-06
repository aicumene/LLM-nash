"""signature_lm.py: the signature code of the ACT-BOID paper, with the output directory from SIG_OUT and MC-SSA optional.
The oscillatory signature of a network's trajectories, measured the same way for every architecture (Section 11.7):
per trajectory, on a montage of 64 pooled channels (random tiles) with the estimators of Section 9.2 —
  spectrum      participation ratio, 95 % energy rank, surrogate-thresholded d_osc (per-channel shuffles)
  modes         TLS-ESPRIT poles in the band of periods <= T/2: count, sustained (|z| >= 0.95), periods
  aperiodic     Monte-Carlo SSA index with the 1/f^chi null (oscillators above the background, top excess)
  synchrony     mean |correlation| of channel pairs
  thirds        PR and synchrony in the first, middle and last third of the axis (20-step windows)
  speed         mean squared change of the standardised state per step; contrast between its 10th and 90th
                percentiles; number of fast phases; step at which the speed falls below the threshold for good
  operator      one linear map for all trajectories in the 64 leading principal axes, predicting the change from the
                current state (q = 1) and from 8 steps of history (q = 8); fit on half of the trajectories, scored on
                the other half; the map's eigenvalues: sustained rotating pairs and their periods
Usage: signature.py <tag> <npz> <key> [--n 64] [--T 0] [--procs 12]   -> ~/ctm/boid/signature_<tag>.json
   <key> is the array in the npz ([N, T, d]); --T truncates the axis (0 = whole).
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
import sys, json, time, argparse, numpy as np, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import mcssa as eeg_jpca                                           # the MC-SSA index with the 1/f null: the standalone CPU port
except Exception:                                                      # noqa: BLE001
    try:
        import eeg_jpca
    except Exception:                                                  # noqa: BLE001
        eeg_jpca = None
B = os.environ.get('SIG_OUT', os.path.expanduser('~/ctm/boid')); FS = 100.0; WIN, L_WIN = 20, 7


def zs(X):
    X = X - X.mean(0); sd = X.std(0); keep = sd > 1e-8
    return X[:, keep] / sd[keep]


def spectrum(X, L):
    T = X.shape[0]; K = T - L + 1; G = X @ X.T; Hg = np.zeros((K, K))
    for l in range(L): Hg += G[l:l + K, l:l + K]
    w, V = np.linalg.eigh(Hg); w = np.clip(w[::-1], 0, None); V = V[:, ::-1]
    return np.sqrt(w), V


def descriptors(s):
    lam = s ** 2; return dict(pr=float(lam.sum() ** 2 / (lam ** 2).sum()), r95=int(np.searchsorted(np.cumsum(lam) / lam.sum(), 0.95) + 1))


def d_osc_shuffle(X, L, s, rng, n_surr=30):
    T, C = X.shape; null = np.empty((n_surr, len(s)))
    for i in range(n_surr):
        idx = np.argsort(rng.random((T, C)), axis=0); null[i] = spectrum(np.take_along_axis(X, idx, axis=0), L)[0]
    thr = np.quantile(null, 0.95, axis=0); ex = s > thr
    return int(len(s)) if ex.all() else int(np.argmin(ex))


def esprit(X, L, s, V, fmin):
    T, C = X.shape; K = T - L + 1; lam = s ** 2
    r = int(np.searchsorted(np.cumsum(lam) / lam.sum(), 0.95) + 1); r = max(2, min(r, K - 2, ((L - 1) * C) // 2))
    H = np.empty((L * C, K))
    for l in range(L): H[l * C:(l + 1) * C] = X[l:l + K].T
    U = (H @ V[:, :r]) / s[:r]; U1, U2 = U[:(L - 1) * C], U[C:]
    M = np.block([[U1.T @ U1, U1.T @ U2], [U2.T @ U1, U2.T @ U2]]); w, Q = np.linalg.eigh(M); Q = Q[:, ::-1]
    z = np.linalg.eigvals(-Q[:r, r:] @ np.linalg.pinv(Q[r:, r:]))
    f = np.angle(z) / (2 * np.pi); a = np.abs(z); band = (f >= fmin) & (f < 0.5)
    return dict(r=r, n_modes=int(band.sum()), n_sust=int((a[band] >= 0.95).sum()), med_abs=float(np.median(a[band])) if band.any() else None,
                periods=sorted((1 / f[band]).round(1).tolist())[:12], moduli=a[band].round(3).tolist()[:12])


def sync(X):
    c = np.corrcoef(X.T); iu = np.triu_indices_from(c, 1); return float(np.nanmean(np.abs(c[iu])))


def runs(mask):
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]])); return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


Z = None; TILES = None; L_FULL = None; FMIN = None
def analyze(i):
    rng = np.random.default_rng(1000 + i); Zi = Z[i].astype(np.float64); T = Zi.shape[0]
    X = zs(Zi[:, TILES].mean(2)); s, V = spectrum(X, L_FULL); out = dict(i=int(i), **descriptors(s), C=int(X.shape[1]))
    out['d_osc'] = d_osc_shuffle(X, L_FULL, s, rng); out.update(esprit(X, L_FULL, s, V, FMIN)); out['sync'] = sync(X)
    try:
        if eeg_jpca is None: raise RuntimeError('eeg_jpca not available')
        j = eeg_jpca.jpca_index(Zi[:, TILES].mean(2), FS, L_FULL, k=16, n_surr=20, rng=rng, null='aperiodic')
        out['mc'] = {k: (float(v) if np.isscalar(v) else [float(x) for x in np.atleast_1d(v)][:8]) for k, v in j.items() if k in ('n_osc', 'excess_top', 'freq_top', 'n_sig_components', 'd_osc_mc', 'freqs_sig')}
    except Exception as ex:
        out['mc'] = {'error': f'{type(ex).__name__}: {ex}'[:100]}
    thirds = {}
    for name, (a, b) in (('first', (0, T // 3)), ('middle', (T // 3, 2 * T // 3)), ('last', (2 * T // 3, T))):
        prs, sys_ = [], []
        for t0 in range(a, b - WIN + 1, 5):
            Xw = zs(Zi[t0:t0 + WIN][:, TILES].mean(2)); sw, _ = spectrum(Xw, L_WIN); prs.append(descriptors(sw)['pr']); sys_.append(sync(Xw))
        thirds[name] = dict(pr=float(np.median(prs)) if prs else None, sync=float(np.median(sys_)) if sys_ else None)
    out['thirds'] = thirds
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('tag'); ap.add_argument('npz'); ap.add_argument('key'); ap.add_argument('--n', type=int, default=64); ap.add_argument('--T', type=int, default=0)
    ap.add_argument('--procs', type=int, default=12); ap.add_argument('--tiles', type=int, default=64); ap.add_argument('--L', type=int, default=0)
    A = ap.parse_args(); t0 = time.time()
    d = np.load(A.npz); Z = d[A.key]
    if Z.ndim == 4: Z = Z[:, :, 0]                                               # several layers stored: take the first
    Z = Z[:A.n].astype(np.float32)
    if A.T: Z = Z[:, :A.T]
    N, T, D = Z.shape; rng = np.random.default_rng(12345); perm = rng.permutation(D)
    TILES = perm[:(D // A.tiles) * A.tiles].reshape(A.tiles, D // A.tiles) if D >= 2 * A.tiles else perm[:, None]
    L_FULL = A.L or max(6, min(25, T // 3)); FMIN = 2.0 / T                      # periods up to T/2 count as oscillatory
    print(f'[{A.tag}] N {N}, T {T}, d {D}; tiles {TILES.shape}, L {L_FULL}, periods counted <= {T / 2:.0f} steps', flush=True)
    with mp.get_context('fork').Pool(A.procs) as pool: res = pool.map(analyze, range(N), chunksize=2)
    # ---- speed of the state, phases
    flat = Z.reshape(-1, D); mu = flat.mean(0); sd = flat.std(0); sd[sd < 1e-6] = 1; Zs = (Z - mu) / sd
    speed = ((Zs[:, 1:] - Zs[:, :-1]) ** 2).mean(2); q10, q90 = np.quantile(speed, 0.1), np.quantile(speed, 0.9); thr = float(np.sqrt(q10 * q90))
    n_fast, settle = [], []
    for i in range(N):
        e = [(a, b) for a, b in runs(speed[i] > thr) if b - a >= 3]; n_fast.append(len(e))
        below = speed[i] < thr; settle.append(int(np.argmax(np.cumsum(below[::-1]) == np.arange(1, T))) if below[-1] else -1)   # last step after which the speed stays below the threshold
    settle = np.array(settle); settle = T - 1 - settle[settle >= 0]
    sp = dict(contrast=float(q90 / q10), median_profile=np.median(speed, 0)[::max(1, (T - 1) // 40)].round(5).tolist(), n_fast_phases_mean=float(np.mean(n_fast)),
              frac_more_than_one=float(np.mean(np.array(n_fast) > 1)), settles_frac=float(len(settle) / N), settle_step_median=float(np.median(settle)) if len(settle) else None)
    # ---- one operator for all trajectories (64 leading axes; the map predicts the change)
    tr = np.arange(N) < N // 2; te = ~tr; U, S, Vt = np.linalg.svd(Zs[tr].reshape(-1, D) - Zs[tr].reshape(-1, D).mean(0), full_matrices=False); P = Vt[:64].T
    Y = Zs @ P; var64 = float((S[:64] ** 2).sum() / (S ** 2).sum())
    def fit(q):
        Xs = np.concatenate([np.concatenate([Y[i, t - q + 1:t + 1][::-1].ravel()[None] for t in range(q - 1, T - 1)]) for i in np.flatnonzero(tr)])
        Ys = np.concatenate([Y[i, q:T] - Y[i, q - 1:T - 1] for i in np.flatnonzero(tr)])
        mx, my = Xs.mean(0), Ys.mean(0); G = (Xs - mx).T @ (Xs - mx); G[np.diag_indices_from(G)] += 1e-2 * np.trace(G) / G.shape[0]
        W = np.linalg.solve(G, (Xs - mx).T @ (Ys - my)); b = my - mx @ W
        Xt = np.concatenate([np.concatenate([Y[i, t - q + 1:t + 1][::-1].ravel()[None] for t in range(q - 1, T - 1)]) for i in np.flatnonzero(te)])
        Yt = np.concatenate([Y[i, q:T] for i in np.flatnonzero(te)]); cur = Xt[:, :64]
        return float(1 - ((cur + Xt @ W + b - Yt) ** 2).sum() / ((cur - Yt) ** 2).sum()), W
    s1, W1 = fit(1); s8, _ = fit(8)
    ev = np.linalg.eigvals((W1 + np.eye(64)).T); rot = (np.abs(ev) >= 0.95) & (np.abs(np.angle(ev)) > 2 * np.pi / (T / 2))
    op = dict(variance_in_64_axes=var64, share_q1=s1, share_q8=s8, n_near_unit=int((np.abs(ev) >= 0.95).sum()), n_sustained_rotating=int(rot.sum()),
              periods_rotating=sorted((2 * np.pi / np.abs(np.angle(ev[rot]))).round(1).tolist())[:12], modulus_quantiles=np.quantile(np.abs(ev), [.1, .5, .9]).round(3).tolist())
    med = lambda k: float(np.median([r[k] for r in res if r.get(k) is not None]))
    mc_n = [r['mc']['n_osc'] for r in res if 'n_osc' in r['mc']]; mc_x = [r['mc']['excess_top'] for r in res if 'excess_top' in r['mc']]
    summary = dict(tag=A.tag, N=int(N), T=int(T), d=int(D), L=int(L_FULL), pr=med('pr'), r95=med('r95'), d_osc=med('d_osc'), n_modes=med('n_modes'), n_sust=med('n_sust'), med_abs=med('med_abs'),
                   periods_median=float(np.median(np.concatenate([r['periods'] for r in res if r['periods']]))) if any(r['periods'] for r in res) else None,
                   sync=med('sync'), mc_n_osc=float(np.median(mc_n)) if mc_n else None, mc_excess_top=float(np.median(mc_x)) if mc_x else None,
                   thirds={k: dict(pr=float(np.median([r['thirds'][k]['pr'] for r in res if r['thirds'][k]['pr'] is not None])), sync=float(np.median([r['thirds'][k]['sync'] for r in res if r['thirds'][k]['sync'] is not None]))) for k in ('first', 'middle', 'last')},
                   speed=sp, operator=op)
    json.dump(dict(summary=summary, per_trajectory=res), open(f'{B}/signature_{A.tag}.json', 'w'))
    print(json.dumps(summary, indent=None)[:1500]); print(f'SIGNATURE_DONE {A.tag} in {time.time() - t0:.0f}s', flush=True)
