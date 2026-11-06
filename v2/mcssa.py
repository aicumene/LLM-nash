"""Monte-Carlo SSA oscillator index with a 1/f^chi (aperiodic) null — standalone CPU port of eeg_jpca.jpca_index
(the version used for the calcium and CTM signatures), so that CMM trajectories are scored the same way.
X: (T, C) window; returns n_osc (planes above the null), excess_top, freq_top (cycles per sample * fs), d_osc_mc, ..."""
import numpy as np
from scipy.signal import welch

EPS = 1e-12


def zscore(X):
    X = X - X.mean(0); sd = X.std(0); sd[sd < 1e-8] = 1.0
    return X / sd


def hankel(X, L):
    T, C = X.shape; K = T - L + 1
    return np.concatenate([X[l:l + K] for l in range(L)], axis=1)


def skew_fit(Z, fs):
    T, k = Z.shape; Zd = np.gradient(Z, 1.0 / fs, axis=0)
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    Dm = np.zeros((T * k, len(pairs)))
    for c, (i, j) in enumerate(pairs):
        col = np.zeros((T, k)); col[:, j] = Z[:, i]; col[:, i] = -Z[:, j]; Dm[:, c] = col.reshape(-1)
    y = Zd.reshape(-1); m, *_ = np.linalg.lstsq(Dm, y, rcond=None)
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + EPS
    skew_r2 = 1.0 - float(np.sum((y - Dm @ m) ** 2)) / ss_tot
    B = np.zeros((k, k))
    for c, (i, j) in enumerate(pairs):
        B[i, j], B[j, i] = m[c], -m[c]
    eig = np.linalg.eigvals(B); rot_freq = float(np.max(np.abs(eig.imag)) / (2 * np.pi)) if eig.size else np.nan
    M, *_ = np.linalg.lstsq(Z, Zd, rcond=None); lin_r2 = 1.0 - float(np.sum((Zd - Z @ M) ** 2)) / ss_tot
    return dict(skew_r2=skew_r2, lin_r2=lin_r2, rot_share=skew_r2 / (lin_r2 + EPS), rot_freq=rot_freq)


def _dom_freq(z, fs):
    f = np.fft.rfftfreq(z.size, 1.0 / fs); P = np.abs(np.fft.rfft(z - z.mean())) ** 2; P[0] = 0
    return float(f[int(np.argmax(P))])


def pair_planes(Z, S, fs, sv_ratio=0.8, freq_tol=0.25):
    k = Z.shape[1]; freqs = [_dom_freq(Z[:, i], fs) for i in range(k)]; planes, i = [], 0
    while i < k - 1:
        same_f = abs(freqs[i] - freqs[i + 1]) <= freq_tol * max(freqs[i], freqs[i + 1], 1e-9)
        if S[i + 1] / (S[i] + EPS) >= sv_ratio and same_f:
            planes.append((i, i + 1, 0.5 * (freqs[i] + freqs[i + 1]))); i += 2
        else:
            i += 1
    cons = []
    for (i, j, f) in planes:
        x, y = Z[:, i], Z[:, j]; dx, dy = np.gradient(x, 1.0 / fs), np.gradient(y, 1.0 / fs); Lz = x * dy - y * dx
        cons.append(abs(Lz.mean()) / (np.abs(Lz).mean() + EPS))
    return planes, np.array(cons)


def aperiodic_fit(x, fs, fmin=0.5, excl=0.3, iters=6):
    fw, P = welch(x, fs=fs, nperseg=min(x.size, int(4 * fs)))
    m = (fw >= fmin) & (P > 0)
    if m.sum() < 8:
        return float(np.log10(P.mean() + EPS)), 0.0
    lx, ly = np.log10(fw[m]), np.log10(P[m]); keep = np.ones(lx.size, bool)
    for _ in range(iters):
        b, a = np.polyfit(lx[keep], ly[keep], 1); new = (ly - (a + b * lx)) < excl
        if new.sum() < 8 or np.array_equal(new, keep):
            break
        keep = new
    return a, b


def aperiodic_surrogates(Xz, fs, n, rng, fmin=0.5):
    T, C = Xz.shape; f = np.fft.rfftfreq(T, 1.0 / fs); out = np.empty((n, T, C), dtype=np.float32)
    for c in range(C):
        a, b = aperiodic_fit(Xz[:, c], fs, fmin); amp = np.zeros(f.size); pos = f >= f[1]
        amp[pos] = np.sqrt(10 ** (a + b * np.log10(np.maximum(f[pos], fmin))))
        W = np.fft.rfft(rng.standard_normal((n, T)), axis=1); x = np.fft.irfft(W * amp[None, :], T, axis=1)
        out[:, :, c] = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + EPS)
    return out


def jpca_index(X, fs, L, k=10, n_surr=20, alpha=0.05, rng=None, null="aperiodic"):
    rng = np.random.default_rng() if rng is None else rng
    Xz = zscore(np.asarray(X, np.float64)).astype(np.float32); T, C = Xz.shape
    H = hankel(Xz, L); H = H - H.mean(axis=0, keepdims=True); k = int(min(k, H.shape[1], H.shape[0] - 2))
    U, S, Vt = np.linalg.svd(H, full_matrices=False); V = Vt[:k].T; Z = (U[:, :k] * S[:k]).astype(np.float64)
    if null == "aperiodic":
        surr = aperiodic_surrogates(Xz, fs, n_surr, rng)
    else:
        surr = np.stack([np.take_along_axis(Xz, np.argsort(rng.random(Xz.shape), axis=0), axis=0) for _ in range(n_surr)])
    lam_null = np.empty((n_surr, k))
    for i in range(n_surr):
        Hs = hankel(surr[i], L); Hs = Hs - Hs.mean(axis=0, keepdims=True); lam_null[i] = np.sum((Hs @ V) ** 2, axis=0)
    lam_obs = S[:k] ** 2; thr_raw = np.quantile(lam_null, 1 - alpha, axis=0)
    n_ref = min(6, n_surr - 3); bias = np.ones(k)
    if n_ref >= 2:
        ratios = []
        for m in range(n_ref):
            Hm = hankel(surr[m], L); Hm = Hm - Hm.mean(axis=0, keepdims=True)
            Um, Sm, Vtm = np.linalg.svd(Hm, full_matrices=False); Vm = Vtm[:k].T
            others = np.stack([np.sum(((hankel(surr[i], L) - hankel(surr[i], L).mean(axis=0, keepdims=True)) @ Vm) ** 2, axis=0) for i in range(n_surr) if i != m])
            ratios.append(Sm[:k] ** 2 / (np.quantile(others, 1 - alpha, axis=0) + EPS))
        bias = np.maximum(np.median(np.stack(ratios), axis=0), 1.0)
    thr = thr_raw * bias
    obs = skew_fit(Z, fs); planes, cons = pair_planes(Z, S[:k], fs)
    sig = np.array([lam_obs[i] > thr[i] and lam_obs[j] > thr[j] for (i, j, _) in planes], bool)
    excess = [float(min(lam_obs[i] / thr[i], lam_obs[j] / thr[j])) for (i, j, _) in planes]
    n_osc = int(sig.sum()); top = int(np.argmax(excess)) if planes else -1
    p = S ** 2 / (S ** 2).sum(); ex_all = lam_obs > thr
    d_mc = int(np.argmin(ex_all)) if (ex_all[0] and not ex_all.all()) else (k if ex_all.all() else 0)
    return dict(skew_r2=obs["skew_r2"], lin_r2=obs["lin_r2"], rot_share=obs["rot_share"], rot_freq=obs["rot_freq"],
                n_planes=len(planes), n_osc=n_osc, d_osc_mc=d_mc, n_sig_components=int(ex_all.sum()),
                freqs_sig=[float(planes[i][2]) for i in range(len(planes)) if sig[i]],
                excess_sig=[excess[i] for i in range(len(planes)) if sig[i]],
                freq_top=float(planes[top][2]) if top >= 0 else np.nan, excess_top=float(excess[top]) if top >= 0 else np.nan,
                cons_top=float(cons[top]) if top >= 0 else np.nan, pr=float(1.0 / np.sum(p ** 2)), k=k)
