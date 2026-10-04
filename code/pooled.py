"""Multi-task (pooled) PPK-FIP: profiles of the same reaction system share the composition path (S_M, S_D, p_M,
p_D) and the lumped constants F_*, while k_E and k_d are condition-specific.  The original paper already pools the
same profiles in its fingerprint step; here the pooling is extended to the dynamic step.
Training data for a target case = its own training rows + all rows of its sibling profiles."""
import numpy as np
import methods
from methods import *
from fitlib import Model, bic

SIB = {"Akerman70": ["Akerman100"], "Akerman100": ["Akerman70"], "Bornadel": [], "Tao025": ["Tao035", "Tao045"],
       "Tao035_045": ["Tao025"], "Mao123": []}

def pooled_spec(dsets, Fs, proc=True, kd=True, share_F=True):
    s = {}
    for d in dsets:
        s[f"kE@{d}"] = ("free", 0.1, 1e-5, 1e3, True)
        if kd: s[f"kd@{d}"] = ("free", 0.01, 1e-8, 2.0, True)
    s["SM"] = ("free", 0.6, 1e-3, 20, True); s["SD"] = ("free", 0.1, 1e-4, 20, True)
    for f in Fs:
        if share_F: s[f] = ("free", 3e-3, 1e-12, 1e3, True)
        else:
            for d in dsets: s[f"{f}@{d}"] = ("free", 3e-3, 1e-12, 1e3, True)
    if proc: s["pM"] = ("free", 0.05, 0.0, 0.95, False); s["pD"] = ("free", 0.05, 0.0, 0.95, False)
    return s

def pooled_map(case, rows, Fs, lam, sp, n_starts=6, share_F=True, tau=None):
    """MAP of the pooled model with the fingerprint-informed prior; returns a Model whose predictions for the target
    case's datasets are used.  Selectivity prior from the pooled fingerprint (same FPG group as the original)."""
    dsets = CASES[case] + SIB[case]
    rows_all = set(rows) | set(P[P.dataset.isin(SIB[case]) & (P.t > 0)].index)
    trainfp = set(rows) | (set(P[P.dataset.isin(FPG[case])].index) - all_rows(case))
    SM0, SD0 = fit_fingerprint(FPG[case], trainfp)
    hw = fingerprint_ci(FPG[case], trainfp, SM0, SD0)
    sdl = lam * np.maximum(hw / np.array([SM0, SD0]) / 1.96, 0.02)
    m = Model(dsets, pooled_spec(dsets, Fs, share_F=share_F), rows=rows_all).fit(n_starts=n_starts)
    sig = np.sqrt(m.F / max(m.nT - m.k, 1))
    pri = {"SM": (np.log(SM0), sdl[0]), "SD": (np.log(SD0), sdl[1]), "pM": (0.0, sp), "pD": (0.0, sp)}
    for d in dsets: pri[f"kd@{d}"] = (np.log(1e-3), 1.5 * np.log(10))
    lo, hi, lg, _ = m._tx(); idx = {k: m.free.index(k) for k in pri}
    hier = []   # partial pooling: log F@d shrunk towards the across-condition mean (scale tau)
    if tau is not None and not share_F:
        for f in Fs:
            hier.append([m.free.index(f"{f}@{d}") for d in dsets])
    def obj(x):
        r = m.resid(m.to_theta(x))
        h = [sig * (x[ii] - x[ii].mean()) / tau for ii in hier]
        return np.r_[r, [sig * (x[idx[k]] - mu) / s for k, (mu, s) in pri.items()], *h]
    from scipy.optimize import least_squares
    best = None
    for s0 in [m.x] + [np.clip(m.x + np.random.default_rng(i).normal(0, 0.5, len(m.x)), lo + 1e-9, hi - 1e-9) for i in range(2)]:
        r = least_squares(obj, s0, bounds=(lo, hi), method="trf", x_scale="jac", xtol=1e-10, ftol=1e-10, max_nfev=methods.MAXNFEV)
        if best is None or r.cost < best.cost: best = r
    m.x = best.x; m.theta = m.to_theta(best.x)
    # training BIC on the pooled data (used for the simplified/full weighting)
    res = m.resid(m.theta); m.F = float(res @ res); m.BIC_total = bic(m.F, len(res), m.k); m.k_total = m.k
    return m

def pooled_ppk(case, rows, lam, sp, share_F=True, tau=None):
    if not SIB[case]: return None
    ms = {tag: pooled_map(case, rows, Fs, lam, sp, share_F=share_F, tau=tau) for tag, Fs in (("s", SIMPL[case]), ("f", FULL))}
    b = np.array([ms["s"].BIC_total, ms["f"].BIC_total]); w = np.exp(-(b - b.min()) / 2); w /= w.sum()
    def pred(rr):
        _, a = predict_rows(ms["s"], rr); _, c = predict_rows(ms["f"], rr); return w[0] * a + w[1] * c
    return pred, ms, w
