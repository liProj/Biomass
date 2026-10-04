"""All competing estimators, each refit from scratch on a set of training rows (row indices of profiles.csv).
Original method (two-step): fingerprint fit (SM, SD) on the training rows of the whole fingerprint group, then
dynamic fit with SM, SD fixed (paper's simplified parameter set, or the full set).
New-method family: joint one-step estimation with optional pseudo-processivity (pM, pD) and first-order
enzyme deactivation (kd); 'GPK' = BIC-selected member (selection uses training data only)."""
import json, numpy as np, os
from scipy.optimize import least_squares
from fitlib import Model, P, bic, MAXNFEV
from models import comp_at_c
from kinetics import SPECIES, JD

FPG = {"Akerman70": ["Akerman70", "Akerman100"], "Akerman100": ["Akerman70", "Akerman100"], "Bornadel": ["Bornadel"],
       "Tao025": ["Tao025", "Tao035", "Tao045"], "Tao035_045": ["Tao025", "Tao035", "Tao045"], "Mao123": ["Mao1", "Mao2", "Mao3"]}
CASES = {"Akerman70": ["Akerman70"], "Akerman100": ["Akerman100"], "Bornadel": ["Bornadel"], "Tao025": ["Tao025"],
         "Tao035_045": ["Tao035", "Tao045"], "Mao123": ["Mao1", "Mao2", "Mao3"]}
SIMPL = {"Akerman70": ["FAM", "FAD"], "Akerman100": ["FAM", "FAD"], "Bornadel": ["FAZ", "FAM", "FAD"],
         "Tao025": ["FAZ", "FAM"], "Tao035_045": ["FAZ", "FAM"], "Mao123": ["FAZ", "FAM"]}
FULL = ["FA", "FAZ", "FAM", "FAD"]

def all_rows(case):
    return set(P[P.dataset.isin(CASES[case]) & (P.t > 0)].index)

def fit_fingerprint(dsets, rows):
    blocks = [P[(P.dataset == d) & (P.t > 0) & P.index.isin(rows)] for d in dsets]
    def f(lth):
        SM, SD = np.exp(lth)
        out = []
        for b in blocks:
            if len(b) == 0: continue
            pr = comp_at_c(b.c.to_numpy(), SM, SD); o = b[SPECIES].to_numpy(float); m = ~np.isnan(o)
            out.append((pr - o)[m])
        return np.concatenate(out)
    best = None
    for a in (0.3, 0.8, 2.0):
        for s in (0.03, 0.3):
            r = least_squares(f, np.log([a, s]), bounds=(np.log([1e-3, 1e-4]), np.log([20, 20])), xtol=1e-12, ftol=1e-12)
            if best is None or r.cost < best.cost: best = r
    return tuple(np.exp(best.x))

def kE_spec(case, s):
    if case == "Mao123":
        for d in CASES[case]: s[f"kE@{d}"] = ("free", 0.07, 1e-5, 1e3, True)
    else:
        s["kE"] = ("free", 0.1, 1e-5, 1e3, True)

def two_step(case, rows, full=False, n_starts=10):
    trainfp = set(rows) | (set(P[P.dataset.isin(FPG[case])].index) - all_rows(case))  # other datasets of the group stay in
    SM, SD = fit_fingerprint(FPG[case], trainfp)
    s = {}; kE_spec(case, s)
    s["SM"] = ("fix", SM); s["SD"] = ("fix", SD)
    for f in (FULL if full else SIMPL[case]): s[f] = ("free", 3e-3, 1e-12, 1e3, True)
    m = Model(CASES[case], s, rows=rows).fit(n_starts=n_starts)
    m.k_total = m.k + 2; m.BIC_total = bic(m.F, m.nT, m.k_total)
    return m

def joint(case, rows, proc=False, kd=False, init=None, n_starts=10, Fset=None):
    s = {}; kE_spec(case, s)
    s["SM"] = ("free", 0.6, 1e-3, 20, True); s["SD"] = ("free", 0.1, 1e-4, 20, True)
    for f in (Fset or SIMPL[case]): s[f] = ("free", 3e-3, 1e-12, 1e3, True)
    if proc: s["pM"] = ("free", 0.05, 0.0, 0.95, False); s["pD"] = ("free", 0.05, 0.0, 0.95, False)
    if kd: s["kd"] = ("free", 0.01, 1e-8, 2.0, True)
    m = Model(CASES[case], s, rows=rows)
    x0s = None
    if init is not None:   # warm start(s) from nested model(s)
        lo, hi, lg, x0 = m._tx()
        x0s = []
        for im in init:
            x = x0.copy()
            for i, k in enumerate(m.free):
                if k in im.free:
                    v = im.theta[im.free.index(k)]
                    x[i] = np.log(max(v, 1e-12)) if lg[i] else v
            x0s.append(np.clip(x, lo + 1e-9, hi - 1e-9))
        x0s.append(np.clip(x0, lo + 1e-9, hi - 1e-9))
    m.fit(n_starts=n_starts, x0s=x0s)
    m.k_total = m.k; m.BIC_total = m.BIC
    return m

def family(case, rows, n_starts=8):
    """Fit the nested family with warm starts; returns dict name -> model."""
    out = {}
    out["joint"] = joint(case, rows, n_starts=n_starts)
    out["joint+kd"] = joint(case, rows, kd=True, init=[out["joint"]], n_starts=n_starts)
    out["joint+proc"] = joint(case, rows, proc=True, init=[out["joint"]], n_starts=n_starts)
    out["joint+proc+kd"] = joint(case, rows, proc=True, kd=True, init=[out["joint+kd"], out["joint+proc"]], n_starts=n_starts)
    return out

def predict_rows(m, rows):
    """prediction matrix (n,4) for given row indices of P (any dataset of the case)."""
    R = P.loc[sorted(rows)]
    out = np.full((len(R), 4), np.nan)
    for d in R.dataset.unique():
        idx = np.where(R.dataset.to_numpy() == d)[0]
        out[idx] = m.pred(m.theta, d, R.t.to_numpy()[idx])
    return R, out

# ------------------------------------------------------------------ Bayesian MAP with fingerprint-informed prior
# Two-step  = infinitely tight prior on (SM, SD) at the fingerprint estimate;  joint = flat prior.
# FIP (fingerprint-informed prior): log SM, log SD ~ N(log fingerprint estimate, (lam * relative CI/1.96)^2), i.e. the
# fingerprint step enters as calibrated prior information instead of a hard constraint.  Shrinkage priors keep
# processivity (pM, pD ~ N+(0, s_p)) and deactivation (log10 kd ~ N(-3, 1.5)) near the original model unless the data
# demand otherwise.  MAP by appending prior residuals sigma_e*(u-mu)/s to the least-squares residual vector
# (sigma_e estimated from the unpenalised fit), so the objective is a proper log-posterior up to a constant.
def fingerprint_ci(dsets, rows, SM, SD):
    from kinetics import wls_ci
    blocks = [P[(P.dataset == d) & (P.t > 0) & P.index.isin(rows)] for d in dsets]
    def f(th):
        out = []
        for b in blocks:
            if len(b) == 0: continue
            pr = comp_at_c(b.c.to_numpy(), th[0], th[1]); o = b[SPECIES].to_numpy(float); m = ~np.isnan(o)
            out.append((pr - o)[m])
        return np.concatenate(out)
    r = f(np.array([SM, SD])); hw, _, _ = wls_ci(f, np.array([SM, SD]), len(r))
    return hw

def map_fit(case, rows, proc=True, kd=True, lam=3.0, s_p=0.1, init=None, n_starts=8, base=None):
    trainfp = set(rows) | (set(P[P.dataset.isin(FPG[case])].index) - all_rows(case))
    SM0, SD0 = fit_fingerprint(FPG[case], trainfp)
    hw = fingerprint_ci(FPG[case], trainfp, SM0, SD0)
    sdl = lam * np.maximum(hw / np.array([SM0, SD0]) / 1.96, 0.02)       # sd of log SM, log SD
    if base is not None:   # reuse an already fitted unpenalised model of the same structure (saves a refit)
        import copy; m = copy.copy(base); m.theta = base.theta.copy(); m.x = base.x.copy()
    else:
        m = joint(case, rows, proc=proc, kd=kd, init=init, n_starts=n_starts)  # unpenalised start + sigma_e
    sig = np.sqrt(m.F / max(m.nT - m.k, 1))
    pri = {"SM": (np.log(SM0), sdl[0], True), "SD": (np.log(SD0), sdl[1], True)}
    if proc: pri["pM"] = (0.0, s_p, False); pri["pD"] = (0.0, s_p, False)
    if kd: pri["kd"] = (np.log(1e-3), 1.5 * np.log(10), True)
    lo, hi, lg, _ = m._tx()
    idx = {k: m.free.index(k) for k in pri}
    def obj(x):
        th = m.to_theta(x); r = m.resid(th)
        pr = [sig * (x[idx[k]] - mu) / s for k, (mu, s, _) in pri.items()]
        return np.r_[r, pr]
    starts = [m.x] + [np.clip(m.x + np.random.default_rng(i).normal(0, 0.5, len(m.x)), lo + 1e-9, hi - 1e-9) for i in range(n_starts - 1)]
    starts.append(np.clip(np.where(lg, m.x, 0.0) * 0 + m.x, lo + 1e-9, hi - 1e-9))
    best = None
    for s in starts:
        x0 = s.copy()
        if "SM" in idx: x0[idx["SM"]] = np.log(SM0); x0[idx["SD"]] = np.log(SD0)
        for x00 in (s, np.clip(x0, lo + 1e-9, hi - 1e-9)):
            try:
                r = least_squares(obj, x00, bounds=(lo, hi), method="trf", x_scale="jac", xtol=1e-10, ftol=1e-10, max_nfev=MAXNFEV)
            except Exception:
                continue
            if best is None or r.cost < best.cost: best = r
    m.x = best.x; m.theta = m.to_theta(best.x)
    res = m.resid(m.theta); m.F = float(res @ res)
    m.BIC = bic(m.F, m.nT, m.k); m.k_total = m.k; m.BIC_total = m.BIC
    m.prior = dict(SM0=SM0, SD0=SD0, sdl=sdl.tolist(), lam=lam, s_p=s_p)
    return m

def ppk_chain(case, rows, Fs, lam, sp, proc=True, kd=True, n_starts=6):
    """FIP-MAP fit built exactly as in cv_all.py: two-step start -> joint -> nested warm starts (+kd, +proc, +both)
    -> MAP with the unpenalised model of the requested structure as base.  Used by every experiment so that all
    PPK-FIP variants share one estimation pipeline."""
    old = SIMPL[case]; SIMPL[case] = Fs
    try:
        ts = two_step(case, rows, full=(Fs == FULL))
        j = joint(case, rows, Fset=Fs, n_starts=n_starts, init=[ts])
        if proc and kd:
            jk = joint(case, rows, Fset=Fs, kd=True, init=[j], n_starts=n_starts)
            jp = joint(case, rows, Fset=Fs, proc=True, init=[j], n_starts=n_starts)
            base = joint(case, rows, Fset=Fs, proc=True, kd=True, init=[jk, jp], n_starts=n_starts)
        elif proc:
            base = joint(case, rows, Fset=Fs, proc=True, init=[j], n_starts=n_starts)
        elif kd:
            base = joint(case, rows, Fset=Fs, kd=True, init=[j], n_starts=n_starts)
        else:
            base = j
        return map_fit(case, rows, proc=proc, kd=kd, lam=lam, s_p=sp, base=base, n_starts=2)
    finally:
        SIMPL[case] = old
