"""Core kinetic models of Nishimura et al. (Biomass 2026, 6, 12) + shared fitting utilities.
State order: A, Z, M, D, T  (molar % relative to initial TMP).  Rate prefactor kE = kA*[E]T."""
import numpy as np, pandas as pd, os
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); JD = os.path.dirname(HERE)
SPECIES = ["Z", "M", "D", "T"]

def load_profiles():
    return pd.read_csv(f"{JD}/data/profiles.csv")

A0 = {"Akerman70": 300.0, "Akerman100": 300.0, "Bornadel": 300.0, "Tao025": 320.0, "Tao035": 320.0,
      "Tao045": 320.0, "Mao1": 1000 / 3.3, "Mao2": 1000 / 3.3, "Mao3": 1000 / 3.3, "Mao4": 1000 / 3.3, "Mao5": 1000 / 3.3}  # 10 mol acid : 3.3 mol TMP (Suppl. S2)

# ---------------------------------------------------------------- fingerprinting model (Eqs 16-19)
def fp_rhs(c, y, SM, SD):
    Z, M, D, T = y
    S = Z + SM * M + SD * D
    S = max(S, 1e-12)
    return [-Z / S, (Z - SM * M) / S, (SM * M - SD * D) / S, SD * D / S]

def fp_predict(cgrid, SM, SD):
    cgrid = np.asarray(cgrid, float)
    cmax = max(cgrid.max(), 1e-9)
    sol = solve_ivp(fp_rhs, (0, cmax), [100, 0, 0, 0], args=(SM, SD), t_eval=np.sort(np.unique(cgrid)),
                    method="LSODA", rtol=1e-10, atol=1e-10)
    out = np.empty((len(cgrid), 4))
    for j, c in enumerate(cgrid):
        out[j] = sol.y[:, np.searchsorted(sol.t, c)]
    return out

# ---------------------------------------------------------------- dynamic model (Eqs 30-34)
def dyn_rhs(t, y, kE, SM, SD, FA, FAZ, FAM, FAD, extra=None):
    A, Z, M, D, T = y
    A = max(A, 0.0)
    den = FA * A + FAZ * A * Z + FAM * A * M + FAD * A * D + Z + SM * M + SD * D
    r = kE * A / max(den, 1e-12)
    if extra is not None:
        r *= extra(t, y)
    rZ, rM, rD = r * Z, r * SM * M, r * SD * D
    return [-(rZ + rM + rD), -rZ, rZ - rM, rM - rD, rD]

def dyn_predict(tgrid, A0_, kE, SM, SD, FA, FAZ, FAM, FAD, extra=None):
    tgrid = np.asarray(tgrid, float)
    tu = np.unique(tgrid)
    sol = solve_ivp(dyn_rhs, (0, tu.max()), [A0_, 100, 0, 0, 0], args=(kE, SM, SD, FA, FAZ, FAM, FAD, extra),
                    t_eval=tu, method="LSODA", rtol=1e-9, atol=1e-9)
    if not sol.success or sol.y.shape[1] != len(tu):
        return np.full((len(tgrid), 4), 1e3)
    return sol.y[1:, np.searchsorted(tu, tgrid)].T

def obs_matrix(d):
    """Observed Z,M,D,T (NaN where not reported) for rows with t>0 (t=0 rows carry zero residual)."""
    return d[SPECIES].to_numpy(float)

def bic(F, nT, k):
    return nT * np.log(F / nT) + k * np.log(nT)

def wls_ci(resid_fun, theta_hat, nT, alpha=0.05, h=1e-5, transform=None):
    """Linearised covariance (Schwaab-type): cov = s^2 (J'J)^-1, s^2 = Fobj/(nT-k); J by central differences
    on the *natural* parameters.  Returns half-widths (t-dist) and correlation matrix."""
    k = len(theta_hat)
    r0 = resid_fun(theta_hat)
    J = np.empty((len(r0), k))
    for i in range(k):
        step = h * max(abs(theta_hat[i]), 1e-12)
        tp, tm = theta_hat.copy(), theta_hat.copy()
        tp[i] += step; tm[i] -= step
        J[:, i] = (resid_fun(tp) - resid_fun(tm)) / (2 * step)
    F = float(r0 @ r0)
    s2 = F / max(nT - k, 1)
    JTJ = J.T @ J
    cov = s2 * np.linalg.pinv(JTJ)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    tq = stats.t.ppf(1 - alpha / 2, max(nT - k, 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / np.outer(se, se)
    return tq * se, corr, cov

def multistart_lsq(fun, x0s, lb=-np.inf, ub=np.inf, **kw):
    best = None
    for x0 in x0s:
        try:
            r = least_squares(fun, x0, bounds=(lb, ub), method="trf", x_scale="jac",
                              xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=4000, **kw)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best

# ---------------------------------------------------------------- exact fast solver (shared-denominator structure)
# Because all three esterifications share one denominator, the composition path in c is the fingerprint
# solution and time is a LINEAR combination of five path integrals:
#   kE * t(c) = I0(c) + FA*I1(c) + FAZ*IZ(c) + FAM*IM(c) + FAD*ID(c)
#   I0 = ln(A0/(A0-c)),  I1 = int dc/S,  IX = int X/S dc,  S = Z + SM*M + SD*D.
from functools import lru_cache

def _fp_aug_rhs(c, y, SM, SD):
    Z, M, D, T, I1, IZ, IM, ID = y
    S = max(Z + SM * M + SD * D, 1e-300)
    return [-Z / S, (Z - SM * M) / S, (SM * M - SD * D) / S, SD * D / S, 1 / S, Z / S, M / S, D / S]

@lru_cache(maxsize=256)
def fp_grid(SM, SD, cend=299.99999, n=8000):
    # dense grid, refined near completion where S -> 0
    u = np.linspace(0, 1, n)
    cg = cend * (1 - (1 - u) ** 2.5)
    sol = solve_ivp(_fp_aug_rhs, (0, cend), [100, 0, 0, 0, 0, 0, 0, 0], args=(SM, SD), t_eval=cg,
                    method="LSODA", rtol=1e-10, atol=1e-12)
    Y = sol.y
    if Y.shape[1] < n:  # integration stopped early (S underflow) -> pad with last state
        pad = np.repeat(Y[:, -1:], n - Y.shape[1], axis=1); Y = np.hstack([Y, pad]); Y[4:, sol.y.shape[1]:] = np.inf
    return cg, Y

def fp_at_c(c, SM, SD):
    cg, Y = fp_grid(float(SM), float(SD))
    return np.stack([np.interp(c, cg, Y[i]) for i in range(4)], 1)

def tau_of_c(SM, SD, A0_, FA, FAZ, FAM, FAD):
    """kE * t as a function of the c grid."""
    cg, Y = fp_grid(float(SM), float(SD))
    with np.errstate(divide="ignore"):
        I0 = np.log(A0_ / np.clip(A0_ - cg, 1e-300, None))
    return cg, Y, I0 + FA * Y[4] + FAZ * Y[5] + FAM * Y[6] + FAD * Y[7]

def dyn_fast(t, A0_, kE, SM, SD, FA, FAZ, FAM, FAD, kd=0.0):
    """Exact composition at times t (optionally with first-order enzyme deactivation kd:
    active time tau = (1-exp(-kd t))/kd replaces t)."""
    t = np.asarray(t, float)
    tau = t if kd <= 0 else (1 - np.exp(-kd * t)) / kd
    cg, Y, tc = tau_of_c(SM, SD, A0_, FA, FAZ, FAM, FAD)
    tc = np.maximum.accumulate(np.nan_to_num(tc, nan=np.inf, posinf=1e300))
    c = np.interp(kE * tau, tc, cg)
    return np.stack([np.interp(c, cg, Y[i]) for i in range(4)], 1), c
