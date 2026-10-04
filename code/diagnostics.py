"""Diagnostics of the original estimation problem + solver benchmark.
 (1) landscape : 200 random starts of the original FULL dynamic model per case -> distribution of local optima
 (2) profile   : profile least-squares of the full model along log FA (Akerman70, Akerman100, Tao025) -> flat ridge
 (3) bench     : exact reparameterised solver vs adaptive ODE integration (accuracy and wall time per evaluation)"""
import json, time, os, numpy as np, pandas as pd
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
from concurrent.futures import ProcessPoolExecutor
from kinetics import *
import repro
from models import predict
OUT = f"{JD}/results/diagnostics"; os.makedirs(OUT, exist_ok=True)
fp = json.load(open(f"{JD}/results/baseline/fingerprint.json")); P = load_profiles()
CASES = {c[0]: c for c in repro.DYN_CASES}

def landscape(case, n=200):
    name, dsets, grp, zero, bk = CASES[case]
    f, nk = repro.dyn_resid_factory(dsets, fp[grp]["SM"], fp[grp]["SD"], repro.FNAMES, bk)
    rng = np.random.default_rng(7); out = []
    lo = np.log([1e-6] * nk + [1e-14] * 4); hi = np.log([1e3] * nk + [1e3] * 4)
    for i in range(n):
        x0 = np.r_[rng.uniform(np.log(1e-3), np.log(10), nk), rng.uniform(np.log(1e-8), np.log(1.0), 4)]
        try:
            r = least_squares(lambda x: f(np.exp(x)), x0, bounds=(lo, hi), method="trf", max_nfev=2000)
            out.append(dict(case=case, start=i, Fobj=2 * r.cost, logFA=r.x[nk], at_bound=bool(np.any(np.isclose(r.x, hi, atol=1e-3)))))
        except Exception:
            pass
    return out

def profile(case, grid=np.linspace(np.log(1e-8), np.log(1e3), 45)):
    name, dsets, grp, zero, bk = CASES[case]
    f, nk = repro.dyn_resid_factory(dsets, fp[grp]["SM"], fp[grp]["SD"], repro.FNAMES, bk)
    out = []; x = None
    for lfa in grid:
        g = lambda z: f(np.exp(np.r_[z[:nk], lfa, z[nk:]]))
        best = None
        for z0 in ([x] if x is not None else []) + [np.r_[np.log([0.1] * nk), np.log([1e-3] * 3)], np.r_[np.log([1.0] * nk), np.log([0.03] * 3)]]:
            r = least_squares(g, z0, bounds=(np.log([1e-6] * nk + [1e-14] * 3), np.log([1e5] * nk + [1e5] * 3)), max_nfev=3000)
            if best is None or r.cost < best.cost: best = r
        x = best.x; out.append(dict(case=case, logFA=lfa, Fobj=2 * best.cost, logkE=best.x[0]))
    return out

def bench():
    t = P[(P.dataset == "Bornadel") & (P.t > 0)].t.to_numpy()
    th = (300.0, 0.995, 0.522, 0.0347, 0.0, 0.0396, 0.0115, 0.0049, 0.196, 0.020, 0.0)
    def rhs(tt, y, kE, SM, SD, FA, FAZ, FAM, FAD, pM, pD, kd):
        A, Z, M, D, T = y; den = FA * A + FAZ * A * Z + FAM * A * M + FAD * A * D + Z + SM * M + SD * D; r = kE * A / den * np.exp(-kd * tt)
        return [-r * (Z * (1 + pM + pM * pD) + SM * M * (1 + pD) + SD * D), -r * Z, r * (Z * (1 - pM) - SM * M),
                r * (Z * pM * (1 - pD) + SM * M * (1 - pD) - SD * D), r * (Z * pM * pD + SM * M * pD + SD * D)]
    ref = solve_ivp(rhs, (0, t.max()), [300, 100, 0, 0, 0], args=th[1:], t_eval=t, method="Radau", rtol=1e-12, atol=1e-12).y[1:].T
    rows = []
    from models import path
    for meth, tol in (("RK45", 1e-6), ("RK45", 1e-9), ("LSODA", 1e-6), ("LSODA", 1e-9), ("Radau", 1e-9)):
        t0 = time.perf_counter(); n = 0
        while time.perf_counter() - t0 < 2.0:
            y = solve_ivp(rhs, (0, t.max()), [300, 100, 0, 0, 0], args=th[1:], t_eval=t, method=meth, rtol=tol, atol=tol).y[1:].T; n += 1
        rows.append(dict(solver=f"{meth} (tol {tol:g})", ms=(time.perf_counter() - t0) / n * 1e3, max_err=float(np.abs(y - ref).max())))
    for label, clear in (("exact reparam. (path recomputed)", True), ("exact reparam. (path cached: kE, F*, kd change)", False)):
        t0 = time.perf_counter(); n = 0
        while time.perf_counter() - t0 < 2.0:
            if clear: path.cache_clear()
            y = predict(t, *th); n += 1
        rows.append(dict(solver=label, ms=(time.perf_counter() - t0) / n * 1e3, max_err=float(np.abs(y - ref).max())))
    return rows

if __name__ == "__main__":
    pd.DataFrame(bench()).to_csv(f"{OUT}/solver_bench.csv", index=False)
    print(pd.read_csv(f"{OUT}/solver_bench.csv").round(6).to_string(index=False), flush=True)
    with ProcessPoolExecutor(6) as ex:
        L = sum(ex.map(landscape, list(CASES)), [])
        Pf = sum(ex.map(profile, ["Akerman70", "Akerman100", "Tao025"]), [])
    pd.DataFrame(L).to_csv(f"{OUT}/landscape.csv", index=False); pd.DataFrame(Pf).to_csv(f"{OUT}/profile_FA.csv", index=False)
    L = pd.DataFrame(L); print(L.groupby("case").Fobj.describe().round(1).to_string())
