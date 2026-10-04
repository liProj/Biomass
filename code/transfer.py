"""Cross-condition fingerprint transfer: learn the composition path (selectivities, and processivity for the new
model) from the OTHER profiles of the same reaction system and predict the held-out profile in the c-domain
(Z, M, D, T versus degree of reaction).  Tests whether the path parameters are transferable system properties.
Folds: Akerman {70 | 100}, Tao {0.25 | 0.35 | 0.45}, Mao {1 | 2 | 3}."""
import numpy as np, pandas as pd, os
from scipy.optimize import least_squares
from models import comp_at_c
from kinetics import load_profiles, SPECIES, JD
P = load_profiles()
SYS = {"Akerman": ["Akerman70", "Akerman100"], "Tao": ["Tao025", "Tao035", "Tao045"], "Mao": ["Mao1", "Mao2", "Mao3"]}

def fit_path(dsets, proc, s_p=None):
    B = [P[(P.dataset == d) & (P.t > 0)] for d in dsets]
    def f(x):
        SM, SD = np.exp(x[:2]); pM, pD = (x[2], x[3]) if proc else (0, 0)
        r = []
        for b in B:
            pr = comp_at_c(b.c.to_numpy(), SM, SD, pM, pD); o = b[SPECIES].to_numpy(float); m = ~np.isnan(o)
            r.append((pr - o)[m])
        r = np.concatenate(r)
        if proc and s_p: r = np.r_[r, np.sqrt(r @ r / len(r)) * np.array([pM, pD]) / s_p]
        return r
    best = None
    for a in (0.3, 0.8, 2.0):
        for s in (0.03, 0.3):
            x0 = np.r_[np.log([a, s]), [0.05, 0.05]] if proc else np.log([a, s])
            lo = np.r_[np.log([1e-3, 1e-4]), [0, 0]] if proc else np.log([1e-3, 1e-4])
            hi = np.r_[np.log([20, 20]), [0.95, 0.95]] if proc else np.log([20, 20])
            r = least_squares(f, x0, bounds=(lo, hi), xtol=1e-12, ftol=1e-12)
            if best is None or r.cost < best.cost: best = r
    x = best.x
    return (np.exp(x[0]), np.exp(x[1]), x[2] if proc else 0, x[3] if proc else 0)

if __name__ == "__main__":
    recs = []
    for sysn, ds in SYS.items():
        for held in ds:
            train = [d for d in ds if d != held]
            b = P[(P.dataset == held) & (P.t > 0)]; o = b[SPECIES].to_numpy(float)
            for name, kw in (("ORIG fingerprint", dict(proc=False)), ("processive fingerprint", dict(proc=True)),
                             ("processive fingerprint (shrunk, s_p=0.1)", dict(proc=True, s_p=0.1))):
                th = fit_path(train, **kw)
                pr = comp_at_c(b.c.to_numpy(), *th)
                for j in range(len(b)):
                    for s, sp in enumerate(SPECIES):
                        if not np.isnan(o[j, s]):
                            recs.append(dict(system=sysn, held_out=held, method=name, c=b.c.iloc[j], species=sp, obs=o[j, s],
                                             pred=pr[j, s], SM=th[0], SD=th[1], pM=th[2], pD=th[3]))
    df = pd.DataFrame(recs); os.makedirs(f"{JD}/results/transfer", exist_ok=True)
    df.to_csv(f"{JD}/results/transfer/fingerprint_transfer.csv", index=False)
    df["se"] = (df.pred - df.obs) ** 2
    print(df.groupby(["held_out", "method"]).se.mean().pow(.5).unstack().round(3).to_string())
    print(df.groupby(["method"]).se.mean().pow(.5).round(3).to_string())
