"""Out-of-sample evaluation.  Protocols:
  loto  : leave-one-time-point-out (all species at one sampling time of one profile held out), refit everything
  fcst  : forward extrapolation - train on the sampling times with t <= 50% of each profile's horizon, predict the rest
For each fold every method is refit from scratch on the training rows only (the original two-step method
re-estimates SM, SD from the training rows too)."""
import sys, json, numpy as np, pandas as pd, os, time
from concurrent.futures import ProcessPoolExecutor
from methods import *

METHODS = ["orig-simpl", "orig-full", "joint", "joint+kd", "joint+proc", "joint+proc+kd", "GPK(BIC)"]

def fit_all(case, rows):
    ms = {"orig-simpl": two_step(case, rows), "orig-full": two_step(case, rows, full=True)}
    ms.update(family(case, rows))
    fam = ["joint", "joint+kd", "joint+proc", "joint+proc+kd"]
    best = min(fam, key=lambda k: ms[k].BIC_total)
    ms["GPK(BIC)"] = ms[best]; ms["_choice"] = best
    return ms

def folds(case, proto):
    R = P[P.dataset.isin(CASES[case]) & (P.t > 0)]
    if proto == "loto":
        return [[i] for i in R.index]
    if proto == "fcst":
        test = []
        for d, g in R.groupby("dataset"):
            test += list(g.index[g.t > 0.5 * g.t.max()])
        return [test]

def run_fold(args):
    case, proto, test = args
    rows = all_rows(case) - set(test)
    ms = fit_all(case, rows)
    recs = []
    for name in METHODS:
        R, pr = predict_rows(ms[name], test)
        o = R[SPECIES].to_numpy(float)
        for j in range(len(R)):
            for s, sp in enumerate(SPECIES):
                if not np.isnan(o[j, s]):
                    recs.append(dict(case=case, proto=proto, method=name, row=int(R.index[j]), dataset=R.dataset.iloc[j],
                                     t=float(R.t.iloc[j]), species=sp, obs=o[j, s], pred=pr[j, s], choice=ms["_choice"]))
    return recs

if __name__ == "__main__":
    protos = sys.argv[1].split(",") if len(sys.argv) > 1 else ["fcst", "loto"]
    out = f"{JD}/results/cv"; os.makedirs(out, exist_ok=True)
    for proto in protos:
        jobs = [(c, proto, f) for c in CASES for f in folds(c, proto)]
        t0 = time.time(); recs = []
        nproc = int(os.environ.get("NPROC", 14))
        with ProcessPoolExecutor(nproc) as ex:
            for i, r in enumerate(ex.map(run_fold, jobs, chunksize=1)):
                recs += r
                if i % 10 == 0:
                    pd.DataFrame(recs).to_csv(f"{out}/{proto}_predictions.partial.csv", index=False)
                    print(f"  {proto}: {i+1}/{len(jobs)} folds, {time.time()-t0:.0f}s", flush=True)
        df = pd.DataFrame(recs); df.to_csv(f"{out}/{proto}_predictions.csv", index=False)
        df["se"] = (df.pred - df.obs) ** 2
        tab = df.groupby(["case", "method"]).se.mean().pow(0.5).unstack()[METHODS]
        print(f"== {proto}  ({len(jobs)} folds, {time.time()-t0:.0f}s)  RMSE (molar %)\n", tab.round(3).to_string(), flush=True)
