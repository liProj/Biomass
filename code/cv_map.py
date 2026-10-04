"""Out-of-sample evaluation of the fingerprint-informed-prior (FIP) MAP estimator over a hyper-parameter grid,
on exactly the same folds as cv.py.  The unpenalised joint fit of each fold is computed once and shared.
Hyper-parameters (lam: prior width multiplier on the fingerprint CI; s_p: processivity shrinkage) are later
chosen per case by leave-one-CASE-out meta-validation (select_hparams.py) - no information from the target case."""
import sys, os, time, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from methods import *
from cv import folds

GRID = [(lam, sp) for lam in (1.0, 3.0, 10.0) for sp in (0.05, 0.2)]

def run_fold(args):
    case, proto, test = args
    rows = all_rows(case) - set(test)
    base = family(case, rows)
    recs = []
    for lam, sp in GRID:
        m = map_fit(case, rows, proc=True, kd=True, lam=lam, s_p=sp, init=[base["joint+proc+kd"], base["joint+kd"], base["joint+proc"]], n_starts=4)
        R, pr = predict_rows(m, test)
        o = R[SPECIES].to_numpy(float)
        for j in range(len(R)):
            for s, spc in enumerate(SPECIES):
                if not np.isnan(o[j, s]):
                    recs.append(dict(case=case, proto=proto, method=f"FIP(lam={lam:g},sp={sp:g})", lam=lam, s_p=sp, row=int(R.index[j]),
                                     dataset=R.dataset.iloc[j], t=float(R.t.iloc[j]), species=spc, obs=o[j, s], pred=pr[j, s]))
    return recs

if __name__ == "__main__":
    proto = sys.argv[1]; out = f"{JD}/results/cv"
    jobs = [(c, proto, f) for c in CASES for f in folds(c, proto)]
    t0 = time.time(); recs = []
    with ProcessPoolExecutor(int(os.environ.get("NPROC", 14))) as ex:
        for i, r in enumerate(ex.map(run_fold, jobs, chunksize=1)):
            recs += r
            if i % 10 == 0:
                pd.DataFrame(recs).to_csv(f"{out}/{proto}_map.partial.csv", index=False)
                print(f"  {i+1}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(recs); df.to_csv(f"{out}/{proto}_map.csv", index=False)
    df["se"] = (df.pred - df.obs) ** 2
    print(df.groupby(["case", "method"]).se.mean().pow(0.5).unstack().round(3).to_string())
