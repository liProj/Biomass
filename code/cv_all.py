"""Unified out-of-sample run: for every fold, fit every candidate on the training rows and store test predictions
together with the training-set Fobj / BIC_total of the candidate (needed for BIC selection & averaging).
Candidates
  orig-simpl, orig-full                       original two-step method (paper's simplified / full F set)
  J[s|f][+kd][+proc]                           joint one-step family, F set simplified (s) or full (f)
  FIP[s|f](lam,sp)                             fingerprint-informed-prior MAP with processivity + deactivation"""
import sys, os, time, json, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from methods import *
from cv import folds

GRID = [(lam, sp) for lam in (1.0, 3.0) for sp in (0.05, 0.2)]

def candidates(case, rows):
    C = {"orig-simpl": two_step(case, rows), "orig-full": two_step(case, rows, full=True)}
    for tag, Fs in (("s", SIMPL[case]), ("f", FULL)):
        j = joint(case, rows, Fset=Fs, n_starts=6, init=[C["orig-simpl"] if tag == "s" else C["orig-full"]])
        jk = joint(case, rows, Fset=Fs, kd=True, init=[j], n_starts=6)
        jp = joint(case, rows, Fset=Fs, proc=True, init=[j], n_starts=6)
        jpk = joint(case, rows, Fset=Fs, proc=True, kd=True, init=[jk, jp], n_starts=6)
        C[f"J{tag}"], C[f"J{tag}+kd"], C[f"J{tag}+proc"], C[f"J{tag}+proc+kd"] = j, jk, jp, jpk
        for lam, sp in GRID:
            C[f"FIP{tag}(lam={lam:g},sp={sp:g})"] = map_fit_F(case, rows, Fs, lam, sp, base=jpk)
    return C

def map_fit_F(case, rows, Fs, lam, sp, init=None, base=None):
    import methods
    old = methods.SIMPL[case]; methods.SIMPL[case] = Fs
    try:
        return map_fit(case, rows, proc=True, kd=True, lam=lam, s_p=sp, init=init, base=base, n_starts=2)
    finally:
        methods.SIMPL[case] = old

def run_fold(args):
    import pickle
    case, proto, test, fid = args
    cache = f"{JD}/results/cv/cache_{proto}/{case}_{fid}.pkl"
    if os.path.exists(cache): return pickle.load(open(cache, "rb"))
    rows = all_rows(case) - set(test)
    C = candidates(case, rows)
    recs = []
    for name, m in C.items():
        R, pr = predict_rows(m, test)
        o = R[SPECIES].to_numpy(float)
        for j in range(len(R)):
            for s, spc in enumerate(SPECIES):
                if not np.isnan(o[j, s]):
                    recs.append(dict(case=case, proto=proto, fold=fid, method=name, row=int(R.index[j]), dataset=R.dataset.iloc[j],
                                     t=float(R.t.iloc[j]), species=spc, obs=o[j, s], pred=pr[j, s],
                                     train_F=m.F, train_nT=m.nT, train_k=m.k_total, train_BIC=m.BIC_total))
    os.makedirs(os.path.dirname(cache), exist_ok=True); pickle.dump(recs, open(cache, "wb"))
    return recs

if __name__ == "__main__":
    proto = sys.argv[1]; out = f"{JD}/results/cv"
    jobs = []
    for c in CASES:
        for i, f in enumerate(folds(c, proto)): jobs.append((c, proto, f, i))
    order = {"Mao123": 0, "Tao035_045": 1, "Tao025": 2, "Bornadel": 3, "Akerman100": 4, "Akerman70": 5}
    jobs.sort(key=lambda j: order[j[0]])          # heaviest first for load balancing
    t0 = time.time(); recs = []
    with ProcessPoolExecutor(int(os.environ.get("NPROC", 14))) as ex:
        for i, r in enumerate(ex.map(run_fold, jobs, chunksize=1)):
            recs += r
            if i % 10 == 0:
                print(f"  {proto} {i+1}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(recs); df.to_parquet(f"{out}/{proto}_all.parquet") if False else df.to_csv(f"{out}/{proto}_all.csv.gz", index=False)
    df["se"] = (df.pred - df.obs) ** 2
    print(df.groupby(["case", "method"]).se.mean().pow(0.5).unstack().T.round(3).to_string())
