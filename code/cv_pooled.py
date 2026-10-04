"""LOTO / fcst evaluation of pooled (multi-task) PPK-FIP for the cases that have sibling profiles."""
import sys, os, time, json, pickle, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from pooled import *
from cv import folds
HP = json.load(open(f"{JD}/results/cv/meta_hparams.json"))

def run_fold(args):
    case, proto, test, fid = args
    SHARE = os.environ.get("SHARE_F", "1") == "1"; TAU = os.environ.get("TAU"); tag = ("" if SHARE else "_path") + (f"_tau{TAU}" if TAU else "")
    cache = f"{JD}/results/cv/cache_pool{tag}_{proto}/{case}_{fid}.pkl"
    if os.path.exists(cache): return pickle.load(open(cache, "rb"))
    rows = all_rows(case) - set(test)
    if proto == "fcst":   # siblings are truncated at the same horizon fraction
        pass
    pred, ms, w = pooled_ppk(case, rows, *HP[case], share_F=SHARE, tau=float(np.log(float(TAU))) if TAU else None)
    R = P.loc[sorted(test)]; pr = pred(test); o = R[SPECIES].to_numpy(float); recs = []
    for j in range(len(R)):
        for s, sp in enumerate(SPECIES):
            if not np.isnan(o[j, s]):
                recs.append(dict(case=case, proto=proto, fold=fid, procedure="PPK-FIP-MT" if SHARE else ("PPK-FIP-MThier" if TAU else "PPK-FIP-MTpath"), row=int(R.index[j]), species=sp, err=pr[j, s] - o[j, s]))
    os.makedirs(os.path.dirname(cache), exist_ok=True); pickle.dump(recs, open(cache, "wb"))
    return recs

if __name__ == "__main__":
    proto = sys.argv[1]
    jobs = [(c, proto, f, i) for c in CASES if SIB[c] for i, f in enumerate(folds(c, proto))]
    t0 = time.time(); recs = []
    with ProcessPoolExecutor(int(os.environ.get("NPROC", 10))) as ex:
        for i, r in enumerate(ex.map(run_fold, jobs, chunksize=1)):
            recs += r
            if i % 10 == 0: print(f"  {proto} {i+1}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(recs); tag = ("" if os.environ.get("SHARE_F", "1") == "1" else "_path") + (f"_tau{os.environ['TAU']}" if os.environ.get("TAU") else "")
    df.to_csv(f"{JD}/results/cv/{proto}_pooled{tag}.csv.gz", index=False)
    df["se"] = df.err ** 2; print(df.groupby("case").se.mean().pow(.5).round(3).to_string())
