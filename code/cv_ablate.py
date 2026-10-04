"""Ablation CV (same folds): PPK-FIP without pseudo-processivity / without deactivation, at the meta-selected
hyper-parameters of each case, simplified and full F sets (BIC-averaged afterwards like PPK-FIP)."""
import sys, os, time, json, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
import methods
from methods import *
from cv import folds
HP = json.load(open(f"{JD}/results/cv/meta_hparams.json"))

def fip_variant(case, rows, Fs, proc, kd, lam, sp):
    return ppk_chain(case, rows, Fs, lam, sp, proc=proc, kd=kd)

def run_fold(args):
    case, proto, test, fid = args
    rows = all_rows(case) - set(test); lam, sp = HP[case]; recs = []
    import pickle
    cache = f"{JD}/results/cv/cache_abl_{proto}/{case}_{fid}.pkl"
    if os.path.exists(cache): return pickle.load(open(cache, "rb"))
    VARS = (("full", True, True),) if proto == "fcst" else ()
    for vname, proc, kd in VARS + (("no-proc", False, True), ("no-kd", True, False), ("no-proc-no-kd", False, False)):
        for tag, Fs in (("s", SIMPL[case]), ("f", FULL)):
            m = fip_variant(case, rows, Fs, proc, kd, lam, sp)
            R, pr = predict_rows(m, test); o = R[SPECIES].to_numpy(float)
            for j in range(len(R)):
                for s, spc in enumerate(SPECIES):
                    if not np.isnan(o[j, s]):
                        recs.append(dict(case=case, proto=proto, fold=fid, method=f"FIP{tag}[{vname}]", row=int(R.index[j]),
                                         species=spc, obs=o[j, s], pred=pr[j, s], train_BIC=m.BIC_total))
    os.makedirs(os.path.dirname(cache), exist_ok=True); pickle.dump(recs, open(cache, "wb"))
    return recs

if __name__ == "__main__":
    proto = sys.argv[1]
    order = {"Mao123": 0, "Tao035_045": 1, "Tao025": 2, "Bornadel": 3, "Akerman100": 4, "Akerman70": 5}
    jobs = sorted([(c, proto, f, i) for c in CASES for i, f in enumerate(folds(c, proto))], key=lambda j: order[j[0]])
    t0 = time.time(); recs = []
    with ProcessPoolExecutor(int(os.environ.get("NPROC", 14))) as ex:
        for i, r in enumerate(ex.map(run_fold, jobs, chunksize=1)):
            recs += r
            if i % 10 == 0: print(f"  {proto} {i+1}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(recs).to_csv(f"{JD}/results/cv/{proto}_ablate.csv.gz", index=False)
