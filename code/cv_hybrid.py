"""PPK-FIP + Gaussian-process discrepancy (Kennedy-O'Hagan-type hybrid), on the same folds.
For each fold: refit PPK-FIP (meta-selected hyper-parameters; simplified/full BIC-averaged) and the original
two-step method; fit a GP (Matern-5/2 + white noise, zero prior mean) to the TRAINING residuals of each species
on (t/t_max, condition) with a zero-residual anchor at t=0; add the GP posterior mean to the mechanistic test
prediction.  Far from data the GP reverts to zero, i.e. to the mechanistic model."""
import sys, os, time, json, pickle, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from concurrent.futures import ProcessPoolExecutor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
import methods
from methods import *
from cv import folds
from cv_all import map_fit_F
HP = json.load(open(f"{JD}/results/cv/meta_hparams.json"))
COND = {"Tao025": 0.25, "Tao035": 0.35, "Tao045": 0.45, "Mao1": 1.0, "Mao2": 2.0, "Mao3": 3.0}

def ppk(case, rows):
    out = {}
    for tag, Fs in (("s", SIMPL[case]), ("f", FULL)):
        out[tag] = ppk_chain(case, rows, Fs, *HP[case])
    b = np.array([out["s"].BIC_total, out["f"].BIC_total]); w = np.exp(-(b - b.min()) / 2); w /= w.sum()
    def pred(rr):
        _, a = predict_rows(out["s"], rr); _, c = predict_rows(out["f"], rr); return w[0] * a + w[1] * c
    return pred

def gp_correct(pred_fn, case, train_rows, test_rows):
    Rtr = P.loc[sorted(train_rows)]; Rte = P.loc[sorted(test_rows)]
    tmax = P[P.dataset.isin(CASES[case])].t.max()
    res = Rtr[SPECIES].to_numpy(float) - pred_fn(train_rows)
    def X(R): return np.c_[R.t.to_numpy() / tmax, R.dataset.map(COND).fillna(0).to_numpy()]
    Xtr, Xte = X(Rtr), X(Rte)
    X0 = np.c_[np.zeros(len(CASES[case])), [COND.get(d, 0) for d in CASES[case]]]
    corr = np.zeros((len(Rte), 4))
    for s in range(4):
        m = ~np.isnan(res[:, s])
        if m.sum() < 4: continue
        Xa = np.vstack([Xtr[m], X0]); ya = np.r_[res[m, s], np.zeros(len(X0))]
        sc = max(ya.std(), 1e-3)
        k = C(1.0, (1e-3, 1e2)) * Matern([0.2, 1.0], (1e-2, 1e2), nu=2.5) + WhiteKernel(0.5, (1e-4, 1e1))
        g = GaussianProcessRegressor(k, normalize_y=False, n_restarts_optimizer=2, random_state=0).fit(Xa, ya / sc)
        corr[:, s] = g.predict(Xte) * sc
    return Rte, pred_fn(test_rows), corr

def run_fold(args):
    case, proto, test, fid = args
    cache = f"{JD}/results/cv/cache_hyb_{proto}/{case}_{fid}.pkl"
    if os.path.exists(cache): return pickle.load(open(cache, "rb"))
    rows = all_rows(case) - set(test)
    fns = {"PPK-FIP": ppk(case, rows)}
    o = two_step(case, rows); fns["ORIG"] = lambda rr, o=o: predict_rows(o, rr)[1]
    recs = []
    for name, fn in fns.items():
        Rte, base, corr = gp_correct(fn, case, rows, test)
        obs = Rte[SPECIES].to_numpy(float)
        for j in range(len(Rte)):
            for s, sp in enumerate(SPECIES):
                if not np.isnan(obs[j, s]):
                    for lab, pr in ((name, base[j, s]), (name + "+GP", base[j, s] + corr[j, s])):
                        recs.append(dict(case=case, proto=proto, fold=fid, procedure=lab, row=int(Rte.index[j]), species=sp,
                                         err=np.clip(pr, 0, 100) - obs[j, s] if "+GP" in lab else pr - obs[j, s]))
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
    df = pd.DataFrame(recs); df.to_csv(f"{JD}/results/cv/{proto}_hybrid.csv.gz", index=False)
    df["se"] = df.err ** 2
    print(df.groupby(["case", "procedure"]).se.mean().pow(.5).unstack().round(3).to_string())
