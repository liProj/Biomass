"""Monte-Carlo simulation: does the proposed procedure recover known truths better than the original two-step
method, and does it avoid over-fitting when the truth has NO processivity / deactivation?
Truths are the fits to the real data (sampling design and noise level of the real case):
  S1 plain   : Akerman70 design, truth = original simplified model (no processivity, no deactivation)
  S2 proc    : Bornadel design, truth = joint + pseudo-processivity
  S3 deact   : Akerman100 design, truth = joint + first-order deactivation
Noise: iid Gaussian with the residual SD of the truth fit, clipped at 0; c recomputed as M+2D+3T."""
import sys, os, json, time, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
import fitlib, methods
from methods import *
from models import predict

SCEN = {"S1 plain (Akerman70 design)": ("Akerman70", "orig"), "S2 pseudo-processive (Bornadel design)": ("Bornadel", "proc"),
        "S3 deactivation (Akerman100 design)": ("Akerman100", "kd")}
HP = [(1.0, 0.05), (1.0, 0.2), (3.0, 0.2)]

def truth(case, kind):
    rows = all_rows(case)
    if kind == "orig": m = two_step(case, rows, n_starts=20)
    elif kind == "proc": m = joint(case, rows, proc=True, n_starts=20)
    else: m = joint(case, rows, kd=True, n_starts=20)
    sig = np.sqrt(m.F / m.nT)
    return m, sig

def curve(m, d, tg):
    return m.pred(m.theta, d, tg)

def run_rep(args):
    scen, rep = args
    case, kind = SCEN[scen]
    orig = fitlib.P.copy()
    mt, sig = truth(case, kind)                 # deterministic (cached per process would be nicer; cheap enough)
    rng = np.random.default_rng(1000 + rep)
    d = CASES[case][0]
    idx = orig.index[(orig.dataset == d) & (orig.t > 0)]
    tt = orig.loc[idx, "t"].to_numpy()
    Y = curve(mt, d, tt) + rng.normal(0, sig, (len(tt), 4))
    Y = np.clip(Y, 0, None)
    for s, sp in enumerate(SPECIES): fitlib.P.loc[idx, sp] = Y[:, s]
    fitlib.P.loc[idx, "c"] = Y[:, 1] + 2 * Y[:, 2] + 3 * Y[:, 3]
    try:
        rows = all_rows(case)
        C = {"ORIG": two_step(case, rows), "ORIG-full": two_step(case, rows, full=True)}
        J = {}
        for tag, Fs in (("s", SIMPL[case]), ("f", FULL)):
            j = joint(case, rows, Fset=Fs, n_starts=5); jk = joint(case, rows, Fset=Fs, kd=True, init=[j], n_starts=5)
            jp = joint(case, rows, Fset=Fs, proc=True, init=[j], n_starts=5); jpk = joint(case, rows, Fset=Fs, proc=True, kd=True, init=[jk, jp], n_starts=5)
            J.update({f"J{tag}": j, f"J{tag}+kd": jk, f"J{tag}+proc": jp, f"J{tag}+proc+kd": jpk})
            for lam, sp in HP:
                from cv_all import map_fit_F
                C[f"FIP{tag}({lam:g},{sp:g})"] = map_fit_F(case, rows, Fs, lam, sp, init=[jpk, jk, jp])
        tg = np.linspace(tt.max() / 100, tt.max(), 100)
        true_c = curve(mt, d, tg)
        out = []
        def rec(name, pred, sm=np.nan, sd=np.nan, struct=""):
            out.append(dict(scenario=scen, rep=rep, method=name, curve_rmse=float(np.sqrt(np.mean((pred - true_c) ** 2))),
                            SM_err=sm, SD_err=sd, structure=struct))
        tSM = mt.params_for(mt.theta, d)["SM"]; tSD = mt.params_for(mt.theta, d)["SD"]
        for name in ("ORIG", "ORIG-full"):
            p = C[name].params_for(C[name].theta, d)
            rec(name, curve(C[name], d, tg), np.log(p["SM"] / tSM), np.log(p["SD"] / tSD))
        jb = min(J, key=lambda k: J[k].BIC_total); p = J[jb].params_for(J[jb].theta, d)
        rec("JOINT-BIC", curve(J[jb], d, tg), np.log(p["SM"] / tSM), np.log(p["SD"] / tSD), jb)
        for lam, sp in HP:
            fs, ff = C[f"FIPs({lam:g},{sp:g})"], C[f"FIPf({lam:g},{sp:g})"]
            w = np.exp(-(np.r_[fs.BIC_total, ff.BIC_total] - min(fs.BIC_total, ff.BIC_total)) / 2); w /= w.sum()
            pr = w[0] * curve(fs, d, tg) + w[1] * curve(ff, d, tg)
            ps, pf = fs.params_for(fs.theta, d), ff.params_for(ff.theta, d)
            sm = w[0] * np.log(ps["SM"]) + w[1] * np.log(pf["SM"]); sdd = w[0] * np.log(ps["SD"]) + w[1] * np.log(pf["SD"])
            rec(f"FIP-BMA(lam={lam:g},sp={sp:g})", pr, sm - np.log(tSM), sdd - np.log(tSD),
                f"pM={w[0]*ps['pM']+w[1]*pf['pM']:.3f};pD={w[0]*ps['pD']+w[1]*pf['pD']:.3f};kd={w[0]*ps['kd']+w[1]*pf['kd']:.4f}")
        return out
    finally:
        fitlib.P.loc[:, :] = orig.values

if __name__ == "__main__":
    nrep = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    jobs = [(s, r) for s in SCEN for r in range(nrep)]
    t0 = time.time(); recs = []
    with ProcessPoolExecutor(int(os.environ.get("NPROC", 14))) as ex:
        for i, r in enumerate(ex.map(run_rep, jobs, chunksize=1)):
            recs += r
            if i % 20 == 0: print(f"  {i+1}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(recs); os.makedirs(f"{JD}/results/sim", exist_ok=True)
    df.to_csv(f"{JD}/results/sim/sim.csv", index=False)
    print(df.groupby(["scenario", "method"])[["curve_rmse", "SM_err", "SD_err"]].agg(["mean", "std"]).round(3).to_string())
