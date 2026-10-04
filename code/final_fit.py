"""Full-data fits of every procedure (paper protocol) + dense prediction grids for the figures.
PPK-FIP = BIC-weighted average of FIP(simplified F set) and FIP(full F set) at the meta-selected (lam, s_p);
its in-sample 'Fobj' is the SSE of the averaged prediction and its parameter count is the BIC-weighted mean k."""
import json, os, sys, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from methods import *
from fitlib import bic
from cv_all import map_fit_F
OUT = f"{JD}/results/final"; os.makedirs(OUT, exist_ok=True)
HP = json.load(open(f"{JD}/results/cv/meta_hparams.json"))

def dense(m, case):
    out = {}
    for d in CASES[case]:
        tm = P[P.dataset == d].t.max(); tg = np.linspace(0, tm * 1.02, 300)
        out[d] = (tg, m.pred(m.theta, d, np.maximum(tg, 1e-9)))
    return out

def run(case):
    rows = all_rows(case); ns = 30
    C = {"ORIG": two_step(case, rows, n_starts=ns), "ORIG-full": two_step(case, rows, full=True, n_starts=ns)}
    for tag, Fs in (("s", SIMPL[case]), ("f", FULL)):
        j = joint(case, rows, Fset=Fs, n_starts=ns, init=[C["ORIG"] if tag == "s" else C["ORIG-full"]])
        jk = joint(case, rows, Fset=Fs, kd=True, init=[j], n_starts=ns); jp = joint(case, rows, Fset=Fs, proc=True, init=[j], n_starts=ns)
        jpk = joint(case, rows, Fset=Fs, proc=True, kd=True, init=[jk, jp], n_starts=ns)
        C.update({f"J{tag}": j, f"J{tag}+kd": jk, f"J{tag}+proc": jp, f"J{tag}+proc+kd": jpk})
        lam, sp = HP[case]
        C[f"FIP{tag}"] = map_fit_F(case, rows, Fs, lam, sp, base=jpk)  # same call as in CV
    b = np.array([C["FIPs"].BIC_total, C["FIPf"].BIC_total]); w = np.exp(-(b - b.min()) / 2); w /= w.sum()
    recs, grids, params = [], {}, {}
    R = P.loc[sorted(rows)]
    for name, m in C.items():
        _, pr = predict_rows(m, rows); o = R[SPECIES].to_numpy(float); e = (pr - o)[~np.isnan(o)]
        recs.append(dict(case=case, method=name, Fobj=float(e @ e), nT=len(e), k=m.k_total, BIC_total=bic(e @ e, len(e), m.k_total),
                         BIC_paper=bic(e @ e, len(e), m.k) if name.startswith("ORIG") else bic(e @ e, len(e), m.k_total)))
        params[name] = {d: m.params_for(m.theta, d) for d in CASES[case]}
        grids[name] = dense(m, case)
    # PPK-FIP (model average)
    _, ps = predict_rows(C["FIPs"], rows); _, pf = predict_rows(C["FIPf"], rows); pr = w[0] * ps + w[1] * pf
    e = (pr - o)[~np.isnan(o)]; kbar = w[0] * C["FIPs"].k_total + w[1] * C["FIPf"].k_total
    recs.append(dict(case=case, method="PPK-FIP", Fobj=float(e @ e), nT=len(e), k=kbar, BIC_total=bic(e @ e, len(e), kbar),
                     BIC_paper=bic(e @ e, len(e), kbar), w_simpl=w[0]))
    grids["PPK-FIP"] = {d: (grids["FIPs"][d][0], w[0] * grids["FIPs"][d][1] + w[1] * grids["FIPf"][d][1]) for d in CASES[case]}
    params["PPK-FIP"] = {"weights": w.tolist(), "hp": HP[case]}
    return case, recs, {k: {d: (g[0].tolist(), g[1].tolist()) for d, g in v.items()} for k, v in grids.items()}, params

if __name__ == "__main__":
    allrecs, G, PAR = [], {}, {}
    with ProcessPoolExecutor(6) as ex:
        for case, recs, grids, params in ex.map(run, CASES):
            allrecs += recs; G[case] = grids; PAR[case] = params
    df = pd.DataFrame(allrecs); df.to_csv(f"{OUT}/insample.csv", index=False)
    json.dump(G, open(f"{OUT}/grids.json", "w")); json.dump(PAR, open(f"{OUT}/params.json", "w"), default=float)
    print(df.round(2).to_string(index=False))
