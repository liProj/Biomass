"""In-sample (paper protocol) comparison on the full data of each case, with many multistarts.
Reports the paper's own metrics (Fobj, BIC with the paper's convention k = dynamic parameters only) and the fair
BIC_total (k counts SM, SD estimated from the same data in step 1), AICc and per-species RMSE."""
import json, numpy as np, pandas as pd, os
from concurrent.futures import ProcessPoolExecutor
from methods import *
from fitlib import bic

PAPER = {  # (Fobj, BIC) as printed: simplified / full
    "Akerman70": ((70.9, 35.1), (59.0, 35.7)), "Akerman100": ((118.0, 50.3), (117.9, 56.9)), "Bornadel": ((329.9, 140.5), (329.9, 145.3)),
    "Tao025": ((123.5, 54.5), (121.2, 61.7)), "Tao035_045": ((326.7, 132.9), (325.8, 142.4)), "Mao123": ((1771.5, 381.6), (1475.9, 365.7))}

def species_rmse(m):
    R, pr = predict_rows(m, all_rows_m(m))
    o = R[SPECIES].to_numpy(float)
    return {sp: float(np.sqrt(np.nanmean((pr[:, s] - o[:, s]) ** 2))) for s, sp in enumerate(SPECIES)}
def all_rows_m(m):
    return set().union(*[set(b.index) for b in m.blocks.values()])

def run(case):
    rows = all_rows(case)
    ms = {"orig-simpl (refit)": two_step(case, rows, n_starts=30), "orig-full (refit)": two_step(case, rows, full=True, n_starts=30)}
    ms.update(family(case, rows, n_starts=30))
    fam = ["joint", "joint+kd", "joint+proc", "joint+proc+kd"]
    best = min(fam, key=lambda k: ms[k].BIC_total)
    recs = []
    for name, m in ms.items():
        k_dyn = m.k; k_tot = m.k_total
        aicc = m.nT * np.log(m.F / m.nT) + 2 * k_tot + 2 * k_tot * (k_tot + 1) / max(m.nT - k_tot - 1, 1)
        recs.append(dict(case=case, method=name, Fobj=m.F, nT=m.nT, k_paper=k_dyn if "orig" in name else k_tot, k_total=k_tot,
                         BIC_paper=bic(m.F, m.nT, k_dyn if "orig" in name else k_tot), BIC_total=m.BIC_total, AICc=aicc,
                         RMSE=np.sqrt(m.F / m.nT), selected=(name == best), params=json.dumps(dict(zip(m.free, map(float, m.theta)))),
                         **{f"RMSE_{k}": v for k, v in species_rmse(m).items()}))
    for lab, (F, B) in zip(("paper simplified (printed)", "paper full (printed)"), PAPER[case]):
        nT = ms["joint"].nT; kp = round((B - nT * np.log(F / nT)) / np.log(nT))
        recs.append(dict(case=case, method=lab, Fobj=F, nT=nT, k_paper=kp, k_total=kp + 2, BIC_paper=B,
                         BIC_total=bic(F, nT, kp + 2), RMSE=np.sqrt(F / nT)))
    return recs

if __name__ == "__main__":
    out = f"{JD}/results/insample"; os.makedirs(out, exist_ok=True)
    with ProcessPoolExecutor(6) as ex:
        recs = sum(ex.map(run, CASES), [])
    df = pd.DataFrame(recs); df.to_csv(f"{out}/insample.csv", index=False)
    print(df[["case", "method", "Fobj", "nT", "k_total", "BIC_paper", "BIC_total", "AICc", "selected"]].round(2).to_string(index=False))
