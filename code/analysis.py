"""Summaries + significance tests for the out-of-sample protocols.
PPK-FIP == procedure 'FIP-BMA-meta' of select_eval.py.  Paired Wilcoxon signed-rank tests on per-observation
squared errors (and on per-fold MSE), Holm correction across the six cases for each comparison."""
import numpy as np, pandas as pd, json, os
from scipy.stats import wilcoxon
from kinetics import JD
OUT = f"{JD}/results/summary"; os.makedirs(OUT, exist_ok=True)
REN = {"FIP-BMA-meta": "PPK-FIP"}
CASES = ["Akerman70", "Akerman100", "Bornadel", "Tao025", "Tao035_045", "Mao123"]

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for i, j in enumerate(o):
        run = max(run, (m - i) * p[j]); adj[j] = min(1.0, run)
    return adj

def load_proc(proto):
    e = pd.read_csv(f"{JD}/results/cv/{proto}_procedures.csv.gz"); e["procedure"] = e.procedure.replace(REN)
    e["se"] = e.err ** 2
    bbf = [f for f in os.listdir(f"{JD}/results/cv") if f.startswith(f"{proto}_bb_")]
    for f in bbf:
        b = pd.read_csv(f"{JD}/results/cv/{f}")
        b = b.rename(columns={"method": "procedure"}); b["se"] = (b.pred - b.obs) ** 2
        b["fold"] = b.row if proto == "loto" else 0
        e = pd.concat([e, b[["case", "fold", "procedure", "row", "species", "se"]]], ignore_index=True)
    return e

def summarise(proto, ref_list=("ORIG", "ORIG-full")):
    e = load_proc(proto)
    rm = e.groupby(["case", "procedure"]).se.mean().pow(0.5).unstack()
    allr = e.groupby("procedure").se.mean().pow(0.5)
    rows = []
    for ref in ref_list:
        ps = []
        for c in CASES:
            a = e[(e.case == c) & (e.procedure == "PPK-FIP")].set_index(["fold", "row", "species"]).se
            b = e[(e.case == c) & (e.procedure == ref)].set_index(["fold", "row", "species"]).se
            j = pd.concat([a, b], axis=1, keys=["new", "ref"]).dropna()
            p = wilcoxon(j.new, j.ref).pvalue if len(j) > 5 and (j.new != j.ref).any() else np.nan
            fa = j.groupby(level="fold").mean()
            win = float((fa.new < fa.ref).mean())
            rows.append(dict(proto=proto, ref=ref, case=c, rmse_new=np.sqrt(j.new.mean()), rmse_ref=np.sqrt(j.ref.mean()),
                             ratio=np.sqrt(j.new.mean() / j.ref.mean()), n_obs=len(j), fold_win_frac=win, p=p,
                             obs_win_frac=float((j.new < j.ref).mean()), median_diff=float((j.new - j.ref).median())))
            ps.append(p)
        a = e[e.procedure == "PPK-FIP"].set_index(["case", "fold", "row", "species"]).se
        b = e[e.procedure == ref].set_index(["case", "fold", "row", "species"]).se
        j = pd.concat([a, b], axis=1, keys=["new", "ref"]).dropna()
        rows.append(dict(proto=proto, ref=ref, case="ALL", rmse_new=np.sqrt(j.new.mean()), rmse_ref=np.sqrt(j.ref.mean()),
                         ratio=np.sqrt(j.new.mean() / j.ref.mean()), n_obs=len(j), p=wilcoxon(j.new, j.ref).pvalue,
                         fold_win_frac=np.nan, obs_win_frac=float((j.new < j.ref).mean()), median_diff=float((j.new - j.ref).median())))
    T = pd.DataFrame(rows)
    for ref in ref_list:
        m = (T.ref == ref) & (T.case != "ALL"); T.loc[m, "p_holm"] = holm(T.loc[m, "p"].fillna(1).values)
    T.to_csv(f"{OUT}/{proto}_tests.csv", index=False); rm.to_csv(f"{OUT}/{proto}_rmse.csv"); allr.to_csv(f"{OUT}/{proto}_rmse_all.csv")
    return T, rm

if __name__ == "__main__":
    import sys
    for proto in sys.argv[1:]:
        T, rm = summarise(proto)
        print(f"== {proto}\n", rm.round(3).to_string(), "\n", T.round(4).to_string(index=False))
