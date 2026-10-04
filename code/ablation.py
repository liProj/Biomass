"""Component ablation of PPK-FIP on the LOTO (and fcst) folds.  Every variant is a procedure computed from the raw
candidate predictions (cv_all.py) or from the dedicated ablation run (cv_ablate.py); RMSE pooled over all
observations and per case; paired Wilcoxon vs the full PPK-FIP."""
import json, os, numpy as np, pandas as pd
from scipy.stats import wilcoxon
from kinetics import JD
from select_eval import bic_weights
R = f"{JD}/results"; HP = json.load(open(f"{R}/cv/meta_hparams.json"))
CASES = ["Akerman70", "Akerman100", "Bornadel", "Tao025", "Tao035_045", "Mao123"]

def build(proto):
    raw = pd.read_csv(f"{R}/cv/{proto}_all.csv.gz")
    out = []
    for (case, fold), d in raw.groupby(["case", "fold"]):
        g = {m: x.set_index(["row", "species"]) for m, x in d.groupby("method")}
        tb = {m: x.train_BIC.iloc[0] for m, x in g.items()}; obs = g["orig-simpl"].obs
        lam, sp = HP[case]
        def fipbma(l, s_):
            a, b = f"FIPs(lam={l:g},sp={s_:g})", f"FIPf(lam={l:g},sp={s_:g})"; w = bic_weights([tb[a], tb[b]])
            return w[0] * g[a].pred + w[1] * g[b].pred
        V = {"PPK-FIP (full)": fipbma(lam, sp),
             "simplified F only (no averaging)": g[f"FIPs(lam={lam:g},sp={sp:g})"].pred,
             "full F only (no averaging)": g[f"FIPf(lam={lam:g},sp={sp:g})"].pred,
             "no fingerprint prior (joint, BMA)": None, "no prior, BIC selection": None,
             "two-step, simplified (ORIG)": g["orig-simpl"].pred, "two-step, full (ORIG-full)": g["orig-full"].pred}
        J = [m for m in g if m.startswith("J")]; w = bic_weights([tb[m] for m in J])
        V["no fingerprint prior (joint, BMA)"] = sum(wi * g[m].pred for wi, m in zip(w, J))
        V["no prior, BIC selection"] = g[min(J, key=lambda m: tb[m])].pred
        for l in (1.0, 3.0):
            for s_ in (0.05, 0.2): V[f"fixed $\\lambda$={l:g}, $s_p$={s_:g} (no meta-selection)"] = fipbma(l, s_)
        for k, v in V.items():
            e = (v - obs).rename("err").reset_index(); e["case"] = case; e["fold"] = fold; e["variant"] = k; out.append(e)
    E = pd.concat(out, ignore_index=True)
    if os.path.exists(f"{R}/cv/{proto}_ablate.csv.gz"):
        A = pd.read_csv(f"{R}/cv/{proto}_ablate.csv.gz")
        for (case, fold), d in A.groupby(["case", "fold"]):
            g = {m: x.set_index(["row", "species"]) for m, x in d.groupby("method")}
            for vn, lab in (("full", "PPK-FIP (re-run, consistency check)"), ("no-proc", "no pseudo-processivity"), ("no-kd", "no deactivation"),
                            ("no-proc-no-kd", "no processivity, no deactivation")):
                if f"FIPs[{vn}]" not in g: continue
                a, b = g[f"FIPs[{vn}]"], g[f"FIPf[{vn}]"]; w = bic_weights([a.train_BIC.iloc[0], b.train_BIC.iloc[0]])
                e = ((w[0] * a.pred + w[1] * b.pred) - a.obs).rename("err").reset_index()
                e["case"] = case; e["fold"] = fold; e["variant"] = lab; E = pd.concat([E, e], ignore_index=True)
    E["se"] = E.err ** 2
    return E

def summarise(E):
    ref = E[E.variant == "PPK-FIP (full)"].set_index(["case", "fold", "row", "species"]).se
    rows = []
    for v, d in E.groupby("variant", sort=False):
        x = d.set_index(["case", "fold", "row", "species"]).se
        j = pd.concat([x, ref], axis=1, keys=["v", "r"]).dropna()
        p = wilcoxon(j.v, j.r).pvalue if (j.v != j.r).any() else np.nan
        r = dict(variant=v, ALL=np.sqrt(x.mean()), p_vs_full=p, n=len(x))
        for c in CASES: r[c] = np.sqrt(d[d.case == c].se.mean())
        rows.append(r)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    for proto in ("loto", "fcst"):
        E = build(proto); S = summarise(E); S.to_csv(f"{R}/summary/{proto}_ablation.csv", index=False)
        print(proto); print(S.round(3).to_string(index=False))
