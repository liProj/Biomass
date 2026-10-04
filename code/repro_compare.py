"""Item-by-item comparison of every number printed in Tables 1-6 / Section 3 of the paper with our reproduction.
Three columns per item: printed value | our re-fit (global multistart) | value at the printed parameter point.
kA is compared through the inferred enzyme loading E = kE/kA (must be a constant per source)."""
import json, numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from kinetics import *
from repro import dyn_resid_factory, FP_GROUPS
B = f"{JD}/results/baseline"
fp = json.load(open(f"{B}/fingerprint.json")); dyn = json.load(open(f"{B}/dynamic.json"))
rows = []
FPP = {"Akerman": (0.389, 0.026, 0.0311, 0.0035, 74.2), "Bornadel": (0.909, 0.096, 0.066, 0.011, 840.6),
       "Tao": (0.761, 0.025, 0.333, 0.012, 535.3), "Mao": (0.916, 0.094, 0.378, 0.037, 1616.0)}
for g, (sm, smc, sd, sdc, ss) in FPP.items():
    v = fp[g]; z = 1.959964 / __import__("scipy").stats.t.ppf(0.975, v["nT"] - 2)
    for item, p, o in (("SM", sm, v["SM"]), ("SM 95% CI", smc, v["SM_ci"] * z), ("SD", sd, v["SD"]), ("SD 95% CI", sdc, v["SD_ci"] * z), ("SS", ss, v["SS"])):
        rows.append(dict(table="Sec. 3 (fingerprint)", case=g, model="fingerprint", item=item, paper=p, refit=o, at_paper_point=np.nan))
# printed dynamic tables: case -> variant -> {param: value}, Fobj, BIC
PT = {
 ("Akerman70", "full"): ({"kA": 0.0866, "FA": 0.000077, "FAZ": 0.00223, "FAM": 0.00495, "FAD": 0.00205}, 59.0, 35.7, "Table 1"),
 ("Akerman100", "full"): ({"kA": 0.1336, "FA": 0.0000252, "FAZ": 1.988e-9, "FAM": 0.0022687, "FAD": 0.0013023}, 117.9, 56.9, "Table 1"),
 ("Akerman70", "simplified"): ({"kA": 0.0530, "FAM": 0.00223, "FAD": 0.00100}, 70.9, 35.1, "Table 2"),
 ("Akerman100", "simplified"): ({"kA": 0.133, "FAM": 0.00225, "FAD": 0.00130}, 118.0, 50.3, "Table 2"),
 ("Bornadel", "full"): ({"kA": 0.199, "FA": 7.164e-9, "FAZ": 0.0395, "FAM": 0.0125, "FAD": 0.00933}, 329.9, 145.3, "Table 3"),
 ("Bornadel", "simplified"): ({"kA": 0.199, "FAZ": 0.0395, "FAM": 0.0125, "FAD": 0.00933}, 329.9, 140.5, "Table 3"),
 ("Tao025", "full"): ({"kA": 0.848, "FA": 0.0259, "FAZ": 0.00888, "FAM": 0.0130, "FAD": 7.52e-10}, 121.2, 61.7, "Table 4"),
 ("Tao035_045", "full"): ({"kA": 0.639, "FA": 0.0288, "FAZ": 0.00737, "FAM": 0.0135, "FAD": 6.86e-9}, 325.8, 142.4, "Table 4"),
 ("Tao025", "simplified"): ({"kA": 0.776, "FAZ": 0.00813, "FAM": 0.0117}, 123.5, 54.5, "Table 5"),
 ("Tao035_045", "simplified"): ({"kA": 0.582, "FAZ": 0.00669, "FAM": 0.0122}, 326.7, 132.9, "Table 5"),
 ("Mao123", "full"): ({"kA@Mao1": 0.0109, "kA@Mao2": 0.00810, "kA@Mao3": 0.00693, "FA": 0.345, "FAZ": 0.0156, "FAM": 0.0154, "FAD": 2.62e-10}, 1475.9, 365.7, "Table 6"),
 ("Mao123", "simplified"): ({"kA@Mao1": 0.00545, "kA@Mao2": 0.00407, "kA@Mao3": 0.00349, "FAZ": 0.00797, "FAM": 0.00702}, 1771.5, 381.6, "Table 6")}
DS = {"Akerman70": ["Akerman70"], "Akerman100": ["Akerman100"], "Bornadel": ["Bornadel"], "Tao025": ["Tao025"],
      "Tao035_045": ["Tao035", "Tao045"], "Mao123": ["Mao1", "Mao2", "Mao3"]}
GRP = {"Akerman70": "Akerman", "Akerman100": "Akerman", "Bornadel": "Bornadel", "Tao025": "Tao", "Tao035_045": "Tao", "Mao123": "Mao"}
for (case, var), (pp, F, Bc, tab) in PT.items():
    d = dyn[f"{case}|{var}"]
    free = [k for k in ["FA", "FAZ", "FAM", "FAD"] if k in pp]
    bk = case == "Mao123"
    f, nk = dyn_resid_factory(DS[case], fp[GRP[case]]["SM"], fp[GRP[case]]["SD"], free, bk)
    kAs = [pp[f"kA@{x}"] for x in DS[case]] if bk else [pp["kA"]]
    # value of Fobj at the printed point, enzyme loading E free (single scalar shared by the batches)
    def Fp(lE):
        th = np.r_[np.array(kAs) * np.exp(lE), [pp[k] for k in free]]
        r = f(th); return r @ r
    opt = minimize_scalar(Fp, bounds=(np.log(1e-3), np.log(1e3)), method="bounded")
    E = float(np.exp(opt.x))
    rows.append(dict(table=tab, case=case, model=var, item="Fobj", paper=F, refit=d["Fobj"], at_paper_point=opt.fun))
    rows.append(dict(table=tab, case=case, model=var, item="BIC", paper=Bc, refit=d["BIC"], at_paper_point=bic(opt.fun, d["nT"], d["k"])))
    rows.append(dict(table=tab, case=case, model=var, item="enzyme loading E = kE/kA implied", paper=np.nan,
                     refit=(d["params"]["kE"] / pp["kA"]) if not bk else d["params"]["kE_Mao1"] / pp["kA@Mao1"], at_paper_point=E))
    for k in free:
        rows.append(dict(table=tab, case=case, model=var, item=k, paper=pp[k], refit=d["params"][k], at_paper_point=pp[k]))
df = pd.DataFrame(rows)
def rel(a, b): return np.abs(a - b) / np.maximum(np.abs(a), 1e-12)
df["refit_rel_dev"] = rel(df.paper, df.refit)
df["status"] = np.where(df.refit_rel_dev <= 0.015, "reproduced", np.where((df.item == "Fobj") & (df.refit < df.paper - 0.05), "improved (paper not at optimum)", "differs"))
df.to_csv(f"{JD}/results/repro_compare.csv", index=False)
pd.set_option("display.width", 220)
print(df.round(5).to_string(index=False))
print(df.groupby("status").size())
