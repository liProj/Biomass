"""Turn the raw candidate predictions of cv_all.py into the competing *procedures* and score them.
Procedures (all decisions use training data or OTHER cases only):
  ORIG            original two-step method, paper-preferred simplified model
  ORIG-full       original two-step method, full model
  ORIG-BIC        original two-step, simplified/full chosen by training BIC (the paper's own criterion)
  JOINT-BIC       joint family J*, choose member by training BIC_total
  JOINT-BMA       BIC-weighted average of the J* members
  FIP-meta        FIP with (F set chosen by training BIC) and (lam, sp) chosen by leave-one-CASE-out meta-validation
  FIP-BMA-meta    BIC-weighted average of FIPs and FIPf at the meta-selected (lam, sp)   <- proposed (PPK-FIP)"""
import sys, numpy as np, pandas as pd
from kinetics import JD

def load(proto):
    return pd.read_csv(f"{JD}/results/cv/{proto}_all.csv.gz")

def bic_weights(b):
    b = np.asarray(b, float); w = np.exp(-(b - b.min()) / 2); return w / w.sum()

def procedures(df, hp=None):
    """df: rows of one fold (all candidates).  Returns dict procedure -> prediction Series indexed by (row,species)."""
    key = ["row", "species"]
    g = {m: d.set_index(key) for m, d in df.groupby("method")}
    out = {"ORIG": g["orig-simpl"].pred, "ORIG-full": g["orig-full"].pred}
    tb = {m: d.train_BIC.iloc[0] for m, d in g.items()}
    out["ORIG-BIC"] = out["ORIG"] if tb["orig-simpl"] <= tb["orig-full"] else out["ORIG-full"]
    J = [m for m in g if m.startswith("J")]
    out["JOINT-BIC"] = g[min(J, key=lambda m: tb[m])].pred
    w = bic_weights([tb[m] for m in J]); out["JOINT-BMA"] = sum(wi * g[m].pred for wi, m in zip(w, J))
    if hp is not None:
        lam, sp = hp
        fs, ff = f"FIPs(lam={lam:g},sp={sp:g})", f"FIPf(lam={lam:g},sp={sp:g})"
        out["FIP-meta"] = g[fs].pred if tb[fs] <= tb[ff] else g[ff].pred
        w = bic_weights([tb[fs], tb[ff]]); out["FIP-BMA-meta"] = w[0] * g[fs].pred + w[1] * g[ff].pred
    return out, g[ "orig-simpl"].obs

def score(df, hpmap=None):
    rows = []
    for (case, fold), d in df.groupby(["case", "fold"]):
        P_, obs = procedures(d, None if hpmap is None else hpmap[case])
        for name, pr in P_.items():
            e = (pr - obs).rename("err").reset_index()
            e["case"] = case; e["fold"] = fold; e["procedure"] = name
            rows.append(e)
    return pd.concat(rows, ignore_index=True)

def meta_select(df):
    """For each target case choose (lam, sp) minimising the mean over the OTHER cases of RMSE(FIP-BMA)/RMSE(ORIG)."""
    fips = sorted({m.split("(")[1].rstrip(")") for m in df.method.unique() if m.startswith("FIP")})
    grid = []
    for s in fips:
        kv = dict(x.split("=") for x in s.split(",")); grid.append((float(kv["lam"]), float(kv["sp"])))
    perf = {}
    for hp in grid:
        e = score(df, {c: hp for c in df.case.unique()})
        r = e.groupby(["case", "procedure"]).err.apply(lambda x: np.sqrt(np.mean(x ** 2))).unstack()
        perf[hp] = r["FIP-BMA-meta"] / r["ORIG"]
    perf = pd.DataFrame(perf)  # index case, columns hp
    choice = {}
    for c in perf.index:
        others = perf.drop(index=c).mean(0)
        choice[c] = others.idxmin()
    return choice, perf

if __name__ == "__main__":
    proto = sys.argv[1]
    df = load(proto)
    choice, perf = meta_select(df)
    if proto != "loto":   # single definition of PPK-FIP: hyper-parameters always from leave-one-case-out LOTO meta-validation
        import json as _j
        choice = {c: tuple(v) for c, v in _j.load(open(f"{JD}/results/cv/meta_hparams.json")).items()}
        print("using LOTO meta-selected hyper-parameters for", proto)
    print("meta-selected (lam, sp) per case (leave-one-case-out):", choice)
    import json
    json.dump({c: [float(a), float(b)] for c, (a, b) in choice.items()}, open(f"{JD}/results/cv/{proto}_meta_hparams.json", "w"))
    if proto == "loto":
        json.dump({c: [float(a), float(b)] for c, (a, b) in choice.items()}, open(f"{JD}/results/cv/meta_hparams.json", "w"))
    e = score(df, choice)
    e.to_csv(f"{JD}/results/cv/{proto}_procedures.csv.gz", index=False)
    r = e.groupby(["case", "procedure"]).err.apply(lambda x: np.sqrt(np.mean(x ** 2))).unstack()
    order = ["ORIG", "ORIG-full", "ORIG-BIC", "JOINT-BIC", "JOINT-BMA", "FIP-meta", "FIP-BMA-meta"]
    print(r[order].round(3).to_string())
    print("\nratio to ORIG:\n", (r[order].div(r["ORIG"], axis=0)).round(3).to_string())
    perf.to_csv(f"{JD}/results/cv/{proto}_meta_grid.csv")
