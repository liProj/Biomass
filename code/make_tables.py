"""Generate paper/numbers.tex (macros) and paper/tables/*.tex from result files.  Missing results render as \\textbf{??}."""
import os, json, numpy as np, pandas as pd
from kinetics import JD, load_profiles
PAP = f"{JD}/paper"; TAB = f"{PAP}/tables"; os.makedirs(TAB, exist_ok=True)
R = f"{JD}/results"; M = {}
def ex(p): return os.path.exists(p)
def mac(name, val): M[name] = val
def f(x, d=2): return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"
def pfmt(p): return "--" if p is None or np.isnan(p) else ("$<$0.001" if p < 0.001 else f"{p:.3f}")
CL = {"Akerman70": "Åkerman 70\\,°C", "Akerman100": "Åkerman 100\\,°C", "Bornadel": "Bornadel", "Tao025": "Tao $a_w$=0.25",
      "Tao035_045": "Tao $a_w$=0.35+0.45", "Mao123": "Mao batches 1--3", "ALL": "All cases (pooled)"}
PAPERI = {"Akerman70": ((70.9, 35.1), (59.0, 35.7)), "Akerman100": ((118.0, 50.3), (117.9, 56.9)), "Bornadel": ((329.9, 140.5), (329.9, 145.3)),
          "Tao025": ((123.5, 54.5), (121.2, 61.7)), "Tao035_045": ((326.7, 132.9), (325.8, 142.4)), "Mao123": ((1771.5, 381.6), (1475.9, 365.7))}
CASES = ["Akerman70", "Akerman100", "Bornadel", "Tao025", "Tao035_045", "Mao123"]
DS_ = {"Akerman70": ["Akerman70"], "Akerman100": ["Akerman100"], "Bornadel": ["Bornadel"], "Tao025": ["Tao025"], "Tao035_045": ["Tao035", "Tao045"], "Mao123": ["Mao1", "Mao2", "Mao3"]}
def write(name, s): open(f"{TAB}/{name}.tex", "w").write(s)
def tabular(cols, rows, align=None, caption="", label="", size="\\footnotesize", notes=""):
    align = align or "l" + "r" * (len(cols) - 1)
    s = f"\\begin{{table}}[t]\n\\centering\n\\caption{{{caption}}}\\label{{{label}}}\n{size}\n\\begin{{adjustbox}}{{max width=\\linewidth}}\n\\begin{{tabular}}{{{align}}}\n\\toprule\n"
    s += " & ".join(cols) + " \\\\\n\\midrule\n"
    for r in rows: s += ("\\midrule\n" if r == "MID" else " & ".join(str(x) for x in r) + " \\\\\n")
    s += "\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n"
    if notes: s += f"\\par\\smallskip\\parbox{{\\linewidth}}{{\\scriptsize {notes}}}\n"
    return s + "\\end{table}\n"

# ---------------------------------------------------------------- data
P = load_profiles()
mac("NRows", len(P)); fit = P[P.dataset.isin(["Akerman70", "Akerman100", "Bornadel", "Tao025", "Tao035", "Tao045", "Mao1", "Mao2", "Mao3"]) & (P.t > 0)]
mac("NFitProfiles", 9); mac("NFitPoints", int(fit[["Z", "M", "D", "T"]].notna().sum().sum()))
rows = []
SRC = {"Akerman70": ("S1", "oleic, CALB, 70\\,°C, 3:1"), "Akerman100": ("S2", "oleic, CALB, 100\\,°C, 3:1"), "Bornadel": ("S3", "oleic, CALB, 3:1"),
       "Tao025": ("S4", "caprylic, \\emph{Candida} sp., $a_w$ 0.25, 3.2:1"), "Tao035": ("S5", "caprylic, $a_w$ 0.35"), "Tao045": ("S6", "caprylic, $a_w$ 0.45"),
       "Mao1": ("S17", "pelargonic, \\emph{Candida} sp., batch 1, 10:3.3"), "Mao2": ("S18", "batch 2"), "Mao3": ("S19", "batch 3"),
       "Mao4": ("S20", "batch 4 (excluded)"), "Mao5": ("S21", "batch 5 (excluded)")}
for d, g in P.groupby("dataset", sort=False):
    rows.append([d.replace("Akerman", "Åkerman "), SRC[d][0], SRC[d][1], len(g) - 1, f"{g.t.max():g}", "Z,M,D,T" if g.Z.notna().any() else "M,D,T (+c)"])
write("data", tabular(["Profile", "Table", "System", "$n_t$", "$t_\\mathrm{max}$ (h)", "Species"], rows, "lllrrl",
      "Reaction profiles used (Supplementary Material of \\cite{nishimura2026}); $n_t$ = sampling times after $t=0$.", "tab:data"))
dc = pd.read_csv(f"{R}/data_check.csv"); hard = dc[dc.match != "INFO"]
mac("NCheckPass", int((hard.match == "YES").sum())); mac("NCheckAll", len(hard))
rows = [[r["item"].replace("_", "\\_").replace("&", "\\&").replace("|", "/"), r["paper"], r["public_data"], r["match"]] for _, r in dc[dc.item.str.contains("Table|Fingerprinting|sources|datasets|batches")].iterrows()]
write("datacheck", tabular(["Check", "Paper", "Public data", "Match"], rows, "p{9.2cm}rrl",
      "Hard data check: every quantity printed in the paper that depends on the sample sizes or on individual values.", "tab:datacheck", "\\scriptsize"))
# ---------------------------------------------------------------- reproduction
rc = pd.read_csv(f"{R}/repro_compare.csv")
fp = rc[rc.model == "fingerprint"]
rows = []
for g in ["Akerman", "Bornadel", "Tao", "Mao"]:
    x = fp[fp.case == g].set_index("item")
    rows.append([g.replace("Akerman", "Åkerman"), f"{x.loc['SM','paper']:.3f} / {x.loc['SM','refit']:.3f}", f"{x.loc['SM 95% CI','paper']:.3f} / {x.loc['SM 95% CI','refit']:.3f}",
                 f"{x.loc['SD','paper']:.4f} / {x.loc['SD','refit']:.4f}", f"{x.loc['SD 95% CI','paper']:.4f} / {x.loc['SD 95% CI','refit']:.4f}", f"{x.loc['SS','paper']:.1f} / {x.loc['SS','refit']:.1f}"])
write("repro_fp", tabular(["System", "$S_M$", "$\\pm$95\\% CI", "$S_D$", "$\\pm$95\\% CI", "SS"], rows, "lrrrrr",
      "Step 1 (fingerprinting) reproduced: printed / reproduced.", "tab:reprofp"))
rows = []
for (case, var), g in rc[rc.model != "fingerprint"].groupby(["case", "model"], sort=False):
    x = g.set_index("item")
    consistent = abs(x.loc["Fobj", "at_paper_point"] - x.loc["Fobj", "paper"]) / x.loc["Fobj", "paper"] < 0.015
    rows.append([CL[case], var, f"{x.loc['Fobj','paper']:.1f}", f"{x.loc['Fobj','at_paper_point']:.1f}", f"{x.loc['enzyme loading E = kE/kA implied','at_paper_point']:.2f}",
                 f"{x.loc['Fobj','refit']:.1f}", f"{x.loc['BIC','paper']:.1f}", f"{x.loc['BIC','refit']:.1f}", "yes" if consistent else "no"])
write("repro_dyn", tabular(["Case", "Model", "$F_\\mathrm{obj}$ printed", "at printed $\\theta$", "$E$", "$F_\\mathrm{obj}$ global", "BIC printed", "BIC global", "consistent"],
      rows, "llrrrrrrl", "Step 2 (dynamic model) audit. ``At printed $\\theta$'': objective evaluated at the printed parameters with the enzyme loading $E=k_E/k_A$ optimised (one scalar per case); ``global'': our 30-start optimum of the same model; ``consistent'': printed $F_\\mathrm{obj}$ recovered within 1.5\\% at the printed parameters.", "tab:reprodyn"))
mac("ReproConsistent", sum(r[-1] == "yes" for r in rows)); mac("ReproAll", len(rows))
# diagnostics
if ex(f"{R}/diagnostics/solver_bench.csv"):
    b = pd.read_csv(f"{R}/diagnostics/solver_bench.csv").set_index("solver")
    mac("SolverErr", f"{b.max_err.filter(like='exact').max():.0e}".replace("e-0", "\\times10^{-") + "}")
    mac("SolverCachedMs", f"{b.ms.filter(like='cached').iloc[0]:.2f}"); mac("SolverLsodaMs", f"{b.loc['LSODA (tol 1e-09)','ms']:.1f}")
    mac("SolverRadauMs", f"{b.loc['Radau (tol 1e-09)','ms']:.0f}")
if ex(f"{R}/diagnostics/landscape.csv"):
    L = pd.read_csv(f"{R}/diagnostics/landscape.csv")
    fr = L.groupby("case").apply(lambda g: np.mean(g.Fobj <= g.Fobj.min() * 1.001))
    mac("LandMinFrac", f"{100*fr.min():.0f}"); mac("LandMaxFrac", f"{100*fr.max():.0f}")
# transfer
if ex(f"{R}/transfer/fingerprint_transfer.csv"):
    t = pd.read_csv(f"{R}/transfer/fingerprint_transfer.csv"); t["se"] = (t.pred - t.obs) ** 2
    rm = t.groupby(["held_out", "method"]).se.mean().pow(.5).unstack()
    rows = [[h, f(r["ORIG fingerprint"], 3), f(r["processive fingerprint (shrunk, s_p=0.1)"], 3), f(r["processive fingerprint (shrunk, s_p=0.1)"] / r["ORIG fingerprint"], 3)] for h, r in rm.iterrows()]
    tot = t.groupby("method").se.mean().pow(.5)
    rows += ["MID", ["All", f(tot["ORIG fingerprint"], 3), f(tot["processive fingerprint (shrunk, s_p=0.1)"], 3), f(tot["processive fingerprint (shrunk, s_p=0.1)"] / tot["ORIG fingerprint"], 3)]]
    write("transfer", tabular(["Held-out profile", "Original path", "Processive path", "Ratio"], rows, "lrrr",
          "Cross-condition transfer of the composition path: RMSE (molar \\%) for the held-out profile predicted against $c$ from the other profiles of the same system.", "tab:transfer"))
    mac("TransferWins", int((rm["processive fingerprint (shrunk, s_p=0.1)"] < rm["ORIG fingerprint"]).sum())); mac("TransferN", len(rm))
    mac("TransferRatio", f(tot["processive fingerprint (shrunk, s_p=0.1)"] / tot["ORIG fingerprint"], 3))

if __name__ == "__main__":
    import sys
    extra = f"{JD}/code/make_tables_results.py"
    if ex(extra): exec(open(extra).read(), globals())
    with open(f"{PAP}/numbers.tex", "w") as fh:
        for k, v in M.items(): fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(len(M), "macros;", sorted(os.listdir(TAB)))
