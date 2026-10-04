"""Result figures (out-of-sample, black-box, ablation, simulation, transfer, in-sample fits)."""
import os, sys, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from plotstyle import *
from kinetics import JD, load_profiles, SPECIES
setup(); P = load_profiles(); R = f"{JD}/results"; ex = os.path.exists
CASES = ["Akerman70", "Akerman100", "Bornadel", "Tao025", "Tao035_045", "Mao123"]
CL = {"Akerman70": "Åkerman 70 °C", "Akerman100": "Åkerman 100 °C", "Bornadel": "Bornadel", "Tao025": "Tao $a_w$ 0.25",
      "Tao035_045": "Tao $a_w$ 0.35+0.45", "Mao123": "Mao batches 1-3"}
DS = {"Akerman70": ["Akerman70"], "Akerman100": ["Akerman100"], "Bornadel": ["Bornadel"], "Tao025": ["Tao025"],
      "Tao035_045": ["Tao035", "Tao045"], "Mao123": ["Mao1", "Mao2", "Mao3"]}

def rmse_table(proto):
    return pd.read_csv(f"{R}/summary/{proto}_rmse.csv", index_col=0)

def fig_oos(proto, name, title):
    rm = rmse_table(proto); T = pd.read_csv(f"{R}/summary/{proto}_tests.csv")
    meths = ["ORIG", "ORIG-full", "JOINT-BMA", "PPK-FIP"]
    fig, axs = plt.subplots(1, 2, figsize=(7.8, 2.9), gridspec_kw=dict(wspace=0.5, width_ratios=[1.45, 1]))
    a = axs[0]; x = np.arange(len(CASES)); w = 0.2
    for i, m in enumerate(meths):
        a.bar(x + (i - 1.5) * w, rm.loc[CASES, m], w * 0.92, color=col(m), label=m)
    a.set_xticks(x); a.set_xticklabels([CL[c] for c in CASES], rotation=25, ha="right", fontsize=7)
    a.set_ylabel("RMSE (molar %)"); a.set_title(f"(a) {title}: RMSE per case", loc="left"); a.legend(ncol=2, fontsize=6.6)
    a = axs[1]
    t = T[(T.ref == "ORIG") & (T.case != "ALL")].set_index("case").loc[CASES]
    t2 = T[(T.ref == "ORIG-full") & (T.case != "ALL")].set_index("case").loc[CASES]
    a.axvline(1, color=INK2, lw=0.8)
    for k, (tt, m, off) in enumerate(((t, "ORIG", -0.12), (t2, "ORIG-full", 0.12))):
        y = np.arange(len(CASES)) + off
        a.scatter(tt.ratio, y, color=col(m), s=26, zorder=3, label=f"vs {m}")
        for yi, r, p, md in zip(y, tt.ratio, tt.p_holm, tt.median_diff):
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            a.text(r * 1.06, yi, star, va="center", fontsize=8, fontweight="bold", color=SLOTS[0] if md < 0 else SLOTS[1])
    a.set_xscale("log")
    from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter
    lo, hi = a.get_xlim(); cand = (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10) if hi / lo > 6 else (0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.5, 2)
    tk = [v for v in cand if lo <= v <= hi]
    a.xaxis.set_major_locator(FixedLocator(tk)); a.xaxis.set_major_formatter(FixedFormatter([f"{v:g}" for v in tk])); a.xaxis.set_minor_locator(NullLocator())
    a.set_yticks(range(len(CASES))); a.set_yticklabels([CL[c] for c in CASES], fontsize=7); a.invert_yaxis()
    a.set_xlabel("RMSE ratio PPK-FIP / reference (log)"); a.set_title("(b) Ratio (<1 = PPK-FIP better)", loc="left")
    a.legend(fontsize=6.5, loc="lower left")
    save(fig, name)

def fig_blackbox():
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.8), gridspec_kw=dict(wspace=0.3))
    for a, proto, ttl in zip(axs, ("loto", "fcst"), ("(a) Leave-one-time-point-out", "(b) Forward extrapolation")):
        if not ex(f"{R}/summary/{proto}_rmse.csv"): continue
        rm = rmse_table(proto); meths = [m for m in ["PPK-FIP", "ORIG", "GP", "MLP-ens", "TabICL", "NODE"] if m in rm.columns]
        x = np.arange(len(CASES)); w = 0.8 / len(meths)
        for i, m in enumerate(meths):
            a.bar(x + (i - (len(meths) - 1) / 2) * w, rm.loc[CASES, m], w * 0.92, color=col(m), label=m)
        a.set_yscale("log"); a.set_xticks(x); a.set_xticklabels([CL[c] for c in CASES], rotation=25, ha="right", fontsize=6.8)
        a.set_ylabel("RMSE (molar %, log)"); a.set_title(ttl, loc="left")
    axs[0].legend(ncol=3, fontsize=6.3, loc="upper left")
    save(fig, "fig13_blackbox")

def fig_forecast():
    cv = pd.read_csv(f"{R}/cv/fcst_all.csv.gz"); proc = pd.read_csv(f"{R}/cv/fcst_procedures.csv.gz")
    bb = pd.read_csv(f"{R}/cv/fcst_bb_GP_MLP-ens_TabICL_NODE.csv")
    show = [("Akerman100", "Akerman100"), ("Bornadel", "Bornadel"), ("Tao025", "Tao025")]
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.6), gridspec_kw=dict(wspace=0.3))
    for a, (case, d) in zip(axs, show):
        g = P[P.dataset == d]; tm = g.t.max()
        a.axvspan(0, 0.5 * tm, color="#f1f0ec", zorder=0)
        for sp in ["M", "D", "T"]:
            a.plot(g.t, g[sp], SPEC_MK[sp], color=SPEC_COL[sp], ms=3.2, label=sp)
            for m, ls in (("ORIG", "--"), ("PPK-FIP", "-")):
                q = proc[(proc.case == case) & (proc.procedure == ("FIP-BMA-meta" if m == "PPK-FIP" else m)) & (proc.species == sp)]
                if len(q):
                    rr = P.loc[q.row]; o = rr[sp].values
                    a.plot(rr.t, o + q.err.values, ls, color=SPEC_COL[sp], lw=1.1)
            q = bb[(bb.case == case) & (bb.method == "NODE") & (bb.species == sp) & (bb.dataset == d)]
            a.plot(q.t, q.pred, ":", color=SPEC_COL[sp], lw=1.0)
        a.set_title(CL[case], loc="left"); a.set_xlabel("t (h)"); a.set_ylim(-3, 103)
    axs[0].set_ylabel("molar %")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=INK2, ls="--", label="ORIG"), Line2D([], [], color=INK2, ls="-", label="PPK-FIP"), Line2D([], [], color=INK2, ls=":", label="neural ODE")]
    h += [Line2D([], [], color=SPEC_COL[s], marker=SPEC_MK[s], ls="", label=s) for s in "MDT"]
    fig.legend(handles=h, fontsize=6.6, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.1))
    save(fig, "fig12_forecast")

if __name__ == "__main__":
    for proto, nm, tt in (("loto", "fig10_loto", "LOTO cross-validation"), ("fcst", "fig11_forecast_bars", "Forward extrapolation")):
        if ex(f"{R}/summary/{proto}_tests.csv"): fig_oos(proto, nm, tt)
    fig_blackbox()
    if ex(f"{R}/cv/fcst_procedures.csv.gz"): fig_forecast()

def fig_transfer():
    t = pd.read_csv(f"{R}/transfer/fingerprint_transfer.csv"); t["se"] = (t.pred - t.obs) ** 2
    rm = t.groupby(["held_out", "method"]).se.mean().pow(.5).unstack()
    order = ["Akerman70", "Akerman100", "Tao025", "Tao035", "Tao045", "Mao1", "Mao2", "Mao3"]; rm = rm.loc[order]
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.7), gridspec_kw=dict(wspace=0.3, width_ratios=[1.3, 1]))
    a = axs[0]; x = np.arange(len(rm)); w = 0.38
    a.bar(x - w / 2, rm["ORIG fingerprint"], w * 0.92, color=col("ORIG"), label="initial path ($p_M=p_D=0$)")
    a.bar(x + w / 2, rm["processive fingerprint (shrunk, s_p=0.1)"], w * 0.92, color=col("PPK-FIP"), label="processive path (shrunk)")
    a.set_xticks(x); a.set_xticklabels([o.replace("Akerman", "Åk").replace("Tao0", "Tao .") for o in order], rotation=30, ha="right", fontsize=7)
    a.set_ylabel("held-out RMSE (molar %)"); a.set_title("(a) Transfer to an unseen condition", loc="left"); a.legend(fontsize=6.5)
    a = axs[1]; d = P[(P.dataset == "Tao045") & (P.t > 0)]; tt = t[t.held_out == "Tao045"]
    for sp in SPECIES:
        a.plot(d.c, d[sp], SPEC_MK[sp], color=SPEC_COL[sp], ms=3.2, label=sp)
        for m, ls in (("ORIG fingerprint", "--"), ("processive fingerprint (shrunk, s_p=0.1)", "-")):
            q = tt[(tt.method == m) & (tt.species == sp)].sort_values("c"); a.plot(q.c, q.pred, ls, color=SPEC_COL[sp], lw=1.1)
    a.set_xlabel("degree of reaction c (%)"); a.set_ylabel("molar %"); a.set_title("(b) Tao $a_w$ 0.45 predicted from 0.25+0.35", loc="left", fontsize=8.3)
    a.legend(fontsize=6.3, ncol=4, loc="upper center")
    save(fig, "fig16_transfer")

def fig_sim():
    s = pd.read_csv(f"{R}/sim/sim.csv")
    sc = list(s.scenario.unique()); meths = ["ORIG", "ORIG-full", "JOINT-BIC"] + sorted([m for m in s.method.unique() if m.startswith("FIP")])
    fig, axs = plt.subplots(2, len(sc), figsize=(7.8, 4.6), gridspec_kw=dict(hspace=0.55, wspace=0.3))
    lab = {m: ("PPK-FIP" if m == "FIP-BMA(lam=3,sp=0.2)" else m.replace("FIP-BMA(lam=1,sp=", "FIP $\\lambda$=1, $s_p$=").replace(")", "")) for m in meths}
    for j, scen in enumerate(sc):
        g = s[s.scenario == scen]
        for i, (var, yl) in enumerate((("curve_rmse", "curve RMSE vs truth"), ("SM_err", "log($\\hat S_M/S_M$)"))):
            a = axs[i, j]; data = [g[g.method == m][var].dropna().values for m in meths]
            bp = a.boxplot(data, widths=0.6, patch_artist=True, showfliers=False)
            for k, b in enumerate(bp["boxes"]):
                b.set_facecolor(col("PPK-FIP") if meths[k].startswith("FIP") else col(meths[k])); b.set_alpha(0.85); b.set_edgecolor(INK2)
            for med in bp["medians"]: med.set_color(INK)
            if var == "SM_err": a.axhline(0, color=INK2, lw=0.8)
            a.set_xticks(range(1, len(meths) + 1)); a.set_xticklabels([lab[m] for m in meths], rotation=35, ha="right", fontsize=6)
            if j == 0: a.set_ylabel(yl)
            if i == 0: a.set_title(scen.split(" (")[0], loc="left", fontsize=8.5)
    save(fig, "fig15_simulation")

LABD = {"Akerman70": "Åkerman 70 °C", "Akerman100": "Åkerman 100 °C", "Bornadel": "Bornadel", "Tao025": "Tao $a_w$ 0.25",
        "Tao035": "Tao $a_w$ 0.35", "Tao045": "Tao $a_w$ 0.45", "Mao1": "Mao batch 1", "Mao2": "Mao batch 2", "Mao3": "Mao batch 3"}
def fig_fits():
    G = json.load(open(f"{R}/final/grids.json"))
    case_of = {d: c for c, ds in DS.items() for d in ds}
    fig, axs = plt.subplots(3, 3, figsize=(7.6, 6.6), gridspec_kw=dict(hspace=0.45, wspace=0.25))
    for a, d in zip(axs.flat, ["Akerman70", "Akerman100", "Bornadel", "Tao025", "Tao035", "Tao045", "Mao1", "Mao2", "Mao3"]):
        c = case_of[d]; g = P[P.dataset == d]
        for m, ls in (("ORIG", "--"), ("PPK-FIP", "-")):
            tg, Y = G[c][m][d]; Y = np.array(Y)
            for s, sp in enumerate(SPECIES):
                if g[sp].notna().any(): a.plot(tg, Y[:, s], ls, color=SPEC_COL[sp], lw=1.2 if ls == "-" else 1.0)
        for sp in SPECIES:
            if g[sp].notna().any(): a.plot(g.t, g[sp], SPEC_MK[sp], color=SPEC_COL[sp], ms=3.0, mec="white", mew=0.4)
        a.set_title(LABD[d], loc="left", fontsize=8); a.set_ylim(-3, 103)
        if a in axs[:, 0]: a.set_ylabel("molar %")
        if a in axs[-1]: a.set_xlabel("t (h)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=INK2, ls="--", label="initial two-step (refit)"), Line2D([], [], color=INK2, ls="-", label="PPK-FIP")]
    h += [Line2D([], [], color=SPEC_COL[s], marker=SPEC_MK[s], ls="", label=s) for s in SPECIES]
    fig.legend(handles=h, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 0.95), fontsize=7)
    save(fig, "fig06_fits_time")

def fig_paths():
    from models import comp_at_c
    Pr = json.load(open(f"{R}/final/params.json")); fp = json.load(open(f"{R}/baseline/fingerprint.json"))
    show = [("Bornadel", "Bornadel", "Bornadel"), ("Tao025", "Tao025", "Tao"), ("Akerman100", "Akerman100", "Akerman")]
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.6), gridspec_kw=dict(wspace=0.25))
    cg = np.linspace(0, 299, 300)
    for a, (case, d, grp) in zip(axs, show):
        g = P[(P.dataset == d)]
        w = Pr[case]["PPK-FIP"]["weights"]; ps, pf = Pr[case]["FIPs"][d], Pr[case]["FIPf"][d]
        Yo = comp_at_c(cg, fp[grp]["SM"], fp[grp]["SD"])
        Yn = w[0] * comp_at_c(cg, ps["SM"], ps["SD"], ps["pM"], ps["pD"]) + w[1] * comp_at_c(cg, pf["SM"], pf["SD"], pf["pM"], pf["pD"])
        for s, sp in enumerate(SPECIES):
            a.plot(cg, Yo[:, s], "--", color=SPEC_COL[sp], lw=1.0); a.plot(cg, Yn[:, s], "-", color=SPEC_COL[sp], lw=1.2)
            a.plot(g.c, g[sp], SPEC_MK[sp], color=SPEC_COL[sp], ms=3.0, mec="white", mew=0.4)
        pm = w[0] * ps["pM"] + w[1] * pf["pM"]; pd_ = w[0] * ps["pD"] + w[1] * pf["pD"]
        a.set_title(f"{LABD[d]}: $p_M$={pm:.2f}, $p_D$={pd_:.2f}", loc="left", fontsize=7.6); a.set_xlabel("degree of reaction c (%)")
    axs[0].set_ylabel("molar %")
    save(fig, "fig07_paths_c")

def fig_insample():
    I = pd.read_csv(f"{R}/final/insample.csv")
    PAPF = {"Akerman70": 59.0, "Akerman100": 117.9, "Bornadel": 329.9, "Tao025": 121.2, "Tao035_045": 325.8, "Mao123": 1475.9}
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.8), gridspec_kw=dict(wspace=0.3))
    x = np.arange(len(CASES)); w = 0.2
    a = axs[0]
    vals = {"best reported fit": [PAPF[c] for c in CASES]}
    for m in ("ORIG", "ORIG-full", "PPK-FIP"): vals[m + (" refit" if m.startswith("ORIG") else "")] = [I[(I.case == c) & (I.method == m)].Fobj.iloc[0] for c in CASES]
    cols = [INK2, col("ORIG"), col("ORIG-full"), col("PPK-FIP")]
    for i, (k, v) in enumerate(vals.items()):
        a.bar(x + (i - 1.5) * w, np.array(v) / np.array(vals["best reported fit"]), w * 0.92, color=cols[i], label=k)
    a.axhline(1, color=INK2, lw=0.7); a.set_ylabel("$F_\\mathrm{obj}$ / best reported $F_\\mathrm{obj}$")
    a.set_xticks(x); a.set_xticklabels([CL[c] for c in CASES], rotation=25, ha="right", fontsize=6.8); a.legend(fontsize=6.2)
    a.set_title("(a) Sum of squares (lower is better)", loc="left")
    a = axs[1]
    for i, m in enumerate(("ORIG", "ORIG-full", "PPK-FIP")):
        b = [I[(I.case == c) & (I.method == m)].BIC_total.iloc[0] - I[(I.case == c) & (I.method == "ORIG")].BIC_total.iloc[0] for c in CASES]
        a.bar(x + (i - 1) * 0.27, b, 0.25, color=col(m), label=m)
    a.axhline(0, color=INK2, lw=0.7); a.set_ylabel("ΔBIC$_\\mathrm{tot}$ vs. ORIG (refit)")
    a.set_xticks(x); a.set_xticklabels([CL[c] for c in CASES], rotation=25, ha="right", fontsize=6.8); a.legend(fontsize=6.2)
    a.set_title("(b) BIC counting all parameters", loc="left")
    save(fig, "fig08_insample")

def fig_resid():
    G = json.load(open(f"{R}/final/grids.json"))
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.7), gridspec_kw=dict(wspace=0.3, width_ratios=[1.3, 1]))
    a = axs[0]; data, pos, cc = [], [], []
    for i, c in enumerate(CASES):
        for j, m in enumerate(("ORIG", "PPK-FIP")):
            r = []
            for d in DS[c]:
                tg, Y = G[c][m][d]; Y = np.array(Y); g = P[(P.dataset == d) & (P.t > 0)]
                pr = np.stack([np.interp(g.t, tg, Y[:, s]) for s in range(4)], 1)
                e = (pr - g[SPECIES].to_numpy(float)).ravel(); r += list(e[~np.isnan(e)])
            data.append(np.abs(r)); pos.append(i * 3 + j); cc.append(col(m))
    bp = a.boxplot(data, positions=pos, widths=0.8, patch_artist=True, showfliers=False)
    for b, c_ in zip(bp["boxes"], cc): b.set_facecolor(c_); b.set_edgecolor(INK2)
    for med in bp["medians"]: med.set_color(INK)
    a.set_xticks([i * 3 + 0.5 for i in range(len(CASES))]); a.set_xticklabels([CL[c] for c in CASES], rotation=25, ha="right", fontsize=6.8)
    a.set_ylabel("|residual| (molar %)"); a.set_title("(a) Absolute residuals, full-data fits", loc="left")
    from matplotlib.patches import Patch
    a.legend(handles=[Patch(color=col("ORIG"), label="ORIG"), Patch(color=col("PPK-FIP"), label="PPK-FIP")], fontsize=6.5)
    a = axs[1]; c = "Bornadel"; d = "Bornadel"; g = P[(P.dataset == d) & (P.t > 0)]
    for m, mk in (("ORIG", "o"), ("PPK-FIP", "s")):
        tg, Y = G[c][m][d]; Y = np.array(Y)
        for s, sp in enumerate(["M", "D"]):
            si = SPECIES.index(sp); pr = np.interp(g.t, tg, Y[:, si])
            a.plot(g.t, pr - g[sp], marker=mk, ls="-" if m == "PPK-FIP" else "--", color=SPEC_COL[sp], ms=3, lw=0.8,
                   label=f"{sp}, {m}")
    a.axhline(0, color=INK2, lw=0.7); a.set_xlabel("t (h)"); a.set_ylabel("residual (molar %)")
    a.set_title("(b) Bornadel: structured residuals of M, D", loc="left"); a.legend(fontsize=6, ncol=2)
    save(fig, "fig09_residuals")

def fig_pooling():
    E = pd.read_csv(f"{R}/cv/loto_procedures.csv.gz"); E = E[E.procedure.isin(["ORIG", "ORIG-full", "FIP-BMA-meta"])]
    E["procedure"] = E.procedure.replace({"FIP-BMA-meta": "L0 (PPK-FIP)"})
    for fn, lv in (("loto_pooled_path.csv.gz", "L1 shared path"), ("loto_pooled_path_tau3.csv.gz", "L1h hierarchical F"), ("loto_pooled.csv.gz", "L2 shared path+F")):
        q = pd.read_csv(f"{R}/cv/{fn}"); q["procedure"] = lv; E = pd.concat([E, q[["case", "fold", "procedure", "row", "species", "err"]]])
    E["se"] = E.err ** 2; rm = E.groupby(["case", "procedure"]).se.mean().pow(.5).unstack()
    sib = ["Akerman70", "Akerman100", "Tao025", "Tao035_045"]
    meths = ["ORIG", "ORIG-full", "L0 (PPK-FIP)", "L1 shared path", "L1h hierarchical F", "L2 shared path+F"]
    cols = [col("ORIG"), col("ORIG-full"), col("PPK-FIP"), SLOTS[2], SLOTS[5], SLOTS[6]]
    fig, a = plt.subplots(figsize=(6.4, 2.7)); x = np.arange(len(sib)); w = 0.8 / len(meths)
    for i, (m, c_) in enumerate(zip(meths, cols)):
        a.bar(x + (i - (len(meths) - 1) / 2) * w, rm.loc[sib, m], w * 0.92, color=c_, label=m)
    a.set_xticks(x); a.set_xticklabels([CL[c] for c in sib], fontsize=7.5); a.set_ylabel("LOTO RMSE (molar %)")
    a.set_title("Multi-task pooling across conditions of the same system", loc="left"); a.legend(fontsize=6.3, ncol=2)
    save(fig, "fig17_pooling")

def fig_ablation():
    A = pd.read_csv(f"{R}/summary/loto_ablation.csv")
    fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.3), gridspec_kw=dict(wspace=0.75, width_ratios=[1, 1.15]))
    a = axs[0]; A = A.iloc[::-1]
    cc = [col("PPK-FIP") if v.startswith("PPK-FIP") else (col("ORIG") if v.startswith("two-step") else "#8a8985") for v in A.variant]
    a.barh(range(len(A)), A.ALL, color=cc); a.set_yticks(range(len(A))); a.set_yticklabels([v.replace("$\\lambda$", "λ").replace("$s_p$", "s_p") for v in A.variant], fontsize=6.3)
    full = A[A.variant == "PPK-FIP (full)"].ALL.iloc[0]; a.axvline(full, color=col("PPK-FIP"), lw=0.8, ls="--")
    a.set_xlim(A.ALL.min() * 0.9, A.ALL.max() * 1.03); a.set_xlabel("pooled LOTO RMSE (molar %)"); a.set_title("(a) Component ablation", loc="left")
    a = axs[1]
    M = A.set_index("variant")[CASES].iloc[::-1]
    rel = np.log(M.div(M.loc["PPK-FIP (full)"], axis=1))
    im = a.imshow(rel.values, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    a.set_xticks(range(len(CASES))); a.set_xticklabels([CL[c] for c in CASES], rotation=35, ha="right", fontsize=6.3)
    a.set_yticks(range(len(rel))); a.set_yticklabels([v.replace("$\\lambda$", "λ").replace("$s_p$", "s_p") for v in rel.index], fontsize=6.0)
    for i in range(rel.shape[0]):
        for j in range(rel.shape[1]):
            a.text(j, i, f"{np.exp(rel.values[i, j]):.2f}", ha="center", va="center", fontsize=5.5, color=INK)
    a.grid(False); a.set_title("(b) RMSE ratio to PPK-FIP per case", loc="left")
    cb = fig.colorbar(im, ax=a, fraction=0.04, pad=0.02); cb.set_label("log ratio (red = worse)", fontsize=6.5); cb.ax.tick_params(labelsize=6)
    save(fig, "fig14_ablation")

def fig_hybrid():
    if not ex(f"{R}/cv/loto_hybrid.csv.gz"): return
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.8), gridspec_kw=dict(wspace=0.3))
    for a, proto, ttl in zip(axs, ("loto", "fcst"), ("(a) Leave-one-time-point-out", "(b) Forward extrapolation")):
        h = pd.read_csv(f"{R}/cv/{proto}_hybrid.csv.gz"); h["se"] = h.err ** 2
        rm = h.groupby(["case", "procedure"]).se.mean().pow(.5).unstack()
        bb = rmse_table(proto)
        meths = [("ORIG", col("ORIG")), ("ORIG+GP", "#f4a582"), ("PPK-FIP", col("PPK-FIP")), ("PPK-FIP+GP", SLOTS[6])]
        x = np.arange(len(CASES)); w = 0.16
        for i, (m, c_) in enumerate(meths):
            a.bar(x + (i - 2) * w, rm.loc[CASES, m], w * 0.92, color=c_, label=m)
        if "GP" in bb.columns: a.bar(x + 2 * w, bb.loc[CASES, "GP"], w * 0.92, color=col("GP"), label="GP alone")
        a.set_yscale("log"); a.set_xticks(x); a.set_xticklabels([CL[c] for c in CASES], rotation=25, ha="right", fontsize=6.8)
        a.set_ylabel("RMSE (molar %, log)"); a.set_title(ttl, loc="left")
    axs[0].legend(fontsize=6.2, ncol=2)
    save(fig, "fig18_hybrid")
