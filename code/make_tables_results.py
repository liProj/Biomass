SHORT = {"Akerman70": "AkSeventy", "Akerman100": "AkHundred", "Bornadel": "Born", "Tao025": "TaoA", "Tao035_045": "TaoB", "Mao123": "Mao"}
# executed inside make_tables.py (shares its namespace: M, mac, f, pfmt, tabular, write, CL, CASES, R, ex)
# ---------------------------------------------------------------- out-of-sample tables
for proto, lab, cap in (("loto", "loto", "Leave-one-time-point-out cross-validation"), ("fcst", "fcst", "Forward extrapolation (train $t\\le0.5t_\\mathrm{max}$)")):
    if not ex(f"{R}/summary/{proto}_tests.csv"): continue
    rm = pd.read_csv(f"{R}/summary/{proto}_rmse.csv", index_col=0); T = pd.read_csv(f"{R}/summary/{proto}_tests.csv")
    allr = pd.read_csv(f"{R}/summary/{proto}_rmse_all.csv", index_col=0).iloc[:, 0]
    cols = ["ORIG", "ORIG-full", "ORIG-BIC", "JOINT-BMA", "PPK-FIP"]
    rows = []
    for c in CASES + ["ALL"]:
        r = rm.loc[c] if c != "ALL" else allr
        t1 = T[(T.ref == "ORIG") & (T.case == c)].iloc[0]; t2 = T[(T.ref == "ORIG-full") & (T.case == c)].iloc[0]
        best = min(r[k] for k in cols)
        cells = [("\\textbf{" + f(r[k], 3) + "}") if abs(r[k] - best) < 1e-9 else f(r[k], 3) for k in cols]
        pv1 = t1.p_holm if c != "ALL" else t1.p; pv2 = t2.p_holm if c != "ALL" else t2.p
        rows.append([CL[c]] + cells + [f(t1.ratio, 3), pfmt(pv1), f(t2.ratio, 3), pfmt(pv2)])
        if c == CASES[-1]: rows.append("MID")
    write(f"{lab}_main", tabular(["Case", "ORIG", "ORIG-full", "ORIG-BIC", "JOINT-BMA", "PPK-FIP", "ratio$_\\mathrm{O}$", "$p_\\mathrm{O}$", "ratio$_\\mathrm{F}$", "$p_\\mathrm{F}$"],
          rows, "lrrrrrrrrr", f"{cap}: RMSE (molar \\%) of every procedure re-estimated from scratch in each fold. ratio = RMSE(PPK-FIP)/RMSE(reference); $p$ = Wilcoxon signed-rank test on paired squared errors, Holm-corrected over the six cases (O: vs. original simplified, F: vs. original full model; pooled row uncorrected).",
          f"tab:{lab}"))
    t1 = T[(T.ref == "ORIG") & (T.case != "ALL")]; t2 = T[(T.ref == "ORIG-full") & (T.case != "ALL")]
    P_ = proto.capitalize()
    mac(f"{P_}WinsOrig", int((t1.ratio < 1).sum())); mac(f"{P_}WinsFull", int((t2.ratio < 1).sum()))
    mac(f"{P_}SigWinsOrig", int(((t1.p_holm < 0.05) & (t1.median_diff < 0)).sum())); mac(f"{P_}SigLossOrig", int(((t1.p_holm < 0.05) & (t1.median_diff > 0)).sum()))
    mac(f"{P_}SigWinsFull", int(((t2.p_holm < 0.05) & (t2.median_diff < 0)).sum())); mac(f"{P_}SigLossFull", int(((t2.p_holm < 0.05) & (t2.median_diff > 0)).sum()))
    a1 = T[(T.ref == "ORIG") & (T.case == "ALL")].iloc[0]; a2 = T[(T.ref == "ORIG-full") & (T.case == "ALL")].iloc[0]
    mac(f"{P_}AllNew", f(a1.rmse_new, 3)); mac(f"{P_}AllOrig", f(a1.rmse_ref, 3)); mac(f"{P_}AllFull", f(a2.rmse_ref, 3))
    mac(f"{P_}AllRedOrig", f(100 * (1 - a1.ratio), 1)); mac(f"{P_}AllRedFull", f(100 * (1 - a2.ratio), 1))
    mac(f"{P_}AllPOrig", pfmt(a1.p)); mac(f"{P_}AllPFull", pfmt(a2.p))
    for c in CASES:
        k = SHORT[c]
        mac(f"{P_}{k}New", f(rm.loc[c, "PPK-FIP"], 3)); mac(f"{P_}{k}Orig", f(rm.loc[c, "ORIG"], 3)); mac(f"{P_}{k}Full", f(rm.loc[c, "ORIG-full"], 3))
    # black-box table
    bbm = [m for m in ["GP", "MLP-ens", "TabICL", "NODE"] if m in rm.columns]
    if bbm:
        rows = [[CL[c]] + [f(rm.loc[c, m], 3) for m in ["ORIG", "PPK-FIP"] + bbm] for c in CASES] + ["MID", [CL["ALL"]] + [f(allr[m], 3) for m in ["ORIG", "PPK-FIP"] + bbm]]
        write(f"{lab}_blackbox", tabular(["Case", "ORIG", "PPK-FIP", "GP", "MLP ens.", "TabICL", "Neural ODE"], rows, "lrrrrrr",
              f"{cap}: mechanistic models versus black-box machine-learning baselines (RMSE, molar \\%).", f"tab:{lab}bb"))
        for m in bbm: mac(f"{P_}BB{m.replace('-','').replace('ens','')}", f(allr[m], 2))
# ---------------------------------------------------------------- in-sample
if ex(f"{R}/final/insample.csv"):
    I = pd.read_csv(f"{R}/final/insample.csv")

    rows = []
    for c in CASES:
        g = I[I.case == c].set_index("method"); pp = PAPERI[c]
        rows.append([CL[c], f"{pp[0][0]:.1f} / {pp[0][1]:.1f}", f"{pp[1][0]:.1f} / {pp[1][1]:.1f}",
                     f"{g.loc['ORIG','Fobj']:.1f} / {g.loc['ORIG','BIC_total']:.1f}", f"{g.loc['ORIG-full','Fobj']:.1f} / {g.loc['ORIG-full','BIC_total']:.1f}",
                     f"\\textbf{{{g.loc['PPK-FIP','Fobj']:.1f}}} / \\textbf{{{g.loc['PPK-FIP','BIC_total']:.1f}}}", f"{g.loc['PPK-FIP','k']:.1f}"])
        k = SHORT[c]
        mac(f"In{k}Fnew", f(g.loc["PPK-FIP", "Fobj"], 1)); mac(f"In{k}Fpaper", f(min(pp[0][0], pp[1][0]), 1))
        mac(f"In{k}Red", f(100 * (1 - g.loc["PPK-FIP", "Fobj"] / min(pp[0][0], pp[1][0])), 1))
    write("insample", tabular(["Case", "Paper simpl. $F_\\mathrm{obj}$/BIC", "Paper full", "ORIG refit $F_\\mathrm{obj}$/BIC$_\\mathrm{tot}$", "ORIG-full refit", "PPK-FIP", "$\\bar k$"],
          rows, "lrrrrrr", "In-sample fit on the full data (paper protocol). Paper columns: printed $F_\\mathrm{obj}$ / printed BIC (paper convention, selectivities not counted). Refit and PPK-FIP columns: $F_\\mathrm{obj}$ / BIC$_\\mathrm{tot}$ counting every estimated parameter (for PPK-FIP the BIC-weighted mean parameter count $\\bar k$).", "tab:insample"))
    wins = sum(I[(I.case == c) & (I.method == "PPK-FIP")].Fobj.iloc[0] < min(PAPERI[c][0][0], PAPERI[c][1][0]) for c in CASES)
    winsB = sum(I[(I.case == c) & (I.method == "PPK-FIP")].BIC_total.iloc[0] < I[(I.case == c) & (I.method.isin(["ORIG", "ORIG-full"]))].BIC_total.min() for c in CASES)
    mac("InFobjWins", wins); mac("InBICWins", winsB)
# ---------------------------------------------------------------- multi-task pooling extension (LOTO)
if ex(f"{R}/cv/loto_pooled.csv.gz"):
    E = pd.read_csv(f"{R}/cv/loto_procedures.csv.gz"); E = E[E.procedure.isin(["ORIG", "ORIG-full", "FIP-BMA-meta"])]
    E["procedure"] = E.procedure.replace({"FIP-BMA-meta": "L0"})
    for fn, lv in (("loto_pooled.csv.gz", "L2"), ("loto_pooled_path.csv.gz", "L1"), ("loto_pooled_path_tau3.csv.gz", "L1h")):
        if ex(f"{R}/cv/{fn}"):
            q = pd.read_csv(f"{R}/cv/{fn}"); q["procedure"] = lv; E = pd.concat([E, q[["case", "fold", "procedure", "row", "species", "err"]]])
    E["se"] = E.err ** 2; rmp = E.groupby(["case", "procedure"]).se.mean().pow(.5).unstack()
    sib = ["Akerman70", "Akerman100", "Tao025", "Tao035_045"]
    lr = np.log(rmp.loc[sib, ["L0", "L1", "L1h", "L2"]].div(rmp.loc[sib, "ORIG"], axis=0))
    choice = {c: lr.drop(index=c).mean().idxmin() for c in sib}
    sel = pd.concat([g[g.procedure == choice.get(c, "L0")] for c, g in E.groupby("case")])
    rows = [[CL[c], f(rmp.loc[c, "ORIG"], 3), f(rmp.loc[c, "ORIG-full"], 3), f(rmp.loc[c, "L0"], 3),
             f(rmp.loc[c, "L1"], 3) if c in sib else "--", f(rmp.loc[c, "L1h"], 3) if c in sib else "--", f(rmp.loc[c, "L2"], 3) if c in sib else "--",
             choice.get(c, "L0")] for c in CASES]
    allsel = np.sqrt(sel.se.mean()); allL0 = np.sqrt(E[E.procedure == "L0"].se.mean())
    rows += ["MID", [CL["ALL"], f(np.sqrt(E[E.procedure == 'ORIG'].se.mean()), 3), f(np.sqrt(E[E.procedure == 'ORIG-full'].se.mean()), 3), f(allL0, 3), "", "", "", f"{allsel:.3f}"]]
    write("pooling", tabular(["Case", "ORIG", "ORIG-full", "L0 (PPK-FIP)", "L1 path", "L1h hier.", "L2 path+$F$", "meta-choice"], rows, "lrrrrrrl",
          "Multi-task extension (LOTO RMSE, molar \\%): profiles of the same system share the composition path (L1), additionally the lumped constants with hierarchical shrinkage ($\\tau=\\ln3$, L1h) or fully (L2). The level is chosen by leave-one-case-out meta-validation; the last row gives the pooled RMSE of the meta-selected levels (last column) versus L0.", "tab:pooling"))
    mac("PoolAllSel", f(allsel, 3)); mac("PoolAllLo", f(allL0, 3)); mac("PoolAkHundredLii", f(rmp.loc["Akerman100", "L2"], 3))
# ---------------------------------------------------------------- simulation
if ex(f"{R}/sim/sim.csv"):
    S = pd.read_csv(f"{R}/sim/sim.csv"); PPK = "FIP-BMA(lam=3,sp=0.2)"
    meths = ["ORIG", "ORIG-full", "JOINT-BIC", "FIP-BMA(lam=1,sp=0.05)", "FIP-BMA(lam=1,sp=0.2)", PPK]
    names = {"FIP-BMA(lam=1,sp=0.05)": "FIP $\\lambda$=1,$s_p$=0.05", "FIP-BMA(lam=1,sp=0.2)": "FIP $\\lambda$=1,$s_p$=0.2", PPK: "PPK-FIP ($\\lambda$=3,$s_p$=0.2)"}
    rows = []
    for sc, g in S.groupby("scenario", sort=False):
        a = g.groupby("method")[["curve_rmse", "SM_err", "SD_err"]].agg(["mean", "std"])
        for m in meths:
            rows.append([sc.split(" (")[0] if m == "ORIG" else "", names.get(m, m), f"{a.loc[m,('curve_rmse','mean')]:.3f} $\\pm$ {a.loc[m,('curve_rmse','std')]:.3f}",
                         f"{a.loc[m,('SM_err','mean')]:+.3f} $\\pm$ {a.loc[m,('SM_err','std')]:.3f}", f"{a.loc[m,('SD_err','mean')]:+.3f} $\\pm$ {a.loc[m,('SD_err','std')]:.3f}"])
        rows.append("MID")
    rows = rows[:-1]
    write("sim", tabular(["Scenario (truth)", "Estimator", "curve RMSE vs truth", "bias $\\ln(\\hat S_M/S_M)$", "bias $\\ln(\\hat S_D/S_D)$"], rows, "llrrr",
          f"Monte-Carlo truth recovery ({S.rep.nunique()} replicates per scenario; mean $\\pm$ SD). Curve RMSE is computed against the noise-free truth on a dense time grid.", "tab:sim"))
    a = S.groupby(["scenario", "method"])[["curve_rmse", "SM_err"]].mean()
    s1, s2, s3 = list(S.scenario.unique())
    mac("SimSMBiasOrig", f"{a.loc[(s2,'ORIG'),'SM_err']:+.2f}"); mac("SimSMBiasNew", f"{a.loc[(s2,PPK),'SM_err']:+.2f}")
    mac("SimSMBiasOrigPct", f"{100*(np.exp(a.loc[(s2,'ORIG'),'SM_err'])-1):.0f}")
    mac("SimSiiOrig", f(a.loc[(s2, "ORIG"), "curve_rmse"], 3)); mac("SimSiiNew", f(a.loc[(s2, PPK), "curve_rmse"], 3))
    mac("SimSiiiOrig", f(a.loc[(s3, "ORIG"), "curve_rmse"], 3)); mac("SimSiiiFull", f(a.loc[(s3, "ORIG-full"), "curve_rmse"], 3)); mac("SimSiiiNew", f(a.loc[(s3, PPK), "curve_rmse"], 3))
    mac("SimSiOrig", f(a.loc[(s1, "ORIG"), "curve_rmse"], 3)); mac("SimSiNew", f(a.loc[(s1, PPK), "curve_rmse"], 3)); mac("SimSiFull", f(a.loc[(s1, "ORIG-full"), "curve_rmse"], 3))
    mac("SimSiJoint", f(a.loc[(s1, "JOINT-BIC"), "curve_rmse"], 3)); mac("NSimReps", S.rep.nunique())
# ---------------------------------------------------------------- misc parameter macros
if ex(f"{R}/final/params.json"):
    Pr = json.load(open(f"{R}/final/params.json"))
    def wpar(case, d, k):
        w = Pr[case]["PPK-FIP"]["weights"]; return w[0] * Pr[case]["FIPs"][d][k] + w[1] * Pr[case]["FIPf"][d][k]
    mac("BornPM", f(wpar("Bornadel", "Bornadel", "pM"), 2)); mac("BornPD", f(wpar("Bornadel", "Bornadel", "pD"), 3))
    mac("BornSM", f(wpar("Bornadel", "Bornadel", "SM"), 2)); mac("BornSD", f(wpar("Bornadel", "Bornadel", "SD"), 3))
    mac("AkHundredKd", f(wpar("Akerman100", "Akerman100", "kd"), 3)); mac("TaoAKd", f(wpar("Tao025", "Tao025", "kd"), 3))
    mac("AkSeventyKd", f"{wpar('Akerman70', 'Akerman70', 'kd'):.1e}"); mac("TaoAPD", f(wpar("Tao025", "Tao025", "pD"), 2))
    CI = pd.read_csv(f"{R}/final/ci.csv") if ex(f"{R}/final/ci.csv") else None
    def ci(c, tag, k):
        if CI is None: return ""
        q = CI[(CI.case == c) & (CI.Fset == tag) & (CI.param == k)]
        return f" $\\pm$ {q.ci95.iloc[0]:.3g}" if len(q) else ""
    rows = []
    for c in CASES:
        tag = "s" if Pr[c]["PPK-FIP"]["weights"][0] >= 0.5 else "f"
        for d in DS_[c]:
            p_ = Pr[c][f"FIP{tag}"][d]
            rows.append([d.replace("Akerman", "Åkerman "), f"{p_['SM']:.3f}" + ci(c, tag, "SM"), f"{p_['SD']:.4f}" + ci(c, tag, "SD"), f"{p_['pM']:.3f}" + ci(c, tag, "pM"),
                         f"{p_['pD']:.3f}" + ci(c, tag, "pD"), f"{p_['kd']:.2e}", f"{p_['kE']:.3g}", tag, f(Pr[c]["PPK-FIP"]["weights"][0], 2)])
    write("params", tabular(["Profile", "$S_M$", "$S_D$", "$p_M$", "$p_D$", "$k_d$ (h$^{-1}$)", "$k_E$", "$F$ set", "$w_\\mathrm{simpl}$"], rows, "lrrrrrrcr",
          "PPK-FIP estimates on the full data for the dominant model form ($F$ set with the larger BIC weight; $w_\\mathrm{simpl}$ = weight of the simplified form) with linearised 95\\% confidence half-widths computed as in the original paper. Values of $k_E$ at the upper bound (1000) indicate the non-identifiable ridge of the full $F$ set (Figure~\\ref{fig:landscape}b), along which only ratios of $k_E$ and the $F$ constants are determined.", "tab:params"))
rc_ = pd.read_csv(f"{R}/repro_compare.csv")
mac("AkHundredFullGlobal", f(rc_[(rc_.case == "Akerman100") & (rc_.model == "full") & (rc_.item == "Fobj")].refit.iloc[0], 1))
mac("MaoFullAtPaper", f(rc_[(rc_.case == "Mao123") & (rc_.model == "full") & (rc_.item == "Fobj")].at_paper_point.iloc[0], 1))
if ex(f"{R}/cv/loto_procedures.csv.gz"):
    _e = pd.read_csv(f"{R}/cv/loto_procedures.csv.gz"); mac("NLotoFolds", _e.groupby(["case", "fold"]).ngroups)
# ---------------------------------------------------------------- ablation
if ex(f"{R}/summary/loto_ablation.csv"):
    A = pd.read_csv(f"{R}/summary/loto_ablation.csv")
    Fc = pd.read_csv(f"{R}/summary/fcst_ablation.csv").set_index("variant") if ex(f"{R}/summary/fcst_ablation.csv") else None
    rows = []
    for _, r in A.iterrows():
        rows.append([r.variant] + [f(r[c], 3) for c in CASES] + [f"\\textbf{{{r.ALL:.3f}}}" if r.variant == "PPK-FIP (full)" else f(r.ALL, 3), pfmt(r.p_vs_full),
                     f(Fc.loc[r.variant, "ALL"], 2) if Fc is not None and r.variant in Fc.index else "--"])
    write("ablation", tabular(["Variant", "Åk70", "Åk100", "Born.", "Tao.25", "Tao.35/.45", "Mao", "LOTO all", "$p$ vs full", "Fcst all"], rows, "lrrrrrrrrr",
          "Ablation of PPK-FIP (LOTO RMSE per case and pooled, molar \\%; $p$: Wilcoxon signed-rank test on paired squared errors against the full method; last column: pooled forward-extrapolation RMSE). Every variant is re-estimated in every fold.", "tab:ablation", "\\scriptsize"))
    g = A.set_index("variant")
    mac("AblLamOne", f(g.loc["fixed $\\lambda$=1, $s_p$=0.2 (no meta-selection)", "ALL"], 3)); mac("AblJoint", f(g.loc["no fingerprint prior (joint, BMA)", "ALL"], 3))
    mac("AblSimplOnly", f(g.loc["simplified F only (no averaging)", "ALL"], 3)); mac("AblFullOnly", f(g.loc["full F only (no averaging)", "ALL"], 3))
    for k, lab in (("AblNoProc", "no pseudo-processivity"), ("AblNoKd", "no deactivation"), ("AblNoBoth", "no processivity, no deactivation"), ("AblRerun", "PPK-FIP (re-run, consistency check)")):
        if lab in g.index:
            mac(k, f(g.loc[lab, "ALL"], 3)); mac(k + "Born", f(g.loc[lab, "Bornadel"], 3)); mac(k + "P", pfmt(g.loc[lab, "p_vs_full"]))
# ---------------------------------------------------------------- hybrid (mechanistic + GP discrepancy)
if ex(f"{R}/cv/loto_hybrid.csv.gz"):
    rows = []; tot = {}
    for proto in ("loto", "fcst"):
        if not ex(f"{R}/cv/{proto}_hybrid.csv.gz"): continue
        h = pd.read_csv(f"{R}/cv/{proto}_hybrid.csv.gz"); h["se"] = h.err ** 2
        rm = h.groupby(["case", "procedure"]).se.mean().pow(.5).unstack(); al = h.groupby("procedure").se.mean().pow(.5); tot[proto] = (rm, al, h)
        for c in CASES + ["ALL"]:
            r = rm.loc[c] if c != "ALL" else al
            rows.append([("LOTO" if proto == "loto" else "Extrapolation") if c == CASES[0] else "", CL[c]] + [f(r[m], 3) for m in ["ORIG", "ORIG+GP", "PPK-FIP", "PPK-FIP+GP"]])
        rows.append("MID")
    rows = rows[:-1]
    write("hybrid", tabular(["Protocol", "Case", "ORIG", "ORIG+GP", "PPK-FIP", "PPK-FIP+GP"], rows, "llrrrr",
          "Mechanistic mean plus Gaussian-process discrepancy (RMSE, molar \\%). The GP is fitted to the training residuals of each species and reverts to zero away from the data.", "tab:hybrid"))
    rm, al, h = tot["loto"]
    a = h[h.procedure == "PPK-FIP+GP"].set_index(["case", "fold", "row", "species"]).se; b = h[h.procedure == "PPK-FIP"].set_index(["case", "fold", "row", "species"]).se
    from scipy.stats import wilcoxon as _w
    mac("HybLotoAll", f(al["PPK-FIP+GP"], 3)); mac("HybLotoBase", f(al["PPK-FIP"], 3)); mac("HybLotoP", pfmt(_w(a, b.loc[a.index]).pvalue))
    mac("HybLotoMao", f(rm.loc["Mao123", "PPK-FIP+GP"], 3)); mac("HybLotoOrigGP", f(al["ORIG+GP"], 3))
    if "fcst" in tot: mac("HybFcstAll", f(tot["fcst"][1]["PPK-FIP+GP"], 3)); mac("HybFcstBase", f(tot["fcst"][1]["PPK-FIP"], 3))
if ex(f"{R}/cv/loto_hybrid.csv.gz"):
    _h = pd.read_csv(f"{R}/cv/loto_hybrid.csv.gz"); _h["se"] = _h.err ** 2; _al = _h.groupby("procedure").se.mean() ** .5
    mac("HybLotoRedOrig", f(100 * (1 - _al["PPK-FIP+GP"] / _al["ORIG"]), 1))
    _k = ["case", "fold", "row", "species"]; _A = _h[_h.procedure == "PPK-FIP+GP"].set_index(_k).se
    from scipy.stats import wilcoxon as _w
    _B = _h[_h.procedure == "ORIG"].set_index(_k).se.loc[_A.index]; mac("HybLotoPOrig", pfmt(_w(_A, _B).pvalue))
    _B = _h[_h.procedure == "ORIG+GP"].set_index(_k).se.loc[_A.index]; mac("HybLotoPOrigGP", pfmt(_w(_A, _B).pvalue))
    _rm = _h.groupby(["case", "procedure"]).se.mean().pow(.5).unstack()
    mac("HybWinsOrig", int((_rm["PPK-FIP+GP"] < _rm["ORIG"]).sum()))
    if ex(f"{R}/cv/fcst_hybrid.csv.gz"):
        _f = pd.read_csv(f"{R}/cv/fcst_hybrid.csv.gz"); _f["se"] = _f.err ** 2; _fa = _f.groupby("procedure").se.mean() ** .5
        mac("HybFcstOrig", f(_fa["ORIG"], 3))
# fold-level (conservative) tests: per-fold MSE, 145 folds
if ex(f"{R}/cv/loto_procedures.csv.gz"):
    from scipy.stats import wilcoxon as _w
    _e = pd.read_csv(f"{R}/cv/loto_procedures.csv.gz"); _e["se"] = _e.err ** 2
    _f = _e.groupby(["case", "fold", "procedure"]).se.mean().unstack()
    mac("FoldN", len(_f))
    for _r, _k in (("ORIG", "Orig"), ("ORIG-full", "Full")):
        mac(f"FoldWins{_k}", int((_f["FIP-BMA-meta"] < _f[_r]).sum())); mac(f"FoldP{_k}", f"{_w(_f['FIP-BMA-meta'], _f[_r]).pvalue:.1e}".replace("e-0", "\\times10^{-").replace("e-", "\\times10^{-") + "}")
    if ex(f"{R}/cv/loto_hybrid.csv.gz"):
        _g = pd.read_csv(f"{R}/cv/loto_hybrid.csv.gz"); _g["se"] = _g.err ** 2; _g = _g.groupby(["case", "fold", "procedure"]).se.mean().unstack()
        mac("FoldHybWinsOrig", int((_g["PPK-FIP+GP"] < _g["ORIG"]).sum()))
