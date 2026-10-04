"""Fig 3 reproduction audit; Fig 4 estimation landscape + profile (identifiability); Fig 5 solver benchmark."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from plotstyle import *
from kinetics import JD
setup()
R = pd.read_csv(f"{JD}/results/repro_compare.csv")
CL = {"Akerman70": "Åk70", "Akerman100": "Åk100", "Bornadel": "Born.", "Tao025": "Tao.25", "Tao035_045": "Tao.35/.45", "Mao123": "Mao1-3",
      "Akerman": "Åkerman", "Tao": "Tao", "Mao": "Mao"}

def fig_repro():
    fig, axs = plt.subplots(1, 3, figsize=(8.0, 2.9), gridspec_kw=dict(wspace=0.55, width_ratios=[1, 1.25, 1.25]))
    a = axs[0]
    fp = R[(R.model == "fingerprint") & R.item.isin(["SM", "SD", "SS"])]
    a.loglog(fp.paper, fp.refit, "o", color=SLOTS[0], ms=5, label="fingerprint SM, SD, SS")
    sm = R[(R.model == "simplified") & R.case.isin(["Akerman70", "Akerman100", "Bornadel"]) & R.item.isin(["Fobj", "FAZ", "FAM", "FAD"])]
    a.loglog(sm.paper, sm.refit, "s", color=SLOTS[1], ms=5, label="simplified dynamic (Åk., Born.)")
    lim = [1e-3, 3e3]; a.plot(lim, lim, color=INK2, lw=0.8, ls="--"); a.set_xlim(lim); a.set_ylim(lim)
    a.set_xlabel("reported value"); a.set_ylabel("re-estimated value"); a.set_title("(a) Fit agreement", loc="left")
    a.legend(loc="lower right", fontsize=6.2)
    a = axs[1]
    F = R[R.item == "Fobj"].copy(); F["lab"] = F.case.map(CL) + np.where(F.model == "full", " F", " S")
    x = np.arange(len(F))
    a.bar(x - 0.27, F.paper, 0.26, color=SLOTS[1], label="reported")
    a.bar(x, F.at_paper_point, 0.26, color=SLOTS[3], label="at reported parameters")
    a.bar(x + 0.27, F.refit, 0.26, color=SLOTS[0], label="best multistart fit")
    a.set_yscale("log"); a.set_xticks(x); a.set_xticklabels(F.lab, rotation=70, fontsize=6.2); a.set_ylabel("Fobj (molar %²)")
    a.set_title("(b) Dynamic fit comparison", loc="left"); a.legend(fontsize=6.2, loc="upper left")
    a = axs[2]
    E = R[R.item.str.startswith("enzyme")].copy(); E["lab"] = E.case.map(CL) + np.where(E.model == "full", " F", " S")
    a.scatter(np.arange(len(E)), E.at_paper_point, color=SLOTS[0], s=22, zorder=3)
    for v in (0.1, 2, 5, 15): a.axhline(v, color=GRID, lw=0.8, zorder=1)
    a.set_yscale("log"); a.set_xticks(np.arange(len(E))); a.set_xticklabels(E.lab, rotation=70, fontsize=6.2)
    a.set_ylabel("implied loading $E=k_E/k_A$"); a.set_title("(c) Implied loading", loc="left")
    save(fig, "fig03_reproduction")

def fig_landscape():
    L = pd.read_csv(f"{JD}/results/diagnostics/landscape.csv"); Pf = pd.read_csv(f"{JD}/results/diagnostics/profile_FA.csv")
    full = R[(R.item == "Fobj") & (R.model == "full")].set_index("case")
    cases = ["Akerman70", "Akerman100", "Bornadel", "Tao025", "Tao035_045", "Mao123"]
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw=dict(wspace=0.3, width_ratios=[1.4, 1]))
    a = axs[0]
    for i, c in enumerate(cases):
        g = L[L.case == c]; best = g.Fobj.min(); y = np.log10(g.Fobj / best)
        jit = np.random.default_rng(i).uniform(-0.18, 0.18, len(y))
        a.scatter(i + jit, np.clip(y, 0, 3), s=6, color=SLOTS[0], alpha=0.35, lw=0)
        a.scatter(i, np.log10(full.loc[c, "paper"] / best), marker="D", s=34, color=SLOTS[1], zorder=4, edgecolor="white", lw=0.6)
        frac = np.mean(g.Fobj <= best * 1.001)
        a.text(i, 3.12, f"{100*frac:.0f}%", ha="center", fontsize=6.5, color=INK2)
    a.set_xticks(range(len(cases))); a.set_xticklabels([CL[c] for c in cases], fontsize=7)
    a.set_ylabel("log$_{10}$(Fobj / best found)"); a.set_ylim(-0.1, 3.3)
    a.set_title("(a) 200 random starts, initial full model", loc="left")
    a.scatter([], [], s=12, color=SLOTS[0], label="local optimum reached"); a.scatter([], [], marker="D", color=SLOTS[1], label="reported objective")
    a.legend(loc="center right", fontsize=6.5)
    a = axs[1]
    for c, col_ in zip(["Akerman70", "Akerman100", "Tao025"], [SLOTS[0], SLOTS[1], SLOTS[2]]):
        g = Pf[Pf.case == c]
        a.plot(g.logFA / np.log(10), g.Fobj / g.Fobj.min(), color=col_, label=CL[c])
    a.set_yscale("log"); a.set_xlabel("log$_{10}$ $F_A$ (fixed)"); a.set_ylabel("profile Fobj / min")
    a.set_title("(b) Profile along $F_A$", loc="left"); a.legend(fontsize=6.5)
    save(fig, "fig04_landscape_identifiability")

def fig_bench():
    B = pd.read_csv(f"{JD}/results/diagnostics/solver_bench.csv")
    fig, a = plt.subplots(figsize=(4.6, 2.6))
    c = [SLOTS[0] if s.startswith("exact") else SLOTS[1] for s in B.solver]
    a.barh(range(len(B)), B.ms, color=c); a.set_xscale("log")
    a.set_yticks(range(len(B))); a.set_yticklabels(B.solver, fontsize=6.6); a.invert_yaxis()
    for i, (m, e) in enumerate(zip(B.ms, B.max_err)):
        a.text(m * 1.15, i, f"{m:.2g} ms, max err {e:.0e}", va="center", fontsize=6.2, color=INK2)
    a.set_xlabel("wall time per model evaluation (ms, log scale)"); a.set_xlim(0.3, 2e4)
    a.set_title("Solver cost (Bornadel profile, processive model)", loc="left")
    save(fig, "fig05_solver_benchmark")

if __name__ == "__main__":
    fig_repro(); fig_landscape(); fig_bench()
