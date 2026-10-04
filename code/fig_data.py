"""Fig 1 workflow schematic (original two-step vs PPK-FIP); Fig 2 data overview (time and degree-of-reaction views)."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from plotstyle import *
from kinetics import load_profiles, SPECIES
setup(); P = load_profiles()
LAB = {"Akerman70": "Åkerman 70 °C", "Akerman100": "Åkerman 100 °C", "Bornadel": "Bornadel", "Tao025": "Tao $a_w$=0.25",
       "Tao035": "Tao $a_w$=0.35", "Tao045": "Tao $a_w$=0.45", "Mao1": "Mao batch 1", "Mao2": "Mao batch 2", "Mao3": "Mao batch 3"}

def box(ax, x, y, w, h, txt, fc, ec, fs=7.6, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02", fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, color=INK, fontweight="bold" if bold else "normal", wrap=True)
def arrow(ax, x0, y0, x1, y1, c=INK2):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.0))

def fig_workflow():
    fig, ax = plt.subplots(figsize=(8.2, 4.6)); ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    o, n = "#fbe3d9", "#dbe8f8"
    ax.text(0.25, 0.97, "(a) Initial two-step method", ha="center", fontsize=9, fontweight="bold")
    ax.text(0.75, 0.97, "(b) Proposed PPK-FIP", ha="center", fontsize=9, fontweight="bold")
    box(ax, 0.02, 0.78, 0.44, 0.11, "Profiles Z, M, D, T vs. degree of reaction c", o, SLOTS[1])
    box(ax, 0.02, 0.60, 0.46, 0.11, "Step 1: irreversible fingerprint fit\n→ selectivities $S_M$, $S_D$ (fixed hereafter)", o, SLOTS[1])
    box(ax, 0.02, 0.42, 0.46, 0.11, "Step 2: dynamic ODE fit vs. time\n$k_A$, $F_A$, $F_{AZ}$, $F_{AM}$, $F_{AD}$ (one local search)", o, SLOTS[1])
    box(ax, 0.02, 0.24, 0.46, 0.11, "Simplified model by BIC\n(in-sample Fobj, BIC, Hessian CIs)", o, SLOTS[1])
    for y in (0.78, 0.60, 0.42): arrow(ax, 0.25, y, 0.25, y - 0.07)
    box(ax, 0.52, 0.78, 0.46, 0.11, "Generalised Ping Pong path in c: selectivities\n+ pseudo-processivity ($p_M$, $p_D$)", n, SLOTS[0])
    box(ax, 0.52, 0.60, 0.46, 0.11, "Exact time map: $k_E\\,\\tau(c)=J_0+F_AJ_1+F_{AZ}J_Z+F_{AM}J_M+F_{AD}J_D$\n+ enzyme deactivation $\\tau=(1-e^{-k_d t})/k_d$", n, SLOTS[0], fs=6.6)
    box(ax, 0.52, 0.42, 0.46, 0.11, "One-step MAP: fingerprint-informed prior on $S_M,S_D$\n+ shrinkage priors on $p_M,p_D,k_d$; multistart search", n, SLOTS[0])
    box(ax, 0.52, 0.24, 0.46, 0.11, "BIC-weighted averaging (simplified / full $F$ set)\nhyper-parameters by leave-one-case-out meta-validation", n, SLOTS[0])
    for y in (0.78, 0.60, 0.42): arrow(ax, 0.75, y, 0.75, y - 0.07)
    box(ax, 0.08, 0.03, 0.84, 0.13, "Evaluation: in-sample fit · leave-one-time-point-out CV · forward extrapolation ·\n"
        "cross-condition transfer · Monte-Carlo truth recovery · black-box ML baselines (GP, MLP, TabICL, neural ODE)", "#f1f0ec", INK2, fs=7.2)
    arrow(ax, 0.25, 0.24, 0.40, 0.165); arrow(ax, 0.75, 0.24, 0.60, 0.165)
    ax.annotate("", (0.52, 0.475), (0.48, 0.62), arrowprops=dict(arrowstyle="-|>", color=SLOTS[0], lw=1.0, ls="--"))
    save(fig, "fig01_workflow")

def fig_overview():
    ds = ["Akerman70", "Akerman100", "Bornadel", "Tao025", "Tao035", "Tao045", "Mao1", "Mao2", "Mao3"]
    fig, axs = plt.subplots(3, 6, figsize=(7.6, 6.0), gridspec_kw=dict(hspace=0.75, wspace=0.42))
    for i, d in enumerate(ds):
        g = P[P.dataset == d]
        a1 = axs[i // 3, (i % 3) * 2]; a2 = axs[i // 3, (i % 3) * 2 + 1]
        for sp in SPECIES:
            if g[sp].notna().any():
                a1.plot(g.t, g[sp], marker=SPEC_MK[sp], color=SPEC_COL[sp], lw=0.8, ms=2.6, label=sp)
                a2.plot(g.c, g[sp], marker=SPEC_MK[sp], color=SPEC_COL[sp], lw=0, ms=2.6)
        a1.set_title(LAB[d], fontsize=7.4, loc="left", x=0.0, y=1.04); a1.set_xlabel("t (h)", fontsize=6.5); a2.set_xlabel("c (%)", fontsize=6.5)
        a1.set_ylim(-3, 103); a2.set_ylim(-3, 103); a2.set_xlim(0, 300); a2.set_yticklabels([])
        if i % 3 == 0: a1.set_ylabel("molar %", fontsize=7)
    axs[0, 0].legend(ncol=4, loc="lower left", bbox_to_anchor=(0.0, 1.22), fontsize=7, handlelength=1.2, columnspacing=0.8)
    save(fig, "fig02_data_overview")

if __name__ == "__main__":
    fig_workflow(); fig_overview()
