"""Shared figure style (validated categorical palette, fixed slot order - colour follows the entity)."""
import os, matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
# entity -> fixed slot
METHOD_COL = {"PPK-FIP": SLOTS[0], "ORIG": SLOTS[1], "ORIG-full": SLOTS[3], "JOINT-BIC": SLOTS[2], "JOINT-BMA": SLOTS[5],
              "FIP-meta": SLOTS[6], "GP": SLOTS[4], "MLP-ens": "#8a8985", "TabICL": SLOTS[7], "NODE": SLOTS[6]}
SPEC_COL = {"Z": SLOTS[0], "M": SLOTS[1], "D": SLOTS[2], "T": SLOTS[3]}   # validated adjacent order; markers add secondary encoding
SPEC_MK = {"Z": "o", "M": "s", "D": "^", "T": "D"}
def setup():
    mpl.rcParams.update({"figure.dpi": 130, "savefig.dpi": 330, "savefig.bbox": "tight", "font.family": "DejaVu Sans",
        "font.size": 8.5, "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.labelsize": 8.5, "axes.edgecolor": INK2,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "legend.frameon": False, "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "lines.linewidth": 1.6, "lines.markersize": 4.5})
def col(m): return METHOD_COL.get(m, "#8a8985")
FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("png", "pdf"): fig.savefig(f"{FIG}/{name}.{ext}")
    plt.close(fig); print("  figure ->", name)
