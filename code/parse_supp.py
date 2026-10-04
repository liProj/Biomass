"""Parse every reaction-profile table of the Supplementary Material of Nishimura et al., Biomass 2026, 6, 12
into tidy CSVs (data/profiles.csv) - no manual typing of numbers."""
import re, os, pymupdf, pandas as pd, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); JD = os.path.dirname(HERE)
PDF = f"{JD}/supp/biomass-06-00012-s001/biomass-4043327-supplementary.pdf"
lines = []
for i, p in enumerate(pymupdf.open(PDF)):
    L = [l.strip() for l in p.get_text().split("\n")]
    # drop the page-number line that opens every page
    if L and L[0] == str(i + 1): L = L[1:]
    lines += L
num = re.compile(r"^-?\d+(\.\d+)?$")
tables = {}
cur = None
for l in lines:
    m = re.match(r"^Table (S\d+)\.", l)
    if m:
        cur = m.group(1); tables[cur] = []; continue
    if l.startswith("*") or l.startswith("S2.") or l.startswith("S3.") or l.startswith("The "):
        cur = None if (l.startswith("S") or l.startswith("The ")) else cur
        continue
    if cur and num.match(l):
        tables[cur].append(float(l))

META = {  # table -> (dataset id, source, columns)
    "S1": ("Akerman70", "Akerman et al. 70 C (oleic acid, CALB, 3:1)", ["t", "c", "Z", "M", "D", "T"]),
    "S2": ("Akerman100", "Akerman et al. 100 C (oleic acid, CALB, 3:1)", ["t", "c", "Z", "M", "D", "T"]),
    "S3": ("Bornadel", "Bornadel et al. exp 1 (oleic acid, CALB, 3:1)", ["t", "c", "Z", "M", "D", "T"]),
    "S4": ("Tao025", "Tao et al. aw=0.25 (caprylic acid, Candida sp. 99-125, 3.2:1)", ["t", "c", "Z", "M", "D", "T"]),
    "S5": ("Tao035", "Tao et al. aw=0.35", ["t", "c", "Z", "M", "D", "T"]),
    "S6": ("Tao045", "Tao et al. aw=0.45", ["t", "c", "Z", "M", "D", "T"]),
    "S17": ("Mao1", "Mao et al. batch 1 (pelargonic acid, Candida sp. 99-125, 3:1)", ["t", "tcum", "c", "M", "D", "T"]),
    "S18": ("Mao2", "Mao et al. batch 2", ["t", "tcum", "c", "M", "D", "T"]),
    "S19": ("Mao3", "Mao et al. batch 3", ["t", "tcum", "c", "M", "D", "T"]),
    "S20": ("Mao4", "Mao et al. batch 4 (excluded by authors)", ["t", "tcum", "c", "M", "D", "T"]),
    "S21": ("Mao5", "Mao et al. batch 5 (excluded by authors)", ["t", "tcum", "c", "M", "D", "T"]),
}
rows = []
for tab, (ds, src, cols) in META.items():
    v = tables[tab]
    k = len(cols)
    assert len(v) % k == 0, (tab, len(v))
    a = np.array(v).reshape(-1, k)
    d = pd.DataFrame(a, columns=cols)
    if "Z" not in d: d["Z"] = np.nan
    if "tcum" not in d: d["tcum"] = d["t"]
    d.insert(0, "dataset", ds); d.insert(1, "table", tab); d["source"] = src
    rows.append(d)
    print(f"{tab:4s} {ds:11s} rows={len(d):3d} t:[{d.t.min():g},{d.t.max():g}]")
P = pd.concat(rows, ignore_index=True)[["dataset", "table", "t", "tcum", "c", "Z", "M", "D", "T", "source"]]
os.makedirs(f"{JD}/data", exist_ok=True)
P.to_csv(f"{JD}/data/profiles.csv", index=False)
print("saved", len(P), "rows")
