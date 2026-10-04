"""Linearised 95% confidence intervals (same construction as the original paper: s^2 (J'J)^-1, t quantile) for the
PPK-FIP estimates of each F-set variant at the saved full-data MAP solution (prior residuals included in J)."""
import json, numpy as np, pandas as pd
from scipy import stats
import methods
from methods import *
from fitlib import Model
Pr = json.load(open(f"{JD}/results/final/params.json")); HP = json.load(open(f"{JD}/results/cv/meta_hparams.json"))
rows = []
for case in CASES:
    for tag, Fs in (("s", SIMPL[case]), ("f", FULL)):
        s = {}; kE_spec(case, s)
        s["SM"] = ("free", 0.6, 1e-3, 20, True); s["SD"] = ("free", 0.1, 1e-4, 20, True)
        for f_ in Fs: s[f_] = ("free", 3e-3, 1e-12, 1e3, True)
        s["pM"] = ("free", 0.05, 0.0, 0.95, False); s["pD"] = ("free", 0.05, 0.0, 0.95, False); s["kd"] = ("free", 0.01, 1e-8, 2.0, True)
        m = Model(CASES[case], s, rows=all_rows(case))
        p0 = Pr[case][f"FIP{tag}"]
        th = []
        for k in m.free:
            base, d = (k.split("@") + [CASES[case][0]])[:2]
            th.append(p0[d][base])
        th = np.array(th, float)
        r = m.resid(th); n = len(r); k = len(th); s2 = (r @ r) / (n - k)
        J = np.empty((n, k))
        for i in range(k):
            h = 1e-6 * max(abs(th[i]), 1e-9); tp, tm = th.copy(), th.copy(); tp[i] += h; tm[i] = max(tm[i] - h, 0) if m.spec[m.free[i]][4] is False else tm[i] - h
            J[:, i] = (m.resid(tp) - m.resid(tm)) / (tp[i] - tm[i])
        cov = s2 * np.linalg.pinv(J.T @ J); hw = stats.t.ppf(0.975, n - k) * np.sqrt(np.clip(np.diag(cov), 0, None))
        for name, v, w in zip(m.free, th, hw):
            rows.append(dict(case=case, Fset=tag, param=name, estimate=v, ci95=w, rel=w / max(abs(v), 1e-12)))
df = pd.DataFrame(rows); df.to_csv(f"{JD}/results/final/ci.csv", index=False)
print(df[df.param.isin(["SM", "SD", "pM", "pD", "kd"])].round(4).to_string(index=False))
