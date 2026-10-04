"""Generic parameter-spec fitter for the model family in models.py.
A spec maps parameter name -> ('fix', value) | ('free', init, lo, hi, log) ; per-dataset parameters are written
'kE@Mao1'.  Optional 'link' functions express hierarchical structure (e.g. kE of batch b = kE0*exp(-delta*(b-1)))."""
import numpy as np, pandas as pd
from scipy.optimize import least_squares
from models import predict
from kinetics import load_profiles, A0 as A0TAB, SPECIES

P = load_profiles()
import os as _os
MAXNFEV = int(_os.environ.get("FIT_MAXNFEV", 3000))
MECH = ["kE", "SM", "SD", "FA", "FAZ", "FAM", "FAD", "pM", "pD", "kd"]
DEFAULT = dict(FA=0.0, FAZ=0.0, FAM=0.0, FAD=0.0, pM=0.0, pD=0.0, kd=0.0)

class Model:
    def __init__(self, dsets, spec, link=None, A0=None, rows=None):
        self.dsets = dsets; self.spec = spec; self.link = link
        self.A0 = A0 or {d: A0TAB[d] for d in dsets}
        self.free = [k for k, v in spec.items() if v[0] == "free"]
        self.blocks = {}
        for d in dsets:
            b = P[(P.dataset == d) & (P.t > 0)]
            if rows is not None: b = b[b.index.isin(rows)]
            self.blocks[d] = b

    def params_for(self, theta, d):
        vals = {}
        named = dict(zip(self.free, theta))
        for k, v in self.spec.items():
            vals[k] = named[k] if v[0] == "free" else v[1]
        out = dict(DEFAULT)
        for k, v in vals.items():
            if "@" in k:
                base, ds = k.split("@")
                if ds == d: out[base] = v
            else:
                out[k] = v
        if self.link: out = self.link(out, d, vals)
        return out

    def pred(self, theta, d, t):
        p = self.params_for(theta, d)
        return predict(t, self.A0[d], p["kE"], p["SM"], p["SD"], p["FA"], p["FAZ"], p["FAM"], p["FAD"], p["pM"], p["pD"], p["kd"])

    def resid(self, theta, blocks=None):
        blocks = blocks or self.blocks
        out = []
        for d, b in blocks.items():
            if len(b) == 0: continue
            pr = self.pred(theta, d, b.t.to_numpy())
            o = b[SPECIES].to_numpy(float); m = ~np.isnan(o)
            out.append((pr - o)[m])
        return np.concatenate(out)

    def _tx(self):
        lo, hi, lg, x0 = [], [], [], []
        for k in self.free:
            _, init, l, h, isl = self.spec[k]
            lg.append(isl)
            lo.append(np.log(l) if isl else l); hi.append(np.log(h) if isl else h)
            x0.append(np.log(init) if isl else init)
        return np.array(lo), np.array(hi), np.array(lg), np.array(x0)

    def to_theta(self, x):
        lo, hi, lg, _ = self._tx()
        return np.where(lg, np.exp(np.clip(x, -700, 700)), x)

    def fit(self, n_starts=12, seed=0, x0s=None):
        lo, hi, lg, x0 = self._tx()
        rng = np.random.default_rng(seed)
        starts = [np.clip(x0, lo + 1e-9, hi - 1e-9)] if x0s is None else list(x0s)
        for _ in range(n_starts - 1):
            starts.append(np.clip(x0 + rng.normal(0, 1.0, len(x0)) * np.where(lg, 1.5, 0.15 * (hi - lo)), lo + 1e-9, hi - 1e-9))
        best = None
        f = lambda x: self.resid(self.to_theta(x))
        for s in starts:
            try:
                r = least_squares(f, s, bounds=(lo, hi), method="trf", x_scale="jac", xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=MAXNFEV)
            except Exception:
                continue
            if best is None or r.cost < best.cost: best = r
        self.x = best.x; self.theta = self.to_theta(best.x)
        res = self.resid(self.theta)
        self.F = float(res @ res); self.nT = len(res); self.k = len(self.free)
        self.BIC = self.nT * np.log(self.F / self.nT) + self.k * np.log(self.nT)
        return self

def bic(F, nT, k): return nT * np.log(F / nT) + k * np.log(nT)
