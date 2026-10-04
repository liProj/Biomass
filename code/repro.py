"""Reproduce every fitted quantity reported in Nishimura et al. (Biomass 2026, 6, 12):
fingerprinting fits (SM, SD, SS), full and simplified dynamic fits (Tables 1-6: kA*[E]T, FA..FAD, Fobj, BIC),
95% CIs and parameter correlation matrices (Tables S22-S36).  Output: results/baseline/*.json|csv"""
import json, itertools, sys, os, numpy as np, pandas as pd
from kinetics import *

OUT = f"{JD}/results/baseline"; os.makedirs(OUT, exist_ok=True)
P = load_profiles()
FP_GROUPS = {"Akerman": ["Akerman70", "Akerman100"], "Bornadel": ["Bornadel"],
             "Tao": ["Tao025", "Tao035", "Tao045"], "Mao": ["Mao1", "Mao2", "Mao3"]}
DYN_CASES = [  # name, datasets, fp group, simplified-model zeroed params, batch-specific kE?
    ("Akerman70", ["Akerman70"], "Akerman", ["FA", "FAZ"], False),
    ("Akerman100", ["Akerman100"], "Akerman", ["FA", "FAZ"], False),
    ("Bornadel", ["Bornadel"], "Bornadel", ["FA"], False),
    ("Tao025", ["Tao025"], "Tao", ["FA", "FAD"], False),
    ("Tao035_045", ["Tao035", "Tao045"], "Tao", ["FA", "FAD"], False),
    ("Mao123", ["Mao1", "Mao2", "Mao3"], "Mao", ["FA", "FAD"], True),
]
FNAMES = ["FA", "FAZ", "FAM", "FAD"]

def fp_resid_factory(dsets):
    blocks = [(P[(P.dataset == d) & (P.t > 0)]) for d in dsets]
    def f(th):
        SM, SD = th
        res = []
        for b in blocks:
            pred = fp_at_c(b.c.to_numpy(), SM, SD)
            obs = obs_matrix(b)
            m = ~np.isnan(obs)
            res.append((pred - obs)[m])
        return np.concatenate(res)
    return f

def fit_fp(group):
    f = fp_resid_factory(FP_GROUPS[group])
    g = lambda lth: f(np.exp(lth))
    starts = [np.log([a, b]) for a in (0.2, 0.5, 1.0, 2.0) for b in (0.01, 0.05, 0.3, 1.0)]
    r = multistart_lsq(g, starts, lb=np.log([1e-4, 1e-5]), ub=np.log([50, 50]))
    th = np.exp(r.x); res = f(th); F = float(res @ res); nT = len(res)
    hw, corr, _ = wls_ci(f, th, nT)
    return dict(group=group, SM=th[0], SD=th[1], SM_ci=hw[0], SD_ci=hw[1], SS=F, nT=nT, corr=corr.tolist())

def dyn_resid_factory(dsets, SM, SD, free, batch_kE, extra_factory=None):
    blocks = [P[(P.dataset == d) & (P.t > 0)] for d in dsets]
    nk = len(dsets) if batch_kE else 1
    def unpack(th):
        kEs = th[:nk]; rest = dict(zip(free, th[nk:]))
        Fp = {n: rest.get(n, 0.0) for n in FNAMES}
        return kEs, Fp
    def f(th):
        kEs, Fp = unpack(th)
        res = []
        for i, (d, b) in enumerate(zip(dsets, blocks)):
            kE = kEs[i if batch_kE else 0]
            pred, _ = dyn_fast(b.t.to_numpy(), A0[d], kE, SM, SD, Fp["FA"], Fp["FAZ"], Fp["FAM"], Fp["FAD"])
            obs = obs_matrix(b); m = ~np.isnan(obs)
            res.append((pred - obs)[m])
        return np.concatenate(res)
    return f, nk

def fit_dyn(dsets, SM, SD, free, batch_kE, n_starts=None):
    f, nk = dyn_resid_factory(dsets, SM, SD, free, batch_kE)
    g = lambda lth: f(np.exp(lth))
    k0 = [0.01, 0.05, 0.2, 1.0, 5.0]
    F0 = [1e-6, 1e-4, 1e-3, 1e-2, 0.1]
    rng = np.random.default_rng(0)
    starts = []
    for kk in k0:
        for v in F0:
            starts.append(np.log([kk] * nk + [v] * len(free)))
    for _ in range(40):
        starts.append(np.log(np.r_[10 ** rng.uniform(-3, 1, nk), 10 ** rng.uniform(-8, 0, len(free))]))
    lb = np.log([1e-6] * nk + [1e-14] * len(free)); ub = np.log([1e3] * nk + [1e3] * len(free))
    r = multistart_lsq(g, starts, lb=lb, ub=ub)
    th = np.exp(r.x); res = f(th); F = float(res @ res); nT = len(res); k = len(th)
    hw, corr, _ = wls_ci(f, th, nT)
    names = ([f"kE_{d}" for d in dsets] if batch_kE else ["kE"]) + free
    return dict(params=dict(zip(names, th.tolist())), ci=dict(zip(names, hw.tolist())), Fobj=F, nT=nT, k=k,
                BIC=bic(F, nT, k), corr=corr.tolist(), names=names, theta=th.tolist())

if __name__ == "__main__":
    fp = {g: fit_fp(g) for g in FP_GROUPS}
    for g, v in fp.items():
        print(f"FP {g:9s} SM={v['SM']:.4f}±{v['SM_ci']:.4f} SD={v['SD']:.5f}±{v['SD_ci']:.5f} SS={v['SS']:.1f} nT={v['nT']}")
    json.dump(fp, open(f"{OUT}/fingerprint.json", "w"), indent=1)
    dyn = {}
    for name, dsets, grp, zero, bk in DYN_CASES:
        SM, SD = fp[grp]["SM"], fp[grp]["SD"]
        for variant, free in (("full", FNAMES), ("simplified", [n for n in FNAMES if n not in zero])):
            v = fit_dyn(dsets, SM, SD, free, bk)
            dyn[f"{name}|{variant}"] = v
            ps = " ".join(f"{n}={v['params'][n]:.4g}±{v['ci'][n]:.3g}" for n in v["names"])
            print(f"DYN {name:11s} {variant:10s} Fobj={v['Fobj']:.1f} BIC={v['BIC']:.1f} nT={v['nT']} | {ps}", flush=True)
    json.dump(dyn, open(f"{OUT}/dynamic.json", "w"), indent=1)
