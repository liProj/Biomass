import json, sys, numpy as np
from fitlib import Model, bic
from concurrent.futures import ProcessPoolExecutor
FP = json.load(open("../results/baseline/fingerprint.json"))
CASES = {"Akerman70": (["Akerman70"], "Akerman", ["FAM", "FAD"]), "Akerman100": (["Akerman100"], "Akerman", ["FAM", "FAD"]),
         "Bornadel": (["Bornadel"], "Bornadel", ["FAZ", "FAM", "FAD"]), "Tao025": (["Tao025"], "Tao", ["FAZ", "FAM"]),
         "Tao035_045": (["Tao035", "Tao045"], "Tao", ["FAZ", "FAM"]), "Mao123": (["Mao1", "Mao2", "Mao3"], "Mao", ["FAZ", "FAM"])}
def spec_for(case, sm_free, simpl, proc, kd, allF=False):
    ds, g, Fs = CASES[case]
    s = {}
    if case == "Mao123":
        for d in ds: s[f"kE@{d}"] = ("free", 0.07, 1e-5, 1e3, True)
    else:
        s["kE"] = ("free", 0.1, 1e-5, 1e3, True)
    s["SM"] = ("free", FP[g]["SM"], 1e-3, 20, True) if sm_free else ("fix", FP[g]["SM"])
    s["SD"] = ("free", FP[g]["SD"], 1e-4, 20, True) if sm_free else ("fix", FP[g]["SD"])
    use = ["FA", "FAZ", "FAM", "FAD"] if allF else Fs
    for f in use: s[f] = ("free", 3e-3, 1e-9, 1e2, True)
    if proc:
        s["pM"] = ("free", 0.1, 0.0, 0.95, False); s["pD"] = ("free", 0.05, 0.0, 0.95, False)
    if kd: s["kd"] = ("free", 0.01, 1e-7, 1.0, True)
    return s
VARIANTS = {"two-step simpl (orig)": (False, True, False, False), "joint simpl": (True, True, False, False),
            "joint+kd": (True, True, False, True), "joint+proc": (True, True, True, False), "joint+proc+kd": (True, True, True, True)}
def run(args):
    case, vname = args
    sm, simpl, proc, kd = VARIANTS[vname]
    m = Model(CASES[case][0], spec_for(case, sm, simpl, proc, kd)).fit(n_starts=16)
    return case, vname, m.F, m.k, m.nT, m.BIC + (0 if sm else 2 * np.log(m.nT)), dict(zip(m.free, np.round(m.theta, 5).tolist()))
if __name__ == "__main__":
    jobs = [(c, v) for c in CASES for v in VARIANTS]
    with ProcessPoolExecutor(18) as ex:
        for case, v, F, k, nT, B, th in ex.map(run, jobs):
            print(f"{case:11s} {v:24s} Fobj={F:8.1f} k={k} nT={nT} BIC_total={B:7.1f} {th}", flush=True)
