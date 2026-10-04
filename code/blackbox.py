"""Black-box / data-driven baselines on the same folds as cv.py:
  GP      : Gaussian process (Matern-5/2 + white noise) per species on (t, condition)
  MLP-ens : 5-seed MLP ensemble per species
  TabICL  : pretrained tabular foundation model (in-context learning, jingang/TabICL regressor v2) per species
  NODE    : black-box neural ODE  dY/dt = f_theta(Y/100, condition) trained through an RK4 solver on GPU
Condition features: Tao -> water activity, Mao -> batch index, others -> none."""
import sys, os, time, warnings, numpy as np, pandas as pd, torch
warnings.filterwarnings("ignore")
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.neural_network import MLPRegressor
from methods import CASES, all_rows, P, SPECIES, JD
from cv import folds

COND = {"Tao025": 0.25, "Tao035": 0.35, "Tao045": 0.45, "Mao1": 1, "Mao2": 2, "Mao3": 3}
DEV = "cpu"   # tiny networks: kernel-launch bound on GPU, ~40x faster on one CPU core
torch.set_num_threads(1)

def feats(R, tmax):
    X = [R.t.to_numpy() / tmax]
    if R.dataset.iloc[0] in COND or any(d in COND for d in R.dataset.unique()):
        X.append(R.dataset.map(COND).to_numpy(float))
    return np.stack(X, 1)

def fit_predict_static(kind, train, test, tmax):
    Xtr, Xte = feats(train, tmax), feats(test, tmax)
    out = np.full((len(test), 4), np.nan)
    for s, sp in enumerate(SPECIES):
        y = train[sp].to_numpy(float); m = ~np.isnan(y)
        if m.sum() < 3: continue
        # anchor the known initial state (t=0: Z=100, others 0) as a training point, as the mechanistic models know it
        X0 = np.zeros((1, Xtr.shape[1])); X0[0, 1:] = Xtr[0, 1:] if Xtr.shape[1] > 1 else 0
        Xa = np.vstack([Xtr[m], feats(P[(P.t == 0) & P.dataset.isin(train.dataset.unique())], tmax)])
        ya = np.r_[y[m], np.full(len(Xa) - m.sum(), 100.0 if sp == "Z" else 0.0)]
        if kind == "GP":
            k = C(1000.0) * Matern(length_scale=[0.3] * Xa.shape[1], nu=2.5) + WhiteKernel(4.0)
            g = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=3, random_state=0).fit(Xa, ya)
            p = g.predict(Xte)
        elif kind == "MLP-ens":
            p = np.mean([MLPRegressor(hidden_layer_sizes=(32, 32), activation="tanh", max_iter=5000, alpha=1e-3, random_state=sd,
                                      learning_rate_init=3e-3).fit(Xa, ya / 100).predict(Xte) * 100 for sd in range(5)], 0)
        elif kind == "TabICL":
            from tabicl import TabICLRegressor
            p = TabICLRegressor(random_state=0).fit(Xa, ya).predict(Xte)
        out[:, s] = np.clip(p, 0, 100)
    return out

class F(torch.nn.Module):
    def __init__(s, nc):
        super().__init__(); s.net = torch.nn.Sequential(torch.nn.Linear(4 + nc, 64), torch.nn.Tanh(), torch.nn.Linear(64, 64),
                                                      torch.nn.Tanh(), torch.nn.Linear(64, 4))
    def forward(s, y, c): return s.net(torch.cat([y, c], -1))

def node_fit_predict(train, test, tmax, seed=0, epochs=1500):
    torch.manual_seed(seed)
    dsets = sorted(set(train.dataset) | set(test.dataset))
    nc = 1 if any(d in COND for d in dsets) else 0
    f = F(nc).to(DEV); opt = torch.optim.Adam(f.parameters(), 3e-3)
    nstep = 60; tg = torch.linspace(0, 1, nstep + 1, device=DEV); h = 1.0 / nstep
    def integrate(cv):
        y = torch.tensor([[1.0, 0, 0, 0]], device=DEV).repeat(len(cv), 1); ys = [y]
        for i in range(nstep):
            k1 = f(y, cv); k2 = f(y + h / 2 * k1, cv); k3 = f(y + h / 2 * k2, cv); k4 = f(y + h * k3, cv)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4); ys.append(y)
        return torch.stack(ys, 1)  # (nd, nstep+1, 4)
    cvec = torch.tensor([[COND.get(d, 0.0)] for d in dsets] if nc else [[]] * len(dsets), device=DEV, dtype=torch.float32)
    def gather(traj, R):
        di = torch.tensor([dsets.index(d) for d in R.dataset], device=DEV)
        pos = torch.tensor(R.t.to_numpy() / tmax * nstep, device=DEV, dtype=torch.float32).clamp(0, nstep)
        lo = pos.floor().long().clamp(max=nstep - 1); w = (pos - lo).unsqueeze(1)
        return (1 - w) * traj[di, lo] + w * traj[di, lo + 1]
    ytr = torch.tensor(train[SPECIES].to_numpy(float) / 100, device=DEV, dtype=torch.float32); mtr = ~torch.isnan(ytr)
    best = (1e9, None)
    for ep in range(epochs):
        opt.zero_grad(); pr = gather(integrate(cvec), train)
        loss = ((pr - torch.nan_to_num(ytr))[mtr] ** 2).mean(); loss.backward(); opt.step()
        if loss.item() < best[0]: best = (loss.item(), {k: v.clone() for k, v in f.state_dict().items()})
    f.load_state_dict(best[1])
    with torch.no_grad():
        return np.clip(gather(integrate(cvec), test).cpu().numpy() * 100, 0, 100)

def run(case, proto, test, kinds):
    rows = all_rows(case) - set(test)
    tr, te = P.loc[sorted(rows)], P.loc[sorted(test)]
    tmax = P[P.dataset.isin(CASES[case])].t.max()
    recs = []
    for kind in kinds:
        pr = node_fit_predict(tr, te, tmax) if kind == "NODE" else fit_predict_static(kind, tr, te, tmax)
        o = te[SPECIES].to_numpy(float)
        for j in range(len(te)):
            for s, sp in enumerate(SPECIES):
                if not np.isnan(o[j, s]):
                    recs.append(dict(case=case, proto=proto, method=kind, row=int(te.index[j]), dataset=te.dataset.iloc[j],
                                     t=float(te.t.iloc[j]), species=sp, obs=o[j, s], pred=pr[j, s]))
    return recs

if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    proto = sys.argv[1]; kinds = sys.argv[2].split(","); nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    jobs = [(c, proto, fo, kinds) for c in CASES for fo in folds(c, proto)]
    recs = []; t0 = time.time()
    with ProcessPoolExecutor(nproc) as ex:
        for r in ex.map(run, *zip(*jobs)):
            recs += r
    print(f"{len(jobs)} folds {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(recs); tag = "_".join(kinds)
    df.to_csv(f"{JD}/results/cv/{proto}_bb_{tag}.csv", index=False)
    df["se"] = (df.pred - df.obs) ** 2
    print(df.groupby(["case", "method"]).se.mean().pow(0.5).unstack().round(3).to_string())
