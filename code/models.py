"""Generalised semi-mechanistic family used by the new method (and its ablations).
Composition path in c (degree of reaction) with optional pseudo-processivity (pM, pD); time follows from the
shared Ping Pong denominator exactly:  kE * tau(c) = J0(c) + FA*J1 + FAZ*JZ + FAM*JM + FAD*JD,
  Sigma = dc/(r dt) = Z(1+pM+pM pD) + SM M (1+pD) + SD D,  S = Z + SM M + SD D,
  J0 = int S/((A0-c) Sigma) dc,  J1 = int 1/Sigma,  JX = int X/Sigma.
Enzyme deactivation (first order, kd) maps real time t to active time tau = (1-exp(-kd t))/kd.
With pM=pD=0 and kd=0 this is exactly Eqs (30)-(34) of the original paper."""
import numpy as np
from functools import lru_cache
from scipy.integrate import solve_ivp

def _rhs(c, y, SM, SD, pM, pD, A0):
    Z, M, D, T = y[:4]
    S = Z + SM * M + SD * D
    Sig = max(Z * (1 + pM + pM * pD) + SM * M * (1 + pD) + SD * D, 1e-300)
    dZ = -Z; dM = Z * (1 - pM) - SM * M
    dD = Z * pM * (1 - pD) + SM * M * (1 - pD) - SD * D
    dT = Z * pM * pD + SM * M * pD + SD * D
    A = max(A0 - c, 1e-300)
    return [dZ / Sig, dM / Sig, dD / Sig, dT / Sig, S / (A * Sig), 1 / Sig, Z / Sig, M / Sig, D / Sig]

import numba

@numba.njit(cache=True)
def _f(c, y, SM, SD, pM, pD, A0, out):
    Z, M, D = y[0], y[1], y[2]
    S = Z + SM * M + SD * D
    Sig = Z * (1 + pM + pM * pD) + SM * M * (1 + pD) + SD * D
    if Sig < 1e-300: Sig = 1e-300
    A = A0 - c
    if A < 1e-300: A = 1e-300
    out[0] = -Z / Sig
    out[1] = (Z * (1 - pM) - SM * M) / Sig
    out[2] = (Z * pM * (1 - pD) + SM * M * (1 - pD) - SD * D) / Sig
    out[3] = (Z * pM * pD + SM * M * pD + SD * D) / Sig
    out[4] = S / (A * Sig); out[5] = 1 / Sig; out[6] = Z / Sig; out[7] = M / Sig; out[8] = D / Sig

@numba.njit(cache=True)
def _rk4_path(cg, SM, SD, pM, pD, A0):
    n = cg.shape[0]
    Y = np.zeros((9, n)); y = np.zeros(9); y[0] = 100.0
    k1 = np.zeros(9); k2 = np.zeros(9); k3 = np.zeros(9); k4 = np.zeros(9); tmp = np.zeros(9)
    Y[:, 0] = y
    for i in range(n - 1):
        h = cg[i + 1] - cg[i]; c = cg[i]
        _f(c, y, SM, SD, pM, pD, A0, k1)
        for j in range(9): tmp[j] = y[j] + 0.5 * h * k1[j]
        _f(c + 0.5 * h, tmp, SM, SD, pM, pD, A0, k2)
        for j in range(9): tmp[j] = y[j] + 0.5 * h * k2[j]
        _f(c + 0.5 * h, tmp, SM, SD, pM, pD, A0, k3)
        for j in range(9): tmp[j] = y[j] + h * k3[j]
        _f(c + h, tmp, SM, SD, pM, pD, A0, k4)
        for j in range(9):
            y[j] += h / 6.0 * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j])
        for j in range(4):
            if y[j] < 0: y[j] = 0.0
        Y[:, i + 1] = y
    return Y

@lru_cache(maxsize=256)
def path(SM, SD, pM, pD, A0, cend=299.99999, n=6000):
    """numba RK4 on a grid refined towards completion (S -> 0); validated against LSODA (path_lsoda)."""
    u = np.linspace(0, 1, n)
    cg = cend * (1 - (1 - u) ** 3)
    return cg, _rk4_path(cg, SM, SD, pM, pD, A0)

def path_lsoda(SM, SD, pM, pD, A0, cend=299.99999, n=6000):
    u = np.linspace(0, 1, n)
    cg = cend * (1 - (1 - u) ** 2.5)
    sol = solve_ivp(_rhs, (0, cend), [100, 0, 0, 0, 0, 0, 0, 0, 0], args=(SM, SD, pM, pD, A0), t_eval=cg,
                    method="LSODA", rtol=1e-9, atol=1e-11)
    Y = sol.y
    if Y.shape[1] < n:
        k = Y.shape[1]; cg = cg[:k]
    return cg, Y

def comp_at_c(c, SM, SD, pM=0.0, pD=0.0, A0=300.0):
    cg, Y = path(float(SM), float(SD), float(pM), float(pD), float(A0))
    return np.stack([np.interp(c, cg, Y[i]) for i in range(4)], 1)

def predict(t, A0, kE, SM, SD, FA=0., FAZ=0., FAM=0., FAD=0., pM=0., pD=0., kd=0.):
    t = np.asarray(t, float)
    tau = t if kd <= 1e-12 else -np.expm1(-kd * t) / kd
    cg, Y = path(float(SM), float(SD), float(pM), float(pD), float(A0))
    tc = Y[4] + FA * Y[5] + FAZ * Y[6] + FAM * Y[7] + FAD * Y[8]
    tc = np.maximum.accumulate(np.nan_to_num(tc, nan=1e300, posinf=1e300))
    c = np.interp(kE * tau, tc, cg)
    return np.stack([np.interp(c, cg, Y[i]) for i in range(4)], 1)
