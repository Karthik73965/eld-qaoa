"""6-bus network model: Y-bus, DC-flow loss formula (B-coefficients) and Newton-Raphson AC load flow.

Bus 1 is the slack, buses 2-3 are PV, buses 4-6 are PQ loads (power factor 0.98).
Generator voltage setpoints are not given in Prada (2017); the Wood & Wollenberg values
V1 = 1.05, V2 = 1.05, V3 = 1.07 pu are used.
"""
from dataclasses import dataclass

import numpy as np

from .data import System

V_SET = {1: 1.05, 2: 1.05, 3: 1.07}


@dataclass
class Network:
    n_bus: int
    base: float
    ybus: np.ndarray
    branches: list
    gen_bus: np.ndarray      # 0-based bus index of each generator
    load_bus: np.ndarray     # 0-based bus index of L4, L5, L6
    vmin: np.ndarray
    vmax: np.ndarray
    pf: float


def build_network(sys: System) -> Network:
    raw = sys.raw
    n = len(raw["buses"])
    Y = np.zeros((n, n), dtype=complex)
    for br in raw["branches"]:
        i, j = br["from"] - 1, br["to"] - 1
        y = 1 / complex(br["r_pu"], br["x_pu"])
        ysh = 1j * br["bc_pu"] / 2
        Y[i, i] += y + ysh
        Y[j, j] += y + ysh
        Y[i, j] -= y
        Y[j, i] -= y
    return Network(
        n_bus=n, base=raw["base_mva"], ybus=Y, branches=raw["branches"],
        gen_bus=np.array([g["bus"] - 1 for g in raw["generators"]]),
        load_bus=np.array([3, 4, 5]),
        vmin=np.array([b["vmin_pu"] for b in raw["buses"]]),
        vmax=np.array([b["vmax_pu"] for b in raw["buses"]]),
        pf=raw["load_power_factor"],
    )


def bus_loads(net: Network, load: dict) -> np.ndarray:
    """Per-bus active load (MW) for one period."""
    Pd = np.zeros(net.n_bus)
    Pd[net.load_bus] = [load["L4"], load["L5"], load["L6"]]
    return Pd


class LossModel:
    """P_L = P^T B P + B0.P + B00 (MW), from DC line flows and line resistances.

    Line flows f = PTDF (Cg P - Pd) with the slack at bus 1; losses = sum_l r_l f_l^2 (per unit, |V| ~ 1).
    Expanding the square gives the Kron loss formula, so B, B0, B00 come straight from the line data.
    """

    def __init__(self, net: Network, Pd_mw: np.ndarray):
        nb, base = net.n_bus, net.base
        A = np.zeros((len(net.branches), nb))           # branch-bus incidence
        for l, br in enumerate(net.branches):
            A[l, br["from"] - 1], A[l, br["to"] - 1] = 1, -1
        xinv = np.diag([1 / br["x_pu"] for br in net.branches])
        Bbus = A.T @ xinv @ A
        ptdf = np.zeros((len(net.branches), nb))
        ptdf[:, 1:] = xinv @ A[:, 1:] @ np.linalg.inv(Bbus[1:, 1:])
        R = np.diag([br["r_pu"] for br in net.branches])
        G = ptdf[:, net.gen_bus]                          # flow sensitivity to each generator (pu/pu)
        f0 = -ptdf @ (Pd_mw / base)                       # flows caused by the loads alone
        # P_L(pu) = (G p + f0)^T R (G p + f0) with p = P/base; convert to MW in P (MW).
        self.B = G.T @ R @ G / base                       # 1/MW
        self.B0 = 2 * G.T @ R @ f0                        # dimensionless
        self.B00 = float(f0 @ R @ f0) * base              # MW
        self.ptdf, self.Pd = ptdf, Pd_mw

    def __call__(self, P):
        P = np.asarray(P, dtype=float)
        return float(P @ self.B @ P + self.B0 @ P + self.B00)

    def grad(self, P):
        return 2 * self.B @ np.asarray(P, dtype=float) + self.B0


@dataclass
class LoadFlow:
    converged: bool
    iterations: int
    V: np.ndarray            # complex bus voltages (pu)
    P_gen: np.ndarray        # MW, slack output solved by the load flow
    Q_gen: np.ndarray        # MVAr
    losses_mw: float
    flows_mva: np.ndarray    # apparent power at the sending end of each line
    voltage_ok: bool
    lines_ok: bool
    q_ok: bool


def newton_raphson(net: Network, sys: System, P_gen_mw, Pd_mw, tol=1e-8, max_iter=20) -> LoadFlow:
    """Polar Newton-Raphson load flow with the analytic Jacobian (dS/dVa, dS/dVm)."""
    nb, base, Y = net.n_bus, net.base, net.ybus
    Qd_mw = Pd_mw * np.tan(np.arccos(net.pf))
    Sg = np.zeros(nb, dtype=complex)
    Sg[net.gen_bus] = np.asarray(P_gen_mw, dtype=float)
    Ssch = (Sg - (Pd_mw + 1j * Qd_mw)) / base

    Vm = np.ones(nb)
    for bus, v in V_SET.items():
        Vm[bus - 1] = v
    Va = np.zeros(nb)
    pv = np.array([1, 2])          # buses 2, 3
    pq = np.array([3, 4, 5])       # buses 4, 5, 6
    pvpq = np.concatenate([pv, pq])

    converged = False
    for it in range(1, max_iter + 1):
        V = Vm * np.exp(1j * Va)
        mis = V * np.conj(Y @ V) - Ssch
        F = np.concatenate([mis.real[pvpq], mis.imag[pq]])
        if np.max(np.abs(F)) < tol:
            converged = True
            break
        Ibus = Y @ V
        dS_dVm = np.diag(V) @ np.conj(Y @ np.diag(V / np.abs(V))) + np.conj(np.diag(Ibus)) @ np.diag(V / np.abs(V))
        dS_dVa = 1j * np.diag(V) @ np.conj(np.diag(Ibus) - Y @ np.diag(V))
        J = np.block([
            [dS_dVa.real[np.ix_(pvpq, pvpq)], dS_dVm.real[np.ix_(pvpq, pq)]],
            [dS_dVa.imag[np.ix_(pq, pvpq)], dS_dVm.imag[np.ix_(pq, pq)]],
        ])
        dx = np.linalg.solve(J, -F)
        Va[pvpq] += dx[:len(pvpq)]
        Vm[pq] += dx[len(pvpq):]

    V = Vm * np.exp(1j * Va)
    S = V * np.conj(Y @ V) * base                        # net injection (MVA)
    Sgen = S + Pd_mw + 1j * Qd_mw
    flows = []
    for br in net.branches:
        i, j = br["from"] - 1, br["to"] - 1
        y, ysh = 1 / complex(br["r_pu"], br["x_pu"]), 1j * br["bc_pu"] / 2
        flows.append(abs(V[i] * np.conj((V[i] - V[j]) * y + V[i] * ysh)) * base)
    flows = np.array(flows)
    ratings = np.array([br["rating_mva"] for br in net.branches])
    qlim = [(g["qmin_mvar"], g["qmax_mvar"]) for g in sys.raw["generators"]]
    Qg = Sgen.imag[net.gen_bus]
    return LoadFlow(
        converged=converged, iterations=it, V=V, P_gen=Sgen.real[net.gen_bus], Q_gen=Qg,
        losses_mw=float(S.real.sum()), flows_mva=flows,
        voltage_ok=bool(np.all((Vm >= net.vmin - 1e-9) & (Vm <= net.vmax + 1e-9))),
        lines_ok=bool(np.all(flows <= ratings + 1e-6)),
        q_ok=bool(all(lo <= q <= hi for q, (lo, hi) in zip(Qg, qlim))),
    )
