"""Validation tests. Run:  python -m pytest tests  (or python tests/test_validation.py)

1. SLSQP and lambda iteration agree for every period.
2. QUBO energies equal cost + penalty of the decoded dispatch; Ising Hamiltonian reproduces them.
3. Newton-Raphson load flow reproduces PYPOWER (Python port of MATPOWER) on the original
   Wood & Wollenberg case6ww base case.
4. The DC-flow loss formula is within 6 % of the AC (Newton-Raphson) losses in every period.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eld.classical import solve_lambda, solve_slsqp
from eld.data import gen_cost, load_system
from eld.network import LossModel, build_network, bus_loads, newton_raphson
from eld.qubo import all_bitstrings, all_energies, build_qubo, to_ising

SYS = load_system()


def test_slsqp_matches_lambda():
    for load in SYS.loads:
        P1, c1 = solve_slsqp(SYS, load["total"])
        P2, c2, _ = solve_lambda(SYS, load["total"])
        assert np.allclose(P1, P2, atol=1e-4) and abs(c1 - c2) < 1e-6


def test_qubo_and_ising_energies():
    q = build_qubo(SYS, 250, n_bits=3, penalty=20.0)
    E = all_energies(q)
    X = all_bitstrings(q.n_vars)
    P = np.array([q.decode(x) for x in X])
    direct = np.array([gen_cost(SYS, p) for p in P]) + 20.0 * (P.sum(1) - 250) ** 2
    assert np.allclose(E, direct)
    H, offset = to_ising(q)
    Z = 1 - 2 * X
    diag = np.zeros(len(X))
    for pauli, idx, coeff in H.to_sparse_list():
        diag += coeff.real * np.prod(Z[:, idx], axis=1)
    assert np.allclose(diag + offset, E)


def test_newton_raphson_matches_pypower():
    from pypower.api import case6ww, ppoption, runpf
    ref, ok = runpf(case6ww(), ppoption(VERBOSE=0, OUT_ALL=0))
    assert ok
    net = build_network(SYS)
    net.pf = np.cos(np.pi / 4)                       # case6ww loads are 70 MW + j70 MVAr
    lf = newton_raphson(net, SYS, ref["gen"][:, 1], ref["bus"][:, 2])
    assert lf.converged
    assert np.allclose(lf.P_gen, ref["gen"][:, 1], atol=1e-3)
    assert np.allclose(lf.Q_gen, ref["gen"][:, 2], atol=1e-3)
    assert np.allclose(np.abs(lf.V), ref["bus"][:, 7], atol=1e-4)
    assert np.allclose(np.degrees(np.angle(lf.V)), ref["bus"][:, 8], atol=1e-3)


def test_loss_formula_close_to_ac():
    net = build_network(SYS)
    for load in SYS.loads:
        Pd = bus_loads(net, load)
        P, _ = solve_slsqp(SYS, load["total"])
        ac = newton_raphson(net, SYS, P, Pd).losses_mw
        assert abs(LossModel(net, Pd)(P) - ac) / ac < 0.06   # DC approximation within 6 % (observed 2-5 %)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
