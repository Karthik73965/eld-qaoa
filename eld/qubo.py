"""ELD -> QUBO -> Ising Hamiltonian.

Each generator output is encoded with n_bits binary variables:
    P_i = Pmin_i + dP_i * sum_k 2^k x_ik,   dP_i = (Pmax_i - Pmin_i) / (2^n_bits - 1)
so P_i stays within its limits for every bitstring. The QUBO energy (in $/h) is
    E(x) = sum_i C_i(P_i(x)) + A * (sum_i P_i(x) - P_D)^2
Variable j = i * n_bits + k is mapped to qubit j.
"""
from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from .data import System


@dataclass
class Qubo:
    const: float
    lin: np.ndarray        # q_j
    quad: np.ndarray       # strictly upper-triangular U_jl, j < l
    weights: np.ndarray    # MW contributed by variable j when x_j = 1
    gen_of: np.ndarray     # generator index of variable j
    pmin: np.ndarray
    step: np.ndarray       # dP_i per generator
    n_bits: int
    penalty: float
    demand: float

    @property
    def n_vars(self):
        return len(self.lin)

    def energy(self, x) -> float:
        x = np.asarray(x, dtype=float)
        return float(self.const + self.lin @ x + x @ self.quad @ x)

    def decode(self, x) -> np.ndarray:
        """Bit vector -> generator outputs in MW."""
        x = np.asarray(x, dtype=float)
        return self.pmin + np.bincount(self.gen_of, weights=self.weights * x, minlength=len(self.pmin))


def build_qubo(sys: System, demand, n_bits=4, penalty=20.0) -> Qubo:
    n_gen = sys.n_gen
    n = n_gen * n_bits
    step = (sys.pmax - sys.pmin) / (2**n_bits - 1)
    gen_of = np.repeat(np.arange(n_gen), n_bits)
    weights = step[gen_of] * 2.0 ** np.tile(np.arange(n_bits), n_gen)

    const, lin, quad = 0.0, np.zeros(n), np.zeros((n, n))

    def add_square(coef, offset, w):
        """Add coef * (offset + w.x)^2, using x_j^2 = x_j."""
        nonlocal const
        const += coef * offset**2
        lin[:] += coef * (2 * offset * w + w**2)
        quad[:] += coef * 2 * np.triu(np.outer(w, w), k=1)

    for i in range(n_gen):
        w_i = np.where(gen_of == i, weights, 0.0)
        add_square(sys.a[i], sys.pmin[i], w_i)                     # a_i P_i^2
        const += sys.b[i] * sys.pmin[i] + sys.c[i]                 # b_i Pmin_i + c_i
        lin[:] += sys.b[i] * w_i                                   # b_i (P_i - Pmin_i)
    add_square(penalty, sys.pmin.sum() - demand, weights)          # A (sum P - P_D)^2

    return Qubo(const, lin, quad, weights, gen_of, sys.pmin.copy(), step, n_bits, penalty, demand)


def all_bitstrings(n) -> np.ndarray:
    """Row r holds the bits of integer r, column j = bit j (= qubit j in Qiskit ordering)."""
    idx = np.arange(2**n)
    return ((idx[:, None] >> np.arange(n)) & 1).astype(float)


def all_energies(qubo: Qubo) -> np.ndarray:
    """QUBO energy of every bitstring (exhaustive / brute force), indexed like Qiskit statevectors."""
    X = all_bitstrings(qubo.n_vars)
    return qubo.const + X @ qubo.lin + np.einsum("rj,jl,rl->r", X, qubo.quad, X)


def to_ising(qubo: Qubo):
    """Substitute x_j = (1 - Z_j)/2. Returns (SparsePauliOp H, offset) with E(x) = <H> + offset."""
    n = qubo.n_vars
    U = qubo.quad
    h = -qubo.lin / 2 - (U.sum(axis=1) + U.sum(axis=0)) / 4
    offset = qubo.const + qubo.lin.sum() / 2 + U.sum() / 4
    terms = [("Z", [j], h[j]) for j in range(n) if h[j] != 0]
    terms += [("ZZ", [j, l], U[j, l] / 4) for j in range(n) for l in range(j + 1, n) if U[j, l] != 0]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n).simplify(), offset
