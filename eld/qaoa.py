"""QAOA for the ELD QUBO, built explicitly from the Ising Hamiltonian and run with Qiskit.

Circuit:  |+>^n  ->  prod_{l=1..p} [ exp(-i gamma_l H_C) exp(-i beta_l H_M) ]  ->  measure
    H_C = sum h_j Z_j + sum J_jl Z_j Z_l   (scaled so the largest |coefficient| is 1)
    H_M = sum X_j
Parameters are trained on the exact statevector (noise-free simulator); final dispatch is
chosen from measured shots on the Aer simulator (or IBM hardware, see scripts/run_ibm.py).
"""
import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize


def qaoa_circuit(H: SparsePauliOp, p: int, measure=False) -> QuantumCircuit:
    n = H.num_qubits
    gamma, beta = ParameterVector("γ", p), ParameterVector("β", p)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    terms = H.to_sparse_list()
    for layer in range(p):
        for pauli, idx, coeff in terms:          # cost layer exp(-i γ H_C)
            if pauli == "Z":
                qc.rz(2 * gamma[layer] * coeff.real, idx[0])
            elif pauli == "ZZ":
                qc.rzz(2 * gamma[layer] * coeff.real, idx[0], idx[1])
        qc.rx(2 * beta[layer], range(n))          # mixer layer exp(-i β H_M)
    if measure:
        qc.measure_all()
    return qc


def scaled_hamiltonian(H: SparsePauliOp):
    scale = float(np.max(np.abs(H.coeffs)))
    return (H / scale).simplify(), scale


def interp_init(params, p):
    """INTERP strategy (Zhou et al., PRX 2020): extend optimal depth-p angles to depth p+1."""
    def extend(v):
        v = np.concatenate([[0.0], v, [0.0]])
        return np.array([(i / p) * v[i] + ((p - i) / p) * v[i + 1] for i in range(p + 1)])
    return np.concatenate([extend(params[:p]), extend(params[p:])])


class QAOARunner:
    """Trains QAOA angles for one QUBO. `energies` = QUBO energy of every basis state ($/h)."""

    def __init__(self, H: SparsePauliOp, energies: np.ndarray, objective="cvar", alpha=0.1):
        self.H, self.scale = scaled_hamiltonian(H)
        self.energies = energies
        self.e_min, self.e_max = energies.min(), energies.max()
        self.order = np.argsort(energies)
        self.objective = objective
        self.alpha = alpha

    def probabilities(self, qc, params):
        return Statevector(qc.assign_parameters(params)).probabilities()

    def expectation(self, probs):
        return float(probs @ self.energies)

    def cvar(self, probs):
        """Mean energy of the lowest-energy alpha fraction of the distribution (Barkoutsos et al., Quantum 2020)."""
        p_sorted = probs[self.order]
        cum = np.cumsum(p_sorted)
        take = np.minimum(p_sorted, np.maximum(self.alpha - (cum - p_sorted), 0.0))
        return float(take @ self.energies[self.order] / self.alpha)

    def loss(self, probs):
        val = self.cvar(probs) if self.objective == "cvar" else self.expectation(probs)
        return (val - self.e_min) / (self.e_max - self.e_min)   # 0 = all weight on the optimum

    def train(self, p, restarts=5, init=None, seed=0, maxiter=400):
        rng = np.random.default_rng(seed)
        qc = qaoa_circuit(self.H, p)
        starts = [] if init is None else [np.asarray(init)]
        while len(starts) < restarts:
            starts.append(np.concatenate([rng.uniform(0, np.pi, p), rng.uniform(0, np.pi / 2, p)]))

        best, t0, nfev = None, time.perf_counter(), 0
        for x0 in starts:
            history = []

            def f(theta):
                v = self.loss(self.probabilities(qc, theta))
                history.append(v)
                return v

            res = minimize(f, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.3})
            nfev += res.nfev
            if best is None or res.fun < best["loss"]:
                best = {"params": res.x, "loss": res.fun, "history": history}
        best["probs"] = self.probabilities(qc, best["params"])
        best["time_s"] = time.perf_counter() - t0
        best["nfev"] = nfev
        best["p"] = p
        return best


def sample_counts(H_scaled: SparsePauliOp, p, params, shots=4096, seed=1, backend=None):
    """Measure the trained circuit. Returns {basis-state index: count} (index bit j = qubit j)."""
    from qiskit_aer import AerSimulator
    backend = backend or AerSimulator(seed_simulator=seed)
    qc = qaoa_circuit(H_scaled, p, measure=True).assign_parameters(params)
    counts = backend.run(transpile(qc, backend), shots=shots).result().get_counts()
    return {int(k.replace(" ", ""), 2): v for k, v in counts.items()}
