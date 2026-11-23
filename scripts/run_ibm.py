"""Run the trained QAOA circuit on real IBM Quantum hardware (period 1, 250 MW) and compare the
measured distribution with the noise-free simulator and with uniform random sampling.

Angles are trained on the simulator (as in run_all.py); only the final sampling runs on the QPU.
Small encodings (2 and 3 bits/generator -> 6 and 9 qubits) are used because the full 12-qubit
all-to-all circuit needs many SWAPs on IBM's heavy-hex lattice and decoheres.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

from eld.classical import solve_slsqp
from eld.data import gen_cost, load_system
from eld.qaoa import QAOARunner, qaoa_circuit
from eld.qubo import all_bitstrings, all_energies, build_qubo, to_ising
from eld.refine import refine

PERIOD, PENALTY, P, SHOTS = 1, 20.0, 1, 4096
OUT = ROOT / "results"


def summarise(dist, E, i_opt, good):
    """dist: {index: probability}"""
    p_opt = sum(v for k, v in dist.items() if np.isclose(E[k], E[i_opt]))
    p_good = sum(v for k, v in dist.items() if good[k])
    mean_e = sum(v * E[k] for k, v in dist.items())
    return {"p_opt": p_opt, "p_good": p_good, "approx_ratio": (E.max() - mean_e) / (E.max() - E.min())}


def main():
    load_dotenv(ROOT / ".env")
    sys_ = load_system()
    demand = next(l["total"] for l in sys_.loads if l["period"] == PERIOD)
    _, c_ref = solve_slsqp(sys_, demand)

    service = QiskitRuntimeService(channel=os.environ["IBM_QUANTUM_CHANNEL"],
                                   token=os.environ["IBM_QUANTUM_TOKEN"], instance="auto")
    backend = service.least_busy(operational=True, simulator=False)
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=1)
    print("backend:", backend.name)

    cases, pubs = [], []
    for n_bits in (2, 3):
        q = build_qubo(sys_, demand, n_bits, PENALTY)
        E = all_energies(q)
        X = all_bitstrings(q.n_vars)
        H, _ = to_ising(q)
        runner = QAOARunner(H, E, objective="cvar", alpha=0.05)
        res = runner.train(P, restarts=8, seed=7)
        isa = pm.run(qaoa_circuit(runner.H, P, measure=True).assign_parameters(res["params"]))
        Pm = q.pmin + np.stack([(X * q.weights * (q.gen_of == i)).sum(1) for i in range(3)], 1)
        cost = (sys_.a * Pm**2 + sys_.b * Pm + sys_.c).sum(1)
        good = (np.abs(Pm.sum(1) - demand) <= q.step.max()) & (cost <= 1.01 * c_ref)
        cases.append(dict(n_bits=n_bits, q=q, E=E, X=X, probs=res["probs"], good=good,
                          params=res["params"].tolist(), isa_depth=isa.depth(),
                          isa_2q=sum(v for k, v in isa.count_ops().items() if k in ("cz", "ecr", "cx", "rzz"))))
        pubs.append(isa)
        print(f"{n_bits} bits: {q.n_vars} qubits, transpiled depth {isa.depth()}, 2q gates {cases[-1]['isa_2q']}")

    job = SamplerV2(mode=backend).run(pubs, shots=SHOTS)
    print("job id:", job.job_id(), "- waiting for the QPU queue...", flush=True)
    result = job.result()

    report = {"backend": backend.name, "job_id": job.job_id(), "period": PERIOD, "demand": demand,
              "p": P, "shots": SHOTS, "cost_slsqp": c_ref, "cases": []}
    for case, pub_res in zip(cases, result):
        E, X, q, good = case["E"], case["X"], case["q"], case["good"]
        i_opt = int(E.argmin())
        counts = {int(k, 2): v for k, v in pub_res.data.meas.get_counts().items()}
        hw = {k: v / SHOTS for k, v in counts.items()}
        ideal = dict(enumerate(case["probs"]))
        uniform = {k: 1 / len(E) for k in range(len(E))}
        i_best = min(counts, key=lambda k: E[k])
        P_q = q.decode(X[i_best])
        P_h, c_h, _ = refine(sys_, P_q, q.step, demand)
        report["cases"].append({
            "n_bits": case["n_bits"], "qubits": q.n_vars, "isa_depth": case["isa_depth"], "isa_2q_gates": case["isa_2q"],
            "params": case["params"],
            "hardware": summarise(hw, E, i_opt, good), "simulator": summarise(ideal, E, i_opt, good),
            "random": summarise(uniform, E, i_opt, good),
            "hw_best_dispatch_mw": P_q.tolist(), "hw_best_cost": gen_cost(sys_, P_q),
            "hw_found_qubo_opt": bool(np.isclose(E[i_best], E[i_opt])),
            "hybrid_dispatch_mw": P_h.tolist(), "hybrid_cost": c_h,
            "hybrid_gap_pct": 100 * (c_h - c_ref) / c_ref,
            "hw_counts": {str(k): v for k, v in sorted(counts.items(), key=lambda kv: -kv[1])},
        })
    OUT.mkdir(exist_ok=True)
    (OUT / "ibm_hardware.json").write_text(json.dumps(report, indent=2, default=float))
    for c in report["cases"]:
        print(c["qubits"], "qubits | P(opt) hw/sim/random:",
              *(round(c[k]["p_opt"], 4) for k in ("hardware", "simulator", "random")),
              "| P(good):", *(round(c[k]["p_good"], 4) for k in ("hardware", "simulator", "random")),
              "| hybrid gap %:", round(c["hybrid_gap_pct"], 4))


if __name__ == "__main__":
    main()
