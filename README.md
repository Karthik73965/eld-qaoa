# QAOA-Based Economic Load Dispatch (6-Bus System)

EEE mini project, RGUKT Nuzvid — J. Karthikeya, A. Krishna Chaitanya, N. Ramesh, M. Harika.
Guide: B. Madhusudhan Rao.

Economic load dispatch (ELD) of a 6-bus, 3-generator test system solved with the Quantum
Approximate Optimization Algorithm (QAOA), benchmarked against classical SciPy SLSQP for all
12 load periods, with a run on real IBM Quantum hardware.

## Pipeline

```
6-bus data -> ELD model -> binary encoding (4 bits/gen, 12 qubits) -> QUBO -> Ising H
          -> CVaR-QAOA (Qiskit, p = 1..4) -> 4096 shots -> decode MW
          -> local classical refinement (exact power balance) -> compare with SLSQP
```

## Key results (simulator, 12 periods, 180–330 MW)

| Metric | Value |
|---|---|
| QAOA sampled the exact QUBO optimum | 12 / 12 periods |
| Hybrid cost vs SLSQP, worst period | +0.040 % |
| 12-period total cost: hybrid vs SLSQP | $43,173.88 vs $43,171.05 /h-sum |
| P(optimal bitstring) vs random, p = 1 → 4 | 4.6× → 6.8× → 8.3× → 10.2× |
| Shots landing near-optimal (≤1 % after refinement) | 20.4 % (random: 3.1 %) |

Real hardware (`ibm_marrakesh`, p = 1, period 1): QAOA beats random sampling on the QPU —
P(optimal) 3.7 % vs 1.6 % (6 qubits); P(near-optimal) 15.5 % vs 6.0 % (9 qubits).

Classical SLSQP solves each period in milliseconds; QAOA is not faster on a 3-generator
problem. The study shows that ELD maps correctly to a QUBO, that QAOA concentrates
probability on economical dispatches, and how that behaves with depth and hardware noise.

## Layout

| Path | Contents |
|---|---|
| `data/sixbus_prada2017.json` | Test system (Prada 2017, App. D, Tables 30–34) |
| `eld/` | `data`, `classical` (SLSQP, λ-iteration), `qubo`, `qaoa`, `refine` |
| `scripts/run_all.py` | Full 12-period study → `results/periods.csv`, `qaoa_depth.csv`, `summary.json` |
| `scripts/run_ibm.py` | Real-hardware run → `results/ibm_hardware.json` |
| `scripts/plots.py` | Figures → `results/figures/` |

## Run

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -r requirements.txt
python scripts/run_all.py && python scripts/plots.py
# hardware: put IBM_QUANTUM_TOKEN and IBM_QUANTUM_CHANNEL=ibm_quantum_platform in .env
python scripts/run_ibm.py
```

## Data source

J. F. Prada, *Ensuring the Reliable Operation of the Power Grid*, PhD thesis, Carnegie Mellon
University, 2017, Appendix D (system adapted from Wood, Wollenberg & Sheblé, *Power Generation,
Operation, and Control*). Base MVA (100) is assumed; it is not stated in the source.
