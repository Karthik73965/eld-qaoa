# QAOA-Based Economic Load Dispatch (6-Bus System)

EEE mini project, RGUKT Nuzvid — J. Karthikeya, A. Krishna Chaitanya, N. Ramesh, M. Harika.
Guide: B. Madhusudhan Rao.

Economic load dispatch (ELD) of a 6-bus, 3-generator test system solved with the Quantum
Approximate Optimization Algorithm (QAOA), benchmarked against SLSQP, λ-iteration, PSO, GA and
simulated annealing for all 12 load periods, run on real IBM Quantum hardware, and extended to
transmission losses with an AC Newton–Raphson check.

## Pipeline

```
6-bus data -> ELD model -> binary encoding (4 bits/gen, 12 qubits) -> QUBO -> Ising H
          -> CVaR-QAOA (Qiskit, p = 1..4) -> 4096 shots -> decode MW
          -> refine 5 best bitstrings (exact power balance) -> compare with SLSQP
          -> losses (network B-coefficients) -> Newton-Raphson AC load flow
```

## Key results (simulator, 12 periods, 180–330 MW)

| Metric | Value |
|---|---|
| QAOA sampled the exact QUBO optimum | 12 / 12 periods |
| Hybrid cost vs SLSQP, worst period | +0.040 % |
| 12-period total cost: hybrid vs SLSQP | $43,173.88 vs $43,171.05 |
| P(optimal bitstring) vs random, p = 1 → 4 | 4.6× → 6.8× → 8.3× → 10.2× |
| Shots landing near-optimal | 20.4 % (random: 3.1 %) |

**Standard algorithms** (240 runs each; success = within 0.1 % of SLSQP and balanced):

| Method | Mean gap | Worst gap | Success |
|---|---|---|---|
| QAOA + refine (12-bit QUBO) | 0.007 % | 0.040 % | 100 % |
| Exhaustive search + refine (same QUBO) | 0.007 % | 0.040 % | 100 % |
| Simulated annealing + refine (same QUBO, same budget) | 0.029 % | 0.437 % | 92.5 % |
| PSO / GA, dependent unit (Gaing 2003) | 0.000 % | 0.000 % | 100 % |
| PSO / GA, penalty | 0.65 / 0.76 % | 2.2 / 3.0 % | 13.8 / 7.5 % |

**Real hardware** (`ibm_marrakesh`, p = 1, period 1): P(optimal) 3.7 % vs 1.6 % random at 6 qubits
(hybrid gap 0 %); at 9 qubits noise removes most of the gain (0.24 % vs 0.20 %).

**Losses + AC check:** losses 2.7–9.2 MW add 2.0 % to daily cost; hybrid stays within 0.08 % of the
loss-inclusive optimum; B-coefficient losses within 1.5–4.3 % of Newton–Raphson; voltages and
reactive power within limits. Line T09 (bus 3–6) is overloaded up to 145 % at the peak — for the
classical optimum too, since ELD ignores line limits.

**No speed-up is claimed:** SLSQP solves each period in ~1 ms; simulated QAOA needs ~22 s training.

## Layout

| Path | Contents |
|---|---|
| `data/sixbus_prada2017.json` | Test system (Prada 2017, App. D, Tables 30–34) |
| `data/reference/` | Archived source thesis and MATPOWER `case6ww.m` |
| `docs/SOURCES.md` | Provenance of every value, method and setting |
| `eld/` | `data`, `classical`, `qubo`, `qaoa`, `refine`, `metaheuristics`, `network` |
| `scripts/` | `run_all`, `run_baselines`, `run_penalty_sweep`, `run_losses`, `run_ibm`, `plots` |
| `tests/test_validation.py` | SLSQP vs λ, QUBO vs Ising, NR vs PYPOWER, loss formula vs AC |
| `results/` | CSV/JSON results and figures |
| `report/` | LaTeX project report (Overleaf-ready) |
| `paper/` | IEEE conference paper (IEEEtran, Overleaf-ready) |
| `viva/VIVA_QA.md` | Viva questions and answers |
| `docs/index.html` | Project website (GitHub Pages, from `/docs`) |

## Run

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -r requirements.txt
python tests/test_validation.py
python scripts/run_all.py
python scripts/run_penalty_sweep.py
python scripts/run_baselines.py
python scripts/run_losses.py
python scripts/plots.py
# hardware: put IBM_QUANTUM_TOKEN and IBM_QUANTUM_CHANNEL=ibm_quantum_platform in .env
python scripts/run_ibm.py
```

## Data source

J. F. Prada, *Ensuring the Reliable Operation of the Power Grid*, PhD thesis, Carnegie Mellon
University, 2017, Appendix D (system adapted from Wood, Wollenberg & Sheblé). The 100 MVA base and
generator voltage set-points come from MATPOWER `case6ww.m`. Full details: `docs/SOURCES.md`.
