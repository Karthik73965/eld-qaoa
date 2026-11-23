# Sources and Provenance Log

Every number, formula, method and setting used in this project, with where it comes from.
Bibliographic details were checked on 24 Nov 2026 against Crossref (`api.crossref.org/works/<DOI>`)
and the arXiv API (`export.arxiv.org/api/query?id_list=<id>`).

## 1. Test-system data

Primary source: **J. F. Prada, *Ensuring the Reliable Operation of the Power Grid: State-Based
and Distributed Approaches to Scheduling Energy and Contingency Reserves*, PhD thesis, Carnegie
Mellon University, 2017, Appendix D, Tables 30–35, pp. 146–148.**
URL: https://www.cmu.edu/ceic/assets/docs/publications/phd-dissertations/2017/jose-prada-phd-thesis-2017.pdf
Local copy: `data/reference/prada2017_thesis.pdf`. Machine-readable transcription: `data/sixbus_prada2017.json`.

| Item | Value(s) | Source location |
|---|---|---|
| Bus types, base kV 230, Vmax 1.07, Vmin 0.95 pu | 6 buses | Table 30, p. 146 |
| Branch R, X, Bc (pu), continuous / short-term ratings (MVA) | 11 lines | Table 31, p. 146 |
| Load per bus (L4, L5, L6) and total, 12 periods | 180–330 MW | Table 32, p. 147 |
| Pmin, Pmax, Qmin, Qmax, ramp rates, min up/down | 3 generators | Table 33, p. 147 |
| Cost coefficients a, b, c; start-up; reserve cost | 3 generators | Table 34, p. 147 |
| Initial output 90 MW each (not used: single-period ELD) | | Table 35, p. 148 |
| Load power factor 0.98 | | Section 8.1, p. 92 |
| System "adapted from [15]" = Wood & Wollenberg | | Section 8.1, p. 91 (Fig. 11 caption) |

Values **not** stated in Prada and taken from the original system's digital record,
**MATPOWER `case6ww.m`** ("6 bus example from pp. 104, 112, 119, 123–124, 549 of *Power
Generation, Operation, and Control*, 2nd ed., Wood & Wollenberg, Wiley, 1996").
URL: https://raw.githubusercontent.com/MATPOWER/matpower/master/data/case6ww.m
Local copy: `data/reference/case6ww.m` (SHA-256 `2DDF9154A81E4A5A7BCE287B116DF328FABF31D9D8DDB55552FFF9C97291C65A`).

| Item | Value | Line in case6ww.m |
|---|---|---|
| System MVA base | 100 MVA | `mpc.baseMVA = 100;` |
| Generator voltage setpoints | V1 = 1.05, V2 = 1.05, V3 = 1.07 pu | `mpc.gen` column `Vg` |
| Slack = bus 1, PV = buses 2–3, PQ = buses 4–6 | | `mpc.bus` column `type` (3, 2, 1) |

Cross-check: R, X, B of all 11 lines and all cost coefficients in case6ww.m are identical to
Prada Tables 31 and 34.

**Where Prada and the original W&W data differ (Prada used, as the project's stated source):**

| Item | Prada (used) | case6ww / W&W original |
|---|---|---|
| Loads | 12-period profile, pf 0.98 | 70 MW + j70 MVAr each |
| Line ratings | 100 MVA (lines from bus 1), 60 MVA (others) | 20–90 MVA, line by line |
| Generator Qmax | 150 / 150 / 120 MVAr | 100 MVAr each |
| Load-bus Vmax | 1.07 pu | 1.05 pu |

## 2. Methods and formulas

| What | Reference (verified) | Used in |
|---|---|---|
| ELD formulation, quadratic cost, equal incremental cost (λ-iteration), Kron loss formula P_L = PᵀBP + B₀ᵀP + B₀₀ | A. J. Wood, B. F. Wollenberg, G. B. Sheblé, *Power Generation, Operation, and Control*, 3rd ed., Wiley, 2014 | `eld/classical.py`, `eld/network.py` |
| SLSQP solver | SciPy `scipy.optimize.minimize(method="SLSQP")`, D. Kraft, "A software package for sequential quadratic programming", DFVLR-FB 88-28, 1988 | `eld/classical.py` |
| QUBO / Ising mapping of constrained problems with quadratic penalty | A. Lucas, "Ising formulations of many NP problems", *Front. Phys.* 2:5, 2014, doi:10.3389/fphy.2014.00005 | `eld/qubo.py` |
| QAOA | E. Farhi, J. Goldstone, S. Gutmann, "A Quantum Approximate Optimization Algorithm", arXiv:1411.4028, 2014 | `eld/qaoa.py` |
| CVaR objective for variational optimisation | P. Kl. Barkoutsos, G. Nannicini, A. Robert, I. Tavernelli, S. Woerner, "Improving Variational Quantum Optimization using CVaR", *Quantum* 4, 256, 2020, doi:10.22331/q-2020-04-20-256 | `QAOARunner.cvar` |
| INTERP initialisation for depth p+1 | L. Zhou, S.-T. Wang, S. Choi, H. Pichler, M. D. Lukin, "Quantum Approximate Optimization Algorithm: Performance, Mechanism, and Implementation on Near-Term Devices", *Phys. Rev. X* 10, 021067, 2020, doi:10.1103/PhysRevX.10.021067 | `interp_init` |
| Newton–Raphson load flow, analytic Jacobian dS/dV | R. D. Zimmerman, C. E. Murillo-Sánchez, R. J. Thomas, "MATPOWER: Steady-State Operations, Planning, and Analysis Tools for Power Systems Research and Education", *IEEE Trans. Power Syst.* 26(1):12–19, 2011, doi:10.1109/TPWRS.2010.2051168 | `newton_raphson` |
| PSO with linearly decreasing inertia weight | Y. Shi, R. Eberhart, "A modified particle swarm optimizer", *Proc. IEEE ICEC*, pp. 69–73, 1998, doi:10.1109/ICEC.1998.699146 | `eld/metaheuristics.py` |
| PSO for ELD, dependent-unit handling of the balance constraint | Z.-L. Gaing, "Particle swarm optimization to solving the economic dispatch considering the generator constraints", *IEEE Trans. Power Syst.* 18(3):1187–1195, 2003, doi:10.1109/TPWRS.2003.814889 | `eld_objective(mode="dependent")` |
| Simulated binary crossover (real-coded GA) | K. Deb, R. B. Agrawal, "Simulated binary crossover for continuous search space", *Complex Systems* 9:115–148, 1995 | `ga` |
| Simulated annealing | S. Kirkpatrick, C. D. Gelatt Jr., M. P. Vecchi, "Optimization by simulated annealing", *Science* 220(4598):671–680, 1983, doi:10.1126/science.220.4598.671 | `simulated_annealing` |

## 3. Related quantum-power-systems literature (cited in the report)

| Reference (verified) | Note |
|---|---|
| X. Han, Z. Li, Y. Xu, "Quantum Assisted Stochastic Economic Dispatch for Renewables Rich Power Systems", arXiv:2404.13073, 2024 | Authors as on arXiv. |
| H. Jing, Y. Wang, Y. Li, "Data-driven quantum approximate optimization algorithm for power systems", *Communications Engineering* 2, 12, 2023, doi:10.1038/s44172-023-00061-8 | **Corrects Review-1 slide ref [4]**, which read "Li et al. (2024)". |
| M. Wang, Z. Cao, Y. Li, "Quantum-Behaved Particle Swarm Optimization Based on Concentration Selection Probability Assignment Weights for Power System Economic Dispatch", *IEEJ Trans. Electr. Electron. Eng.* 19(5):640–652, 2024, doi:10.1002/tee.24015 | |

## 4. Software (exact versions, `requirements.txt`)

Python 3.11.9; qiskit 2.6.2; qiskit-aer 0.18.4; qiskit-ibm-runtime 0.52.1; numpy 2.4.6;
scipy 1.17.1; pandas 3.1.1; matplotlib 3.11.2; python-dotenv 1.2.4; PYPOWER 5.1.21 (validation only).

Hardware: IBM Quantum Platform (https://quantum.cloud.ibm.com), Open plan, backend `ibm_marrakesh`
(156-qubit Heron), job id `dc299oanj2vs73dgbsd0`. Runtime channel name `ibm_quantum_platform`.

## 5. Settings chosen in this project (design choices, not taken from a source)

These are tuning decisions. Each is stated here so nothing is hidden; sensitivity is reported where noted.

| Setting | Value | Reason |
|---|---|---|
| Bits per generator | 4 (12 qubits); 2 and 3 on hardware | Largest size that trains in ~20 s/period on a laptop statevector simulator; hardware limited by noise |
| Penalty weight A | 20 $/MW² | Of A ∈ {1, 5, 20, 100}, the smallest whose QUBO optimum meets demand to ≤ 0.5 MW in all 12 periods (max mismatch 7, 2, 0.5, 0 MW; A = 100 balances exactly but its grid optimum costs +0.18 %). Data: `results/penalty_sweep.csv` |
| QAOA depth p | 1–4 (INTERP) | Reported for every p |
| CVaR α | 0.05 | Within the range studied by Barkoutsos et al. |
| Optimiser | COBYLA, 8 random restarts at p = 1, max 400 iterations | Gradient-free, standard for QAOA |
| Shots | 4096 | Equal to the number of bitstrings (2¹²), giving the same evaluation budget as exhaustive search and SA |
| Refinement | top-5 sampled bitstrings, box ±1 grid step | Classical post-processing of samples |
| PSO | 30 particles, 100 iterations, w 0.9→0.4, c1 = c2 = 2, vmax = 20 % of range | Common settings (Shi & Eberhart 1998); budget ≈ QAOA shots |
| GA | 30 individuals, 100 generations, pc 0.9, pm 0.1, SBX ηc = 15 | Common real-coded GA settings; same budget |
| SA | 4096 evaluations, T₀ = mean single-flip ΔE, geometric cooling to T₀·10⁻⁴; the 5 lowest-energy distinct states visited are refined | Same budget and same top-5 refinement as QAOA |
| Random seeds | 20 per stochastic method | Mean, worst and success rate reported |
| Success criterion | cost within 0.1 % of SLSQP and |mismatch| ≤ 0.01 MW | |

## 6. Losses study (`scripts/run_losses.py`)

| Item | Value / method | Source |
|---|---|---|
| Loss coefficients B, B0, B00 | Derived per period from DC PTDF (slack bus 1) and line resistances: P_L = Σ r_l f_l² expanded into Kron form | Wood & Wollenberg (2014); line data Prada Table 31 |
| Hybrid with losses | QAOA p = 3 on QUBO with demand P_D + P_L estimate, refine with loss-inclusive balance, iterate (max 4) | This project |
| AC check | Newton–Raphson, validated against PYPOWER case6ww (tests/test_validation.py) | Zimmerman et al. (2011) |
| Line ratings used for the overload check | 60 MVA continuous / 75 MVA short-term for T09 | Prada Table 31 |
