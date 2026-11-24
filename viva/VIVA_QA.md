# Viva Q&A — QAOA-Based Economic Load Dispatch

Short, honest answers. Every number matches `results/` and the report.

## 1. Basics of ELD

**Q1. What is economic load dispatch?**
Sharing the demand among committed generators so that the total fuel cost is minimum, while meeting
power balance (ΣP = P_D, plus losses if modelled) and each unit's P_min–P_max limits.

**Q2. What is the cost model?**
Quadratic: C_i(P_i) = a_i P_i² + b_i P_i + c_i ($/h). Coefficients are from Prada (2017), App. D,
Tables 33–34.

**Q3. What is the equal incremental cost criterion?**
At the optimum every unit not at a limit runs at the same incremental cost dC_i/dP_i = 2a_i P_i + b_i = λ.
λ-iteration searches λ by bisection until ΣP_i(λ) = P_D.

**Q4. Why is G1 at 50 MW in periods 2–6?**
G1 is the most expensive unit (highest b). At low demand its λ would be below its minimum output, so it
is clamped at P_min = 50 MW and G2/G3 set λ.

**Q5. Why SLSQP as the reference?**
The problem is convex (positive a_i, linear constraint), so SLSQP finds the global optimum. We checked it
against λ-iteration: they agree to 10⁻⁶ $/h in all 12 periods.

**Q6. Where does the data come from?**
6-bus, 3-generator Wood & Wollenberg system, using Prada's PhD thesis (CMU, 2017), App. D. The 100 MVA base
and voltage set-points come from MATPOWER `case6ww.m`. Every value is logged in `docs/SOURCES.md`.

## 2. QUBO formulation

**Q7. What is a QUBO?**
Quadratic Unconstrained Binary Optimization: minimise E(x) = xᵀQx over binary x. It is the input format
for QAOA and quantum annealers.

**Q8. How are continuous MW values made binary?**
P_i = P_min + ΔP·Σ 2^k x_ik with 4 bits per unit, so 16 levels. Steps: 10, 7.5 and 9 MW. Three units give
12 bits = 12 qubits = 4096 candidate dispatches.

**Q9. How are generator limits handled?**
They need no penalty. The encoding only produces values between P_min and P_max.

**Q10. How is power balance handled?**
A quadratic penalty A(ΣP − P_D)². It is quadratic in x because P is linear in x.

**Q11. How did you pick A = 20?**
By an exhaustive sweep over A = 1, 5, 20, 100 in all periods:
- A = 1: the QUBO optimum under-supplies by up to 7 MW.
- A = 100: balance dominates, and the cost is worse by 0.18%.
- A = 20: the smallest value that keeps the mismatch ≤ 0.5 MW.

**Q12. Why do some raw QAOA results look cheaper than SLSQP?**
They supply 0.5 MW less than demand. They are infeasible, not cheaper. Refinement fixes the balance.

**Q13. How do you go from QUBO to Ising?**
Substitute x = (1 − Z)/2. This gives H = Σ h_j Z_j + Σ J_jl Z_j Z_l + constant: 12 Z terms and 66 ZZ terms,
since every pair of bits is coupled by the penalty. We verified E(x) = ⟨x|H|x⟩ + const for all 4096
states.

## 3. QAOA

**Q14. What is QAOA?**
Farhi et al. (2014). Start in |+⟩^⊗n, which is all 4096 dispatches equally. Then alternate p layers of:
- the cost unitary e^{−iγH_C}, which marks good dispatches with phases;
- the mixer e^{−iβH_M}, which turns the phases into probability.

A classical optimiser tunes γ and β. Measuring then gives good dispatches more often.

**Q15. Which gates are used?**
- H on all qubits.
- RZ(2γh_j) and RZZ(2γJ_jl) for the cost layer.
- RX(2β) for the mixer.
- Measurement.

There are 2p parameters.

**Q16. What is CVaR and why use it?**
Conditional value-at-risk (Barkoutsos et al., 2020) is the mean energy of the best α fraction of outcomes
(α = 0.05). We want the optimum to appear in the samples, not a good average, so CVaR trains the circuit
for that.

**Q17. What is INTERP?**
Zhou et al. (2020): angles optimised at depth p are interpolated to initialise depth p+1. This avoids bad
random starts. At p = 1 we use 8 random starts.

**Q18. Which optimiser?**
COBYLA, which is gradient-free, with at most 400 iterations.

**Q19. Why is refinement needed? Isn't it cheating?**
The 4-bit grid is 7.5–10 MW wide, so the best grid point cannot hit the continuous optimum exactly.
SLSQP only moves each unit by at most one grid step around QAOA's top-5 bitstrings. It cannot find the
answer if QAOA picks the wrong cell. We apply the exact same refinement to exhaustive search and SA, so
the comparison is fair.

**Q20. What is "amplification"?**
P(optimal bitstring after QAOA) ÷ (1/4096). It is 4.6× at p = 1, 6.8× at p = 2, 8.3× at p = 3 and
10.2× at p = 4 on average, with a maximum of 31× in period 4.

**Q21. Why does deeper p help?**
More layers give more freedom to interfere amplitudes towards low-energy states. As p → ∞, QAOA approaches
adiabatic evolution, which reaches the ground state.

**Q22. In the convergence plot, p = 3 ends above p = 2 in period 1. Why?**
COBYLA is local, and a single INTERP start doesn't guarantee monotone improvement. p = 4 still reaches
the lowest CVaR.

## 4. Results

**Q23. Headline results?**
- QAOA sampled the QUBO optimum in 12/12 periods.
- The hybrid dispatch is within 0.040% of SLSQP in the worst period: $43,173.88 vs $43,171.05 per day.
- 20.4% of shots are near-optimal, against 3.1% for random.

**Q24. Why isn't the gap zero?**
Exhaustive search plus the same refinement gives the same gap, so it comes from the grid and box, not
QAOA. In period 5 the QUBO optimum (60, 67.5, 72) is more than one step from the SLSQP optimum
(50, 83.5, 66.5).

**Q25. How did you compare with standard algorithms?**
240 runs each (20 seeds × 12 periods); success means within 0.1% of SLSQP and balanced.

| Method | Success |
|---|---|
| QAOA + refine | 100% |
| Exhaustive + refine | 100% |
| SA + refine (same QUBO, same budget) | 92.5% |
| PSO / GA with the dependent unit (Gaing 2003) | 100% |
| PSO / GA with a penalty | 13.8% / 7.5% |

**Q26. So is QAOA better than PSO?**
Not in accuracy. Dependent-unit PSO works on continuous variables and reaches the exact optimum. QAOA beats
SA on the same binary problem. The lesson about PSO is that constraint handling matters as much as the
algorithm.

**Q27. Is QAOA faster?**
No. SLSQP takes about 1 ms. Simulated QAOA takes about 22 s of training per period, because simulating
qubits classically costs exponential time. We claim probability amplification and correctness, not
speed-up.

**Q28. Then why use quantum at all?**
ELD is convex and easy; we use it to validate the method end to end. The QUBO route pays off for
problems that are binary by nature, like unit commitment, where classical solvers face combinatorial
growth.

## 5. Hardware

**Q29. What did you run on real hardware?**
IBM `ibm_marrakesh` (156-qubit Heron), job `dc299oanj2vs73dgbsd0`, period 1, p = 1, 4096 shots.

**Q30. Results?**
- 6 qubits (2 bits per unit), 46 two-qubit gates: P(opt) was 3.69% against 1.56% random, about 90% of the
  simulator's 4.11%. Hybrid gap 0%.
- 9 qubits, 144 two-qubit gates: 0.24% against 0.20%, so noise removes most of the gain. The near-optimal
  probability is still 15.5% against 6.1%.

**Q31. Why not 12 qubits on hardware?**
All-to-all coupling needs many SWAPs on heavy-hex hardware. The circuit depth grows quickly and gate
errors accumulate. 9 qubits already shows the limit.

**Q32. How could hardware results improve?**
Error mitigation (readout, ZNE, dynamical decoupling), better qubit layout, and fewer couplings
(sparser encodings).

## 6. Losses and AC validation

**Q33. How are losses modelled?**
Kron's formula P_L = PᵀBP + B₀ᵀP + B₀₀. We derived the coefficients ourselves from the line data:
DC PTDF with slack at bus 1, and losses Σ r·f².

**Q34. How do you put losses in a QUBO, which must stay quadratic?**
Outer iteration: solve the QUBO for P_D + P_L^(k), refine with the loss-inclusive balance, update P_L,
and repeat. It converged in 2–3 rounds.

**Q35. Loss results?**
- Losses are 2.7–9.2 MW and add 2.0% to the daily cost.
- The hybrid stays within 0.08% of the loss-inclusive optimum.

**Q36. Why Newton–Raphson?**
To check the dispatch on the real AC network: voltages, reactive power and line flows. Our NR solver
matches PYPOWER on `case6ww` exactly and converged in 4 iterations every time.

**Q37. What did the AC check show?**
- The loss formula is within 1.5–4.3% of AC.
- Voltages are 1.009–1.070 pu, and Q is within limits.
- **But** line T09 (bus 3–6) is overloaded in periods 1 and 7–12, up to 145% at 330 MW.

**Q38. Is that overload QAOA's fault?**
No. The SLSQP optimum overloads it too (142.7%). ELD ignores line limits. The fix is security-constrained
dispatch, which is our future work.

## 7. Tricky questions

**Q39. Is this quantum advantage?**
No, and we don't claim it. It's a correct, validated, honest NISQ demonstration.

**Q40. What is NISQ?**
Noisy Intermediate-Scale Quantum: today's devices with tens to hundreds of noisy qubits and no error
correction.

**Q41. What limits scaling?**
- Qubits grow as n_bits × N units.
- All-to-all penalty couplings mean deep circuits on hardware.
- Classical simulation is exponential.
- Precision is set by the bits per unit.

**Q42. Why 4 bits, not 6?**
6 bits would mean 18 qubits and 262k states. Simulation and training would be much slower, and hardware
far noisier. 4 bits plus a refinement step gives the same final accuracy.

**Q43. How is this different from quantum-inspired PSO?**
QPSO (e.g. Wang 2024) is a classical algorithm inspired by quantum mechanics. We run an actual quantum
circuit, both simulated and on IBM hardware.

**Q44. What would you do next?**
- Line-flow penalties (SCED) to remove the T09 overload.
- Unit commitment.
- IEEE 14/30-bus systems.
- Error mitigation.
- Multi-period ramp constraints.

**Q45. Can someone reproduce your results?**
Yes. Everything is public: the code, data and fixed seeds are on GitHub (`Karthik73965/eld-qaoa`), and
the run order is in the README. 4 validation tests pass.
