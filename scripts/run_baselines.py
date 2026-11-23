"""Compare hybrid QAOA with standard ELD methods over all 12 periods and 20 random seeds.

Methods: SLSQP and lambda iteration (deterministic), exhaustive QUBO search, PSO, GA,
simulated annealing on the QUBO, and QAOA (trained angles from run_all.py, re-sampled per seed).
Writes results/baselines.csv and results/baselines_summary.csv.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eld.classical import solve_lambda, solve_slsqp
from eld.data import gen_cost, load_system
from eld.metaheuristics import eld_objective, ga, pso, simulated_annealing
from eld.qaoa import sample_counts, scaled_hamiltonian
from eld.qubo import all_bitstrings, all_energies, build_qubo, to_ising
from eld.refine import refine

N_BITS, PENALTY, P_QAOA, SHOTS, TOP_K, SEEDS = 4, 20.0, 4, 4096, 5, range(20)
OUT = ROOT / "results"


def best_refined(sys_, q, X, candidates, demand):
    refined = [refine(sys_, q.decode(X[k]), q.step, demand)[:2] for k in candidates]
    return min(refined, key=lambda r: r[1])


def main():
    sys_ = load_system()
    depth = pd.read_csv(OUT / "qaoa_depth.csv")
    rows = []

    def record(period, demand, method, seed, P, t, evals, c_ref):
        c = gen_cost(sys_, P)
        rows.append({"period": period, "demand": demand, "method": method, "seed": seed,
                     "cost": c, "gap_pct": 100 * (c - c_ref) / c_ref, "mismatch_mw": float(np.sum(P) - demand),
                     "time_ms": 1e3 * t, "evaluations": evals})

    for load in sys_.loads:
        period, demand = load["period"], load["total"]
        t = time.perf_counter(); P_ref, c_ref = solve_slsqp(sys_, demand)
        record(period, demand, "SLSQP", 0, P_ref, time.perf_counter() - t, np.nan, c_ref)
        t = time.perf_counter(); P_l, _, _ = solve_lambda(sys_, demand)
        record(period, demand, "Lambda iteration", 0, P_l, time.perf_counter() - t, np.nan, c_ref)

        q = build_qubo(sys_, demand, N_BITS, PENALTY)
        X = all_bitstrings(q.n_vars)
        t = time.perf_counter()
        E = all_energies(q)
        P_bf, _ = best_refined(sys_, q, X, np.argsort(E)[:TOP_K], demand)
        record(period, demand, "Exhaustive QUBO + refine", 0, P_bf, time.perf_counter() - t, len(E), c_ref)

        H_s, _ = scaled_hamiltonian(to_ising(q)[0])
        params = json.loads(depth[(depth.period == period) & (depth.p == P_QAOA)].params.iloc[0])
        pow2 = 2 ** np.arange(q.n_vars)

        for s in SEEDS:
            for mode, label in (("penalty", "penalty"), ("dependent", "dependent unit")):
                obj, lo, hi, to_P = eld_objective(sys_, demand, mode)
                for name, algo in (("PSO", pso), ("GA", ga)):
                    t = time.perf_counter(); x, ev = algo(obj, lo, hi, seed=s)
                    record(period, demand, f"{name} ({label})", s, to_P(x)[0], time.perf_counter() - t, ev, c_ref)

            t = time.perf_counter()
            _, _, ev, top = simulated_annealing(lambda x: E[int(x @ pow2)], q.n_vars, n_evals=SHOTS,
                                                seed=s, top_k=TOP_K)
            P, _ = best_refined(sys_, q, X, [int(x @ pow2) for x in top], demand)
            record(period, demand, "SA (QUBO) + refine", s, P, time.perf_counter() - t, ev, c_ref)

            t = time.perf_counter()
            counts = sample_counts(H_s, P_QAOA, params, shots=SHOTS, seed=100 + s)
            P, _ = best_refined(sys_, q, X, sorted(counts, key=lambda k: E[k])[:TOP_K], demand)
            record(period, demand, "QAOA (QUBO) + refine", s, P, time.perf_counter() - t, SHOTS, c_ref)
        print(f"period {period} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "baselines.csv", index=False)
    ok = (df.gap_pct <= 0.1) & (df.mismatch_mw.abs() <= 0.01)
    summary = df.assign(success=ok).groupby("method", sort=False).agg(
        mean_gap_pct=("gap_pct", "mean"), worst_gap_pct=("gap_pct", "max"), std_gap_pct=("gap_pct", "std"),
        max_abs_mismatch_mw=("mismatch_mw", lambda m: m.abs().max()), success_rate=("success", "mean"),
        mean_time_ms=("time_ms", "mean"), evaluations=("evaluations", "first"), runs=("cost", "size"))
    summary.to_csv(OUT / "baselines_summary.csv")
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(summary)


if __name__ == "__main__":
    main()
