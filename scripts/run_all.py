"""Run the full study: classical ELD, QUBO brute force, QAOA (p = 1..P_MAX), sampling and hybrid
refinement for all 12 load periods. Writes results/*.csv and results/summary.json."""
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
from eld.qaoa import QAOARunner, interp_init, qaoa_circuit, sample_counts
from eld.qubo import all_bitstrings, all_energies, build_qubo, to_ising
from eld.refine import refine

N_BITS, PENALTY, P_MAX = 4, 20.0, 4
OBJECTIVE, ALPHA = "cvar", 0.05
RESTARTS, SHOTS, SEED = 8, 4096, 7
TOP_K = 5
OUT = ROOT / "results"


def good_mask(sys_, q, demand, c_ref):
    """Bitstrings that the refinement step turns into a dispatch within 1% of the SLSQP cost."""
    X = all_bitstrings(q.n_vars)
    P = q.pmin + np.stack([(X * q.weights * (q.gen_of == i)).sum(1) for i in range(sys_.n_gen)], 1)
    cost = (sys_.a * P**2 + sys_.b * P + sys_.c).sum(1)
    return (np.abs(P.sum(1) - demand) <= q.step.max()) & (cost <= 1.01 * c_ref)


def run_period(sys_, load):
    demand = load["total"]
    row = {"period": load["period"], "demand": demand}

    t = time.perf_counter()
    P_cl, c_cl = solve_slsqp(sys_, demand)
    row["t_slsqp_ms"] = 1e3 * (time.perf_counter() - t)
    _, c_lam, lam = solve_lambda(sys_, demand)
    row.update({f"{g}_slsqp": v for g, v in zip(sys_.names, P_cl)})
    row.update(cost_slsqp=c_cl, cost_lambda=c_lam, lambda_=lam)

    q = build_qubo(sys_, demand, N_BITS, PENALTY)
    E = all_energies(q)
    X = all_bitstrings(q.n_vars)
    i_opt = int(E.argmin())
    n_opt = int(np.isclose(E, E[i_opt]).sum())
    P_bf = q.decode(X[i_opt])
    row.update({f"{g}_qubo_opt": v for g, v in zip(sys_.names, P_bf)})
    row.update(cost_qubo_opt=gen_cost(sys_, P_bf), mismatch_qubo_opt=P_bf.sum() - demand)

    H, _ = to_ising(q)
    runner = QAOARunner(H, E, objective=OBJECTIVE, alpha=ALPHA)
    good = good_mask(sys_, q, demand, c_cl)
    u = 1 / len(E)
    depth_rows, prev = [], None
    for p in range(1, P_MAX + 1):
        res = runner.train(p, restarts=RESTARTS if prev is None else 1,
                           init=None if prev is None else interp_init(prev["params"], p - 1), seed=SEED)
        pr = res["probs"]
        depth_rows.append({
            "period": load["period"], "demand": demand, "p": p,
            "p_opt": pr[i_opt], "p_opt_random": n_opt * u, "amp_opt": pr[i_opt] / (n_opt * u),
            "p_good": pr[good].sum(), "p_good_random": good.sum() * u, "amp_good": pr[good].sum() / (good.sum() * u),
            "approx_ratio": (E.max() - pr @ E) / (E.max() - E.min()),
            "train_time_s": res["time_s"], "nfev": res["nfev"],
            "params": json.dumps(np.round(res["params"], 5).tolist()),
            "history": json.dumps(np.round(res["history"], 6).tolist()),
        })
        prev = res

    # Measure the deepest trained circuit and keep the lowest-energy measured bitstring.
    counts = sample_counts(runner.H, P_MAX, prev["params"], shots=SHOTS, seed=SEED)
    i_best = min(counts, key=lambda k: E[k])
    P_q = q.decode(X[i_best])
    row.update({f"{g}_qaoa": v for g, v in zip(sys_.names, P_q)})
    row.update(cost_qaoa=gen_cost(sys_, P_q), mismatch_qaoa=P_q.sum() - demand,
               qaoa_found_qubo_opt=bool(np.isclose(E[i_best], E[i_opt])),
               shots_hitting_opt=int(sum(v for k, v in counts.items() if np.isclose(E[k], E[i_opt]))),
               shots_good=int(sum(v for k, v in counts.items() if good[k])),
               train_time_s=sum(r["train_time_s"] for r in depth_rows))

    # Refine the TOP_K lowest-energy measured bitstrings and keep the cheapest balanced dispatch.
    candidates = sorted(counts, key=lambda k: E[k])[:TOP_K]
    refined = [(*refine(sys_, q.decode(X[k]), q.step, demand), rank) for rank, k in enumerate(candidates, 1)]
    P_h, c_h, widen, rank = min(refined, key=lambda r: r[1])
    row.update({f"{g}_hybrid": v for g, v in zip(sys_.names, P_h)})
    row.update(cost_hybrid=c_h, mismatch_hybrid=P_h.sum() - demand, refine_box=widen, refine_rank=rank,
               gap_qaoa_pct=100 * (row["cost_qaoa"] - c_cl) / c_cl,
               gap_hybrid_pct=100 * (c_h - c_cl) / c_cl)
    return row, depth_rows


def main():
    OUT.mkdir(exist_ok=True)
    sys_ = load_system()
    rows, depth = [], []
    for load in sys_.loads:
        t = time.perf_counter()
        row, d = run_period(sys_, load)
        rows.append(row)
        depth += d
        print(f"P{row['period']:>2} {row['demand']} MW | SLSQP {row['cost_slsqp']:.2f} | QAOA {row['cost_qaoa']:.2f} "
              f"(mismatch {row['mismatch_qaoa']:+.2f}) | hybrid {row['cost_hybrid']:.2f} ({row['gap_hybrid_pct']:+.4f}%) "
              f"| amp_opt p{P_MAX} {d[-1]['amp_opt']:.1f}x | {time.perf_counter() - t:.0f}s", flush=True)

    periods = pd.DataFrame(rows)
    depth_df = pd.DataFrame(depth)
    periods.to_csv(OUT / "periods.csv", index=False)
    depth_df.to_csv(OUT / "qaoa_depth.csv", index=False)

    last = depth_df[depth_df.p == P_MAX]
    summary = {
        "config": {"n_bits": N_BITS, "qubits": 3 * N_BITS, "penalty_A": PENALTY, "p_max": P_MAX,
                   "objective": OBJECTIVE, "cvar_alpha": ALPHA, "shots": SHOTS, "seed": SEED, "refine_top_k": TOP_K},
        "total_cost_slsqp": periods.cost_slsqp.sum(),
        "total_cost_qaoa_raw": periods.cost_qaoa.sum(),
        "total_cost_hybrid": periods.cost_hybrid.sum(),
        "max_abs_gap_hybrid_pct": periods.gap_hybrid_pct.abs().max(),
        "max_abs_mismatch_qaoa_mw": periods.mismatch_qaoa.abs().max(),
        "periods_qaoa_found_qubo_opt": int(periods.qaoa_found_qubo_opt.sum()),
        "mean_amp_opt_by_p": depth_df.groupby("p").amp_opt.mean().to_dict(),
        "mean_amp_good_by_p": depth_df.groupby("p").amp_good.mean().to_dict(),
        "mean_p_good_pmax": last.p_good.mean(),
        "mean_p_good_random": last.p_good_random.mean(),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
