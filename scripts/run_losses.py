"""Extension: ELD with transmission losses, validated by Newton-Raphson AC load flow.

For every period:
  1. Loss formula P_L = P'BP + B0'P + B00 built from the 11-line data (DC flows x line resistance).
  2. Classical loss-inclusive ELD (SLSQP with sum P = P_D + P_L(P)).
  3. Hybrid QAOA with losses: QUBO demand is updated iteratively to P_D + P_L(P_prev)
     (the standard loss-update iteration), refinement enforces the loss-inclusive balance.
  4. Newton-Raphson AC load flow of the lossless and loss-inclusive dispatches: actual losses,
     slack-bus output, bus voltages, line loadings and generator MVAr vs limits.
Writes results/losses.csv.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eld.classical import solve_slsqp, solve_slsqp_losses
from eld.data import gen_cost, load_system
from eld.network import LossModel, build_network, bus_loads, newton_raphson
from eld.qaoa import QAOARunner, interp_init, sample_counts
from eld.qubo import all_bitstrings, all_energies, build_qubo, to_ising
from eld.refine import refine

N_BITS, PENALTY, P_QAOA, SHOTS, TOP_K, SEED, LOSS_ITERS = 4, 20.0, 3, 4096, 5, 7, 4
OUT = ROOT / "results"


def hybrid_with_losses(sys_, demand, loss):
    P_prev, history = None, []
    for it in range(LOSS_ITERS):
        pl_est = 0.0 if P_prev is None else loss(P_prev)
        q = build_qubo(sys_, demand + pl_est, N_BITS, PENALTY)
        E = all_energies(q)
        X = all_bitstrings(q.n_vars)
        runner = QAOARunner(to_ising(q)[0], E, objective="cvar", alpha=0.05)
        res = runner.train(1, restarts=8, seed=SEED)
        for p in range(2, P_QAOA + 1):
            res = runner.train(p, restarts=1, init=interp_init(res["params"], p - 1), seed=SEED)
        counts = sample_counts(runner.H, P_QAOA, res["params"], shots=SHOTS, seed=SEED)
        cands = sorted(counts, key=lambda k: E[k])[:TOP_K]
        refined = [refine(sys_, q.decode(X[k]), q.step, demand, loss=loss)[:2] for k in cands]
        P, c = min(refined, key=lambda r: r[1])
        history.append(loss(P))
        if P_prev is not None and abs(loss(P) - loss(P_prev)) < 0.05:
            break
        P_prev = P
    return P, c, it + 1


def main():
    sys_ = load_system()
    net = build_network(sys_)
    ratings = np.array([b["rating_mva"] for b in net.branches])
    ste = np.array([b["ste_rating_mva"] for b in net.branches])
    rows = []
    for load in sys_.loads:
        period, demand = load["period"], load["total"]
        Pd = bus_loads(net, load)
        loss = LossModel(net, Pd)
        P0, c0 = solve_slsqp(sys_, demand)
        P1, c1 = solve_slsqp_losses(sys_, demand, loss)
        t = time.perf_counter()
        Ph, ch, iters = hybrid_with_losses(sys_, demand, loss)
        t_h = time.perf_counter() - t
        row = {"period": period, "demand": demand,
               **{f"G{i+1}_lossless": v for i, v in enumerate(P0)}, "cost_lossless": c0,
               **{f"G{i+1}_slsqp_loss": v for i, v in enumerate(P1)}, "cost_slsqp_loss": c1,
               "ploss_formula_slsqp": loss(P1),
               **{f"G{i+1}_hybrid_loss": v for i, v in enumerate(Ph)}, "cost_hybrid_loss": ch,
               "ploss_formula_hybrid": loss(Ph), "gap_hybrid_loss_pct": 100 * (ch - c1) / c1,
               "loss_iterations": iters, "time_hybrid_s": t_h}
        for tag, P in (("lossless", P0), ("slsqp_loss", P1), ("hybrid_loss", Ph)):
            lf = newton_raphson(net, sys_, P, Pd)
            row.update({
                f"nr_{tag}_converged": lf.converged, f"nr_{tag}_iterations": lf.iterations,
                f"nr_{tag}_losses_mw": lf.losses_mw, f"nr_{tag}_slack_mw": lf.P_gen[0],
                f"nr_{tag}_slack_dev_mw": lf.P_gen[0] - P[0],
                f"nr_{tag}_vmin_pu": float(np.abs(lf.V).min()), f"nr_{tag}_vmax_pu": float(np.abs(lf.V).max()),
                f"nr_{tag}_max_line_loading_pct": float(np.max(lf.flows_mva / ratings) * 100),
                f"nr_{tag}_max_loaded_line": net.branches[int(np.argmax(lf.flows_mva / ratings))]["id"],
                f"nr_{tag}_lines_over_ste": int(np.sum(lf.flows_mva > ste)),
                f"nr_{tag}_voltage_ok": lf.voltage_ok, f"nr_{tag}_lines_ok": lf.lines_ok, f"nr_{tag}_q_ok": lf.q_ok,
                f"nr_{tag}_cost_actual": gen_cost(sys_, lf.P_gen),
            })
        rows.append(row)
        print(f"P{period:>2} {demand} MW | loss formula {row['ploss_formula_slsqp']:.2f} MW, NR {row['nr_slsqp_loss_losses_mw']:.2f} MW "
              f"| cost lossless {c0:.2f} -> with losses {c1:.2f}, hybrid {ch:.2f} ({row['gap_hybrid_loss_pct']:+.4f}%) "
              f"| NR slack dev {row['nr_hybrid_loss_slack_dev_mw']:+.2f} MW | V {row['nr_hybrid_loss_vmin_pu']:.3f}-{row['nr_hybrid_loss_vmax_pu']:.3f} "
              f"| lines {row['nr_hybrid_loss_max_line_loading_pct']:.0f}% ({row['nr_hybrid_loss_max_loaded_line']}) "
              f"| Q ok {row['nr_hybrid_loss_q_ok']}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "losses.csv", index=False)


if __name__ == "__main__":
    main()
