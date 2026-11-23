"""Penalty-weight study: exhaustive QUBO optimum for A in {1, 5, 20, 100} and every period.
Writes results/penalty_sweep.csv."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eld.classical import solve_slsqp
from eld.data import gen_cost, load_system
from eld.qubo import all_bitstrings, all_energies, build_qubo

sys_ = load_system()
X = all_bitstrings(12)
rows = []
for A in (1, 5, 20, 100):
    for load in sys_.loads:
        d = load["total"]
        q = build_qubo(sys_, d, 4, A)
        P = q.decode(X[all_energies(q).argmin()])
        c_ref = solve_slsqp(sys_, d)[1]
        rows.append({"A": A, "period": load["period"], "demand": d, "mismatch_mw": P.sum() - d,
                     "cost": gen_cost(sys_, P), "gap_pct": 100 * (gen_cost(sys_, P) - c_ref) / c_ref})
df = pd.DataFrame(rows)
df.to_csv(ROOT / "results" / "penalty_sweep.csv", index=False)
print(df.groupby("A").agg(max_abs_mismatch=("mismatch_mw", lambda m: m.abs().max()),
                          mean_gap_pct=("gap_pct", "mean")).round(4))
