"""Load the 6-bus test system (Prada 2017, Appendix D) and evaluate generation cost."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "sixbus_prada2017.json"


@dataclass
class System:
    names: list
    pmin: np.ndarray   # MW
    pmax: np.ndarray   # MW
    a: np.ndarray      # $/MW^2 h
    b: np.ndarray      # $/MWh
    c: np.ndarray      # $/h
    loads: list        # one dict per period: period, L4, L5, L6, total
    raw: dict          # full JSON (bus/branch data for the losses extension)

    @property
    def n_gen(self):
        return len(self.names)


def load_system(path=DATA_FILE) -> System:
    raw = json.loads(Path(path).read_text())
    g = raw["generators"]
    col = lambda key: np.array([x[key] for x in g], dtype=float)
    return System(
        names=[x["id"] for x in g],
        pmin=col("pmin_mw"), pmax=col("pmax_mw"),
        a=col("a"), b=col("b"), c=col("c"),
        loads=raw["loads_mw"], raw=raw,
    )


def gen_cost(sys: System, P) -> float:
    """Total generation cost  sum_i a_i P_i^2 + b_i P_i + c_i  in $/h."""
    P = np.asarray(P, dtype=float)
    return float(np.sum(sys.a * P**2 + sys.b * P + sys.c))


def incremental_cost(sys: System, P) -> np.ndarray:
    """dC_i/dP_i = 2 a_i P_i + b_i  in $/MWh."""
    return 2 * sys.a * np.asarray(P, dtype=float) + sys.b
