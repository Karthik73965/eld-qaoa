"""Classical ELD solvers: SciPy SLSQP (benchmark) and lambda iteration (textbook cross-check)."""
import numpy as np
from scipy.optimize import minimize

from .data import System, gen_cost


def solve_slsqp(sys: System, demand, lo=None, hi=None):
    """Minimise total cost s.t. sum(P) = demand and lo <= P <= hi (defaults: generator limits)."""
    lo = sys.pmin if lo is None else lo
    hi = sys.pmax if hi is None else hi
    x0 = np.clip(np.full(sys.n_gen, demand / sys.n_gen), lo, hi)
    res = minimize(
        lambda P: gen_cost(sys, P),
        x0,
        jac=lambda P: 2 * sys.a * P + sys.b,
        bounds=list(zip(lo, hi)),
        constraints=[{"type": "eq", "fun": lambda P: P.sum() - demand, "jac": lambda P: np.ones_like(P)}],
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 500},
    )
    if not res.success:
        raise RuntimeError(f"SLSQP failed: {res.message}")
    return res.x, gen_cost(sys, res.x)


def solve_slsqp_losses(sys: System, demand, loss, lo=None, hi=None, x0=None):
    """Minimise total cost s.t. sum(P) = demand + P_L(P) (Kron loss formula) and lo <= P <= hi."""
    lo = sys.pmin if lo is None else lo
    hi = sys.pmax if hi is None else hi
    # start from the lossless optimum scaled up by the estimated losses (several starts for robustness)
    P_ll = solve_slsqp(sys, demand, lo, hi)[0] if lo.sum() <= demand <= hi.sum() else (lo + hi) / 2
    starts = [P_ll * (1 + loss(P_ll) / demand), P_ll, (lo + hi) / 2]
    if x0 is not None:
        starts.insert(0, x0)
    for start in starts:
        res = minimize(
            lambda P: gen_cost(sys, P),
            np.clip(start, lo, hi),
            jac=lambda P: 2 * sys.a * P + sys.b,
            bounds=list(zip(lo, hi)),
            constraints=[{"type": "eq", "fun": lambda P: P.sum() - demand - loss(P),
                          "jac": lambda P: np.ones_like(P) - loss.grad(P)}],
            method="SLSQP",
            options={"ftol": 1e-10, "maxiter": 500},
        )
        if res.success and abs(res.x.sum() - demand - loss(res.x)) < 1e-6:
            return res.x, gen_cost(sys, res.x)
    raise RuntimeError(f"SLSQP (losses) failed: {res.message}")


def solve_lambda(sys: System, demand, tol=1e-9):
    """Equal-incremental-cost dispatch by bisection on lambda, respecting generator limits."""
    def output(lam):
        return np.clip((lam - sys.b) / (2 * sys.a), sys.pmin, sys.pmax)

    lo = float(np.min(2 * sys.a * sys.pmin + sys.b))
    hi = float(np.max(2 * sys.a * sys.pmax + sys.b))
    while hi - lo > tol:
        lam = 0.5 * (lo + hi)
        if output(lam).sum() < demand:
            lo = lam
        else:
            hi = lam
    lam = 0.5 * (lo + hi)
    P = output(lam)
    return P, gen_cost(sys, P), lam
