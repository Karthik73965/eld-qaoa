"""Hybrid step: restore exact power balance around the dispatch chosen by QAOA.

QAOA picks a point on the binary grid. Because the grid is coarse (dP_i = 7.5-10 MW with
4 bits), that point usually misses P_D by a few MW. Each generator is then allowed to move
at most one grid step (+/- dP_i) from its QAOA level, and the small continuous problem is
solved classically. The box is widened only if balance is impossible inside it.
With `loss` given, the balance is sum(P) = P_D + P_L(P).
"""
import numpy as np

from .classical import solve_slsqp, solve_slsqp_losses
from .data import System


def refine(sys: System, P_qaoa, step, demand, max_widen=4, loss=None):
    P_qaoa = np.asarray(P_qaoa, dtype=float)
    for k in range(1, max_widen + 1):
        lo = np.maximum(sys.pmin, P_qaoa - k * step)
        hi = np.minimum(sys.pmax, P_qaoa + k * step)
        if loss is None:
            if lo.sum() <= demand <= hi.sum():
                P, cost = solve_slsqp(sys, demand, lo, hi)
                return P, cost, k
        else:
            # balance reachable inside the box? (loss changes little across one grid step)
            if lo.sum() - loss(lo) <= demand <= hi.sum() - loss(hi):
                try:
                    P, cost = solve_slsqp_losses(sys, demand, loss, lo, hi, x0=P_qaoa)
                    return P, cost, k
                except RuntimeError:
                    continue
    raise RuntimeError("demand not reachable near the QAOA dispatch")
