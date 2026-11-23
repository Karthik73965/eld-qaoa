"""Standard metaheuristic baselines used in the ELD literature.

GA and PSO work on the continuous ELD (P in MW). Two standard ways of handling the
power-balance equality are provided:
  * "penalty":   search all n outputs, add BALANCE_PENALTY * (sum P - P_D)^2
  * "dependent": search n-1 outputs, the last generator supplies P_D - sum(others)
                 and is penalised only if that violates its limits (dependent-unit method)
Simulated annealing works on the same QUBO bitstrings as QAOA, followed by the same refinement.
"""
import numpy as np

from .data import System

BALANCE_PENALTY = 1e4   # $/MW^2


def eld_objective(sys: System, demand, mode):
    """Returns (objective over rows of x, lower bounds, upper bounds, x -> P in MW)."""
    if mode == "penalty":
        def to_P(x):
            return np.atleast_2d(x)

        def obj(x):
            P = to_P(x)
            return (sys.a * P**2 + sys.b * P + sys.c).sum(1) + BALANCE_PENALTY * (P.sum(1) - demand) ** 2
        return obj, sys.pmin, sys.pmax, to_P

    if mode == "dependent":
        def to_P(x):
            x = np.atleast_2d(x)
            return np.hstack([x, demand - x.sum(1, keepdims=True)])

        def obj(x):
            P = to_P(x)
            viol = np.maximum(sys.pmin[-1] - P[:, -1], 0) + np.maximum(P[:, -1] - sys.pmax[-1], 0)
            return (sys.a * P**2 + sys.b * P + sys.c).sum(1) + BALANCE_PENALTY * viol**2
        return obj, sys.pmin[:-1], sys.pmax[:-1], to_P

    raise ValueError(mode)


def pso(obj, lo, hi, n_particles=30, iters=100, w=(0.9, 0.4), c1=2.0, c2=2.0, seed=0):
    """Particle swarm optimisation with linearly decreasing inertia weight (Shi & Eberhart, 1998)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(lo, hi, (n_particles, len(lo)))
    v = np.zeros_like(x)
    vmax = 0.2 * (hi - lo)
    f = obj(x)
    pbest, pbest_f = x.copy(), f.copy()
    g = pbest[f.argmin()].copy()
    for t in range(iters):
        wt = w[0] - (w[0] - w[1]) * t / (iters - 1)
        r1, r2 = rng.random(x.shape), rng.random(x.shape)
        v = np.clip(wt * v + c1 * r1 * (pbest - x) + c2 * r2 * (g - x), -vmax, vmax)
        x = np.clip(x + v, lo, hi)
        f = obj(x)
        better = f < pbest_f
        pbest[better], pbest_f[better] = x[better], f[better]
        g = pbest[pbest_f.argmin()].copy()
    return g, n_particles * (iters + 1)


def ga(obj, lo, hi, pop=30, gens=100, pc=0.9, pm=0.1, eta_c=15, seed=0):
    """Real-coded GA: tournament selection, SBX crossover (Deb & Agrawal 1995), Gaussian mutation, elitism."""
    rng = np.random.default_rng(seed)
    n = len(lo)
    x = rng.uniform(lo, hi, (pop, n))
    f = obj(x)
    for _ in range(gens):
        # binary tournament
        i, j = rng.integers(pop, size=(2, pop))
        parents = np.where((f[i] < f[j])[:, None], x[i], x[j])
        children = parents.copy()
        for k in range(0, pop - 1, 2):
            if rng.random() < pc:
                u = rng.random(n)
                beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta_c + 1)), (1 / (2 * (1 - u))) ** (1 / (eta_c + 1)))
                p1, p2 = parents[k], parents[k + 1]
                children[k] = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
                children[k + 1] = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
        mutate = rng.random(children.shape) < pm
        children += mutate * rng.normal(0, 0.05 * (hi - lo), children.shape)
        children = np.clip(children, lo, hi)
        fc = obj(children)
        # elitism: keep the best individual of the previous generation
        best = f.argmin()
        worst = fc.argmax()
        children[worst], fc[worst] = x[best], f[best]
        x, f = children, fc
    return x[f.argmin()], pop * (gens + 1)


def simulated_annealing(energy_fn, n_vars, n_evals=4096, t0=None, t1=None, seed=0, top_k=1):
    """Single-bit-flip simulated annealing (Kirkpatrick et al., 1983) with geometric cooling.
    Also returns the top_k lowest-energy distinct states visited, so SA candidates can be
    refined exactly like the top_k sampled QAOA bitstrings."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, n_vars).astype(float)
    e = energy_fn(x)
    best_x, best_e = x.copy(), e
    seen = {tuple(x): e}
    if t0 is None:  # initial temperature from typical single-flip energy changes
        deltas = []
        for j in range(n_vars):
            y = x.copy(); y[j] = 1 - y[j]
            deltas.append(abs(energy_fn(y) - e))
        t0 = float(np.mean(deltas))
        t1 = t1 or t0 * 1e-4
    alpha = (t1 / t0) ** (1 / max(n_evals - 1, 1))
    T = t0
    for _ in range(n_evals - 1):
        j = rng.integers(n_vars)
        y = x.copy(); y[j] = 1 - y[j]
        ey = energy_fn(y)
        seen[tuple(y)] = ey
        if ey <= e or rng.random() < np.exp(-(ey - e) / T):
            x, e = y, ey
            if e < best_e:
                best_x, best_e = x.copy(), e
        T *= alpha
    top = [np.array(s) for s in sorted(seen, key=seen.get)[:top_k]]
    return best_x, best_e, n_evals, top
