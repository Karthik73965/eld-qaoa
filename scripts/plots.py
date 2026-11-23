"""Make report/slide figures from results/*.csv and results/*.json -> results/figures/*.png"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = RES / "figures"

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#ffffff", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"   # blue, orange, aqua, violet (fixed order)
GEN_COLORS = [S1, S2, S3]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "legend.frameon": False, "legend.labelcolor": INK2, "lines.linewidth": 2, "lines.markersize": 7,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=200)
    plt.close(fig)


def dispatch(periods):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = periods.period.values
    bottom = np.zeros(len(x))
    for g, col in zip(["G1", "G2", "G3"], GEN_COLORS):
        vals = periods[f"{g}_hybrid"].values
        ax.bar(x, vals, bottom=bottom, width=0.7, color=col, label=g, edgecolor=SURFACE, linewidth=1.5)
        bottom += vals
    for xi, d in zip(x, periods.demand):
        ax.text(xi, d + 4, f"{d}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x)
    ax.set_xlabel("Load period")
    ax.set_ylabel("Generation (MW)")
    ax.set_ylim(0, periods.demand.max() * 1.12)
    ax.set_title("Hybrid QAOA dispatch for the 12-period load profile")
    ax.legend(ncol=3, loc="upper left")
    save(fig, "dispatch_by_period.png")


def cost_gap(periods):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = periods.period.values
    w = 0.38
    ax.bar(x - w / 2, periods.gap_qaoa_pct, w, color=S2, label="QAOA (raw grid point)", edgecolor=SURFACE)
    ax.bar(x + w / 2, periods.gap_hybrid_pct, w, color=S1, label="QAOA + refinement (hybrid)", edgecolor=SURFACE)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_xticks(x)
    ax.set_xlabel("Load period")
    ax.set_ylabel("Cost difference vs SLSQP (%)")
    ax.set_title("Operating cost relative to the classical SLSQP optimum")
    ax.legend(loc="upper right")
    fig.text(0.01, 0.01, "Negative raw values are not savings: those grid points supply 0.5 MW less than demand.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "cost_gap.png", dpi=200)
    plt.close(fig)


def mismatch(periods):
    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = periods.period.values
    w = 0.38
    ax.bar(x - w / 2, periods.mismatch_qaoa, w, color=S2, label="QAOA (raw)", edgecolor=SURFACE)
    ax.bar(x + w / 2, periods.mismatch_hybrid, w, color=S1, label="Hybrid", edgecolor=SURFACE)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_xticks(x)
    ax.set_xlabel("Load period")
    ax.set_ylabel("ΣP − P_D (MW)")
    ax.set_title("Power-balance mismatch")
    ax.legend(loc="upper right")
    save(fig, "mismatch.png")


def amplification(depth):
    g = depth.groupby("p").agg(opt=("amp_opt", "mean"), good=("amp_good", "mean"))
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(g.index, g.opt, "-o", color=S1, label="Optimal bitstring")
    ax.plot(g.index, g.good, "-o", color=S2, label="Near-optimal (≤1% after refinement)")
    ax.axhline(1, color=MUTED, linestyle="--", linewidth=1.2)
    ax.text(g.index.max(), 1.15, "random sampling = 1×", ha="right", va="bottom", fontsize=8, color=MUTED)
    for p_, v in g.opt.items():
        ax.annotate(f"{v:.1f}×", (p_, v), textcoords="offset points", xytext=(0, -16), ha="center", fontsize=8, color=INK2)
    ax.set_xticks(g.index)
    ax.set_xlabel("QAOA depth p")
    ax.set_ylabel("Probability gain over random (×)")
    ax.set_ylim(0, max(g.opt.max(), g.good.max()) * 1.25)
    ax.set_title("QAOA concentrates probability on good dispatches")
    ax.legend(loc="upper left")
    save(fig, "amplification_vs_depth.png")


def p_good(depth):
    g = depth.groupby("p").agg(q=("p_good", "mean"), r=("p_good_random", "mean"))
    fig, ax = plt.subplots(figsize=(6, 3.8))
    labels = ["Random"] + [f"QAOA p={p}" for p in g.index]
    vals = [g.r.iloc[0] * 100] + list(g.q * 100)
    cols = [MUTED] + [S1] * len(g)
    ax.bar(labels, vals, color=cols, width=0.6, edgecolor=SURFACE)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.4, f"{v:.1f}%", ha="center", fontsize=9, color=INK2)
    ax.set_ylabel("Shots landing near-optimal (%)")
    ax.set_title("Chance one measurement is usable (mean of 12 periods)")
    save(fig, "p_good.png")


def convergence(depth, period=1):
    d = depth[depth.period == period]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for (_, r), col in zip(d.iterrows(), [S1, S2, S3, S4]):
        h = np.minimum.accumulate(json.loads(r.history))
        ax.plot(np.arange(1, len(h) + 1), h, color=col, label=f"p = {r.p}")
    ax.set_yscale("log")
    ax.set_xlabel("Optimizer iteration (COBYLA)")
    ax.set_ylabel("Normalised CVaR cost (log scale)")
    ax.set_title(f"QAOA training convergence, period {period} ({int(d.demand.iloc[0])} MW)")
    ax.legend()
    save(fig, "convergence.png")


def hardware():
    f = RES / "ibm_hardware.json"
    if not f.exists():
        return
    rep = json.loads(f.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, metric, title in zip(axes, ["p_opt", "p_good"], ["P(optimal bitstring)", "P(near-optimal)"]):
        labels = [f"{c['qubits']} qubits" for c in rep["cases"]]
        x = np.arange(len(labels))
        w = 0.26
        for k, (src, col, name) in enumerate([("random", MUTED, "Random"), ("hardware", S2, rep["backend"]),
                                              ("simulator", S1, "Ideal simulator")]):
            vals = [100 * c[src][metric] for c in rep["cases"]]
            ax.bar(x + (k - 1) * w, vals, w, color=col, label=name, edgecolor=SURFACE)
        ax.set_xticks(x, labels)
        ax.set_ylabel("%")
        ax.set_title(title, fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, fontsize=9)
    fig.suptitle(f"Real IBM hardware vs simulator (period {rep['period']}, p = {rep['p']})",
                 x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIG / "ibm_hardware.png", dpi=200)
    plt.close(fig)


def baselines():
    f = RES / "baselines_summary.csv"
    if not f.exists():
        return
    s = pd.read_csv(f)
    s = s[s.runs > 12].sort_values("success_rate")          # stochastic methods, 240 runs each
    fig, ax = plt.subplots(figsize=(8, 3.8))
    cols = [S1 if m.startswith("QAOA") else MUTED for m in s.method]
    ax.barh(s.method, 100 * s.success_rate, color=cols, height=0.6, edgecolor=SURFACE)
    for i, (v, g) in enumerate(zip(s.success_rate, s.mean_gap_pct)):
        ax.text(100 * v + 1, i, f"{100 * v:.1f}%  (mean gap {g:.3f}%)", va="center", fontsize=8, color=INK2)
    ax.set_xlim(0, 135)
    ax.set_xticks(range(0, 101, 20))
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Runs within 0.1% of the SLSQP optimum and meeting demand (%)")
    ax.set_title("Success rate over 240 runs (12 periods × 20 seeds)")
    save(fig, "baselines_success.png")


def losses():
    f = RES / "losses.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    x = d.period.values
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(x - w / 2, d.ploss_formula_hybrid, w, color=S1, label="Loss formula (B-coefficients)", edgecolor=SURFACE)
    ax.bar(x + w / 2, d.nr_hybrid_loss_losses_mw, w, color=S2, label="Newton–Raphson AC load flow", edgecolor=SURFACE)
    ax.set_xticks(x)
    ax.set_xlabel("Load period")
    ax.set_ylabel("Transmission loss (MW)")
    ax.set_title("Losses of the hybrid QAOA dispatch: formula vs AC load flow")
    ax.legend(loc="upper left")
    save(fig, "losses.png")

    fig, ax = plt.subplots(figsize=(8, 3.6))
    load = d.nr_hybrid_loss_max_line_loading_pct
    ax.bar(x, load, 0.6, color=[S2 if v > 100 else S1 for v in load], edgecolor=SURFACE)
    for xi, v in zip(x, load):
        ax.text(xi, v + 3, f"{v:.0f}%", ha="center", fontsize=8, color=INK2, zorder=4,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1))
    for lvl, txt in ((100, "continuous rating 60 MVA"), (125, "short-term rating 75 MVA")):
        ax.axhline(lvl, color=MUTED, linestyle="--", linewidth=1, zorder=1)
        ax.text(3.5, lvl + 2, txt, fontsize=8, color=MUTED, va="bottom", ha="center")
    ax.set_xticks(x)
    ax.set_ylim(0, 165)
    ax.set_xlabel("Load period")
    ax.set_ylabel("Loading of line T09, bus 3–6 (%)")
    ax.set_title("Most loaded line under the hybrid QAOA dispatch (AC load flow)")
    save(fig, "line_loading.png")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    baselines()
    losses()
    periods = pd.read_csv(RES / "periods.csv")
    depth = pd.read_csv(RES / "qaoa_depth.csv")
    dispatch(periods)
    cost_gap(periods)
    mismatch(periods)
    amplification(depth)
    p_good(depth)
    convergence(depth)
    hardware()
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
