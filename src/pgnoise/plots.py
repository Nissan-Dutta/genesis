"""Figures. Every function takes tidy tables produced by the analysis modules and writes one PNG."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

METHOD_STYLE = {
    "percentile": ("Bootstrap percentile (naive)", "#d62728"),
    "marginal_single_step": ("Pairwise max-t, marginal, single-step", "#1f77b4"),
    "marginal_stepdown": ("Pairwise max-t, marginal, step-down", "#9ecae1"),
    "simultaneous_single_step": ("Pairwise max-t, simultaneous, single-step", "#a1d99b"),
    "simultaneous_stepdown": ("Pairwise max-t, simultaneous, step-down", "#2ca02c"),
}


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def reproduction(rep: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, col, label in [
        (axes[0], "Average_Spearman", "Average Spearman"),
        (axes[1], "Bootstrap_standard_error_Spearman", "Bootstrap SE of gap to #1"),
    ]:
        x, y = rep[f"{col}__published"], np.round(rep[f"{col}__ours"], 3)
        ax.scatter(x, y, s=14, alpha=0.8)
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], color="grey", lw=0.8, ls="--")
        n_exact = int((np.abs(x - y) < 1e-9).sum())
        ax.set(xlabel=f"Published {label}", ylabel=f"Recomputed {label}",
               title=f"{label}: {n_exact}/{len(x)} identical at 3 dp")
    fig.suptitle("Reproducing the ProteinGym DMS-substitution leaderboard (97 models)")
    return _save(fig, path)


def rank_intervals(ru: pd.DataFrame, path: Path, top: int = 40) -> Path:
    d = ru.head(top)
    fig, ax = plt.subplots(figsize=(9, 0.26 * top + 1.5))
    y = np.arange(len(d))
    spec = [("pct", "percentile", -0.25), ("marg", "marginal_single_step", 0.0),
            ("simul", "simultaneous_stepdown", 0.25)]
    for prefix, method, off in spec:
        label, color = METHOD_STYLE[method]
        ax.hlines(y + off, d[f"{prefix}_lo"], d[f"{prefix}_hi"], color=color, lw=2.2, label=label)
    ax.scatter(d["rank"], y, color="black", s=12, zorder=3, label="Published rank")
    ax.set_yticks(y, [f"{m}{' *' if b else ''}" for m, b in zip(d.index, d["in_best_set"])], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Rank (95% interval)")
    ax.set_title(f"Rank uncertainty, top {top} models (* = cannot be ruled out as #1)")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="upper right", fontsize=7.5)
    return _save(fig, path)


def pairwise_heatmap(z: pd.DataFrame, sig: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9.5, 8))
    lim = 4
    im = ax.imshow(np.clip(z.to_numpy(), -lim, lim), cmap="RdBu_r", vmin=-lim, vmax=lim)
    n = len(z)
    for i in range(n):
        for j in range(n):
            if sig.iat[i, j]:
                ax.text(j, i, "•", ha="center", va="center", fontsize=9, color="black")
    ax.set_xticks(range(n), z.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(n), z.index, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.75, label="z = (row - column) / SE (clipped at ±4)")
    ax.set_title("Top 20 pairwise differences (• = significant after simultaneous max-t correction)")
    return _save(fig, path)


def leave_one_group_out(table: pd.DataFrame, path: Path, models: list[str]) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(table))
    for m in models:
        ax.plot(x, table[m], marker="o", lw=1.4, label=m)
    ax.set_xticks(x, [f"drop {c}" if c != "(none)" else "all groups" for c in table.index], rotation=20)
    ax.set_ylabel("Rank")
    ax.invert_yaxis()
    ax.set_yticks(range(1, int(table[models].to_numpy().max()) + 1))
    ax.set_title("Leaderboard rank when one function group is left out")
    ax.legend(fontsize=7.5, ncol=2, loc="lower left")
    ax.grid(alpha=0.3)
    return _save(fig, path)


def sim_coverage_by_rank(per_model: dict[str, pd.DataFrame], path: Path,
                         methods: tuple[str, ...] = ("percentile", "marginal_single_step", "marginal_stepdown",
                                                     "simultaneous_stepdown")) -> Path:
    names = list(per_model)
    fig, axes = plt.subplots(2, len(names), figsize=(4.2 * len(names), 7), sharex=True, squeeze=False)
    for col, name in enumerate(names):
        pm = per_model[name]
        for method in methods:
            label, color = METHOD_STYLE[method]
            d = pm[pm.method == method].sort_values(["true_rank_min", "model"])
            pos = np.arange(1, len(d) + 1)
            axes[0, col].plot(pos, d["coverage"], color=color, lw=1.2, label=label)
            axes[1, col].plot(pos, d["mean_width"], color=color, lw=1.2, label=label)
        axes[0, col].axhline(0.95, color="grey", ls="--", lw=0.8)
        axes[0, col].set(title=name, ylim=(0, 1.02))
        axes[1, col].set(xlabel="True rank position")
    axes[0, 0].set_ylabel("Coverage of true rank (set)")
    axes[1, 0].set_ylabel("Mean interval width (ranks)")
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Coverage simulation: per-model coverage and width by true rank")
    return _save(fig, path)


def sim_summary(summary: pd.DataFrame, best: pd.DataFrame, path: Path) -> Path:
    scenarios = list(dict.fromkeys(summary["scenario"]))
    methods = list(METHOD_STYLE)
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    w = 0.8 / len(methods)
    x = np.arange(len(scenarios))
    for k, method in enumerate(methods):
        label, color = METHOD_STYLE[method]
        d = summary[summary.method == method].set_index("scenario").loc[scenarios]
        for ax, col in zip(axes[:3], ["mean_coverage", "min_model_coverage", "joint_coverage"]):
            ax.bar(x + (k - 2) * w, d[col], width=w, color=color, label=label)
    for ax, title in zip(axes[:3], ["Average per-model coverage", "Worst-model coverage", "Joint coverage (all models)"]):
        ax.axhline(0.95, color="grey", ls="--", lw=0.8)
        ax.set(title=title, ylim=(0, 1.05))
        ax.set_xticks(x, scenarios, rotation=25, fontsize=8)
    b = best.set_index("scenario").loc[scenarios]
    axes[3].bar(x - 0.27, b["best_set_per_true_best"], width=0.27, color="#9ecae1",
                label="Max-t best set, per true #1")
    axes[3].bar(x, b["best_set_coverage"], width=0.27, color="#1f77b4", label="Max-t best set, all true #1s")
    axes[3].bar(x + 0.27, b["naive_best_set_coverage"], width=0.27, color="#d62728",
                label="Within 1.96 SE of #1, all true #1s")
    axes[3].axhline(0.95, color="grey", ls="--", lw=0.8)
    axes[3].set(title="Best-model set contains the true #1", ylim=(0, 1.05))
    axes[3].set_xticks(x, scenarios, rotation=25, fontsize=8)
    h0, l0 = axes[0].get_legend_handles_labels()
    h3, l3 = axes[3].get_legend_handles_labels()
    fig.legend(h0 + h3, l0 + l3, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.16))
    fig.suptitle("Coverage simulation summary (nominal 95%)")
    return _save(fig, path)
