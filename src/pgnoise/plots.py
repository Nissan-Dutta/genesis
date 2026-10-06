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
    "marginal_stepdown_t": ("Bootstrap-t, marginal, step-down", "#9467bd"),
    "simultaneous_stepdown_t": ("Bootstrap-t, simultaneous, step-down", "#8c564b"),
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
                                                     "marginal_stepdown_t", "simultaneous_stepdown")) -> Path:
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
            ax.bar(x + (k - (len(methods) - 1) / 2) * w, d[col], width=w, color=color, label=label)
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


def robustness_heatmap(long: pd.DataFrame, path: Path, combos: list[tuple[str, str]], top: int = 10) -> Path:
    """Rank of each model (rows) under each leaderboard (columns); ★ = cannot be ruled out as #1."""
    tables = {c: long[(long.metric == c[0]) & (long.scheme == c[1])].set_index("model") for c in combos}
    base = tables[combos[0]]
    models = []
    for c in combos:
        models += list(tables[c].sort_values("rank").index[:top])
    models = sorted(set(models), key=lambda m: base.loc[m, "rank"])
    R = np.array([[tables[c].loc[m, "rank"] for c in combos] for m in models], dtype=float)
    best = np.array([[tables[c].loc[m, "in_best_set"] for c in combos] for m in models])
    fig, ax = plt.subplots(figsize=(1.0 * len(combos) + 4, 0.32 * len(models) + 2))
    im = ax.imshow(np.minimum(R, 40), cmap="viridis_r", vmin=1, vmax=40, aspect="auto")
    for i in range(len(models)):
        for j in range(len(combos)):
            txt = f"{int(R[i, j])}{'★' if best[i, j] else ''}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if R[i, j] > 18 else "black", fontweight="bold" if best[i, j] else None)
    ax.set_xticks(range(len(combos)), [f"{m}\n{s}" for m, s in combos], fontsize=7.5, rotation=35, ha="right")
    ax.set_yticks(range(len(models)), models, fontsize=7.5)
    ax.axvline(len([c for c in combos if c[1] == combos[0][1]]) - 0.5, color="white", lw=2)
    fig.colorbar(im, ax=ax, shrink=0.6, label="Rank (capped at 40)")
    ax.set_title(f"Rank under other metrics and aggregation schemes (union of top {top}s; ★ = possible #1)")
    return _save(fig, path)


def _short(model: str) -> str:
    return model.replace(" Protein-RAG (16B)", "-RAG")


def robustness_grid(summary: pd.DataFrame, path: Path) -> Path:
    metrics = list(dict.fromkeys(summary.metric))
    schemes = list(dict.fromkeys(summary.scheme))
    tau = summary.pivot(index="metric", columns="scheme", values="kendall_tau_vs_published").loc[metrics, schemes]
    fig, ax = plt.subplots(figsize=(15, 6))
    im = ax.imshow(tau.to_numpy(), cmap="magma", vmin=0.6, vmax=1.0, aspect="auto")
    for i, m in enumerate(metrics):
        for j, s in enumerate(schemes):
            row = summary[(summary.metric == m) & (summary.scheme == s)].iloc[0]
            ax.text(j, i, f"#1 {_short(row.top1)}\npossible #1: {row.best_set_size}\nτ = {row.kendall_tau_vs_published:.2f}",
                    ha="center", va="center", fontsize=7, color="white" if row.kendall_tau_vs_published < 0.85 else "black")
    ax.set_xticks(range(len(schemes)), schemes, fontsize=8)
    ax.set_yticks(range(len(metrics)), metrics, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Kendall τ vs published leaderboard")
    ax.set_title("Leader, size of the possible-#1 set, and agreement with the published ranking")
    return _save(fig, path)


def power_curves(curves: pd.DataFrame, required: pd.DataFrame, current_units: int, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    colors = {0.005: "#d62728", 0.01: "#1f77b4"}
    for delta, color in colors.items():
        d = curves[curves.delta == delta]
        band = d.groupby("units")["analytic_power"]
        ax.fill_between(band.min().index, band.min(), band.max(), color=color, alpha=0.15,
                        label=f"Δ = {delta}: analytic, range over top-10 pairs")
        med = d[d.pair == "median"]
        ax.plot(med.units, med.analytic_power, color=color, lw=2, label=f"Δ = {delta}: analytic, median pair")
        ax.scatter(med.units, med.simulated_power, color=color, s=18, zorder=3, label=f"Δ = {delta}: simulated")
    ax.axhline(0.8, color="grey", ls="--", lw=0.8)
    ax.axvline(current_units, color="black", ls=":", lw=1, label=f"today ({current_units} units)")
    ax.set(xscale="log", xlabel="(UniProt, function) units in the benchmark  (assays ≈ 1.085 × units)",
           ylabel="Power (two-sided, α = 0.05)", title="Power to detect a gain over the current #1", ylim=(0, 1.02))
    ax.legend(fontsize=7, loc="upper left")
    ax = axes[1]
    for pair, d in required.groupby("pair"):
        ax.plot(d.delta, d.units_needed, color="grey", lw=0.8, alpha=0.6)
    med = required.groupby("delta")["units_needed"].median()
    ax.plot(med.index, med.values, color="black", lw=2, label="median over top-10 pairs")
    ax.axhline(current_units, color="black", ls=":", lw=1, label="today")
    for delta in colors:
        ax.axvline(delta, color=colors[delta], ls="--", lw=0.8)
    ax.set(xscale="log", yscale="log", xlabel="True gain Δ (Spearman)", ylabel="Units needed for 80% power",
           title="Benchmark size needed (grey: individual top-10 pairs)")
    ax.legend(fontsize=8)
    return _save(fig, path)


HEADLINE_RC = {
    "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "pdf.fonttype": 42, "ps.fonttype": 42,
}
POSSIBLE_TOP1, OTHER = "#D55E00", "#0072B2"


def _headline_label(model: str) -> str:
    return model.replace(" Protein-RAG (16B)", " Protein-RAG")


def headline(ru: pd.DataFrame, neighbours: pd.DataFrame, per_pair: pd.DataFrame, required: pd.DataFrame,
             n_units: int, n_assays: int, incomplete: dict[str, int], paths: list[Path], top: int = 20,
             rank_panel: int = 7) -> list[Path]:
    """Single-column figure: (a) power / minimum detectable difference; (b) compact rank-interval support."""
    assays_per_unit = n_assays / n_units
    d = ru.head(rank_panel)
    n_best = int(ru.in_best_set.sum())
    with plt.rc_context(HEADLINE_RC):
        fig = plt.figure(figsize=(3.45, 6.15))
        # Fixed vertical bands: (a) power, gutter, (b) ranks + in-panel legend — no shared fig.legend.
        bx = fig.add_axes((0.14, 0.48, 0.84, 0.44))
        ax = fig.add_axes((0.22, 0.17, 0.74, 0.22))

        grid = required.groupby("delta")["units_needed"]
        lo, med, hi = (grid.min() * assays_per_unit, grid.median() * assays_per_unit, grid.max() * assays_per_unit)
        bx.fill_between(med.index, lo, hi, color="0.86", lw=0, label="#1 vs #2–#11 (range)")
        bx.plot(med.index, med, color="black", lw=1.2, label="median pair")
        today_line = plt.Line2D([], [], color=POSSIBLE_TOP1, lw=0.9, ls=(0, (3, 1.5)),
                                 label=f"today ({n_assays} assays)")
        bx.axhline(n_assays, color=POSSIBLE_TOP1, lw=0.9, ls=(0, (3, 1.5)))
        mdd_med = float(per_pair["mdd_now"].median())
        mdd_min = float(per_pair["mdd_now"].min())
        bx.plot([mdd_med], [n_assays], "o", ms=3.2, color=POSSIBLE_TOP1, zorder=4)
        bx.annotate(f"median MDD ≈ {mdd_med:.3f}", (mdd_med, n_assays), xytext=(0.022, 280),
                    fontsize=5.5, ha="right", va="bottom",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="0.35", shrinkA=0, shrinkB=2))
        bx.axvline(mdd_min, color="0.45", lw=0.7, ls=(0, (2, 2)))
        bx.text(0.00235, 70, f"best pair\n{mdd_min:.3f}", fontsize=5.5, color="0.35", ha="left", va="bottom")
        for delta in (0.005, 0.01):
            need = float(per_pair[f"assays_for_{delta}"].median())
            bx.plot([delta], [need], "o", ms=3.0, color="black", zorder=4)
            bx.annotate(f"{delta:g} → {50 * round(need / 50):,.0f}",
                        (delta, need), xytext=(0, 7), textcoords="offset points",
                        fontsize=5.5, ha="center", va="bottom")
        bx.set(xscale="log", yscale="log", xlim=(0.002, 0.03), ylim=(50, 5.5e4),
               xlabel="True gain Δ over the current #1 (Spearman)", ylabel="Assays for 80% power")
        bx.set_xticks([0.002, 0.005, 0.01, 0.02, 0.03], ["0.002", "0.005", "0.01", "0.02", "0.03"])
        bx.set_yticks([1e2, 1e3, 1e4], ["100", "1,000", "10,000"])
        bx.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        bx.grid(which="major", lw=0.4, alpha=0.35)
        power_handles, power_labels = bx.get_legend_handles_labels()
        bx.legend(power_handles + [today_line], power_labels + [today_line.get_label()],
                  loc="upper left", frameon=False, borderaxespad=0.2, fontsize=5.5)
        fig.text(0.02, 0.935, "a", fontweight="bold", fontsize=8, va="top")
        fig.text(0.07, 0.935, "Does this gain support a decision? (80% power, paired test, α = 0.05)",
                 fontsize=7, va="top")

        y = np.arange(len(d))
        for yi, (m, r) in zip(y, d.iterrows()):
            c = POSSIBLE_TOP1 if r.in_best_set else OTHER
            ax.hlines(yi, r.simul_lo - 0.4, r.simul_hi + 0.4, color=c, lw=1.0, alpha=0.4, capstyle="butt")
            ax.hlines(yi, r.marg_lo - 0.4, r.marg_hi + 0.4, color=c, lw=3.4, capstyle="butt")
            ax.plot(r["rank"], yi, "o", ms=2.4, mfc="white", mec="black", mew=0.6, zorder=3)
        labels = [f"{int(r['rank'])}. {_headline_label(m)}{'†' if m in incomplete else ''}" for m, r in d.iterrows()]
        ax.set_yticks(y, labels, fontsize=6)
        for tick, best in zip(ax.get_yticklabels(), d.in_best_set):
            if best:
                tick.set_color(POSSIBLE_TOP1)
                tick.set_fontweight("bold")
        x_max = min(int(d.simul_hi.max()) + 1, 22)
        ax.set_xlim(0.3, x_max)
        ax.set_ylim(len(d) - 0.35, -0.45)
        ax.set_xticks([1, 5, 10, 15, 20] if x_max >= 20 else [1, 5, 10, 15])
        ax.tick_params(axis="y", length=0, pad=1)
        ax.set_xlabel("Rank (95% CI)", labelpad=1, fontsize=6.5)
        ax.grid(axis="x", lw=0.35, alpha=0.35)
        rank_handles = [
            plt.Line2D([], [], color=POSSIBLE_TOP1, lw=3.0, label="possible #1"),
            plt.Line2D([], [], color=OTHER, lw=3.0, label="marginal"),
            plt.Line2D([], [], color=OTHER, lw=1.0, alpha=0.45, label="simultaneous"),
            plt.Line2D([], [], marker="o", ls="", ms=2.4, mfc="white", mec="black", mew=0.6, label="published"),
        ]
        ax.legend(handles=rank_handles, loc="upper center", bbox_to_anchor=(0.5, -0.38),
                  ncol=4, frameon=False, handlelength=1.2, columnspacing=0.8, fontsize=5.5)
        fig.text(0.02, 0.415, "b", fontweight="bold", fontsize=8, va="top")
        fig.text(0.07, 0.415, f"Supporting rank CIs (top {rank_panel}; {n_best} not ruled out as #1)",
                 fontsize=7, va="top")

        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=300, metadata={"CreationDate": None} if path.suffix == ".pdf" else None)
        plt.close(fig)
    return paths


ESM2_PARAMS = {"8M": 8e6, "35M": 35e6, "150M": 150e6, "650M": 650e6, "3B": 3e9, "15B": 15e9}


def esm2_replication(table: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    assays = list(dict.fromkeys(table.assay))
    colors = dict(zip(assays, plt.cm.tab10.colors))
    sizes = list(dict.fromkeys(table["size"]))
    markers = dict(zip(sizes, "osD^vP"))
    short = {a: f"{a.split('_')[0]} ({table.loc[table.assay == a, 'function'].iloc[0]}, "
                f"L={table.loc[table.assay == a, 'seq_len'].iloc[0]})" for a in assays}

    ax = axes[0]
    for r in table.itertuples():
        c, m = colors[r.assay], markers[r.size]
        ax.scatter(r.spearman_published, r.spearman_ours, color=c, marker=m, s=46, zorder=3)
        ax.scatter(r.spearman_published, r.spearman_ours_wt_marginals, facecolors="none", edgecolors=c, marker=m,
                   s=46, zorder=2)
        ax.plot([r.spearman_published] * 2, [r.spearman_ours, r.spearman_ours_wt_marginals], color=c, lw=0.6,
                alpha=0.5)
    lo = min(table[["spearman_published", "spearman_ours", "spearman_ours_wt_marginals"]].min()) - 0.03
    hi = max(table[["spearman_published", "spearman_ours", "spearman_ours_wt_marginals"]].max()) + 0.03
    ax.plot([lo, hi], [lo, hi], color="black", lw=0.8, ls="--")
    max_diff = float((table.spearman_ours - table.spearman_published).abs().max())
    max_wt = float((table.spearman_ours_wt_marginals - table.spearman_published).abs().max())
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Published ProteinGym Spearman (3 dp)", ylabel="Our Spearman (CPU)",
           title=f"Masked marginals reproduce the leaderboard (max |Δ| = {max_diff:.4f});\n"
                 f"wild-type marginals do not (max |Δ| = {max_wt:.3f})")
    handles = [plt.Line2D([], [], color=colors[a], marker="o", ls="", label=short[a]) for a in assays]
    handles += [plt.Line2D([], [], color="grey", marker=markers[s], ls="", label=f"ESM-2 {s}") for s in sizes]
    handles += [plt.Line2D([], [], color="grey", marker="o", ls="", label="masked marginals (ProteinGym's method)"),
                plt.Line2D([], [], markerfacecolor="none", markeredgecolor="grey", marker="o", ls="",
                           label="wild-type marginals")]
    ax.legend(handles=handles, fontsize=7, loc="upper left")

    ax = axes[1]
    x = np.array([ESM2_PARAMS[s] for s in sizes])
    for k, a in enumerate(assays):
        d = table[table.assay == a].set_index("size").loc[sizes]
        jitter = x * (1 + 0.06 * (k - (len(assays) - 1) / 2))
        ax.errorbar(jitter, d.spearman_ours, yerr=1.96 * d.spearman_boot_se, color=colors[a], marker="o", ms=4,
                    lw=1.2, capsize=2, label=short[a])
        ax.scatter(jitter, d.spearman_published, color="black", marker="x", s=22, zorder=4)
    ax.scatter([], [], color="black", marker="x", label="published")
    ax.set(xscale="log", xticks=x, xticklabels=sizes, xlabel="ESM-2 size",
           ylabel="Spearman  (±1.96 bootstrap SE over mutants)",
           title="Within-assay sampling noise dwarfs replication error;\nbigger is not reliably better")
    ax.minorticks_off()
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
    fig.tight_layout()
    return _save(fig, path)
