"""Real-data analyses: leaderboard reproduction and rank uncertainty."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import ranks
from .data import FUNCTION_GROUPS, AssayTable, UnitTable, to_units
from .stats import bootstrap_scores, group_means, proteingym_gap_se, proteingym_score

ALPHA = 0.05


def _category_average(assays: AssayTable, column: str, categories: list[str]) -> pd.DataFrame:
    """ProteinGym's breakdown scheme: mean per (UniProt, category), then per category."""
    frame = assays.scores.join(assays.meta[["uniprot", column]])
    per_unit = frame.groupby(["uniprot", column]).mean(numeric_only=True)
    return per_unit.groupby(column).mean(numeric_only=True).T[categories]


def reproduce_leaderboard(
    assays: AssayTable, summary: pd.DataFrame, n_boot: int = 10_000, seed: int = 0
) -> pd.DataFrame:
    """Recompute every published summary column and line it up with ProteinGym's numbers."""
    units = to_units(assays)
    score = proteingym_score(units.X, units.groups, units.n_groups)
    top = int(np.argmax(score))
    se = proteingym_gap_se(units.X, units.groups, units.n_groups, top, n_boot, np.random.default_rng(seed))
    gm = pd.DataFrame(group_means(units.X, units.groups, units.n_groups).T, index=units.models,
                      columns=[f"Function_{g}" for g in units.group_names])
    msa = _category_average(assays, "msa_depth", ["Low", "Medium", "High"])
    msa.columns = ["Low_MSA_depth", "Medium_MSA_depth", "High_MSA_depth"]
    taxa = _category_average(assays, "taxon", ["Human", "Eukaryote", "Prokaryote", "Virus"])
    taxa.columns = ["Taxa_Human", "Taxa_Other_Eukaryote", "Taxa_Prokaryote", "Taxa_Virus"]

    ours = pd.DataFrame({"Average_Spearman": score, "Bootstrap_standard_error_Spearman": se}, index=units.models)
    ours = ours.join(gm).join(msa).join(taxa)
    ours["rank"] = ranks.ranks_desc(score)

    out = pd.DataFrame(index=units.models)
    for col in ours.columns:
        if col == "rank":
            continue
        out[f"{col}__ours"] = ours[col]
        out[f"{col}__published"] = summary.loc[units.models, col]
    out["rank__ours"] = ours["rank"]
    out["rank__published"] = summary.loc[units.models, "Model_rank"]
    return out.sort_values("rank__published")


def reproduction_report(rep: pd.DataFrame) -> pd.DataFrame:
    """Per published column: how many models match at 3 dp and the largest absolute difference."""
    rows = []
    for col in sorted({c.split("__")[0] for c in rep.columns}):
        ours, pub = rep[f"{col}__ours"], rep[f"{col}__published"]
        if col == "rank":
            d = (ours - pub).abs()
        else:
            d = (np.round(ours.astype(float), 3) - pub).abs()
        rows.append({"column": col, "n_models": int(d.notna().sum()),
                     "n_exact_3dp": int((d < 1e-9).sum()), "max_abs_diff": float(d.max())})
    return pd.DataFrame(rows).set_index("column")


def rank_uncertainty(units: UnitTable, n_boot: int = 10_000, seed: int = 1, alpha: float = ALPHA) -> tuple[pd.DataFrame, dict]:
    """Point scores, bootstrap SEs, P(#1) and three kinds of 95% rank interval for every model."""
    theta = proteingym_score(units.X, units.groups, units.n_groups)
    boot = bootstrap_scores(units.X, units.groups, units.n_groups, n_boot, np.random.default_rng(seed))
    ms = ranks.max_statistics(theta, boot)
    pl, pu = ranks.percentile_rank_ci(boot, alpha)
    # Headline: marginal single-step and simultaneous step-down (both hold coverage in every
    # simulated scenario); marginal step-down under-covers when models are exactly tied.
    ml, mu = ranks.pairwise_rank_ci(theta, boot, alpha, "marginal", ms, stepdown=False)
    sl, su = ranks.pairwise_rank_ci(theta, boot, alpha, "simultaneous", ms)
    msl, msu = ranks.pairwise_rank_ci(theta, boot, alpha, "marginal", ms)
    ssl, ssu = ranks.pairwise_rank_ci(theta, boot, alpha, "simultaneous", ms, stepdown=False)
    best = ranks.best_confidence_set(theta, boot, alpha, ms)
    naive_best = ranks.naive_best_set(theta, boot, alpha)
    boot_ranks = ranks.ranks_desc(boot)
    df = pd.DataFrame(
        {
            "score": theta,
            "rank": ranks.ranks_desc(theta),
            "boot_se": boot.std(axis=0, ddof=1),
            "p_rank1": (boot_ranks == 1).mean(axis=0),
            "pct_lo": pl, "pct_hi": pu,
            "marg_lo": ml, "marg_hi": mu,
            "simul_lo": sl, "simul_hi": su,
            "marg_stepdown_lo": msl, "marg_stepdown_hi": msu,
            "simul_single_lo": ssl, "simul_single_hi": ssu,
            "in_best_set": best,
            "in_naive_best_set": naive_best,
        },
        index=units.models,
    ).sort_values("score", ascending=False)
    extras = {"theta": theta, "boot": boot, "ms": ms}
    return df, extras


def neighbour_tests(units: UnitTable, ru: pd.DataFrame, boot: np.ndarray, top: int = 20,
                    alpha: float = ALPHA) -> pd.DataFrame:
    """Paired bootstrap z-tests between each pair of adjacent models in the top ``top``."""
    order = list(ru.index[:top])
    idx = [units.models.index(m) for m in order]
    theta = ru["score"].to_numpy()
    se = ranks.pairwise_se(boot)
    rows = []
    for a in range(top - 1):
        i, j = idx[a], idx[a + 1]
        gap = theta[a] - theta[a + 1]
        z = gap / se[i, j]
        rows.append({"rank_hi": a + 1, "model_hi": order[a], "model_lo": order[a + 1],
                     "gap": gap, "se": se[i, j], "z": z, "p": 2 * sps.norm.sf(abs(z))})
    df = pd.DataFrame(rows)
    df["p_holm"] = ranks.holm(df["p"].to_numpy())
    df["sig_uncorrected"] = df["p"] < alpha
    df["sig_holm"] = df["p_holm"] < alpha
    return df


def top_gap_power(units: UnitTable, ru: pd.DataFrame, boot: np.ndarray, k: int = 10) -> pd.DataFrame:
    """Gap, SE and minimum detectable difference for #1 against each of the next ``k`` models."""
    se = ranks.pairwise_se(boot)
    i1 = units.models.index(ru.index[0])
    rows = []
    for r in range(1, k + 1):
        m = ru.index[r]
        s = se[i1, units.models.index(m)]
        rows.append({"rank": r + 1, "model": m, "gap_to_1": ru["score"].iloc[0] - ru["score"].iloc[r],
                     "se": s, "sig_threshold_1.96se": 1.96 * s,
                     "mdd_80pct_power": ranks.minimum_detectable_difference(s)})
    return pd.DataFrame(rows)


def leave_one_group_out(units: UnitTable, top_k: int = 5) -> pd.DataFrame:
    """Recompute the leaderboard with each function group removed; report the new top ``top_k``."""
    rows = []
    for drop in [None, *units.group_names]:
        sub = units if drop is None else units.drop_groups([drop])
        s = pd.Series(proteingym_score(sub.X, sub.groups, sub.n_groups), index=units.models)
        s = s.sort_values(ascending=False)
        row = {"dropped": drop or "(none)", "n_units": int(sub.X.shape[0]),
               "margin_1_vs_2": s.iloc[0] - s.iloc[1]}
        for r in range(top_k):
            row[f"#{r + 1}"] = f"{s.index[r]} ({s.iloc[r]:.3f})"
        rows.append(row)
    return pd.DataFrame(rows)


def leave_one_group_out_ranks(units: UnitTable) -> pd.DataFrame:
    """Rank of every model under each leave-one-function-group-out leaderboard."""
    out = {}
    for drop in [None, *units.group_names]:
        sub = units if drop is None else units.drop_groups([drop])
        out[drop or "(none)"] = ranks.ranks_desc(proteingym_score(sub.X, sub.groups, sub.n_groups))
    return pd.DataFrame(out, index=units.models).T


def version_drift(summaries: dict[str, pd.DataFrame], reference: str) -> pd.DataFrame:
    """Change in each model's published Average Spearman between releases (models in both)."""
    ref = summaries[reference]["Average_Spearman"]
    rows = []
    names = list(summaries)
    for a, b in zip(names[:-1], names[1:]):
        sa, sb = summaries[a]["Average_Spearman"], summaries[b]["Average_Spearman"]
        common = sa.index.intersection(sb.index)
        d = sb.loc[common] - sa.loc[common]
        big = d.abs().sort_values(ascending=False).head(3)
        rows.append({"from": a, "to": b, "n_models_from": len(sa), "n_models_to": len(sb),
                     "n_common": len(common), "median_change": d.median(), "max_abs_change": d.abs().max(),
                     "largest_changes": "; ".join(f"{m} {d[m]:+.3f}" for m in big.index),
                     "top_model_to": sb.idxmax()})
    df = pd.DataFrame(rows)
    df.attrs["reference_top"] = ref.idxmax()
    return df


def assay_coverage(assays: AssayTable, units: UnitTable, ru: pd.DataFrame, n_boot: int = 4000,
                   seed: int = 2) -> dict:
    """Which models skip assays, and what happens when everyone is scored on the common subset."""
    n_scored = assays.scores.notna().sum()
    incomplete = n_scored[n_scored < len(assays.scores)]
    out: dict = {"n_assays": len(assays.scores),
                 "incomplete": {m: int(n) for m, n in incomplete.items()}}
    common = assays.scores.notna().all(axis=1)
    sub = AssayTable(assays.scores[common], assays.meta[common])
    su = to_units(sub)
    theta = proteingym_score(su.X, su.groups, su.n_groups)
    boot = bootstrap_scores(su.X, su.groups, su.n_groups, n_boot, np.random.default_rng(seed))
    ml, mu = ranks.pairwise_rank_ci(theta, boot, ALPHA, "marginal", stepdown=False)
    common_df = pd.DataFrame({"score_common": theta, "rank_common": ranks.ranks_desc(theta),
                              "marg_lo_common": ml, "marg_hi_common": mu}, index=su.models)
    common_df = ru[["score", "rank"]].join(common_df).sort_values("rank")
    out["n_common_assays"] = int(common.sum())
    out["n_common_units"] = int(su.X.shape[0])
    out["common_table"] = common_df
    missing = assays.scores.index[~common]
    out["missing_assays"] = assays.meta.loc[missing, ["uniprot", "function", "n_mutants"]]
    others = [m for m in assays.models if m not in incomplete.index]
    out["others_mean_on_missing"] = float(assays.scores.loc[missing, others].mean().mean())
    out["others_mean_on_common"] = float(assays.scores.loc[common, others].mean().mean())
    out["missing_by_function"] = assays.meta.loc[missing, "function"].value_counts().reindex(FUNCTION_GROUPS, fill_value=0)
    return out
