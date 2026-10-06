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
    assays: AssayTable, summary: pd.DataFrame, n_boot: int = 10_000, seed: int = 0, metric: str = "Spearman"
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

    ours = pd.DataFrame({f"Average_{metric}": score, f"Bootstrap_standard_error_{metric}": se}, index=units.models)
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


def _leaderboard_on_assays(assays: AssayTable, mask: pd.Series, n_boot: int, seed: int) -> tuple[pd.Series, pd.Series, list[str]]:
    """ProteinGym aggregate, ranks and 95% best-model set on a fixed assay subset."""
    sub = AssayTable(assays.scores[mask], assays.meta[mask])
    su = to_units(sub)
    theta = proteingym_score(su.X, su.groups, su.n_groups)
    boot = bootstrap_scores(su.X, su.groups, su.n_groups, n_boot, np.random.default_rng(seed))
    ms = ranks.max_statistics(theta, boot)
    best = ranks.best_confidence_set(theta, boot, ALPHA, ms)
    models = su.models
    return (
        pd.Series(theta, index=models),
        pd.Series(ranks.ranks_desc(theta), index=models),
        [m for m, ok in zip(models, best, strict=True) if ok],
    )


def assay_coverage(assays: AssayTable, units: UnitTable, ru: pd.DataFrame, n_boot: int = 4000,
                   seed: int = 2, top_k: int = 10) -> dict:
    """Which models skip assays, and ranks / possible-#1 sets on shared assay subsets."""
    n_scored = assays.scores.notna().sum()
    incomplete = n_scored[n_scored < len(assays.scores)]
    out: dict = {"n_assays": len(assays.scores),
                 "incomplete": {m: int(n) for m, n in incomplete.items()}}
    published_top = list(ru.index[:top_k])
    common_all = assays.scores.notna().all(axis=1)
    common_topk = assays.scores[published_top].notna().all(axis=1)

    def _subset(name: str, mask: pd.Series, seed_off: int) -> dict:
        score, rank, best = _leaderboard_on_assays(assays, mask, n_boot, seed + seed_off)
        return {
            "n_assays": int(mask.sum()),
            "n_units": int(to_units(AssayTable(assays.scores[mask], assays.meta[mask])).X.shape[0]),
            "top1": rank.idxmin(),
            "best_set": best,
            "score": score,
            "rank": rank,
        }

    all_common = _subset("all_models", common_all, 0)
    topk_common = _subset(f"top{top_k}", common_topk, 1)

    common_df = pd.DataFrame(
        {
            "score": ru["score"],
            "rank": ru["rank"],
            "in_best_set": ru["in_best_set"],
            "score_all_common": all_common["score"],
            "rank_all_common": all_common["rank"],
            "score_topk_common": topk_common["score"],
            "rank_topk_common": topk_common["rank"],
        },
        index=ru.index,
    )
    common_df["in_best_set_all_common"] = common_df.index.isin(all_common["best_set"])
    common_df["in_best_set_topk_common"] = common_df.index.isin(topk_common["best_set"])
    common_df = common_df.sort_values("rank")

    su_all = to_units(AssayTable(assays.scores[common_all], assays.meta[common_all]))
    boot = bootstrap_scores(su_all.X, su_all.groups, su_all.n_groups, n_boot, np.random.default_rng(seed))
    theta_all = proteingym_score(su_all.X, su_all.groups, su_all.n_groups)
    ml, mu = ranks.pairwise_rank_ci(theta_all, boot, ALPHA, "marginal", stepdown=False)
    common_df["marg_lo_all_common"] = pd.Series(ml, index=su_all.models)
    common_df["marg_hi_all_common"] = pd.Series(mu, index=su_all.models)

    out["n_common_assays"] = all_common["n_assays"]
    out["n_common_units"] = all_common["n_units"]
    out["top_on_common"] = all_common["top1"]
    out["best_set_all_common"] = all_common["best_set"]
    out[f"n_top{top_k}_common_assays"] = topk_common["n_assays"]
    out[f"n_top{top_k}_common_units"] = topk_common["n_units"]
    out[f"top_on_top{top_k}_common"] = topk_common["top1"]
    out[f"best_set_top{top_k}_common"] = topk_common["best_set"]
    out["published_top_k"] = top_k
    out["common_table"] = common_df
    out["shared_assay_summary"] = pd.DataFrame(
        [
            {
                "subset": "published (all assays)",
                "n_assays": len(assays.scores),
                "top1": ru.index[0],
                "best_set": "; ".join(ru.index[ru["in_best_set"]]),
                "n_possible_1": int(ru["in_best_set"].sum()),
            },
            {
                "subset": f"all {all_common['n_assays']} assays scored by every model",
                "n_assays": all_common["n_assays"],
                "top1": all_common["top1"],
                "best_set": "; ".join(all_common["best_set"]),
                "n_possible_1": len(all_common["best_set"]),
            },
            {
                "subset": f"{topk_common['n_assays']} assays scored by published top {top_k}",
                "n_assays": topk_common["n_assays"],
                "top1": topk_common["top1"],
                "best_set": "; ".join(topk_common["best_set"]),
                "n_possible_1": len(topk_common["best_set"]),
            },
        ]
    )
    missing = assays.scores.index[~common_all]
    out["missing_assays"] = assays.meta.loc[missing, ["uniprot", "function", "n_mutants"]]
    others = [m for m in assays.models if m not in incomplete.index]
    out["others_mean_on_missing"] = float(assays.scores.loc[missing, others].mean().mean())
    out["others_mean_on_common"] = float(assays.scores.loc[common_all, others].mean().mean())
    out["missing_by_function"] = assays.meta.loc[missing, "function"].value_counts().reindex(FUNCTION_GROUPS, fill_value=0)
    if "Protriever" in common_df.index:
        prot = common_df.loc["Protriever"]
        out["protriever_common"] = {k: float(prot[k]) for k in
                                    ["score", "rank", "score_all_common", "rank_all_common"]}
        out["protriever_common"]["marg_lo_common"] = float(prot["marg_lo_all_common"])
        out["protriever_common"]["marg_hi_common"] = float(prot["marg_hi_all_common"])
    else:
        out["protriever_common"] = None
    return out
