"""How the top of the leaderboard moves across ProteinGym metrics and aggregation schemes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import ranks
from .aggregation import SCHEMES, bootstrap_scheme, make_design, point_scores
from .data import DEFAULT_DATA_DIR, METRICS, load_assays

BASELINE = ("Spearman", "proteingym")


def rank_table(metric: str, scheme_name: str, n_boot: int, seed: int, alpha: float = 0.05,
               data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Scores, ranks, headline rank intervals, P(#1) and best-set membership for one leaderboard."""
    design = make_design(load_assays(data_dir, metric))
    scheme = SCHEMES[scheme_name]
    theta = point_scores(design, scheme)
    boot = bootstrap_scheme(design, scheme, n_boot, np.random.default_rng(seed))
    ms = ranks.max_statistics(theta, boot)
    ml, mu = ranks.pairwise_rank_ci(theta, boot, alpha, "marginal", ms, stepdown=False)
    sl, su = ranks.pairwise_rank_ci(theta, boot, alpha, "simultaneous", ms)
    return pd.DataFrame({
        "metric": metric, "scheme": scheme_name, "model": design.models, "score": theta,
        "rank": ranks.ranks_desc(theta), "p_rank1": (ranks.ranks_desc(boot) == 1).mean(axis=0),
        "marg_lo": ml, "marg_hi": mu, "simul_lo": sl, "simul_hi": su,
        "in_best_set": ranks.best_confidence_set(theta, boot, alpha, ms),
    })


def robustness_grid(n_boot: int = 4000, seed: int = 11, metrics: list[str] | None = None,
                    schemes: list[str] | None = None, data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Long table: one row per (metric, scheme, model)."""
    tables = []
    for i, metric in enumerate(metrics or METRICS):
        for j, scheme in enumerate(schemes or list(SCHEMES)):
            tables.append(rank_table(metric, scheme, n_boot, seed + 100 * i + j, data_dir=data_dir))
    return pd.concat(tables, ignore_index=True)


def summarise(long: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """Per leaderboard: #1, best-model set, top 10, and agreement with the published leaderboard."""
    base = long[(long.metric == BASELINE[0]) & (long.scheme == BASELINE[1])].set_index("model")
    base_top = list(base.sort_values("rank").index[:top])
    rows = []
    for (metric, scheme), d in long.groupby(["metric", "scheme"], sort=False):
        d = d.set_index("model").loc[base.index]
        order = d.sort_values(["rank", "score"], ascending=[True, False]).index
        top_list = list(order[:top])
        moved = (d.loc[base_top, "rank"] - base.loc[base_top, "rank"]).abs()
        rows.append({
            "metric": metric, "scheme": scheme, "top1": order[0],
            "top1_p_rank1": float(d.loc[order[0], "p_rank1"]),
            "best_set": "; ".join(d.index[d.in_best_set].tolist()),
            "best_set_size": int(d.in_best_set.sum()),
            f"top{top}": "; ".join(top_list),
            f"overlap_with_published_top{top}": len(set(top_list) & set(base_top)),
            "kendall_tau_vs_published": float(sps.kendalltau(d["score"], base["score"]).statistic),
            f"max_rank_shift_published_top{top}": int(moved.max()),
            f"mean_marginal_width_top{top}": float((d.loc[top_list, "marg_hi"] - d.loc[top_list, "marg_lo"]).mean()),
        })
    return pd.DataFrame(rows)


def proteingym_uniprot_level_average(assays, reference: pd.DataFrame) -> pd.Series:
    """The 'Average' row of ProteinGym's published Uniprot_level CSV, reproduced exactly.

    ProteinGym merges per-protein means with non-deduplicated per-assay lookups before averaging,
    so proteins with several assays / MSA-depth categories are counted more than once.
    """
    ref = reference.copy()
    ref["MSA_Neff_L_category"] = ref["MSA_Neff_L_category"].str.capitalize()
    per_protein = (assays.scores.join(assays.meta[["uniprot"]]).rename(columns={"uniprot": "UniProt_ID"})
                   .groupby("UniProt_ID").mean(numeric_only=True).reset_index())
    merged = (per_protein
              .merge(ref[["UniProt_ID", "MSA_Neff_L_category"]].drop_duplicates(), on="UniProt_ID", how="left")
              .merge(ref[["UniProt_ID", "taxon"]].drop_duplicates(), on="UniProt_ID", how="left")
              .merge(ref[["UniProt_ID", "coarse_selection_type"]], on="UniProt_ID", how="left"))
    return merged[assays.models].mean()
