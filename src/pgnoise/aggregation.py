"""Alternative ways to aggregate per-assay scores into a leaderboard score.

Every scheme is bootstrapped with the same sampling model as ProteinGym's error bars: resample
(UniProt, function group) units with replacement within each function group. Only the estimand
(how assays are weighted and summarised) changes between schemes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import AssayTable, to_units
from .stats import stratified_counts, weighted_group_means


@dataclass(frozen=True)
class Scheme:
    name: str
    description: str
    assay_weight: str  # "unit": 1 / assays in its unit; "uniprot": 1 / assays of its protein; "flat": 1
    balanced: bool  # average function-group means (True) or pool all assays (False)
    statistic: str = "mean"  # "mean" | "median"


SCHEMES: dict[str, Scheme] = {
    s.name: s
    for s in [
        Scheme("proteingym", "ProteinGym: assays -> (UniProt, function) units -> function groups -> mean",
               "unit", True),
        Scheme("function_group_mean", "Mean of the 5 function-group means of raw assays (no UniProt step)",
               "flat", True),
        Scheme("uniprot_weighted", "Each protein weighted equally, ignoring function groups", "uniprot", False),
        Scheme("flat_mean", "Plain mean over all 217 assays", "flat", False),
        Scheme("median", "Median over all 217 assays", "flat", False, "median"),
    ]
}


@dataclass(frozen=True)
class Design:
    """Per-assay scores plus the index maps every scheme needs."""

    S: np.ndarray  # (n_assays, p) with NaN for unscored
    unit_of_assay: np.ndarray  # (n_assays,) index into units
    group_of_assay: np.ndarray  # (n_assays,)
    unit_groups: np.ndarray  # (n_units,)
    n_groups: int
    weights: dict[str, np.ndarray]  # (n_assays, p) base weights per rule, 0 where unscored
    models: list[str]
    unit_X: np.ndarray  # ProteinGym unit matrix, for the exact ProteinGym path


def make_design(assays: AssayTable) -> Design:
    units = to_units(assays)
    unit_index = {u: i for i, u in enumerate(units.units)}
    keys = list(zip(assays.meta["uniprot"], assays.meta["function"]))
    unit_of_assay = np.array([unit_index[k] for k in keys])
    group_of_assay = units.groups[unit_of_assay]
    S = assays.scores.to_numpy(dtype=float)
    observed = pd.DataFrame(~np.isnan(S), index=assays.scores.index)

    def per_model_inverse_count(labels: pd.Series) -> np.ndarray:
        # Each model averages only its own scored assays within a protein / unit.
        counts = observed.groupby(labels.to_numpy()).transform("sum").to_numpy(dtype=float)
        with np.errstate(divide="ignore"):
            return np.where(observed.to_numpy(), 1.0 / counts, 0.0)

    unit_labels = pd.Series(unit_of_assay, index=assays.scores.index)
    return Design(
        S=S,
        unit_of_assay=unit_of_assay,
        group_of_assay=group_of_assay,
        unit_groups=units.groups,
        n_groups=units.n_groups,
        weights={"unit": per_model_inverse_count(unit_labels),
                 "uniprot": per_model_inverse_count(assays.meta["uniprot"]),
                 "flat": observed.to_numpy(dtype=float)},
        models=units.models,
        unit_X=units.X,
    )


def _weighted_median(S: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Weighted median of each column of S for every row of weights W -> (B, p); NaNs get weight 0."""
    B, p = W.shape[0], S.shape[1]
    out = np.full((B, p), np.nan)
    for m in range(p):
        observed = ~np.isnan(S[:, m])
        order = np.argsort(S[observed, m], kind="stable")
        values = S[observed, m][order]
        cw = np.cumsum(W[:, observed][:, order], axis=1)
        half = cw[:, -1:] / 2
        # When the cumulative weight hits exactly half, average with the next value (pandas' rule).
        lower = np.argmax(cw >= half - 1e-12, axis=1)
        upper = np.argmax(cw > half + 1e-12, axis=1)
        out[:, m] = (values[lower] + values[upper]) / 2
    return out


def scheme_scores(design: Design, scheme: Scheme, unit_counts: np.ndarray) -> np.ndarray:
    """Scores (B, p) for each row of unit multiplicities (B, n_units); a row of ones = point estimate."""
    if scheme.name == "proteingym":
        with np.errstate(invalid="ignore"):
            return np.nanmean(weighted_group_means(design.unit_X, design.unit_groups, design.n_groups,
                                                   unit_counts), axis=1)
    W = unit_counts[:, design.unit_of_assay].astype(float)
    if scheme.statistic == "median":
        if scheme.balanced or scheme.assay_weight != "flat":
            raise NotImplementedError("only the pooled, flat-weighted median is defined")
        return _weighted_median(design.S, W)
    base = design.weights[scheme.assay_weight]
    numer = np.where(np.isnan(design.S), 0.0, design.S) * base
    if scheme.balanced:
        gm = np.empty((W.shape[0], design.n_groups, design.S.shape[1]))
        for g in range(design.n_groups):
            rows = design.group_of_assay == g
            with np.errstate(invalid="ignore", divide="ignore"):
                gm[:, g] = (W[:, rows] @ numer[rows]) / (W[:, rows] @ base[rows])
        with np.errstate(invalid="ignore"):
            return np.nanmean(gm, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (W @ numer) / (W @ base)


def point_scores(design: Design, scheme: Scheme) -> np.ndarray:
    return scheme_scores(design, scheme, np.ones((1, design.unit_groups.size)))[0]


def bootstrap_scheme(design: Design, scheme: Scheme, n_boot: int, rng: np.random.Generator,
                     chunk: int = 1000) -> np.ndarray:
    reps = []
    for start in range(0, n_boot, chunk):
        counts = stratified_counts(design.unit_groups, min(chunk, n_boot - start), rng)
        reps.append(scheme_scores(design, scheme, counts))
    return np.vstack(reps)
