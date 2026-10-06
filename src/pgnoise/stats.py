"""ProteinGym aggregation and the stratified (within function group) bootstrap over units."""

from __future__ import annotations

import numpy as np


def group_means(X: np.ndarray, groups: np.ndarray, n_groups: int) -> np.ndarray:
    """NaN-skipping mean of each model within each function group -> (n_groups, n_models)."""
    out = np.full((n_groups, X.shape[1]), np.nan)
    for g in range(n_groups):
        block = X[groups == g]
        observed = ~np.isnan(block)
        counts = observed.sum(axis=0)
        sums = np.where(observed, block, 0.0).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[g] = np.where(counts > 0, sums / counts, np.nan)
    return out


def proteingym_score(X: np.ndarray, groups: np.ndarray, n_groups: int) -> np.ndarray:
    """ProteinGym's headline score: mean over function groups of the within-group unit means.

    Groups where a model has no data are skipped, matching pandas' ``mean(skipna=True)``.
    """
    return np.nanmean(group_means(X, groups, n_groups), axis=0)


def stratified_counts(groups: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap multiplicities (n_boot, n_units), resampling units with replacement within each group."""
    counts = np.zeros((n_boot, groups.size), dtype=np.int32)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        draws = rng.integers(0, idx.size, size=(n_boot, idx.size))
        np.add.at(counts, (np.arange(n_boot)[:, None], idx[draws]), 1)
    return counts


def weighted_group_means(
    X: np.ndarray, groups: np.ndarray, n_groups: int, weights: np.ndarray
) -> np.ndarray:
    """Group means for many weight vectors at once -> (n_weights, n_groups, n_models)."""
    observed = ~np.isnan(X)
    X0 = np.where(observed, X, 0.0)
    out = np.empty((weights.shape[0], n_groups, X.shape[1]))
    for g in range(n_groups):
        rows = groups == g
        W = weights[:, rows].astype(float)
        num = W @ X0[rows]
        den = W @ observed[rows].astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, g, :] = np.where(den > 0, num / den, np.nan)
    return out


def bootstrap_scores(
    X: np.ndarray,
    groups: np.ndarray,
    n_groups: int,
    n_boot: int,
    rng: np.random.Generator,
    chunk: int = 2000,
) -> np.ndarray:
    """Bootstrap replicates of the ProteinGym score for every model -> (n_boot, n_models)."""
    reps = []
    for start in range(0, n_boot, chunk):
        counts = stratified_counts(groups, min(chunk, n_boot - start), rng)
        with np.errstate(invalid="ignore"):
            reps.append(np.nanmean(weighted_group_means(X, groups, n_groups, counts), axis=1))
    return np.vstack(reps)


def proteingym_gap_se(
    X: np.ndarray,
    groups: np.ndarray,
    n_groups: int,
    top: int,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """ProteinGym's published error bar: bootstrap SE of each model's gap to the top model.

    Replicates ``compute_bootstrap_standard_error_functional_categories``: differences are taken
    at the unit level, group means skip NaNs, the group means are averaged *without* skipping
    NaNs, and the SD (ddof=1) is taken over finite replicates.
    """
    D = X - X[:, [top]]
    reps = []
    for start in range(0, n_boot, 2000):
        counts = stratified_counts(groups, min(2000, n_boot - start), rng)
        reps.append(weighted_group_means(D, groups, n_groups, counts).mean(axis=1))
    reps = np.vstack(reps)
    out = np.empty(X.shape[1])
    for m in range(X.shape[1]):
        finite = reps[np.isfinite(reps[:, m]), m]
        out[m] = finite.std(ddof=1) if finite.size > 1 else np.nan
    return out
