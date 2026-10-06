"""Rank uncertainty for the ProteinGym DMS-substitution leaderboard (zero-shot Spearman).

Everything else in the repo is built from these five functions:
  load               read the two ProteinGym CSVs
  strata             one (protein x model) matrix per function category
  leaderboard_score  ProteinGym's published aggregate
  bootstrap          stratified cluster bootstrap of that aggregate
  rank_intervals     rank confidence intervals from the bootstrap draws
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

DMS_FILE = "DMS_substitutions_Spearman_DMS_level.csv"
SUMMARY_FILE = "Summary_performance_DMS_substitutions_Spearman.csv"


def load(data_dir):
    """Return (assay-level scores, published summary, model names in published order)."""
    d = pd.read_csv(f"{data_dir}/{DMS_FILE}")
    s = pd.read_csv(f"{data_dir}/{SUMMARY_FILE}")
    return d, s, list(s.Model_name)


def strata(d, models):
    """{function category: (UniProt clusters x models) array}.

    Assays of the same protein within a category are averaged first, so a protein with
    several assays counts once. This is ProteinGym's own aggregation step.
    """
    cats = sorted(d["Selection Type"].unique())
    return {c: d[d["Selection Type"] == c].groupby("UniProt ID")[models].mean().to_numpy() for c in cats}


def leaderboard_score(U):
    """Mean over proteins within each category, then the unweighted mean of the category means.

    nanmean because one model (Protriever) has no score for some proteins; it is then averaged
    over the proteins it does have, which is what the published leaderboard does too.
    """
    return np.mean([np.nanmean(u, axis=0) for u in U.values()], axis=0)


def bootstrap(U, B, rng, small_sample_correction=False):
    """B draws of leaderboard_score from resampling proteins with replacement within each category.

    This is the same resampling ProteinGym uses for its published standard errors, so the
    uncertainty it measures is "which proteins happened to be in the benchmark", not the noise
    inside each assay.

    small_sample_correction rescales each category's bootstrap deviations by sqrt(n / (n - 1)).
    The plain bootstrap variance of a mean of n clusters is (n - 1) / n times the usual unbiased
    estimate; that matters for Binding, which has only 12 proteins.
    """
    total = 0.0
    for u in U.values():
        n = u.shape[0]
        draws = np.nanmean(u[rng.integers(0, n, size=(B, n))], axis=1)
        if small_sample_correction:
            centre = np.nanmean(u, axis=0)
            draws = centre + np.sqrt(n / (n - 1)) * (draws - centre)
        total = total + draws
    return total / len(U)


def rank_intervals(theta, boot, alpha=0.05, method="marginal"):
    """Rank confidence intervals (rank 1 = best), following Mogstad, Romano, Shaikh & Wilhelm (2024).

    Build a confidence interval for every pairwise difference theta_j - theta_k with half-width
    crit * se_jk, where crit comes from the bootstrap distribution of the largest |t|. Then
        lower_j = 1 + (number of models significantly better than j)
        upper_j = M - (number of models significantly worse than j)

    method="marginal": crit_j is the max over j's own M-1 comparisons. Each model's interval
        covers that model's true rank with probability >= 1 - alpha, but the M intervals do NOT
        all hold at once.
    method="simultaneous": one crit from the max over all pairs. All M intervals hold jointly
        with probability >= 1 - alpha, so they are wider.
    method="unadjusted": ordinary per-pair 95% intervals (crit = 1.96), no correction for the
        M-1 comparisons. Not valid; kept as the naive baseline the coverage simulation should
        catch.
    """
    M = len(theta)
    centred = boot - theta                                   # B x M
    cov = np.cov(centred, rowvar=False)
    var = np.diag(cov)
    se = np.sqrt(np.maximum(var[:, None] + var[None, :] - 2 * cov, 0.0))
    np.fill_diagonal(se, np.inf)                             # a model is never compared with itself

    # max_t[b, j] = max over k of |t_jk| in bootstrap draw b (chunked to keep memory small)
    max_t = np.empty_like(centred)
    for i in range(0, len(centred), 256):
        c = centred[i:i + 256]
        max_t[i:i + 256] = (np.abs(c[:, :, None] - c[:, None, :]) / se).max(axis=2)

    if method == "simultaneous":
        crit = np.full(M, np.quantile(max_t.max(axis=1), 1 - alpha))
    elif method == "marginal":
        crit = np.quantile(max_t, 1 - alpha, axis=0)
    elif method == "unadjusted":
        crit = np.full(M, norm.ppf(1 - alpha / 2))
    else:
        raise ValueError(method)

    diff = theta[:, None] - theta[None, :]                   # diff[j, k] = theta_j - theta_k
    half = crit[:, None] * se
    better = (diff + half < 0).sum(axis=1)                   # k confidently above j
    worse = (diff - half > 0).sum(axis=1)                    # k confidently below j
    return 1 + better, M - worse


def ranks(theta):
    """1 = highest score."""
    return (-theta).argsort().argsort() + 1
