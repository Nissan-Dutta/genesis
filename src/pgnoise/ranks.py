"""Rank confidence intervals, pairwise tests and confidence sets for the best model.

Rank convention throughout: ``rank_j = 1 + #{k : theta_k > theta_j}`` (1 = best, ties share the
better rank). Pairwise-comparison intervals follow Mogstad, Romano, Shaikh & Wilhelm (2024),
"Inference for Ranks with Applications to Mobility across Neighbourhoods and Academic
Achievement across Countries" (single-step, studentised max-|t| bootstrap critical values).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


def ranks_desc(theta: np.ndarray) -> np.ndarray:
    """Min-ranks of ``theta`` along the last axis, 1 = largest."""
    return stats.rankdata(-theta, method="min", axis=-1).astype(int)


def percentile_rank_ci(theta_boot: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Naive bootstrap percentile interval of each model's rank across replicates."""
    r = ranks_desc(theta_boot)
    lower = np.quantile(r, alpha / 2, axis=0, method="lower").astype(int)
    upper = np.quantile(r, 1 - alpha / 2, axis=0, method="higher").astype(int)
    return lower, upper


def pairwise_se(theta_boot: np.ndarray) -> np.ndarray:
    """Bootstrap SE of every pairwise difference theta_j - theta_k -> (p, p), inf on the diagonal."""
    C = np.cov(theta_boot, rowvar=False)
    v = np.diag(C)
    var = v[:, None] + v[None, :] - 2 * C
    se = np.sqrt(np.clip(var, 0, None))
    np.fill_diagonal(se, np.inf)
    return np.where(se > 0, se, np.inf)


@dataclass(frozen=True)
class MaxStats:
    """Bootstrap distributions of the studentised max statistics needed by every procedure."""

    marginal_two_sided: np.ndarray  # (B, p): max_k |T_jk| for each j
    simultaneous_two_sided: np.ndarray  # (B,): max_{j,k} |T_jk|
    best_one_sided: np.ndarray  # (B, p): max_k T_kj, i.e. evidence some k beats j
    se: np.ndarray  # (p, p)
    diff: np.ndarray  # (p, p): theta_hat_j - theta_hat_k


def max_statistics(theta_hat: np.ndarray, theta_boot: np.ndarray, chunk: int = 500) -> MaxStats:
    se = pairwise_se(theta_boot)
    diff = theta_hat[:, None] - theta_hat[None, :]
    B, p = theta_boot.shape
    marg = np.empty((B, p))
    simul = np.empty(B)
    best = np.empty((B, p))
    for s in range(0, B, chunk):
        tb = theta_boot[s : s + chunk]
        T = (tb[:, :, None] - tb[:, None, :] - diff) / se  # T[b, j, k]
        absT = np.abs(T)
        marg[s : s + chunk] = absT.max(axis=2)
        simul[s : s + chunk] = absT.max(axis=(1, 2))
        best[s : s + chunk] = T.max(axis=1)
    return MaxStats(marg, simul, best, se, diff)


def _bounds_from_critical(diff: np.ndarray, se: np.ndarray, crit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """crit[j] is the critical value used for every comparison involving model j."""
    p = diff.shape[0]
    c = crit[:, None]
    better_than_j = (diff.T > c * se).sum(axis=1)  # k with theta_k - theta_j significantly > 0
    worse_than_j = (diff > c * se).sum(axis=1)
    return 1 + better_than_j, p - worse_than_j


_DENSE_LIMIT = 40_000_000  # elements of the (B, p, p) |T| tensor we are willing to hold in memory


def _abs_t(tb: np.ndarray, diff: np.ndarray, se: np.ndarray) -> np.ndarray:
    return np.abs((tb[:, :, None] - tb[:, None, :] - diff) / se).astype(np.float32)


def _masked_critical(theta_boot: np.ndarray, diff: np.ndarray, se: np.ndarray, active: np.ndarray,
                     mode: str, alpha: float, dense: np.ndarray | None = None,
                     chunk: int = 500) -> np.ndarray:
    """(1 - alpha) quantile of the max |T| over the still-active comparisons, per model or pooled."""
    B, p = theta_boot.shape
    maxima = np.empty((B, p)) if mode == "marginal" else np.empty(B)
    for s in range(0, B, chunk):
        absT = dense[s : s + chunk] if dense is not None else _abs_t(theta_boot[s : s + chunk], diff, se)
        absT = np.where(active, absT, 0.0)
        maxima[s : s + chunk] = absT.max(axis=2) if mode == "marginal" else absT.max(axis=(1, 2))
    crit = np.quantile(maxima, 1 - alpha, axis=0)
    return crit if mode == "marginal" else np.full(p, crit)


def pairwise_rank_ci(
    theta_hat: np.ndarray,
    theta_boot: np.ndarray,
    alpha: float = 0.05,
    mode: str = "marginal",
    ms: MaxStats | None = None,
    stepdown: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Rank intervals from studentised pairwise comparisons.

    ``mode="marginal"``: each model's interval covers its own rank with prob >= 1 - alpha.
    ``mode="simultaneous"``: all intervals cover all ranks jointly with prob >= 1 - alpha.
    ``stepdown``: Romano-Wolf refinement; recompute the critical value over the comparisons not
    yet rejected until nothing new is rejected. The first step is the single-step procedure.
    """
    if mode not in ("marginal", "simultaneous"):
        raise ValueError(f"unknown mode {mode!r}")
    if not stepdown:
        ms = ms or max_statistics(theta_hat, theta_boot)
        if mode == "marginal":
            crit = np.quantile(ms.marginal_two_sided, 1 - alpha, axis=0)
        else:
            crit = np.full(theta_hat.size, np.quantile(ms.simultaneous_two_sided, 1 - alpha))
        return _bounds_from_critical(ms.diff, ms.se, crit)

    se = ms.se if ms is not None else pairwise_se(theta_boot)
    diff = theta_hat[:, None] - theta_hat[None, :]
    p = theta_hat.size
    dense = _abs_t(theta_boot, diff, se) if theta_boot.shape[0] * p * p <= _DENSE_LIMIT else None
    active = ~np.eye(p, dtype=bool)
    rejected = np.zeros((p, p), dtype=bool)
    while active.any():
        crit = _masked_critical(theta_boot, diff, se, active, mode, alpha, dense)
        with np.errstate(invalid="ignore"):
            new = active & (np.abs(diff) > crit[:, None] * se)
        if not new.any():
            break
        rejected |= new
        active &= ~new
    better_than_j = (rejected & (diff.T > 0)).sum(axis=1)
    worse_than_j = (rejected & (diff > 0)).sum(axis=1)
    return 1 + better_than_j, p - worse_than_j


def best_confidence_set(
    theta_hat: np.ndarray,
    theta_boot: np.ndarray,
    alpha: float = 0.05,
    ms: MaxStats | None = None,
) -> np.ndarray:
    """Models that cannot be ruled out as #1; contains the true best with prob >= 1 - alpha.

    Model j is excluded only if some k is significantly better, using a one-sided max-t
    critical value over the p - 1 comparisons against j (this is all the coverage of the
    true best needs, since only comparisons against it matter).
    """
    ms = ms or max_statistics(theta_hat, theta_boot)
    crit = np.quantile(ms.best_one_sided, 1 - alpha, axis=0)
    beaten = (ms.diff.T > crit[:, None] * ms.se).any(axis=1)
    return ~beaten


def naive_best_set(theta_hat: np.ndarray, theta_boot: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Uncorrected: j survives unless the current leader beats it in a single two-sided z-test."""
    se = pairwise_se(theta_boot)
    top = int(np.argmax(theta_hat))
    z = (theta_hat[top] - theta_hat) / se[top]
    keep = z <= stats.norm.ppf(1 - alpha / 2)
    keep[top] = True
    return keep


def pairwise_z(theta_hat: np.ndarray, theta_boot: np.ndarray) -> np.ndarray:
    se = pairwise_se(theta_boot)
    return (theta_hat[:, None] - theta_hat[None, :]) / se


def holm(pvalues: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = p.size
    adj = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out


def minimum_detectable_difference(se: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest true gap a two-sided z-test at level alpha detects with the given power."""
    return float((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) * se)
