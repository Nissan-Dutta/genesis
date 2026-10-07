"""Studentised (bootstrap-t) pairwise rank intervals for the ProteinGym score.

The single-step / step-down procedures in ``ranks`` divide every bootstrap difference by one
fixed bootstrap SE. With small strata (Binding has 12 units) that ignores the sampling noise in
the SE itself, which makes an exact-level final step-down test liberal. Here each bootstrap
replicate gets its own linearised SE, so the max-|t| distribution includes SE noise.

Linearisation: theta_j = (1/G) sum_g mean_{u in g, observed} X_uj, so unit u contributes
psi_uj = o_uj (X_uj - m_gj) / (G n_gj) and
    Var(theta_j - theta_k) ~= sum_g n_g / (n_g - 1) * sum_{u in g} (psi_uj - psi_uk)^2.
"""

from __future__ import annotations

import numpy as np

from .stats import stratified_counts, weighted_group_means


def pairwise_variance(X: np.ndarray, groups: np.ndarray, n_groups: int, weights: np.ndarray) -> np.ndarray:
    """Linearised variance of every pairwise difference, one (p, p) matrix per weight row."""
    observed = ~np.isnan(X)
    X0 = np.where(observed, X, 0.0)
    B, p = weights.shape[0], X.shape[1]
    V = np.zeros((B, p, p))
    for g in range(n_groups):
        rows = groups == g
        n = int(rows.sum())
        W = weights[:, rows].astype(float)  # (B, n)
        Mg, Xg = observed[rows].astype(float), X0[rows]
        den = W @ Mg  # (B, p)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(den > 0, (W @ Xg) / den, 0.0)
            scale = np.where(den > 0, 1.0 / (n_groups * den), 0.0)
        psi = Mg[None] * (Xg[None] - mean[:, None, :]) * scale[:, None, :]  # (B, n, p)
        weighted = psi * W[:, :, None]
        V += (n / max(n - 1, 1)) * np.matmul(weighted.transpose(0, 2, 1), psi)
    d = np.diagonal(V, axis1=1, axis2=2)
    return d[:, :, None] + d[:, None, :] - 2 * V


def studentized_rank_ci(
    X: np.ndarray,
    groups: np.ndarray,
    n_groups: int,
    n_boot: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
    modes: tuple[str, ...] = ("marginal", "simultaneous"),
    chunk: int = 250,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Bootstrap-t step-down rank intervals for each requested mode."""
    p = X.shape[1]
    ones = np.ones((1, X.shape[0]))
    theta = np.nanmean(weighted_group_means(X, groups, n_groups, ones), axis=1)[0]
    diff = theta[:, None] - theta[None, :]
    se = np.sqrt(np.clip(pairwise_variance(X, groups, n_groups, ones)[0], 0, None))
    np.fill_diagonal(se, np.inf)
    se = np.where(se > 0, se, np.inf)
    t_obs = np.abs(diff) / se

    absT = np.empty((n_boot, p, p), dtype=np.float32)
    for s in range(0, n_boot, chunk):
        counts = stratified_counts(groups, min(chunk, n_boot - s), rng)
        with np.errstate(invalid="ignore"):
            tb = np.nanmean(weighted_group_means(X, groups, n_groups, counts), axis=1)
        se_b = np.sqrt(np.clip(pairwise_variance(X, groups, n_groups, counts), 0, None))
        with np.errstate(invalid="ignore", divide="ignore"):
            T = np.abs(tb[:, :, None] - tb[:, None, :] - diff) / se_b
        T[~np.isfinite(T)] = 0.0
        absT[s : s + counts.shape[0]] = T

    out = {}
    for mode in modes:
        active = ~np.eye(p, dtype=bool)
        rejected = np.zeros((p, p), dtype=bool)
        while active.any():
            masked = np.where(active, absT, 0.0)
            maxima = masked.max(axis=2) if mode == "marginal" else masked.max(axis=(1, 2))
            crit = np.quantile(maxima, 1 - alpha, axis=0)
            crit = crit if mode == "marginal" else np.full(p, crit)
            new = active & (t_obs > crit[:, None])
            if not new.any():
                break
            rejected |= new
            active &= ~new
        out[mode] = (1 + (rejected & (diff.T > 0)).sum(axis=1), p - (rejected & (diff > 0)).sum(axis=1))
    return out
