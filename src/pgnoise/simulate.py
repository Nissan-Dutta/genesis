"""Synthetic leaderboards calibrated to ProteinGym, with known true ranks, for coverage checks.

Estimand. Each model m has a true benchmark aggregate
    theta_m = mean over function groups g of E[score of m on a random (UniProt, g) unit],
i.e. ProteinGym's headline score in the limit of infinitely many proteins per group. We cover
rank(theta_m) = 1 + #{k : theta_k > theta_m}. This is *not* a prediction interval for the
model's rank on a new assay (the target of Neuhof & Benjamini 2026); per-assay ranks vary far
more than the rank of the aggregate.

Generator, per synthetic leaderboard:
    X[u, m] = mu[g(u), m] + a[u] + eps[u, m]
with the real number of units per group, ``mu`` the observed group means (optionally adjusted
to create ties), ``a`` resampled real unit effects, and ``eps`` either resampled real residual
rows (keeps cross-model correlation, heteroscedasticity and tails) or Gaussian draws with the
exact within-group empirical covariance.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import ranks
from .data import UnitTable
from .stats import bootstrap_scores, group_means, proteingym_score
from .studentized import studentized_rank_ci

METHODS = (
    "percentile",
    "marginal_single_step",
    "marginal_stepdown",
    "simultaneous_single_step",
    "simultaneous_stepdown",
    "marginal_stepdown_t",
    "simultaneous_stepdown_t",
)


@dataclass(frozen=True)
class Calibration:
    mu: np.ndarray  # (n_groups, p) group means
    unit_effects: list[np.ndarray]  # per group: (n_g,) row effects
    residuals: list[np.ndarray]  # per group: (n_g, p) residual rows, df-corrected
    group_sizes: np.ndarray
    models: list[str]


def calibrate(units: UnitTable) -> Calibration:
    """Two-way decomposition of the real unit x model matrix (complete-coverage models only)."""
    complete = ~np.isnan(units.X).any(axis=0)
    models = [m for m, c in zip(units.models, complete) if c]
    X = units.X[:, complete]
    mu = group_means(X, units.groups, units.n_groups)
    effects, resid, sizes = [], [], []
    for g in range(units.n_groups):
        Xg = X[units.groups == g] - mu[g]
        a = Xg.mean(axis=1)
        e = Xg - a[:, None]
        n = Xg.shape[0]
        effects.append(a)
        # Double-centred residuals lose ~one df per row and column; rescale to unbiased size.
        resid.append(e * np.sqrt(n / (n - 1)))
        sizes.append(n)
    return Calibration(mu, effects, resid, np.array(sizes), models)


def true_scores(mu: np.ndarray) -> np.ndarray:
    return mu.mean(axis=0)


def set_true_scores(mu: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Shift each model's group profile by a constant so its aggregate equals ``target``."""
    return mu + (target - true_scores(mu))[None, :]


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    noise: str = "empirical"  # "empirical" | "gaussian"
    tie_blocks: tuple[tuple[int, int], ...] = ()  # 0-based [start, stop) rank positions forced equal
    near_tie_blocks: tuple[tuple[int, int, float], ...] = ()  # (start, stop, spacing)
    spread: float = 1.0  # multiply true gaps around the mean

    def mu(self, cal: Calibration) -> np.ndarray:
        theta = true_scores(cal.mu)
        order = np.argsort(-theta)
        target = theta.mean() + self.spread * (theta - theta.mean())
        for start, stop in self.tie_blocks:
            target[order[start:stop]] = target[order[start]]
        for start, stop, spacing in self.near_tie_blocks:
            idx = order[start:stop]
            target[idx] = target[order[start]] - spacing * np.arange(idx.size)
        return set_true_scores(cal.mu, target)


SCENARIOS: dict[str, Scenario] = {
    s.name: s
    for s in [
        Scenario("calibrated", "True scores = observed group means; resampled real residual rows"),
        Scenario("calibrated_gaussian", "As calibrated, Gaussian residuals with empirical covariance",
                 noise="gaussian"),
        Scenario("exact_ties", "Top 5 exactly tied, and ranks 20-29 exactly tied",
                 tie_blocks=((0, 5), (19, 29))),
        Scenario("near_ties", "Top 5 spaced 0.002 apart (well below the MDD), ranks 20-29 spaced 0.001",
                 near_tie_blocks=((0, 5, 0.002), (19, 29, 0.001))),
        Scenario("separated", "True gaps x3 (an easier, well-separated leaderboard)", spread=3.0),
    ]
}


def generate(cal: Calibration, mu: np.ndarray, noise: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One synthetic unit x model matrix and its group labels."""
    blocks, groups = [], []
    for g, n in enumerate(cal.group_sizes):
        a = rng.choice(cal.unit_effects[g], size=n, replace=True)
        R = cal.residuals[g]
        if noise == "empirical":
            eps = R[rng.integers(0, R.shape[0], size=n)]
        elif noise == "gaussian":
            eps = rng.standard_normal((n, R.shape[0])) @ R / np.sqrt(R.shape[0])
        else:
            raise ValueError(f"unknown noise {noise!r}")
        blocks.append(mu[g][None, :] + a[:, None] + eps)
        groups.append(np.full(n, g))
    return np.vstack(blocks), np.concatenate(groups)


def true_rank_sets(theta: np.ndarray, tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """For each model, the smallest and largest rank it could legitimately be given (ties)."""
    better = (theta[None, :] > theta[:, None] + tol).sum(axis=1)
    not_worse = (theta[None, :] >= theta[:, None] - tol).sum(axis=1)
    return 1 + better, not_worse


@dataclass
class RepResult:
    covered_strict: dict[str, np.ndarray] = field(default_factory=dict)
    covered_lenient: dict[str, np.ndarray] = field(default_factory=dict)
    width: dict[str, np.ndarray] = field(default_factory=dict)
    best_covered: bool = False
    best_any: bool = False
    best_per_model: float = 0.0
    best_size: int = 0
    naive_best_covered: bool = False
    naive_best_size: int = 0
    se_top_gap: float = 0.0


def one_rep(cal: Calibration, mu: np.ndarray, noise: str, n_boot: int, alpha: float,
            rng: np.random.Generator, correction: bool = False) -> RepResult:
    theta = true_scores(mu)
    rmin, rmax = true_rank_sets(theta)
    X, groups = generate(cal, mu, noise, rng)
    n_groups = mu.shape[0]
    est = proteingym_score(X, groups, n_groups)
    boot = bootstrap_scores(X, groups, n_groups, n_boot, rng, small_sample_correction=correction)
    ms = ranks.max_statistics(est, boot)
    intervals = {
        "percentile": ranks.percentile_rank_ci(boot, alpha),
        "marginal_single_step": ranks.pairwise_rank_ci(est, boot, alpha, "marginal", ms, stepdown=False),
        "marginal_stepdown": ranks.pairwise_rank_ci(est, boot, alpha, "marginal", ms),
        "simultaneous_single_step": ranks.pairwise_rank_ci(est, boot, alpha, "simultaneous", ms, stepdown=False),
        "simultaneous_stepdown": ranks.pairwise_rank_ci(est, boot, alpha, "simultaneous", ms),
    }
    student = studentized_rank_ci(X, groups, n_groups, n_boot, rng, alpha)
    intervals["marginal_stepdown_t"] = student["marginal"]
    intervals["simultaneous_stepdown_t"] = student["simultaneous"]
    res = RepResult()
    for name, (lo, hi) in intervals.items():
        res.covered_strict[name] = (lo <= rmin) & (hi >= rmax)
        res.covered_lenient[name] = (lo <= rmax) & (hi >= rmin)
        res.width[name] = hi - lo
    true_best = rmin == 1
    best = ranks.best_confidence_set(est, boot, alpha, ms)
    naive = ranks.naive_best_set(est, boot, alpha)
    res.best_covered = bool(best[true_best].all())
    res.best_any = bool(best[true_best].any())
    res.best_per_model = float(best[true_best].mean())
    res.best_size = int(best.sum())
    res.naive_best_covered = bool(naive[true_best].all())
    res.naive_best_size = int(naive.sum())
    order = np.argsort(-theta)
    res.se_top_gap = float(ms.se[order[0], order[1]])
    return res


def _rep_batch(cal: Calibration, mu: np.ndarray, noise: str, n_boot: int, alpha: float,
               seeds: list[np.random.SeedSequence], correction: bool) -> list[RepResult]:
    return [one_rep(cal, mu, noise, n_boot, alpha, np.random.default_rng(s), correction) for s in seeds]


def run_scenario(cal: Calibration, scenario: Scenario, n_reps: int, n_boot: int = 1000,
                 alpha: float = 0.05, seed: int = 0, n_jobs: int = 1, correction: bool = False) -> dict:
    """Coverage, widths and best-set behaviour of every interval method over ``n_reps`` leaderboards.

    Each replicate gets its own spawned seed, so results do not depend on ``n_jobs``.
    """
    mu = scenario.mu(cal)
    theta = true_scores(mu)
    rmin, rmax = true_rank_sets(theta)
    seeds = np.random.SeedSequence(seed).spawn(n_reps)
    if n_jobs == 1:
        reps = _rep_batch(cal, mu, scenario.noise, n_boot, alpha, seeds, correction)
    else:
        batches = [seeds[i::n_jobs] for i in range(n_jobs)]
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = [pool.submit(_rep_batch, cal, mu, scenario.noise, n_boot, alpha, b, correction) for b in batches]
            results = [f.result() for f in futures]
        reps = [None] * n_reps
        for i, batch in enumerate(results):
            reps[i::n_jobs] = batch
    per_model, summary = [], []
    for method in METHODS:
        strict = np.array([r.covered_strict[method] for r in reps])
        lenient = np.array([r.covered_lenient[method] for r in reps])
        width = np.array([r.width[method] for r in reps])
        per_model.append(pd.DataFrame({
            "method": method, "model": cal.models, "true_rank_min": rmin, "true_rank_max": rmax,
            "coverage": strict.mean(axis=0), "coverage_lenient": lenient.mean(axis=0),
            "mean_width": width.mean(axis=0),
        }))
        summary.append({
            "scenario": scenario.name, "method": method,
            "mean_coverage": strict.mean(), "min_model_coverage": strict.mean(axis=0).min(),
            "mean_coverage_lenient": lenient.mean(),
            "joint_coverage": strict.all(axis=1).mean(),
            "mean_width": width.mean(), "mean_width_top10": width[:, np.argsort(-theta)[:10]].mean(),
        })
    best = {
        "scenario": scenario.name,
        "n_true_best": int((rmin == 1).sum()),
        "best_set_coverage": np.mean([r.best_covered for r in reps]),
        "best_set_any_true_best": np.mean([r.best_any for r in reps]),
        "best_set_per_true_best": np.mean([r.best_per_model for r in reps]),
        "best_set_mean_size": np.mean([r.best_size for r in reps]),
        "naive_best_set_coverage": np.mean([r.naive_best_covered for r in reps]),
        "naive_best_set_mean_size": np.mean([r.naive_best_size for r in reps]),
        "mean_se_top_gap": np.mean([r.se_top_gap for r in reps]),
    }
    return {"summary": pd.DataFrame(summary), "per_model": pd.concat(per_model, ignore_index=True),
            "best": best, "n_reps": n_reps, "n_boot": n_boot}
