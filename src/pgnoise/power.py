"""Power at the top of the leaderboard: how many assays to detect a gain of delta Spearman?

Setting: a new model beats the current leader by a true delta on ProteinGym's aggregate. Its
paired per-unit differences with the leader are assumed to be as variable as the observed
differences between the leader and the current top-10 models. More assays are assumed to arrive
as new (UniProt, function) units, keeping today's function-group mix and ~1.085 assays per unit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps

from .data import UnitTable
from .stats import proteingym_score


@dataclass(frozen=True)
class PairedDiffs:
    label: str
    D: np.ndarray  # per-unit paired differences (challenger - leader), complete units only
    groups: np.ndarray
    n_groups: int

    @property
    def group_sizes(self) -> np.ndarray:
        return np.bincount(self.groups, minlength=self.n_groups)


def paired_diffs(units: UnitTable, a: str, b: str) -> PairedDiffs:
    ia, ib = units.models.index(a), units.models.index(b)
    D = units.X[:, ib] - units.X[:, ia]
    keep = ~np.isnan(D)
    return PairedDiffs(f"{a} vs {b}", D[keep], units.groups[keep], units.n_groups)


def top_pairs(units: UnitTable, k: int = 10) -> list[PairedDiffs]:
    """Paired differences between the current #1 and each of the next ``k`` complete-coverage models."""
    order = np.argsort(-proteingym_score(units.X, units.groups, units.n_groups))
    complete = ~np.isnan(units.X).any(axis=0)
    leader = units.models[order[0]]
    challengers = [units.models[i] for i in order[1:] if complete[i]][:k]
    return [paired_diffs(units, leader, m) for m in challengers]


def analytic_se(D: np.ndarray, groups: np.ndarray, n_groups: int, scale: float = 1.0) -> float:
    """SE of the group-balanced mean difference with every group's size multiplied by ``scale``."""
    var = 0.0
    for g in range(n_groups):
        d = D[groups == g]
        var += d.var(ddof=1) / (d.size * scale)
    return float(np.sqrt(var) / n_groups)


def analytic_power(se: np.ndarray | float, delta: float, alpha: float = 0.05) -> np.ndarray:
    """Two-sided z-test power for a true gap ``delta`` with standard error ``se``."""
    z = sps.norm.ppf(1 - alpha / 2)
    shift = delta / np.asarray(se, dtype=float)
    return sps.norm.cdf(shift - z) + sps.norm.cdf(-shift - z)


def required_units(se0: float, n0: int, delta: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Units needed (proportional allocation) so a true gap ``delta`` is detected with ``power``."""
    z = sps.norm.ppf(1 - alpha / 2) + sps.norm.ppf(power)
    return float(n0 * (z * se0 / delta) ** 2)


def simulated_power(pd_: PairedDiffs, delta: float, scales: np.ndarray, n_sims: int,
                    rng: np.random.Generator, alpha: float = 0.05) -> np.ndarray:
    """Resample real paired differences (re-centred so the true gap is ``delta``) at scaled sizes.

    Each simulated benchmark is tested exactly as on the real data: group-balanced mean
    difference over its plug-in SE, two-sided at level ``alpha``. Returns power per scale.
    """
    z = sps.norm.ppf(1 - alpha / 2)
    out = np.empty(len(scales))
    for i, s in enumerate(scales):
        mean = np.zeros(n_sims)
        var = np.zeros(n_sims)
        for g in range(pd_.n_groups):
            d = pd_.D[pd_.groups == g]
            centred = d - d.mean() + delta
            n = max(2, int(round(s * d.size)))
            draw = centred[rng.integers(0, d.size, size=(n_sims, n))]
            mean += draw.mean(axis=1)
            var += draw.var(axis=1, ddof=1) / n
        t = (mean / pd_.n_groups) / (np.sqrt(var) / pd_.n_groups)
        out[i] = np.mean(np.abs(t) > z)
    return out


ASSAYS_PER_UNIT = 217 / 200


def power_tables(units: UnitTable, deltas: tuple[float, ...] = (0.005, 0.01), k: int = 10,
                 n_sims: int = 4000, seed: int = 5, alpha: float = 0.05, power: float = 0.8) -> dict:
    """Per-pair requirements, power curves (analytic for every pair, simulated for the median pair)
    and the simulated size of the test (delta = 0)."""
    pairs = top_pairs(units, k)
    se0 = np.array([analytic_se(p.D, p.groups, p.n_groups) for p in pairs])
    n0 = np.array([p.D.size for p in pairs])
    median_idx = int(np.argsort(se0)[len(se0) // 2])
    rng = np.random.default_rng(seed)

    per_pair = []
    for p, s, n in zip(pairs, se0, n0):
        row = {"pair": p.label, "se": s, "units_now": int(n),
               "mdd_now": float((sps.norm.ppf(1 - alpha / 2) + sps.norm.ppf(power)) * s)}
        for d in deltas:
            need = required_units(s, n, d, alpha, power)
            row[f"units_for_{d}"] = need
            row[f"assays_for_{d}"] = need * ASSAYS_PER_UNIT
            row[f"x_today_for_{d}"] = need / n
        per_pair.append(row)

    scales = np.geomspace(0.25, 40, 24)
    curves = []
    for d in deltas:
        sim = simulated_power(pairs[median_idx], d, scales, n_sims, rng, alpha)
        for i, (p, s, n) in enumerate(zip(pairs, se0, n0)):
            for j, sc in enumerate(scales):
                curves.append({"pair": "median" if i == median_idx else p.label, "delta": d, "units": sc * n,
                               "analytic_power": float(analytic_power(s / np.sqrt(sc), d, alpha)),
                               "simulated_power": float(sim[j]) if i == median_idx else np.nan})
    size = simulated_power(pairs[median_idx], 0.0, np.array([0.25, 0.5, 1, 2, 4]), n_sims, rng, alpha)

    grid = np.geomspace(0.002, 0.03, 30)
    required = [{"pair": p.label, "delta": d, "units_needed": required_units(s, n, d, alpha, power)}
                for p, s, n in zip(pairs, se0, n0) for d in grid]
    return {"per_pair": pd.DataFrame(per_pair), "curves": pd.DataFrame(curves),
            "required": pd.DataFrame(required), "median_pair": pairs[median_idx].label,
            "size_at_delta0": dict(zip(["0.25x", "0.5x", "1x", "2x", "4x"], size.round(4).tolist()))}
