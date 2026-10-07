from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pgnoise.data import AssayTable


@pytest.fixture
def toy_assays() -> AssayTable:
    """Six assays on four proteins in two function groups, three models, one missing score."""
    scores = pd.DataFrame(
        {
            "A": [0.5, 0.7, 0.4, 0.6, 0.3, 0.5],
            "B": [0.4, 0.6, 0.5, 0.2, 0.2, 0.4],
            "C": [0.1, np.nan, 0.3, 0.3, 0.1, 0.2],
        },
        index=[f"dms{i}" for i in range(6)],
    )
    meta = pd.DataFrame(
        {
            "uniprot": ["P1", "P1", "P2", "P3", "P3", "P4"],
            "function": ["Activity", "Activity", "Activity", "Stability", "Stability", "Stability"],
            "taxon": ["Human"] * 6,
            "msa_depth": ["High"] * 6,
            "n_mutants": [100] * 6,
        },
        index=scores.index,
    )
    return AssayTable(scores, meta)


def separated_leaderboard(p: int = 6, n_per_group: int = 40, gap: float = 0.5, noise: float = 0.05,
                          seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit x model matrix whose true model means are far apart relative to the noise."""
    rng = np.random.default_rng(seed)
    theta = gap * np.arange(p)[::-1]
    groups = np.repeat([0, 1], n_per_group)
    X = theta[None, :] + rng.normal(0, 0.2, (groups.size, 1)) + rng.normal(0, noise, (groups.size, p))
    return X, groups, theta
