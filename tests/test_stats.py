import numpy as np
import pandas as pd
import pytest

from pgnoise.data import to_units
from pgnoise.stats import (
    bootstrap_scores,
    group_means,
    proteingym_gap_se,
    proteingym_score,
    stratified_counts,
    weighted_group_means,
)


def test_to_units_matches_pandas_groupby(toy_assays):
    units = to_units(toy_assays)
    expected = toy_assays.scores.join(toy_assays.meta[["uniprot", "function"]]).groupby(
        ["uniprot", "function"]).mean(numeric_only=True)
    np.testing.assert_allclose(units.X, expected.to_numpy(), equal_nan=True)
    assert units.group_names == ["Activity", "Stability"]
    assert units.units == list(expected.index)


def test_unit_mean_skips_missing_assay(toy_assays):
    units = to_units(toy_assays)
    p1 = units.units.index(("P1", "Activity"))
    assert units.X[p1, units.models.index("C")] == pytest.approx(0.1)


def test_proteingym_score_is_mean_of_group_means(toy_assays):
    units = to_units(toy_assays)
    score = proteingym_score(units.X, units.groups, units.n_groups)
    # Model A: Activity units P1=0.6, P2=0.4 -> 0.5; Stability units P3=0.45, P4=0.5 -> 0.475.
    assert score[0] == pytest.approx((0.5 + 0.475) / 2)


def test_group_without_data_is_skipped():
    X = np.array([[0.2, 0.4], [0.4, np.nan], [0.9, np.nan]])
    groups = np.array([0, 1, 1])
    score = proteingym_score(X, groups, 2)
    assert score[1] == pytest.approx(0.4)
    assert score[0] == pytest.approx((0.2 + 0.65) / 2)


def test_stratified_counts_stay_within_groups():
    groups = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    counts = stratified_counts(groups, 500, np.random.default_rng(0))
    for g in np.unique(groups):
        assert (counts[:, groups == g].sum(axis=1) == (groups == g).sum()).all()
    assert counts.min() >= 0


def test_unit_weights_reproduce_group_means(toy_assays):
    units = to_units(toy_assays)
    w = np.ones((1, units.X.shape[0]))
    np.testing.assert_allclose(
        weighted_group_means(units.X, units.groups, units.n_groups, w)[0],
        group_means(units.X, units.groups, units.n_groups),
        equal_nan=True,
    )


def test_bootstrap_scores_centre_on_estimate():
    rng = np.random.default_rng(1)
    X = rng.normal(0.5, 0.1, (60, 4))
    groups = np.repeat([0, 1, 2], 20)
    boot = bootstrap_scores(X, groups, 3, 4000, rng)
    assert boot.shape == (4000, 4)
    np.testing.assert_allclose(boot.mean(axis=0), proteingym_score(X, groups, 3), atol=3e-3)


def test_gap_se_matches_analytic_single_group():
    rng = np.random.default_rng(2)
    n = 80
    X = rng.normal(0, 1, (n, 3))
    se = proteingym_gap_se(X, np.zeros(n, int), 1, top=0, n_boot=20_000, rng=rng)
    assert se[0] == 0
    D = X[:, 1] - X[:, 0]
    analytic = D.std(ddof=0) / np.sqrt(n)
    assert se[1] == pytest.approx(analytic, rel=0.05)


def test_gap_se_handles_missing_units():
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, (40, 2))
    X[:5, 1] = np.nan
    se = proteingym_gap_se(X, np.repeat([0, 1], 20), 2, top=0, n_boot=2000, rng=rng)
    assert np.isfinite(se).all() and se[1] > 0


def test_pandas_route_equals_numpy_route(toy_assays):
    """The aggregation must match ProteinGym's pandas code path exactly."""
    frame = toy_assays.scores.join(toy_assays.meta[["uniprot", "function"]])
    pg = frame.groupby(["uniprot", "function"]).mean(numeric_only=True).groupby("function").mean().mean()
    units = to_units(toy_assays)
    ours = pd.Series(proteingym_score(units.X, units.groups, units.n_groups), index=units.models)
    pd.testing.assert_series_equal(ours, pg[units.models], check_names=False)
