import numpy as np
import pytest
from conftest import separated_leaderboard

from pgnoise import ranks, studentized
from pgnoise.stats import bootstrap_scores


def test_linearised_variance_single_group_is_sample_variance_over_n():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 3))
    V = studentized.pairwise_variance(X, np.zeros(30, int), 1, np.ones((1, 30)))[0]
    d = X[:, 0] - X[:, 1]
    assert V[0, 1] == pytest.approx(d.var(ddof=1) / 30)
    assert V[0, 0] == pytest.approx(0)


def test_linearised_variance_tracks_bootstrap_variance():
    X, groups, _ = separated_leaderboard(p=5, gap=0.01, noise=0.05, seed=3)
    V = studentized.pairwise_variance(X, groups, 2, np.ones((1, X.shape[0])))[0]
    se_boot = ranks.pairwise_se(bootstrap_scores(X, groups, 2, 8000, np.random.default_rng(1)))
    iu = np.triu_indices(5, 1)
    np.testing.assert_allclose(np.sqrt(V[iu]), se_boot[iu], rtol=0.08)


def test_linearised_variance_handles_missing():
    X, groups, _ = separated_leaderboard(p=4, seed=5)
    X[:6, 2] = np.nan
    V = studentized.pairwise_variance(X, groups, 2, np.ones((1, X.shape[0])))[0]
    assert np.isfinite(V).all() and (V[np.triu_indices(4, 1)] > 0).all()


def test_studentized_intervals_exact_when_separated():
    X, groups, theta = separated_leaderboard()
    out = studentized.studentized_rank_ci(X, groups, 2, 400, np.random.default_rng(0))
    truth = ranks.ranks_desc(theta)
    for lo, hi in out.values():
        np.testing.assert_array_equal(lo, truth)
        np.testing.assert_array_equal(hi, truth)


def test_studentized_intervals_contain_point_rank():
    X, groups, _ = separated_leaderboard(p=8, gap=0.01, noise=0.08, seed=4)
    theta = np.nanmean([X[groups == g].mean(axis=0) for g in (0, 1)], axis=0)
    point = ranks.ranks_desc(theta)
    out = studentized.studentized_rank_ci(X, groups, 2, 400, np.random.default_rng(2))
    m_lo, m_hi = out["marginal"]
    s_lo, s_hi = out["simultaneous"]
    assert (m_lo <= point).all() and (point <= m_hi).all()
    assert (s_lo <= m_lo).all() and (s_hi >= m_hi).all()
