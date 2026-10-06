import numpy as np
import pytest
from conftest import separated_leaderboard

from pgnoise import ranks
from pgnoise.stats import bootstrap_scores, proteingym_score


def _fit(X, groups, n_boot=2000, seed=0):
    theta = proteingym_score(X, groups, 2)
    boot = bootstrap_scores(X, groups, 2, n_boot, np.random.default_rng(seed))
    return theta, boot


def test_ranks_desc_uses_min_rank_for_ties():
    np.testing.assert_array_equal(ranks.ranks_desc(np.array([0.3, 0.5, 0.5, 0.1])), [3, 1, 1, 4])


def test_separated_models_get_exact_ranks():
    X, groups, theta_true = separated_leaderboard()
    theta, boot = _fit(X, groups)
    truth = ranks.ranks_desc(theta_true)
    for mode in ("marginal", "simultaneous"):
        for stepdown in (False, True):
            lo, hi = ranks.pairwise_rank_ci(theta, boot, mode=mode, stepdown=stepdown)
            np.testing.assert_array_equal(lo, truth)
            np.testing.assert_array_equal(hi, truth)
    lo, hi = ranks.percentile_rank_ci(boot)
    np.testing.assert_array_equal(lo, truth)
    np.testing.assert_array_equal(hi, truth)


def test_identical_models_familywise_error_near_nominal():
    """Under a global null every interval should be [1, p] in ~95% of datasets."""
    n_sets, errors = 60, 0
    for seed in range(n_sets):
        X, groups, _ = separated_leaderboard(gap=0.0, seed=seed)
        theta, boot = _fit(X, groups, n_boot=500, seed=seed + 1000)
        lo, hi = ranks.pairwise_rank_ci(theta, boot, mode="simultaneous")
        errors += bool((lo > 1).any() or (hi < X.shape[1]).any())
    assert errors / n_sets <= 0.15


def test_interval_nesting():
    X, groups, _ = separated_leaderboard(p=10, gap=0.01, noise=0.08, seed=4)
    theta, boot = _fit(X, groups)
    point = ranks.ranks_desc(theta)
    ms = ranks.max_statistics(theta, boot)
    m1 = ranks.pairwise_rank_ci(theta, boot, mode="marginal", ms=ms, stepdown=False)
    md = ranks.pairwise_rank_ci(theta, boot, mode="marginal", ms=ms)
    s1 = ranks.pairwise_rank_ci(theta, boot, mode="simultaneous", ms=ms, stepdown=False)
    sd = ranks.pairwise_rank_ci(theta, boot, mode="simultaneous", ms=ms)
    for lo, hi in (m1, md, s1, sd):
        assert (lo <= point).all() and (point <= hi).all()
    # Simultaneous contains marginal; step-down is never wider than single-step.
    assert (s1[0] <= m1[0]).all() and (s1[1] >= m1[1]).all()
    assert (md[0] >= m1[0]).all() and (md[1] <= m1[1]).all()
    assert (sd[0] >= s1[0]).all() and (sd[1] <= s1[1]).all()


def test_best_set_keeps_leader_and_drops_clear_losers():
    X, groups, _ = separated_leaderboard(p=5, gap=0.2)
    X[:, 1] = X[:, 0] + np.random.default_rng(9).normal(0, 0.05, X.shape[0])  # model 1 ~ ties model 0
    theta, boot = _fit(X, groups)
    best = ranks.best_confidence_set(theta, boot)
    assert best[0] and best[1]
    assert not best[2:].any()


def test_naive_best_set_contains_leader():
    X, groups, _ = separated_leaderboard(p=5, gap=0.2)
    theta, boot = _fit(X, groups)
    naive = ranks.naive_best_set(theta, boot)
    assert naive[np.argmax(theta)] and naive.sum() == 1


def test_pairwise_se_symmetric_with_infinite_diagonal():
    boot = np.random.default_rng(0).normal(size=(500, 4))
    se = ranks.pairwise_se(boot)
    np.testing.assert_allclose(se, se.T)
    assert np.isinf(np.diag(se)).all()
    assert se[0, 1] == pytest.approx(np.std(boot[:, 0] - boot[:, 1], ddof=1), rel=1e-6)


def test_holm_known_values():
    np.testing.assert_allclose(ranks.holm(np.array([0.01, 0.04, 0.03])), [0.03, 0.06, 0.06])
    assert ranks.holm(np.array([0.5, 0.9])).max() <= 1


def test_minimum_detectable_difference():
    assert ranks.minimum_detectable_difference(1.0) == pytest.approx(1.959964 + 0.841621, rel=1e-5)
    assert ranks.minimum_detectable_difference(0.005) == pytest.approx(0.0140, abs=1e-4)
