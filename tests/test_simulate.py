import numpy as np
import pytest
from conftest import separated_leaderboard

from pgnoise import simulate
from pgnoise.data import UnitTable


@pytest.fixture
def calibration() -> simulate.Calibration:
    X, groups, _ = separated_leaderboard(p=8, n_per_group=30, gap=0.01, noise=0.04, seed=11)
    units = UnitTable(X, groups, ["g0", "g1"], [f"m{i}" for i in range(8)], [(f"u{i}", "g") for i in range(len(X))])
    return simulate.calibrate(units)


def test_true_rank_sets_with_ties():
    rmin, rmax = simulate.true_rank_sets(np.array([0.5, 0.5, 0.3, 0.5, 0.1]))
    np.testing.assert_array_equal(rmin, [1, 1, 4, 1, 5])
    np.testing.assert_array_equal(rmax, [3, 3, 4, 3, 5])


def test_set_true_scores_hits_target(calibration):
    target = np.linspace(0.6, 0.4, len(calibration.models))
    mu = simulate.set_true_scores(calibration.mu, target)
    np.testing.assert_allclose(simulate.true_scores(mu), target)


def test_exact_tie_scenario_creates_ties(calibration):
    sc = simulate.Scenario("t", "", tie_blocks=((0, 3),))
    theta = simulate.true_scores(sc.mu(calibration))
    top = np.sort(theta)[::-1]
    assert top[0] == pytest.approx(top[1]) and top[1] == pytest.approx(top[2])
    assert top[2] > top[3]


def test_near_tie_scenario_spacing(calibration):
    sc = simulate.Scenario("n", "", near_tie_blocks=((0, 4, 0.002),))
    top = np.sort(simulate.true_scores(sc.mu(calibration)))[::-1][:4]
    np.testing.assert_allclose(np.diff(top), -0.002)


def test_residuals_are_centred(calibration):
    for R, a in zip(calibration.residuals, calibration.unit_effects):
        np.testing.assert_allclose(R.mean(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(R.mean(axis=1), 0, atol=1e-12)
        assert abs(a.mean()) < 1e-12


@pytest.mark.parametrize("noise", ["empirical", "gaussian"])
def test_generate_shapes_and_unbiasedness(calibration, noise):
    rng = np.random.default_rng(0)
    draws = [simulate.generate(calibration, calibration.mu, noise, rng) for _ in range(400)]
    X, groups = draws[0]
    assert X.shape == (calibration.group_sizes.sum(), len(calibration.models))
    est = np.mean([simulate.proteingym_score(Xi, gi, 2) for Xi, gi in draws], axis=0)
    np.testing.assert_allclose(est, simulate.true_scores(calibration.mu), atol=5e-3)


def test_small_coverage_run_is_sane(calibration):
    res = simulate.run_scenario(calibration, simulate.Scenario("x", ""), n_reps=40, n_boot=300, seed=1)
    s = res["summary"].set_index("method")
    assert set(s.index) == set(simulate.METHODS)
    assert s.loc["simultaneous_single_step", "joint_coverage"] >= 0.85
    assert (s["mean_width"] >= 0).all()
    assert 0 <= res["best"]["best_set_coverage"] <= 1


def test_parallel_matches_serial(calibration):
    sc = simulate.Scenario("x", "")
    a = simulate.run_scenario(calibration, sc, n_reps=8, n_boot=200, seed=3, n_jobs=1)
    b = simulate.run_scenario(calibration, sc, n_reps=8, n_boot=200, seed=3, n_jobs=2)
    np.testing.assert_allclose(a["summary"]["mean_coverage"], b["summary"]["mean_coverage"])
