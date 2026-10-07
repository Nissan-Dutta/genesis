import numpy as np
import pytest
from scipy import stats as sps

from pgnoise import power


def test_analytic_se_single_group():
    D = np.random.default_rng(0).normal(size=50)
    assert power.analytic_se(D, np.zeros(50, int), 1) == pytest.approx(D.std(ddof=1) / np.sqrt(50))
    assert power.analytic_se(D, np.zeros(50, int), 1, scale=4) == pytest.approx(D.std(ddof=1) / np.sqrt(200))


def test_power_is_alpha_at_zero_gap():
    assert power.analytic_power(0.01, 0.0) == pytest.approx(0.05)


def test_required_units_hits_target_power():
    se0, n0, delta = 0.006, 200, 0.01
    n = power.required_units(se0, n0, delta)
    achieved = power.analytic_power(se0 * np.sqrt(n0 / n), delta)
    assert achieved == pytest.approx(0.8, abs=0.002)  # the far tail adds a negligible sliver
    z = sps.norm.ppf(0.975) + sps.norm.ppf(0.8)
    assert n == pytest.approx(n0 * (z * se0 / delta) ** 2)


def test_simulated_power_matches_analytic_for_gaussian_differences():
    rng = np.random.default_rng(1)
    groups = np.repeat([0, 1], 60)
    pd_ = power.PairedDiffs("toy", rng.normal(0, 0.05, 120), groups, 2)
    se0 = power.analytic_se(pd_.D, groups, 2)
    scales = np.array([1.0, 3.0])
    sim = power.simulated_power(pd_, 0.01, scales, 4000, rng)
    np.testing.assert_allclose(sim, power.analytic_power(se0 / np.sqrt(scales), 0.01), atol=0.04)
    size = power.simulated_power(pd_, 0.0, np.array([1.0]), 4000, rng)[0]
    assert size == pytest.approx(0.05, abs=0.015)
