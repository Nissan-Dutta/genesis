import numpy as np
import pytest

from pgnoise import aggregation as ag
from pgnoise.data import to_units
from pgnoise.stats import proteingym_score


@pytest.fixture
def design(toy_assays):
    return ag.make_design(toy_assays)


def _point(design, name):
    return ag.point_scores(design, ag.SCHEMES[name])


def test_proteingym_scheme_matches_core_score(toy_assays, design):
    units = to_units(toy_assays)
    np.testing.assert_allclose(_point(design, "proteingym"), proteingym_score(units.X, units.groups, units.n_groups))


def test_flat_and_median_match_pandas(toy_assays, design):
    np.testing.assert_allclose(_point(design, "flat_mean"), toy_assays.scores.mean().to_numpy())
    # Model C has 5 observed assays, A and B have 6 (even counts use the midpoint rule).
    np.testing.assert_allclose(_point(design, "median"), toy_assays.scores.median().to_numpy())


def test_uniprot_and_function_group_schemes_match_pandas(toy_assays, design):
    frame = toy_assays.scores.join(toy_assays.meta[["uniprot", "function"]])
    by_protein = frame.groupby("uniprot").mean(numeric_only=True).mean()
    by_function = frame.groupby("function").mean(numeric_only=True).mean()
    np.testing.assert_allclose(_point(design, "uniprot_weighted"), by_protein[design.models].to_numpy())
    np.testing.assert_allclose(_point(design, "function_group_mean"), by_function[design.models].to_numpy())


def test_weighted_median_equals_median_of_repeated_values():
    rng = np.random.default_rng(0)
    S = rng.normal(size=(9, 3))
    w = rng.integers(0, 4, size=(1, 9))
    w[0, 0] = 1
    expected = [np.median(np.repeat(S[:, m], w[0])) for m in range(3)]
    np.testing.assert_allclose(ag._weighted_median(S, w.astype(float))[0], expected)


@pytest.mark.parametrize("name", list(ag.SCHEMES))
def test_bootstrap_shapes_and_unit_weights(design, name):
    boot = ag.bootstrap_scheme(design, ag.SCHEMES[name], 50, np.random.default_rng(1))
    assert boot.shape == (50, len(design.models))
    assert np.isfinite(boot).all()


def test_generic_unit_weighted_path_equals_proteingym(toy_assays, design):
    generic = ag.Scheme("pg_generic", "", "unit", True)
    np.testing.assert_allclose(ag.point_scores(design, generic), _point(design, "proteingym"))
    counts = np.random.default_rng(3).integers(0, 3, size=(20, design.unit_groups.size))
    counts[:, 0] = 1
    counts[:, -1] = 1
    np.testing.assert_allclose(ag.scheme_scores(design, generic, counts),
                               ag.scheme_scores(design, ag.SCHEMES["proteingym"], counts), equal_nan=True)
