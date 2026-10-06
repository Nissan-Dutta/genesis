"""Integration checks against the real ProteinGym files; skipped until `pgnoise download` has run."""

import numpy as np
import pytest

from pgnoise import analysis, data

pytestmark = pytest.mark.skipif(
    not (data.DEFAULT_DATA_DIR / "DMS_substitutions_Spearman_DMS_level.csv").exists(),
    reason="ProteinGym data not downloaded",
)


def test_published_averages_and_ranks_reproduce_exactly():
    rep = analysis.reproduce_leaderboard(data.load_assays(), data.load_summary(), n_boot=200)
    np.testing.assert_array_equal(np.round(rep["Average_Spearman__ours"], 3), rep["Average_Spearman__published"])
    np.testing.assert_array_equal(rep["rank__ours"], rep["rank__published"])


def test_unit_structure():
    units = data.to_units(data.load_assays())
    assert units.X.shape == (200, 97)
    assert units.group_names == data.FUNCTION_GROUPS
