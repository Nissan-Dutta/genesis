"""Integration checks against the real ProteinGym files; skipped until `pgnoise download` has run."""

import numpy as np
import pytest

import pandas as pd

from pgnoise import analysis, data, robustness

pytestmark = pytest.mark.skipif(
    not (data.DEFAULT_DATA_DIR / "DMS_substitutions_Spearman_DMS_level.csv").exists(),
    reason="ProteinGym data not downloaded",
)


def test_published_averages_and_ranks_reproduce_exactly():
    rep = analysis.reproduce_leaderboard(data.load_assays(), data.load_summary(), n_boot=200)
    np.testing.assert_array_equal(np.round(rep["Average_Spearman__ours"], 3), rep["Average_Spearman__published"])
    np.testing.assert_array_equal(rep["rank__ours"], rep["rank__published"])


@pytest.mark.parametrize("metric", data.METRICS[1:])
def test_other_metric_averages_reproduce(metric):
    rep = analysis.reproduce_leaderboard(data.load_assays(metric=metric), data.load_summary(metric=metric),
                                         n_boot=100, metric=metric)
    np.testing.assert_array_equal(np.round(rep[f"Average_{metric}__ours"], 3), rep[f"Average_{metric}__published"])
    np.testing.assert_array_equal(rep["rank__ours"], rep["rank__published"])


def test_published_uniprot_level_average_quirk():
    assays = data.load_assays()
    ours = robustness.proteingym_uniprot_level_average(assays, pd.read_csv(data.path_for("reference")))
    published = pd.read_csv(data.path_for("uniprot_level")).iloc[-1]
    cols = [c for c in published.index if c not in ("UniProt_ID", "MSA_Neff_L_category", "Taxon", "Selection Type")]
    np.testing.assert_array_equal(np.round(ours.to_numpy(), 3), published[cols].astype(float).to_numpy())


def test_unit_structure():
    units = data.to_units(data.load_assays())
    assert units.X.shape == (200, 97)
    assert units.group_names == data.FUNCTION_GROUPS
