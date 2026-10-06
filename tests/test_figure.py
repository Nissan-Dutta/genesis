from __future__ import annotations

import shutil
from pathlib import Path

from pgnoise import cli

RESULTS = Path(__file__).resolve().parents[1] / "results"


def test_headline_figure_builds_from_committed_tables(tmp_path: Path) -> None:
    out = tmp_path / "results"
    (out / "tables").mkdir(parents=True)
    for name in ["rank_intervals.csv", "neighbour_tests_top20.csv", "power_per_pair.csv", "power_required_units.csv"]:
        shutil.copy(RESULTS / "tables" / name, out / "tables" / name)
    shutil.copy(RESULTS / "summary.json", out / "summary.json")
    cli.main(["--out", str(out), "figure"])
    png, pdf = out / "figures" / "headline.png", out / "figures" / "headline.pdf"
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert pdf.read_bytes()[:5] == b"%PDF-"
