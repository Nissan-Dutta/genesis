from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from pgnoise import paper

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"


@pytest.fixture(scope="module")
def nums() -> list[paper.Number]:
    return paper.numbers(json.loads((ROOT / "results" / "summary.json").read_text()))


def test_generated_number_files_are_current(nums: list[paper.Number]) -> None:
    assert (PAPER / "numbers.tex").read_text() == paper.to_tex(nums), "run `uv run pgnoise numbers`"
    assert (PAPER / "numbers.md").read_text() == paper.to_markdown(nums), "run `uv run pgnoise numbers`"


def test_macro_names_are_valid_latex(nums: list[paper.Number]) -> None:
    names = [n.macro for n in nums]
    assert len(names) == len(set(names))
    assert all(re.fullmatch(r"[A-Za-z]+", name) for name in names)


def test_markdown_note_quotes_every_number_used_in_the_latex_note(nums: list[paper.Number]) -> None:
    by_macro = {n.macro: n for n in nums}
    used = {m for m in re.findall(r"\\([A-Za-z]+)", (PAPER / "technical-note.tex").read_text()) if m in by_macro}
    assert len(used) >= 40
    markdown = (PAPER / "technical-note.md").read_text()
    missing = {m: paper.markdown_value(by_macro[m]) for m in used if paper.markdown_value(by_macro[m]) not in markdown}
    assert not missing


def test_percentages_round_half_up() -> None:
    assert paper._pct(0.0595) == "6.0"
    assert paper._pct(0.0935) == "9.4"
    assert paper._pct(0.4921, 0) == "49"
