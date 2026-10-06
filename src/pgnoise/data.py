"""Download and load ProteinGym's public DMS-substitution Spearman leaderboard data."""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Pinned to the latest ProteinGym commit that touched the benchmark files at the time of writing
# ("fixing missing benchmark files", 2026-03-25). Bump deliberately and update the checksums.
PROTEINGYM_COMMIT = "144fe22b07dfaeec2b366f2346203a9838a55b4c"
RAW_BASE = f"https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/{PROTEINGYM_COMMIT}"
_SPEARMAN_DIR = "benchmarks/DMS_zero_shot/substitutions/Spearman"

FILES: dict[str, tuple[str, str]] = {
    "dms_level": (
        f"{_SPEARMAN_DIR}/DMS_substitutions_Spearman_DMS_level.csv",
        "f432423b87f79ac9778dfac86e3d95be041d246bc618dc0a406b35b0b7466437",
    ),
    "uniprot_level": (
        f"{_SPEARMAN_DIR}/DMS_substitutions_Spearman_Uniprot_level.csv",
        "c8f88f24f80fa0afc23783ce7c11813f8a90a1561645eaa04929d2f780188e97",
    ),
    "function_level": (
        f"{_SPEARMAN_DIR}/DMS_substitutions_Spearman_Uniprot_Selection_Type_level.csv",
        "8a0d00e6092242ca1ce5b4770612af347487ebf31802aaaebb3ddbd0660ea013",
    ),
    "summary": (
        f"{_SPEARMAN_DIR}/Summary_performance_DMS_substitutions_Spearman.csv",
        "ee61a7c09efbfbd8d9a6c9355092c798032a578fa58c6f6256b5da8778555fd4",
    ),
    "reference": (
        "reference_files/DMS_substitutions.csv",
        "a8f498011532a74aa9fe556a50555a75e928c5837d19c06a87592ae04049b308",
    ),
}

METRICS = ["Spearman", "AUC", "MCC", "NDCG", "Top_recall"]

_OTHER_METRIC_CHECKSUMS = {
    "AUC": ("83683a894f4ef693ecd367144557c530f6a33f12f9ec6c176a4c19aaaaff3068",
            "880f04503b28cb579cb2302c4b56c51505f87cce3f72807e40e2120dcc3ff24a"),
    "MCC": ("5c1e48975e40a130d208a754bb68ea909abd43874120021b9f7aacba7025e6cd",
            "a6d9b892a7b040f2c2dcccac049ab2e928a120135e86984b37b706da5ec504a0"),
    "NDCG": ("b2debc91a6305fb183e930f2b8aeabf91f8d8b49351506d0fbf0cee6b2d22aae",
             "bcac893d0e73a883ff0c3c6b44bb35583e770cd985e88c8367d023762302d253"),
    "Top_recall": ("ab9cee646fbc8a92728bff83f1eefc5bf0097b07a12bbe64db9715403305d7de",
                   "c08af40426a0300beb81c9b9fdba5dd2c2be4e3db64a3b80bc6f0c6733f5d913"),
}
for _metric, (_dms_sha, _summary_sha) in _OTHER_METRIC_CHECKSUMS.items():
    _dir = f"benchmarks/DMS_zero_shot/substitutions/{_metric}"
    FILES[f"dms_level_{_metric}"] = (f"{_dir}/DMS_substitutions_{_metric}_DMS_level.csv", _dms_sha)
    FILES[f"summary_{_metric}"] = (f"{_dir}/Summary_performance_DMS_substitutions_{_metric}.csv", _summary_sha)


def _metric_key(kind: str, metric: str) -> str:
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")
    return kind if metric == "Spearman" else f"{kind}_{metric}"

DEFAULT_DATA_DIR = Path("data/raw")

DMS_META_COLUMNS = [
    "DMS ID",
    "Number of Mutants",
    "Selection Type",
    "UniProt ID",
    "MSA_Neff_L_category",
    "Taxon",
]

FUNCTION_GROUPS = ["Activity", "Binding", "Expression", "OrganismalFitness", "Stability"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(data_dir: Path = DEFAULT_DATA_DIR, force: bool = False) -> dict[str, Path]:
    """Fetch the pinned ProteinGym files into ``data_dir`` and verify their checksums."""
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, (remote, expected) in FILES.items():
        path = data_dir / Path(remote).name
        if force or not path.exists() or _sha256(path) != expected:
            urllib.request.urlretrieve(f"{RAW_BASE}/{remote}", path)
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch for {path}: expected {expected}, got {actual}")
        paths[key] = path
    return paths


RELEASE_TAGS = ["PG_v1.0", "PG_v1.1", "PG_v1.2", "PG_v1.3"]


def download_release_summaries(data_dir: Path = DEFAULT_DATA_DIR) -> dict[str, pd.DataFrame]:
    """Published summary tables at each release tag plus the pinned commit, oldest first."""
    out_dir = data_dir / "versions"
    out_dir.mkdir(parents=True, exist_ok=True)
    remote = FILES["summary"][0]
    out = {}
    for ref in [*RELEASE_TAGS, PROTEINGYM_COMMIT]:
        label = ref if ref != PROTEINGYM_COMMIT else f"pinned@{ref[:7]}"
        path = out_dir / f"summary_{label.replace('@', '_')}.csv"
        if not path.exists():
            base = f"https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/{ref}"
            urllib.request.urlretrieve(f"{base}/{remote}", path)
        out[label] = pd.read_csv(path).set_index("Model_name")
    return out


def path_for(key: str, data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    path = data_dir / Path(FILES[key][0]).name
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `uv run pgnoise download` first")
    return path


@dataclass(frozen=True)
class AssayTable:
    """Per-assay (DMS-level) Spearman scores plus the metadata ProteinGym aggregates over."""

    scores: pd.DataFrame  # index: DMS ID, columns: model names, NaN where a model was not scored
    meta: pd.DataFrame  # index: DMS ID; columns: uniprot, function, taxon, msa_depth, n_mutants

    @property
    def models(self) -> list[str]:
        return list(self.scores.columns)


def load_assays(data_dir: Path = DEFAULT_DATA_DIR, metric: str = "Spearman") -> AssayTable:
    df = pd.read_csv(path_for(_metric_key("dms_level", metric), data_dir))
    ref = pd.read_csv(path_for("reference", data_dir)).set_index("DMS_id")
    df = df.set_index("DMS ID")
    models = [c for c in df.columns if c not in DMS_META_COLUMNS]
    meta = pd.DataFrame(
        {
            "uniprot": df["UniProt ID"],
            "function": df["Selection Type"],
            "taxon": df["Taxon"],
            "msa_depth": df["MSA_Neff_L_category"].str.capitalize(),
            "n_mutants": df["Number of Mutants"].astype(int),
        }
    )
    coarse = ref.loc[meta.index, "coarse_selection_type"]
    if not (coarse.values == meta["function"].values).all():
        raise ValueError("Selection Type in DMS-level file disagrees with reference coarse_selection_type")
    return AssayTable(scores=df[models].astype(float), meta=meta)


def load_summary(data_dir: Path = DEFAULT_DATA_DIR, metric: str = "Spearman") -> pd.DataFrame:
    return pd.read_csv(path_for(_metric_key("summary", metric), data_dir)).set_index("Model_name")


@dataclass(frozen=True)
class UnitTable:
    """Scores aggregated to ProteinGym's bootstrap unit: one row per (UniProt ID, function group).

    ``X`` is (n_units, n_models) with NaN where a model has no scored assay in the unit;
    ``groups`` holds the integer function-group index of each unit.
    """

    X: np.ndarray
    groups: np.ndarray
    group_names: list[str]
    models: list[str]
    units: list[tuple[str, str]]

    @property
    def n_groups(self) -> int:
        return len(self.group_names)

    def subset_models(self, models: list[str]) -> UnitTable:
        idx = [self.models.index(m) for m in models]
        return UnitTable(self.X[:, idx], self.groups, self.group_names, list(models), self.units)

    def drop_groups(self, drop: list[str]) -> UnitTable:
        keep_names = [g for g in self.group_names if g not in drop]
        keep_rows = np.isin(np.array(self.group_names)[self.groups], keep_names)
        remap = {self.group_names.index(g): i for i, g in enumerate(keep_names)}
        groups = np.array([remap[g] for g in self.groups[keep_rows]])
        units = [u for u, k in zip(self.units, keep_rows) if k]
        return UnitTable(self.X[keep_rows], groups, keep_names, self.models, units)


def to_units(assays: AssayTable) -> UnitTable:
    """Average assays within (UniProt, function) units, skipping NaNs as pandas' groupby-mean does."""
    frame = assays.scores.join(assays.meta[["uniprot", "function"]])
    unit_means = frame.groupby(["uniprot", "function"]).mean(numeric_only=True)
    group_names = sorted(unit_means.index.get_level_values("function").unique())
    groups = np.array([group_names.index(f) for f in unit_means.index.get_level_values("function")])
    return UnitTable(
        X=unit_means[assays.models].to_numpy(dtype=float),
        groups=groups,
        group_names=group_names,
        models=assays.models,
        units=list(unit_means.index),
    )
