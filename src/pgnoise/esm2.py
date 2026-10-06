"""Zero-shot ESM-2 scoring of ProteinGym DMS assays, replicating ProteinGym's own pipeline on CPU.

ProteinGym scores ESM-2 with ``proteingym/baselines/esm/compute_fitness.py --scoring-strategy
masked-marginals`` (``scripts/scoring_DMS_zero_shot/scoring_ESM2_substitutions.sh``): for every
token position i of the wild type it masks i alone, runs the model, and keeps log p(. | x_{-i}).
A mutant's score is the sum over its mutations of log p(mt) - log p(wt) at the masked position,
so multi-mutants are scored additively. Sequences longer than 1022 residues are cut to a
1024-token window around each masked position (``get_optimal_window``). The per-assay metric is
Spearman between that score and ``DMS_score`` over all mutants in the assay file.
"""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import time
import urllib.request
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path

import esm
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr

from . import data

PROTEINGYM_RELEASE = "v1.3"
_MARKS = f"https://marks.hms.harvard.edu/proteingym/ProteinGym_{PROTEINGYM_RELEASE}"
DMS_ZIP_URL = f"{_MARKS}/DMS_ProteinGym_substitutions.zip"
DMS_ZIP_SHA256 = "3a83766254ac9ac9984ec25cb73c6e010ea4418f5e35f143933e6b6e6473b921"
# 1.9 GB archive of every model's per-mutant scores; only the selected assays are fetched, by HTTP range requests.
SCORES_ZIP_URL = f"{_MARKS}/zero_shot_substitutions_scores.zip"
WEIGHTS_URL = "https://dl.fbaipublicfiles.com/fair-esm/models/{name}.pt"


@dataclass(frozen=True)
class AssaySpec:
    dms_sha256: str  # DMS_ProteinGym_substitutions/<id>.csv
    scores_sha256: str  # zero_shot_substitutions_scores.zip:<id>.csv (ProteinGym's own per-mutant model scores)


# The shortest assay in each function group, plus enough mutants for a stable Spearman (>= 170).
ASSAYS: dict[str, AssaySpec] = {
    "TCRG1_MOUSE_Tsuboyama_2023_1E0L": AssaySpec(
        "2f98ed1f2e1e896cd8f4022e5c27aef4df2617ab4639ef52248d6774f5cc0739",
        "11c9fbc4228346504e0e092777be46ad41e57d641511989be62dae962063bcca"),
    "ENVZ_ECOLI_Ghose_2023": AssaySpec(
        "99cea4f638ad1b5c069e7b0c4385e7feb362c65322faf840c3c29ccebe665acd",
        "25192ea8758897b603096f0ecf35c8806db533d582316c8584103c7a8ea44682"),
    "IF1_ECOLI_Kelsic_2016": AssaySpec(
        "5faadfd056c0ba9b17e971848ed47ac392ac4a0374c3550cdd0292d808076e06",
        "839d69df598a5a22970f39ed352f422b2751668e1e86bd08c39366c4a6e7c4ab"),
    "GLPA_HUMAN_Elazar_2016": AssaySpec(
        "ed4c4cd92f86da4a3e4e0e15eabcdb227e3cfbbc6fe0dce4709f1db0ff933912",
        "9f037bb34e300abef127af3c9eb11bf04279290cf14d7f571d4beef50694568a"),
    "B2L11_HUMAN_Dutta_2010_binding-Mcl-1": AssaySpec(
        "0768cc07f06ccfa28e20cfd2b1b0ffc60fbbb7d21174b5d2b3b780930c4b80be",
        "e9c5f7a4754c3171668c49a04520d5f463fd9556e916e3f84ed9b7c12bf26f58"),
}


@dataclass(frozen=True)
class ModelSpec:
    checkpoint: str
    sha256: str
    leaderboard_name: str  # column in DMS_substitutions_Spearman_DMS_level.csv
    scores_column: str  # column in the per-mutant score files


MODELS: dict[str, ModelSpec] = {
    "8M": ModelSpec("esm2_t6_8M_UR50D", "46f002a9870c9bdecd0ea887acb1f9a38a6b561e8f8bf8a6990b679b9d31b928",
                    "ESM2 (8M)", "ESM2_8M"),
    "35M": ModelSpec("esm2_t12_35M_UR50D", "7f21e80e61d16a71735163ef555d3009afb0c98da74c48e29df08606973cc55e",
                     "ESM2 (35M)", "ESM2_35M"),
    "150M": ModelSpec("esm2_t30_150M_UR50D", "881c7176cf198ef8dec26a3c375d40eb58d0c33df95c22562ca6cc6d3f812c62",
                      "ESM2 (150M)", "ESM2_150M"),
    "650M": ModelSpec("esm2_t33_650M_UR50D", "ea9d0522b335a8778dea6535a65301f10208dece28cd5865482b0b1fc446168c",
                      "ESM2 (650M)", "ESM2_650M"),
}

MODEL_WINDOW = 1024


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check(path: Path, expected: str) -> Path:
    actual = _sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"Checksum mismatch for {path}: expected {expected}, got {actual}")
    return path


def _fetch(url: str, path: Path, expected: str) -> Path:
    if path.exists() and _sha256_file(path) == expected:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with urllib.request.urlopen(url) as r, part.open("wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    part.replace(path)
    return _check(path, expected)


class _HTTPRangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file, so ``zipfile`` can pull single members by range request."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            self.size = int(r.headers["Content-Length"])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self.pos = {io.SEEK_SET: offset, io.SEEK_CUR: self.pos + offset, io.SEEK_END: self.size + offset}[whence]
        return self.pos

    def readinto(self, buf) -> int:
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(buf), self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req) as r:
            chunk = r.read()
        buf[: len(chunk)] = chunk
        self.pos += len(chunk)
        return len(chunk)


def esm2_dir(data_dir: Path) -> Path:
    return data_dir.parent / "esm2"


def assay_path(dms_id: str, data_dir: Path = data.DEFAULT_DATA_DIR) -> Path:
    return esm2_dir(data_dir) / "assays" / f"{dms_id}.csv"


def reference_scores_path(dms_id: str, data_dir: Path = data.DEFAULT_DATA_DIR) -> Path:
    return esm2_dir(data_dir) / "proteingym_scores" / f"{dms_id}.csv"


def weights_path(size: str, data_dir: Path = data.DEFAULT_DATA_DIR) -> Path:
    return esm2_dir(data_dir) / "weights" / f"{MODELS[size].checkpoint}.pt"


def download(data_dir: Path = data.DEFAULT_DATA_DIR, assays: list[str] | None = None,
             sizes: list[str] | None = None) -> None:
    """Fetch the pinned assay CSVs, ProteinGym's per-mutant scores for them, and ESM-2 weights; verify all."""
    assays, sizes = assays or list(ASSAYS), sizes or list(MODELS)
    unknown = sorted(set(assays) - set(ASSAYS))
    if unknown:
        raise ValueError(f"no pinned checksums for {unknown}; add them to esm2.ASSAYS first")
    root = esm2_dir(data_dir)
    dms_zip = _fetch(DMS_ZIP_URL, root / "DMS_ProteinGym_substitutions.zip", DMS_ZIP_SHA256)
    with zipfile.ZipFile(dms_zip) as z:
        for dms_id in assays:
            out = assay_path(dms_id, data_dir)
            if not (out.exists() and _sha256_file(out) == ASSAYS[dms_id].dms_sha256):
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(z.read(f"DMS_ProteinGym_substitutions/{dms_id}.csv"))
            _check(out, ASSAYS[dms_id].dms_sha256)
    todo = [a for a in assays if not (reference_scores_path(a, data_dir).exists()
                                      and _sha256_file(reference_scores_path(a, data_dir)) == ASSAYS[a].scores_sha256)]
    if todo:
        with zipfile.ZipFile(io.BufferedReader(_HTTPRangeFile(SCORES_ZIP_URL), buffer_size=1 << 16)) as z:
            for dms_id in todo:
                out = reference_scores_path(dms_id, data_dir)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(z.read(f"{dms_id}.csv"))
    for dms_id in assays:
        _check(reference_scores_path(dms_id, data_dir), ASSAYS[dms_id].scores_sha256)
    for size in sizes:
        _fetch(WEIGHTS_URL.format(name=MODELS[size].checkpoint), weights_path(size, data_dir), MODELS[size].sha256)


@dataclass(frozen=True)
class Assay:
    dms_id: str
    sequence: str
    function: str
    frame: pd.DataFrame  # mutant, DMS_score, n_mut


def load_assay(dms_id: str, data_dir: Path = data.DEFAULT_DATA_DIR) -> Assay:
    ref = pd.read_csv(data.path_for("reference", data_dir)).set_index("DMS_id").loc[dms_id]
    frame = pd.read_csv(assay_path(dms_id, data_dir))[["mutant", "DMS_score"]]
    frame["n_mut"] = frame["mutant"].str.count(":") + 1
    return Assay(dms_id, ref["target_seq"].upper(), ref["coarse_selection_type"], frame)


def load_model(size: str, data_dir: Path = data.DEFAULT_DATA_DIR) -> tuple[torch.nn.Module, esm.Alphabet]:
    spec = MODELS[size]
    # weights_only=False because fair-esm checkpoints pickle their argparse config; the file is SHA-256 pinned.
    # mmap keeps peak memory near one copy of the weights (650M is 2.6 GB).
    model_data = torch.load(weights_path(size, data_dir), map_location="cpu", weights_only=False, mmap=True)
    with warnings.catch_warnings():
        # ProteinGym also loads without the contact-regression head, which masked marginals never use.
        warnings.filterwarnings("ignore", message="Regression weights not found")
        model, alphabet = esm.pretrained.load_model_and_alphabet_core(spec.checkpoint, model_data, None)
    return model.eval(), alphabet


def optimal_window(position: int, seq_len_with_special: int, model_window: int = MODEL_WINDOW) -> tuple[int, int]:
    """ProteinGym's ``get_optimal_window``: token slice [start, end) scored when position ``position`` is masked."""
    half = model_window // 2
    if seq_len_with_special <= model_window:
        return 0, seq_len_with_special
    if position < half:
        return 0, model_window
    if position >= seq_len_with_special - half:
        return seq_len_with_special - model_window, seq_len_with_special
    return max(0, position - half), min(seq_len_with_special, position + half)


def _tokens(alphabet: esm.Alphabet, sequence: str) -> torch.Tensor:
    _, _, tokens = alphabet.get_batch_converter()([("wt", sequence)])
    return tokens[0]


@torch.no_grad()
def masked_marginals(model: torch.nn.Module, alphabet: esm.Alphabet, sequence: str,
                     batch_size: int = 16, model_window: int = MODEL_WINDOW) -> np.ndarray:
    """(L, vocab) log-probabilities at each residue with that residue alone masked.

    Same quantity as ProteinGym's loop, which runs one masked copy at a time; here equal-length masked
    copies are batched. BOS/EOS positions are skipped because no mutant reads them.
    """
    tokens = _tokens(alphabet, sequence)
    n_tok = tokens.numel()
    out = np.empty((len(sequence), len(alphabet)), dtype=np.float64)
    by_length: dict[int, list[tuple[int, int, int]]] = {}
    for i in range(1, n_tok - 1):
        start, end = optimal_window(i, n_tok, model_window)
        by_length.setdefault(end - start, []).append((i, start, end))
    for group in by_length.values():
        for b in range(0, len(group), batch_size):
            chunk = group[b:b + batch_size]
            windows = []
            for i, start, end in chunk:
                masked = tokens.clone()
                masked[i] = alphabet.mask_idx
                windows.append(masked[start:end])
            logp = torch.log_softmax(model(torch.stack(windows))["logits"], dim=-1)
            for k, (i, start, _) in enumerate(chunk):
                out[i - 1] = logp[k, i - start].double().numpy()
    return out


@torch.no_grad()
def wt_marginals(model: torch.nn.Module, alphabet: esm.Alphabet, sequence: str) -> np.ndarray:
    """(L, vocab) log-probabilities from one unmasked pass over the wild type (ESM-1v's cheaper alternative)."""
    tokens = _tokens(alphabet, sequence)
    logp = torch.log_softmax(model(tokens[None])["logits"], dim=-1)[0]
    return logp[1:-1].double().numpy()


def score_mutants(mutants: pd.Series, sequence: str, logp: np.ndarray, alphabet: esm.Alphabet,
                  offset: int = 1) -> np.ndarray:
    """ProteinGym's ``label_row``: sum of log p(mt) - log p(wt) over ``:``-separated mutations."""
    idx = {aa: alphabet.get_idx(aa) for aa in alphabet.all_toks}
    scores = np.empty(len(mutants))
    for n, mutant in enumerate(mutants):
        total = 0.0
        for m in mutant.split(":"):
            wt, pos, mt = m[0], int(m[1:-1]) - offset, m[-1]
            if sequence[pos] != wt:
                raise ValueError(f"{m}: wild type {wt} does not match {sequence[pos]} at position {pos + offset}")
            total += logp[pos, idx[mt]] - logp[pos, idx[wt]]
        scores[n] = total
    return scores


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(spearmanr(x, y)[0])


def _bootstrap_spearmans(dms: np.ndarray, scores: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """(n_boot, k) Spearman of ``dms`` with each column of ``scores`` (n, k) on mutant resamples ``idx``."""
    rx = rankdata(dms[idx], axis=1)
    rx -= rx.mean(1, keepdims=True)
    out = np.empty((idx.shape[0], scores.shape[1]))
    for k in range(scores.shape[1]):
        ry = rankdata(scores[idx, k], axis=1)
        ry -= ry.mean(1, keepdims=True)
        out[:, k] = (rx * ry).sum(1) / np.sqrt((rx ** 2).sum(1) * (ry ** 2).sum(1))
    return out


def bootstrap_spearman_se(dms: np.ndarray, score: np.ndarray, n_boot: int, rng: np.random.Generator) -> float:
    """SD of Spearman over resampled mutants: the assay's own sampling noise, for scale."""
    idx = rng.integers(0, len(dms), (n_boot, len(dms)))
    return float(_bootstrap_spearmans(dms, score[:, None], idx)[:, 0].std(ddof=1))


def size_differences(per_mutant: dict[str, pd.DataFrame], sizes: list[str], n_boot: int = 2000,
                     seed: int = 0) -> pd.DataFrame:
    """Spearman gain of each size over the next smaller one, and 8M -> largest, with a paired mutant bootstrap SE."""
    pairs = list(zip(sizes, sizes[1:]))
    if len(sizes) > 2:
        pairs.append((sizes[0], sizes[-1]))
    rows = []
    for k, (dms_id, frame) in enumerate(per_mutant.items()):
        dms = frame["DMS_score"].to_numpy()
        scores = frame[[f"ours_mm_{s}" for s in sizes]].to_numpy()
        point = np.array([spearman(dms, scores[:, j]) for j in range(len(sizes))])
        idx = np.random.default_rng([seed, k]).integers(0, len(dms), (n_boot, len(dms)))
        boot = _bootstrap_spearmans(dms, scores, idx)
        for small, large in pairs:
            i, j = sizes.index(small), sizes.index(large)
            diff, se = point[j] - point[i], float((boot[:, j] - boot[:, i]).std(ddof=1))
            rows.append({"assay": dms_id, "smaller": small, "larger": large, "gain": diff, "paired_se": se,
                         "z": diff / se})
    return pd.DataFrame(rows)


def replicate(data_dir: Path = data.DEFAULT_DATA_DIR, assays: list[str] | None = None,
              sizes: list[str] | None = None, batch_size: int = 16, threads: int | None = None,
              n_boot: int = 2000, seed: int = 0, log=print) -> dict[str, pd.DataFrame]:
    """Score each assay with each ESM-2 size and compare with ProteinGym's per-mutant scores and leaderboard."""
    assays, sizes = assays or list(ASSAYS), sizes or list(MODELS)
    torch.set_num_threads(threads or os.cpu_count() or 1)
    leaderboard = pd.read_csv(data.path_for("dms_level", data_dir)).set_index("DMS ID")
    loaded = {a: load_assay(a, data_dir) for a in assays}
    per_mutant = {a: loaded[a].frame.copy() for a in assays}
    rows = []
    for size in sizes:
        spec = MODELS[size]
        t0 = time.perf_counter()
        model, alphabet = load_model(size, data_dir)
        load_s = time.perf_counter() - t0
        for a in assays:
            assay, frame = loaded[a], per_mutant[a]
            reference = pd.read_csv(reference_scores_path(a, data_dir)).set_index("mutant")
            t0 = time.perf_counter()
            mm = masked_marginals(model, alphabet, assay.sequence, batch_size)
            mm_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            wt = wt_marginals(model, alphabet, assay.sequence)
            wt_s = time.perf_counter() - t0
            frame[f"ours_mm_{size}"] = score_mutants(frame["mutant"], assay.sequence, mm, alphabet)
            frame[f"ours_wt_{size}"] = score_mutants(frame["mutant"], assay.sequence, wt, alphabet)
            frame[f"proteingym_{size}"] = reference.loc[frame["mutant"], spec.scores_column].to_numpy()
            if not np.allclose(reference.loc[frame["mutant"], "DMS_score"].to_numpy(), frame["DMS_score"]):
                raise ValueError(f"{a}: DMS_score differs between the assay file and ProteinGym's score file")
            dms = frame["DMS_score"].to_numpy()
            ours, pg, ours_wt = (frame[f"{c}_{size}"].to_numpy() for c in ("ours_mm", "proteingym", "ours_wt"))
            single = frame["n_mut"].eq(1).to_numpy()
            rng = np.random.default_rng([seed, sizes.index(size), assays.index(a)])
            row = {
                "assay": a, "function": assay.function, "seq_len": len(assay.sequence), "n_mutants": len(frame),
                "n_multi": int((~single).sum()), "size": size,
                "spearman_published": float(leaderboard.loc[a, spec.leaderboard_name]),
                "spearman_proteingym_scores": spearman(dms, pg),
                "spearman_ours": spearman(dms, ours),
                "spearman_ours_wt_marginals": spearman(dms, ours_wt),
                "spearman_ours_singles": spearman(dms[single], ours[single]),
                "spearman_proteingym_singles": spearman(dms[single], pg[single]),
                "per_mutant_pearson_vs_proteingym": float(np.corrcoef(ours, pg)[0, 1]),
                "per_mutant_max_abs_diff": float(np.max(np.abs(ours - pg))),
                "spearman_boot_se": bootstrap_spearman_se(dms, ours, n_boot, rng),
                "model_load_s": load_s, "masked_marginals_s": mm_s, "wt_marginals_s": wt_s,
            }
            row["diff_vs_published"] = round(round(row["spearman_ours"], 3) - row["spearman_published"], 3)
            rows.append(row)
            log(f"[{size:>4s}] {a:40s} L={row['seq_len']:3d} rho ours={row['spearman_ours']:.4f} "
                f"published={row['spearman_published']:.3f} wt-marg={row['spearman_ours_wt_marginals']:.3f} "
                f"max|d score|={row['per_mutant_max_abs_diff']:.1e} ({mm_s:.1f}s)")
        del model
    return {"table": pd.DataFrame(rows), "per_mutant": per_mutant}
