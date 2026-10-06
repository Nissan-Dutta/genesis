"""ESM-2 scoring: equivalence with ProteinGym's compute_fitness.py, plus checks on the pinned real data."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

torch = pytest.importorskip("torch", reason="ESM-2 tests need the `esm` extra (uv sync --extra esm)")
esm = pytest.importorskip("esm")

from pgnoise import data, esm2  # noqa: E402

AA = "ACDEFGHIKLMNPQRSTVWY"


def pg_get_optimal_window(mutation_position_relative, seq_len_wo_special, model_window):
    """Verbatim from proteingym/utils/scoring_utils.py at the pinned commit."""
    half_model_window = model_window // 2
    if seq_len_wo_special <= model_window:
        return [0, seq_len_wo_special]
    elif mutation_position_relative < half_model_window:
        return [0, model_window]
    elif mutation_position_relative >= seq_len_wo_special - half_model_window:
        return [seq_len_wo_special - model_window, seq_len_wo_special]
    else:
        return [max(0, mutation_position_relative - half_model_window),
                min(seq_len_wo_special, mutation_position_relative + half_model_window)]


def pg_masked_marginals(model, alphabet, sequence, model_window):
    """ProteinGym's masked-marginals loop (compute_fitness.py), one masked copy per forward pass, BOS/EOS included."""
    _, _, batch_tokens = alphabet.get_batch_converter()([("protein1", sequence)])
    all_token_probs = []
    for i in range(batch_tokens.size(1)):
        batch_tokens_masked = batch_tokens.clone()
        batch_tokens_masked[0, i] = alphabet.mask_idx
        if batch_tokens.size(1) > model_window:
            start, end = pg_get_optimal_window(i, len(sequence) + 2, model_window)
            batch_tokens_masked = batch_tokens_masked[:, start:end]
        else:
            start = 0
        with torch.no_grad():
            token_probs = torch.log_softmax(model(batch_tokens_masked)["logits"], dim=-1)
        all_token_probs.append(token_probs[:, i - start])
    return torch.cat(all_token_probs, dim=0).unsqueeze(0)


def pg_label_row(row, sequence, token_probs, alphabet, offset_idx):
    score = 0
    for mutation in row.split(":"):
        wt, idx, mt = mutation[0], int(mutation[1:-1]) - offset_idx, mutation[-1]
        assert sequence[idx] == wt
        wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)
        score += (token_probs[0, 1 + idx, mt_encoded] - token_probs[0, 1 + idx, wt_encoded]).item()
    return score


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
    model = esm.model.esm2.ESM2(num_layers=2, embed_dim=32, attention_heads=4, alphabet=alphabet, token_dropout=True)
    return model.eval(), alphabet


def random_mutants(sequence: str, n: int, rng: np.random.Generator, max_order: int = 3) -> list[str]:
    out = []
    for _ in range(n):
        positions = sorted(rng.choice(len(sequence), rng.integers(1, max_order + 1), replace=False))
        out.append(":".join(f"{sequence[p]}{p + 1}{rng.choice([a for a in AA if a != sequence[p]])}"
                            for p in positions))
    return out


@pytest.mark.parametrize("window", [1024, 16, 17])
def test_optimal_window_matches_proteingym(window):
    for n_tok in [10, 16, 17, 40, 1024, 1025, 3000]:
        for i in range(n_tok):
            assert esm2.optimal_window(i, n_tok, window) == tuple(pg_get_optimal_window(i, n_tok, window))


@pytest.mark.parametrize("length,window", [(30, 1024), (40, 16), (41, 17)])
def test_batched_masked_marginals_match_proteingym_loop(tiny_model, length, window):
    model, alphabet = tiny_model
    rng = np.random.default_rng(length)
    sequence = "".join(rng.choice(list(AA), length))
    mutants = random_mutants(sequence, 60, rng)
    ref = pg_masked_marginals(model, alphabet, sequence, window)
    expected = [pg_label_row(m, sequence, ref, alphabet, 1) for m in mutants]
    for batch_size in (1, 7):
        logp = esm2.masked_marginals(model, alphabet, sequence, batch_size=batch_size, model_window=window)
        np.testing.assert_allclose(esm2.score_mutants(pd.Series(mutants), sequence, logp, alphabet), expected,
                                   atol=1e-5)


def test_wt_marginals_is_one_unmasked_pass(tiny_model):
    model, alphabet = tiny_model
    sequence = "MKTAYIAKQRQISFVKSHFSRQ"
    _, _, tokens = alphabet.get_batch_converter()([("p", sequence)])
    with torch.no_grad():
        full = torch.log_softmax(model(tokens)["logits"], dim=-1)[0, 1:-1].numpy()
    np.testing.assert_allclose(esm2.wt_marginals(model, alphabet, sequence), full, atol=1e-6)


def test_score_mutants_is_additive_and_checks_wild_type(tiny_model):
    _, alphabet = tiny_model
    sequence = "ACDE"
    logp = np.random.default_rng(1).normal(size=(4, len(alphabet)))
    g = alphabet.get_idx
    single_a = logp[0, g("W")] - logp[0, g("A")]
    single_d = logp[2, g("K")] - logp[2, g("D")]
    got = esm2.score_mutants(pd.Series(["A1W", "D3K", "A1W:D3K"]), sequence, logp, alphabet)
    np.testing.assert_allclose(got, [single_a, single_d, single_a + single_d])
    np.testing.assert_allclose(esm2.score_mutants(pd.Series(["A11W"]), sequence, logp, alphabet, offset=11),
                               [single_a])
    with pytest.raises(ValueError, match="does not match"):
        esm2.score_mutants(pd.Series(["C1W"]), sequence, logp, alphabet)


def test_bootstrap_spearman_se_matches_scipy_loop():
    rng = np.random.default_rng(3)
    x = rng.normal(size=80)
    y = x + rng.normal(size=80)
    y[::7] = y[0]  # ties are averaged in both implementations
    se = esm2.bootstrap_spearman_se(x, y, 300, np.random.default_rng(9))
    idx = np.random.default_rng(9).integers(0, 80, (300, 80))
    ref = np.std([spearmanr(x[i], y[i])[0] for i in idx], ddof=1)
    assert se == pytest.approx(ref, abs=1e-10)


def test_size_differences_are_paired():
    rng = np.random.default_rng(4)
    dms = rng.normal(size=200)
    frame = pd.DataFrame({"DMS_score": dms, "ours_mm_8M": dms + rng.normal(0, 2, 200),
                          "ours_mm_35M": dms + rng.normal(0, 0.5, 200)})
    frame["ours_mm_150M"] = frame["ours_mm_35M"]
    out = esm2.size_differences({"toy": frame}, ["8M", "35M", "150M"], n_boot=300).set_index(["smaller", "larger"])
    assert list(out.index) == [("8M", "35M"), ("35M", "150M"), ("8M", "150M")]
    assert out.loc[("35M", "150M"), "gain"] == 0 and out.loc[("35M", "150M"), "paired_se"] == 0
    assert out.loc[("8M", "35M"), "z"] > 5
    assert out.loc[("8M", "35M"), "gain"] == pytest.approx(out.loc[("8M", "150M"), "gain"])


needs_reference = pytest.mark.skipif(
    not (data.DEFAULT_DATA_DIR / "DMS_substitutions.csv").exists()
    or not all(esm2.reference_scores_path(a).exists() for a in esm2.ASSAYS),
    reason="run `uv run pgnoise download` and `uv run pgnoise esm2 --download-only` first",
)


@needs_reference
def test_pinned_assays_cover_every_function_group():
    ref = pd.read_csv(data.path_for("reference")).set_index("DMS_id").loc[list(esm2.ASSAYS)]
    assert sorted(ref.coarse_selection_type) == sorted(data.FUNCTION_GROUPS)
    assert (ref.seq_len + 2 <= esm2.MODEL_WINDOW).all()


@needs_reference
def test_proteingym_per_mutant_scores_reproduce_published_spearman():
    leaderboard = pd.read_csv(data.path_for("dms_level")).set_index("DMS ID")
    for dms_id in esm2.ASSAYS:
        assay = esm2.load_assay(dms_id)
        scores = pd.read_csv(esm2.reference_scores_path(dms_id)).set_index("mutant").loc[assay.frame.mutant]
        np.testing.assert_allclose(scores.DMS_score, assay.frame.DMS_score)
        for spec in esm2.MODELS.values():
            rho = esm2.spearman(assay.frame.DMS_score.to_numpy(), scores[spec.scores_column].to_numpy())
            assert round(rho, 3) == leaderboard.loc[dms_id, spec.leaderboard_name]


@needs_reference
@pytest.mark.skipif(not esm2.weights_path("8M").exists(), reason="ESM-2 8M weights not downloaded")
def test_esm2_8m_reproduces_proteingym_on_multi_mutant_assay():
    res = esm2.replicate(assays=["TCRG1_MOUSE_Tsuboyama_2023_1E0L"], sizes=["8M"], n_boot=50, log=lambda _: None)
    row = res["table"].iloc[0]
    assert row.n_multi > 0
    assert round(row.spearman_ours, 3) == row.spearman_published
    assert row.per_mutant_max_abs_diff < 1e-3
