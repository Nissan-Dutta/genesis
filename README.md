# pgnoise: how much of ProteinGym's leaderboard is noise?

[ProteinGym](https://github.com/OATML-Markslab/ProteinGym) ranks 97 zero-shot protein fitness models on 217 deep
mutational scanning assays, but it reports no uncertainty for the ranks themselves. `pgnoise` reproduces the
published leaderboard exactly and puts statistically valid 95% confidence intervals on every model's rank: marginal
and simultaneous pairwise max-t intervals with a stratified bootstrap over proteins, checked by a coverage
simulation with known true ranks. It then asks how many assays the benchmark would need to detect realistic gains,
and how far the top ranks move with the metric, the aggregation and assay coverage. Three models cannot be ruled out
as #1. Only 2 of the 19 adjacent pairs in the top 20 are statistically distinguishable. Detecting a 0.01 Spearman
gain over the current leader with 80% power would take about 850 assays, 3.9× today's benchmark. Everything runs on
a laptop from about 0.7 MB of pinned public CSVs.

<p align="center"><img src="results/figures/headline.png" width="420"
alt="Top-20 rank intervals with the possible-#1 set highlighted, and assays needed for 80% power versus true gain"></p>

**(a)** 95% intervals for the rank of each top-20 model's benchmark aggregate (thick: marginal; thin: simultaneous).
Orange models cannot be ruled out as #1; dashed lines mark the only two significant gaps between neighbours.
**(b)** Assays needed to detect a true gain Δ over the current #1 with 80% power; today's 217 assays detect about
0.020.

The two-page technical note is in [`paper/technical-note.pdf`](paper/technical-note.pdf), with a Markdown version
in [`paper/technical-note.md`](paper/technical-note.md).

## Key results

All numbers come from `results/summary.json`, using ProteinGym commit `144fe22` with 10,000 bootstrap replicates.

- **Reproduction.** All 97 published averages and ranks match exactly, and so do the AUC, MCC, NDCG and top-K
  recall leaderboards. 91 of 97 published error bars match at 3 dp; the other 6 lie within 0.0002 of a rounding
  boundary and change with the bootstrap seed.
- **#1 is a coin flip.** AIDO Protein-RAG and VenusREM differ by 0.00004, and each is #1 in about half of the
  bootstrap replicates. The 95% set of possible #1s is {AIDO Protein-RAG, VenusREM, ProSST (K=4096)}.
- **Ranks are wide.** #4 has a marginal 95% rank interval of 1–13 and #10 of 6–28 (simultaneous: 1–18 and 6–42).
- **Power.** Today's benchmark detects a gain of about 0.020 over #1 (0.015–0.027 across the next ten models).
  A gain of 0.01 needs about 850 assays (3.9× today), and a gain of 0.005 about 3,380 (15.6×).
- **Robustness.** Spearman, AUC and MCC agree on the possible-#1 set under four mean-based aggregations. NDCG puts
  S3F-MSA first, and only 4 of the published top 10 stay in its top 10.
- **Coverage.** Protriever is scored on 200 of 217 assays. The 17 it skips are harder for everyone else, and on the
  common assays it falls from #8 to #10.
- **Validity.** In 1,000 simulated leaderboards per scenario, the headline intervals keep worst-model coverage
  at 0.973 or above, including under exact ties. Naive bootstrap-percentile rank intervals fall to 0.081.
- **Replication.** Re-scoring five assays with ESM-2 (8M–650M) on CPU reproduces all 20 published Spearman values
  at 3 dp.

## Quickstart

```bash
# needs uv (https://docs.astral.sh/uv/); Python 3.12 and every pinned dependency come from uv.lock
uv sync
uv run pgnoise download   # ~0.7 MB of CSVs into data/raw/ (git-ignored); SHA-256 checksums verified
uv run pgnoise ranks      # rank intervals, best-model set, neighbour tests, Protriever coverage (~15 s)
uv run pgnoise figure     # the headline figure, from the tables in results/
uv run pytest             # unit and integration tests
```

`uv run pgnoise all` runs every analysis except the ESM-2 replication (about 25 minutes on 4 cores, mostly the
simulation). Without uv, `pip install -r requirements.txt && pip install -e .` works too; the requirements file is
exported from the lock file.

## Reproducing each result

Each command writes tables to `results/tables/`, figures to `results/figures/`, and its headline numbers to a
section of `results/summary.json`. `reproduce`, `ranks`, `robustness`, `power`, `simulate` and `esm2` read the files
fetched by `pgnoise download`; `figure` and `numbers` need only the committed `results/`. Runtimes are for 4 cores.

| Result | Command | Main outputs | Time |
|---|---|---|---|
| Published leaderboard and error bars, recomputed | `pgnoise reproduce` | `reproduction_full.csv`, `reproduction_report.csv`, `reproduction.png` | 2 s |
| Score drift between ProteinGym releases | `pgnoise versions` | `version_drift.csv` | 2 s |
| Rank intervals, possible #1s, neighbour tests, minimum detectable difference, leave-one-group-out, Protriever coverage | `pgnoise ranks` | `rank_intervals.csv`, `neighbour_tests_top20.csv`, `top_gap_power.csv`, `leave_one_group_out*.csv`, `common_assay_leaderboard.csv`, `protriever_missing_assays.csv`, `rank_intervals_top40.png`, `pairwise_top20.png`, `leave_one_group_out.png` | 15 s |
| 5 metrics × 5 aggregation schemes | `pgnoise robustness` | `robustness_long.csv`, `robustness_summary.csv`, `reproduction_other_metrics.csv`, `robustness_rank_heatmap.png`, `robustness_grid.png` | 40 s |
| Power curves and benchmark size needed | `pgnoise power` | `power_per_pair.csv`, `power_curves.csv`, `power_required_units.csv`, `power_curves.png` | 5 s |
| Coverage simulation (5 scenarios × 1,000 leaderboards) | `pgnoise simulate` | `sim_summary.csv`, `sim_best_set.csv`, `sim_summary.png`, `sim_coverage_by_rank.png` | 25 min |
| ESM-2 replication on 5 assays | `uv sync --extra esm`, then `pgnoise esm2` | `esm2_replication.csv`, `esm2_size_differences.csv`, `esm2_scores/`, `esm2_replication.png` | 5 min |
| Headline figure | `pgnoise figure` | `headline.png`, `headline.pdf` | 2 s |
| Numbers in the technical note | `pgnoise numbers` | `paper/numbers.tex`, `paper/numbers.md` | < 1 s |
| Technical note PDF | `make -C paper` (needs pdflatex and bibtex) | `paper/technical-note.pdf` | 5 s |

To try the simulation quickly, use `uv run pgnoise simulate --reps 50`. Seeds are fixed (re-running `pgnoise ranks`,
for example, reproduces its committed tables exactly), and the figure and note PDFs are byte-reproducible.

The ESM-2 extra installs CPU-only `torch==2.8.0` and `fair-esm==2.0.0`, then downloads about 3.4 GB of pinned
inputs into `data/esm2/`. Without uv, use `pip install -e ".[esm]" --extra-index-url https://download.pytorch.org/whl/cpu`.

**Number provenance.** `pgnoise numbers` turns `results/summary.json` into LaTeX macros (`paper/numbers.tex`) and a
table that maps each macro to its JSON source (`paper/numbers.md`). The LaTeX note uses only these macros for
results. `tests/test_paper.py` checks that both files are current and that the Markdown note quotes every value the
LaTeX note uses.

## Data and version

- **Source.** `OATML-Markslab/ProteinGym`, commit `144fe22` (2026-03-25). The per-assay file at this commit is
  byte-identical to the June 2025 "Added Protriever" commit, so the data are **release v1.3 plus two later
  additions, AIDO Protein-RAG and Protriever**: 97 models.
- **Files used.** The per-assay scores `DMS_substitutions_{Spearman,AUC,MCC,NDCG,Top_recall}_DMS_level.csv`, rounded
  to 3 dp as ProteinGym itself aggregates them, and `reference_files/DMS_substitutions.csv` for each assay's UniProt
  ID and `coarse_selection_type`. The published `Summary_performance_...csv` files are the reproduction targets.
- **ESM-2 inputs.** The ProteinGym v1.3 assay archive and ProteinGym's own per-mutant scores, from
  `marks.hms.harvard.edu/proteingym`, and the fair-esm checkpoints. All are SHA-256 pinned in `esm2.py`.
- **Version drift** (`results/tables/version_drift.csv`; see
  [ProteinGym issue #99](https://github.com/OATML-Markslab/ProteinGym/issues/99)). From v1.0 to v1.1, published
  averages moved by up to 0.011 (ESM-1v), and by 0.158 for Wavenet. Since v1.2, scores of existing models have not
  changed.

## Methods in brief

**Aggregation.** This replicates `proteingym/performance_DMS_benchmarks.py`. Per-assay scores are averaged within
200 (UniProt ID, function) units, then within 5 function groups, and the headline score is the mean of the 5 group
means. Missing scores are skipped at every level.

**Estimand.** Each interval targets the rank of a model's *benchmark aggregate*: its ProteinGym score in the limit
of infinitely many proteins per function group. This is not a prediction interval for a model's rank on a new assay,
which is the target of Neuhof & Benjamini (2026, arXiv:2606.08679).

**Uncertainty.** All methods share ProteinGym's own bootstrap: units are resampled with replacement within function
groups.

| Method | Guarantee |
|---|---|
| Bootstrap percentile of ranks | none; fails near ties (Hall & Miller 2009) |
| Pairwise max-t, marginal (Mogstad, Romano, Shaikh & Wilhelm 2024) | each model's interval covers its rank with probability ≥ 95% |
| Pairwise max-t, simultaneous | all intervals cover all ranks jointly with probability ≥ 95% |
| Step-down variants (Romano–Wolf) | same guarantees and tighter, but under-cover with exact ties in small strata |
| Bootstrap-t step-down (`studentized.py`) | as above, with each bootstrap replicate studentised by its own linearised SE |
| Best-model set (one-sided max-t) | contains the true #1 with probability ≥ 95% |
| "Within 1.96 SE of #1" (the usual reading of ProteinGym's error bars) | none |

The headline intervals are marginal single-step and simultaneous step-down.

**Power.** A new model is assumed to beat #1 by a true Δ, with paired per-unit differences as variable as those
between #1 and each of the next 10 complete-coverage models, and new assays are assumed to arrive in today's
function mix. Required units = `n0 × ((z_0.975 + z_0.8) × SE0 / Δ)²`. The formula is checked by resampling.

**Simulation.** `X[u, m] = mu[g(u), m] + a[u] + eps[u, m]` on the real unit and group structure. The residuals are
either resampled real rows or Gaussian with the empirical covariance. The scenarios are calibrated, Gaussian, exact
ties, near ties and well separated. A tied model's interval counts as covering only if it contains every rank the
model can take.

**Aggregation schemes** (`aggregation.py`). The schemes are ProteinGym's, the function-group mean, the
UniProt-weighted mean, the flat mean over assays and the median. All use the same bootstrap, so only the estimand
changes. The "Average" row of ProteinGym's `Uniprot_level.csv` is not an equal-per-protein mean: it is reproduced
exactly only by replicating the script's non-deduplicated merge. The leaderboard itself does not use it.

**ESM-2 replication** (`esm2.py`). The five shortest assays, one per function group with at least 170 mutants, are
scored with ProteinGym's `masked-marginals` strategy: each residue is masked in turn, and multi-mutants are scored
additively. Sequences longer than 1,022 residues would use ProteinGym's `get_optimal_window`. Batching is the only
change from ProteinGym's loop, and a test checks it against a verbatim port of that loop. The full write-up is in
the technical note.

## Layout

```
src/pgnoise/
  data.py        download (pinned commit + checksums), loading, (UniProt, function) units
  stats.py       ProteinGym aggregation, stratified bootstrap, ProteinGym's gap-to-#1 SE
  ranks.py       rank intervals, best-model set, neighbour tests, Holm, minimum detectable difference
  studentized.py bootstrap-t step-down rank intervals
  aggregation.py alternative aggregation schemes under the same bootstrap
  robustness.py  metric x scheme grid
  power.py       power curves and required benchmark size
  analysis.py    real-data analyses
  simulate.py    calibrated generator, scenarios, coverage experiment
  esm2.py        pinned ESM-2 inputs, masked and wild-type marginals, comparison with the leaderboard
  paper.py       numbers for the technical note, derived from results/summary.json
  plots.py, cli.py
paper/           technical note (LaTeX + Markdown), bibliography, generated numbers, Makefile
results/         summary.json, tables/, figures/
tests/           unit tests on toy data, plus integration tests on the real files
```

## Data licence and attribution

This repository contains no raw ProteinGym data. `pgnoise download` fetches it from its source, into git-ignored
`data/`. The committed `results/` contain values derived from ProteinGym, and the per-mutant files under
`results/tables/esm2_scores/` include ProteinGym's own model scores and DMS measurements for five assays.

- **ProteinGym** is released under the MIT licence: Copyright (c) 2023 OATML-Markslab, Pascal Notin, Aaron Kollasch,
  Daniel Ritter, Lood van Niekerk. See the
  [ProteinGym LICENSE](https://github.com/OATML-Markslab/ProteinGym/blob/main/LICENSE). If you use these results,
  please cite ProteinGym:

  > P. Notin, A. Kollasch, D. Ritter, L. van Niekerk, S. Paul, H. Spinner, N. Rollins, A. Shaw, R. Orenbuch,
  > R. Weitzman, J. Frazer, M. Dias, D. Franceschi, Y. Gal and D. Marks. ProteinGym: Large-Scale Benchmarks for
  > Protein Fitness Prediction and Design. *Advances in Neural Information Processing Systems 36*, 64331–64379, 2023.

- **The DMS assays** were produced by the original studies listed in ProteinGym's
  `reference_files/DMS_substitutions.csv`. Please cite them when using assay-level results; for example, the five
  ESM-2 replication assays are by Tsuboyama et al. 2023, Ghose et al. 2023, Kelsic et al. 2016, Elazar et al. 2016
  and Dutta et al. 2010.
- **ESM-2** weights and the `fair-esm` package come from
  [facebookresearch/esm](https://github.com/facebookresearch/esm), released under the MIT licence (Lin et al.,
  *Science* 2023).
