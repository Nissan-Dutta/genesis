# pgnoise: how much of ProteinGym's leaderboard is noise?

This repo is a reproducible rank-uncertainty analysis of the
[ProteinGym](https://github.com/OATML-Markslab/ProteinGym) zero-shot DMS-substitution leaderboard
(97 models, 217 assays, Spearman). It does five things:

1. Downloads ProteinGym's public per-assay Spearman table and assay metadata. Files are pinned to
   one commit and checked against SHA-256 checksums.
2. Recomputes the published leaderboard averages, function/MSA/taxon breakdowns and bootstrap
   error bars using ProteinGym's own aggregation scheme, and reports every mismatch.
3. Puts 95% confidence intervals on every model's **rank**: naive bootstrap-percentile, marginal
   and simultaneous pairwise max-t, each single-step and step-down. It also tests neighbouring
   pairs in the top 20, gives a confidence set for the #1 model, the minimum detectable
   difference at the top, a leave-one-function-group-out check of #1, and the effect of
   Protriever's incomplete assay coverage.
4. Runs a coverage simulation calibrated to the real data, with known true ranks. Scenarios
   include exact ties and near ties.
5. Writes tables and figures to `results/`.

## Quick start

```bash
# needs uv (https://docs.astral.sh/uv/); Python 3.12 and all pinned dependencies come from uv.lock
uv sync
uv run pgnoise download          # ~0.7 MB of CSVs into data/raw/ (git-ignored), checksums verified
uv run pgnoise reproduce         # published averages and error bars vs ours   (~2 s)
uv run pgnoise versions          # score drift between ProteinGym releases     (~2 s)
uv run pgnoise ranks             # rank intervals and related analyses         (~15 s)
uv run pgnoise simulate          # coverage simulation, 5 scenarios x 1000 reps (~15-20 min on 4 cores)
uv run pytest                    # unit + integration tests
```

`uv run pgnoise all` runs everything in order. To try the simulation quickly, use
`uv run pgnoise simulate --reps 50`. If you don't use uv, `pip install -r requirements.txt && pip install -e .`
works too. The requirements file is exported from the lock file.

## Data and version

- Source: `OATML-Markslab/ProteinGym`, commit `144fe22` (2026-03-25). The per-assay file at this
  commit is byte-identical to the June 2025 "Added Protriever" commit, so it is **release v1.3
  plus the two later additions, AIDO Protein-RAG and Protriever**. That gives 97 models.
- Files used: `DMS_substitutions_Spearman_DMS_level.csv` (per-assay Spearman, rounded to 3 dp, as
  ProteinGym itself aggregates) and `reference_files/DMS_substitutions.csv` (UniProt ID and
  `coarse_selection_type`). The published `Summary_performance_...csv` is the reproduction target.
- Version drift (`results/tables/version_drift.csv`, cf.
  [ProteinGym issue #99](https://github.com/OATML-Markslab/ProteinGym/issues/99)): from v1.0 to
  v1.1, published averages moved by up to 0.011 (ESM-1v) and by 0.158 for Wavenet. Since v1.2,
  scores of existing models have not changed.

## Methods

**ProteinGym's aggregation.** This replicates `proteingym/performance_DMS_benchmarks.py`. Per-assay
Spearman values are averaged within (UniProt ID, coarse selection type) units, which gives 200
units. The units are then averaged within each of 5 function groups (Activity, Binding,
Expression, OrganismalFitness, Stability), and the headline score is the mean of the 5 group
means. Missing scores are skipped at every level, as pandas does. The published error bar is the
SD over 10,000 bootstrap replicates of each model's unit-level gap to #1, resampling units with
replacement within each function group.

**Estimand.** Each interval targets the rank of a model's *benchmark aggregate*: ProteinGym's
score in the limit of infinitely many proteins per function group. Rank is
`1 + #{models with a strictly higher aggregate}`. This is **not** a prediction interval for a
model's rank on a new assay, which is the target of Neuhof & Benjamini (2026, arXiv 2606.08679).
Per-assay ranks vary far more than the rank of the aggregate.

**Uncertainty.** All methods use the same stratified bootstrap over units as ProteinGym's error
bars (10,000 replicates on real data, 1,000 inside the simulation).

| Method | Guarantee |
|---|---|
| Bootstrap percentile of ranks | none (known to fail near ties; Hall & Miller 2009) |
| Pairwise max-t, marginal (Mogstad, Romano, Shaikh & Wilhelm 2024) | each model's interval covers its rank w.p. ≥ 95% |
| Pairwise max-t, simultaneous | all intervals cover all ranks jointly w.p. ≥ 95% |
| Step-down variants | same guarantees, tighter (Romano-Wolf) |
| Best-model set (one-sided max-t vs each model) | contains the true #1 w.p. ≥ 95% |
| "Within 1.96 SE of #1" (how ProteinGym error bars are usually read) | none |

**Neighbour tests.** Paired bootstrap z-tests on the 19 adjacent pairs in the top 20, reported
both uncorrected and with Holm correction. **MDD:** `(z_0.975 + z_0.8) × SE(gap)`, i.e. the true
gap detectable with 80% power.

**Simulation.** `X[u, m] = mu[g(u), m] + a[u] + eps[u, m]` on the real unit/group structure. The
96 complete-coverage models start from their observed group means. Unit effects are resampled
from the real ones. Residuals are either resampled real residual rows (keeping cross-model
correlation, heteroscedasticity and tails) or Gaussian with the exact within-group covariance.
Scenarios: `calibrated`, `calibrated_gaussian`, `exact_ties` (top 5 tied, and ranks 20–29 tied),
`near_ties` (top 5 spaced 0.002 apart, ranks 20–29 spaced 0.001), `separated` (true gaps × 3).
With ties, an interval counts as covering only if it contains every rank the tied model could
legitimately take (`coverage`). `coverage_lenient` asks only that it contain at least one such rank.

## Outputs

- `results/summary.json`: headline numbers from every command.
- `results/tables/`: reproduction, rank intervals, neighbour tests, top-gap power,
  leave-one-group-out, common-assay leaderboard, Protriever's missing assays, version drift, and
  simulation summaries.
- `results/figures/`: `reproduction.png`, `rank_intervals_top40.png`, `pairwise_top20.png`,
  `leave_one_group_out.png`, `sim_summary.png`, `sim_coverage_by_rank.png`.

## Layout

```
src/pgnoise/
  data.py        download (pinned commit + checksums), loading, (UniProt, function) units
  stats.py       ProteinGym aggregation, stratified bootstrap, ProteinGym's gap-to-#1 SE
  ranks.py       rank intervals, best-model set, neighbour tests, Holm, MDD
  analysis.py    real-data analyses
  simulate.py    calibrated generator, scenarios, coverage experiment
  plots.py, cli.py
tests/           unit tests on toy data, plus integration tests on the real files
```
