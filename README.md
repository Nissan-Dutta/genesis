# pgnoise: how much of ProteinGym's leaderboard is noise?

This repo is a reproducible rank-uncertainty analysis of the
[ProteinGym](https://github.com/OATML-Markslab/ProteinGym) zero-shot DMS-substitution leaderboard
(97 models, 217 assays). It does seven things:

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
5. Re-runs the rank analysis under ProteinGym's other metrics (AUC, MCC, NDCG, top-K recall) and
   under four alternative aggregation schemes.
6. Computes power curves: how many assays it takes to detect a gain of 0.005 or 0.01 at the top.
7. Writes tables and figures to `results/`.

## Quick start

```bash
# needs uv (https://docs.astral.sh/uv/); Python 3.12 and all pinned dependencies come from uv.lock
uv sync
uv run pgnoise download          # ~0.7 MB of CSVs into data/raw/ (git-ignored), checksums verified
uv run pgnoise reproduce         # published averages and error bars vs ours   (~2 s)
uv run pgnoise versions          # score drift between ProteinGym releases     (~2 s)
uv run pgnoise ranks             # rank intervals and related analyses         (~15 s)
uv run pgnoise robustness        # 5 metrics x 5 aggregation schemes           (~40 s)
uv run pgnoise power             # power curves and benchmark size needed      (~5 s)
uv run pgnoise simulate          # coverage simulation, 5 scenarios x 1000 reps (~25 min on 4 cores)
uv run pytest                    # unit + integration tests
```

`uv run pgnoise all` runs everything in order. To try the simulation quickly, use
`uv run pgnoise simulate --reps 50`. If you don't use uv, `pip install -r requirements.txt && pip install -e .`
works too. The requirements file is exported from the lock file.

## Data and version

- Source: `OATML-Markslab/ProteinGym`, commit `144fe22` (2026-03-25). The per-assay file at this
  commit is byte-identical to the June 2025 "Added Protriever" commit, so it is **release v1.3
  plus the two later additions, AIDO Protein-RAG and Protriever**. That gives 97 models.
- Files used: `DMS_substitutions_{Spearman,AUC,MCC,NDCG,Top_recall}_DMS_level.csv` (per-assay
  scores, rounded to 3 dp, as ProteinGym itself aggregates) and `reference_files/DMS_substitutions.csv` (UniProt ID and
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
| Step-down variants | same guarantees, tighter (Romano-Wolf); under-covers with exact ties in small strata |
| Bootstrap-t step-down (`studentized.py`) | as above, each bootstrap replicate studentised with its own linearised SE |
| Best-model set (one-sided max-t vs each model) | contains the true #1 w.p. ≥ 95% |
| "Within 1.96 SE of #1" (how ProteinGym error bars are usually read) | none |

**Neighbour tests.** Paired bootstrap z-tests on the 19 adjacent pairs in the top 20, reported
both uncorrected and with Holm correction. **MDD:** `(z_0.975 + z_0.8) × SE(gap)`, i.e. the true
gap detectable with 80% power.

**Aggregation schemes** (`aggregation.py`). Every scheme uses the same stratified unit bootstrap,
so only the estimand changes:

| Scheme | Definition |
|---|---|
| `proteingym` | assays → (UniProt, function) units → 5 function groups → mean (the published one) |
| `function_group_mean` | mean of the 5 function-group means of raw assays (no UniProt step) |
| `uniprot_weighted` | every protein weighted equally, function groups ignored |
| `flat_mean` | plain mean over 217 assays |
| `median` | median over 217 assays; its bootstrap is rough because scores are rounded to 3 dp |

A side note: the "Average" row of ProteinGym's published `Uniprot_level.csv` is *not* an
equal-per-protein mean. It is reproduced exactly (97/97) only by replicating the script's
non-deduplicated merge, which counts some proteins up to 4 times. It differs from equal weights
by up to 0.024. The leaderboard itself does not use it.

**Power** (`power.py`). A new model is assumed to beat the current #1 by a true Δ, with paired
per-unit differences as variable as the real ones between #1 and each of the next 10
complete-coverage models. Assays are assumed to arrive as new units with today's function mix.
Required units = `n0 × ((z_0.975 + z_0.8) × SE0 / Δ)²`. The formula is checked by resampling the
real paired differences, re-centred at Δ, at scaled benchmark sizes and running the same z-test.

**Simulation.** `X[u, m] = mu[g(u), m] + a[u] + eps[u, m]` on the real unit/group structure. The
96 complete-coverage models start from their observed group means. Unit effects are resampled
from the real ones. Residuals are either resampled real residual rows (keeping cross-model
correlation, heteroscedasticity and tails) or Gaussian with the exact within-group covariance.
Scenarios: `calibrated`, `calibrated_gaussian`, `exact_ties` (top 5 tied, and ranks 20–29 tied),
`near_ties` (top 5 spaced 0.002 apart, ranks 20–29 spaced 0.001), `separated` (true gaps × 3).
With ties, an interval counts as covering only if it contains every rank the tied model could
legitimately take (`coverage`). `coverage_lenient` asks only that it contain at least one such rank.

## Headline results (pinned data, 10,000 bootstrap reps)

- **Reproduction.** All 97 published averages and ranks match exactly, as do the MSA-depth and
  taxon breakdowns. 4 of 485 function-group cells differ by 0.001: they are exact half-way values
  (e.g. 0.3115), where floating-point summation order decides the rounding. 91 of 97 error bars
  match at 3 dp. The other 6 lie within 0.0001 of a rounding boundary and flip between bootstrap
  seeds.
- **#1 is a coin flip.** AIDO Protein-RAG and VenusREM differ by 0.00004, and each is #1 in about
  50% of bootstrap replicates. Dropping Activity or OrganismalFitness assays makes VenusREM #1.
- **Models that cannot be ruled out as #1** (max-t, 95%): AIDO Protein-RAG, VenusREM, ProSST (K=4096).
- **Top-20 neighbours.** Only 2 of 19 adjacent pairs differ significantly, with or without Holm
  correction.
- **Marginal 95% rank intervals.** #4 ProSST (K=4096) is ranks 1–13; #10 ProSST (K=512) is
  ranks 6–28. Simultaneous intervals are 1–18 and 6–42.
- **Power at the top.** #1 vs #3 gap = 0.011 with SE 0.006 (z = 1.8). The minimum detectable
  difference against #1 (80% power) is 0.015–0.026 across the next ten models.
- **Coverage.** Protriever is scored on 200/217 assays. Its 17 missing assays are harder (other
  models average 0.379 on them vs 0.410 elsewhere). On the 200 common assays it falls from #8
  to #10.
- **Simulation** (1,000 synthetic leaderboards per scenario). Marginal single-step and both
  simultaneous intervals reach ≥ 97% worst-model coverage in every scenario. Naive bootstrap
  percentile intervals fall to 8% coverage for some exactly tied models, and to 0% joint
  coverage. Marginal step-down under-covers with exact ties (worst model 91%), so it is not
  used for headline numbers. Studentising it (bootstrap-t) lifts the exact-tie worst case to
  94.0%. That is still just short of 95%, and it is no narrower than single-step (top-10 width
  about 10 ranks for both), so single-step remains the headline. The simultaneous bootstrap-t
  variant is valid but much wider (top-10 width 18–23 vs 14).

- **Other metrics.** All published AUC, MCC, NDCG and top-K-recall averages and ranks also
  reproduce exactly. AUC and MCC agree with Spearman: the possible-#1 set is always within
  {AIDO, VenusREM, ProSST K=4096}. NDCG puts S3F-MSA first under every mean-based scheme
  (possible #1: 4–6 models). Only 4 of the published top 10 stay in its top 10, and Kendall τ
  against the published ranking is 0.65–0.70. Top-K recall cannot separate the top: 9–12
  models remain possible #1s.
- **Other aggregation schemes (Spearman).** #1 flips to VenusREM under the function-group mean
  and the median. UniProt-weighted and flat means narrow the possible-#1 set to {AIDO,
  VenusREM}. Medians widen it to 12.
- **Power.** Detecting a 0.01 gain over #1 with 80% power needs about 780 units (≈ 850 assays,
  3.9× today; range 450–1,460 units over the top-10 pairs). A 0.005 gain needs about 3,100 units
  (≈ 3,400 assays, 15.6× today; range 1,800–5,800). Today's benchmark detects about 0.020
  (range 0.015–0.027).

## Outputs

- `results/summary.json`: headline numbers from every command.
- `results/tables/`: reproduction, rank intervals, neighbour tests, top-gap power,
  leave-one-group-out, common-assay leaderboard, Protriever's missing assays, version drift, and
  simulation summaries.
- `results/figures/`: `reproduction.png`, `rank_intervals_top40.png`, `pairwise_top20.png`,
  `leave_one_group_out.png`, `robustness_rank_heatmap.png`, `robustness_grid.png`,
  `power_curves.png`, `sim_summary.png`, `sim_coverage_by_rank.png`.

## Layout

```
src/pgnoise/
  data.py        download (pinned commit + checksums), loading, (UniProt, function) units
  stats.py       ProteinGym aggregation, stratified bootstrap, ProteinGym's gap-to-#1 SE
  ranks.py       rank intervals, best-model set, neighbour tests, Holm, MDD
  studentized.py bootstrap-t step-down rank intervals (linearised per-replicate SEs)
  aggregation.py alternative aggregation schemes under the same bootstrap
  robustness.py  metric x scheme grid
  power.py       power curves and required benchmark size
  analysis.py    real-data analyses
  simulate.py    calibrated generator, scenarios, coverage experiment
  plots.py, cli.py
tests/           unit tests on toy data, plus integration tests on the real files
```
