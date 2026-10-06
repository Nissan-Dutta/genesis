# How much of the ProteinGym leaderboard is noise?

[ProteinGym](https://github.com/OATML-Markslab/ProteinGym) ranks 97 models that predict the effects of
mutations, using 217 deep mutational scanning assays. This repo asks which of the differences on that
leaderboard are real. It covers how uncertain each model's rank is, how large a gain the benchmark can
detect, and which parts of the benchmark the uncertainty comes from.

Scope so far: the zero-shot substitution leaderboard (Spearman), at ProteinGym commit `144fe22`
(2026-03-25). The uncertainty modelled is which proteins happen to be in the benchmark. It uses the
stratified cluster bootstrap that ProteinGym uses for its own published standard errors. Noise inside
each assay is not modelled yet.

## Reproduce

```bash
uv sync
uv run python replicate.py                     # about 2 s: every number in "Results"
uv run python coverage_sim.py                  # about 100 s on 8 cores
uv run python coverage_sim.py --shrink 0.5     # stress test with all true gaps halved
```

`data/fetch.sh` re-downloads the two input tables from the pinned commit and checks their SHA-256.

| file | what it does |
|---|---|
| `pgrank.py` | the whole method: aggregation, bootstrap, rank intervals |
| `replicate.py` | reproduces the published leaderboard and the pilot, then prints the results below |
| `coverage_sim.py` | checks whether the intervals contain the true rank as often as they claim |
| `results/` | `rank_intervals.csv` (all 97 models), `coverage*.md`, `coverage_per_model*.csv` |

## Results

1. **Replication.** The published mean Spearman is reproduced exactly (3 dp) for all 97 models. The published
   standard errors are reproduced within 0.001 for 96 of 97. The exception is Protriever (0.008 here vs 0.006
   published), the only model with missing assays.
2. **The top two are tied.** AIDO Protein-RAG (16B) leads VenusREM by 0.00004. The standard error of
   that gap is 0.005.
3. **Ranks below the top two are uncertain by many places.** The 95% rank intervals for the top 20 average
   18 ranks wide with the per-model method (guaranteed coverage, conservative). They average 10.6 ranks
   wide even with no multiple-comparison correction. For example, the #10 model (ProSST K=512) is
   somewhere in [6, 30], or [7, 22] without correction.
4. **Only 2 of the 19 neighbouring pairs in the top 20 are distinguishable**, even with no correction for
   multiple comparisons, so with a correction it is 2 at most.
5. **Smallest detectable gain at the top.** About 0.015 Spearman (80% power, two-sided 5%) for a pair
   like #1 vs #2. The gap between #1 and #3 is 0.011.
6. **Binding dominates the uncertainty.** The leaderboard weights its five function categories equally. So
   the 12 Binding proteins (6% of assays) account for a median 37% of the variance in the top models' gaps
   to #1. The 66 Stability proteins account for 6%. Per assay added, Binding is by far the most efficient
   place to make the top of the leaderboard more decisive.
7. **Protriever is ranked on an easier subset.** It has no scores for 17 assays, one alphabetical block
   from GFP_AEQVI to MBD11_ARATH. On those assays the other top-10 models average 0.441 Spearman, against
   0.520 on the rest. Scored on the same 200 assays as everyone else, Protriever moves from #8 to #10.

## Do the intervals contain the true rank 95% of the time?

`coverage_sim.py` treats the observed proteins as the whole population, so the true ranks are known.
It draws 1,000 new benchmarks of the same shape and reruns the full analysis on each one.

| method | each model's interval contains its true rank | all 97 contain theirs at once | mean width, top 20 |
|---|---|---|---|
| per-model (the pilot's) | ≥ 99.6% for every model | 97.5% | 17.6 ranks |
| all at once (simultaneous) | ≥ 99.9% | 100% | 23.2 ranks |
| no multiple-comparison correction | ≥ 94.6% | **50.5%** | 10.6 ranks |

- The per-model method is valid but conservative: it never falls below 95%, but it is far above it. The
  intervals in the Results section therefore overstate rank uncertainty somewhat.
- The harness catches a bad method. Without the multiple-comparison correction, all 97 intervals hold at
  once in only half the simulated benchmarks.
- Bootstrap standard errors come out about 3.5% too small, the usual small-sample bias with only 12 Binding
  proteins. Rescaling each category's bootstrap by √(n/(n−1)) fixes it (ratio 0.965 → 0.997). ProteinGym's
  published SEs share this bias. It doesn't change any conclusion here because the intervals are conservative.
- The simulation assumes the true scores equal the observed ones. Observed gaps are noisier, and so more
  spread out, than true gaps. With every true gap halved (`--shrink 0.5`) the conclusions are unchanged:
  per-model coverage ≥ 99.7%, 97.7% all at once, and 52.2% for the uncorrected method.

## Careful with these claims

- The pilot calls its intervals "simultaneous". They are per-model ("marginal" in Mogstad et al. 2024):
  each covers its own model's rank, with no guarantee that all 97 hold together. They happened to hold
  jointly 97.5% of the time here. Use `method="simultaneous"` for any figure that shows every model at once.
- "The #1 spot changes if you drop Activity or Organismal Fitness" follows directly from a 0.00004 gap.
  Don't present it as a separate finding.
- The pilot's headline example, "#4 could be anywhere from 1 to 13", is fragile. That model has an unusually
  large standard error, and without correction its interval is [3, 7]. Likewise, 3 models can't be ruled
  out as #1 under the per-model method, but only 2 can't without correction. Lead with the top-20 average
  widths or the #10 example, which hold up under every method.
- The bootstrap treats the 14 proteins that appear in two categories as independent draws.
- The novelty claim (first rank-uncertainty analysis of ProteinGym) rests on a quick literature search.
  Re-check it before stating it.

## Next

- Tighten the guaranteed intervals with the step-down version of the same method (Mogstad et al. 2024),
  then re-run the coverage check.
- Plot detectable effect size against the number of assays, broken down by which category gets them.
- Add mutation-level noise from ProteinGym's per-mutant score files (4.4 GB).
