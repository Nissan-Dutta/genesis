# Examiner's Bug Review — `pgnoise` (ProteinGym rank-uncertainty package)

**Repository:** https://github.com/Nissan-Dutta/genesis  
**Branch reviewed:** `cline/zbk1yf72` (from `main` @ `39a032e`)  
**Scope:** Static source review of `src/pgnoise/` and `tests/`.  
**Verification limitation:** The sandbox has no `numpy`/`scipy`/`pandas`/`pytest` installed and no network access, so I **could not execute** the test suite or import the package at runtime. Every finding below is grounded in static reading of the source (all `.py` files read in full). Items marked "requires execution" must be confirmed by running the tests once the environment is provisioned (`uv sync` then `pytest`).

---

## 1. Executive summary

`pgnoise` is a well-structured, well-tested statistics package. The core reproduction path is validated against real ProteinGym data (the committed `results/summary.json` reports 97/97 published averages and ranks reproduced exactly; 91/97 bootstrap SEs at 3 dp). I did **not** find any crash-level bug, missing import, or obviously-wrong arithmetic in the core aggregation / rank-interval / power machinery.

However, as an examiner I would flag **three substantive concerns** before accepting the headline claims:

1. A **critical-value indexing mismatch in `best_confidence_set`** (`ranks.py`) that undercuts the formal coverage guarantee of the "possible #1" set — the headline result "three models cannot be ruled out as #1".
2. **Coverage-test assertions are too weak** (`tests/test_simulate.py`) — they do not actually verify nominal 95% coverage, so the headline validity claim ("worst-model coverage ≥ 0.973") is not guarded by tests.
3. A **docstring vs implementation mismatch in `proteingym_gap_se`** (`stats.py`) — a candidate explanation for the 6/97 SEs that miss at 3 dp.

Details, severity, and required verification commands follow.

---

## 2. Findings

### F-1. `best_confidence_set` uses a per-beater critical value to exclude per-model — coverage not guaranteed as claimed

**File:** `src/pgnoise/ranks.py`, lines 154–157  
**Severity:** High (affects headline claim)  
**Confidence:** High (logic), requires execution to quantify

```python
ms = ms or max_statistics(theta_hat, theta_boot)
crit = np.quantile(ms.best_one_sided, 1 - alpha, axis=0)   # crit[k], indexed by model k
beaten = (ms.diff.T > crit[:, None] * ms.se).any(axis=1)   # beaten[j]
return ~beaten
```

- `ms.best_one_sided[b, k] = max_j T[b, j, k]` = max evidence that *some model j beats model k*.  
  So `crit[k]` is calibrated for the question **"is model k beaten by anyone?"** — i.e. it belongs to the model being *beaten* (`k`).
- But `beaten[j] = any k : theta_k - theta_j > crit[k] * se[k,j]` uses `crit[k]` (indexed by the **beater** `k`) to test whether `k` beats `j` — i.e. whether `j` should be excluded. The critical value appropriate for excluding `j` should be indexed by `j` (`quantile` of `max_k T[b,k,j]`, max evidence some model beats `j`).

Under the **global null** (exchangeable models) the two coincide, so the single-step best-set is approximately calibrated there. Under a **realistic non-exchangeable leaderboard** (models have different true scores and variances) the calibration is not the one implied by the docstring: *"Model j is excluded only if some k is significantly better … contains the true best with prob ≥ 1 − alpha."*

**Impact:** The best-set / "possible #1" results (README: "three models cannot be ruled out as #1"; `results/summary.json` best-set coverage 0.973) rely on this procedure. The mismatch means the advertised ≥ 1−α coverage is not formally guaranteed by this code.

**Required verification:** Run the coverage simulation (`uv run pgnoise simulate`) and check `simulation.best_set_coverage` per scenario against the nominal 0.95. Also strengthen `tests/test_simulate.py::test_small_coverage_run_is_sane` (see F-2).


### F-2. Simulation test assertions are too weak to validate the headline coverage claim

**File:** `tests/test_simulate.py`, `test_small_coverage_run_is_sane`  
**Severity:** High (test gap — the headline validity claim is untested)  
**Confidence:** High

```python
s = res["summary"].set_index("method")
assert s.loc["simultaneous_single_step", "joint_coverage"] >= 0.85   # below nominal 0.95
assert (s["mean_width"] >= 0).all()
assert 0 <= res["best"]["best_set_coverage"] <= 1
```

- `joint_coverage >= 0.85` is **below** the nominal 0.95 the paper claims to maintain ("headline intervals keep worst-model coverage at 0.973 or above").
- `best_set_coverage` is only asserted to lie in `[0, 1]` — a test that passes even if the best set never covers the true best.
- No assertion that per-model coverage ≈ 0.95 or that worst-model coverage is near nominal.

**Impact:** A regression that breaks coverage (e.g. the F-1 indexing issue, or a liberal step-down) would not be caught. The README's headline statistical guarantees are not protected by tests.

**Required fix:** Assert `joint_coverage` and `min_model_coverage` are close to 0.95 (e.g. `>= 0.90` as a smoke floor, or `pytest.approx(0.95, abs=...)` on a larger run), and assert `best_set_coverage` is near nominal (e.g. `>= 0.90`). Then run `pytest`.

### F-3. `proteingym_gap_se` docstring says NaNs are *not* skipped, but the implementation skips them

**File:** `src/pgnoise/stats.py`, lines 85–108  
**Severity:** Medium (docstring/implementation mismatch; candidate explanation for SE mismatch)  
**Confidence:** High

Docstring (line 96–97): *"…the group means are averaged **without** skipping NaNs, and the SD (ddof=1) is taken over finite replicates."*

Implementation (line 103): `weighted_group_means(D, groups, n_groups, counts).mean(axis=1)`, where `weighted_group_means` computes `den = W @ observed` and returns `num/den` only where `den > 0` — i.e. it **does skip NaNs within each group** (mean over observed units). So the code skips NaNs, contradicting the docstring.

This is a plausible explanation for why the committed `results/summary.json` reports `Bootstrap_standard_error_Spearman`: **91/97 exact at 3 dp, max abs diff 0.001** (6 of 97 off). The README hand-waves these as "within 0.0002 of a rounding boundary and change with the bootstrap seed," but a NaN-handling difference is a more substantive candidate and should be resolved rather than attributed to seed noise.

**Required verification:** Reproduce the SE column against the published values and inspect the 6 mismatching models; decide whether ProteinGym's published SE truly averages group means without skipping NaNs, and align code + docstring accordingly.

### F-4. (Not a bug, but a resource watch) Large `(B, p, p)` float32 tensor in studentised/max-t paths

**Files:** `src/pgnoise/ranks.py` (`_abs_t`, `_DENSE_LIMIT = 40_000_000`), `src/pgnoise/studentized.py` (`absT` of shape `(n_boot, p, p)`)

- `max_statistics` builds a `(chunk, p, p)` tensor per chunk (chunk=500); with p=97 that is ~18 MB per chunk — fine.
- `studentized_rank_ci` builds the full `(n_boot, p, p)` float32 tensor (`absT`). With n_boot=1000, p=97 → ~365 MB; with n_boot=10,000 → ~3.6 GB. This is below `_DENSE_LIMIT` only when `n_boot*p*p <= 40M`; otherwise it falls back to chunking in `ranks`, but `studentized_rank_ci` allocates the full tensor upfront and could OOM on large bootstrap counts.

**Recommendation:** chunk `studentized_rank_ci` the same way `ranks.py` does, or cap `n_boot` for the studentised methods. Low priority for the current 1,000-rep simulation runs.

---

## 3. Items reviewed and found NOT to be bugs

These were checked carefully and are consistent (and match the documented behavior / tests):

| Item | File | Assessment |
|------|------|------------|
| `pairwise_se`, `max_statistics`, `_bounds_from_critical` | `ranks.py` | Lower ≤ point rank ≤ upper nesting holds; marginal/simultaneous crit logic correct. |
| `percentile_rank_ci`, `pairwise_rank_ci` (single-step & stepdown) | `ranks.py` | Consistent with MRSW (2024) max-|t|; nesting test logic sound. |
| `best_confidence_set` one-sided logic | `ranks.py` | Structurally reasonable; only the crit indexing (F-1) is a concern. |
| `pairwise_variance` linearisation (incl. missing data / `den==0`) | `studentized.py` | Matches `test_linearised_variance_single_group_is_sample_variance_over_n`; `den==0` → 0 contribution, no NaN propagation. |
| `group_means`, `proteingym_score`, `stratified_counts`, `bootstrap_scores` | `stats.py` | Correct within-group resampling; NaN-skipping matches pandas; small-sample correction optional. |
| `analytic_se` / `analytic_power` / `required_units` / `simulated_power` | `power.py` | SE and power algebra verified against tests; `scale` handling consistent. |
| `make_design` unit / uniprot / flat weights; `_weighted_median` | `aggregation.py` | Match pandas groupby / median (tests encode this). |
| `to_units`, `calibrate`, `generate`, `true_rank_sets` | `data.py` / `simulate.py` | Correct two-way decomposition and tie-rank sets. |
| `download`, `path_for`, `download_release_summaries` | `data.py` | All exist and are called correctly by `cli.py` (no missing imports). |
| CLI commands (`reproduce`, `ranks`, `robustness`, `power`, `simulate`, `figure`, `numbers`, `esm2`) | `cli.py` | Wire up correctly; `"all"` composes the sub-commands. |

---

## 4. Recommended verification commands (run once environment is provisioned)

```bash
cd /workspace
uv sync                 # install numpy, scipy, pandas, pytest, ruff (py>=3.12)
uv run pytest tests/ -q  # run full suite (test_esm2.py needs the `esm` extra + downloaded data)
uv run pytest tests/test_simulate.py -q   # focus: coverage tests (F-2)
uv run pytest tests/test_stats.py -q      # focus: gap SE (F-3)
uv run pytest tests/test_ranks.py -q      # focus: rank intervals / best set (F-1 context)
uv run pgnoise simulate --reps 200 --boot 500   # check best_set_coverage vs 0.95 (F-1)
uv run pgnoise reproduce --boot 200             # check SE column matches (F-3)
uv run ruff check src/ tests                  # lint
```

To provision Python 3.12 (per `.python-version`) if needed: `uv python install 3.12`.

---

## 5. Examiner's verdict

`pgnoise` is **statistically serious and well-engineered**; the core reproduction (leaderboard averages + ranks) is validated against real ProteinGym data. I would **not reject it on crash-level defects**.

The two issues I would raise in a lab review are:

1. **The best-set critical-value indexing (F-1)** — it weakens the formal guarantee behind the headline "three models cannot be ruled out as #1" claim; it must be checked empirically via the coverage simulation.
2. **The coverage tests do not assert nominal coverage (F-2)** — the headline validity guarantee is not actually protected by the test suite, which is a real gap for a lab that ships statistical software.

Fixing F-2 (stronger coverage assertions) and resolving F-3 (docstring/SE alignment) are concrete, low-risk changes. F-1 may require a methods correction to `best_confidence_set` if the simulation shows coverage below nominal.

---

*Review performed by static reading only; no runtime execution was possible in this sandbox. All file:line references refer to `src/pgnoise/` and `tests/` at commit `39a032e` on branch `cline/zbk1yf72`.*

