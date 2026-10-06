"""Reproduce the pilot's numbers with pgrank.py and check them against the published leaderboard.

Usage: uv run python replicate.py [data_dir]
Writes results/rank_intervals.csv (one row per model).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import pgrank as pg

# The pilot's printed top-15 rank intervals (seed 0, B=4000); used as a regression check.
PILOT_TOP15 = [(1, 4), (1, 4), (2, 7), (1, 13), (2, 9), (4, 12), (4, 20), (3, 20), (5, 17),
               (6, 30), (6, 22), (6, 22), (6, 33), (5, 25), (6, 30)]

data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
d, s, models = pg.load(data_dir)
M = len(models)
U = pg.strata(d, models)
theta = pg.leaderboard_score(U)
order = np.argsort(-theta)
pub = s.set_index("Model_name").loc[models]

print(f"{M} models, {len(d)} assays; proteins per category: { {c: u.shape[0] for c, u in U.items()} }")

# 1. Point estimates
gap = np.abs(np.round(theta, 3) - pub.Average_Spearman.to_numpy())
assert gap.max() == 0, "leaderboard means do not replicate"
print("Published Average_Spearman reproduced exactly (all models, 3 dp).")
print(f"#1 vs #2 unrounded gap: {theta[order[0]] - theta[order[1]]:.5f}  "
      f"({models[order[0]]} vs {models[order[1]]})")

# 2. Standard errors (same seed and draws as the pilot)
boot = pg.bootstrap(U, 4000, np.random.default_rng(0))
leader = order[0]
se_vs_leader = (boot - boot[:, [leader]]).std(axis=0, ddof=1)
se_gap = np.abs(se_vs_leader - pub.Bootstrap_standard_error_Spearman.to_numpy())
off = [(models[j], round(se_vs_leader[j], 4), pub.Bootstrap_standard_error_Spearman.iloc[j])
       for j in np.where(se_gap > 0.001)[0]]
print(f"SE of gap to #1 matches the published value within 0.001 for {M - len(off)}/{M} models; "
      f"mismatches (ours, published): {off}")

# 3. Rank intervals, per-model (marginal) and joint (simultaneous)
lo, hi = pg.rank_intervals(theta, boot)
slo, shi = pg.rank_intervals(theta, boot, method="simultaneous")
nlo, nhi = pg.rank_intervals(theta, boot, method="unadjusted")
assert [(lo[j], hi[j]) for j in order[:15]] == PILOT_TOP15, "marginal intervals differ from the pilot"
print("Per-model intervals match the pilot's top 15 exactly.\n")
print("Three 95% rank intervals. 'per-model' is the pilot's (guaranteed for each model, conservative);")
print("'all at once' holds jointly for all 97; 'no correction' is the optimistic lower bracket")
print("(no guarantee; see coverage_sim.py).\n")
print("rank  per-model  all at once  no correction  score  model")
for r, j in enumerate(order[:20], 1):
    print(f"{r:>4}  [{lo[j]:>2},{hi[j]:>2}]    [{slo[j]:>2},{shi[j]:>2}]      [{nlo[j]:>2},{nhi[j]:>2}]        "
          f"{theta[j]:.3f}  {models[j]}")
top20 = order[:20]
print(f"mean width, top 20: per-model {(hi - lo + 1)[top20].mean():.1f}, all at once "
      f"{(shi - slo + 1)[top20].mean():.1f}, no correction {(nhi - nlo + 1)[top20].mean():.1f} ranks")

for name, l in [("per-model", lo), ("all at once", slo), ("no correction", nlo)]:
    print(f"Not ruled out as #1 ({name}): {[models[j] for j in range(M) if l[j] == 1]}")

# 4. Adjacent pairs and minimum detectable effect (unchanged from the pilot)
sep = sum(np.quantile(boot[:, a] - boot[:, b], 0.025) > 0 for a, b in zip(order[:19], order[1:20]))
print(f"Adjacent top-20 pairs separable at 95% (no multiplicity correction): {sep}/19")
print(f"MDE at 80% power, two-sided 5%, #1 vs #2: {2.8 * se_vs_leader[order[1]]:.4f}; "
      f"observed #1 - #3 gap: {theta[order[0]] - theta[order[2]]:.4f}")

# 5. Protriever on the same assays as everyone else
missing = d["Protriever"].isna()
theta_common = pg.leaderboard_score(pg.strata(d[~missing], models))
p = models.index("Protriever")
others = [j for j in order[:10] if j != p]
print(f"\nProtriever: missing {missing.sum()} of {len(d)} assays "
      f"({d.loc[missing, 'DMS ID'].iloc[0]} ... {d.loc[missing, 'DMS ID'].iloc[-1]}).")
print(f"  published rank {pg.ranks(theta)[p]} -> rank {pg.ranks(theta_common)[p]} "
      f"when all models are scored on its {(~missing).sum()} assays")
print(f"  other top-10 models, mean assay Spearman on the {missing.sum()} missing assays: "
      f"{d.loc[missing, [models[j] for j in others]].to_numpy().mean():.3f} vs "
      f"{d.loc[~missing, [models[j] for j in others]].to_numpy().mean():.3f} on the rest")

# 6. Which category the uncertainty comes from. Var(gap) = sum over categories of
#    (1/5)^2 * var_across_proteins(gap) / n_proteins, so each term is that category's share.
share = {}
for c, u in U.items():
    g = u[:, order[1:10]] - u[:, [leader]]                 # gaps of #2..#10 to #1, per protein
    share[c] = np.nanvar(g, axis=0, ddof=1) / (~np.isnan(g)).sum(axis=0) / len(U) ** 2
total = sum(share.values())
print("\nShare of the uncertainty in #2-#10's gap to #1, by category (median over the 9 models):")
for c, u in U.items():
    print(f"  {c:<18} {len(u):>3} proteins  {np.median(share[c] / total):.0%}")

Path("results").mkdir(exist_ok=True)
pd.DataFrame({"model": models, "score": theta, "rank": pg.ranks(theta),
              "rank_lo_per_model": lo, "rank_hi_per_model": hi,
              "rank_lo_simultaneous": slo, "rank_hi_simultaneous": shi,
              "rank_lo_no_correction": nlo, "rank_hi_no_correction": nhi,
              "se_gap_to_leader": se_vs_leader}).sort_values("rank").to_csv(
    "results/rank_intervals.csv", index=False, float_format="%.5f")
print("\nwrote results/rank_intervals.csv")
