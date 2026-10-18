"""Independent check of pgnoise's robustness grid and power table (genesis main, merged PR #1).

Recomputes every leaderboard in results/tables/robustness_summary.csv and every row of
results/tables/power_per_pair.csv from the raw ProteinGym tables with pgrank.py only, and asserts
they agree. Where scores are exactly tied, the order of tied models is arbitrary, so those checks
are reported instead of asserted.

Usage: uv run python crosscheck.py <pgnoise data/raw> <pgnoise results/>
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, norm

import pgrank as pg

raw, results = sys.argv[1], sys.argv[2]
summary = pd.read_csv(f"{results}/tables/robustness_summary.csv").set_index(["metric", "scheme"])

# Each averaging scheme as one line of pandas. "proteingym" is ProteinGym's own aggregate.
SCHEMES = {
    "proteingym": lambda d, m: pd.Series(pg.leaderboard_score(pg.strata(d, m)), index=m),
    "function_group_mean": lambda d, m: d.groupby("Selection Type")[m].mean().mean(),
    "uniprot_weighted": lambda d, m: d.groupby("UniProt ID")[m].mean().mean(),
    "flat_mean": lambda d, m: d[m].mean(),
    "median": lambda d, m: d[m].median(),
}


def top(scores, k):
    return list(scores.sort_values(ascending=False, kind="stable").index[:k])


print("1. Leaderboards across metrics and averaging schemes (vs robustness_summary.csv)\n")
print(f"{'metric':<11}{'scheme':<20}{'#1':<24}{'top-10 overlap':>15}{'tau':>8}{'|d tau|':>10}  ties  possible #1 (ours | pgnoise)")
base = None
for metric in pg.METRICS:
    d, s, models = pg.load(raw, metric)
    for scheme, f in SCHEMES.items():
        sc = f(d, models)
        if scheme == "proteingym":
            gap = np.abs(np.round(sc.to_numpy(), 3) - s.set_index("Model_name").loc[models, f"Average_{metric}"])
            assert gap.max() == 0, f"{metric}: published Average_{metric} not reproduced"
        if (metric, scheme) == ("Spearman", "proteingym"):
            base = sc
        sc = sc.loc[base.index]
        ref = summary.loc[(metric, scheme)]
        tau = kendalltau(sc, base).statistic
        overlap = len(set(top(sc, 10)) & set(top(base, 10)))
        r = np.round(sc.sort_values(ascending=False).to_numpy(), 10)
        tied = bool(len(np.unique(np.round(sc.to_numpy(), 10))) < len(sc))
        if not tied:
            assert abs(tau - ref.kendall_tau_vs_published) < 1e-9, (metric, scheme, "tau")
        if r[0] != r[1]:
            assert top(sc, 1)[0] == ref.top1, (metric, scheme, "#1")
        if r[9] != r[10]:
            assert overlap == ref.overlap_with_published_top10, (metric, scheme, "overlap")
        possible = ""
        if scheme == "proteingym":
            U = pg.strata(d, models)
            lo, _ = pg.rank_intervals(pg.leaderboard_score(U), pg.bootstrap(U, 4000, np.random.default_rng(0)))
            possible = f"{len([j for j in range(len(models)) if lo[j] == 1])} | {ref.best_set_size}"
        print(f"{metric:<11}{scheme:<20}{top(sc, 1)[0][:23]:<24}{overlap:>15}{tau:>8.3f}"
              f"{abs(tau - ref.kendall_tau_vs_published):>10.1e}  {'yes' if tied else 'no ':<4}  {possible}")
print("All published averages reproduced; every untied #1, overlap and tau matches pgnoise.\n")

print("2. Power at the top (vs power_per_pair.csv)\n")
d, s, models = pg.load(raw)
U = pg.strata(d, models)
boot = pg.bootstrap(U, 4000, np.random.default_rng(0), small_sample_correction=True)
z = norm.ppf(0.975) + norm.ppf(0.8)
pp = pd.read_csv(f"{results}/tables/power_per_pair.csv")
print(f"{'challenger':<20}{'SE closed form':>15}{'SE bootstrap':>14}{'units for 0.01':>16}{'for 0.005':>11}")
for row in pp.to_dict("records"):
    a, b = (models.index(x) for x in row["pair"].split(" vs "))
    gaps = [(u[:, b] - u[:, a])[~np.isnan(u[:, b] - u[:, a])] for u in U.values()]
    se = np.sqrt(sum(g.var(ddof=1) / len(g) for g in gaps)) / len(U)
    n0 = sum(len(g) for g in gaps)
    assert abs(se - row["se"]) < 1e-12 and n0 == row["units_now"], row["pair"]
    for delta in (0.005, 0.01):
        assert np.isclose(n0 * (z * se / delta) ** 2, row[f"units_for_{delta}"], rtol=1e-9), row["pair"]
    se_boot = (boot[:, b] - boot[:, a]).std(ddof=1)
    print(f"{models[b][:19]:<20}{se:>15.5f}{se_boot:>14.5f}{n0 * (z * se / 0.01) ** 2:>16.0f}"
          f"{n0 * (z * se / 0.005) ** 2:>11.0f}")
med = pp.iloc[int(np.argsort(pp.se.to_numpy())[len(pp) // 2])]
print(f"Closed-form SEs and required units match pgnoise exactly for all {len(pp)} pairs.")
print(f"Median pair ({med.pair}): {med['units_for_0.01']:.0f} units = {med['assays_for_0.01']:.0f} assays for 0.01.")
size = json.load(open(f"{results}/summary.json"))["power"]["size_at_delta0"]
print(f"Simulated false-positive rate of the test at zero gap (target 0.05), by benchmark size: {size}")
