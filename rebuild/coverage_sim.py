"""Do the "95% rank intervals" contain the true rank 95% of the time?

Plasmode simulation: treat ProteinGym's observed proteins as the whole population, so the true
leaderboard (the published one) and every model's true rank are known exactly. Each replicate
  1. draws a new benchmark of the same shape (n_c proteins per category, with replacement),
  2. reruns the full analysis on it (its own bootstrap, its own rank intervals),
  3. records whether each model's interval contains that model's true rank.
A method that works should hit the true rank at least 95% of the time.

The simulated benchmarks are as small and lumpy as the real one (12 Binding proteins, one model
with missing assays), so this tests the method under the conditions where it is actually used.

Weak spot: the observed leaderboard is a noisy estimate, so its gaps are more spread out than the
true gaps, and the real benchmark probably has more near-ties than this simulated one.
--shrink 0.5 stress-tests that by halving every model's distance from the average score (a
constant shift per model, so the protein-to-protein noise is untouched).

Usage: uv run python coverage_sim.py [--reps 1000] [--boot 1000] [--seed 1] [--shrink 1.0] [--data data]
Writes results/coverage[_shrinkX].md and results/coverage_per_model[_shrinkX].csv.
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import pgrank as pg

# name: (pgrank.rank_intervals method, small_sample_correction)
METHODS = {
    "per-model (marginal)": ("marginal", False),
    "per-model, small-n fix": ("marginal", True),
    "simultaneous": ("simultaneous", False),
    "naive: no multiple-comparison correction": ("unadjusted", False),
}


def true_ranks(theta):
    """1 + number of models strictly better (ties share the better rank)."""
    return 1 + (theta[None, :] > theta[:, None]).sum(axis=1)


def run_chunk(U, theta_true, reps, B, seed):
    rng = np.random.default_rng(seed)
    M = len(theta_true)
    r_true = true_ranks(theta_true)
    best = np.argmax(theta_true)
    out = {m: {"cover": np.zeros((reps, M), bool), "best_kept": np.zeros(reps, bool),
               "width": np.zeros((reps, M))} for m in METHODS}
    out["gap"] = np.zeros((reps, M))                       # estimated gap to the true #1
    out["se"] = {fix: np.zeros((reps, M)) for fix in (False, True)}  # its bootstrap SE
    for r in range(reps):
        Ustar = {c: u[rng.integers(0, len(u), len(u))] for c, u in U.items()}
        theta = pg.leaderboard_score(Ustar)
        out["gap"][r] = theta - theta[best]
        inner_seed = rng.integers(2**63)
        for fix in (False, True):
            # same resampling indices with and without the fix, so the comparison is paired
            boot = pg.bootstrap(Ustar, B, np.random.default_rng(inner_seed), small_sample_correction=fix)
            boot = boot[~np.isnan(boot).any(axis=1)]       # astronomically rare all-missing draws
            out["se"][fix][r] = (boot - boot[:, [best]]).std(axis=0, ddof=1)
            for name, (method, method_fix) in METHODS.items():
                if method_fix != fix:
                    continue
                lo, hi = pg.rank_intervals(theta, boot, method=method)
                out[name]["cover"][r] = (lo <= r_true) & (r_true <= hi)
                out[name]["best_kept"][r] = lo[best] == 1
                out[name]["width"][r] = hi - lo + 1
    return out


def merge(chunks):
    def cat(key_path):
        parts = []
        for ch in chunks:
            x = ch
            for k in key_path:
                x = x[k]
            parts.append(x)
        return np.concatenate(parts)
    res = {m: {k: cat((m, k)) for k in ("cover", "best_kept", "width")} for m in METHODS}
    res["gap"] = cat(("gap",))
    res["se"] = {fix: cat(("se", fix)) for fix in (False, True)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--data", default="data")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--shrink", type=float, default=1.0)
    a = ap.parse_args()

    d, s, models = pg.load(a.data)
    U = pg.strata(d, models)
    theta_obs = pg.leaderboard_score(U)
    shift = -(1 - a.shrink) * (theta_obs - theta_obs.mean())
    U = {c: u + shift for c, u in U.items()}
    theta_true = pg.leaderboard_score(U)
    r_true = true_ranks(theta_true)
    top20 = np.argsort(-theta_true)[:20]
    best = np.argmax(theta_true)

    n_chunks = max(1, min(a.workers * 4, a.reps))
    sizes = [len(x) for x in np.array_split(np.arange(a.reps), n_chunks)]
    seeds = [int(ss.generate_state(1)[0]) for ss in np.random.SeedSequence(a.seed).spawn(n_chunks)]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        chunks = list(ex.map(run_chunk, [U] * n_chunks, [theta_true] * n_chunks, sizes,
                             [a.boot] * n_chunks, seeds))
    res = merge(chunks)
    R = a.reps
    mc = np.sqrt(0.95 * 0.05 / R)

    title = f"# Coverage of 95% rank intervals ({R} simulated benchmarks, {a.boot} bootstrap draws each"
    title += ")" if a.shrink == 1 else f", true gaps shrunk to {a.shrink:g}x the observed gaps)"
    lines = [title,
             "",
             f"Monte Carlo standard error of a coverage near 95%: {mc:.3f}. Run time {time.time() - t0:.0f}s.",
             "",
             "| method | mean per-model coverage | worst model | top-20 models: mean (min) | all 97 at once | true #1 kept in 'could be #1' set | mean width, top 20 |",
             "|---|---|---|---|---|---|---|"]
    per_model = {"model": models, "true_rank": r_true}
    for name in METHODS:
        cov = res[name]["cover"].mean(axis=0)
        per_model[f"coverage: {name}"] = cov
        w = np.argmin(cov)
        lines.append(
            f"| {name} | {cov.mean():.3f} | {cov[w]:.3f} ({models[w]}, true rank {r_true[w]}) | "
            f"{cov[top20].mean():.3f} ({cov[top20].min():.3f}) | {res[name]['cover'].all(axis=1).mean():.3f} | "
            f"{res[name]['best_kept'].mean():.3f} | {res[name]['width'][:, top20].mean():.1f} ranks |")

    true_sd = res["gap"].std(axis=0, ddof=1)
    others = [j for j in top20 if j != best]
    ratio = {fix: (res["se"][fix].mean(axis=0)[others] / true_sd[others]) for fix in (False, True)}
    lines += ["",
              "## Are the bootstrap standard errors the right size?",
              "",
              "Ratio of the average bootstrap SE of each top-20 model's gap to the true #1, to the actual "
              "spread of that gap across simulated benchmarks (1.00 = right size, below 1 = too narrow):",
              "",
              f"- plain bootstrap: median {np.median(ratio[False]):.3f}, range {ratio[False].min():.3f} to {ratio[False].max():.3f}",
              f"- with small-n fix: median {np.median(ratio[True]):.3f}, range {ratio[True].min():.3f} to {ratio[True].max():.3f}"]
    report = "\n".join(lines)
    print(report)
    suffix = "" if a.shrink == 1 else f"_shrink{a.shrink:g}"
    Path("results").mkdir(exist_ok=True)
    Path(f"results/coverage{suffix}.md").write_text(report + "\n")
    pd.DataFrame(per_model).sort_values("true_rank").to_csv(f"results/coverage_per_model{suffix}.csv",
                                                            index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
