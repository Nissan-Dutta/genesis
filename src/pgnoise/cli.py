"""Command-line entry point: ``uv run pgnoise <command>``."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import aggregation, analysis, data, plots, power, ranks, robustness, simulate


def _update_summary(out: Path, section: str, values: dict) -> None:
    path = out / "summary.json"
    current = json.loads(path.read_text()) if path.exists() else {}
    current[section] = values
    path.write_text(json.dumps(current, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def cmd_download(args: argparse.Namespace) -> None:
    paths = data.download(args.data_dir, force=args.force)
    for key, path in paths.items():
        print(f"{key:15s} {path}")
    print(f"ProteinGym commit {data.PROTEINGYM_COMMIT}; checksums verified")


def cmd_reproduce(args: argparse.Namespace) -> None:
    assays, summary = data.load_assays(args.data_dir), data.load_summary(args.data_dir)
    rep = analysis.reproduce_leaderboard(assays, summary, n_boot=args.boot, seed=args.seed)
    report = analysis.reproduction_report(rep)
    tables = args.out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    rep.to_csv(tables / "reproduction_full.csv")
    report.to_csv(tables / "reproduction_report.csv")
    plots.reproduction(rep, args.out / "figures" / "reproduction.png")
    print(report.to_string())
    se_col = "Bootstrap_standard_error_Spearman"
    se_off = rep.index[(np.round(rep[f"{se_col}__ours"], 3) - rep[f"{se_col}__published"]).abs() > 1e-9]
    _update_summary(args.out, "reproduction", {
        "proteingym_commit": data.PROTEINGYM_COMMIT,
        "n_models": len(rep), "n_assays": len(assays.scores),
        "columns": report.to_dict(orient="index"),
        "se_mismatches": {m: [round(float(rep.loc[m, f"{se_col}__ours"]), 5), float(rep.loc[m, f"{se_col}__published"])]
                          for m in se_off},
    })


def cmd_versions(args: argparse.Namespace) -> None:
    summaries = data.download_release_summaries(args.data_dir)
    drift = analysis.version_drift(summaries, reference=list(summaries)[-1])
    (args.out / "tables").mkdir(parents=True, exist_ok=True)
    drift.to_csv(args.out / "tables" / "version_drift.csv", index=False)
    print(drift.to_string(index=False))
    _update_summary(args.out, "version_drift", drift.to_dict(orient="records"))


def cmd_ranks(args: argparse.Namespace) -> None:
    assays = data.load_assays(args.data_dir)
    units = data.to_units(assays)
    tables, figs = args.out / "tables", args.out / "figures"
    tables.mkdir(parents=True, exist_ok=True)

    ru, extra = analysis.rank_uncertainty(units, n_boot=args.boot, seed=args.seed)
    ru.to_csv(tables / "rank_intervals.csv", index_label="model")
    boot = extra["boot"]
    plots.rank_intervals(ru, figs / "rank_intervals_top40.png")

    nb = analysis.neighbour_tests(units, ru, boot)
    nb.to_csv(tables / "neighbour_tests_top20.csv", index=False)

    top20 = list(ru.index[:20])
    idx = [units.models.index(m) for m in top20]
    theta = extra["theta"]
    z = ranks.pairwise_z(theta, boot)[np.ix_(idx, idx)]
    ms = extra["ms"]
    crit = np.quantile(ms.simultaneous_two_sided, 0.95)
    sig = np.abs(z) > crit
    zdf = pd.DataFrame(z, index=top20, columns=top20)
    zdf.to_csv(tables / "pairwise_z_top20.csv")
    plots.pairwise_heatmap(zdf, pd.DataFrame(sig, index=top20, columns=top20), figs / "pairwise_top20.png")

    power = analysis.top_gap_power(units, ru, boot)
    power.to_csv(tables / "top_gap_power.csv", index=False)

    logo = analysis.leave_one_group_out(units)
    logo.to_csv(tables / "leave_one_group_out.csv", index=False)
    logo_ranks = analysis.leave_one_group_out_ranks(units)
    logo_ranks.to_csv(tables / "leave_one_group_out_ranks.csv")
    plots.leave_one_group_out(logo_ranks, figs / "leave_one_group_out.png", list(ru.index[:8]))

    cov = analysis.assay_coverage(assays, units, ru, seed=args.seed + 1)
    cov["common_table"].to_csv(tables / "common_assay_leaderboard.csv", index_label="model")
    cov["missing_assays"].to_csv(tables / "protriever_missing_assays.csv")

    pd.set_option("display.width", 200)
    cols = ["score", "rank", "p_rank1", "pct_lo", "pct_hi", "marg_lo", "marg_hi", "simul_lo", "simul_hi", "in_best_set"]
    print(ru[cols].head(25).to_string(float_format=lambda x: f"{x:.3f}"))
    print(nb.to_string(float_format=lambda x: f"{x:.4f}"))
    print(power.to_string(float_format=lambda x: f"{x:.4f}"))
    print(logo.to_string(index=False))
    print(cov["common_table"].head(12).to_string(float_format=lambda x: f"{x:.3f}"))

    pos = {m: i for i, m in enumerate(ru.index)}
    se_1_3 = float(ranks.pairwise_se(boot)[units.models.index(ru.index[0]), units.models.index(ru.index[2])])
    prot = cov["common_table"].loc["Protriever"] if "Protriever" in cov["common_table"].index else None
    _update_summary(args.out, "ranks", {
        "n_boot": args.boot,
        "n_units": int(units.X.shape[0]),
        "top": ru.index[0],
        "intervals_by_rank": {int(pos[m]) + 1: {"model": m, **{c: int(ru.loc[m, c]) for c in
                              ["pct_lo", "pct_hi", "marg_lo", "marg_hi", "simul_lo", "simul_hi",
                               "marg_stepdown_lo", "marg_stepdown_hi", "simul_single_lo", "simul_single_hi"]},
                              "p_rank1": float(ru.loc[m, "p_rank1"])} for m in ru.index[:20]},
        "best_set": list(ru.index[ru["in_best_set"]]),
        "naive_best_set": list(ru.index[ru["in_naive_best_set"]]),
        "neighbours_sig_uncorrected": int(nb["sig_uncorrected"].sum()),
        "neighbours_sig_holm": int(nb["sig_holm"].sum()),
        "neighbours_sig_pairs_uncorrected": [f"{a} > {b}" for a, b, s in zip(nb.model_hi, nb.model_lo, nb.sig_uncorrected) if s],
        "gap_1_vs_3": float(ru["score"].iloc[0] - ru["score"].iloc[2]),
        "se_1_vs_3": se_1_3,
        "mdd_80_median_top10": float(power["mdd_80pct_power"].median()),
        "sig_threshold_median_top10": float(power["sig_threshold_1.96se"].median()),
        "logo_top": dict(zip(logo["dropped"], [s.split(" (")[0] for s in logo["#1"]])),
        "coverage": {"incomplete": cov["incomplete"], "n_common_assays": cov["n_common_assays"],
                     "missing_by_function": cov["missing_by_function"].to_dict(),
                     "others_mean_on_missing": cov["others_mean_on_missing"],
                     "others_mean_on_common": cov["others_mean_on_common"],
                     "protriever_common": None if prot is None else {k: float(v) for k, v in prot.items()}},
    })


def cmd_robustness(args: argparse.Namespace) -> None:
    tables, figs = args.out / "tables", args.out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    checks = []
    for metric in data.METRICS:
        rep = analysis.reproduce_leaderboard(data.load_assays(args.data_dir, metric),
                                             data.load_summary(args.data_dir, metric), n_boot=2000, metric=metric)
        r = analysis.reproduction_report(rep)
        checks.append({"metric": metric, "average_exact": int(r.loc[f"Average_{metric}", "n_exact_3dp"]),
                       "rank_exact": int(r.loc["rank", "n_exact_3dp"]), "n_models": int(r.loc["rank", "n_models"])})
    assays = data.load_assays(args.data_dir)
    published = pd.read_csv(data.path_for("uniprot_level", args.data_dir)).iloc[-1]
    quirk = robustness.proteingym_uniprot_level_average(assays, pd.read_csv(data.path_for("reference", args.data_dir)))
    clean = pd.Series(aggregation.point_scores(aggregation.make_design(assays), aggregation.SCHEMES["uniprot_weighted"]),
                      index=assays.models)
    cols = [c for c in published.index if c not in ("UniProt_ID", "MSA_Neff_L_category", "Taxon", "Selection Type")]
    pub = published[cols].astype(float).to_numpy()
    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(tables / "reproduction_other_metrics.csv", index=False)

    long = robustness.robustness_grid(n_boot=args.boot, seed=args.seed, data_dir=args.data_dir)
    long.to_csv(tables / "robustness_long.csv", index=False)
    summary = robustness.summarise(long)
    summary.to_csv(tables / "robustness_summary.csv", index=False)
    combos = [(m, "proteingym") for m in data.METRICS] + [("Spearman", s) for s in aggregation.SCHEMES if s != "proteingym"]
    plots.robustness_heatmap(long, figs / "robustness_rank_heatmap.png", combos)
    plots.robustness_grid(summary, figs / "robustness_grid.png")
    pd.set_option("display.width", 250)
    print(checks_df.to_string(index=False))
    print(summary.drop(columns=["top10", "best_set"]).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    mean_based = summary[~summary.scheme.eq("median")]
    _update_summary(args.out, "robustness", {
        "n_boot": args.boot,
        "reproduction_other_metrics": checks,
        "uniprot_level_average_quirk": {
            "published_matches_duplicated_merge": int((np.abs(np.round(quirk.to_numpy(), 3) - pub) < 1e-9).sum()),
            "published_matches_equal_protein_weights": int((np.abs(np.round(clean.to_numpy(), 3) - pub) < 1e-9).sum()),
            "max_abs_diff_vs_equal_weights": float(np.abs(quirk - clean).max()),
        },
        "possible_top1_union_mean_schemes": sorted({m for s in mean_based.best_set for m in s.split("; ")}),
        "leaders": {f"{r.metric}/{r.scheme}": r.top1 for r in summary.itertuples()},
        "summary": summary.to_dict(orient="records"),
    })


def cmd_power(args: argparse.Namespace) -> None:
    units = data.to_units(data.load_assays(args.data_dir))
    res = power.power_tables(units, n_sims=args.sims, seed=args.seed)
    tables = args.out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    res["per_pair"].to_csv(tables / "power_per_pair.csv", index=False)
    res["curves"].to_csv(tables / "power_curves.csv", index=False)
    res["required"].to_csv(tables / "power_required_units.csv", index=False)
    plots.power_curves(res["curves"], res["required"], int(units.X.shape[0]), args.out / "figures" / "power_curves.png")
    print(res["per_pair"].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    med = res["curves"][res["curves"].pair == "median"]
    gap = (med.simulated_power - med.analytic_power).abs()
    print("median pair:", res["median_pair"], "| size at delta=0:", res["size_at_delta0"],
          "| max |sim - analytic| power:", round(float(gap.max()), 3))
    pp = res["per_pair"]
    _update_summary(args.out, "power", {
        "median_pair": res["median_pair"],
        "size_at_delta0": res["size_at_delta0"],
        "max_abs_sim_minus_analytic": float(gap.max()),
        **{f"units_for_{d}": {"median": float(pp[f"units_for_{d}"].median()), "min": float(pp[f"units_for_{d}"].min()),
                              "max": float(pp[f"units_for_{d}"].max())} for d in (0.005, 0.01)},
        **{f"assays_for_{d}_median": float(pp[f"assays_for_{d}"].median()) for d in (0.005, 0.01)},
        "mdd_now": {"median": float(pp["mdd_now"].median()), "min": float(pp["mdd_now"].min()),
                    "max": float(pp["mdd_now"].max())},
        "per_pair": pp.to_dict(orient="records"),
    })


def cmd_simulate(args: argparse.Namespace) -> None:
    units = data.to_units(data.load_assays(args.data_dir))
    cal = simulate.calibrate(units)
    tables, figs = args.out / "tables", args.out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    names = args.scenarios or list(simulate.SCENARIOS)
    summaries, per_model, best = [], {}, []
    for k, name in enumerate(names):
        t0 = time.time()
        res = simulate.run_scenario(cal, simulate.SCENARIOS[name], args.reps, args.boot,
                                    seed=args.seed + k, n_jobs=args.jobs)
        summaries.append(res["summary"])
        per_model[name] = res["per_model"]
        best.append(res["best"])
        res["per_model"].to_csv(tables / f"sim_per_model_{name}.csv", index=False)
        print(f"[{name}] {args.reps} reps in {time.time() - t0:.0f}s")
        print(res["summary"].drop(columns="scenario").to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["best"].items()})
    summary = pd.concat(summaries, ignore_index=True)
    best_df = pd.DataFrame(best)
    summary.to_csv(tables / "sim_summary.csv", index=False)
    best_df.to_csv(tables / "sim_best_set.csv", index=False)
    plots.sim_summary(summary, best_df, figs / "sim_summary.png")
    shown = [n for n in ["calibrated", "exact_ties", "near_ties"] if n in per_model] or names[:3]
    plots.sim_coverage_by_rank({n: per_model[n] for n in shown}, figs / "sim_coverage_by_rank.png")
    _update_summary(args.out, "simulation", {
        "n_reps": args.reps, "n_boot": args.boot, "n_models": len(cal.models),
        "scenarios": {n: simulate.SCENARIOS[n].description for n in names},
        "summary": summary.to_dict(orient="records"), "best_set": best_df.to_dict(orient="records"),
    })


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pgnoise", description="How much of ProteinGym's leaderboard is noise?")
    parser.add_argument("--data-dir", type=Path, default=data.DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=Path, default=Path("results"))
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="fetch pinned ProteinGym CSVs and verify checksums")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("reproduce", help="recompute the published leaderboard and its error bars")
    p.add_argument("--boot", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_reproduce)

    p = sub.add_parser("versions", help="score drift between ProteinGym releases")
    p.set_defaults(func=cmd_versions)

    p = sub.add_parser("ranks", help="rank intervals, neighbour tests, best set, MDD, leave-one-group-out")
    p.add_argument("--boot", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(func=cmd_ranks)

    p = sub.add_parser("robustness", help="rank intervals under other metrics and aggregation schemes")
    p.add_argument("--boot", type=int, default=4000)
    p.add_argument("--seed", type=int, default=11)
    p.set_defaults(func=cmd_robustness)

    p = sub.add_parser("power", help="assays needed to detect gains at the top")
    p.add_argument("--sims", type=int, default=4000)
    p.add_argument("--seed", type=int, default=5)
    p.set_defaults(func=cmd_power)

    p = sub.add_parser("simulate", help="coverage simulation with known true ranks")
    p.add_argument("--reps", type=int, default=1000)
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--scenarios", nargs="*", choices=list(simulate.SCENARIOS))
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("all", help="download, reproduce, versions, ranks, robustness, power, simulate")
    p.add_argument("--reps", type=int, default=1000)
    p.add_argument("--jobs", type=int, default=4)
    p.set_defaults(func=None)

    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command != "all":
        args.func(args)
        return
    cmd_download(argparse.Namespace(**vars(args), force=False))
    cmd_reproduce(argparse.Namespace(**vars(args), boot=10_000, seed=0))
    cmd_versions(args)
    cmd_ranks(argparse.Namespace(**vars(args), boot=10_000, seed=1))
    cmd_robustness(argparse.Namespace(**vars(args), boot=4000, seed=11))
    cmd_power(argparse.Namespace(**vars(args), sims=4000, seed=5))
    cmd_simulate(argparse.Namespace(**{**vars(args), "boot": 1000, "seed": 2026, "scenarios": None}))


if __name__ == "__main__":
    main()
