"""Command-line entry point: ``uv run pgnoise <command>``."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import analysis, data, plots, ranks, simulate


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
                               "marg_single_lo", "marg_single_hi", "simul_single_lo", "simul_single_hi"]},
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

    p = sub.add_parser("simulate", help="coverage simulation with known true ranks")
    p.add_argument("--reps", type=int, default=1000)
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--scenarios", nargs="*", choices=list(simulate.SCENARIOS))
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("all", help="download, reproduce, versions, ranks, simulate")
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
    cmd_simulate(argparse.Namespace(**{**vars(args), "boot": 1000, "seed": 2026, "scenarios": None}))


if __name__ == "__main__":
    main()
