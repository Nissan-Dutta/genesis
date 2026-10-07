"""Every number quoted in the technical note, derived from ``results/summary.json``.

``pgnoise numbers`` writes them as LaTeX macros (``paper/numbers.tex``, used by the LaTeX note) and as a
provenance table (``paper/numbers.md``) that maps each macro to its value and its source in the summary.
"""

from __future__ import annotations

import math
import re
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

Summary = dict


@dataclass(frozen=True)
class Number:
    macro: str
    value: str
    source: str


def _int(x: float) -> str:
    return f"{int(round(x)):,}"


def _round_to(x: float, step: int) -> str:
    return f"{int(step * round(x / step)):,}"


def _f(x: float, dp: int) -> str:
    return f"{x:.{dp}f}"


def _pct(x: float, dp: int = 1) -> str:
    """Percent with half-up rounding on the decimal representation (0.0595 -> 6.0, not 5.9)."""
    return str((Decimal(repr(x)) * 100).quantize(Decimal(1).scaleb(-dp), rounding=ROUND_HALF_UP))


def _sci(x: float, digits: int) -> str:
    mantissa, exp = f"{x:.{digits}e}".split("e")
    return f"{mantissa}e{int(exp)}"


def _short(model: str) -> str:
    return model.replace(" Protein-RAG (16B)", " Protein-RAG")


def _names(models: list[str]) -> str:
    models = [_short(m) for m in models]
    return ", ".join(models[:-1]) + " and " + models[-1] if len(models) > 1 else models[0]


SIM_METHODS = {"Pct": "percentile", "MargSS": "marginal_single_step", "MargSD": "marginal_stepdown",
               "SimSD": "simultaneous_stepdown", "MargSDt": "marginal_stepdown_t"}
SIM_SCENARIOS = {"Cal": "calibrated", "Gauss": "calibrated_gaussian", "Ties": "exact_ties", "Near": "near_ties",
                 "Sep": "separated"}


def _sim(s: Summary, method: str, scenario: str) -> dict:
    return next(r for r in s["simulation"]["summary"] if r["method"] == method and r["scenario"] == scenario)


def _best(s: Summary, scenario: str) -> dict:
    return next(r for r in s["simulation"]["best_set"] if r["scenario"] == scenario)


def _rob(s: Summary, metric: str, scheme: str) -> dict:
    return next(r for r in s["robustness"]["summary"] if r["metric"] == metric and r["scheme"] == scheme)


MEAN_SCHEMES = ["proteingym", "function_group_mean", "uniprot_weighted", "flat_mean"]


def _rank(s: Summary, r: int) -> dict:
    return s["ranks"]["intervals_by_rank"][str(r)]


def _esm(s: Summary, assay_prefix: str, size: str) -> dict:
    return next(r for r in s["esm2"]["rows"] if r["assay"].startswith(assay_prefix) and r["size"] == size)


def _step(s: Summary, assay_prefix: str, smaller: str, larger: str) -> dict:
    return next(r for r in s["esm2"]["size_differences"]
                if r["assay"].startswith(assay_prefix) and r["smaller"] == smaller and r["larger"] == larger)


def _drift(s: Summary, frm: str) -> dict:
    return next(r for r in s["version_drift"] if r["from"] == frm)


def _change_of(s: Summary, frm: str, model: str) -> float:
    for part in _drift(s, frm)["largest_changes"].split("; "):
        name, change = part.rsplit(" ", 1)
        if name == model:
            return float(change)
    raise KeyError(model)


def _spec() -> list[tuple[str, str, Callable[[Summary], str]]]:
    """(macro, source description, value from summary)."""
    rk = lambda s: s["ranks"]  # noqa: E731
    pw = lambda s: s["power"]  # noqa: E731
    cov = lambda s: s["ranks"]["coverage"]  # noqa: E731
    spec: list[tuple[str, str, Callable[[Summary], str]]] = [
        # data and reproduction
        ("pgCommit", "reproduction.proteingym_commit (first 7 characters)", lambda s: s["reproduction"]["proteingym_commit"][:7]),
        ("nModels", "reproduction.n_models", lambda s: str(s["reproduction"]["n_models"])),
        ("nAssays", "reproduction.n_assays", lambda s: str(s["reproduction"]["n_assays"])),
        ("nUnits", "ranks.n_units", lambda s: str(rk(s)["n_units"])),
        ("unitsBinding", "ranks.units_by_group.Binding", lambda s: str(rk(s)["units_by_group"]["Binding"])),
        ("unitsExpression", "ranks.units_by_group.Expression", lambda s: str(rk(s)["units_by_group"]["Expression"])),
        ("nExactAverage", "reproduction.columns.Average_Spearman.n_exact_3dp",
         lambda s: str(s["reproduction"]["columns"]["Average_Spearman"]["n_exact_3dp"])),
        ("nExactRank", "reproduction.columns.rank.n_exact_3dp",
         lambda s: str(s["reproduction"]["columns"]["rank"]["n_exact_3dp"])),
        ("nExactSE", "reproduction.columns.Bootstrap_standard_error_Spearman.n_exact_3dp",
         lambda s: str(s["reproduction"]["columns"]["Bootstrap_standard_error_Spearman"]["n_exact_3dp"])),
        ("nSEOff", "number of reproduction.se_mismatches", lambda s: str(len(s["reproduction"]["se_mismatches"]))),
        ("seOffBoundary", "max over reproduction.se_mismatches of |ours - nearest 3-dp rounding boundary| (4 dp, rounded up)",
         lambda s: _f(math.ceil(1e4 * max(abs(1e3 * ours - (math.floor(1e3 * ours) + 0.5)) / 1e3
                                          for ours, _ in s["reproduction"]["se_mismatches"].values()) - 1e-9) / 1e4, 4)),
        ("nExactOtherMetrics", "min over AUC, MCC, NDCG, Top_recall of robustness.reproduction_other_metrics.average_exact",
         lambda s: str(min(r["average_exact"] for r in s["robustness"]["reproduction_other_metrics"]
                           if r["metric"] != "Spearman"))),
        ("driftEsmOneV", "version_drift[PG_v1.0].largest_changes, ESM-1v (single), absolute",
         lambda s: _f(abs(_change_of(s, "PG_v1.0", "ESM-1v (single)")), 3)),
        ("driftMaxOneZero", "version_drift[PG_v1.0].max_abs_change", lambda s: _f(_drift(s, "PG_v1.0")["max_abs_change"], 3)),
        ("driftMaxOneTwo", "version_drift[PG_v1.1].max_abs_change", lambda s: _f(_drift(s, "PG_v1.1")["max_abs_change"], 3)),
        ("driftSinceOneTwo", "max of version_drift[from PG_v1.2 on].max_abs_change",
         lambda s: _f(max(r["max_abs_change"] for r in s["version_drift"] if r["from"] not in ("PG_v1.0", "PG_v1.1")), 3)),
        ("nBoot", "ranks.n_boot", lambda s: _int(rk(s)["n_boot"])),
        # ranks
        ("topOne", "ranks.intervals_by_rank.1.model", lambda s: _short(_rank(s, 1)["model"])),
        ("topTwo", "ranks.intervals_by_rank.2.model", lambda s: _short(_rank(s, 2)["model"])),
        ("topThree", "ranks.intervals_by_rank.3.model", lambda s: _short(_rank(s, 3)["model"])),
        ("topFour", "ranks.intervals_by_rank.4.model", lambda s: _short(_rank(s, 4)["model"])),
        ("topTen", "ranks.intervals_by_rank.10.model", lambda s: _short(_rank(s, 10)["model"])),
        ("gapOneTwo", "ranks.gap_1_vs_2", lambda s: _f(rk(s)["gap_1_vs_2"], 5)),
        ("pRankOneTop", "ranks.intervals_by_rank.1.p_rank1 (percent)", lambda s: _pct(_rank(s, 1)["p_rank1"], 0)),
        ("pRankOneSecond", "ranks.intervals_by_rank.2.p_rank1 (percent)", lambda s: _pct(_rank(s, 2)["p_rank1"], 0)),
        ("bestSet", "ranks.best_set", lambda s: _names(rk(s)["best_set"])),
        ("nBestSet", "len(ranks.best_set)", lambda s: str(len(rk(s)["best_set"]))),
        ("naiveBestSet", "ranks.naive_best_set", lambda s: _names(rk(s)["naive_best_set"])),
        ("nNeighbourSig", "ranks.neighbours_sig_holm", lambda s: str(rk(s)["neighbours_sig_holm"])),
        ("nNeighbourSigRaw", "ranks.neighbours_sig_uncorrected", lambda s: str(rk(s)["neighbours_sig_uncorrected"])),
        ("zSigA", "ranks.neighbours_sig_z, first pair", lambda s: _f(list(rk(s)["neighbours_sig_z"].values())[0], 1)),
        ("zSigB", "ranks.neighbours_sig_z, second pair", lambda s: _f(list(rk(s)["neighbours_sig_z"].values())[1], 1)),
        ("sigPairA", "ranks.neighbours_sig_z, first key",
         lambda s: _short(list(rk(s)["neighbours_sig_z"])[0]).replace(" > ", " $>$ ")),
        ("sigPairB", "ranks.neighbours_sig_z, second key",
         lambda s: _short(list(rk(s)["neighbours_sig_z"])[1]).replace(" > ", " $>$ ")),
        ("margFourLo", "ranks.intervals_by_rank.4.marg_lo", lambda s: str(_rank(s, 4)["marg_lo"])),
        ("margFourHi", "ranks.intervals_by_rank.4.marg_hi", lambda s: str(_rank(s, 4)["marg_hi"])),
        ("simFourHi", "ranks.intervals_by_rank.4.simul_hi", lambda s: str(_rank(s, 4)["simul_hi"])),
        ("margTenLo", "ranks.intervals_by_rank.10.marg_lo", lambda s: str(_rank(s, 10)["marg_lo"])),
        ("margTenHi", "ranks.intervals_by_rank.10.marg_hi", lambda s: str(_rank(s, 10)["marg_hi"])),
        ("simTenHi", "ranks.intervals_by_rank.10.simul_hi", lambda s: str(_rank(s, 10)["simul_hi"])),
        ("margOneHi", "ranks.intervals_by_rank.1.marg_hi", lambda s: str(_rank(s, 1)["marg_hi"])),
        ("margThreeLo", "ranks.intervals_by_rank.3.marg_lo", lambda s: str(_rank(s, 3)["marg_lo"])),
        ("gapOneThree", "ranks.gap_1_vs_3", lambda s: _f(rk(s)["gap_1_vs_3"], 3)),
        ("seOneThree", "ranks.se_1_vs_3", lambda s: _f(rk(s)["se_1_vs_3"], 4)),
        ("zOneThree", "ranks.gap_1_vs_3 / ranks.se_1_vs_3", lambda s: _f(rk(s)["gap_1_vs_3"] / rk(s)["se_1_vs_3"], 1)),
        ("logoFlipGroups", "function groups g with ranks.logo_top[g] != ranks.logo_top['(none)']",
         lambda s: " or ".join(g for g, m in rk(s)["logo_top"].items() if m != rk(s)["logo_top"]["(none)"])),
        ("logoFlipLeader", "ranks.logo_top for those groups (distinct)",
         lambda s: _names(sorted({m for m in rk(s)["logo_top"].values() if m != rk(s)["logo_top"]["(none)"]}))),
        # Protriever coverage
        ("protScored", "ranks.coverage.incomplete.Protriever", lambda s: str(cov(s)["incomplete"]["Protriever"])),
        ("protMissing", "reproduction.n_assays - ranks.coverage.n_common_assays",
         lambda s: str(s["reproduction"]["n_assays"] - cov(s)["n_common_assays"])),
        ("protMissAct", "ranks.coverage.missing_by_function.Activity", lambda s: str(cov(s)["missing_by_function"]["Activity"])),
        ("protMissExp", "ranks.coverage.missing_by_function.Expression", lambda s: str(cov(s)["missing_by_function"]["Expression"])),
        ("protMissOrg", "ranks.coverage.missing_by_function.OrganismalFitness",
         lambda s: str(cov(s)["missing_by_function"]["OrganismalFitness"])),
        ("protMissStab", "ranks.coverage.missing_by_function.Stability", lambda s: str(cov(s)["missing_by_function"]["Stability"])),
        ("nCommonAssays", "ranks.coverage.n_common_assays", lambda s: str(cov(s)["n_common_assays"])),
        ("othersOnMissing", "ranks.coverage.others_mean_on_missing", lambda s: _f(cov(s)["others_mean_on_missing"], 3)),
        ("othersOnCommon", "ranks.coverage.others_mean_on_common", lambda s: _f(cov(s)["others_mean_on_common"], 3)),
        ("protRank", "ranks.coverage.protriever_common.rank", lambda s: _int(cov(s)["protriever_common"]["rank"])),
        ("protRankCommon", "ranks.coverage.protriever_common.rank_all_common",
         lambda s: _int(cov(s)["protriever_common"]["rank_all_common"])),
        ("mddAtTop", "power.mdd_now.min (headline MDD)", lambda s: _f(pw(s)["mdd_now"]["min"], 3)),
        ("topPublished", "ranks.top", lambda s: _short(rk(s)["top"])),
        ("topAllCommon", "ranks.coverage.top_on_common", lambda s: _short(cov(s)["top_on_common"])),
        ("bestSetAllCommon", "ranks.coverage.best_set_all_common", lambda s: _names(cov(s)["best_set_all_common"])),
        ("nTopTenCommonAssays", "ranks.coverage.n_top10_common_assays", lambda s: str(cov(s)["n_top10_common_assays"])),
        ("topTopTenCommon", "ranks.coverage.top_on_top10_common", lambda s: _short(cov(s)["top_on_top10_common"])),
        ("bestSetTopTenCommon", "ranks.coverage.best_set_top10_common", lambda s: _names(cov(s)["best_set_top10_common"])),
        ("minMargWorst", "simulation.summary marginal_single_step min_model_coverage min over scenarios",
         lambda s: _f(min(r["min_model_coverage"] for r in s["simulation"]["summary"]
                        if r["method"] == "marginal_single_step"), 3)),
        ("minSimulWorst", "simulation.summary simultaneous_stepdown min_model_coverage min over scenarios",
         lambda s: _f(min(r["min_model_coverage"] for r in s["simulation"]["summary"]
                        if r["method"] == "simultaneous_stepdown"), 3)),
        ("topOnCommon", "ranks.coverage.top_on_common", lambda s: _short(cov(s)["top_on_common"])),
        # simulation
        ("simReps", "simulation.n_reps", lambda s: _int(s["simulation"]["n_reps"])),
        ("simBoot", "simulation.n_boot", lambda s: _int(s["simulation"]["n_boot"])),
        ("simModels", "simulation.n_models", lambda s: str(s["simulation"]["n_models"])),
        ("bestSingleMin", "min over single-#1 scenarios of simulation.best_set.best_set_coverage (percent)",
         lambda s: _pct(min(r["best_set_coverage"] for r in s["simulation"]["best_set"] if r["n_true_best"] == 1))),
        ("bestTiesEach", "simulation.best_set[exact_ties].best_set_per_true_best (percent)",
         lambda s: _pct(_best(s, "exact_ties")["best_set_per_true_best"])),
        ("bestTiesAll", "simulation.best_set[exact_ties].best_set_coverage (percent)",
         lambda s: _pct(_best(s, "exact_ties")["best_set_coverage"])),
        ("meanCovMargSSMin", "min over scenarios of simulation.summary[marginal_single_step].mean_coverage",
         lambda s: _f(min(_sim(s, "marginal_single_step", c)["mean_coverage"] for c in SIM_SCENARIOS.values()), 3)),
        ("naiveTiesAll", "simulation.best_set[exact_ties].naive_best_set_coverage (percent)",
         lambda s: _pct(_best(s, "exact_ties")["naive_best_set_coverage"])),
        # power
        ("mddMed", "power.mdd_now.median", lambda s: _f(pw(s)["mdd_now"]["median"], 3)),
        ("mddMin", "power.mdd_now.min", lambda s: _f(pw(s)["mdd_now"]["min"], 3)),
        ("mddMax", "power.mdd_now.max", lambda s: _f(pw(s)["mdd_now"]["max"], 3)),
        ("unitsTenMed", "power.units_for_0.01.median (nearest 10)", lambda s: _round_to(pw(s)["units_for_0.01"]["median"], 10)),
        ("unitsTenMin", "power.units_for_0.01.min (nearest 10)", lambda s: _round_to(pw(s)["units_for_0.01"]["min"], 10)),
        ("unitsTenMax", "power.units_for_0.01.max (nearest 10)", lambda s: _round_to(pw(s)["units_for_0.01"]["max"], 10)),
        ("assaysTenMed", "power.assays_for_0.01_median (nearest 10)", lambda s: _round_to(pw(s)["assays_for_0.01_median"], 10)),
        ("timesTenMed", "power.units_for_0.01.median / ranks.n_units",
         lambda s: _f(pw(s)["units_for_0.01"]["median"] / rk(s)["n_units"], 1)),
        ("unitsFiveMed", "power.units_for_0.005.median (nearest 10)", lambda s: _round_to(pw(s)["units_for_0.005"]["median"], 10)),
        ("unitsFiveMin", "power.units_for_0.005.min (nearest 10)", lambda s: _round_to(pw(s)["units_for_0.005"]["min"], 10)),
        ("unitsFiveMax", "power.units_for_0.005.max (nearest 10)", lambda s: _round_to(pw(s)["units_for_0.005"]["max"], 10)),
        ("assaysFiveMed", "power.assays_for_0.005_median (nearest 10)", lambda s: _round_to(pw(s)["assays_for_0.005_median"], 10)),
        ("timesFiveMed", "power.units_for_0.005.median / ranks.n_units",
         lambda s: _f(pw(s)["units_for_0.005"]["median"] / rk(s)["n_units"], 1)),
        ("powerSimGap", "power.max_abs_sim_minus_analytic", lambda s: _f(pw(s)["max_abs_sim_minus_analytic"], 2)),
        ("sizeToday", "power.size_at_delta0.1x (percent)", lambda s: _pct(pw(s)["size_at_delta0"]["1x"])),
        ("sizeQuarter", "power.size_at_delta0.0.25x (percent)", lambda s: _pct(pw(s)["size_at_delta0"]["0.25x"])),
        # robustness
        ("robBoot", "robustness.n_boot", lambda s: _int(s["robustness"]["n_boot"])),
        ("robPossibleSSA", "union of robustness.summary.best_set over Spearman/AUC/MCC x mean-based schemes",
         lambda s: _names(sorted({m for met in ("Spearman", "AUC", "MCC") for sc in MEAN_SCHEMES
                                  for m in _rob(s, met, sc)["best_set"].split("; ")}))),
        ("robFgmLeader", "robustness.leaders.Spearman/function_group_mean", lambda s: _short(s["robustness"]["leaders"]["Spearman/function_group_mean"])),
        ("ndcgLeader", "robustness.leaders.NDCG/proteingym", lambda s: _short(s["robustness"]["leaders"]["NDCG/proteingym"])),
        ("ndcgOverlap", "robustness.summary[NDCG/proteingym].overlap_with_published_top10",
         lambda s: str(_rob(s, "NDCG", "proteingym")["overlap_with_published_top10"])),
        ("ndcgTauMin", "min over mean-based schemes of robustness.summary[NDCG].kendall_tau_vs_published",
         lambda s: _f(min(_rob(s, "NDCG", sc)["kendall_tau_vs_published"] for sc in MEAN_SCHEMES), 2)),
        ("ndcgTauMax", "max over mean-based schemes of robustness.summary[NDCG].kendall_tau_vs_published",
         lambda s: _f(max(_rob(s, "NDCG", sc)["kendall_tau_vs_published"] for sc in MEAN_SCHEMES), 2)),
        ("ndcgBestMin", "min over mean-based schemes of robustness.summary[NDCG].best_set_size",
         lambda s: str(min(_rob(s, "NDCG", sc)["best_set_size"] for sc in MEAN_SCHEMES))),
        ("ndcgBestMax", "max over mean-based schemes of robustness.summary[NDCG].best_set_size",
         lambda s: str(max(_rob(s, "NDCG", sc)["best_set_size"] for sc in MEAN_SCHEMES))),
        ("topkBestMin", "min over mean-based schemes of robustness.summary[Top_recall].best_set_size",
         lambda s: str(min(_rob(s, "Top_recall", sc)["best_set_size"] for sc in MEAN_SCHEMES))),
        ("topkBestMax", "max over mean-based schemes of robustness.summary[Top_recall].best_set_size",
         lambda s: str(max(_rob(s, "Top_recall", sc)["best_set_size"] for sc in MEAN_SCHEMES))),
        ("medianBest", "robustness.summary[Spearman/median].best_set_size", lambda s: str(_rob(s, "Spearman", "median")["best_set_size"])),
        # ESM-2
        ("esmExact", "esm2.n_exact_3dp", lambda s: str(s["esm2"]["n_exact_3dp"])),
        ("esmCompared", "esm2.n_compared", lambda s: str(s["esm2"]["n_compared"])),
        ("esmMaxDiff", "esm2.max_abs_spearman_diff_vs_published", lambda s: _f(s["esm2"]["max_abs_spearman_diff_vs_published"], 4)),
        ("esmMaxDiffPG", "esm2.max_abs_spearman_diff_vs_proteingym_scores",
         lambda s: _sci(s["esm2"]["max_abs_spearman_diff_vs_proteingym_scores"], 1)),
        ("esmMutDiff", "esm2.max_per_mutant_abs_diff", lambda s: _sci(s["esm2"]["max_per_mutant_abs_diff"], 0)),
        ("esmWtMax", "esm2.max_abs_wt_marginals_shift", lambda s: _f(s["esm2"]["max_abs_wt_marginals_shift"], 3)),
        ("esmWtMed", "median over esm2.rows of |spearman_ours_wt_marginals - spearman_ours|",
         lambda s: _f(statistics.median(abs(r["spearman_ours_wt_marginals"] - r["spearman_ours"]) for r in s["esm2"]["rows"]), 3)),
        ("esmSeMin", "min of esm2.rows.spearman_boot_se", lambda s: _f(min(r["spearman_boot_se"] for r in s["esm2"]["rows"]), 3)),
        ("esmSeMax", "max of esm2.rows.spearman_boot_se", lambda s: _f(max(r["spearman_boot_se"] for r in s["esm2"]["rows"]), 3)),
        ("esmStepsNeg", "esm2.size_steps_negative", lambda s: str(s["esm2"]["size_steps_negative"])),
        ("esmSteps", "esm2.size_steps", lambda s: str(s["esm2"]["size_steps"])),
        ("esmStepsNegSig", "esm2.size_steps_negative_significant", lambda s: str(s["esm2"]["size_steps_negative_significant"])),
        ("esmTcrgPooled", "esm2.rows[TCRG1, 650M].spearman_ours", lambda s: _f(_esm(s, "TCRG1", "650M")["spearman_ours"], 3)),
        ("esmTcrgSingles", "esm2.rows[TCRG1, 650M].spearman_ours_singles",
         lambda s: _f(_esm(s, "TCRG1", "650M")["spearman_ours_singles"], 3)),
        ("esmBclGain", "esm2.size_differences[B2L11, 150M->650M].gain", lambda s: _f(_step(s, "B2L11", "150M", "650M")["gain"], 3)),
        ("esmBclSe", "esm2.size_differences[B2L11, 150M->650M].paired_se",
         lambda s: _f(_step(s, "B2L11", "150M", "650M")["paired_se"], 3)),
        ("esmMinutes", "esm2.total_runtime_s / 60", lambda s: _f(s["esm2"]["total_runtime_s"] / 60, 1)),
    ]
    for mk, method in SIM_METHODS.items():
        for sk, scenario in SIM_SCENARIOS.items():
            spec.append((f"cov{mk}{sk}", f"simulation.summary[{scenario}, {method}].min_model_coverage",
                         lambda s, m=method, c=scenario: _f(_sim(s, m, c)["min_model_coverage"], 3)))
        spec.append((f"joint{mk}Min", f"min over scenarios of simulation.summary[{method}].joint_coverage",
                     lambda s, m=method: _f(min(_sim(s, m, c)["joint_coverage"] for c in SIM_SCENARIOS.values()), 2)))
        spec.append((f"joint{mk}Max", f"max over scenarios of simulation.summary[{method}].joint_coverage",
                     lambda s, m=method: _f(max(_sim(s, m, c)["joint_coverage"] for c in SIM_SCENARIOS.values()), 2)))
        spec.append((f"width{mk}", f"simulation.summary[calibrated, {method}].mean_width_top10",
                     lambda s, m=method: _f(_sim(s, m, "calibrated")["mean_width_top10"], 1)))
    return spec


def numbers(summary: Summary) -> list[Number]:
    return [Number(macro, fn(summary), source) for macro, source, fn in _spec()]


def tex_value(n: Number) -> str:
    """The value as LaTeX: scientific notation becomes $m\\times10^{e}$."""
    return re.sub(r"^(\d(?:\.\d+)?)e(-?\d+)$", r"$\1\\times10^{\2}$", n.value)


def to_tex(nums: list[Number]) -> str:
    lines = ["% Generated by `uv run pgnoise numbers` from results/summary.json. Do not edit by hand."]
    lines += [f"\\newcommand{{\\{n.macro}}}{{{tex_value(n)}}}  % {n.source}" for n in nums]
    return "\n".join(lines) + "\n"


def markdown_value(n: Number) -> str:
    """The value as it is written in the Markdown note."""
    return n.value.replace("$>$", ">")


def to_markdown(nums: list[Number]) -> str:
    lines = ["# Number provenance", "",
             "Generated by `uv run pgnoise numbers` from `results/summary.json`. Every number in the technical note "
             "is one of these values; `tests/test_paper.py` checks that the Markdown note quotes each one.", "",
             "| Macro | Value | Source in `results/summary.json` |", "|---|---|---|"]
    lines += [f"| `\\{n.macro}` | {markdown_value(n)} | {n.source} |" for n in nums]
    return "\n".join(lines) + "\n"


def write(summary: Summary, paper_dir: Path) -> list[Path]:
    nums = numbers(summary)
    paper_dir.mkdir(parents=True, exist_ok=True)
    tex, md = paper_dir / "numbers.tex", paper_dir / "numbers.md"
    tex.write_text(to_tex(nums))
    md.write_text(to_markdown(nums))
    return [tex, md]
