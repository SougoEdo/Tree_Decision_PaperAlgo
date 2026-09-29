#!/usr/bin/env python3
"""Experiments and figures for the planted-threshold test on daily data (see planted_threshold_test.py).

  python planted_threshold_experiments.py time    one job per trading rate, to estimate the run time
  python planted_threshold_experiments.py run     every experiment, in parallel; saves report/figures/results_daily_threshold.json
  python planted_threshold_experiments.py run-settings   the settings experiment (risk aversion, fee, depth, quartile
                                     thresholds); saves report/figures/results_risk_fee_depth.json
  python planted_threshold_experiments.py run-enhancements   the split-search options (quantile candidates, refinement,
                                     soft splits); saves report/figures/results_split_options.json
  python planted_threshold_experiments.py run-pruning   the pruning hurdle in standard errors; saves report/figures/results_pruning_hurdle.json
  python planted_threshold_experiments.py run-options   sharper smoothing weights and the pruning hurdle below the root;
                                     saves report/figures/results_split_options_2.json
  python planted_threshold_experiments.py run-v1   model v1 (the frozen configuration of 24 September) on 100 datasets;
                                     saves report/figures/results_model_v1.json
  python planted_threshold_experiments.py run-resolution   the threshold against price volatility and feature noise;
                                     saves report/figures/results_resolution.json
  python planted_threshold_experiments.py run-v2   model v1 against the strength of the signal, the price volatility
                                     and the half-life of x, at λ = 1 and 5, with and without fees;
                                     saves report/figures/results_model_v1_sweeps.json
  python planted_threshold_experiments.py run-regime   the regime-dependent volatility model (its own report);
                                     saves report/figures/results_regime_volatility.json
  python planted_threshold_experiments.py run-regime-v1   the regime-dependent volatility model, on model v1;
                                     saves report/figures/results_regime_volatility_v1.json
  python planted_threshold_experiments.py plot    the figures, from the saved results, into report/figures/

Every cell of the experiment is one trading rate (days between decisions) with
one strength of the planted drift (edge = drift / volatility per day) or one
reversion speed k of the feature; a cell holds independent datasets that
differ only by their seed.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from functools import lru_cache
import json
import os
import sys
import time

import numpy as np

from planted_threshold_test import (
    DAYS_PER_YEAR,
    FEATURES,
    Case,
    annual,
    expected_return_curve,
    run_case,
    sign_change,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "report", "figures"))
RESULTS = os.path.join(OUT, "results_daily_threshold.json")

BASE = Case()  # the parameters set in planted_threshold_test.py
STEPS = [1, 2, 5, 10, 21]  # days between two decisions: daily to monthly
EDGES = [0.02, 0.05, 0.10]  # drift / volatility per day
K_VALUES = [
    0.01,
    0.02,
    0.05,
    0.10,
]  # reversion speed per day: half-lives 69, 35, 14, 7 days
DATASETS_PER_CELL = 50
HISTOGRAM_EDGES = [0.05, 0.10]  # these cells get more datasets, for the histograms
DATASETS_PER_HISTOGRAM = 100
RATE_NAMES = {
    1: "daily",
    2: "twice a week",
    5: "weekly",
    10: "every 2 weeks",
    21: "monthly",
}

# A second experiment: the decision problem and the tree are changed, one or two things at a time.
SETTINGS = {
    "base": {},  # the run of the note
    "depth 3": {"max_depth": 3},
    "λ = 5, depth 3": {"risk_aversion": 5.0, "max_depth": 3},
    "λ = 10, depth 3": {"risk_aversion": 10.0, "max_depth": 3},
    "fee 10 bp": {"fee": 0.001},
    "quartiles, depth 3": {"threshold_quantiles": (0.25, 0.5, 0.75), "max_depth": 3},
}
SETTING_STEPS = [2, 5, 10, 21]
SETTING_EDGES = [0.05, 0.10]
DATASETS_PER_SETTING = 40
SETTINGS_RESULTS = os.path.join(OUT, "results_risk_fee_depth.json")
POSITION_BINS = np.linspace(
    -2.5, 2.5, 26
)  # the mean position by bin of x, on the test rows
SCORE_GRID = np.linspace(-2.5, 2.5, 51)  # the tree's score as a function of x

# A third experiment: the options that change how a split is chosen (see SPOPortfolioTree).
DECILES = tuple(float(q) for q in np.round(np.arange(0.1, 1.0, 0.1), 2))
ENHANCEMENTS = {
    "base": {},
    "deciles": {"threshold_quantiles": DECILES, "min_leaf_fraction": 0.10},
    "deciles + refinement": {
        "threshold_quantiles": DECILES,
        "min_leaf_fraction": 0.10,
        "refine_thresholds": True,
        "refine_standard_errors": 1.0,
    },
    "smoothing": {"smoothing": 1.0},
    "deciles + refinement + smoothing": {
        "threshold_quantiles": DECILES,
        "min_leaf_fraction": 0.10,
        "refine_thresholds": True,
        "smoothing": 1.0,
    },
}
ENHANCEMENTS_RESULTS = os.path.join(OUT, "results_split_options.json")

# A fourth experiment: the pruning hurdle, in standard errors of the checking-row regret gain.
PRUNING = {
    "base": {},
    "hurdle 1 s.e.": {"prune_standard_errors": 1.0},
    "hurdle 2 s.e.": {"prune_standard_errors": 2.0},
    "deciles + refinement": {
        "threshold_quantiles": DECILES,
        "min_leaf_fraction": 0.10,
        "refine_thresholds": True,
        "refine_standard_errors": 1.0,
    },
    "deciles + refinement + hurdle 2 s.e.": {
        "threshold_quantiles": DECILES,
        "min_leaf_fraction": 0.10,
        "refine_thresholds": True,
        "refine_standard_errors": 1.0,
        "prune_standard_errors": 2.0,
    },
}
PRUNING_RESULTS = os.path.join(OUT, "results_pruning_hurdle.json")

# A seventh experiment: sharper smoothing weights and the pruning hurdle below the root only.
OPTIONS2 = {
    "base": {},
    "smoothing κ = 0.5": {"smoothing": 0.5},
    "smoothing κ = 0.25": {"smoothing": 0.25},
    "gaussian smoothing κ = 1": {"smoothing": 1.0, "smoothing_kernel": "gaussian"},
    "hurdle 2 s.e. below the root": {
        "prune_standard_errors": 2.0,
        "prune_hurdle_from_depth": 1,
    },
    "deciles + refinement + smoothing κ = 0.25": {
        "threshold_quantiles": DECILES,
        "min_leaf_fraction": 0.10,
        "refine_thresholds": True,
        "smoothing": 0.25,
    },
    "deciles + refinement + hurdle below the root": {
        "threshold_quantiles": DECILES,
        "min_leaf_fraction": 0.10,
        "refine_thresholds": True,
        "refine_standard_errors": 1.0,
        "prune_standard_errors": 2.0,
        "prune_hurdle_from_depth": 1,
    },
}
OPTIONS2_STEPS = SETTING_STEPS  # every trading rate, as in the other option experiments
OPTIONS2_RESULTS = os.path.join(OUT, "results_split_options_2.json")

# Model v1 (24 September 2026): the configuration frozen for the first presentation, validated as a whole on
# 100 datasets (seeds 0-99: the datasets of the base run of Section 4, so the comparison with it is paired).
V1 = {
    "model v1": {
        "max_depth": 3,
        "threshold_quantiles": DECILES,
        "min_leaf_fraction": 0.10,
        "refine_thresholds": True,
        "refine_standard_errors": 1.0,
        "prune_standard_errors": 2.0,
        "prune_hurdle_from_depth": 1,
    },
}
V1_DATASETS = 100
V1_RESULTS = os.path.join(OUT, "results_model_v1.json")

# Model v1 sweeps (28 September 2026): the threshold found against the strength of the signal (the drift, at
# 16% volatility), against the price volatility (at a drift of +-25% a year) and against the half-life of x,
# at λ = 1 and 5, with the 3 bp fee and without fees. Weekly decisions; the half-life sweep also every 2 days
# (with and without fees).
# The cells that E9 already ran (λ = 1, 3 bp, edges 0.05 and 0.10) are reused, not run again.
V1_SETTING = V1["model v1"]
V2_LAMBDAS = [1.0, 5.0]
V2_FEES = [0.0003, 0.0]
V2_EDGES = [0.02, 0.035, 0.05, 0.07, 0.10, 0.15]  # drift / volatility per day: a price Sharpe of 0.3 to 2.4
V2_VOLATILITY_RATIOS = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]  # times the base volatility: 8 to 48% a year
V2_K = [0.35, 0.2, 0.1, 0.05, 0.02, 0.01]  # reversion speed per day: half-lives 2, 3.5, 7, 14, 35, 69 days
V2_DATASETS = 50
V2_RESULTS = os.path.join(OUT, "results_model_v1_sweeps.json")

# A fifth experiment: the resolution of the threshold against the price volatility (at a fixed
# drift of +-25% a year) and against observation noise on the feature (x has standard deviation 1).
RESOLUTION_STEPS = [5, 21]
RESOLUTION_VOLATILITIES = [
    0.004,
    0.006,
    0.008,
    0.01,
    0.015,
    0.02,
    0.03,
    0.04,
]  # per day
RESOLUTION_NOISES = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
DATASETS_PER_RESOLUTION = 60
RESOLUTION_RESULTS = os.path.join(OUT, "results_resolution.json")

# A sixth experiment, its own report: the price volatility also follows a threshold on x.
REGIME = {
    "constant volatility, λ = 1": {},
    "constant volatility, λ = 10": {"risk_aversion": 10.0},
    "volatility threshold at d, λ = 1": {"up_volatility_ratio": 0.5},
    "volatility threshold at d, λ = 10": {
        "up_volatility_ratio": 0.5,
        "risk_aversion": 10.0,
    },
    "volatility threshold at d + 1, λ = 1": {
        "up_volatility_ratio": 0.5,
        "volatility_threshold": 1.0,
    },
    "volatility threshold at d + 1, λ = 10": {
        "up_volatility_ratio": 0.5,
        "volatility_threshold": 1.0,
        "risk_aversion": 10.0,
    },
    "volatility threshold at d + 1, λ = 10, 20-day covariance": {
        "up_volatility_ratio": 0.5,
        "volatility_threshold": 1.0,
        "risk_aversion": 10.0,
        "window": 20,
    },
}
REGIME_RESULTS = os.path.join(OUT, "results_regime_volatility.json")
# The same seven settings on model v1 (28 September 2026): E7 was run on the base pipeline.
REGIME_V1_RESULTS = os.path.join(OUT, "results_regime_volatility_v1.json")


def x_grid(case):
    std = np.sqrt(case.feature_variance)
    return np.linspace(case.mean - 4 * std, case.mean + 4 * std, 321)


@lru_cache(maxsize=None)
def curve_of(k, feature_variance, edge, volatility, d, mean, step, feature_noise=0.0):
    case = replace(
        BASE,
        k=k,
        feature_variance=feature_variance,
        edge=edge,
        volatility=volatility,
        d=d,
        mean=mean,
        step=step,
        feature_noise=feature_noise,
    )
    grid = x_grid(case)
    return grid, expected_return_curve(case, grid, n_paths=10000)


def curve(case):
    return curve_of(
        case.k,
        case.feature_variance,
        case.edge,
        case.volatility,
        case.d,
        case.mean,
        case.step,
        case.feature_noise,
    )


def position_profile(x, weights):
    """Mean weight by bin of x (None where fewer than 3 decisions fall)."""
    which = np.digitize(x, POSITION_BINS) - 1
    return [
        float(weights[which == b].mean()) if np.sum(which == b) >= 3 else None
        for b in range(len(POSITION_BINS) - 1)
    ]


def job(task):
    tag, case, seed, extra = task
    grid, expected = curve(case)
    result = run_case(case, seed, curve=(grid, expected))
    paths = result["paths"]
    x_test = result["rows"].X[case.n_train :, 0]
    return {
        "tag": tag,
        "step": case.step,
        "edge": case.edge,
        "k": case.k,
        "volatility": case.volatility,
        "seed": seed,
        "sign_change": sign_change(grid, expected),
        "feature_noise": case.feature_noise,
        "up_volatility_ratio": case.up_volatility_ratio,
        "volatility_threshold": case.volatility_threshold,
        "risk_aversion": case.risk_aversion,
        "fee": case.fee,
        "window": case.window,
        "grown": result["grown"],
        "kept": result["kept"],
        "n_leaves": len(result["kept"]) + 1,
        "spreads": result["spreads"],
        "score_profile": [
            float(v) for v in result["tree"].predict_returns(SCORE_GRID[:, None])[:, 0]
        ],
        "test_mean": {
            name: float(path.net_returns.mean()) for name, path in paths.items()
        },
        "test_sharpe": {name: float(path.sharpe) for name, path in paths.items()},
        "test_turnover": {
            name: float(path.turnover.mean()) for name, path in paths.items()
        },
        "profile": {
            name: position_profile(x_test, paths[name].weights[:, 0])
            for name in ("tree", "ideal rule", "ideal rule, true variance")
        },
        **extra,
    }


def all_tasks():
    tasks = []
    for step in STEPS:  # the daily jobs are the slowest: submit them first
        for edge in EDGES:
            n = DATASETS_PER_HISTOGRAM if edge in HISTOGRAM_EDGES else DATASETS_PER_CELL
            tasks += [
                ("edge", replace(BASE, step=step, edge=edge), seed, {})
                for seed in range(n)
            ]
        for k in K_VALUES:
            if k != BASE.k:
                tasks += [
                    ("speed", replace(BASE, step=step, k=k), seed, {})
                    for seed in range(DATASETS_PER_CELL)
                ]
    return tasks


def settings_tasks():
    return [
        (
            "settings",
            replace(BASE, step=step, edge=edge, **changes),
            seed,
            {"setting": name},
        )
        for step in SETTING_STEPS
        for name, changes in SETTINGS.items()
        for edge in SETTING_EDGES
        for seed in range(DATASETS_PER_SETTING)
    ]


def setting_tasks(tag, settings, steps=SETTING_STEPS, datasets=DATASETS_PER_SETTING):
    return [
        (tag, replace(BASE, step=step, edge=edge, **changes), seed, {"setting": name})
        for step in steps
        for name, changes in settings.items()
        for edge in SETTING_EDGES
        for seed in range(datasets)
    ]


def enhancement_tasks():
    return setting_tasks("enhancements", ENHANCEMENTS)


def resolution_tasks():
    mu = BASE.mu  # the drift stays at +-25% a year
    tasks = []
    for step in RESOLUTION_STEPS:
        for volatility in RESOLUTION_VOLATILITIES:
            case = replace(BASE, step=step, volatility=volatility, edge=mu / volatility)
            tasks += [
                ("resolution", case, seed, {"axis": "volatility", "value": volatility})
                for seed in range(DATASETS_PER_RESOLUTION)
            ]
        for noise in RESOLUTION_NOISES:
            case = replace(BASE, step=step, feature_noise=noise)
            tasks += [
                ("resolution", case, seed, {"axis": "noise", "value": noise})
                for seed in range(DATASETS_PER_RESOLUTION)
            ]
    return tasks


def price_sharpe(edge):
    """Annualized Sharpe ratio of the regime drift: drift over volatility per day, times sqrt(252)."""
    return float(edge * np.sqrt(DAYS_PER_YEAR))


def annual_volatility(daily):
    return float(daily * np.sqrt(DAYS_PER_YEAR))


def half_life(k):
    return float(np.log(2) / k)


def case_key(case, seed):
    return (
        case.step,
        round(case.edge, 6),
        round(case.k, 6),
        round(case.volatility, 9),
        case.risk_aversion,
        round(case.fee, 8),
        seed,
    )


def record_key(r):
    return (
        r["step"],
        round(r["edge"], 6),
        round(r["k"], 6),
        round(r["volatility"], 9),
        r["risk_aversion"],
        round(r.get("fee", BASE.fee), 8),
        r["seed"],
    )


def v2_cases():
    """The distinct cases of the model v1 sweeps, each with the sweeps it belongs to."""
    cases = {}

    def add(sweep, **changes):
        case = replace(BASE, **V1_SETTING, **changes)
        cases.setdefault(case_key(case, 0)[:-1], (case, []))[1].append(sweep)

    for lam in V2_LAMBDAS:
        for fee in V2_FEES:
            for edge in V2_EDGES:
                add("signal", step=5, edge=edge, risk_aversion=lam, fee=fee)
            for k in V2_K:
                add("half-life", step=5, k=k, risk_aversion=lam, fee=fee)
        for ratio in V2_VOLATILITY_RATIOS:
            daily = ratio * BASE.volatility
            add(
                "volatility",
                step=5,
                volatility=daily,
                edge=round(BASE.mu / daily, 6),
                risk_aversion=lam,
            )
        for fee in V2_FEES:
            for k in V2_K:
                add("half-life", step=2, k=k, risk_aversion=lam, fee=fee)
    return list(cases.values())


def v2_records():
    """The records of the model v1 sweeps: the sweep file plus the E9 cells they share."""
    results = []
    if os.path.exists(V2_RESULTS):
        with open(V2_RESULTS) as handle:
            results = json.load(handle)
    keys = {case_key(case, 0)[:-1] for case, _ in v2_cases()}
    if os.path.exists(V1_RESULTS):
        with open(V1_RESULTS) as handle:
            results += [
                r
                for r in (dict(r, fee=BASE.fee) for r in json.load(handle))
                if record_key(r)[:-1] in keys
            ]
    return results


def v2_cell(results, step, edge, k, volatility, lam, fee):
    key = (step, round(edge, 6), round(k, 6), round(volatility, 9), lam, round(fee, 8))
    return [r for r in results if record_key(r)[:-1] == key]


def run_v2():
    existing = v2_records()
    done = {record_key(r) for r in existing}
    tasks = [
        ("v1 sweeps", case, seed, {"sweeps": sweeps})
        for case, sweeps in v2_cases()
        for seed in range(V2_DATASETS)
        if case_key(case, seed) not in done
    ]
    tasks.sort(key=lambda t: t[1].step)  # the 2-day fits are the slowest: submit them first
    print(
        f"{len(tasks)} fits to run; {len(done)} already available (E9 and earlier runs)",
        flush=True,
    )
    run(tasks, V2_RESULTS, [r for r in existing if r["tag"] == "v1 sweeps"])


def v2_summary(cell, step):
    s = setting_summary(cell, step)
    first = [th for r in cell for f, th in [root_of(r)] if f == "x"]
    s["mean"] = float(np.mean(first)) if first else np.nan
    s["std"] = float(np.std(first, ddof=1)) if len(first) > 1 else np.nan
    # full moves between the asset and cash per year (the turnover counts both legs)
    s["switches"] = {
        name: value / 2 * DAYS_PER_YEAR / step for name, value in s["turnover"].items()
    }
    return s


def v2_sweep_cells():
    """(sweep, label, step, edge, k, volatility, λ, fee) of every cell, in reading order."""
    cells = []
    for lam in V2_LAMBDAS:
        for fee in V2_FEES:
            for edge in V2_EDGES:
                cells.append(
                    ("signal", f"Sharpe {price_sharpe(edge):.2f}", 5, edge, BASE.k, BASE.volatility, lam, fee)
                )
        for ratio in V2_VOLATILITY_RATIOS:
            daily = ratio * BASE.volatility
            cells.append(
                (
                    "volatility",
                    f"{annual_volatility(daily):.0%} a year",
                    5,
                    round(BASE.mu / daily, 6),
                    BASE.k,
                    daily,
                    lam,
                    BASE.fee,
                )
            )
        for step in (5, 2):
            for fee in V2_FEES:
                for k in V2_K:
                    cells.append(
                        ("half-life", f"{half_life(k):.2g} days", step, BASE.edge, k, BASE.volatility, lam, fee)
                    )
    return cells


def print_v2_summary(results):
    print(f"\nmodel v1 sweeps ({V2_DATASETS} datasets per cell; the E9 cells have 100)")
    print(
        f"{'sweep':>10} {'value':>14} {'λ':>3} {'fee':>5} {'rate':>12} {'n':>4} {'on x':>5} {'none':>5} "
        f"{'mean':>6} {'std':>5} {'<=.25':>5} {'2nd':>4} {'leaves':>6} {'tree':>6} {'ideal':>6} {'switches/yr':>14}"
    )
    for sweep, value, step, edge, k, vol, lam, fee in v2_sweep_cells():
        cell = v2_cell(results, step, edge, k, vol, lam, fee)
        if not cell:
            continue
        s = v2_summary(cell, step)
        print(
            f"{sweep:>10} {value:>14} {lam:>3g} {fee * 1e4:>3g}bp {RATE_NAMES[step]:>12} {s['n']:>4d} "
            f"{s['on_x']:>5.0f} {s['none']:>5.0f} {s['mean']:>+6.2f} {s['std']:>5.2f} {s['within']:>5.0f} "
            f"{s['second']:>4.0f} {s['leaves']:>6.2f} {s['sharpe']['tree']:>6.2f} {s['sharpe']['ideal rule']:>6.2f} "
            f"{s['switches']['tree']:>6.1f} ({s['switches']['ideal rule']:.1f})"
        )
    print(
        "\nmean, std: first threshold on x over the datasets that split (target d = 0); switches: full moves "
        "between the asset and cash per year, tree (ideal rule)."
    )


def run(tasks, path, previous=None):
    results, start = list(previous or []), time.perf_counter()
    os.makedirs(OUT, exist_ok=True)
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        futures = [pool.submit(job, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if done % 50 == 0 or done == len(tasks):
                print(
                    f"progress: {done}/{len(tasks)} fits after {time.perf_counter() - start:.0f}s",
                    flush=True,
                )
    with open(path, "w") as handle:
        json.dump(results, handle)
    print(f"saved {path}")


# ----------------------------------------------------------------------------- summaries
def records(results, step, edge=BASE.edge, k=BASE.k):
    """The datasets of one cell."""
    return [
        r for r in results if r["step"] == step and r["edge"] == edge and r["k"] == k
    ]


def root_of(record):
    roots = [(f, th) for depth, f, th in record["kept"] if depth == 0]
    return roots[0] if roots else (None, None)


def annual_sharpe(cell, name, step):
    """Mean over the datasets of the annualized test Sharpe ratio of one strategy."""
    return float(np.nanmean([r["test_sharpe"][name] for r in cell])) * np.sqrt(
        DAYS_PER_YEAR / step
    )


def annual_gain(cell, name, step):
    """Mean test return of a strategy over the tree without splits, per year."""
    return (
        float(
            np.mean(
                [
                    r["test_mean"][name] - r["test_mean"]["tree without splits"]
                    for r in cell
                ]
            )
        )
        * DAYS_PER_YEAR
        / step
    )


def cell_summary(cell, step):
    roots = [root_of(r) for r in cell]
    first = [th for f, th in roots if f == "x"]
    return {
        "n": len(cell),
        "on_x": 100 * sum(f == "x" for f, _ in roots) / len(cell),
        "none": 100 * sum(f is None for f, _ in roots) / len(cell),
        "median": float(np.median(first)) if first else np.nan,
        "quartiles": tuple(np.percentile(first, [25, 75]))
        if first
        else (np.nan, np.nan),
        "sharpe": {
            name: annual_sharpe(cell, name, step)
            for name in ("tree", "ideal rule", "rule at d", "always the asset")
        },
        "captured": (
            100
            * annual_gain(cell, "tree", step)
            / annual_gain(cell, "ideal rule", step)
            if annual_gain(cell, "ideal rule", step) > 0.005
            else np.nan
        ),
    }


def print_summary(results):
    print(
        f"\n{'rate':>14} {'edge':>5} {'k':>5} {'n':>4} {'on x':>5} {'none':>5} {'median':>7} "
        f"{'quartiles':>15} {'tree':>6} {'ideal':>6} {'at d':>6} {'asset':>6} {'captured':>9}"
    )
    print(
        f"{'':>14} {'':>5} {'':>5} {'':>4} {'%':>5} {'%':>5} {'':>7} {'':>15} "
        f"{'annualized test Sharpe ratio':^27} {'%':>9}"
    )
    cells = [(step, edge, BASE.k) for edge in EDGES for step in STEPS]
    cells += [(step, BASE.edge, k) for k in K_VALUES if k != BASE.k for step in STEPS]
    for step, edge, k in cells:
        cell = records(results, step, edge, k)
        if not cell:
            continue
        s = cell_summary(cell, step)
        print(
            f"{RATE_NAMES[step]:>14} {edge:>5g} {k:>5g} {s['n']:>4d} {s['on_x']:>5.0f} {s['none']:>5.0f} "
            f"{s['median']:>+7.2f} {s['quartiles'][0]:>+7.2f}{s['quartiles'][1]:>+8.2f} "
            f"{s['sharpe']['tree']:>6.2f} {s['sharpe']['ideal rule']:>6.2f} "
            f"{s['sharpe']['rule at d']:>6.2f} {s['sharpe']['always the asset']:>6.2f} "
            f"{s['captured']:>9.0f}"
        )
    print(
        "\nrate: days between decisions; on x / none: share of datasets whose first split is on x / "
        "that keep no split;\nmedian and quartiles: first threshold on x over the datasets that "
        "split (x has standard deviation 1, the target is d = 0);\ncaptured: test gain of the tree "
        "over the tree without splits, as a share of the ideal rule's gain."
    )


def setting_records(results, name, step, edge):
    return [
        r
        for r in results
        if r.get("setting") == name and r["step"] == step and r["edge"] == edge
    ]


def setting_summary(cell, step):
    s = cell_summary(cell, step)
    s["leaves"] = float(np.mean([r.get("n_leaves", len(r["kept"]) + 1) for r in cell]))
    s["turnover"] = {
        name: float(np.mean([r["test_turnover"][name] for r in cell]))
        for name in ("tree", "ideal rule")
    }
    first = [th for r in cell for f, th in [root_of(r)] if f == "x"]
    s["within"] = 100 * float(np.mean(np.abs(first) <= 0.25)) if first else np.nan
    root_spreads = [
        r["spreads"][0] for r in cell if r.get("spreads") and root_of(r)[0] == "x"
    ]
    s["spread"] = float(np.mean(root_spreads)) if root_spreads else 0.0
    s["second"] = 100 * float(
        np.mean([any(d >= 1 for d, _, _ in r["kept"]) for r in cell])
    )
    return s


def print_resolution_summary(results):
    for axis_name in ("volatility", "noise"):
        print(
            f"\nresolution experiment, {axis_name} sweep, {DATASETS_PER_RESOLUTION} datasets per point"
        )
        print(
            f"{'value':>8} {'edge':>6} {'rate':>8} {'on x':>5} {'none':>5} {'median |th|':>11} {'quartiles':>15} "
            f"{'tree':>6} {'ideal':>6} {'true var':>8} {'captured':>9}"
        )
        for step in RESOLUTION_STEPS:
            for value, cell in resolution_cells(results, axis_name, step).items():
                s = cell_summary(cell, step)
                first = np.abs([th for r in cell for f, th in [root_of(r)] if f == "x"])
                q = np.percentile(first, [25, 50, 75]) if len(first) else [np.nan] * 3
                true_var = float(
                    np.nanmean(
                        [r["test_sharpe"]["ideal rule, true variance"] for r in cell]
                    )
                ) * np.sqrt(DAYS_PER_YEAR / step)
                print(
                    f"{value:>8.3g} {cell[0]['edge']:>6.3g} {RATE_NAMES[step]:>8} {s['on_x']:>5.0f} {s['none']:>5.0f} "
                    f"{q[1]:>11.2f} {q[0]:>7.2f}{q[2]:>8.2f} {s['sharpe']['tree']:>6.2f} "
                    f"{s['sharpe']['ideal rule']:>6.2f} {true_var:>8.2f} {s['captured']:>9.0f}"
                )


def print_settings_summary(results, names, title, steps=SETTING_STEPS):
    for edge in SETTING_EDGES:
        n_cell = max(
            (len(setting_records(results, n, st, edge)) for n in names for st in steps),
            default=0,
        )
        print(f"\n{title}, edge {edge:g}, {n_cell} datasets per cell")
        print(
            f"{'setting':>32} {'rate':>14} {'leaves':>6} {'on x':>5} {'none':>5} {'median':>7} "
            f"{'quartiles':>15} {'<=.25':>5} {'spread':>6} {'2nd':>4} {'tree':>6} {'ideal':>6} {'captured':>9} {'turnover':>14}"
        )
        for name in names:
            for step in steps:
                cell = setting_records(results, name, step, edge)
                if not cell:
                    continue
                s = setting_summary(cell, step)
                print(
                    f"{name:>32} {RATE_NAMES[step]:>14} {s['leaves']:>6.2f} {s['on_x']:>5.0f} "
                    f"{s['none']:>5.0f} {s['median']:>+7.2f} {s['quartiles'][0]:>+7.2f}{s['quartiles'][1]:>+8.2f} "
                    f"{s['within']:>5.0f} {s['spread']:>6.2f} {s['second']:>4.0f} {s['sharpe']['tree']:>6.2f} "
                    f"{s['sharpe']['ideal rule']:>6.2f} {s['captured']:>9.0f} "
                    f"{s['turnover']['tree']:>6.2f} ({s['turnover']['ideal rule']:.2f})"
                )
    print(
        "\nleaves: mean number of leaves kept; <=.25: share of first thresholds within 0.25 of d; "
        "spread: mean spread of the root split (soft splits); 2nd: share of datasets with a second-level split; turnover: per decision, tree (ideal rule in brackets)."
    )


def print_paired_summary(v1, base):
    """Model v1 minus the base, dataset by dataset (same seeds): mean difference and its standard error."""
    print(
        f"\nmodel v1 minus the base, paired over datasets (annualized test Sharpe ratio; 2 s.e. in brackets)"
    )
    print(
        f"{'edge':>5} {'rate':>14} {'n':>4} {'base':>6} {'v1':>6} {'diff':>7} {'2 s.e.':>8} {'v1 better':>10} {'leaves':>13}"
    )
    for edge in SETTING_EDGES:
        for step in SETTING_STEPS:
            a = {r["seed"]: r for r in base if r["step"] == step and r["edge"] == edge}
            b = {r["seed"]: r for r in v1 if r["step"] == step and r["edge"] == edge}
            seeds = sorted(set(a) & set(b))
            if not seeds:
                continue
            scale = np.sqrt(DAYS_PER_YEAR / step)
            sa = np.array([a[s]["test_sharpe"]["tree"] for s in seeds]) * scale
            sb = np.array([b[s]["test_sharpe"]["tree"] for s in seeds]) * scale
            diff = sb - sa
            error = 2 * diff.std(ddof=1) / np.sqrt(len(seeds))
            leaves_a = np.mean(
                [a[s].get("n_leaves", len(a[s]["kept"]) + 1) for s in seeds]
            )
            leaves_b = np.mean(
                [b[s].get("n_leaves", len(b[s]["kept"]) + 1) for s in seeds]
            )
            print(
                f"{edge:>5g} {RATE_NAMES[step]:>14} {len(seeds):>4d} {sa.mean():>6.2f} {sb.mean():>6.2f} "
                f"{diff.mean():>+7.2f} {error:>8.2f} {100 * np.mean(diff > 0):>9.0f}% "
                f"{leaves_a:>5.2f} -> {leaves_b:.2f}"
            )


# ----------------------------------------------------------------------------- figures
GREEN, RED, BLUE, ORANGE, GREY = "#2e8b57", "#c0392b", "#1f4e9c", "#e67e22", "#555555"
EDGE_COLORS = {0.02: "#f5c99b", 0.05: ORANGE, 0.10: "#a04000"}


def shade_regimes(axis, days, x, d):
    """Light green while x >= d (drift +mu), light red while x < d (drift -mu)."""
    above = x >= d
    edges = np.flatnonzero(np.diff(above.astype(int))) + 1
    for start, stop in zip(np.r_[0, edges], np.r_[edges, len(x)]):
        axis.axvspan(
            days[start],
            days[min(stop, len(x) - 1)],
            color=GREEN if above[start] else RED,
            alpha=0.13,
            lw=0,
        )


def tree_scores(result, x_values):
    """The tree's predicted return of the asset when x takes these values."""
    X = np.tile(
        result["rows"].X[: result["case"].n_train].mean(axis=0), (len(x_values), 1)
    )
    X[:, 0] = x_values
    return result["tree"].predict_returns(X)


def position(result, scores):
    """Weight of the asset chosen by the optimizer, from half-and-half holdings."""
    n = len(scores)
    sigma = np.tile(
        result["rows"].Sigma[: result["case"].n_train].mean(axis=0), (n, 1, 1)
    )
    return result["optimizer"].solve(scores, np.full((n, 2), 0.5), sigma)[:, 0]


def binned_means(x, r, n_bins=16):
    """Mean realised return by quantile bin of x, with its standard error."""
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    which = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n_bins - 1)
    centers, means, errors = [], [], []
    for b in range(n_bins):
        inside = which == b
        if inside.sum() >= 2:
            centers.append(x[inside].mean())
            means.append(r[inside].mean())
            errors.append(r[inside].std(ddof=1) / np.sqrt(inside.sum()))
    return np.array(centers), np.array(means), np.array(errors)


def figure_world(result, plt):
    case = result["case"]
    x, observed, prices = result["series"]
    days = np.arange(case.window, case.window + 2 * DAYS_PER_YEAR)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    for axis in (top, bottom):
        shade_regimes(axis, days, x[days], case.d)
    top.plot(days, x[days], color=BLUE, lw=1.0)
    top.axhline(
        case.d,
        color="black",
        ls="--",
        lw=1,
        label=f"planted threshold d = {case.d:g}, the long-run mean of x",
    )
    top.set_ylabel("feature x")
    top.legend(loc="upper right", fontsize=8)
    top.set_title(
        f"The planted world (two years): drift {case.mu:+.2%} a day while x is above d "
        f"(green), {-case.mu:+.2%} a day below (red); volatility {case.volatility:.0%} a day",
        fontsize=9,
    )
    bottom.semilogy(days, prices[days], color="black", lw=1.0)
    decisions = days[(days - case.window) % case.step == 0]
    bottom.plot(
        decisions,
        np.full(len(decisions), prices[days].min() * 0.98),
        "|",
        color=GREY,
        ms=6,
    )
    bottom.set_ylabel("price of the asset (log scale)")
    bottom.set_xlabel(f"day (grey ticks: one decision every {case.step} days)")
    fig.tight_layout()
    return fig


def figure_learned(result, grid, expected, plt):
    case = result["case"]
    rows = result["rows"]
    train = slice(0, case.n_train)
    crossing = sign_change(grid, expected)
    thresholds = [th for _, f, th in result["kept"] if f == "x"]
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(9, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
    )
    centers, means, errors = binned_means(rows.X[train, 0], rows.R[train, 0])
    top.errorbar(
        centers,
        means,
        yerr=errors,
        fmt="o",
        color=GREY,
        ms=4,
        lw=1,
        capsize=2,
        label=f"realised {case.step}-day returns, mean by bin of x (training decisions, "
        f"± 1 standard error)",
    )
    top.plot(
        grid,
        expected,
        color=BLUE,
        lw=2,
        label=f"true expected {case.step}-day return given x",
    )
    limit = 1.25 * max(np.abs(expected).max(), (np.abs(means) + errors).max())
    scores = tree_scores(result, grid)[:, 0]
    top.plot(
        grid,
        np.clip(scores, -limit, limit),
        color=ORANGE,
        lw=2.2,
        label="the tree: score of each leaf (clipped to the axis: scores are inputs "
        "to the optimizer, not forecasts)",
    )
    top.axhline(0, color="black", lw=0.6)
    top.set_ylabel(
        f"{case.step}-day return (grey, blue) and leaf score (orange), same units"
    )
    top.set_ylim(-limit, limit)
    top.set_title(
        "What the tree should learn, and what it learned (one dataset)", fontsize=10
    )
    ideal = position(result, np.column_stack([expected, np.zeros(len(grid))]))
    bottom.plot(
        grid,
        ideal,
        color=BLUE,
        lw=2,
        label="ideal position (optimizer fed with the true curve)",
    )
    bottom.plot(
        grid,
        position(result, tree_scores(result, grid)),
        color=ORANGE,
        lw=2.2,
        label="position chosen with the tree's leaves",
    )
    bottom.set_ylabel("weight of the asset")
    bottom.set_xlabel("feature x on the decision day")
    bottom.set_ylim(-0.05, 1.08)
    for axis in (top, bottom):
        axis.axvline(case.d, color="black", ls="--", lw=1)
        if crossing is not None:
            axis.axvline(crossing, color=BLUE, ls=":", lw=1.4)
        for th in thresholds:
            axis.axvline(th, color=ORANGE, lw=1)
    if crossing is not None and abs(crossing - case.d) < 0.05:
        top.annotate(
            f"d = {case.d:g}: the sign change\n(x = {crossing:.2f}) is at d",
            (case.d, 0.9 * limit),
            fontsize=8,
            ha="right",
            xytext=(-4, 0),
            textcoords="offset points",
        )
    else:
        top.annotate(
            f"d = {case.d:g}",
            (case.d, 0.9 * limit),
            fontsize=8,
            ha="left",
            xytext=(3, 0),
            textcoords="offset points",
        )
        if crossing is not None:
            top.annotate(
                f"sign change\nx = {crossing:.2f}",
                (crossing, 0.75 * limit),
                fontsize=8,
                ha="right",
                color=BLUE,
                xytext=(-3, 0),
                textcoords="offset points",
            )
    for th in thresholds:
        top.annotate(
            f"tree\n{th:+.2f}",
            (th, -0.9 * limit),
            fontsize=8,
            ha="left",
            color=ORANGE,
            xytext=(3, 0),
            textcoords="offset points",
        )
    top.legend(loc="lower right", fontsize=7.5)
    bottom.legend(loc="center right", fontsize=8)
    top.set_xlim(
        case.mean - 3.2 * result["feature_std"], case.mean + 3.2 * result["feature_std"]
    )
    fig.tight_layout()
    return fig


def figure_decisions(result, plt):
    case = result["case"]
    rows = result["rows"]
    x = result["series"][0]
    shown = 60
    first = case.n_train
    decisions = np.arange(first, first + shown)
    days = np.arange(rows.days[first], rows.days[first + shown])
    fig, (top, middle, bottom) = plt.subplots(
        3, 1, figsize=(9, 8.5), gridspec_kw={"height_ratios": [2, 1.3, 2.2]}
    )
    middle.sharex(top)
    for axis in (top, middle):
        shade_regimes(axis, days, x[days], case.d)
    top.plot(days, x[days], color=BLUE, lw=1)
    top.axhline(case.d, color="black", ls="--", lw=1)
    for th in [th for _, f, th in result["kept"] if f == "x"]:
        top.axhline(th, color=ORANGE, lw=1.2)
    top.set_ylabel("feature x")
    top.set_title(
        f"Test period, never seen in training: the first {shown} decisions "
        f"(regimes in green/red, the tree's threshold in orange)",
        fontsize=10,
    )
    middle.step(
        rows.days[decisions],
        result["paths"]["tree"].weights[:shown, 0],
        where="post",
        color=ORANGE,
        lw=1.6,
        label="tree",
    )
    middle.step(
        rows.days[decisions],
        result["paths"]["ideal rule"].weights[:shown, 0],
        where="post",
        color=BLUE,
        lw=1,
        ls="--",
        label="ideal rule",
    )
    middle.set_ylabel("weight of the asset")
    middle.set_ylim(-0.08, 1.5)
    middle.set_xlabel("day")
    middle.legend(loc="upper right", fontsize=8, ncol=2)
    styles = {
        "tree": (ORANGE, "-", 2),
        "ideal rule": (BLUE, "--", 1.4),
        "rule at d": (GREEN, ":", 1.6),
        "always the asset": ("black", "-", 1),
    }
    years = (rows.days[first:] + case.step - rows.days[first]) / DAYS_PER_YEAR
    for name, (color, ls, lw) in styles.items():
        path = result["paths"][name]
        wealth = np.cumprod(1 + path.net_returns)
        bottom.plot(
            years,
            wealth,
            color=color,
            ls=ls,
            lw=lw,
            label=f"{name} (Sharpe {annual(path, case)['sharpe']:.2f})",
        )
    bottom.axhline(1, color=GREY, lw=0.8)
    bottom.set_ylabel("wealth (start = 1)")
    bottom.set_xlabel("years into the test period")
    bottom.set_title(
        f"The whole test period, {case.test_days / DAYS_PER_YEAR:g} years "
        f"(annualized Sharpe ratios in brackets)",
        fontsize=10,
    )
    bottom.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        fontsize=8,
        ncol=4,
        frameon=False,
    )
    fig.tight_layout()
    return fig


def figure_histograms(results, plt):
    fig, axes = plt.subplots(
        len(STEPS), len(HISTOGRAM_EDGES), figsize=(8.5, 11.5), sharex=True
    )
    bins = np.linspace(-1.5, 1.5, 31)
    for i, step in enumerate(STEPS):
        for j, edge in enumerate(HISTOGRAM_EDGES):
            axis = axes[i, j]
            cell = records(results, step, edge=edge)
            if not cell:
                continue
            found = {
                depth: [
                    th
                    for r in cell
                    for dp, f, th in r["kept"]
                    if f == "x" and dp == depth
                ]
                for depth in (0, 1)
            }
            axis.hist(
                [
                    np.clip(found[0], bins[0], bins[-1]),
                    np.clip(found[1], bins[0], bins[-1]),
                ],
                bins=bins,
                stacked=True,
                color=[ORANGE, "#f5c99b"],
                label=["first split", "second-level split"],
            )
            axis.axvline(
                BASE.d,
                color="black",
                ls="--",
                lw=1.2,
                label=f"planted d = {BASE.d:g}, the mean of x: the target at every rate",
            )
            s_ = cell_summary(cell, step)
            axis.set_title(
                f"{RATE_NAMES[step]} (every {step} d), edge {edge:g} "
                f"({edge * BASE.volatility * DAYS_PER_YEAR:+.0%} a year)\n"
                f"first split on x {s_['on_x']:.0f}%, no split {s_['none']:.0f}%, "
                f"median {s_['median']:+.2f}",
                fontsize=9,
            )
            if j == 0:
                axis.set_ylabel("number of splits")
            if i == len(STEPS) - 1:
                axis.set_xlabel("threshold found on x")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=False)
    n = len(records(results, STEPS[0], edge=HISTOGRAM_EDGES[0]))
    fig.suptitle(
        f"Where the tree puts its thresholds on x (k = {BASE.k:g}, {n} datasets per panel; "
        f"x has standard deviation 1)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.98))
    return fig


def heatmap(axis, values, title, text, cmap, vmin, vmax, xlabels, xlabel, steps=STEPS):
    image = axis.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    axis.set_xticks(
        range(len(xlabels)), xlabels, fontsize=9 if len(xlabels) <= 5 else 7
    )
    axis.set_yticks(
        range(len(steps)),
        [f"{RATE_NAMES[s]}\n(every {s} d)" for s in steps],
        fontsize=9,
    )
    axis.set_xlabel(xlabel, fontsize=9.5)
    axis.set_title(title, fontsize=10)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                dark = cmap != "RdYlGn" and (values[i, j] - vmin) / (vmax - vmin) > 0.6
                axis.text(
                    j,
                    i,
                    text(i, j),
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white" if dark else "black",
                )
    return image


def figure_maps(results, plt, key, columns, xlabels, xlabel):
    """One row per trading rate, one column per value of `key` (edge or k)."""
    shape = (len(STEPS), len(columns))
    on_x, none, captured, tree, ideal = (np.full(shape, np.nan) for _ in range(5))
    for i, step in enumerate(STEPS):
        for j, value in enumerate(columns):
            cell = records(results, step, **{key: value})
            if not cell:
                continue
            s = cell_summary(cell, step)
            on_x[i, j], none[i, j], captured[i, j] = s["on_x"], s["none"], s["captured"]
            tree[i, j], ideal[i, j] = s["sharpe"]["tree"], s["sharpe"]["ideal rule"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9))
    axes = axes.ravel()
    heatmap(
        axes[0],
        on_x,
        "first split on the true feature x\n(% of the datasets)",
        lambda i, j: f"{on_x[i, j]:.0f}",
        "Greens",
        0,
        100,
        xlabels,
        xlabel,
    )
    heatmap(
        axes[1],
        none,
        "no split kept after pruning\n(% of the datasets)",
        lambda i, j: f"{none[i, j]:.0f}",
        "Greys",
        0,
        100,
        xlabels,
        xlabel,
    )
    heatmap(
        axes[2],
        captured,
        "test gain over the tree without splits,\nas % of the ideal "
        "rule's gain (blank: under 0.5% a year to gain)",
        lambda i, j: f"{captured[i, j]:.0f}",
        "RdYlGn",
        -50,
        110,
        xlabels,
        xlabel,
    )
    heatmap(
        axes[3],
        tree,
        "annualized test Sharpe ratio of the tree\n(ideal rule in brackets)",
        lambda i, j: f"{tree[i, j]:.2f}\n({ideal[i, j]:.2f})",
        "Blues",
        0,
        max(1.0, np.nanmax(tree)),
        xlabels,
        xlabel,
    )
    axes[0].set_ylabel("trading rate", fontsize=9.5)
    axes[2].set_ylabel("trading rate", fontsize=9.5)
    fig.tight_layout()
    return fig


def figure_thresholds_by_rate(results, plt):
    fig, axis = plt.subplots(figsize=(8.5, 4.6))
    offsets = np.linspace(-0.27, 0.27, len(EDGES))
    for offset, edge in zip(offsets, EDGES):
        color = EDGE_COLORS[edge]
        for i, step in enumerate(STEPS):
            cell = records(results, step, edge=edge)
            found = [root_of(r)[1] for r in cell if root_of(r)[0] == "x"]
            if found:
                axis.boxplot(
                    found,
                    positions=[i + offset],
                    widths=0.22,
                    showfliers=True,
                    boxprops={"color": color},
                    medianprops={"color": color, "lw": 2},
                    whiskerprops={"color": color},
                    capprops={"color": color},
                    flierprops={"marker": ".", "markeredgecolor": color, "ms": 4},
                )
            axis.text(
                i + offset,
                1.0,
                f"{len(found)}/{len(cell)}",
                ha="center",
                va="bottom",
                fontsize=7,
                color=color,
                transform=axis.get_xaxis_transform(),
            )
        axis.plot(
            [],
            [],
            color=color,
            lw=2,
            label=f"edge {edge:g} ({edge * BASE.volatility * DAYS_PER_YEAR:+.0%} a year)",
        )
    axis.axhline(
        BASE.d,
        color="black",
        ls="--",
        lw=1,
        label=f"planted d = {BASE.d:g}, the mean of x",
    )
    axis.set_xticks(
        range(len(STEPS)), [f"{RATE_NAMES[s]}\n(every {s} d)" for s in STEPS]
    )
    axis.set_xlabel("trading rate (above: datasets whose first split is on x)")
    axis.set_ylabel("first threshold found on x (standard deviation of x = 1)")
    axis.set_ylim(-1.6, 1.6)
    axis.set_title(
        f"Threshold found against the trading rate and the strength of the drift "
        f"(k = {BASE.k:g}, volatility {BASE.volatility:.0%} a day)",
        fontsize=10,
        pad=18,
    )
    axis.legend(fontsize=8, loc="lower left", ncol=2)
    fig.tight_layout()
    return fig


def figure_settings_positions(results, plt, names, step=5):
    """Mean test position by x: the tree (median and quartiles over datasets) against the ideal rule."""
    fig, axes = plt.subplots(
        len(names),
        len(SETTING_EDGES),
        figsize=(8.5, 1.6 * len(names) + 0.8),
        sharex=True,
        sharey=True,
    )
    centers = (POSITION_BINS[:-1] + POSITION_BINS[1:]) / 2
    for j, name in enumerate(names):
        for i, edge in enumerate(SETTING_EDGES):
            axis = axes[j, i]
            cell = setting_records(results, name, step, edge)
            if not cell:
                continue
            tree = np.array(
                [
                    [np.nan if v is None else v for v in r["profile"]["tree"]]
                    for r in cell
                ]
            )
            ideal = np.array(
                [
                    [np.nan if v is None else v for v in r["profile"]["ideal rule"]]
                    for r in cell
                ]
            )
            low, mid, high = np.nanpercentile(tree, [25, 50, 75], axis=0)
            axis.fill_between(
                centers,
                low,
                high,
                color=ORANGE,
                alpha=0.25,
                lw=0,
                label="tree: quartiles over datasets",
            )
            axis.plot(
                centers, mid, color=ORANGE, lw=2, label="tree: median over datasets"
            )
            axis.plot(
                centers,
                np.nanmedian(ideal, axis=0),
                color=BLUE,
                lw=1.6,
                ls="--",
                label="ideal rule: median over datasets",
            )
            axis.axvline(BASE.d, color="black", ls=":", lw=1)
            s = setting_summary(cell, step)
            axis.set_title(
                f"{name}\nedge {edge:g}: {s['leaves']:.1f} leaves, Sharpe "
                f"{s['sharpe']['tree']:.2f} ({s['sharpe']['ideal rule']:.2f})",
                fontsize=8,
            )
            if i == 0:
                axis.set_ylabel("mean weight of the asset")
            if j == len(names) - 1:
                axis.set_xlabel("x on the decision day")
    axes[0, 0].set_ylim(-0.05, 1.05)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle(
        f"Position taken on the test decisions, by x ({RATE_NAMES[step]} decisions, "
        f"{DATASETS_PER_SETTING} datasets per panel)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    return fig


def figure_settings_maps(
    results, plt, names, edge=0.10, title="The settings experiment", steps=SETTING_STEPS
):
    shape = (len(steps), len(names))
    captured, tree, ideal, leaves, turnover, turn_ideal = (
        np.full(shape, np.nan) for _ in range(6)
    )
    for i, step in enumerate(steps):
        for j, name in enumerate(names):
            cell = setting_records(results, name, step, edge)
            if not cell:
                continue
            s = setting_summary(cell, step)
            captured[i, j], leaves[i, j] = s["captured"], s["leaves"]
            tree[i, j], ideal[i, j] = s["sharpe"]["tree"], s["sharpe"]["ideal rule"]
            turnover[i, j], turn_ideal[i, j] = (
                s["turnover"]["tree"],
                s["turnover"]["ideal rule"],
            )
    import textwrap

    labels = [
        "\n".join(
            textwrap.wrap(
                n.replace("volatility", "vol.")
                .replace("refinement", "refin.")
                .replace("smoothing", "smooth."),
                14,
            )
        )
        for n in names
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))
    axes = axes.ravel()
    heatmap(
        axes[0],
        captured,
        "test gain over the tree without splits,\nas % of the ideal rule's gain",
        lambda i, j: f"{captured[i, j]:.0f}",
        "RdYlGn",
        -50,
        110,
        labels,
        "",
        steps,
    )
    heatmap(
        axes[1],
        tree,
        "annualized test Sharpe ratio of the tree\n(ideal rule in brackets)",
        lambda i, j: f"{tree[i, j]:.2f}\n({ideal[i, j]:.2f})",
        "Blues",
        0,
        max(1.0, np.nanmax(tree)),
        labels,
        "",
        steps,
    )
    heatmap(
        axes[2],
        leaves,
        "mean number of leaves kept",
        lambda i, j: f"{leaves[i, j]:.1f}",
        "Oranges",
        1,
        8,
        labels,
        "",
        steps,
    )
    heatmap(
        axes[3],
        turnover,
        "turnover per decision of the tree\n(ideal rule in brackets)",
        lambda i, j: f"{turnover[i, j]:.2f}\n({turn_ideal[i, j]:.2f})",
        "Purples",
        0,
        max(1.0, np.nanmax(turnover)),
        labels,
        "",
        steps,
    )
    for axis in axes[:2]:
        axis.set_ylabel("trading rate", fontsize=9.5)
    n_cell = max(
        (len(setting_records(results, n, st, edge)) for n in names for st in steps),
        default=0,
    )
    fig.suptitle(f"{title} at edge {edge:g} ({n_cell} datasets per cell)", fontsize=11)
    fig.tight_layout()
    return fig


def figure_scores(results, plt, names, step=5, edge=0.10):
    """The tree's score for the asset as a function of x: median and quartiles over datasets."""
    rows_ = (len(names) + 1) // 2
    fig, axes = plt.subplots(
        rows_, 2, figsize=(9, 3.1 * rows_), sharey=True, sharex=True
    )
    axes = np.atleast_1d(axes).ravel()
    for axis in axes[len(names) :]:
        axis.set_visible(False)
    for axis, name in zip(axes, names):
        cell = setting_records(results, name, step, edge)
        if not cell:
            continue
        scores = np.array([r["score_profile"] for r in cell])
        low, mid, high = np.percentile(scores, [25, 50, 75], axis=0)
        axis.fill_between(
            SCORE_GRID,
            low,
            high,
            color=ORANGE,
            alpha=0.25,
            lw=0,
            label="quartiles over datasets",
        )
        axis.plot(SCORE_GRID, mid, color=ORANGE, lw=2, label="median over datasets")
        axis.axhline(0, color="black", lw=0.6)
        axis.axvline(BASE.d, color="black", ls=":", lw=1)
        s = setting_summary(cell, step)
        axis.set_title(
            f"{name}\nmedian threshold {s['median']:+.2f}, spread {s['spread']:.2f}",
            fontsize=9,
        )
        axis.set_xlabel("x on the decision day")
    for axis in axes[0::2]:
        axis.set_ylabel("score of the asset given by the tree")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9, frameon=False)
    fig.suptitle(
        f"What the tree says as a function of x ({RATE_NAMES[step]} decisions, edge {edge:g}, "
        f"{DATASETS_PER_SETTING} datasets per panel)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return fig


def figure_thresholds_by_setting(results, plt, names, step=5):
    """Histograms of the first threshold on x, one row per setting, one column per edge."""
    fig, axes = plt.subplots(
        len(names),
        len(SETTING_EDGES),
        figsize=(8.5, 1.6 * len(names) + 0.8),
        sharex=True,
        sharey="row",
    )
    bins = np.linspace(-1.5, 1.5, 31)
    for j, name in enumerate(names):
        for i, edge in enumerate(SETTING_EDGES):
            axis = axes[j, i]
            cell = setting_records(results, name, step, edge)
            if not cell:
                continue
            first = [th for r in cell for f, th in [root_of(r)] if f == "x"]
            axis.hist(
                np.clip(first, bins[0], bins[-1]),
                bins=bins,
                color=ORANGE,
                label="first split on x",
            )
            axis.axvline(
                BASE.d, color="black", ls="--", lw=1.2, label=f"planted d = {BASE.d:g}"
            )
            s = setting_summary(cell, step)
            axis.set_title(
                f"{name}, edge {edge:g}\non x {s['on_x']:.0f}%, median {s['median']:+.2f} "
                f"[{s['quartiles'][0]:+.2f}, {s['quartiles'][1]:+.2f}], within 0.25: {s['within']:.0f}%",
                fontsize=8,
            )
            if i == 0:
                axis.set_ylabel("datasets")
            if j == len(names) - 1:
                axis.set_xlabel("first threshold on x")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9, frameon=False)
    n_cell = max(
        (
            len(setting_records(results, n, step, e))
            for n in names
            for e in SETTING_EDGES
        ),
        default=0,
    )
    fig.suptitle(
        f"Where the first threshold lands ({RATE_NAMES[step]} decisions, "
        f"{n_cell} datasets per panel)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    return fig


def resolution_cells(results, axis, step):
    """{value: records} along one axis of the resolution experiment."""
    cells = {}
    for r in results:
        if r.get("axis") == axis and r["step"] == step:
            cells.setdefault(r["value"], []).append(r)
    return dict(sorted(cells.items()))


def figure_resolution(results, plt):
    """Found rate, threshold error, Sharpe and captured gain against the price volatility
    (fixed drift) and against the observation noise on the feature, by trading rate."""
    fig, axes = plt.subplots(2, 4, figsize=(13, 7))
    colors = {5: ORANGE, 21: BLUE}
    for i, (axis_name, xlabel) in enumerate(
        (
            ("volatility", "price volatility per day (drift +25% a year)"),
            ("noise", "observation noise on x (x has standard deviation 1)"),
        )
    ):
        for step in RESOLUTION_STEPS:
            cells = resolution_cells(results, axis_name, step)
            if not cells:
                continue
            values = np.array(list(cells))
            summaries = [cell_summary(cell, step) for cell in cells.values()]
            found = [s["on_x"] for s in summaries]
            errors = []
            for cell in cells.values():
                first = np.abs([th for r in cell for f, th in [root_of(r)] if f == "x"])
                errors.append(
                    np.percentile(first, [25, 50, 75]) if len(first) else [np.nan] * 3
                )
            errors = np.array(errors)
            tree = [s["sharpe"]["tree"] for s in summaries]
            ideal = [s["sharpe"]["ideal rule"] for s in summaries]
            captured = [s["captured"] for s in summaries]
            label = RATE_NAMES[step]
            axes[i, 0].plot(values, found, "o-", color=colors[step], label=label)
            axes[i, 1].plot(values, errors[:, 1], "o-", color=colors[step], label=label)
            axes[i, 1].fill_between(
                values, errors[:, 0], errors[:, 2], color=colors[step], alpha=0.2, lw=0
            )
            axes[i, 2].plot(
                values, tree, "o-", color=colors[step], label=f"tree, {label}"
            )
            axes[i, 2].plot(
                values, ideal, "--", color=colors[step], label=f"ideal rule, {label}"
            )
            axes[i, 3].plot(values, captured, "o-", color=colors[step], label=label)
        for j, (title, ylim) in enumerate(
            (
                ("first split on x (% of datasets)", (0, 105)),
                (
                    "distance of the first threshold to d\n(median, quartiles)",
                    (0, None),
                ),
                ("annualized test Sharpe ratio", (None, None)),
                ("test gain captured (% of the ideal rule's)", (-20, 110)),
            )
        ):
            axis = axes[i, j]
            axis.set_title(title, fontsize=9.5)
            axis.set_xlabel(xlabel, fontsize=8.5)
            axis.set_ylim(*ylim)
            if axis_name == "volatility":
                axis.set_xscale("log")
                axis.set_xticks(
                    RESOLUTION_VOLATILITIES,
                    [f"{v:.1%} ({BASE.mu / v:.2g})" for v in RESOLUTION_VOLATILITIES],
                    fontsize=7,
                    rotation=45,
                    ha="right",
                )
                axis.minorticks_off()
            if j == 2:
                axis.legend(fontsize=7)
            elif i == 0 and j == 0:
                axis.legend(fontsize=8)
    axes[0, 0].set_ylabel("price volatility sweep", fontsize=10)
    axes[1, 0].set_ylabel("observation noise sweep", fontsize=10)
    fig.suptitle(
        f"Resolution of the threshold ({DATASETS_PER_RESOLUTION} datasets per point; "
        f"in brackets: the edge, drift over volatility per day)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def figure_regime_second_splits(results, plt, names, step=5, edge=0.10):
    """All kept thresholds below the root, per setting, with the drift and volatility thresholds."""
    fig, axes = plt.subplots(
        len(names), 1, figsize=(7.5, 1.4 * len(names) + 0.6), sharex=True
    )
    bins = np.linspace(-2.5, 2.5, 51)
    for axis, name in zip(np.atleast_1d(axes), names):
        cell = setting_records(results, name, step, edge)
        if not cell:
            continue
        second = [th for r in cell for d, f, th in r["kept"] if d >= 1]
        axis.hist(
            np.clip(second, bins[0], bins[-1]),
            bins=bins,
            color="#f5c99b",
            label="second-level thresholds",
        )
        first = [th for r in cell for f, th in [root_of(r)] if f == "x"]
        axis.hist(
            np.clip(first, bins[0], bins[-1]),
            bins=bins,
            color=ORANGE,
            alpha=0.7,
            label="first threshold",
        )
        axis.axvline(BASE.d, color="black", ls="--", lw=1.2, label="drift threshold d")
        d_sigma = cell[0]["volatility_threshold"]
        if cell[0]["up_volatility_ratio"] != 1:
            axis.axvline(
                BASE.d if d_sigma is None else d_sigma,
                color=RED,
                ls=":",
                lw=1.6,
                label="volatility threshold",
            )
        share = 100 * np.mean(
            [
                any(
                    abs(th - (BASE.d if d_sigma is None else d_sigma)) <= 0.25
                    for d, f, th in r["kept"]
                    if d >= 1
                )
                for r in cell
            ]
        )
        axis.set_title(
            f"{name}\n{len(second)} second-level splits in {len(cell)} datasets; "
            f"{share:.0f}% of the datasets have one within 0.25 of the volatility threshold",
            fontsize=8,
        )
        axis.set_ylabel("splits")
    np.atleast_1d(axes)[-1].set_xlabel("threshold on x")
    handles, labels = np.atleast_1d(axes)[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle(
        f"Where the splits land ({RATE_NAMES[step]} decisions, edge {edge:g})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return fig


def figure_split_shares(results, plt, names, steps=SETTING_STEPS):
    """Per setting and trading rate: first split on x, no split kept, a second-level split kept (% of datasets)."""
    keys = [
        ("on_x", "first split on x"),
        ("none", "no split kept"),
        ("second", "second-level split kept"),
    ]
    fig, axes = plt.subplots(
        len(keys), len(SETTING_EDGES), figsize=(9, 3.0 * len(keys)), sharey=True
    )
    width = 0.8 / len(names)
    for i, edge in enumerate(SETTING_EDGES):
        for j, (key, label) in enumerate(keys):
            axis = axes[j, i]
            for k, name in enumerate(names):
                values = [
                    setting_summary(setting_records(results, name, step, edge), step)[
                        key
                    ]
                    if setting_records(results, name, step, edge)
                    else np.nan
                    for step in steps
                ]
                axis.bar(
                    np.arange(len(steps)) + (k - (len(names) - 1) / 2) * width,
                    values,
                    width,
                    label=name,
                    color=plt.cm.Oranges(0.3 + 0.6 * k / max(1, len(names) - 1)),
                )
            axis.set_xticks(
                range(len(steps)), [RATE_NAMES[s] for s in steps], fontsize=8
            )
            axis.set_title(f"{label} (edge {edge:g})", fontsize=9.5)
            if i == 0:
                axis.set_ylabel("% of datasets")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8.5, frameon=False)
    n_cell = max(
        (
            len(setting_records(results, n, st, e))
            for n in names
            for st in steps
            for e in SETTING_EDGES
        ),
        default=0,
    )
    fig.suptitle(f"What the trees keep ({n_cell} datasets per cell)", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    return fig


def v2_lines(results, lam, fee, cells):
    """(x, summaries, cells) of the cells [(x, step, edge, k, volatility)] that have records."""
    xs, summaries, kept = [], [], []
    for x, step, edge, k, vol in cells:
        cell = v2_cell(results, step, edge, k, vol, lam, fee)
        if cell:
            xs.append(x)
            summaries.append(v2_summary(cell, step))
            kept.append((x, step, edge, k, vol))
    return np.array(xs), summaries, kept


V2_PANELS = [
    ("mean", "first threshold on x: mean ± std over the\ndatasets that split (planted d = 0, std of x = 1)"),
    ("within", "first threshold within 0.25 of d\n(% of datasets)"),
    ("none", "no split kept after pruning\n(% of datasets)"),
    ("second", "a second-level split kept\n(% of datasets)"),
]


def figure_v2_recovery(results, plt, keys=None, figsize=(14.5, 4.2), font=9):
    """The first threshold against the Sharpe ratio of the regime drift: the drift swept at 16% volatility
    (lines), the volatility swept at a drift of +-25% a year (hollow squares); λ = 1 and 5, with and without fees.
    keys selects the panels (default: all four); the deck uses two panels at a larger font."""
    colors, styles = {1.0: ORANGE, 5.0: BLUE}, {0.0003: "-", 0.0: "--"}
    panels = [p for p in V2_PANELS if keys is None or p[0] in keys]
    plt.rc("font", size=font)
    fig, axes = plt.subplots(1, len(panels), figsize=figsize)
    axes = np.atleast_1d(axes)
    signal = [(price_sharpe(e), 5, e, BASE.k, BASE.volatility) for e in V2_EDGES]
    volatility = []
    for ratio in V2_VOLATILITY_RATIOS:
        daily = ratio * BASE.volatility
        edge = round(BASE.mu / daily, 6)
        volatility.append((price_sharpe(edge), 5, edge, BASE.k, daily))
    for lam in V2_LAMBDAS:
        for fee in V2_FEES:
            xs, summaries, _ = v2_lines(results, lam, fee, signal)
            if not len(xs):
                continue
            label = (
                f"λ = {lam:g}, fee {fee * 1e4:g} bp"
                if keys is not None
                else f"λ = {lam:g}, fee {fee * 1e4:g} bp: drift ±5 to ±38% a year at 16% volatility"
            )
            for axis, (key, _) in zip(axes, panels):
                values = [s[key] for s in summaries]
                if key == "mean":
                    axis.errorbar(
                        xs,
                        values,
                        yerr=[s["std"] for s in summaries],
                        color=colors[lam],
                        ls=styles[fee],
                        marker="o",
                        ms=4,
                        capsize=2,
                        lw=1.4,
                        label=label,
                    )
                else:
                    axis.plot(xs, values, color=colors[lam], ls=styles[fee], marker="o", ms=4, lw=1.4, label=label)
        xs, summaries, kept = v2_lines(results, lam, BASE.fee, volatility)
        if not len(xs):
            continue
        label = (
            f"λ = {lam:g}, volatility sweep"
            if keys is not None
            else f"λ = {lam:g}, fee 3 bp: volatility 8 to 48% a year at a drift of ±25% a year"
        )
        for axis, (key, _) in zip(axes, panels):
            values = [s[key] for s in summaries]
            if key == "mean":
                axis.errorbar(
                    xs,
                    values,
                    yerr=[s["std"] for s in summaries],
                    color=colors[lam],
                    marker="s",
                    mfc="white",
                    ms=6,
                    capsize=2,
                    ls="none",
                    label=label,
                )
            else:
                axis.plot(xs, values, color=colors[lam], marker="s", mfc="white", ms=6, ls="none", label=label)
            if lam == V2_LAMBDAS[0]:
                for x, value, (_, _, _, _, daily) in zip(xs, values, kept):
                    axis.annotate(
                        f"{annual_volatility(daily):.0%}",
                        (x, value),
                        fontsize=6.5,
                        color=GREY,
                        xytext=(4, 4),
                        textcoords="offset points",
                    )
    ticks = [price_sharpe(e) for e in V2_EDGES]
    for axis, (key, title) in zip(axes, panels):
        axis.set_xscale("log")
        axis.set_xticks(ticks, [f"{t:.2f}" for t in ticks], fontsize=font - 1)
        axis.minorticks_off()
        axis.set_xlabel("annualized Sharpe ratio of the regime drift", fontsize=font - 0.5)
        axis.set_title(title, fontsize=font)
        top = axis.secondary_xaxis("top", functions=(lambda s: s * 15.87, lambda d: d / 15.87))
        top.set_xlabel("drift a year at 16% volatility (%)", fontsize=font - 1.5)
        top.set_xticks([5, 10, 25, 38], [f"±{d:g}" for d in (5, 10, 25, 38)], fontsize=font - 2)
        top.minorticks_off()
        if key == "mean":
            axis.axhline(BASE.d, color="black", ls="--", lw=0.8)
            axis.set_ylim(-1.0, 1.0)
        else:
            axis.set_ylim(0, 105)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=3 if len(panels) > 2 else 2, fontsize=font - 1, frameon=False
    )
    if len(panels) > 2:
        fig.suptitle(
            f"Model v1, weekly decisions, half-life of x 35 days: recovery of the planted threshold against the "
            f"strength of the signal ({V2_DATASETS} datasets per point; the E9 points 100)",
            fontsize=font + 1.5,
        )
    fig.tight_layout(rect=(0, 0.1 if len(panels) > 2 else 0.22, 1, 0.95 if len(panels) > 2 else 1))
    plt.rc("font", size=9)
    return fig


def figure_v2_second_splits(results, plt):
    """Where the thresholds land, weekly decisions, drift +-25% a year, 16% volatility, half-life 35 days."""
    fig, axes = plt.subplots(
        len(V2_LAMBDAS), len(V2_FEES), figsize=(9.5, 2.3 * len(V2_LAMBDAS) + 0.9), sharex=True, sharey=True
    )
    bins = np.linspace(-2.5, 2.5, 51)
    for i, lam in enumerate(V2_LAMBDAS):
        for j, fee in enumerate(V2_FEES):
            axis = axes[i, j]
            cell = v2_cell(results, 5, BASE.edge, BASE.k, BASE.volatility, lam, fee)
            if not cell:
                continue
            first = [th for r in cell for f, th in [root_of(r)] if f == "x"]
            second = [th for r in cell for d, f, th in r["kept"] if d >= 1]
            axis.hist(
                [np.clip(first, bins[0], bins[-1]), np.clip(second, bins[0], bins[-1])],
                bins=bins,
                stacked=True,
                color=[ORANGE, "#f5c99b"],
                label=["first threshold", "second-level thresholds"],
            )
            axis.axvline(BASE.d, color="black", ls="--", lw=1.2, label="planted d = 0")
            with_second = 100 * np.mean([any(d >= 1 for d, _, _ in r["kept"]) for r in cell])
            tails = (
                f"{100 * np.mean(np.abs(second) > 1):.0f}% of them beyond ±1 (the tails of x)"
                if second
                else "none"
            )
            axis.set_title(
                f"λ = {lam:g}, fee {fee * 1e4:g} bp ({len(cell)} datasets): {with_second:.0f}% keep a second-level "
                f"split;\n{len(second)} such thresholds, {tails}",
                fontsize=8,
            )
            if j == 0:
                axis.set_ylabel("splits")
            if i == len(V2_LAMBDAS) - 1:
                axis.set_xlabel("threshold on x (std of x = 1)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8.5, frameon=False)
    fig.suptitle(
        "Model v1: where the second-level thresholds land (weekly, drift ±25% a year, 16% volatility, "
        "half-life 35 days)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    return fig


def figure_v2_halflife(results, plt, lambdas=None, keys=None, figsize=None, font=9):
    """The tree against the half-life of x: threshold, recovery, Sharpe and trading, one row per λ.
    keys selects the panels (default: all four); the deck uses λ = 1 and three panels at a larger font."""
    lambdas = V2_LAMBDAS if lambdas is None else list(lambdas)
    series = [  # label, step, fee, color, line style
        ("weekly, fee 3 bp", 5, 0.0003, ORANGE, "-"),
        ("weekly, no fee", 5, 0.0, ORANGE, "--"),
        ("every 2 days, fee 3 bp", 2, 0.0003, BLUE, "-"),
        ("every 2 days, no fee", 2, 0.0, BLUE, "--"),
    ]
    all_panels = [
        ("mean", "first threshold on x: mean ± std\n(planted d = 0, std of x = 1)"),
        ("within", "first threshold within 0.25 of d (%)\n(dotted: no split kept)"),
        ("sharpe", "annualized test Sharpe ratio\n(grey: the ideal rule)"),
        ("switches", "full moves between asset and cash\nper year (grey: the ideal rule)"),
    ]
    panels = [p for p in all_panels if keys is None or p[0] in keys]
    if figsize is None:
        figsize = (3.6 * len(panels) + 0.5, 3.5 * len(lambdas) + 0.6)
    plt.rc("font", size=font)
    fig, axes = plt.subplots(len(lambdas), len(panels), figsize=figsize, sharex=True)
    axes = np.atleast_2d(axes).reshape(len(lambdas), len(panels))
    lives = [half_life(k) for k in V2_K]
    for i, lam in enumerate(lambdas):
        for label, step, fee, color, ls in series:
            cells = [(half_life(k), step, BASE.edge, k, BASE.volatility) for k in V2_K]
            xs, summaries, _ = v2_lines(results, lam, fee, cells)
            if not len(xs):
                continue
            for axis, (key, _) in zip(axes[i], panels):
                if key == "mean":
                    axis.errorbar(
                        xs,
                        [s["mean"] for s in summaries],
                        yerr=[s["std"] for s in summaries],
                        color=color,
                        ls=ls,
                        marker="o",
                        ms=4,
                        capsize=2,
                        lw=1.4,
                        label=label,
                    )
                elif key == "within":
                    axis.plot(xs, [s["within"] for s in summaries], color=color, ls=ls, marker="o", ms=4, lw=1.4, label=label)
                    axis.plot(xs, [s["none"] for s in summaries], color=color, ls=":", marker=".", ms=3, lw=1)
                else:
                    axis.plot(xs, [s[key]["tree"] for s in summaries], color=color, ls=ls, marker="o", ms=4, lw=1.4, label=label)
                    axis.plot(xs, [s[key]["ideal rule"] for s in summaries], color=GREY, ls=ls, lw=1)
        for axis, (key, title) in zip(axes[i], panels):
            axis.set_xscale("log")
            axis.set_xticks(lives, [f"{h:.2g}" for h in lives])
            axis.minorticks_off()
            if i == 0:
                axis.set_title(title, fontsize=font)
            if i == len(lambdas) - 1:
                axis.set_xlabel("half-life of x (days)", fontsize=font - 0.5)
            if key == "mean":
                axis.axhline(BASE.d, color="black", ls="--", lw=0.8)
                axis.set_ylim(-1.0, 1.0)
            elif key == "within":
                axis.set_ylim(0, 105)
        if len(lambdas) > 1:
            axes[i, 0].set_ylabel(f"λ = {lam:g}", fontsize=font + 1.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=font, frameon=False)
    if len(lambdas) > 1:
        fig.suptitle(
            f"Model v1 against the half-life of x (drift ±25% a year, 16% volatility; {V2_DATASETS} datasets per point)",
            fontsize=font + 1.5,
        )
    fig.tight_layout(rect=(0, 0.06 if len(lambdas) > 1 else 0.14, 1, 0.95 if len(lambdas) > 1 else 1))
    plt.rc("font", size=9)
    return fig



# ----------------------------------------------------------------------------- deck figures (model v1: E9, E10)
# One figure per frame of the short deck: two or three panels, at most three curves per panel, distinct colours.
DECK_FONT = 14
LAMBDA_COLORS = {1.0: ORANGE, 5.0: BLUE}
RATE_COLORS = {5: ORANGE, 2: BLUE}
LINE = {"marker": "o", "ms": 6, "lw": 2.2}


def deck_start(plt, cols, width=10.5, height=4.6):
    plt.rc("font", size=DECK_FONT)
    fig, axes = plt.subplots(1, cols, figsize=(width, height))
    return fig, np.atleast_1d(axes)


def deck_end(fig, plt, axis, ncol=3, bottom=0.17):
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=ncol, fontsize=DECK_FONT - 1, frameon=False)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    plt.rc("font", size=9)
    return fig


def signal_axis(axis):
    ticks = [price_sharpe(e) for e in V2_EDGES]
    axis.set_xscale("log")
    axis.set_xticks(ticks, [f"{t:.1f}" for t in ticks])
    axis.minorticks_off()
    axis.set_xlabel("Sharpe ratio of the regime drift (annualized)")


def halflife_axis(axis):
    lives = [half_life(k) for k in V2_K]
    axis.set_xscale("log")
    axis.set_xticks(lives, [f"{h:.2g}" for h in lives])
    axis.minorticks_off()
    axis.set_xlabel("half-life of the feature x (days)")


def signal_series(results, lam, fee):
    return v2_lines(results, lam, fee, [(price_sharpe(e), 5, e, BASE.k, BASE.volatility) for e in V2_EDGES])


def volatility_series(results, lam):
    cells = []
    for ratio in V2_VOLATILITY_RATIOS:
        daily = ratio * BASE.volatility
        cells.append((100 * annual_volatility(daily), 5, round(BASE.mu / daily, 6), BASE.k, daily))
    return v2_lines(results, lam, BASE.fee, cells)


def halflife_series(results, lam, step, fee):
    return v2_lines(results, lam, fee, [(half_life(k), step, BASE.edge, k, BASE.volatility) for k in V2_K])


def draw_mean(axis, xs, summaries, color, label):
    axis.errorbar(
        xs, [s["mean"] for s in summaries], yerr=[s["std"] for s in summaries], color=color, capsize=3, label=label, **LINE
    )
    axis.axhline(BASE.d, color="black", ls=":", lw=1)
    axis.set_ylabel("first threshold on x")
    axis.set_ylim(-0.8, 1.0)


def draw_hist(axis, values, bins, color, label):
    """The deck's histogram: a solid outline with a transparent fill, clipped to the bins."""
    clipped = np.clip(values, bins[0], bins[-1]) if len(values) else values
    axis.hist(clipped, bins=bins, histtype="step", lw=2.4, color=color, label=label)
    axis.hist(clipped, bins=bins, color=color, alpha=0.18, lw=0)


def draw_value(axis, xs, summaries, key, color, label, sub=None):
    axis.plot(xs, [s[key] if sub is None else s[key][sub] for s in summaries], color=color, label=label, **LINE)


def figure_deck_signal_threshold(results, plt):
    fig, (left, right) = deck_start(plt, 2)
    for lam in V2_LAMBDAS:
        xs, summaries, _ = signal_series(results, lam, BASE.fee)
        if len(xs):
            draw_mean(left, xs, summaries, LAMBDA_COLORS[lam], f"λ = {lam:g}")
            draw_value(right, xs, summaries, "within", LAMBDA_COLORS[lam], f"λ = {lam:g}")
    left.set_title("first threshold: mean ± std")
    right.set_title("first threshold within 0.25 of d (%)")
    right.set_ylabel("datasets (%)")
    right.set_ylim(0, 100)
    for axis in (left, right):
        signal_axis(axis)
    return deck_end(fig, plt, left, ncol=2)


def figure_deck_signal_sharpe(results, plt):
    fig, axes = deck_start(plt, 2)
    for axis, lam in zip(axes, V2_LAMBDAS):
        xs, summaries, _ = signal_series(results, lam, BASE.fee)
        if len(xs):
            draw_value(axis, xs, summaries, "sharpe", ORANGE, "tree, fee 3 bp", "tree")
            draw_value(axis, xs, summaries, "sharpe", GREY, "ideal rule, fee 3 bp", "ideal rule")
        xs, summaries, _ = signal_series(results, lam, 0.0)
        if len(xs):
            draw_value(axis, xs, summaries, "sharpe", GREEN, "tree, no fee", "tree")
        axis.set_title(f"λ = {lam:g}")
        axis.set_ylabel("annualized test Sharpe ratio")
        signal_axis(axis)
    return deck_end(fig, plt, axes[0])


def figure_deck_signal_splits(results, plt):
    fig, (left, right) = deck_start(plt, 2)
    for lam in V2_LAMBDAS:
        xs, summaries, _ = signal_series(results, lam, BASE.fee)
        if len(xs):
            draw_value(left, xs, summaries, "none", LAMBDA_COLORS[lam], f"λ = {lam:g}")
            draw_value(right, xs, summaries, "second", LAMBDA_COLORS[lam], f"λ = {lam:g}")
    left.set_title("trees that keep no split at all")
    right.set_title("trees that keep a second-level split")
    left.set_ylim(0, 100)
    right.set_ylim(0, 30)
    for axis in (left, right):
        axis.set_ylabel("datasets (%)")
        signal_axis(axis)
    return deck_end(fig, plt, left, ncol=2)


def figure_deck_volatility(results, plt, lam=1.0):
    """The drift sweep (16% volatility) and the volatility sweep (drift +-25% a year) on the same axis, the
    Sharpe ratio of the regime drift: threshold, recovery and Sharpe of the tree; the volatility points are
    labelled with the annual volatility."""
    fig, axes = deck_start(plt, 3, width=13)
    drift = signal_series(results, lam, BASE.fee)
    cells = []
    for ratio in V2_VOLATILITY_RATIOS:
        daily = ratio * BASE.volatility
        edge = round(BASE.mu / daily, 6)
        cells.append((price_sharpe(edge), 5, edge, BASE.k, daily))
    vol = v2_lines(results, lam, BASE.fee, cells)
    for (xs, summaries, kept), color, label in (
        (drift, ORANGE, "drift swept, volatility 16% a year"),
        (vol, BLUE, "volatility swept from 8 to 48% a year, drift ±25% a year"),
    ):
        if not len(xs):
            continue
        draw_mean(axes[0], xs, summaries, color, label)
        draw_value(axes[1], xs, summaries, "within", color, label)
        draw_value(axes[2], xs, summaries, "sharpe", color, label, "tree")
    axes[0].set_title("first threshold\nmean ± std")
    axes[0].set_ylabel("threshold on x")
    axes[1].set_title("first threshold\nwithin 0.25 of d")
    axes[1].set_ylabel("datasets (%)")
    axes[1].set_ylim(0, 100)
    axes[2].set_title("test Sharpe of the tree\n(annualized)")
    axes[2].set_ylabel("Sharpe ratio")
    ticks = [0.3, 0.5, 1, 2, 3]
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xticks(ticks, [f"{t:g}" for t in ticks])
        axis.minorticks_off()
        axis.set_xlabel("Sharpe ratio of the regime drift")
    return deck_end(fig, plt, axes[0], ncol=2, bottom=0.2)


def figure_deck_second_splits(results, plt):
    fig, axes = deck_start(plt, 2)
    bins = np.linspace(-1.5, 2.0, 36)
    for axis, lam in zip(axes, V2_LAMBDAS):
        cell = v2_cell(results, 5, BASE.edge, BASE.k, BASE.volatility, lam, BASE.fee)
        if not cell:
            continue
        first = [th for r in cell for f, th in [root_of(r)] if f == "x"]
        second = [th for r in cell for d, f, th in r["kept"] if d >= 1]
        draw_hist(axis, first, bins, ORANGE, "first threshold")
        draw_hist(axis, second, bins, BLUE, "second-level threshold")
        axis.axvline(BASE.d, color="black", ls=":", lw=1.2, label="planted d = 0")
        share = 100 * np.mean([any(d >= 1 for d, _, _ in r["kept"]) for r in cell])
        axis.set_title(f"λ = {lam:g}: {share:.0f}% keep a second-level split\n({len(cell)} datasets)")
        axis.set_xlabel("threshold on x")
        axis.set_ylabel("number of splits")
    return deck_end(fig, plt, axes[0])


def figure_deck_halflife_threshold(results, plt, lam=1.0):
    fig, (left, right) = deck_start(plt, 2)
    for step in (5, 2):
        xs, summaries, _ = halflife_series(results, lam, step, BASE.fee)
        if len(xs):
            draw_mean(left, xs, summaries, RATE_COLORS[step], f"decisions {RATE_NAMES[step]}")
            draw_value(right, xs, summaries, "within", RATE_COLORS[step], f"decisions {RATE_NAMES[step]}")
    left.set_title("first threshold: mean ± std")
    right.set_title("first threshold within 0.25 of d (%)")
    right.set_ylabel("datasets (%)")
    right.set_ylim(0, 100)
    for axis in (left, right):
        halflife_axis(axis)
    return deck_end(fig, plt, left, ncol=2)


def figure_deck_halflife_sharpe(results, plt, lam=1.0, key="sharpe"):
    fig, axes = deck_start(plt, 2)
    for axis, step in zip(axes, (5, 2)):
        xs, summaries, _ = halflife_series(results, lam, step, BASE.fee)
        if len(xs):
            draw_value(axis, xs, summaries, key, ORANGE, "tree, fee 3 bp", "tree")
            draw_value(axis, xs, summaries, key, GREY, "ideal rule, fee 3 bp", "ideal rule")
        xs, summaries, _ = halflife_series(results, lam, step, 0.0)
        if len(xs):
            draw_value(axis, xs, summaries, key, GREEN, "tree, no fee", "tree")
        axis.set_title(f"decisions {RATE_NAMES[step]}")
        axis.set_ylabel(
            "annualized test Sharpe ratio" if key == "sharpe" else "asset / cash switches per year"
        )
        halflife_axis(axis)
    if key == "sharpe":
        low = min(axis.get_ylim()[0] for axis in axes)
        high = max(axis.get_ylim()[1] for axis in axes)
        for axis in axes:
            axis.set_ylim(low, high)
    return deck_end(fig, plt, axes[0])


def figure_deck_e9(v1, base, plt):
    """Model v1 minus the base, paired on the same datasets, and the second-level splits each keeps."""
    fig, (left, right) = deck_start(plt, 2)
    positions = np.arange(len(SETTING_STEPS))
    for edge, color in ((0.10, ORANGE), (0.05, BLUE)):
        means, errors = [], []
        for step in SETTING_STEPS:
            a = {r["seed"]: r for r in base if r["step"] == step and r["edge"] == edge}
            b = {r["seed"]: r for r in v1 if r["step"] == step and r["edge"] == edge}
            seeds = sorted(set(a) & set(b))
            scale = np.sqrt(DAYS_PER_YEAR / step)
            diff = np.array([b[s]["test_sharpe"]["tree"] - a[s]["test_sharpe"]["tree"] for s in seeds]) * scale
            means.append(diff.mean())
            errors.append(2 * diff.std(ddof=1) / np.sqrt(len(seeds)))
        left.errorbar(
            positions, means, yerr=errors, color=color, capsize=4, label=f"drift ±{edge * 252:.0f}% a year", **LINE
        )
    left.axhline(0, color="black", ls=":", lw=1)
    left.set_title("v1 minus base: annualized Sharpe\n(paired, ± 2 s.e.)")
    left.set_ylabel("Sharpe difference")
    width = 0.38
    for k, (name, records_, color) in enumerate((("base", base, GREY), ("model v1", v1, ORANGE))):
        shares = [
            setting_summary([r for r in records_ if r["step"] == step and r["edge"] == 0.10], step)["second"]
            for step in SETTING_STEPS
        ]
        right.bar(positions + (k - 0.5) * width, shares, width, color=color, label=name)
    right.set_title("trees with a second-level split\n(drift ±25% a year)")
    right.set_ylabel("datasets (%)")
    for axis in (left, right):
        axis.set_xticks(
            positions,
            [RATE_NAMES[s].replace(" a ", "\na ").replace("every ", "every\n") for s in SETTING_STEPS],
        )
        axis.set_xlabel("decision rate")
    handles = left.get_legend_handles_labels()[0] + right.get_legend_handles_labels()[0]
    labels = left.get_legend_handles_labels()[1] + right.get_legend_handles_labels()[1]
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=DECK_FONT - 1, frameon=False)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    plt.rc("font", size=9)
    return fig




def in_band(record):
    """True if one of the tree's leaf scores lies within the no-trade band of its own fee (|score| <= f);
    without a fee there is no band."""
    fee = record.get("fee", BASE.fee)
    return fee > 0 and any(abs(v) <= fee for v in set(np.round(record["score_profile"], 6)))


def figure_deck_halflife_band(results, plt, lam=1.0):
    """Trees with a leaf score inside the no-trade band: how many (left) and the Sharpe ratio they lose against
    the other trees of the same cell (right), for the same series in the same colours."""
    fig, (left, right) = deck_start(plt, 2)
    lives = [half_life(k) for k in V2_K]
    for step, fee, color, label in (
        (2, 0.0003, BLUE, "twice a week, fee 3 bp"),
        (5, 0.0003, ORANGE, "weekly, fee 3 bp"),
        (2, 0.0, GREEN, "twice a week, no fee"),
    ):
        shares, losses = [], []
        for k in V2_K:
            cell = v2_cell(results, step, BASE.edge, k, BASE.volatility, lam, fee)
            if not cell:
                shares.append(np.nan)
                losses.append(np.nan)
                continue
            flags = np.array([in_band(r) for r in cell])
            sharpe = np.array([r["test_sharpe"]["tree"] for r in cell]) * np.sqrt(DAYS_PER_YEAR / step)
            shares.append(100 * flags.mean())
            losses.append(
                sharpe[~flags].mean() - sharpe[flags].mean() if flags.any() and (~flags).any() else np.nan
            )
        left.plot(lives, shares, color=color, label=label, **LINE)
        right.plot(lives, losses, color=color, label=label, **LINE)
    left.set_title("trees with a leaf score inside the band")
    left.set_ylabel("datasets (%)")
    left.set_ylim(0, 50)
    right.set_title("Sharpe ratio lost by those trees")
    right.set_ylabel("Sharpe of the others minus theirs")
    right.axhline(0, color="black", ls=":", lw=1)
    for axis in (left, right):
        halflife_axis(axis)
    return deck_end(fig, plt, left, ncol=3, bottom=0.18)



def first_thresholds(cell):
    """The first threshold on x of every tree of the cell that splits on x."""
    return [th for r in cell for f, th in [root_of(r)] if f == "x"]


def draw_boxes(axis, groups, positions, color, width=0.6, counts=True):
    """Boxplots of the groups (lists of thresholds) at the positions, in one colour; the count above each box."""
    kept = [(g, p) for g, p in zip(groups, positions) if len(g)]
    if not kept:
        return
    axis.boxplot(
        [g for g, _ in kept],
        positions=[p for _, p in kept],
        widths=width,
        showfliers=True,
        boxprops={"color": color, "lw": 1.6},
        medianprops={"color": color, "lw": 2.4},
        whiskerprops={"color": color, "lw": 1.4},
        capprops={"color": color, "lw": 1.4},
        flierprops={"marker": ".", "markeredgecolor": color, "ms": 5},
    )
    if counts:
        for g, p in kept:
            axis.text(p, 1.42, f"{len(g)}", ha="center", va="top", fontsize=DECK_FONT - 4, color=color)


def threshold_axis(axis, positions, labels, xlabel):
    axis.axhline(BASE.d, color="black", ls=":", lw=1.2)
    axis.set_xticks(positions, labels)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("first threshold on x")
    axis.set_ylim(-1.5, 1.5)


def figure_deck_threshold_signal(results, plt):
    """The first threshold by signal strength: boxplots over the datasets that split, λ = 1 and λ = 5."""
    fig, axes = deck_start(plt, 2)
    positions = np.arange(len(V2_EDGES))
    labels = [f"{price_sharpe(e):.1f}" for e in V2_EDGES]
    for axis, lam in zip(axes, V2_LAMBDAS):
        groups = [
            first_thresholds(v2_cell(results, 5, e, BASE.k, BASE.volatility, lam, BASE.fee)) for e in V2_EDGES
        ]
        draw_boxes(axis, groups, positions, LAMBDA_COLORS[lam])
        threshold_axis(axis, positions, labels, "Sharpe ratio of the regime drift (annualized)")
        axis.set_title(f"λ = {lam:g}")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    plt.rc("font", size=9)
    return fig


def figure_deck_threshold_halflife(results, plt, lam=1.0):
    """The first threshold by half-life of x: boxplots, weekly and twice a week."""
    fig, axes = deck_start(plt, 2)
    positions = np.arange(len(V2_K))
    labels = [f"{half_life(k):.2g}" for k in V2_K]
    for axis, step in zip(axes, (5, 2)):
        groups = [
            first_thresholds(v2_cell(results, step, BASE.edge, k, BASE.volatility, lam, BASE.fee)) for k in V2_K
        ]
        draw_boxes(axis, groups, positions, RATE_COLORS[step])
        threshold_axis(axis, positions, labels, "half-life of the feature x (days)")
        axis.set_title(f"decisions {RATE_NAMES[step]}")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    plt.rc("font", size=9)
    return fig


def figure_deck_threshold_rate(v1, base, plt):
    """E9: the first threshold by decision rate, v1 against the base, at the two drifts."""
    fig, axes = deck_start(plt, 2)
    positions = np.arange(len(SETTING_STEPS))
    labels = [RATE_NAMES[s].replace(" a ", "\na ").replace("every ", "every\n") for s in SETTING_STEPS]
    for axis, edge in zip(axes, (0.10, 0.05)):
        for records_, color, shift, label in ((base, GREY, -0.2, "base"), (v1, ORANGE, 0.2, "model v1")):
            groups = [
                first_thresholds([r for r in records_ if r["step"] == step and r["edge"] == edge])
                for step in SETTING_STEPS
            ]
            draw_boxes(axis, groups, positions + shift, color, width=0.34, counts=False)
            axis.plot([], [], color=color, lw=2.4, label=label)
        threshold_axis(axis, positions, labels, "decision rate")
        axis.set_title(f"drift ±{edge * 252:.0f}% a year")
    return deck_end(fig, plt, axes[0], ncol=2, bottom=0.16)


def figure_deck_threshold_hist(v1, base, plt, step=5):
    """E9: the distribution of the first threshold, v1 against the base, weekly, at the two drifts."""
    fig, axes = deck_start(plt, 2)
    bins = np.linspace(-1.5, 1.5, 31)
    for axis, edge in zip(axes, (0.10, 0.05)):
        for records_, color, label in ((base, GREY, "base"), (v1, ORANGE, "model v1")):
            first = first_thresholds([r for r in records_ if r["step"] == step and r["edge"] == edge])
            draw_hist(axis, first, bins, color, label)
        axis.axvline(BASE.d, color="black", ls=":", lw=1.2)
        axis.set_title(f"drift ±{edge * 252:.0f}% a year, {RATE_NAMES[step]} decisions")
        axis.set_xlabel("first threshold on x")
        axis.set_ylabel("number of trees")
    return deck_end(fig, plt, axes[0], ncol=2, bottom=0.16)


def deck_grid(plt, rows, cols, width=10.5, height=5.0):
    """A grid of small panels for the deck, with slightly smaller type than deck_start."""
    plt.rc("font", size=DECK_FONT - 1)
    fig, axes = plt.subplots(rows, cols, figsize=(width, height), sharex=True)
    return fig, axes


def figure_deck_threshold_rate_hist(v1, base, plt, steps=(2, 10, 21)):
    """E9: the distribution of the first threshold at the decision rates other than weekly (which has its own
    frame), v1 against the base, at the two drifts."""
    fig, axes = deck_grid(plt, 2, len(steps))
    bins = np.linspace(-1.5, 1.5, 31)
    for row, edge in zip(axes, (0.10, 0.05)):
        for axis, step in zip(row, steps):
            counts = []
            for records_, color, label in ((base, GREY, "base"), (v1, ORANGE, "model v1")):
                cell = [r for r in records_ if r["step"] == step and r["edge"] == edge]
                first = first_thresholds(cell)
                draw_hist(axis, first, bins, color, label)
                counts.append(f"{len(first)}/{len(cell)}")
            axis.axvline(BASE.d, color="black", ls=":", lw=1.2)
            axis.set_title(f"drift ±{edge * 252:.0f}% a year, {RATE_NAMES[step]}", fontsize=DECK_FONT - 2)
    for axis in axes[-1]:
        axis.set_xlabel("first threshold on x")
    for row in axes:
        row[0].set_ylabel("number of trees")
    return deck_end(fig, plt, axes[0][0], ncol=2, bottom=0.11)


def figure_deck_halflife_hist(results, plt, lam=1.0):
    """E10: the distribution of the first threshold by half-life of x, weekly against twice a week
    (λ = 1, fee 3 bp)."""
    fig, axes = deck_grid(plt, 2, 3)
    bins = np.linspace(-1.5, 1.5, 31)
    for axis, k in zip(axes.flat, V2_K):
        counts = []
        for step in (5, 2):
            cell = v2_cell(results, step, BASE.edge, k, BASE.volatility, lam, BASE.fee)
            first = first_thresholds(cell)
            draw_hist(axis, first, bins, RATE_COLORS[step], f"decisions {RATE_NAMES[step]}")
            counts.append(f"{len(first)}/{len(cell)}")
        axis.axvline(BASE.d, color="black", ls=":", lw=1.2)
        axis.set_title(f"half-life {half_life(k):.2g} days", fontsize=DECK_FONT - 2)
    for axis in axes[-1]:
        axis.set_xlabel("first threshold on x")
    for row in axes:
        row[0].set_ylabel("number of trees")
    return deck_end(fig, plt, axes[0][0], ncol=2, bottom=0.11)


def figure_deck_regime_positions(results, plt, step=5, edge=0.10):
    """Mean position by x on the test rows when the volatility threshold sits at d + 1: the tree against the
    ideal rule with the same lagged covariance and the ideal rule with the true variance; λ = 1 and λ = 10."""
    fig, axes = deck_start(plt, 2)
    centers = (POSITION_BINS[:-1] + POSITION_BINS[1:]) / 2
    for axis, lam in zip(axes, (1.0, 10.0)):
        cell = setting_records(results, f"volatility threshold at d + 1, λ = {lam:g}", step, edge)
        if not cell:
            continue
        for key, color, label in (
            ("tree", ORANGE, "tree"),
            ("ideal rule", GREY, "ideal rule, same lagged covariance"),
            ("ideal rule, true variance", GREEN, "ideal rule, true variance"),
        ):
            values = np.array([[np.nan if v is None else v for v in r["profile"][key]] for r in cell])
            axis.plot(centers, np.nanmedian(values, axis=0), color=color, label=label, lw=2.2)
        axis.axvline(BASE.d, color="black", ls=":", lw=1.2, label="drift threshold d = 0")
        axis.axvline(cell[0]["volatility_threshold"], color=RED, ls=":", lw=1.6, label="volatility threshold d + 1")
        axis.set_title(f"λ = {lam:g} ({len(cell)} datasets, median position)")
        axis.set_xlabel("x on the decision day")
        axis.set_ylabel("mean weight of the asset")
        axis.set_ylim(-0.05, 1.05)
    return deck_end(fig, plt, axes[0], ncol=3, bottom=0.24)


def figure_deck_regime_splits(results, plt, step=5, edge=0.10):
    """Every threshold kept at λ = 10 with the volatility threshold at d + 1, 180-day and 20-day covariance."""
    fig, axes = deck_start(plt, 2)
    bins = np.linspace(-2.0, 2.5, 46)
    names = (
        ("volatility threshold at d + 1, λ = 10", "λ = 10, 180-day covariance"),
        ("volatility threshold at d + 1, λ = 10, 20-day covariance", "λ = 10, 20-day covariance"),
    )
    for axis, (name, title) in zip(axes, names):
        cell = setting_records(results, name, step, edge)
        if not cell:
            continue
        d_sigma = cell[0]["volatility_threshold"]
        first = [th for r in cell for f, th in [root_of(r)] if f == "x"]
        second = [th for r in cell for d, f, th in r["kept"] if d >= 1]
        draw_hist(axis, first, bins, ORANGE, "first threshold")
        draw_hist(axis, second, bins, BLUE, "second-level threshold")
        axis.axvline(BASE.d, color="black", ls=":", lw=1.2, label="drift threshold d = 0")
        axis.axvline(d_sigma, color=RED, ls=":", lw=1.6, label="volatility threshold d + 1")
        near = 100 * np.mean(
            [any(abs(th - d_sigma) <= 0.25 for d, f, th in r["kept"] if d >= 1) for r in cell]
        )
        axis.set_title(f"{title} ({len(cell)} datasets)\n{near:.0f}% split near the volatility threshold")
        axis.set_xlabel("threshold on x")
        axis.set_ylabel("number of splits")
    return deck_end(fig, plt, axes[0], ncol=2, bottom=0.27)


# The short deck, section "How the tree learns, in plain words": one dataset fitted with model v1.
BIGPICTURE_COLORS = {"navy": "#1F4E9C", "brick": "#C0392B", "leaf": "#2E8B57", "orange": "#E67E22"}


def fitting_weeks(result):
    """x and the return (in %) of the fitting weeks: the first 75% of the training decisions (run_case prunes on the last 25%)."""
    case, rows = result["case"], result["rows"]
    n_fit = case.n_train - int(round(0.25 * case.n_train))
    return rows.X[:n_fit, 0], rows.R[:n_fit, 0] * 100


def wrong_weeks_cost(returns):
    """Money lost on the wrong weeks by the better of the two answers: the losses taken in the asset if the
    weeks gain in total, the gains missed in cash otherwise. It is the decision regret without fees or risk term."""
    if len(returns) == 0:
        return 0.0
    return min(-returns[returns < 0].sum(), returns[returns > 0].sum())


def figure_deck_bigpicture_split(result, plt):
    """The fitting weeks sorted by x, their returns, the root cut and the answer of each leaf."""
    x, r = fitting_weeks(result)
    roots = [th for depth, feature, th in result["kept"] if depth == 0 and feature == "x"]
    c = BIGPICTURE_COLORS
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.scatter(x, r, s=4, color="0.72", label="one training week")
    bins = np.linspace(-2.75, 2.75, 23)
    idx = np.digitize(x, bins)
    full = [b for b in range(1, len(bins)) if (idx == b).sum() >= 6]
    ax.plot([0.5 * (bins[b - 1] + bins[b]) for b in full], [r[idx == b].mean() for b in full],
            "o-", color=c["navy"], ms=4, lw=1.4, label="average return per interval of $x$")
    if roots:
        theta = roots[0]
        left, right = x < theta, x >= theta
        ax.hlines(r[left].mean(), x.min(), theta, color=c["orange"], lw=3.5)
        ax.hlines(r[right].mean(), theta, x.max(), color=c["orange"], lw=3.5, label="leaf average")
        ax.axvline(theta, color=c["brick"], lw=1.6, ls="--", label=f"the cut: $x = {theta:.2f}$")
        ax.text(theta - 0.12, 7.6, f"leaf 1: {r[left].mean():+.2f}%\ncash", ha="right", va="top", color=c["brick"], fontsize=7.5)
        ax.text(theta + 0.12, 7.6, f"leaf 2: {r[right].mean():+.2f}%\nasset", ha="left", va="top", color=c["leaf"], fontsize=7.5)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-8, 8)
    ax.set_xlabel("$x$ on the decision day", fontsize=8)
    ax.set_ylabel("return of the next 5 days (%)", fontsize=8)
    ax.tick_params(labelsize=7.5)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=7, frameon=False, columnspacing=1.0, handlelength=1.6)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    return fig


def figure_deck_bigpicture_cost(result, plt):
    """The cost of every admissible cut position under the simplified rule, against the cost without a cut."""
    case = result["case"]
    x, r = fitting_weeks(result)
    n_fit = len(x)
    m_star = max(case.min_samples_leaf, int(np.ceil(case.min_leaf_fraction * n_fit)))
    candidates = np.sort(x)[m_star:-m_star]
    costs = np.array([(wrong_weeks_cost(r[x < th]) + wrong_weeks_cost(r[x >= th])) / n_fit for th in candidates])
    without = wrong_weeks_cost(r) / n_fit
    best = candidates[np.argmin(costs)]
    c = BIGPICTURE_COLORS
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.plot(candidates, costs, color=c["navy"], lw=1.5, label="cost with one cut at $\\theta$")
    ax.axhline(without, color="0.35", ls=":", lw=1.4, label=f"cost without a cut: {without:.2f}%")
    ax.plot([best], [costs.min()], "o", color=c["brick"], ms=7, label=f"cheapest cut: $\\theta = {best:.2f}$, {costs.min():.2f}%")
    ax.set_xlabel("position of the cut $\\theta$", fontsize=8)
    ax.set_ylabel("cost (% lost a week)", fontsize=8)
    ax.tick_params(labelsize=7.5)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=1, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    return fig


def plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
    )
    os.makedirs(OUT, exist_ok=True)
    grid = x_grid(BASE)
    expected = expected_return_curve(
        BASE, grid, n_paths=100000
    )  # smooth enough to draw
    single = run_case(BASE, seed=42, curve=(grid, expected))
    print(
        f"single dataset: splits kept {single['kept']}; expected return changes sign at "
        f"x = {sign_change(grid, expected):.3f}"
    )
    for name, path in single["paths"].items():
        figures = annual(path, BASE)
        print(
            f"   test {name:22s} return {figures['return']:.1%}  Sharpe {figures['sharpe']:.2f}  "
            f"turnover {figures['turnover']:.2f} per decision"
        )
    figures = {
        "planted_world_one_dataset": figure_world(single, plt),
        "learned_vs_ideal_one_dataset": figure_learned(single, grid, expected, plt),
        "test_decisions_one_dataset": figure_decisions(single, plt),
    }
    if os.path.exists(RESULTS):
        with open(RESULTS) as handle:
            results = json.load(handle)
        print_summary(results)
        edge_labels = [
            f"edge {e:g}\n({e * BASE.volatility * DAYS_PER_YEAR:+.0%} a year)"
            for e in EDGES
        ]
        k_labels = [f"k = {k:g}\n(half-life {np.log(2) / k:.0f} d)" for k in K_VALUES]
        figures.update(
            {
                "thresholds_by_rate_histograms": figure_histograms(results, plt),
                "maps_rate_vs_edge": figure_maps(
                    results,
                    plt,
                    "edge",
                    EDGES,
                    edge_labels,
                    "drift / volatility per day (drift per year)",
                ),
                "maps_rate_vs_speed": figure_maps(
                    results,
                    plt,
                    "k",
                    K_VALUES,
                    k_labels,
                    "reversion speed of x per day",
                ),
                "thresholds_by_rate_boxplots": figure_thresholds_by_rate(results, plt),
            }
        )
    if os.path.exists(SETTINGS_RESULTS):
        with open(SETTINGS_RESULTS) as handle:
            settings = json.load(handle)
        print_settings_summary(settings, list(SETTINGS), "settings experiment")
        figures.update(
            {
                "risk_fee_depth_positions": figure_settings_positions(
                    settings, plt, list(SETTINGS)
                ),
                "risk_fee_depth_maps": figure_settings_maps(
                    settings, plt, list(SETTINGS)
                ),
            }
        )
    if os.path.exists(ENHANCEMENTS_RESULTS):
        with open(ENHANCEMENTS_RESULTS) as handle:
            enhancements = json.load(handle)
        names = list(ENHANCEMENTS)
        print_settings_summary(enhancements, names, "enhancements experiment")
        figures.update(
            {
                "split_options_scores": figure_scores(enhancements, plt, names),
                "split_options_thresholds": figure_thresholds_by_setting(
                    enhancements, plt, names
                ),
                "split_options_maps": figure_settings_maps(
                    enhancements, plt, names, title="The enhancements experiment"
                ),
            }
        )
    if os.path.exists(PRUNING_RESULTS):
        with open(PRUNING_RESULTS) as handle:
            pruning = json.load(handle)
        names = list(PRUNING)
        print_settings_summary(pruning, names, "pruning experiment")
        figures.update(
            {
                "pruning_hurdle_splits": figure_split_shares(pruning, plt, names),
                "pruning_hurdle_thresholds": figure_thresholds_by_setting(
                    pruning, plt, names
                ),
                "pruning_hurdle_maps": figure_settings_maps(
                    pruning, plt, names, title="The pruning experiment"
                ),
            }
        )
    if os.path.exists(V1_RESULTS) and os.path.exists(RESULTS):
        with open(V1_RESULTS) as handle:
            v1 = json.load(handle)
        with (
            open(RESULTS) as handle
        ):  # the base of Section 4: same seeds, same pipeline with every option off
            base = [
                dict(r, setting="base")
                for r in json.load(handle)
                if r["tag"] == "edge"
                and r["k"] == BASE.k
                and r["edge"] in SETTING_EDGES
                and r["step"] in SETTING_STEPS
                and r["seed"] < V1_DATASETS
            ]
        names = ["base", "model v1"]
        print_settings_summary(v1 + base, names, "model v1 against the base")
        print_paired_summary(v1, base)
        figures.update(
            {
                "model_v1_thresholds": figure_thresholds_by_setting(
                    v1 + base, plt, names
                ),
                "model_v1_splits": figure_split_shares(v1 + base, plt, names),
                "model_v1_maps": figure_settings_maps(
                    v1 + base, plt, names, title="Model v1 against the base"
                ),
                "deck_e9_paired": figure_deck_e9(v1, base, plt),
                "deck_threshold_rate": figure_deck_threshold_rate(v1, base, plt),
                "deck_threshold_hist": figure_deck_threshold_hist(v1, base, plt),
                "deck_threshold_rate_hist": figure_deck_threshold_rate_hist(v1, base, plt),
            }
        )
    if os.path.exists(OPTIONS2_RESULTS):
        with open(OPTIONS2_RESULTS) as handle:
            options2 = json.load(handle)
        names = list(OPTIONS2)
        print_settings_summary(
            options2,
            names,
            "sharper smoothing and hurdle-below-root experiment",
            OPTIONS2_STEPS,
        )
        figures.update(
            {
                "split_options_2_scores": figure_scores(options2, plt, names),
                "split_options_2_thresholds": figure_thresholds_by_setting(
                    options2, plt, names
                ),
                "split_options_2_splits": figure_split_shares(
                    options2, plt, names, OPTIONS2_STEPS
                ),
                "split_options_2_maps": figure_settings_maps(
                    options2,
                    plt,
                    names,
                    title="Sharper smoothing and the hurdle below the root",
                    steps=OPTIONS2_STEPS,
                ),
            }
        )
    if os.path.exists(RESOLUTION_RESULTS):
        with open(RESOLUTION_RESULTS) as handle:
            resolution = json.load(handle)
        print_resolution_summary(resolution)
        figures["resolution_curves"] = figure_resolution(resolution, plt)
    if os.path.exists(REGIME_RESULTS):
        with open(REGIME_RESULTS) as handle:
            regime = json.load(handle)
        names = list(REGIME)
        print_settings_summary(regime, names, "regime-volatility experiment")
        figures.update(
            {
                "regime_volatility_positions": figure_settings_positions(
                    regime, plt, names
                ),
                "regime_volatility_maps": figure_settings_maps(
                    regime, plt, names, title="The regime-volatility experiment"
                ),
                "regime_volatility_thresholds": figure_thresholds_by_setting(
                    regime, plt, names
                ),
                "regime_volatility_splits": figure_regime_second_splits(
                    regime, plt, names
                ),
            }
        )
    if os.path.exists(REGIME_V1_RESULTS):
        with open(REGIME_V1_RESULTS) as handle:
            regime_v1 = json.load(handle)
        names = list(REGIME)
        print_settings_summary(regime_v1, names, "regime-volatility experiment on model v1")
        figures.update(
            {
                "regime_volatility_v1_positions": figure_settings_positions(regime_v1, plt, names),
                "regime_volatility_v1_maps": figure_settings_maps(
                    regime_v1, plt, names, title="The regime-volatility experiment on model v1"
                ),
                "regime_volatility_v1_splits": figure_regime_second_splits(regime_v1, plt, names),
                "deck_regime_positions": figure_deck_regime_positions(regime_v1, plt),
                "deck_regime_splits": figure_deck_regime_splits(regime_v1, plt),
            }
        )
    v2 = v2_records()
    if v2:
        print_v2_summary(v2)
        figures.update(
            {
                "model_v1_recovery_vs_sharpe": figure_v2_recovery(v2, plt),
                "model_v1_second_splits": figure_v2_second_splits(v2, plt),
                "model_v1_halflife": figure_v2_halflife(v2, plt),
                # the short deck: one figure per frame
                "deck_signal_threshold": figure_deck_signal_threshold(v2, plt),
                "deck_signal_sharpe": figure_deck_signal_sharpe(v2, plt),
                "deck_signal_splits": figure_deck_signal_splits(v2, plt),
                "deck_volatility": figure_deck_volatility(v2, plt),
                "deck_second_splits": figure_deck_second_splits(v2, plt),
                "deck_halflife_threshold": figure_deck_halflife_threshold(v2, plt),
                "deck_halflife_sharpe": figure_deck_halflife_sharpe(v2, plt),
                "deck_halflife_switches": figure_deck_halflife_sharpe(v2, plt, key="switches"),
                "deck_halflife_band": figure_deck_halflife_band(v2, plt),
                "deck_threshold_signal": figure_deck_threshold_signal(v2, plt),
                "deck_threshold_halflife": figure_deck_threshold_halflife(v2, plt),
                "deck_halflife_hist": figure_deck_halflife_hist(v2, plt),
            }
        )
    # the short deck, "How the tree learns, in plain words": model v1 on one dataset
    single_v1 = run_case(replace(BASE, **V1_SETTING), seed=42)
    figures.update(
        {
            "deck_bigpicture_split": figure_deck_bigpicture_split(single_v1, plt),
            "deck_bigpicture_cost": figure_deck_bigpicture_cost(single_v1, plt),
        }
    )
    for name, fig in figures.items():
        fig.savefig(os.path.join(OUT, name + ".pdf"))
        fig.savefig(os.path.join(OUT, name + ".png"), dpi=130)
        print("wrote", name)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "time":
        tasks = all_tasks()
        total = 0.0
        for step in STEPS:
            task = next(t for t in tasks if t[1].step == step)
            started = time.perf_counter()
            job(task)
            one = time.perf_counter() - started
            count = sum(t[1].step == step for t in tasks)
            total += one * count
            print(f"one fit every {step} days: {one:.1f}s, {count} fits", flush=True)
        print(
            f"{len(tasks)} fits, about {total / 3600:.1f} core-hours; on about 6 effective "
            f"cores: about {total / 6 / 60:.0f} minutes"
        )
    elif command == "run":
        run(all_tasks(), RESULTS)
    elif command == "run-settings":
        run(settings_tasks(), SETTINGS_RESULTS)
    elif command == "run-enhancements":
        run(enhancement_tasks(), ENHANCEMENTS_RESULTS)
    elif command == "run-pruning":
        run(setting_tasks("pruning", PRUNING), PRUNING_RESULTS)
    elif command == "run-options":
        run(setting_tasks("options2", OPTIONS2, OPTIONS2_STEPS), OPTIONS2_RESULTS)
    elif command == "run-v1":
        run(setting_tasks("v1", V1, datasets=V1_DATASETS), V1_RESULTS)
    elif command == "run-resolution":
        run(resolution_tasks(), RESOLUTION_RESULTS)
    elif command == "run-regime":
        run(setting_tasks("regime", REGIME), REGIME_RESULTS)
    elif command == "run-regime-v1":
        regime_v1 = {name: {**V1_SETTING, **changes} for name, changes in REGIME.items()}
        run(setting_tasks("regime v1", regime_v1), REGIME_V1_RESULTS)
    elif command == "run-v2":
        run_v2()
    elif command == "plot":
        plot()
    else:
        print(__doc__)
