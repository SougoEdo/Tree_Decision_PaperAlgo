#!/usr/bin/env python3
"""Experiments and figures for the planted-threshold test (see recovery_test.py).

  python recovery_figures.py time    one job, to estimate the run time
  python recovery_figures.py run     every experiment, in parallel; saves report/figures/recovery_results.json
  python recovery_figures.py plot    the figures, from the saved results, into report/figures/
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

from recovery_test import (FEATURES, Case, expected_return_curve, run_case, sign_change)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "report", "figures")
RESULTS = os.path.join(OUT, "recovery_results.json")

BASE = Case()                                   # the parameters set in recovery_test.py
K_VALUES = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
VOLATILITIES = [0.01, np.sqrt(BASE.price_variance), 0.10, 0.20, 0.30]   # per step
DATASETS_PER_CELL = 20
DATASETS_PER_HISTOGRAM = 100
TEST_DECISIONS = 300                            # enough when 20 datasets are pooled


def x_grid(case):
    std = np.sqrt(case.feature_variance)
    return np.linspace(case.mean - 4 * std, case.mean + 4 * std, 321)


@lru_cache(maxsize=None)
def curve_of(k, feature_variance, mu, d, mean, step):
    case = replace(BASE, k=k, feature_variance=feature_variance, mu=mu, d=d, mean=mean, step=step)
    grid = x_grid(case)
    return grid, expected_return_curve(case, grid, n_paths=10000)


def curve(case):
    return curve_of(case.k, case.feature_variance, case.mu, case.d, case.mean, case.step)


def job(task):
    tag, case, seed = task
    grid, expected = curve(case)
    result = run_case(case, seed, curve=(grid, expected))
    return {
        "tag": tag, "k": case.k, "volatility": float(np.sqrt(case.price_variance)),
        "step": case.step, "seed": seed, "sign_change": sign_change(grid, expected),
        "grown": result["grown"], "kept": result["kept"],
        "test_mean": {name: float(path.net_returns.mean()) for name, path in result["paths"].items()},
        "test_sharpe": {name: float(path.sharpe) for name, path in result["paths"].items()},
    }


def all_tasks():
    short = replace(BASE, n_test=TEST_DECISIONS)
    tasks = [("grid", replace(short, k=k, price_variance=float(v) ** 2), seed)
             for k in K_VALUES for v in VOLATILITIES for seed in range(DATASETS_PER_CELL)]
    tasks += [("histogram", short, 1000 + seed) for seed in range(DATASETS_PER_HISTOGRAM)]
    tasks += [("histogram", replace(short, step=1), 1000 + seed)
              for seed in range(DATASETS_PER_HISTOGRAM)]
    return tasks


def run():
    tasks, results, start = all_tasks(), [], time.perf_counter()
    os.makedirs(OUT, exist_ok=True)
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        futures = [pool.submit(job, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if done % 50 == 0 or done == len(tasks):
                print(f"progress: {done}/{len(tasks)} fits after {time.perf_counter() - start:.0f}s",
                      flush=True)
    with open(RESULTS, "w") as handle:
        json.dump(results, handle)
    print(f"saved {RESULTS}")


# ----------------------------------------------------------------------------- figures
GREEN, RED, BLUE, ORANGE, GREY = "#2e8b57", "#c0392b", "#1f4e9c", "#e67e22", "#555555"


def shade_regimes(axis, steps, x, d):
    """Light green while x >= d (drift +mu), light red while x < d (drift -mu)."""
    above = x >= d
    edges = np.flatnonzero(np.diff(above.astype(int))) + 1
    for start, stop in zip(np.r_[0, edges], np.r_[edges, len(x)]):
        axis.axvspan(steps[start], steps[min(stop, len(x) - 1)],
                     color=GREEN if above[start] else RED, alpha=0.13, lw=0)


def tree_scores(result, x_values):
    """The tree's predicted return of the asset when x takes these values (decoys at their mean)."""
    X = np.tile(result["rows"].X[:result["case"].n_train].mean(axis=0), (len(x_values), 1))
    X[:, 0] = x_values
    return result["tree"].predict_returns(X)


def position(result, scores):
    """Weight of the asset chosen by the optimizer, from half-and-half holdings."""
    n = len(scores)
    sigma = np.tile(result["rows"].Sigma[:result["case"].n_train].mean(axis=0), (n, 1, 1))
    return result["optimizer"].solve(scores, np.full((n, 2), 0.5), sigma)[:, 0]


def figure_world(result, plt):
    case = result["case"]
    x, _, _, prices = result["series"]
    steps = np.arange(case.window, case.window + 400)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    for axis in (top, bottom):
        shade_regimes(axis, steps, x[steps], case.d)
    top.plot(steps, x[steps], color=BLUE, lw=1.2)
    top.axhline(case.d, color="black", ls="--", lw=1, label=f"planted threshold d = {case.d:g}")
    top.axhline(case.mean, color=GREY, ls=":", lw=1, label=f"long-run mean of x = {case.mean:g}")
    top.set_ylabel("feature x")
    top.legend(loc="upper right", fontsize=8, ncol=2)
    top.set_title(f"The planted world: drift +{case.mu:.0%} per step while x is above d (green), "
                  f"-{case.mu:.0%} while below (red)", fontsize=10)
    bottom.semilogy(steps, prices[steps], color="black", lw=1.2)
    decisions = steps[(steps - case.window) % case.step == 0]
    bottom.plot(decisions, np.full(len(decisions), prices[steps].min() * 0.8), "|", color=GREY, ms=7)
    bottom.set_ylabel("price of the asset (log scale)")
    bottom.set_xlabel(f"step (grey ticks: one decision every {case.step} steps)")
    fig.tight_layout()
    return fig


def figure_learned(result, grid, expected, plt):
    case = result["case"]
    rows = result["rows"]
    train = slice(0, case.n_train)
    crossing = sign_change(grid, expected)
    thresholds = [th for _, f, th in result["kept"] if f == "x"]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 2]})
    top.scatter(rows.X[train, 0], rows.R[train, 0], s=9, color=GREY, alpha=0.35,
                label=f"realised {case.step}-step returns (training decisions)")
    top.plot(grid, expected, color=BLUE, lw=2, label=f"true expected {case.step}-step return given x")
    top.plot(grid, tree_scores(result, grid)[:, 0], color=ORANGE, lw=2.2,
             label="the tree: predicted return of each leaf")
    top.axhline(0, color="black", lw=0.6)
    top.set_ylabel(f"return of the asset over {case.step} steps")
    top.set_ylim(-0.45, 0.6)
    top.set_title("What the tree should learn, and what it learned (one dataset)", fontsize=10)
    ideal = position(result, np.column_stack([expected, np.zeros(len(grid))]))
    bottom.plot(grid, ideal, color=BLUE, lw=2, label="ideal position (optimizer fed with the true curve)")
    bottom.plot(grid, position(result, tree_scores(result, grid)), color=ORANGE, lw=2.2,
                label="position chosen with the tree's leaves")
    bottom.set_ylabel("weight of the asset")
    bottom.set_xlabel("feature x at the decision")
    bottom.set_ylim(-0.05, 1.08)
    for axis in (top, bottom):
        axis.axvline(case.d, color="black", ls="--", lw=1)
        axis.axvline(crossing, color=BLUE, ls=":", lw=1.4)
        for th in thresholds:
            axis.axvline(th, color=ORANGE, lw=1)
    top.annotate(f"d = {case.d:g}", (case.d, 0.52), fontsize=8, ha="left", xytext=(3, 0),
                 textcoords="offset points")
    top.annotate(f"sign change\nx = {crossing:.2f}", (crossing, 0.47), fontsize=8, ha="right",
                 color=BLUE, xytext=(-3, 0), textcoords="offset points")
    top.legend(loc="lower right", fontsize=8)
    bottom.legend(loc="center right", fontsize=8)
    top.set_xlim(case.mean - 3.2 * result["feature_std"], case.mean + 3.2 * result["feature_std"])
    fig.tight_layout()
    return fig


def figure_decisions(result, plt):
    case = result["case"]
    rows = result["rows"]
    x = result["series"][0]
    shown = 60
    first = case.n_train
    decisions = np.arange(first, first + shown)
    steps = np.arange(rows.steps[first], rows.steps[first + shown])
    fig, (top, middle, bottom) = plt.subplots(3, 1, figsize=(9, 7), sharex=True,
                                              gridspec_kw={"height_ratios": [2, 1.3, 2]})
    for axis in (top, middle):
        shade_regimes(axis, steps, x[steps], case.d)
    top.plot(steps, x[steps], color=BLUE, lw=1)
    top.axhline(case.d, color="black", ls="--", lw=1)
    for th in [th for _, f, th in result["kept"] if f == "x"]:
        top.axhline(th, color=ORANGE, lw=1.2)
    top.set_ylabel("feature x")
    top.set_title("Test period, never seen in training: the regimes (green/red), the tree's "
                  "threshold (orange) and its decisions", fontsize=10)
    middle.step(rows.steps[decisions], result["paths"]["tree"].weights[:shown, 0], where="post",
                color=ORANGE, lw=1.6, label="tree")
    middle.step(rows.steps[decisions], result["paths"]["ideal rule"].weights[:shown, 0],
                where="post", color=BLUE, lw=1, ls="--", label="ideal rule")
    middle.set_ylabel("weight of the asset")
    middle.set_ylim(-0.08, 1.08)
    middle.legend(loc="center right", fontsize=8, ncol=2)
    styles = {"tree": (ORANGE, "-", 2), "ideal rule": (BLUE, "--", 1.4), "rule at d": (GREEN, ":", 1.6),
              "always the asset": ("black", "-", 1)}
    for name, (color, ls, lw) in styles.items():
        wealth = np.cumprod(1 + result["paths"][name].net_returns[:shown])
        bottom.semilogy(rows.steps[decisions] + case.step, wealth, color=color, ls=ls, lw=lw, label=name)
    bottom.axhline(1, color=GREY, lw=0.8)
    bottom.set_ylabel("wealth (log scale, start = 1)")
    bottom.set_xlabel("step")
    bottom.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def root_of(record):
    roots = [(f, th) for depth, f, th in record["kept"] if depth == 0]
    return roots[0] if roots else (None, None)


def figure_histograms(results, plt):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for axis, step in zip(axes, (1, BASE.step)):
        records = [r for r in results if r["tag"] == "histogram" and r["step"] == step]
        found = {depth: [th for r in records for dp, f, th in r["kept"] if f == "x" and dp == depth]
                 for depth in (0, 1)}
        bins = np.linspace(-0.3, 0.3, 41) if step == 1 else np.linspace(-2.5, 1.0, 36)
        axis.hist([found[0], found[1]], bins=bins, stacked=True, color=[ORANGE, "#f5c99b"],
                  label=["first split", "second-level split"])
        axis.axvline(BASE.d, color="black", ls="--", lw=1.2, label=f"planted d = {BASE.d:g}")
        # With one decision per step the expected return jumps from -mu to +mu exactly at d.
        crossing = BASE.d if step == 1 else records[0]["sign_change"]
        axis.axvline(crossing, color=BLUE, ls=":", lw=1.6,
                     label="where the expected return over the holding period changes sign")
        roots = [root_of(r)[0] for r in records]
        counts = ", ".join(f"{name}: {roots.count(name)}" for name in FEATURES if roots.count(name))
        axis.set_title(f"one decision every {step} step{'s' if step > 1 else ''} "
                       f"(sign change at x = {crossing:.2f})\n"
                       f"first split on {counts}; no split kept: {roots.count(None)}  "
                       f"({len(records)} datasets)", fontsize=9)
        axis.set_xlabel("threshold found on x")
    axes[0].set_ylabel("number of splits")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("Where the tree puts its thresholds on x (note the different horizontal scales)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def heatmap(axis, values, title, fmt, cmap, vmin, vmax):
    image = axis.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    axis.set_xticks(range(len(VOLATILITIES)),
                    [f"{v:.0%}\n({BASE.mu / v:.2g})" for v in VOLATILITIES], fontsize=8)
    axis.set_yticks(range(len(K_VALUES)), [f"{k:g}" for k in K_VALUES], fontsize=8)
    axis.set_xlabel("price volatility per step  (drift / volatility)", fontsize=8.5)
    axis.set_title(title, fontsize=9.5)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                dark = cmap != "RdYlGn" and (values[i, j] - vmin) / (vmax - vmin) > 0.6
                axis.text(j, i, fmt(values[i, j]), ha="center", va="center", fontsize=8.5,
                          color="white" if dark else "black")
    return image


def figure_maps(results, plt):
    grid = [r for r in results if r["tag"] == "grid"]
    on_x = np.full((len(K_VALUES), len(VOLATILITIES)), np.nan)
    none = np.full_like(on_x, np.nan)
    captured = np.full_like(on_x, np.nan)
    for i, k in enumerate(K_VALUES):
        for j, v in enumerate(VOLATILITIES):
            cell = [r for r in grid if r["k"] == k and abs(r["volatility"] - v) < 1e-9]
            roots = [root_of(r)[0] for r in cell]
            on_x[i, j] = 100 * roots.count("x") / len(cell)
            none[i, j] = 100 * roots.count(None) / len(cell)
            gain = sum(r["test_mean"]["tree"] - r["test_mean"]["tree without splits"] for r in cell)
            best = sum(r["test_mean"]["ideal rule"] - r["test_mean"]["tree without splits"] for r in cell)
            if best / len(cell) > 0.0005:            # otherwise there is nothing to gain
                captured[i, j] = 100 * gain / best
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    heatmap(axes[0], on_x, "first split on the true feature x\n(% of 20 datasets)",
            lambda v: f"{v:.0f}", "Greens", 0, 100)
    heatmap(axes[1], none, "no split kept after pruning\n(% of 20 datasets)",
            lambda v: f"{v:.0f}", "Greys", 0, 100)
    heatmap(axes[2], captured, "test gain over the tree without splits,\nas % of the ideal rule's gain "
            "(blank: nothing to gain)", lambda v: f"{v:.0f}", "RdYlGn", -50, 110)
    axes[0].set_ylabel("reversion speed k of the feature, per step", fontsize=8.5)
    fig.tight_layout()
    return fig


def figure_thresholds_by_k(results, plt):
    fig, axis = plt.subplots(figsize=(8, 4.2))
    base_vol = float(np.sqrt(BASE.price_variance))
    for position_, k in enumerate(K_VALUES):
        cell = [r for r in results if r["tag"] == "grid" and r["k"] == k
                and abs(r["volatility"] - base_vol) < 1e-9]
        found = [root_of(r)[1] for r in cell if root_of(r)[0] == "x"]
        if found:
            axis.boxplot(found, positions=[position_], widths=0.5, showfliers=True,
                         boxprops={"color": ORANGE}, medianprops={"color": ORANGE, "lw": 2},
                         whiskerprops={"color": ORANGE}, capprops={"color": ORANGE},
                         flierprops={"marker": ".", "markeredgecolor": ORANGE})
        crossing = cell[0]["sign_change"]
        if crossing is not None:
            axis.plot(position_, crossing, "D", color=BLUE, ms=7,
                      label="expected return changes sign" if position_ == 0 else None)
        axis.text(position_, 1.0, f"{len(found)}/20", ha="center", va="bottom", fontsize=8,
                  color=GREY, transform=axis.get_xaxis_transform())
    axis.axhline(BASE.d, color="black", ls="--", lw=1, label=f"planted d = {BASE.d:g}")
    axis.set_xticks(range(len(K_VALUES)), [f"{k:g}" for k in K_VALUES])
    axis.set_xlabel("reversion speed k of the feature, per step (grey, above: datasets whose first split is on x)")
    axis.set_ylabel("first threshold found on x")
    axis.set_title(f"Threshold found against k (price volatility {base_vol:.1%} per step, "
                   f"one decision every {BASE.step} steps)", fontsize=10, pad=16)
    axis.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    return fig


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    os.makedirs(OUT, exist_ok=True)
    grid = x_grid(BASE)
    expected = expected_return_curve(BASE, grid, n_paths=100000)     # smooth enough to draw
    single = run_case(BASE, seed=42, curve=(grid, expected))
    print(f"single dataset: splits kept {single['kept']}; expected return changes sign at "
          f"x = {sign_change(grid, expected):.3f}")
    for name, path in single["paths"].items():
        print(f"   test {name:22s} mean {path.net_returns.mean():.3%}  Sharpe {path.sharpe:.3f}  "
              f"turnover {path.turnover.mean():.2f}")
    figures = {"recovery_world": figure_world(single, plt),
               "recovery_learned": figure_learned(single, grid, expected, plt),
               "recovery_decisions": figure_decisions(single, plt)}
    if os.path.exists(RESULTS):
        with open(RESULTS) as handle:
            results = json.load(handle)
        figures.update({"recovery_histograms": figure_histograms(results, plt),
                        "recovery_maps": figure_maps(results, plt),
                        "recovery_thresholds_by_k": figure_thresholds_by_k(results, plt)})
    for name, fig in figures.items():
        fig.savefig(os.path.join(OUT, name + ".pdf"))
        fig.savefig(os.path.join(OUT, name + ".png"), dpi=130)
        print("wrote", name)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "time":
        started = time.perf_counter()
        job(all_tasks()[0])
        one = time.perf_counter() - started
        print(f"one fit: {one:.1f}s; {len(all_tasks())} fits on about 4.5 effective cores: "
              f"about {one * len(all_tasks()) / 4.5 / 60:.0f} minutes")
    elif command == "run":
        run()
    elif command == "plot":
        plot()
    else:
        print(__doc__)
