# PaperAlgo: an SPO decision tree for portfolio decisions

## Layout

```
code/                                the model and the experiments; run the commands from this folder
                                     with the project's Python: ../.venv/bin/python
  spo_tree.py                        the optimizer, the replay, the SPO tree, weekly rows from daily data
  synthetic_market.py                the synthetic market: mean-reverting feature, regime-switching price
  planted_threshold_test.py          one planted-threshold dataset, fitted and printed in detail
  planted_threshold_experiments.py   the experiment sweeps and the figures:
                                     run | run-settings | run-enhancements | run-pruning |
                                     run-resolution | run-regime | run-options | plot | time
  run_real_data.py                   template for real data (fill in load_data)
  tests/                             unit tests:  python -m unittest discover -s tests
report/
  model_and_training_pipeline.tex    the model: inputs, decision problem, loss, tree, training step by step
  experiments_planted_threshold.tex  the experiments on the first market (constant volatility), one section per sweep
  experiments_regime_volatility.tex  the experiments on the second market (volatility follows a threshold too)
  motivations.md                     the motivations of the project
  figures/                           figures and results files (results_*.json) written by the experiments
```

## Commands

```
cd code
../.venv/bin/python -m unittest discover -s tests          # 59 tests
../.venv/bin/python planted_threshold_test.py               # one dataset, printed in detail
../.venv/bin/python planted_threshold_experiments.py run    # the daily-data sweep (about 100 minutes)
../.venv/bin/python planted_threshold_experiments.py plot   # every figure and summary table, from the saved results
cd ../report && latexmk -pdf model_and_training_pipeline.tex experiments_planted_threshold.tex experiments_regime_volatility.tex
```
