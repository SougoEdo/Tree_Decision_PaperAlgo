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
                                     run-resolution | run-regime | run-options | run-v1 | plot | time
  run_real_data.py                   template for real data (fill in load_data)
  tests/                             unit tests:  python -m unittest discover -s tests
report/
  model_and_training_pipeline.tex    the model: inputs, decision problem, loss, tree, training step by step
  pipeline_algorithm.tex             the core algorithm as pseudo-code and flowcharts: replay and fees, score search,
                                     growth round (regret gate, ranking, Sharpe gate), candidates, pruning
  experiments_planted_threshold.tex  the experiments on the first market (constant volatility), one section per sweep
  experiments_regime_volatility.tex  the experiments on the second market (volatility follows a threshold too)
  presentation_model_v1.tex          beamer deck: model v1 (24 September 2026), its training and the evidence behind it
  presentation_model_v1_short.tex    the 15-minute version of the deck: model with equations, then the results
  model_history.md                   dated history of the pipeline versions, the markets and the experiments
  motivations.md                     the motivations of the project
  figures/                           figures and results files (results_*.json) written by the experiments
```

## Commands

```
cd code
../.venv/bin/python -m unittest discover -s tests          # 64 tests
../.venv/bin/python planted_threshold_test.py               # one dataset, printed in detail
../.venv/bin/python planted_threshold_experiments.py run    # the daily-data sweep (about 100 minutes)
../.venv/bin/python planted_threshold_experiments.py run-v1 # model v1 on 100 datasets (about 25 minutes)
../.venv/bin/python planted_threshold_experiments.py plot   # every figure and summary table, from the saved results
cd ../report && latexmk -pdf model_and_training_pipeline.tex pipeline_algorithm.tex experiments_planted_threshold.tex experiments_regime_volatility.tex presentation_model_v1.tex presentation_model_v1_short.tex
```
