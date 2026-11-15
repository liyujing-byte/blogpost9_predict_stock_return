# Blog Post 9 — Predicting Stock Returns with Gradient Boosting

## Project Overview

Reproduce the classroom XGBoost prediction task, test shallower trees, and compare out-of-sample forecast errors and Top 10 portfolio performance.

## Data

The frozen input in `data/raw/prices_snapshot.rds` contains the daily price data returned by `tidyquant::tq_get()` from Yahoo Finance for 95 classroom tickers and SPY, from 2022-01-03 through 2026-11-13. The RDS contains a list with `prices` and `spy` tibbles. The original ticker list and download calls are in `code/classroom_original.R`.

## Methods

Use the classroom feature construction and target. Respect both the signal date and the label end date at each split. Tune depth and minimum node size on 2023 validation data. Refit the classroom and tuned specifications on the same pre-2024 training sample and compare both with a zero-return forecast on the same 66,595 stock-date rows. Use the same Top 10 policy and non-overlapping, matched holding windows for both models and SPY.

## Main Results

The tuned model reduces test RMSE from 9.418% to 8.981%. Zero-return RMSE is 8.999% and its MAE is slightly lower than the tuned model's. The tuned portfolio trails the classroom portfolio in this particular historical backtest.

## Repository Structure

- `code/classroom_original.R` contains the classroom script as supplied.
- `code/analysis.R` builds the common sample, selects and fits models, and generates the tables and figure.
- `data/raw/` contains the frozen input data.
- `data/results/` contains validation, prediction, and investment metrics and periodic backtest returns.
- `figures/post/` contains the cumulative wealth chart.

## How to Reproduce

From the repository root, install the analysis packages in R if needed, then run the script. The snapshot makes the analysis independent of a live price download.

```r
install.packages(c("tidyverse", "tidymodels", "slider", "xgboost"))
```

```sh
Rscript code/analysis.R
```

The run used R 4.5.3. Package versions are recorded in `data/results/session_info.txt`. To rerun the original live-download classroom script separately, also install `tidyquant`.
