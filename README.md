# MGDPR

MGDPR is a PyTorch implementation for graph-based stock movement prediction. The default open-source workflow uses ratio-only node features for representation learning, log-transformed OHLCV features for graph construction, and continuous mutual information adjacency.

## Install

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Install the PyTorch build that matches your platform if the generic `torch` wheel is not appropriate for your CUDA setup.

## Prepare Data

The repository can fetch adjusted OHLCV data from Yahoo Finance. If no ticker file is supplied for `SP500`, the current S&P 500 constituent list is read from Wikipedia and written to `SP500_tickers.csv`.

```bash
python -m data.prepare_market_data \
  --market SP500 \
  --raw_dir data/raw \
  --start 2022-12-01 \
  --end 2025-01-10
```

To only create the ticker list:

```bash
python -m data.prepare_market_data --market SP500 --raw_dir data/raw --symbols_only
```

For a custom ticker list, create a one-symbol-per-line CSV and pass it explicitly:

```bash
python -m data.prepare_market_data \
  --market SP500 \
  --raw_dir data/raw \
  --ticker_path my_tickers.csv \
  --start 2022-12-01 \
  --end 2025-01-10
```

## Train

Default training settings are:

- `representation_mode=ratio_ohlcv`
- `graph_feature_mode=log_ohlcv`
- `graph_mode=continuous_mi`
- `mi_neighbors=round(sqrt(time_steps * 5))`, clamped to valid sklearn bounds

Example SP500 run using freshly fetched data:

```bash
python train/train_val_test.py \
  --market SP500 \
  --raw_dir data/raw \
  --generated_dir data/generated \
  --fetch_data \
  --train_dates 2023-01-01 2023-12-31 \
  --val_dates 2024-01-01 2024-06-30 \
  --test_dates 2024-07-01 2024-12-31 \
  --epochs 3000 \
  --early_stop_patience 10 \
  --save_dir checkpoints/sp500_default \
  --results_path results/sp500_default.csv
```

If raw CSVs already exist, omit `--fetch_data` and pass the ticker list:

```bash
python train/train_val_test.py \
  --market SP500 \
  --raw_dir data/raw \
  --generated_dir data/generated \
  --ticker_path SP500_tickers.csv
```

## Feature Modes

`ratio_ohlcv` uses five ratio features per day: `Open/prev Close`, `High/Open`, `Low/Open`, `Close/Open`, and `Volume/prev Volume`. With `time_steps=21`, both `ratio_ohlcv` and `log_ohlcv` produce `105` learning features per ticker.

`graph_feature_mode=log_ohlcv` computes graph construction inputs from `log1p(OHLCV)`. `graph_mode=continuous_mi` builds one adjacency matrix for each OHLCV channel using sklearn continuous mutual information.

Yahoo Finance and Wikipedia are external research data sources. Their availability and current S&P 500 membership can change over time.
