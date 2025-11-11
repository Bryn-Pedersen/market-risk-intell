#!/bin/bash
# run for the risk prediction pipeline

set -e

echo "Downloading price data"
python -m src.data.ingest_prices --tickers SPY XLK XLF XLE XLI XLY XLP XLV HYG LQD --interval 1d --start 2010-01-01


echo "Creating features and labels"
python -m src.features.labels --horizon 3 --vol_window 10 --vol_thresh_p 0.8


echo "Training model"
python -m src.train model=lstm data.window=60 data.horizon=3


echo "Complete | artifacts/metrics.json for results."

