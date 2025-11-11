#!/bin/bash
# Run script for the risk prediction pipeline

set -e

echo "Step 1: Downloading price data..."
python -m src.data.ingest_prices --tickers SPY XLK XLF XLE XLI XLY XLP XLV HYG LQD --interval 1d --start 2010-01-01

echo ""
echo "Step 2: Creating features and labels..."
python -m src.features.labels --horizon 3 --vol_window 10 --vol_thresh_p 0.8

echo ""
echo "Step 3: Training model..."
python -m src.train model=lstm data.window=60 data.horizon=3

echo ""
echo "Complete. Check artifacts/metrics.json for results."

