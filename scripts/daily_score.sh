#!/usr/bin/env bash
# Daily scoring pipeline for risk predictions
# This script updates price data, recomputes features, and generates predictions

set -euo pipefail

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Warning: .venv not found. Make sure dependencies are installed."
fi

echo "=========================================="
echo "Daily Scoring Pipeline"
echo "=========================================="
echo ""

# Step 1: Download/update price data
echo "Step 1: Downloading price data..."
python -m src.data.ingest_prices \
    --tickers SPY XLK XLF XLE XLI XLY XLP XLV HYG LQD \
    --interval 1d \
    --start 2010-01-01 \
    --out data/raw/prices.parquet

echo ""
echo "Step 2: Creating features and labels..."
python -m src.features.labels \
    --horizon 3 \
    --vol_window 10 \
    --vol_thresh_p 0.8

echo ""
echo "Step 3: Generating predictions..."
python -m src.predict model.checkpoint_path=artifacts/best_lstm.pt

echo ""
echo "Step 4: (Optional) Export to database..."
# Uncomment and configure one of these if you want automatic DB export:
# 
# PostgreSQL:
# python -m src.bi.export_predictions \
#     --db-type postgres \
#     --connection-string "postgresql://user:password@host:5432/dbname" \
#     --table risk_predictions \
#     --if-exists append
#
# BigQuery:
# python -m src.bi.export_predictions \
#     --db-type bigquery \
#     --connection-string "your-gcp-project-id" \
#     --dataset risk_intelligence \
#     --table risk_predictions \
#     --if-exists append

echo ""
echo "=========================================="
echo "Scoring complete!"
echo "Output files:"
echo "  - data/processed/predictions.parquet"
echo "  - data/processed/predictions.csv"
echo "=========================================="

