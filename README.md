# Market Risk Intelligence

Binary classifier for near-term volatility spike prediction using price and volume data. Base implementation for future integration with news, filings, and social media data.

## Overview

Predicts risk days (volatility spikes) in the next 3 days using historical price and volume patterns. Uses LSTM neural networks trained on realized volatility, returns, and volume features.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


### Data Ingestion


```bash
python -m src.data.ingest_prices --tickers SPY XLK XLF XLE XLI XLY XLP XLV HYG LQD --interval 1d --start 2010-01-01
```

Output: `data/raw/prices.parquet`

### Feature Engineering


```bash
python -m src.features.labels --horizon 3 --vol_window 10 --vol_thresh_p 0.8
```

Output: `data/processed/mvp_features.parquet`

### Training


```bash
python -m src.train model=lstm data.window=60 data.horizon=3
```

Output:
- artifacts/best_lstm.pt - Best model by PR-AUC
- artifacts/last_lstm.pt - Final epoch model
- artifacts/metrics.json - Validation metrics

### Prediction / Scoring

Generate predictions for all data and export to flat table format for BI tools:

```bash
python -m src.predict model.checkpoint_path=artifacts/best_lstm.pt
```

Output:
- `data/processed/predictions.parquet` - Predictions in Parquet format
- `data/processed/predictions.csv` - Predictions in CSV format (for Tableau/PowerBI)

The predictions table includes:
- `date`, `ticker` - Identifiers
- `label` - Ground truth (0/1)
- `pred_prob` - Predicted probability P(risk day in next 3 days)
- `pred_label` - Binary prediction (based on threshold)
- `realized_vol_fwd_max` - Forward-looking max volatility (for post-hoc validation)
- `ret`, `rv10`, `volume_z` - Feature values for slicing/analysis

Override config from command line:
```bash
python -m src.predict predict.threshold=0.7 predict.output_csv=data/predictions_custom.csv
```

## Analytics / BI Integration

The prediction export (`data/processed/predictions.csv`) is designed for direct integration with BI tools like Tableau, PowerBI, or databases (Postgres, BigQuery).

### Tableau Dashboard Examples

**Risk Heatmap**
- Rows: `date`
- Columns: `ticker`
- Color: `pred_prob` (dark = high spike risk)

**Precision/Recall Over Time**
- Use `label` vs threshold on `pred_prob` (e.g. >0.7) to compute confusion metrics by month

**Lift Curves by Volatility Percentile**
- Group dates into deciles by `pred_prob`
- Show realized spike rate per decile using `label`

**Ticker-Detail View**
- Filter by `ticker`
- Show time series with price, realized volatility, and predicted risk band

### Production Scoring Pipeline

For production use, set up a daily scoring job that:
1. Ingests new prices
2. Recomputes features
3. Scores latest window
4. Appends to predictions table
5. Tableau dashboard refreshes on schedule

**Automated Daily Scoring:**

Run the daily scoring script:
```bash
./scripts/daily_score.sh
```

This script:
- Downloads/updates price data for all tickers
- Recomputes features and labels
- Generates predictions for all available data
- Optionally exports to database (see below)

**Schedule with cron (macOS/Linux):**
```bash
# Edit crontab
crontab -e

# Add line to run daily at 6 AM
0 6 * * * cd /Users/bryn/idea && ./scripts/daily_score.sh >> logs/score.log 2>&1
```

**Database Export:**

Export predictions to PostgreSQL or BigQuery for direct Tableau connection:

```bash
# PostgreSQL
python -m src.bi.export_predictions \
    --db-type postgres \
    --connection-string "postgresql://user:password@host:5432/dbname" \
    --table risk_predictions \
    --if-exists append

# BigQuery
python -m src.bi.export_predictions \
    --db-type bigquery \
    --connection-string "your-gcp-project-id" \
    --dataset risk_intelligence \
    --table risk_predictions \
    --if-exists append
```

**Optional Dependencies:**
- PostgreSQL: `pip install sqlalchemy psycopg2-binary`
- BigQuery: `pip install google-cloud-bigquery`

## Configuration

Hyperparameters are configured via Hydra configs:
- config/default.yaml - Main configuration
- config/model/lstm.yaml - Model architecture

override from command line:
```bash
python -m src.train data.window=90 model.hidden_size=128
```

## Project Structure

```
config/                 
  default.yaml
  predict.yaml          # Prediction/scoring configuration
  model/
    lstm.yaml
src/
  data/
    ingest_prices.py    # download from Yahoo Finance
  features/
    labels.py           # feat engineering and labeling
  models/
    lstm_classifier.py  # LSTM model architecture
  bi/
    export_predictions.py # Database export for BI tools
  train.py              # Training loop and evaluation
  predict.py            # Batch scoring and prediction export
scripts/
  daily_score.sh        # Automated daily scoring pipeline
data/                   # Data directory (auto-created)
  processed/
    predictions.parquet # Prediction outputs for BI tools
    predictions.csv
artifacts/              # Model checkpoints and metrics (auto-created)
```

## Features

- Realized volatility, 10-day rolling, annualized
- Daily returns
- Volume z-scores (60-day rolling)

## Labels

Binary: 1 if forward-looking max volatility exceeds 80th percentile threshold, 0 otherwise. Horizon: 3 days ahead.

## Metrics

- PR-AUC
- ROC-AUC
  
## Roadmap

- ✅ v0.2: Prediction export for BI tools (Tableau, PowerBI) - **Complete**
- v0.3: News sentiment integration (FinBERT embeddings)
- v0.4: EDGAR filings parsing (8-K/10-Q)
- v0.5: Multi-modal transformer with cross-attention
- v0.6: Sector-specific heads and uncertainty quantification
- v0.7: Online updating and model registry

## Requirements

- Python 3.10+
- PyTorch
- pandas, numpy
- scikit-learn
- yfinance
- hydra-core
- omegaconf
- pyarrow
