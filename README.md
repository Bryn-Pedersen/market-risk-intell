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
  model/
    lstm.yaml
src/
  data/
    ingest_prices.py    # download from Yahoo Finance
  features/
    labels.py           # feat engineering and labeling
  models/
    lstm_classifier.py  # LSTM model architecture
  train.py              # Training loop and evaluation
data/                  
artifacts/              
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
  
## Future Versions

- v0.2: News sentiment integration (FinBERT embeddings??)
- v0.3: EDGAR filings parsing
- v0.4: Multi-modal transformer 
- v0.5: Sector-specific heads and uncertainty quantification
- v0.6: Online updating and model registry

## Requirements

- Python 3.10+
- PyTorch
- pandas, numpy
- scikit-learn
- yfinance
- hydra-core
- omegaconf
- pyarrow
