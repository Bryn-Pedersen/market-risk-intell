import os
from pathlib import Path
import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import hydra
from omegaconf import DictConfig, OmegaConf

from src.models.lstm_classifier import LSTMClassifier

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent


class SeqDataset(Dataset):
    """Dataset that builds rolling windows from time series data (matches training)"""
    
    def __init__(self, df, tickers, window, feature_cols, label_col='risk_label', scalers=None):
        """
        Initialize dataset for prediction
        
        Args:
            df: DataFrame with date, ticker, features and label
            tickers: List of ticker symbols to include
            window: Sequence length (number of time steps)
            feature_cols: List of feature column names
            label_col: Name of label column
            scalers: Dict of pre-fitted scalers (required for prediction)
        """
        self.df = df[df['ticker'].isin(tickers)].copy()
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.window = window
        self.scalers = scalers or {c: StandardScaler() for c in feature_cols}
        
        # Build index of (ticker, start_idx) pairs
        self.index = []
        self.metadata = []  # Store (ticker, date) for each sample
        self.df = self.df.reset_index(drop=True)
        for t in tickers:
            sub = self.df[self.df['ticker'] == t].copy()
            sub = sub.reset_index(drop=True)
            for i in range(len(sub) - window):
                y = sub.loc[i + window - 1, self.label_col]
                # Check if label and all features in window are finite
                feature_window = sub.loc[i:i + window - 1, self.feature_cols]
                if np.isfinite(y) and feature_window.notna().all().all():
                    self.index.append((t, i))
                    # Store metadata for this sample (ticker and date)
                    date = sub.loc[i + window - 1, 'date']
                    self.metadata.append((t, date))
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        t, i = self.index[idx]
        sub = self.df[self.df['ticker'] == t].reset_index(drop=True)
        X = sub.loc[i:i + self.window - 1, self.feature_cols].values.astype(np.float32)
        # Scale per feature (same as training)
        for j, c in enumerate(self.feature_cols):
            X[:, j] = self.scalers[c].transform(X[:, j].reshape(-1, 1)).ravel()
        y = sub.loc[i + self.window - 1, self.label_col].astype(np.float32)
        return torch.tensor(X), torch.tensor(y), idx


def _device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def fit_scalers_from_training_data(df, tickers, train_start, valid_start, window, feature_cols):
    """
    Fit scalers on training data to match training pipeline
    This ensures predictions use the same scaling as training
    """
    from src.train import SeqDataset as TrainSeqDataset
    train_ds = TrainSeqDataset(
        df, tickers, train_start, valid_start,
        window, feature_cols, fit_scaler=True
    )
    return train_ds.scalers


@hydra.main(config_path="../config", config_name="predict", version_base=None)
def main(cfg: DictConfig):
    print("Config:\n", OmegaConf.to_yaml(cfg))

    device = _device(cfg.predict.device)

    # Hydra changes working directory, so get project root
    try:
        from hydra.utils import get_original_cwd
        project_root = Path(get_original_cwd())
    except:
        project_root = PROJECT_ROOT

    # Resolve paths relative to project root
    proc_dir = project_root / cfg.paths.proc_dir
    artifacts_dir = project_root / cfg.paths.artifacts_dir

    # Load processed features
    feat_path = proc_dir / cfg.predict.features_filename
    df = pd.read_parquet(feat_path)
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)

    feature_cols = cfg.data.features
    label_col = cfg.predict.label_col
    
    # Check required columns
    required_cols = feature_cols + [label_col, cfg.predict.date_col, cfg.predict.ticker_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Get all tickers
    tickers = df['ticker'].unique().tolist()

    # Fit scalers on training data to match training pipeline
    print("Fitting scalers on training data...")
    scalers = fit_scalers_from_training_data(
        df, tickers, cfg.data.train_start, cfg.data.valid_start,
        cfg.data.window, feature_cols
    )

    # Create dataset for prediction (use all data, but with training scalers)
    ds = SeqDataset(
        df=df,
        tickers=tickers,
        window=cfg.data.window,
        feature_cols=feature_cols,
        label_col=label_col,
        scalers=scalers
    )

    loader = DataLoader(
        ds, 
        batch_size=cfg.predict.batch_size, 
        shuffle=False,
        num_workers=cfg.predict.get('num_workers', 0)
    )

    # Load model
    model = LSTMClassifier(
        in_dim=len(feature_cols),
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
    )

    # Resolve checkpoint path (Hydra should have already resolved ${paths.artifacts_dir})
    ckpt_path = Path(cfg.model.checkpoint_path)
    if not ckpt_path.is_absolute():
        ckpt_path = artifacts_dir / ckpt_path
    
    model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
    model.to(device)
    model.eval()

    all_indices = []
    all_probs = []
    all_labels = []

    print("Generating predictions...")
    with torch.no_grad():
        for x, y, idx in loader:
            x = x.to(device)
            logits = model(x)

            if logits.ndim == 2:
                logits = logits.squeeze(1)

            probs = torch.sigmoid(logits)

            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(y.numpy().tolist())
            all_indices.extend(idx.numpy().tolist())

    # Build output dataframe with predictions and metadata
    # Use metadata from dataset to get date, ticker
    metadata_rows = []
    for idx in all_indices:
        ticker, date = ds.metadata[idx]
        metadata_rows.append({
            'ticker': ticker,
            'date': date
        })
    
    pred_df = pd.DataFrame(metadata_rows)
    pred_df['label'] = all_labels
    pred_df['pred_prob'] = all_probs
    pred_df['pred_label'] = (pred_df['pred_prob'] >= cfg.predict.threshold).astype(int)

    # Merge with original dataframe to get additional columns using date and ticker
    # This is more reliable than trying to track row indices
    # Only include columns that exist in the dataframe
    merge_cols = [cfg.predict.date_col, cfg.predict.ticker_col]
    optional_cols = ['rv_fwd_max', 'ret', 'rv10', 'volume_z']
    merge_cols.extend([c for c in optional_cols if c in df.columns])
    
    pred_df = pred_df.merge(
        df[merge_cols],
        left_on=['date', 'ticker'],
        right_on=[cfg.predict.date_col, cfg.predict.ticker_col],
        how='left'
    )
    
    # Drop duplicate date/ticker columns from merge
    if cfg.predict.date_col != 'date':
        pred_df = pred_df.drop(columns=[cfg.predict.date_col])
    if cfg.predict.ticker_col != 'ticker':
        pred_df = pred_df.drop(columns=[cfg.predict.ticker_col])

    # Rename rv_fwd_max to realized_vol_fwd_max for clarity
    pred_df = pred_df.rename(columns={'rv_fwd_max': 'realized_vol_fwd_max'})

    # Select and order final columns
    output_cols = [
        'date',
        'ticker',
        'label',
        'pred_prob',
        'realized_vol_fwd_max',
        'ret',
        'rv10',
        'volume_z'
    ]
    
    # Add any additional columns if specified
    if cfg.predict.get('extra_columns'):
        output_cols.extend(cfg.predict.extra_columns)
    
    # Only include columns that exist
    output_cols = [c for c in output_cols if c in pred_df.columns]
    pred_df = pred_df[output_cols].sort_values(['ticker', 'date'])

    # Resolve output paths (Hydra should have already resolved ${paths.proc_dir})
    output_parquet = Path(cfg.predict.output_parquet)
    output_csv = Path(cfg.predict.output_csv)
    if not output_parquet.is_absolute():
        # If relative, resolve relative to proc_dir
        output_parquet = proc_dir / output_parquet
    if not output_csv.is_absolute():
        # If relative, resolve relative to proc_dir
        output_csv = proc_dir / output_csv

    # Save
    os.makedirs(output_parquet.parent, exist_ok=True)
    os.makedirs(output_csv.parent, exist_ok=True)
    
    pred_df.to_parquet(str(output_parquet), index=False)
    pred_df.to_csv(str(output_csv), index=False)

    print(f"\nSaved predictions to:")
    print(f"  - {output_parquet}")
    print(f"  - {output_csv}")
    print(f"\nShape: {pred_df.shape}")
    print(f"\nColumns: {list(pred_df.columns)}")
    print(f"\nFirst few rows:")
    print(pred_df.head(10))
    print(f"\nSummary statistics:")
    print(pred_df[['pred_prob', 'label']].describe())


if __name__ == "__main__":
    main()
