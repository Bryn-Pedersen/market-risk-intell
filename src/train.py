import os
import json
import math
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score
from omegaconf import OmegaConf
import hydra
from omegaconf import DictConfig
from src.models.lstm_classifier import LSTMClassifier

# Get the project root directory parent of src
PROJECT_ROOT = Path(__file__).parent.parent


class SeqDataset(Dataset):
    """Dataset that builds rolling windows from time series data"""
    
    def __init__(self, df, tickers, start, end, window, feature_cols, label_col='risk_label', fit_scaler=False, scalers=None):
        """
        Initialize dataset
        
        Args:
            df: DataFrame with date ticker features and label
            tickers: List of ticker symbols to include
            start: Start date inclusive
            end: End date exclusive
            window: Sequence length number of time steps
            feature_cols: List of feature column names
            label_col: Name of label column
            fit_scaler: Whether to fit scalers on this dataset
            scalers: Dict of pre-fitted scalers for validation/test
        """
        self.df = df[(df['date'] >= start) & (df['date'] < end) & (df['ticker'].isin(tickers))].copy()
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.window = window
        self.scalers = scalers or {c: StandardScaler() for c in feature_cols}
        
        # Build index of ticker start_idx
        self.index = []
        for t in tickers:
            sub = self.df[self.df['ticker'] == t].reset_index(drop=True)
            # fit scalers on train slice if requested
            if fit_scaler:
                for c in feature_cols:
                    vals = sub[c].values.reshape(-1, 1)
                    m = np.isfinite(vals).ravel()
                    if m.any():
                        self.scalers[c].fit(vals[m].reshape(-1, 1))
            for i in range(len(sub) - window):
                y = sub.loc[i + window - 1, self.label_col]
                # Check if label and all features in window are finite
                feature_window = sub.loc[i:i + window - 1, self.feature_cols]
                if np.isfinite(y) and feature_window.notna().all().all():
                    self.index.append((t, i))
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        t, i = self.index[idx]
        sub = self.df[self.df['ticker'] == t].reset_index(drop=True)
        X = sub.loc[i:i + self.window - 1, self.feature_cols].values.astype(np.float32)
        # scale per feature
        for j, c in enumerate(self.feature_cols):
            X[:, j] = self.scalers[c].transform(X[:, j].reshape(-1, 1)).ravel()
        y = sub.loc[i + self.window - 1, self.label_col].astype(np.float32)
        return torch.tensor(X), torch.tensor(y)


def time_splits(cfg):
    """Get time split configuration"""
    return dict(
        train_start=cfg.data.train_start,
        valid_start=cfg.data.valid_start,
        test_start=cfg.data.test_start,
        end="2099-01-01"
    )


def evaluate(model, loader, device):
    """
    Evaluate model on a dataset
    
    Returns:
        Dict with pr_auc and roc_auc metrics
    """
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits = model(X).detach().cpu().numpy()
            ps.extend(logits.tolist())
            ys.extend(y.numpy().tolist())
    ys = np.array(ys)
    ps = 1 / (1 + np.exp(-np.array(ps)))
    pr_auc = average_precision_score(ys, ps) if len(np.unique(ys)) > 1 else float('nan')
    roc = roc_auc_score(ys, ps) if len(np.unique(ys)) > 1 else float('nan')
    return dict(pr_auc=pr_auc, roc_auc=roc)


@hydra.main(config_path='../config', config_name='default', version_base=None)
def main(cfg: DictConfig):
    # Set random seeds for reproducibility
    if hasattr(cfg, 'seed'):
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
    
    # Hydra changes working directory so get project root
    # Use original cwd if available Hydra sets this otherwise use PROJECT_ROOT
    try:
        from hydra.utils import get_original_cwd
        project_root = Path(get_original_cwd())
    except:
        project_root = PROJECT_ROOT
    
    # Resolve paths relative to project root
    proc_dir = project_root / cfg.paths.proc_dir
    artifacts_dir = project_root / cfg.paths.artifacts_dir
    
    os.makedirs(artifacts_dir, exist_ok=True)
    df = pd.read_parquet(proc_dir / 'mvp_features.parquet')
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    
    tickers = df['ticker'].unique().tolist()
    splits = time_splits(cfg)
    
    # Build train fit scalers and valid reuse scalers
    train_ds = SeqDataset(
        df, tickers, splits['train_start'], splits['valid_start'],
        cfg.data.window, cfg.data.features, fit_scaler=True
    )
    valid_ds = SeqDataset(
        df, tickers, splits['valid_start'], splits['test_start'],
        cfg.data.window, cfg.data.features, scalers=train_ds.scalers
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=cfg.train.num_workers, drop_last=True
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=cfg.train.num_workers
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMClassifier(
        in_dim=len(cfg.data.features),
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    best_pr = -1.0
    for epoch in range(cfg.train.max_epochs):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optim.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            optim.step()
            total_loss += loss.item() * X.size(0)
        train_loss = total_loss / len(train_loader.dataset)
        
        metrics = evaluate(model, valid_loader, device)
        print(f"Epoch {epoch+1}/{cfg.train.max_epochs} | train_loss={train_loss:.4f} | val_pr_auc={metrics['pr_auc']:.4f} | val_roc_auc={metrics['roc_auc']:.4f}")
        
        if not math.isnan(metrics['pr_auc']) and metrics['pr_auc'] > best_pr:
            best_pr = metrics['pr_auc']
            torch.save(model.state_dict(), artifacts_dir / 'best_lstm.pt')
            with open(artifacts_dir / 'metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
    
    torch.save(model.state_dict(), artifacts_dir / 'last_lstm.pt')
    print("Training complete. Artifacts saved to:", artifacts_dir)


if __name__ == "__main__":
    main()

