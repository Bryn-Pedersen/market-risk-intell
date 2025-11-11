import argparse
import os
import pandas as pd
import numpy as np


def realized_vol(returns, window=10):
    """
    Compute annualized realized volatility
    
    Args:
        returns: Series of daily returns
        window: Rolling window size
    
    Returns:
        Series of annualized realized volatility
    """
    return np.sqrt(252) * returns.rolling(window).std()


def make_labels(df, horizon=3, vol_window=10, vol_thresh_p=0.8):
    """
    Create risk labels and basic features
    
    Args:
        df: DataFrame with price data must have close volume ticker date
        horizon: Number of days forward to look for risk spike
        vol_window: Window size for realized volatility calculation
        vol_thresh_p: Percentile threshold for risk label 0.8 = top 20%
    
    Returns:
        Tuple of DataFrame with labels/features threshold value
    """
    df = df.copy()
    df['ret'] = df.groupby('ticker')['close'].pct_change()
    
    # Use transform to compute realized volatility preserves index alignment
    df['rv10'] = df.groupby('ticker')['ret'].transform(
        lambda s: realized_vol(s, vol_window)
    )
    
    # Forward max RV over next H days starting from tomorrow not today
    # For day t compute max(rv10[t+1], rv10[t+2], ..., rv10[t+horizon])
    # Process each ticker group separately to avoid pandas index alignment issues
    rv_fwd_max_list = []
    for ticker, group in df.groupby('ticker'):
        # Work with numpy array to avoid index complications
        rv_values = group['rv10'].values
        result_values = np.full(len(rv_values), np.nan, dtype=float)
        
        # For each day look ahead up to horizon days
        for i in range(len(rv_values)):
            if i + 1 < len(rv_values):
                end_idx = min(i + 1 + horizon, len(rv_values))
                future_values = rv_values[i+1:end_idx]
                if len(future_values) > 0:
                    result_values[i] = np.nanmax(future_values)
        
        # Create Series with original group index
        rv_fwd_max = pd.Series(result_values, index=group.index, name='rv_fwd_max')
        rv_fwd_max_list.append(rv_fwd_max)
    
    # Combine all tickers and align with original DataFrame index
    df['rv_fwd_max'] = pd.concat(rv_fwd_max_list).reindex(df.index)
    
    # Threshold for label
    thresh = df['rv_fwd_max'].quantile(vol_thresh_p)
    df['risk_label'] = (df['rv_fwd_max'] >= thresh).astype(int)
    
    # Auxiliary feature volume z-score 60-day
    df['volume_z'] = df.groupby('ticker')['volume'].transform(
        lambda s: (s - s.rolling(60).mean()) / (s.rolling(60).std() + 1e-6)
    )
    
    return df, float(thresh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=str, default='data/raw/prices.parquet')
    ap.add_argument('--output', type=str, default='data/processed/mvp_features.parquet')
    ap.add_argument('--horizon', type=int, default=3)
    ap.add_argument('--vol_window', type=int, default=10)
    ap.add_argument('--vol_thresh_p', type=float, default=0.8)
    args = ap.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df = pd.read_parquet(args.input)
    df, thresh = make_labels(df, args.horizon, args.vol_window, args.vol_thresh_p)
    df.to_parquet(args.output)
    print(f"Saved features to {args.output}; label threshold={thresh:.4f}")


if __name__ == "__main__":
    main()

