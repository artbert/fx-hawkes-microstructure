"""
Market Microstructure & Hawkes Process Utilities.
Provides tools for high-frequency FX data processing and point process estimation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from typing import Optional, Dict, List, Any, Tuple
from scipy.optimize import minimize
import Hawkes as hk

# Global Plotting Settings
sns.set(style="whitegrid")

# --- DATA PREPARATION ---

def exclude_midnight(
    df: pd.DataFrame,
    start_time: datetime.time,
    end_time: datetime.time
) -> pd.DataFrame:
    """Filters out low-liquidity periods (e.g., rollover) from the dataset."""
    df = df.copy()
    mask_to_exclude = (df['timestamp'].dt.time >= start_time) | (df['timestamp'].dt.time <= end_time)
    return df[~mask_to_exclude].reset_index(drop=True)

def load_and_prepare(
    path: str,
    drop_rollover: bool = True,
    rollover_start: str = '23:00',
    rollover_end: str = '01:15',
    sep: str = '\t'
) -> pd.DataFrame:
    """
    Loads raw tick data and performs initial cleaning.

    Args:
        path: Path to the CSV file.
        drop_rollover: Whether to remove low-liquidity midnight periods.
        rollover_start: Start of exclusion period.
        rollover_end: End of exclusion period.
        sep: CSV separator.
    """
    df = pd.read_csv(path, sep=sep)
    expected = {'<DATE>', '<TIME>', '<BID>', '<ASK>'}

    if not expected.issubset(set(df.columns)):
        raise ValueError(f"Missing expected columns. Found: {df.columns.tolist()}")

    # Standardize column names
    df = df.rename(columns={'<DATE>':'date', '<TIME>':'time', '<BID>':'bid', '<ASK>':'ask'})
    df['timestamp'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'])

    # Preserve raw values for point process identification
    df['bid_raw'] = df['bid']
    df['ask_raw'] = df['ask']

    # Forward-fill quotes for mid-price/spread calculations
    df['bid'] = df['bid_raw'].ffill()
    df['ask'] = df['ask_raw'].ffill()

    if drop_rollover:
        start = pd.to_datetime(rollover_start).time()
        end = pd.to_datetime(rollover_end).time()
        df = exclude_midnight(df, start, end)

    df = df.sort_values('timestamp').reset_index(drop=True)
    return df[['timestamp', 'bid_raw', 'ask_raw', 'bid', 'ask']]

def add_tick_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes microstructure features: price deltas, time deltas, and spread."""
    df = df.copy()

    # Price changes (NaN if no update at that millisecond)
    df['delta_bid'] = df['bid'].diff()
    df['delta_ask'] = df['ask'].diff()
    df.loc[df['bid_raw'].isna(), 'delta_bid'] = np.nan
    df.loc[df['ask_raw'].isna(), 'delta_ask'] = np.nan

    # Time deltas
    df['delta_t'] = df['timestamp'].diff().dt.total_seconds()
    df['delta_t_bid'] = df.loc[df['bid_raw'].notna(), 'timestamp'].diff().dt.total_seconds()
    df['delta_t_ask'] = df.loc[df['ask_raw'].notna(), 'timestamp'].diff().dt.total_seconds()

    # Spread and Mid-price dynamics
    df['spread'] = df['ask'] - df['bid']
    df['mid'] = (df['bid'] + df['ask']) / 2

    # Vectorized Mid-Quote Delta (handles asymmetric updates)
    mask_bid = df['bid_raw'].notna()
    mask_ask = df['ask_raw'].notna()

    df['delta_mid_q'] = np.nan
    df.loc[mask_bid & mask_ask, 'delta_mid_q'] = 0.5 * (df['delta_bid'] + df['delta_ask'])
    df.loc[mask_bid & ~mask_ask, 'delta_mid_q'] = df['delta_bid']
    df.loc[mask_ask & ~mask_bid, 'delta_mid_q'] = df['delta_ask']

    # Classification of movement
    df['direction'] = np.sign(df['delta_mid_q'].fillna(0))
    df['abs_delta_price'] = df['delta_mid_q'].abs()

    return df

# --- MODELING UTILS ---

def prepare_times(ts: pd.Series, t0: Optional[pd.Timestamp] = None) -> np.ndarray:
    """Converts timestamps to seconds relative to start time."""
    if t0 is None:
        t0 = ts.iloc[0]
    return (ts - t0).dt.total_seconds().values

def fit_hawkes_1d(
    T: np.ndarray,
    itv: List[float],
    kernel: str = 'exp',
    baseline: str = 'const',
    num_basis: Optional[int] = None
) -> Dict[str, Any]:
    """Fits a 1D Hawkes model using MLE."""
    model = hk.estimator()
    model.set_kernel(kernel)
    if num_basis:
        model.set_baseline(baseline, num_basis=num_basis)
    else:
        model.set_baseline(baseline)

    model.fit(T, itv)

    return {
        'mu': model.parameter.get('mu'),
        'alpha': model.parameter.get('alpha'),
        'beta': model.parameter.get('beta'),
        'br': model.br,
        'L': model.L,
        'AIC': model.AIC
    }

def hawkes_2d_loglik(
    params: np.ndarray,
    T_bid: np.ndarray,
    T_ask: np.ndarray,
    T_end: float,
    check_lambda: bool = False
) -> float:
    """
    Log-Likelihood for 2D Hawkes process with exponential kernel.
    Uses recursive formulation for O(N) complexity.

    Args:
        params: [mu_b, mu_a, alpha_bb, alpha_ba, alpha_ab, alpha_aa, beta]
    """
    mu_b, mu_a, a_bb, a_ba, a_ab, a_aa, beta = params

    if min(params) <= 0:
        return 1e10  # Penalty for non-positive parameters

    # Combine events and track origin (0 for bid, 1 for ask)
    times = np.concatenate([T_bid, T_ask])
    types = np.concatenate([np.zeros(len(T_bid)), np.ones(len(T_ask))])

    sort_idx = np.argsort(times)
    times, types = times[sort_idx], types[sort_idx]

    Rb, Ra = 0.0, 0.0
    last_time = 0.0
    loglik = 0.0

    for t, typ in zip(times, types):
        dt = t - last_time
        decay = np.exp(-beta * dt)

        # Update recursive intensity components
        Rb *= decay
        Ra *= decay

        lambda_b = mu_b + a_bb * beta * Rb + a_ba * beta * Ra
        lambda_a = mu_a + a_ab * beta * Rb + a_aa * beta * Ra

        if typ == 0:
            if check_lambda and lambda_b <= 0: return 1e10
            loglik += np.log(lambda_b)
            Rb += 1
        else:
            if check_lambda and lambda_a <= 0: return 1e10
            loglik += np.log(lambda_a)
            Ra += 1

        last_time = t

    # Compensator part (Integral of intensity function)
    integral = (
        (mu_b + mu_a) * T_end +
        (a_bb + a_ab) * np.sum(1 - np.exp(-beta * (T_end - T_bid))) +
        (a_ba + a_aa) * np.sum(1 - np.exp(-beta * (T_end - T_ask)))
    )

    return -(loglik - integral)

# --- VISUALIZATION ---

def plot_distribution(series: pd.Series, title: str = "Distribution", bins: int = 50):
    """Plots histogram with density and basic statistics."""
    x = series.dropna()
    plt.figure(figsize=(9, 5))
    sns.histplot(x, bins=bins, kde=True, stat="density", color="steelblue", alpha=0.6)

    if not x.empty:
        mean, std = x.mean(), x.std()
        plt.axvline(mean, color="red", linestyle="--", label=f"Mean: {mean:.5e}")
        plt.axvline(mean-std, color="orange", linestyle=":", label=f"-1 STD = {mean-std:.5f}")
        plt.axvline(mean+std, color="orange", linestyle=":", label=f"+1 STD = {mean+std:.5f}")
        plt.legend()

    plt.title(title)
    plt.show()

def plot_num_basis_vs_alpha(num_basis_values, alpha_bids, alpha_asks):
    """Plots num_basis vs alpha."""
    plt.figure(figsize=(9, 5))
    plt.plot(num_basis_values, alpha_bids, marker='o', label='Alpha Bid')
    plt.plot(num_basis_values, alpha_asks, marker='o', label='Alpha Ask')
    plt.xlabel('num_basis')
    plt.ylabel('Alpha')
    plt.title('Num_basis vs Alpha')
    plt.legend()
    plt.grid(True)
    plt.show()

def basic_distribution(series: pd.Series) -> Dict[str, Any]:
    """Helper for metadata extraction."""
    x = series.dropna()
    return {
        "count": int(len(x)),
        "mean": float(x.mean()) if not x.empty else 0.0,
        "std": float(x.std()) if not x.empty else 0.0,
        "min": float(x.min()) if not x.empty else 0.0,
        "max": float(x.max()) if not x.empty else 0.0
    }
