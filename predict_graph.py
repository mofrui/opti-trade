#!/usr/bin/env python
"""
estimate.py  –  walk-forward Ridge / Lasso back-test + plot

Example:
    python estimate.py --inst_id 7 --lookback 250
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import (
    RidgeCV,
    LassoCV,
    Ridge,
    Lasso,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────
def loadPrices(fn: Path) -> np.ndarray:
    """Load whitespace-separated price file → shape (nInst, nt)."""
    df = pd.read_csv(fn, sep=r"\s+", header=None, index_col=None)
    return df.values.T          # instruments × time

def make_lagged_matrix(prices: np.ndarray, lookback: int):
    """Return X (n-samples × lookback) and y (n-samples,) for next-day forecast."""
    X, y = [], []
    for i in range(lookback, len(prices)):
        X.append(prices[i - lookback : i])
        y.append(prices[i])
    return np.asarray(X), np.asarray(y)

# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────
def directional_accuracy(actual: np.ndarray, pred: np.ndarray) -> float:
    """
    Return the fraction of times the model got the *direction* of tomorrow's
    move right.  Expects: `pred` may contain NaNs for days not predicted.
    """
    valid_idx = np.where(~np.isnan(pred))[0]            # indices with a forecast
    if len(valid_idx) == 0:                             # safety guard
        return np.nan
    # Compare sign of (price_t − price_{t-1}) with sign of (pred_t − price_{t-1})
    true_dir = np.sign(actual[valid_idx] - actual[valid_idx - 1])
    pred_dir = np.sign(pred[valid_idx]   - actual[valid_idx - 1])
    return (true_dir == pred_dir).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Core routine
# ──────────────────────────────────────────────────────────────────────────────
def predictFull(insID, lookback, prcAll, which_model="lasso"):
    # ------------------------------------------------------------------ data --
    prices = prcAll[insID]
    print(f"Using instrument ID: {insID}, with {len(prices)} price points.")

    X, y = make_lagged_matrix(prices, lookback)

    # ----------------------------------------------------- initial CV training --
    tscv   = TimeSeriesSplit(n_splits=5)
    alphas = np.logspace(-4, 2, 60)

    ridge_model = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=alphas, cv=tscv, fit_intercept=True),
    )
    lasso_model = make_pipeline(
        StandardScaler(),
        LassoCV(alphas=alphas, cv=tscv, fit_intercept=True, max_iter=15_000),
    )

    ridge_model.fit(X, y)
    lasso_model.fit(X, y)

    est_full = ridge_model if which_model == "ridge" else lasso_model

    # ----------------------------------------------- (A) in-sample fitted line --
    X_full_std   = est_full[:-1].transform(X)      # standardised X
    y_fitted_all = est_full[-1].predict(X_full_std)
    fitted_series = np.concatenate([np.full(lookback, np.nan), y_fitted_all])

    # ------------------------------------ (B) walk-forward 1-step-ahead preds --
    walk_pred = np.full_like(prices, np.nan, dtype=float)

    # chosen α for quick re-fit inside loop
    alpha_ridge = ridge_model[-1].alpha_
    alpha_lasso = lasso_model[-1].alpha_

    for t in range(lookback + 1, len(prices)):
        X_train, y_train = make_lagged_matrix(prices[:t], lookback)

        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)

        if which_model == "ridge":
            mdl = Ridge(alpha=alpha_ridge, fit_intercept=True)
        else:
            mdl = Lasso(alpha=alpha_lasso, fit_intercept=True, max_iter=60_000)

        mdl.fit(X_train_s, y_train)

        X_t = prices[t - lookback : t].reshape(1, -1)
        walk_pred[t] = mdl.predict(scaler.transform(X_t))[0]
    
    
    # ─────────────────────────────────────────  evaluation  ───────────────────
    da = directional_accuracy(prices, walk_pred)
    print(f"\nDirectional accuracy on walk-forward window: {da*100:.2f}% "
          f"(n={np.sum(~np.isnan(walk_pred))} days)")


    # ------------------------------------------------------------------ plot --
    plt.figure(figsize=(12, 5))
    plt.plot(prices, label="Actual")
    plt.plot(fitted_series, label=f"{which_model.title()} fitted (in-sample)")
    plt.plot(walk_pred, label=f"{which_model.title()} walk-forward (OOS)")
    plt.title(
        f"Instrument {insID} — Actual vs {which_model.title()} Predicted\n"
        f"lookback = {lookback} days"
    )
    plt.xlabel("Day index")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Walk-forward Ridge / Lasso stock-price forecaster"
    )
    parser.add_argument(
        "--inst_id", type=int, default=0,
        help="Instrument index (0-based)"
    )
    parser.add_argument(
        "--lookback", type=int, default=500,
        help="Number of lag days used as features"
    )
    parser.add_argument(
        "--prices", type=Path, default=Path("./prices.txt"),
        help="Path to whitespace-delimited price file"
    )
    args = parser.parse_args()

    prcAll = loadPrices(args.prices)
    print(
        f"Loaded {prcAll.shape[0]} instruments for {prcAll.shape[1]} days "
        f"from {args.prices}"
    )

    predictFull(args.inst_id, args.lookback, prcAll, which_model="lasso")

if __name__ == "__main__":
    # mute harmless convergence chatter
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
