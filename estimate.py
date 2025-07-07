"""
estimate.py
Predict next-day stock prices with Ridge & Lasso, taking:
  --inst_id  Index of the instrument (0-based, 0-49)
  --lookback Number of lag days used as features
Example:
  python ridge_lasso_predict.py --inst_id 7 --lookback 30
"""
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import(
    mean_absolute_error,
    mean_squared_error,
)

def loadPrices(fn):
    global nt, nInst
    df=pd.read_csv(fn, sep=r'\s+', header=None, index_col=None)
    (nt,nInst) = df.shape
    return (df.values).T

def make_lagged_matrix(prices, lookback):
    """
    Create a lagged matrix of prices for the given lookback period.
    """
    X, y = [], []
    n = len(prices)
    for i in range(lookback, n):
        X.append(prices[i-lookback:i])
        y.append(prices[i])
    X = np.array(X)
    y = np.array(y)
    return X,y

def train_test_split_time(X, y, test_size=0.2):
    """
    Split the data into training and testing sets based on time.
    """
    split = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    return X_train, X_test, y_train, y_test

def evaluate(y_test, y_pred, label="model"):
    """
    Evaluate the model using Mean Absolute Error and Mean Squared Error.
    """
    mae = mean_absolute_error(y_test, y_pred) # Calculate Mean Absolute Error
    mse = mean_squared_error(y_test, y_pred) # Calculate Mean Squared Error
    rmse = np.sqrt(mse) # Calculate Root Mean Squared Error
    print(f"{label} - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")


# Global variables
nInst = 0
nt = 0
commRate = 0.0005
dlrPosLimit = 10000

def main():
    global nInst, nt

    parser = argparse.ArgumentParser(description="Predict next-day stock prices with Ridge & Lasso")
    parser.add_argument('--inst_id', type=int, default=0, help='Index of the instrument (0-based, 0-49)')
    parser.add_argument('--lookback', type=int, default=500, help='Number of lag days used as features')
    args = parser.parse_args()

    

    # Load all prices from file
    pricesFile="./prices.txt"
    prcAll = loadPrices(pricesFile)
    print ("Loaded %d instruments for %d days" % (nInst, nt))
    
    insID = args.inst_id
    lookback = args.lookback
    if insID < 0 or insID >= nInst:
        raise ValueError(f"Invalid instrument ID: {insID}. Must be between 0 and {nInst-1}.")
    
    print(f"Using instrument ID: {insID}, lookback period: {lookback} days")
    
    # Select the prices for the specified instrument
    prices = prcAll[insID]  # Prices for the instrument insID
    print(f"Using instrument ID: {insID}, with {len(prices)} price points.")

    # Prepare features and target variable
    X,y = make_lagged_matrix(prices, lookback)
    X_train, X_test, y_train, y_test = train_test_split_time(X, y, test_size=0.2)

    # Split data into training and testing sets
    # Prepare models with expanding-window cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    alphas = np.logspace(-4, 2, 60)  # Range of alpha values for Ridge and Lasso
    
    # Ridge Regression
    ridge_model = make_pipeline(StandardScaler(), RidgeCV(cv=tscv, alphas=alphas, fit_intercept=True))
    ridge_model.fit(X_train, y_train)
    
    # Lasso Regression
    lasso_model = make_pipeline(StandardScaler(), LassoCV(cv=tscv, alphas=alphas, fit_intercept=True, max_iter=15_000))
    lasso_model.fit(X_train, y_train)
    
    # Evaluate models
    ridge_pred, lasso_pred = ridge_model.predict(X_test), lasso_model.predict(X_test)

    print(f"Held-out 20 % test window (inst_id={insID}, lookback={lookback}):")
    evaluate(y_test, ridge_pred, label="Ridge")
    evaluate(y_test, lasso_pred, label="Lasso")

    print("\nChosen α values:")
    print(" Ridge α →", ridge_model[-1].alpha_)
    print(" Lasso α →", lasso_model[-1].alpha_)

    # 5) Full-history refit & tomorrow’s forecast
    ridge_model.fit(X, y)
    lasso_model.fit(X, y)
    latest_window = prices[-lookback:].reshape(1, -1)
    print(f"\nTomorrow forecast (Ridge) : {ridge_model.predict(latest_window)[0]:.4f}")
    print(f"Tomorrow forecast (Lasso) : {lasso_model.predict(latest_window)[0]:.4f}")


if __name__ == "__main__":
    main()