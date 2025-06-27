import numpy as np

nInst = 50
currentPos = np.zeros(nInst, dtype=int)

def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    """
    Dual Moving‐Average Incremental Rebalancing Strategy

    1. Calculate 5‐day (short) and 20‐day (long) simple moving averages.
    2. If short MA > long MA, signal = +1 (go long); otherwise signal = -1 (go short).
    3. Compute target shares per instrument so that market value ≈ $10,000.
    4. Trade only the difference: delta = targetShares - currentPos.
    5. Update currentPos += delta and return it.
    """
    global currentPos

    ninst, nt = prcSoFar.shape

    # do nothing until we have at least 20 days
    if nt < 20:
        return currentPos

    # compute moving averages
    short_ma = prcSoFar[:, -5:].mean(axis=1)   # last 5 days
    long_ma  = prcSoFar[:, -20:].mean(axis=1)  # last 20 days

    signal = np.where(short_ma > long_ma, 1, -1)

    # determine target position in shares
    latest_price = prcSoFar[:, -1]
    targetPos = np.floor(10_000 * signal / latest_price).astype(int)

    delta = targetPos - currentPos
    currentPos = currentPos + delta

    return currentPos