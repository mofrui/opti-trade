import numpy as np
from collections import deque

nInst         = 50
SHORT_W       = 5
LONG_W        = 20
CONFIRM_D     = 3
CAP_PER_TICK  = 10000.0
THRESH_DIFF   = 0.005
ATR_WINDOW    = 20

currentPos     = np.zeros(nInst, dtype=int)
signals_hist   = deque(maxlen=CONFIRM_D)
prev_confirmed = np.zeros(nInst, dtype=int)

def getMyPosition(prcSoFar):
    global currentPos, signals_hist, prev_confirmed

    n, t = prcSoFar.shape
    if t < LONG_W+1:
        return currentPos.copy()

    latest = prcSoFar[:, -1]

    short_ma = prcSoFar[:, -SHORT_W:].mean(axis=1)
    long_ma  = prcSoFar[:, -LONG_W :].mean(axis=1)
    diff_pct = (short_ma - long_ma) / long_ma

    hh = prcSoFar[:, -ATR_WINDOW:].max(axis=1)
    ll = prcSoFar[:, -ATR_WINDOW:].min(axis=1)
    prev_close = prcSoFar[:, -ATR_WINDOW-1:-1][:, -1]
    tr1 = hh - ll
    tr2 = np.abs(hh - prev_close)
    tr3 = np.abs(ll - prev_close)
    tr  = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = tr.mean(axis=0) if tr.ndim>1 else tr

    atr_med = np.median(atr)
    trend_mask = (atr > atr_med)

    sig = np.zeros(n, dtype=int)
    sig[diff_pct >  THRESH_DIFF] =  1
    sig[diff_pct < -THRESH_DIFF] = -1
    sig = sig * trend_mask.astype(int)

    signals_hist.append(sig)
    if len(signals_hist) < CONFIRM_D:
        confirmed = prev_confirmed
    else:
        hist = np.stack(signals_hist, axis=0)
        mask = np.all(hist == sig, axis=0)
        confirmed = np.where(mask, sig, prev_confirmed)
    prev_confirmed = confirmed.copy()

    max_shares = np.floor(CAP_PER_TICK / latest).astype(int)
    target     = confirmed * max_shares

    currentPos = target.copy()
    return currentPos.copy()
