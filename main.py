import numpy as np
from collections import deque

nInst = 50
currentPos = np.zeros(nInst, dtype=int)

CONFIRM_DAYS = 2

signals_history = deque(maxlen=CONFIRM_DAYS)

prev_confirmed = np.zeros(nInst, dtype=int)

def getMyPosition(prcSoFar):
    global currentPos, signals_history, prev_confirmed
    nins, nt = prcSoFar.shape
    if nt < 20:
        return currentPos

    short_ma = prcSoFar[:, -5:].mean(axis=1)
    long_ma  = prcSoFar[:, -20:].mean(axis=1)
    today_signal = np.where(short_ma > long_ma, 1, -1)

    signals_history.append(today_signal)

    if len(signals_history) < CONFIRM_DAYS:
        confirmed = prev_confirmed
    else:
        hist = np.stack(signals_history, axis=0)
        mask = np.all(hist == today_signal, axis=0)
        confirmed = np.where(mask, today_signal, prev_confirmed)

    prev_confirmed = confirmed.copy()

    latest_price = prcSoFar[:, -1]
    target = np.floor(10000 * confirmed / latest_price).astype(int)

    delta = target - currentPos
    currentPos += delta

    return currentPos
