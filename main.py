import numpy as np
from collections import deque

class TrendStrategy:
    def __init__(
        self,
        n_inst: int,
        short_w: int = 7,
        long_w: int = 31,
        confirm_d: int = 3,
        cap_per_tick: float = 10000.0,
        thresh_diff: float = 0.003,
        atr_window: int = 20,
    ):
        self.n_inst = n_inst
        self.short_w = short_w
        self.long_w = long_w
        self.confirm_d = confirm_d
        self.cap_per_tick = cap_per_tick
        self.thresh_diff = thresh_diff
        self.atr_window = atr_window

        # internal state
        self.current_pos = np.zeros(n_inst, dtype=int)
        self.signals_hist = deque(maxlen=confirm_d)
        self.prev_confirmed = np.zeros(n_inst, dtype=int)

    def _compute_atr(self, prices: np.ndarray) -> np.ndarray:
        """Compute one-period True Range for each instrument."""
        w = self.atr_window
        if prices.shape[1] < w + 1:
            return np.zeros(self.n_inst)

        high = prices[:, -w:].max(axis=1)
        low = prices[:, -w:].min(axis=1)
        prev_close = prices[:, -w-1:-1][:, -1]

        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        return np.maximum(np.maximum(tr1, tr2), tr3)

    def _compute_ma_diff(self, prices: np.ndarray) -> np.ndarray:
        """Compute percentage difference between short and long moving averages."""
        short_ma = prices[:, -self.short_w:].mean(axis=1)
        long_ma = prices[:, -self.long_w:].mean(axis=1)
        return (short_ma - long_ma) / long_ma

    def _raw_signal(self, ma_diff: np.ndarray, atr: np.ndarray) -> np.ndarray:
        """Generate raw buy/sell signal and filter by ATR median."""
        sig = np.zeros(self.n_inst, dtype=int)
        sig[ma_diff > self.thresh_diff] = 1
        sig[ma_diff < -self.thresh_diff] = -1

        atr_med = np.median(atr)
        return sig * (atr > atr_med).astype(int)

    def _confirm_signal(self, sig: np.ndarray) -> np.ndarray:
        """
        Keep signal only if it repeats for confirm_d days;
        otherwise keep last confirmed.
        """
        self.signals_hist.append(sig)
        if len(self.signals_hist) < self.confirm_d:
            return self.prev_confirmed.copy()

        hist = np.stack(self.signals_hist, axis=0)
        keep = np.all(hist == sig, axis=0)
        confirmed = np.where(keep, sig, self.prev_confirmed)
        self.prev_confirmed = confirmed.copy()
        return confirmed

    def get_position(self, prices: np.ndarray) -> np.ndarray:
        """
        Main entry: given price history (n_inst × t),
        return target positions for today.
        """
        n, t = prices.shape
        if t < self.long_w + 1:
            return self.current_pos.copy()

        ma_diff = self._compute_ma_diff(prices)
        atr = self._compute_atr(prices)
        raw = self._raw_signal(ma_diff, atr)
        sig = self._confirm_signal(raw)

        latest = prices[:, -1]
        max_shares = np.floor(self.cap_per_tick / latest).astype(int)
        target = sig * max_shares

        self.current_pos = target.copy()
        return self.current_pos.copy()


# instantiate once for eval.py
_strategy = TrendStrategy(n_inst=50)

def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    return _strategy.get_position(prcSoFar)
