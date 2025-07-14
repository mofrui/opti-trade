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
        use_slope: bool = True,
        use_roc: bool = False,
        use_rsi: bool = False,
        use_dynamic_params: bool = True,
    ):
        self.n_inst = n_inst
        self.short_w = short_w
        self.long_w = long_w
        self.confirm_d = confirm_d
        self.cap_per_tick = cap_per_tick
        self.thresh_diff = thresh_diff
        self.atr_window = atr_window

        self.use_slope = use_slope
        self.use_roc = use_roc
        self.use_rsi = use_rsi
        self.use_dynamic_params = use_dynamic_params

        self.current_pos = np.zeros(n_inst, dtype=int)
        self.signals_hist = deque(maxlen=confirm_d)
        self.prev_confirmed = np.zeros(n_inst, dtype=int)

        self.param_table = {
            'low':    dict(short_w=5, long_w=21, thresh_diff=0.002),
            'medium': dict(short_w=7, long_w=31, thresh_diff=0.003),
            'high':   dict(short_w=10, long_w=50, thresh_diff=0.005),
        }

    def _compute_atr(self, prices: np.ndarray) -> np.ndarray:
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
        return (prices[:, -self.short_w:].mean(axis=1) - prices[:, -self.long_w:].mean(axis=1)) / prices[:, -self.long_w:].mean(axis=1)

    def _compute_slope(self, prices: np.ndarray) -> np.ndarray:
        if prices.shape[1] < self.short_w:
            return np.zeros(self.n_inst)
        x = np.arange(self.short_w)
        y = prices[:, -self.short_w:]
        x_mean = x.mean()
        y_mean = y.mean(axis=1, keepdims=True)
        num = ((x - x_mean) * (y - y_mean)).sum(axis=1)
        den = ((x - x_mean)**2).sum()
        return num / den / prices[:, -1]

    def _compute_roc(self, prices: np.ndarray, period: int = 10) -> np.ndarray:
        if prices.shape[1] < period + 1:
            return np.zeros(self.n_inst)
        return (prices[:, -1] - prices[:, -period-1]) / prices[:, -period-1]

    def _compute_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        if prices.shape[1] < period + 1:
            return np.full(self.n_inst, 50.0)
        delta = np.diff(prices[:, -period-1:], axis=1)
        up = np.maximum(delta, 0).mean(axis=1)
        down = np.abs(np.minimum(delta, 0)).mean(axis=1)
        rs = np.divide(up, down, out=np.ones_like(up), where=down != 0)
        return 100 - (100 / (1 + rs))

    def _select_param_set(self, atr: np.ndarray):
        atr_med = np.median(atr)
        p33, p66 = np.percentile(atr, [33, 66])
        regime = 'low' if atr_med < p33 else ('medium' if atr_med < p66 else 'high')
        params = self.param_table[regime]
        self.short_w = params['short_w']
        self.long_w = params['long_w']
        self.thresh_diff = params['thresh_diff']

    def _raw_signal(self, score: np.ndarray, atr: np.ndarray) -> np.ndarray:
        sig = np.zeros(self.n_inst, dtype=int)
        sig[score > self.thresh_diff] = 1
        sig[score < -self.thresh_diff] = -1
        atr_med = np.median(atr)
        return sig * (atr > atr_med).astype(int)

    def _confirm_signal(self, sig: np.ndarray) -> np.ndarray:
        self.signals_hist.append(sig)
        if len(self.signals_hist) < self.confirm_d:
            return self.prev_confirmed.copy()
        hist = np.stack(self.signals_hist, axis=0)
        keep = np.all(hist == sig, axis=0)
        confirmed = np.where(keep, sig, self.prev_confirmed)
        self.prev_confirmed = confirmed.copy()
        return confirmed

    def get_position(self, prices: np.ndarray) -> np.ndarray:
        n, t = prices.shape
        if t < self.long_w + 1:
            return self.current_pos.copy()

        atr = self._compute_atr(prices)

        if self.use_dynamic_params:
            self._select_param_set(atr)

        ma_diff = self._compute_ma_diff(prices)
        score = ma_diff

        if self.use_slope:
            slope = self._compute_slope(prices)
            score += 0.3 * slope

        if self.use_roc:
            roc = self._compute_roc(prices)
            score += 0.2 * roc

        if self.use_rsi:
            rsi = self._compute_rsi(prices)
            score *= (1 - np.abs(rsi - 50) / 50)

        raw = self._raw_signal(score, atr)
        sig = self._confirm_signal(raw)

        latest = prices[:, -1]
        max_shares = np.floor(self.cap_per_tick / latest).astype(int)
        target = sig * max_shares

        self.current_pos = target.copy()
        return self.current_pos.copy()


_strategy = TrendStrategy(n_inst=50)

def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    return _strategy.get_position(prcSoFar)
