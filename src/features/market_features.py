from collections import deque
from typing import Optional

class RollingStats:
    def __init__(self, window=60):
        self.window = window
        self.buf = deque(maxlen=window)

    def push(self, x: float):
        self.buf.append(float(x))

    def mean(self) -> float:
        if not self.buf: return 0.0
        return sum(self.buf) / max(1, len(self.buf))

    def std(self) -> float:
        if not self.buf: return 0.0
        m = self.mean()
        return (sum((x - m)**2 for x in self.buf) / max(1, len(self.buf))) ** 0.5

def spread_atr(spread_points: float, atr_points: float) -> float:
    atr_points = max(1e-9, float(atr_points))
    return float(spread_points) / atr_points

def zscore(x: float, mean: float, std: float) -> float:
    std = max(1e-9, std)
    return (float(x) - float(mean)) / std

def simple_breakout_quality(high: float, low: float, close: float) -> float:
    rng = max(1e-9, high - low)
    clv = (close - low) / rng  # 0..1
    return max(0.0, min(1.0, clv))

def simple_regime_trend(slope: float, atr: float) -> float:
    atr = max(1e-9, atr)
    norm = min(1.0, abs(slope) / atr)  # crude normalization
    return norm


def linear_regression_slope(values: list[float]) -> float:
    """Return slope per index using simple OLS on x=0..n-1."""
    n = len(values)
    if n < 2:
        return 0.0
    sum_x = (n - 1) * n / 2
    sum_xx = (n - 1) * n * (2 * n - 1) / 6
    sum_y = float(sum(values))
    sum_xy = float(sum(i * v for i, v in enumerate(values)))
    denom = (n * sum_xx - sum_x * sum_x)
    if denom == 0:
        return 0.0
    return (n * sum_xy - sum_x * sum_y) / denom


def simple_structure_break(candles: list[dict], lookback: int = 20) -> float:
    """Detect a simple range break using real candles.

    Returns:
        +1.0 if last close > prior lookback high
        -1.0 if last close < prior lookback low
        0.0 otherwise
    """
    if not candles or len(candles) < 3:
        return 0.0

    # Use the most recent candle as the "break" candle, compare to prior range.
    last = candles[-1]
    prior = candles[:-1]
    lb = max(2, min(int(lookback), len(prior)))
    window = prior[-lb:]

    prior_high = max(c['high'] for c in window)
    prior_low = min(c['low'] for c in window)
    close = float(last.get('close'))

    if close > prior_high:
        return 1.0
    if close < prior_low:
        return -1.0
    return 0.0
