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
