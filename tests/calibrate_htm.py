import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ai_core.regime_detector import RegimeDetector
from src.ai_core.tick_pressure import TickPressureAnalyzer

def calibrate():
    geo = RegimeDetector()
    phys = TickPressureAnalyzer()
    
    print("=== CALIBRATION RUN ===")
    
    # 1. ENTROPY & HURST
    # A. Random Walk
    np.random.seed(42)
    random_returns = np.random.normal(0, 1, 100)
    prices_random = 1000 + np.cumsum(random_returns)
    candles_random = [{'close': p} for p in prices_random]
    
    entropy_r = geo._calculate_shannon_entropy(candles_random)
    hurst_r = geo._calculate_hurst_exponent(candles_random)
    print(f"RANDOM WALK -> Entropy: {entropy_r:.4f} (Expect >0.8), Hurst: {hurst_r:.4f} (Expect ~0.5)")
    
    # B. Linear Trend
    prices_trend = np.linspace(1000, 1100, 100)
    # Add tiny noise to avoid divide by zero entropy
    prices_trend += np.random.normal(0, 0.001, 100)
    candles_trend = [{'close': p} for p in prices_trend]
    
    entropy_t = geo._calculate_shannon_entropy(candles_trend)
    hurst_t = geo._calculate_hurst_exponent(candles_trend)
    print(f"LINEAR TREND -> Entropy: {entropy_t:.4f} (Expect <0.5), Hurst: {hurst_t:.4f} (Expect >0.5)")
    
    # 2. REYNOLDS NUMBER
    # A. Turbulent (High Vol/Vel, Low Spread)
    phys.ticks.clear()
    start_time = 1000.0
    for i in range(100):
        price = 2000.0 + (i % 2) * 10 
        phys.ticks.append((price, start_time + (i/100.0)))
    spread_points = 1.0
    re_turb = phys.calculate_reynolds_number(spread_points)
    print(f"TURBULENT -> Re: {re_turb:.2f} (Expect >500)")
    
    # B. Laminar (Low Vol/Vel, High Spread)
    phys.ticks.clear()
    for i in range(5): # Low count
        phys.ticks.append((2000.0, start_time + i)) # 1 tick per sec
    spread_points = 10.0 # High Spread
    re_lam = phys.calculate_reynolds_number(spread_points)
    print(f"LAMINAR   -> Re: {re_lam:.2f} (Expect <100)")

if __name__ == '__main__':
    calibrate()
