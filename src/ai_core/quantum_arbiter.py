import numpy as np
import logging
from collections import deque

logger = logging.getLogger("QuantumArbiter")

class QuantumArbiter:
    """
    The 'Quantum' Agent: Uses Advanced Physics-Based Filtering (Kalman Filter)
    to estimate the 'True State' of the market amidst noise.
    
    Upgrade: Replaced simple Z-Score with a 1-Dimensional Kalman Filter.
    This allows for instantaneous adaptation to changing correlations without
    the lag of a moving average.
    """
    def __init__(self, window_size=100, z_threshold=2.0):
        self.window_size = window_size
        self.z_threshold = z_threshold
        
        # Memory
        self.ratios = deque(maxlen=window_size)
        
        # --- KALMAN FILTER STATE ---
        # x: State Estimate (The "True" Ratio)
        # P: Estimate Covariance (Error in the estimate)
        # Q: Process Noise Covariance (How much the true ratio jumps around)
        # R: Measurement Noise Covariance (How much market noise exists)
        self.x = 0.0 
        self.P = 1.0
        self.Q = 1e-5 # Small process noise (Ratio evolves slowly)
        self.R = 1e-3 # Moderate measurement noise (Market is noisy)
        self.first_run = True

    def update(self, price_a, price_b):
        """
        Updates the Kalman Filter with new price data.
        Returns: (Signal, Confidence, Metadata)
        """
        if price_b == 0: return "NEUTRAL", 0.0, {}
        
        # Measurement (Observed Ratio)
        z = price_a / price_b
        self.ratios.append(z)
        
        # Initialize on first run
        if self.first_run:
            self.x = z
            self.first_run = False
            return "NEUTRAL", 0.0, {"ratio": z, "kalman_est": z}
            
        # --- KALMAN FILTER STEP ---
        # 1. Predict
        # x_pred = x_prev (Assume constant state model for the ratio)
        # P_pred = P_prev + Q
        x_pred = self.x
        P_pred = self.P + self.Q
        
        # 2. Update
        # K = P_pred / (P_pred + R) (Kalman Gain)
        # x_new = x_pred + K * (z - x_pred)
        # P_new = (1 - K) * P_pred
        K = P_pred / (P_pred + self.R)
        self.x = x_pred + K * (z - x_pred)
        self.P = (1 - K) * P_pred
        
        # Calculate Deviation (Innovation)
        # How far is the current measurement from our "True State" estimate?
        deviation = z - self.x
        
        # Calculate Dynamic Standard Deviation (Volatility of the deviation)
        # We use a small window of recent deviations to normalize
        std_dev = np.std(self.ratios) if len(self.ratios) > 10 else 0.001
        if std_dev == 0: std_dev = 0.001
        
        # Z-Score based on Kalman Estimate
        z_score = deviation / std_dev
        
        meta = {
            "ratio": z,
            "kalman_est": self.x,
            "deviation": deviation,
            "z_score": z_score,
            "kalman_gain": K
        }
        
        # Logic:
        # If Z > Threshold: Price A is artificially high relative to True Ratio.
        confidence = min(abs(z_score) / 4.0, 1.0)
        
        if z_score > self.z_threshold:
            return "SELL_A_BUY_B", confidence, meta
        elif z_score < -self.z_threshold:
            return "BUY_A_SELL_B", confidence, meta
            
        return "NEUTRAL", 0.0, meta
