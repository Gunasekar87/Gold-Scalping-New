import time
from collections import deque

class TickPressureAnalyzer:
    """
    "Holographic" Market View: Simulates Order Flow (Level 2) using Tick Velocity.
    Detects Institutional Aggression vs Retail Noise.
    """
    def __init__(self, window_seconds=5):
        self.window_seconds = window_seconds
        self.ticks = deque() # Stores (price, time)
        
    def add_tick(self, tick):
        """
        Add a new tick to the analyzer.
        Args:
            tick: Dictionary containing 'bid', 'ask', 'time'
        """
        if not tick:
            return
            
        # Use Bid price for pressure analysis (or Mid)
        price = tick.get('bid', 0.0)
        current_time = tick.get('time', time.time())
        
        if price > 0:
            self.ticks.append((price, current_time))
            self._cleanup(current_time)
        
    def _cleanup(self, current_time):
        while self.ticks and (current_time - self.ticks[0][1] > self.window_seconds):
            self.ticks.popleft()
            
    def get_pressure_metrics(self, point_value=0.01):
        """
        Calculates Tick Pressure (Aggression).
        Returns: Dictionary with pressure metrics
        """
        if len(self.ticks) < 5:
            return {
                'pressure_score': 0.0,
                'intensity': 'LOW',
                'dominance': 'NEUTRAL',
                'state': 'INSUFFICIENT_DATA'
            }
            
        start_price = self.ticks[0][0]
        end_price = self.ticks[-1][0]
        price_change = end_price - start_price
        
        tick_count = len(self.ticks)
        duration = self.ticks[-1][1] - self.ticks[0][1]
        if duration <= 0: duration = 0.001
        
        # Velocity = Ticks per Second (Speed of orders)
        velocity = tick_count / duration
        
        # Normalized Price Change (in Points)
        price_delta_points = price_change / point_value
        
        # Pressure = Price Delta * Velocity
        # Example: +10 points * 5 ticks/sec = +50 (Strong Buy)
        # Example: +1 points * 50 ticks/sec = +50 (Massive Absorption/Buy)
        pressure = price_delta_points * velocity
        
        state = "NEUTRAL"
        intensity = "NORMAL"
        dominance = "NEUTRAL"
        
        # Thresholds (Tuned for Gold)
        if pressure > 50.0: 
            state = "INSTITUTIONAL_BUY"
            intensity = "HIGH"
            dominance = "BUY"
        elif pressure > 15.0: 
            state = "STRONG_BUY"
            intensity = "MEDIUM"
            dominance = "BUY"
        elif pressure < -50.0: 
            state = "INSTITUTIONAL_SELL"
            intensity = "HIGH"
            dominance = "SELL"
        elif pressure < -15.0: 
            state = "STRONG_SELL"
            intensity = "MEDIUM"
            dominance = "SELL"
        elif velocity > 10.0 and abs(price_delta_points) < 2.0: 
            state = "ABSORPTION_FIGHT" # High volume, no movement (Trap)
            intensity = "HIGH"
            dominance = "NEUTRAL"
        elif velocity < 1.0:
            state = "LOW_LIQUIDITY"
            intensity = "LOW"
            
        return {
            'pressure_score': pressure,
            'intensity': intensity,
            'dominance': dominance,
            'state': state,
            'velocity': velocity
        }
