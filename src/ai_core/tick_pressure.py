import time
from collections import deque

class TickPressureAnalyzer:
    """
    "Holographic" Market View: Simulates Order Flow (Level 2) using Tick Velocity.
    Detects Institutional Aggression vs Retail Noise.
    
    ENHANCEMENT 2: Added Order Flow Imbalance Analysis (Jan 4, 2026)
    """
    def __init__(self, window_seconds=5):
        self.window_seconds = window_seconds
        self.ticks = deque() # Stores (price, time)
        
        # ENHANCEMENT 2: Order Flow Imbalance Tracking
        self.buy_volume_buffer = deque(maxlen=100)
        self.sell_volume_buffer = deque(maxlen=100)
        self.max_buffer_size = 100
        
    def add_tick(self, tick):
        """
        Add a new tick to the analyzer.
        Args:
            tick: Dictionary containing 'bid', 'ask', 'time'
        """
        if not tick:
            return
            
        # Use Bid price for pressure analysis.
        # IMPORTANT: Use local wall-clock time for timing/velocity.
        # MT5 tick timestamps are often coarse (seconds) which can make duration ~0
        # and artificially saturate velocity/pressure.
        price = tick.get('bid', 0.0)
        current_time = time.time()
        
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
    
    # ============================================================================
    # ENHANCEMENT 2: Order Flow Imbalance Analysis
    # Added: January 4, 2026
    # Purpose: Detect buy vs sell aggressor volume for better entry timing
    # ============================================================================
    
    def analyze_order_flow(self, tick):
        """
        Analyze buy vs sell volume imbalance from tick data.
        
        Detects aggressor side (who crossed the spread) and tracks volume imbalance.
        
        Args:
            tick: Dictionary with 'last', 'bid', 'ask', 'volume' keys
            
        Returns:
            Dictionary with:
            - order_flow_imbalance: -1 to 1 (negative = sell pressure, positive = buy pressure)
            - buy_pressure: Ratio of buy volume (0-1)
            - sell_pressure: Ratio of sell volume (0-1)
        """
        if not tick:
            return {
                'order_flow_imbalance': 0.0,
                'buy_pressure': 0.5,
                'sell_pressure': 0.5
            }
        
        try:
            last_price = tick.get('last', tick.get('bid', 0))
            bid = tick.get('bid', 0)
            ask = tick.get('ask', 0)
            volume = tick.get('volume', 1)
            
            # Detect aggressor side (who crossed the spread)
            # If last price >= ask, it's a buy aggressor (market buy order)
            # If last price <= bid, it's a sell aggressor (market sell order)
            
            if last_price >= ask and ask > 0:
                # Buy aggressor - someone hit the ask
                self.buy_volume_buffer.append(volume)
            elif last_price <= bid and bid > 0:
                # Sell aggressor - someone hit the bid
                self.sell_volume_buffer.append(volume)
            else:
                # Mid-price trade or limit order fill - neutral
                # Add half to each side
                self.buy_volume_buffer.append(volume / 2)
                self.sell_volume_buffer.append(volume / 2)
            
            # Calculate imbalance from buffers
            buy_vol = sum(self.buy_volume_buffer) if self.buy_volume_buffer else 0
            sell_vol = sum(self.sell_volume_buffer) if self.sell_volume_buffer else 0
            total_vol = buy_vol + sell_vol
            
            if total_vol > 0:
                # Imbalance: -1 (all sell) to +1 (all buy)
                imbalance = (buy_vol - sell_vol) / total_vol
                buy_pressure = buy_vol / total_vol
                sell_pressure = sell_vol / total_vol
            else:
                imbalance = 0.0
                buy_pressure = 0.5
                sell_pressure = 0.5
            
            return {
                'order_flow_imbalance': float(imbalance),
                'buy_pressure': float(buy_pressure),
                'sell_pressure': float(sell_pressure),
                'buy_volume': float(buy_vol),
                'sell_volume': float(sell_vol)
            }
            
        except Exception as e:
            # On error, return neutral
            return {
                'order_flow_imbalance': 0.0,
                'buy_pressure': 0.5,
                'sell_pressure': 0.5
            }
    
    def get_combined_analysis(self, tick, point_value=0.01):
        """
        Get both pressure metrics and order flow analysis in one call.
        
        Args:
            tick: Current tick data
            point_value: Point value for pressure calculation
            
        Returns:
            Combined dictionary with all metrics
        """
        pressure = self.get_pressure_metrics(point_value)
        order_flow = self.analyze_order_flow(tick)
        
        # Combine both analyses
        combined = {
            **pressure,
            **order_flow
        }
        
        return combined
