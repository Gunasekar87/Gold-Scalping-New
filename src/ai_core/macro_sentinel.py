import numpy as np
import logging
from datetime import datetime, time as dtime
import pytz
from enum import Enum

logger = logging.getLogger("MacroSentinel")

class MarketRegime(Enum):
    RANGING = "RANGING"       # Low ADX, Normal Volatility -> Grid Friendly
    TRENDING_UP = "TRENDING_UP" # High ADX, Price > MA -> Trend Following Only
    TRENDING_DOWN = "TRENDING_DOWN" # High ADX, Price < MA -> Trend Following Only
    VOLATILE = "VOLATILE"     # Extreme BB Width -> Sniper Mode (Reduced Risk)
    UNKNOWN = "UNKNOWN"

class MacroSentinel:
    def __init__(self, volatility_threshold=5.0):
        self.news_flag = "GREEN"
        self.volatility_threshold = volatility_threshold # Multiplier of normal ATR
        
        # Market Hours (UTC)
        # Forex typically closes Friday 22:00 UTC and opens Sunday 22:00 UTC
        self.market_close_day = 4 # Friday
        self.market_close_hour = 21 # 21:00 UTC (Buffer for 22:00)
        
        self.market_open_day = 6 # Sunday
        self.market_open_hour = 22 # 22:00 UTC
        
        # Major Holidays (Month, Day)
        self.holidays = [
            (1, 1),   # New Year's Day
            (12, 25), # Christmas
        ]

    def detect_regime(self, candles: list) -> MarketRegime:
        """
        The Chameleon: Detects the current market regime (Ranging, Trending, Volatile).
        
        Args:
            candles: List of dicts [{'close': float, 'high': float, 'low': float}, ...]
                     Needs at least 20 candles.
        """
        if not candles or len(candles) < 20:
            return MarketRegime.UNKNOWN

        try:
            # Extract arrays
            closes = np.array([c['close'] for c in candles])
            highs = np.array([c['high'] for c in candles])
            lows = np.array([c['low'] for c in candles])
            
            # 1. Calculate Volatility (Bollinger Band Width)
            period = 20
            sma = np.mean(closes[-period:])
            std = np.std(closes[-period:])
            upper_band = sma + (2 * std)
            lower_band = sma - (2 * std)
            
            bb_width = (upper_band - lower_band) / sma
            
            # If BB Width is extreme (> 0.5% for Forex/Gold on M1 is huge, adjust threshold as needed)
            # For XAUUSD, 0.5% is approx $10-15 move.
            if bb_width > 0.005: 
                return MarketRegime.VOLATILE

            # 2. Calculate Trend Strength (Simplified ADX-like)
            # We'll use a simpler slope + consistency check for speed
            
            # Calculate SMA 50 (if enough data) or SMA 20
            ma_short = np.mean(closes[-10:])
            ma_long = np.mean(closes[-20:])
            
            # Trend Strength: Difference between MAs normalized by price
            trend_strength = abs(ma_short - ma_long) / ma_long
            
            # Threshold for "Trending" (0.05% deviation)
            is_trending = trend_strength > 0.0005 
            
            if is_trending:
                if ma_short > ma_long:
                    return MarketRegime.TRENDING_UP
                else:
                    return MarketRegime.TRENDING_DOWN
            
            # Default to Ranging
            return MarketRegime.RANGING

        except Exception as e:
            logger.error(f"Regime detection failed: {e}")
            return MarketRegime.UNKNOWN

    def is_market_open(self):
        """
        Checks if the Forex market is currently open.
        Returns: True if Open, False if Closed (Weekend/Holiday).
        """
        now_utc = datetime.now(pytz.utc)
        
        # 1. Check Holidays
        if (now_utc.month, now_utc.day) in self.holidays:
            return False, "Holiday"
            
        # 2. Check Weekend
        weekday = now_utc.weekday() # Mon=0, Sun=6
        hour = now_utc.hour
        
        # Friday after Close
        if weekday == self.market_close_day and hour >= self.market_close_hour:
            return False, "Weekend (Friday Close)"
            
        # Saturday (All Day)
        if weekday == 5:
            return False, "Weekend (Saturday)"
            
        # Sunday before Open
        if weekday == self.market_open_day and hour < self.market_open_hour:
            return False, "Weekend (Sunday Pre-Open)"
            
        return True, "Market Open"

    def check_news(self):
        # Placeholder for FinBERT or Calendar API
        # In a real implementation, this would check ForexFactory or similar
        # For now, we return GREEN unless manually flagged
        return self.news_flag

    def set_news_flag(self, status):
        self.news_flag = status

    def analyze_market_condition(self, tick_history):
        """
        Analyzes recent price action to detect 'Flash' events or News Spikes.
        Returns: "SAFE" or "DANGEROUS"
        """
        if len(tick_history) < 20:
            return "SAFE" # Not enough data yet
            
        # Convert to numpy array
        if len(tick_history) > 0 and isinstance(tick_history[0], dict):
            # Extract close prices if we have candle data
            prices = np.array([c.get('close', 0.0) for c in tick_history])
        else:
            # Assume it's already a list of prices
            prices = np.array(tick_history)
        
        # Calculate Returns (Percentage change)
        returns = np.diff(prices)
        
        # Calculate Volatility (Standard Deviation of returns)
        current_volatility = np.std(returns)
        
        # If we had a rolling average of volatility, we could compare.
        # For now, we use a heuristic: If the last tick moved > 3x the average move of the last 20.
        avg_move = np.mean(np.abs(returns))
        last_move = np.abs(returns[-1])
        
        if last_move > (avg_move * self.volatility_threshold):
            logger.warning(f"High Volatility Detected! Move: {last_move:.5f} vs Avg: {avg_move:.5f}")
            return "DANGEROUS"
            
        return "SAFE"
