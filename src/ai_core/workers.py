"""
Worker Agents - Specialized Trading Strategies.

This module implements the Worker Agents that execute specific strategies
assigned by the Supervisor.

1. RangeWorker: Mean Reversion (Buy Low, Sell High)
2. TrendWorker: Momentum (Buy High, Sell Higher)

Author: AETHER Development Team
License: MIT
Version: 5.0.0
"""

import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger("Workers")

class BaseWorker:
    def get_signal(self, market_data: Dict[str, Any]) -> Tuple[str, float, str]:
        """Returns (Action, Confidence, Reason)"""
        raise NotImplementedError

class RangeWorker(BaseWorker):
    """
    Specialist in Sideways Markets.
    Strategy: Aggressive Mean Reversion (Scalping).
    """
    def get_signal(self, market_data: Dict[str, Any]) -> Tuple[str, float, str]:
        rsi = market_data.get('rsi', 50.0)
        
        # Aggressive Scalping: Trade the immediate deviation
        # If RSI < 50, we are in the lower half -> BUY
        # If RSI > 50, we are in the upper half -> SELL
        # Removed deadband to ensure continuous trading as requested
        
        if rsi < 50:
            confidence = (50 - rsi) / 50.0
            # Boost confidence to ensure trade execution
            return "BUY", max(0.7, confidence), f"Range Low (RSI {rsi:.1f})"
        elif rsi > 50:
            confidence = (rsi - 50) / 50.0
            # Boost confidence to ensure trade execution
            return "SELL", max(0.7, confidence), f"Range High (RSI {rsi:.1f})"
            
        # Only hold if exactly 50.0
        return "HOLD", 0.0, "Equilibrium (Exact 50.0)"

class TrendWorker(BaseWorker):
    """
    Specialist in Trending Markets.
    Strategy: Aggressive Trend Following.
    """
    def get_signal(self, market_data: Dict[str, Any]) -> Tuple[str, float, str]:
        trend_strength = market_data.get('trend_strength', 0.0)
        rsi = market_data.get('rsi', 50.0)
        
        # Aggressive Trend Following
        # If Trend is Up (RSI > 50 usually in uptrend) -> BUY
        # If Trend is Down (RSI < 50 usually in downtrend) -> SELL
        
        if rsi > 50:
            return "BUY", max(0.6, trend_strength), f"Uptrend Momentum (RSI {rsi:.1f})"
        elif rsi < 50:
            return "SELL", max(0.6, trend_strength), f"Downtrend Momentum (RSI {rsi:.1f})"
            
        return "HOLD", 0.0, "Neutral"
