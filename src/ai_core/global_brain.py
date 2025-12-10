"""
Global Brain - Inter-Market Correlation Engine (Layer 9)

This module implements "God Mode" intelligence by monitoring correlated assets
(DXY, US10Y, SPX500) to predict Gold (XAUUSD) movements before they happen.

Concept:
- DXY (Dollar) UP -> Gold DOWN (Inverse Correlation)
- US10Y (Yields) UP -> Gold DOWN (Inverse Correlation)
- SPX500 (Risk On) UP -> Gold DOWN (Safe Haven Outflow)

The engine calculates a "Correlation Score" (-1.0 to +1.0) to bias the main trading engine.
"""

import logging
import time
import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass

logger = logging.getLogger("GlobalBrain")

@dataclass
class CorrelationSignal:
    score: float  # -1.0 (Strong Bearish for Gold) to +1.0 (Strong Bullish for Gold)
    driver: str   # Which asset is driving the move (e.g., "DXY_SPIKE")
    confidence: float
    timestamp: float

class GlobalBrain:
    def __init__(self, market_data_manager):
        self.market_data = market_data_manager
        self.correlations = {
            "DXY": -0.85,   # Strong Inverse
            "US10Y": -0.70, # Strong Inverse
            "SPX500": -0.40 # Weak Inverse (Risk On/Off)
        }
        # Thresholds for "Significant Move" (percent change in last 1 min)
        self.thresholds = {
            "DXY": 0.02,    # 0.02% move is significant for currency
            "US10Y": 0.05,
            "SPX500": 0.05
        }
        self.last_prices = {}
        self.last_update = 0
        
    def update_reference_prices(self, prices: Dict[str, float]):
        """Update the baseline prices for calculation."""
        for symbol, price in prices.items():
            self.last_prices[symbol] = price
        self.last_update = time.time()

    def analyze_impact(self, current_prices: Dict[str, float]) -> CorrelationSignal:
        """
        Analyze global markets to predict Gold's next move.
        Returns a signal bias for XAUUSD.
        """
        total_score = 0.0
        primary_driver = "NEUTRAL"
        max_impact = 0.0
        
        for symbol, correlation in self.correlations.items():
            if symbol not in current_prices or symbol not in self.last_prices:
                continue
                
            # Calculate % Change
            curr = current_prices[symbol]
            prev = self.last_prices[symbol]
            if prev == 0: continue
            
            pct_change = ((curr - prev) / prev) * 100.0
            
            # Check if move is significant (Noise Filter)
            threshold = self.thresholds.get(symbol, 0.05)
            if abs(pct_change) < threshold:
                continue
                
            # Calculate Impact Score
            # Example: DXY moves +0.1% (Bullish Dollar). Correlation is -0.85.
            # Impact = 0.1 * -0.85 = -0.085 (Bearish for Gold)
            impact = pct_change * correlation * 10.0 # Multiplier for sensitivity
            
            total_score += impact
            
            if abs(impact) > max_impact:
                max_impact = abs(impact)
                direction = "SURGE" if pct_change > 0 else "DUMP"
                primary_driver = f"{symbol}_{direction}"
                
        # Normalize Score to -1.0 to 1.0
        final_score = np.clip(total_score, -1.0, 1.0)
        
        # Confidence based on how many assets agree
        confidence = min(abs(total_score), 1.0)
        
        if abs(final_score) > 0.3:
            logger.info(f"[GLOBAL_BRAIN] Signal: {final_score:.2f} | Driver: {primary_driver} | DXY Impact: {total_score:.2f}")
            
        return CorrelationSignal(
            score=final_score,
            driver=primary_driver,
            confidence=confidence,
            timestamp=time.time()
        )

    def get_bias(self) -> float:
        """Get the current trading bias (-1.0 to 1.0)."""
        # In a real implementation, this would fetch live prices from MT5
        # For now, we return 0.0 if no data, or the calculated score
        return 0.0
