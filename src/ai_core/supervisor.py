"""
Supervisor Agent - The "Boss" of the Hierarchical AI System.

This module implements the Supervisor agent which is responsible for:
1. Detecting the current Market Regime (Range, Trend, Chaos)
2. Routing control to the appropriate Worker Agent
3. Managing global risk parameters based on regime

Author: AETHER Development Team
License: MIT
Version: 5.0.0
"""

import logging
from typing import Dict, Any
from dataclasses import dataclass

logger = logging.getLogger("Supervisor")

@dataclass
class Regime:
    name: str
    confidence: float
    description: str

class Supervisor:
    """
    The Supervisor Agent determines the market regime and selects the active strategy.
    """
    
    def __init__(self):
        self.current_regime = "UNKNOWN"
        logger.info("[SUPERVISOR] Agent Initialized")

    def detect_regime(self, market_data: Dict[str, Any]) -> Regime:
        """
        Analyze market data to classify the current regime.
        
        Logic:
        - Low Volatility + Low Trend Strength = RANGE
        - High Volatility + High Trend Strength = TREND
        - Extreme Volatility = CHAOS
        """
        try:
            atr = market_data.get('atr', 0.0)
            trend_strength = market_data.get('trend_strength', 0.0) # 0.0 to 1.0 (ADX proxy)
            volatility_ratio = market_data.get('volatility_ratio', 1.0)
            
            # [PHASE 1 DATA] Macro Context
            macro_context = market_data.get('macro_context', [0.0, 0.0])
            usd_velocity = abs(macro_context[0])
            
            # Classification Logic
            if volatility_ratio > 2.5 or usd_velocity > 5.0:
                return Regime("CHAOS", 1.0, "Extreme Volatility / News Event")
                
            if trend_strength > 0.3:
                return Regime("TREND", trend_strength, "Strong Directional Movement")
                
            return Regime("RANGE", 1.0 - trend_strength, "Sideways / Mean Reversion")
            
        except Exception as e:
            logger.error(f"Regime detection failed: {e}")
            return Regime("RANGE", 0.0, "Fallback due to error")

    def get_active_worker(self, regime: Regime) -> str:
        """
        Select the appropriate worker for the detected regime.
        """
        if regime.name == "TREND":
            return "TREND_WORKER"
        elif regime.name == "RANGE":
            return "RANGE_WORKER"
        else:
            return "DEFENSIVE_WORKER"
