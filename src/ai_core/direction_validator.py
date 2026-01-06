"""
Direction Validator - 7-Factor Intelligence for Trade Direction Accuracy

This module provides rigorous validation of trade direction to prevent
"Buy when market goes down" scenarios. It acts as a guidance system that
adjusts confidence and can invert signals when necessary.
"""

import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger("DirectionValidator")


@dataclass
class ValidationResult:
    """Result of direction validation."""
    score: float  # 0.0 to 1.0 (Alignment Score)
    confidence_multiplier: float  # 0.5 to 1.5
    should_invert: bool
    passed_factors: int # Kept for compatibility
    total_factors: int # Kept for compatibility
    failed_factors: list # Now used for "Opposing Factors"
    reasoning: str


class DirectionValidator:
    """
    7-Factor 'Market Analyst' for Strategic Directional Advice.
    
    Instead of 'Blocking', this module analyzes the market to determine the
    dominant bias (Bullish vs Bearish) and advises the Trading Engine.
    
    It calculates a Weighted Bias Score (-1.0 to +1.0).
    """
    
    def __init__(self):
        self.validation_history = []
        logger.info("DirectionValidator initialized as Market Analyst")
    
    def validate_direction(self, 
                          direction: str,  # 'BUY' or 'SELL'
                          market_data: Dict,
                          worker_confidence: float) -> ValidationResult:
        """
        Analyze trade direction against market bias.
        """
        # 1. Calculate Biases for each Factor (-1.0 Sell, 0.0 Neutral, 1.0 Buy)
        biases = {}
        
        try:
            biases['trend'] = self._analyze_trend(market_data)
            biases['momentum'] = self._analyze_momentum(market_data)
            biases['levels'] = self._analyze_levels(market_data)
            biases['mtf'] = self._analyze_mtf(market_data)
            biases['flow'] = self._analyze_flow(market_data)
            biases['ai'] = self._analyze_ai(direction, market_data)
            biases['trajectory'] = self._analyze_trajectory(market_data)
        except Exception as e:
            logger.error(f"[ANALYST] Analysis failed: {e}")
            return ValidationResult(0.5, 1.0, False, 0, 0, [], "Analysis Error (Defaulting to Neutral)")

        # 2. Apply Weights (optimized for SCALPING)
        weights = {
            'trajectory': 0.20,
            'trend': 0.20,
            'momentum': 0.15,
            'ai': 0.15,
            'levels': 0.15,
            'mtf': 0.10,
            'flow': 0.05
        }
        
        total_bias = 0.0
        total_weight = 0.0
        details = []
        
        for factor, bias in biases.items():
            w = weights.get(factor, 0.0)
            if w > 0:
                total_bias += bias * w
                total_weight += w
            
                if abs(bias) > 0.3:
                    sentiment = "BULLISH" if bias > 0 else "BEARISH"
                    details.append(f"{factor.upper()}: {sentiment}({bias:.1f})")

        # 3. Normalize (-1.0 to 1.0)
        market_bias = total_bias / total_weight if total_weight > 0 else 0.0
        
        # 4. Compare Market Bias to Proposed Direction
        proposed_numeric = 1.0 if direction == 'BUY' else -1.0
        alignment = market_bias * proposed_numeric
        score = (alignment + 1.0) / 2.0
        
        # 5. Formulate Advice
        confidence_multiplier = 1.0
        should_invert = False
        opposing_factors = [k for k, v in biases.items() if (v * proposed_numeric) < -0.2]
        supporting_factors = [k for k, v in biases.items() if (v * proposed_numeric) > 0.2]
        
        narrative = f"Market Bias: {market_bias:.2f}."
        
        if score > 0.65:
            confidence_multiplier = 1.2
            narrative = f"Strong {direction} Alignment. bias={market_bias:.2f}. Supports: {','.join(supporting_factors[:3])}"
        elif score > 0.40:
            confidence_multiplier = 1.0
            narrative = f"Neutral/Moderate Alignment. bias={market_bias:.2f}."
        elif score > 0.25:
            confidence_multiplier = 0.7
            narrative = f"Weak Alignment (Caution). Market leans opposite. Opposes: {','.join(opposing_factors[:3])}"
        else:
            should_invert = True
            confidence_multiplier = 0.5
            narrative = f"CRITICAL DIVERGENCE! Market strongly opposes {direction}. Suggest INVERSION."

        return ValidationResult(
            score=score,
            confidence_multiplier=confidence_multiplier,
            should_invert=should_invert,
            passed_factors=len(supporting_factors),
            total_factors=len(weights),
            failed_factors=opposing_factors, # Meaningful "Opposing" list
            reasoning=narrative
        )

    # --- ANALYTIC METHODS (Return Bias: -1.0 Sell, 1.0 Buy) ---

    def _analyze_trend(self, market_data: Dict) -> float:
        """Analyze Trend Direction and Strength."""
        trend_str = str(market_data.get('trend', 'NEUTRAL')).upper()
        regime = str(market_data.get('regime', '')).upper()
        
        score = 0.0
        if 'UP' in regime or 'BULL' in trend_str: score += 0.8
        elif 'DOWN' in regime or 'BEAR' in trend_str: score -= 0.8
        
        return score

    def _analyze_momentum(self, market_data: Dict) -> float:
        """Analyze Momentum (RSI + MACD)."""
        rsi = market_data.get('rsi', 50)
        macd = market_data.get('macd', {})
        score = 0.0
        
        # RSI
        if rsi > 70: score -= 0.8 # Overbought -> Bearish Pressure
        elif rsi < 30: score += 0.8 # Oversold -> Bullish Pressure
        elif rsi > 55: score += 0.3 # Mild Bullish
        elif rsi < 45: score -= 0.3 # Mild Bearish
        
        # MACD
        hist = macd.get('histogram', 0)
        if hist > 0: score += 0.2
        elif hist < 0: score -= 0.2
        
        return max(-1.0, min(1.0, score))

    def _analyze_levels(self, market_data: Dict) -> float:
        """Analyze proximity to Support/Resistance."""
        price = market_data.get('current_price', 0)
        if not price: return 0.0
        
        support = market_data.get('support', price - 10)
        resistance = market_data.get('resistance', price + 10)
        
        dist_supp = abs(price - support)
        dist_res = abs(price - resistance)
        total = dist_supp + dist_res
        if total == 0: return 0.0
        
        # If closer to support -> Bullish Bounce (+1)
        # If closer to resistance -> Bearish Rejection (-1)
        
        # Normalize position: -1 (at res) to +1 (at sup)
        # Ratio of distance
        # proximity = (dist_res - dist_supp) / total
        
        # Example: Price 105, Supp 100, Res 110. D_S=5, D_R=5. Total=10. (5-5)/10 = 0. Neutral.
        # Example: Price 101, Supp 100, Res 110. D_S=1, D_R=9. Total=10. (9-1)/10 = 0.8. Bullish.
        
        return (dist_res - dist_supp) / total

    def _analyze_mtf(self, market_data: Dict) -> float:
        """Analyze Multi-Timeframe Alignment."""
        m1 = str(market_data.get('m1_trend', '')).upper()
        m5 = str(market_data.get('m5_trend', '')).upper()
        m15 = str(market_data.get('m15_trend', '')).upper()
        
        score = 0.0
        for t in [m1, m5, m15]:
            if 'UP' in t or 'BULL' in t: score += 0.33
            elif 'DOWN' in t or 'BEAR' in t: score -= 0.33
            
        return score

    def _analyze_ai(self, direction: str, market_data: Dict) -> float:
        """Analyze AI Consensus Bias."""
        oracle_pred = str(market_data.get('oracle_prediction', 'NEUTRAL')).upper()
        if oracle_pred in ('UP', 'BUY'): return 0.8
        if oracle_pred in ('DOWN', 'SELL'): return -0.8
        return 0.0

    def _analyze_trajectory(self, market_data: Dict) -> float:
        """Analyze predicted path of next few candles."""
        traj = market_data.get('trajectory', [])
        current_price = market_data.get('current_price', 0)
        
        if not traj or not current_price or len(traj) == 0:
            return 0.0
            
        end_price = traj[-1]
        pct_change = (end_price - current_price) / current_price if current_price > 0 else 0
        
        if abs(pct_change) < 0.001:
            return 0.0
        elif pct_change > 0.005:
            return 1.0
        elif pct_change < -0.005:
            return -1.0
        else:
            return max(-1.0, min(1.0, pct_change * 200))

    def _analyze_flow(self, market_data: Dict) -> float:
        """Order flow analysis - currently neutral."""
        return 0.0

# Singleton instance
_validator_instance = None

def get_direction_validator() -> DirectionValidator:
    """Get or create the singleton DirectionValidator instance."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = DirectionValidator()
    return _validator_instance
