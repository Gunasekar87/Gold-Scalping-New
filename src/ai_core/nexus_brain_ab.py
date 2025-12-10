"""
NexusBrain A/B Testing Wrapper
===============================

Wraps NexusBrain to support A/B testing between Legacy and Enhanced models.

Maintains backward compatibility with existing code while enabling
parallel evaluation of both model versions.

Author: AETHER Development Team
Version: 3.0.0 - Phase 1B
Date: November 25, 2025
"""

import sys
import logging
from typing import Tuple, List, Dict, Optional
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.ai_core.ab_testing_manager import ABTestingManager
from src.constants import TradingSignals

logger = logging.getLogger("NexusBrainAB")


class NexusBrainAB:
    """
    A/B Testing wrapper for NexusBrain
    
    Maintains compatibility with existing NexusBrain interface while
    routing predictions through A/B testing manager.
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        enable_ab_testing: bool = True,
        enhanced_allocation: float = 0.20
    ):
        """
        Initialize NexusBrain with A/B testing support
        
        Args:
            config: Configuration dictionary (for compatibility)
            enable_ab_testing: If True, use A/B testing; if False, use legacy only
            enhanced_allocation: Percentage of predictions using Enhanced model
        """
        self.config = config
        self.enable_ab_testing = enable_ab_testing
        
        if self.enable_ab_testing:
            logger.info("NexusBrain initialized with A/B TESTING enabled")
            logger.info(f"  Enhanced allocation: {enhanced_allocation*100:.1f}%")
            logger.info(f"  Legacy allocation: {(1-enhanced_allocation)*100:.1f}%")
            
            self.ab_manager = ABTestingManager(
                enhanced_allocation=enhanced_allocation,
                enable_auto_rollback=True,
                rollback_threshold=0.02  # 2% Sharpe difference triggers rollback
            )
        else:
            logger.info("NexusBrain initialized in LEGACY-ONLY mode")
            self.ab_manager = None
        
        # Track which model was used for last prediction (for outcome recording)
        self.last_model_used = None
        self.last_prediction_class = None
    
    def predict(
        self,
        candles: List[Dict],
        force_model: Optional[str] = None
    ) -> Tuple[str, float, float]:
        """
        Generate trading signal (compatible with original NexusBrain interface)
        
        Args:
            candles: List of candle dictionaries with keys: open, high, low, close, volume
            force_model: Optional, force specific model ('legacy' or 'enhanced')
            
        Returns:
            Tuple of (signal, confidence, volatility) where:
            - signal: "BUY", "SELL", or "NEUTRAL"
            - confidence: Float in [0, 1]
            - volatility: Predicted volatility
        """
        if not self.enable_ab_testing:
            # Legacy-only mode - not implemented yet, would use original NexusBrain
            logger.warning("Legacy-only mode not yet implemented, using A/B manager")
        
        if self.ab_manager is None:
            logger.warning("A/B manager not initialized, returning HOLD")
            return TradingSignals.HOLD.value, 0.0, 0.0
        
        # Get prediction from A/B manager
        try:
            result = self.ab_manager.predict(candles, force_model=force_model)
            
            # Store for outcome recording
            self.last_model_used = result['model']
            self.last_prediction_class = result['prediction']
            
            # Convert prediction class to signal string
            # Note: Using HOLD instead of NEUTRAL for compatibility with Council
            signal_map = {
                0: TradingSignals.SELL.value,
                1: TradingSignals.HOLD.value,  # Council expects HOLD, not NEUTRAL
                2: TradingSignals.BUY.value
            }
            
            signal = signal_map.get(result['prediction'], TradingSignals.HOLD.value)
            confidence = result['confidence']
            volatility = result.get('volatility', 0.0)
            
            logger.debug(
                f"Prediction: {signal} ({confidence:.2%} confidence) "
                f"using {result['model']} model ({result.get('features_used', 0)} features)"
            )
            
            return signal, confidence, volatility
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return TradingSignals.HOLD.value, 0.0, 0.0
    
    def record_outcome(
        self,
        actual_signal: str,
        pnl: float = 0.0
    ):
        """
        Record the outcome of the last prediction for A/B testing metrics
        
        Args:
            actual_signal: Actual market direction ("BUY", "SELL", or "NEUTRAL")
            pnl: Profit/Loss from this trade
        """
        if self.ab_manager is None or self.last_model_used is None:
            return
        
        # Convert signal string to class
        signal_map = {
            TradingSignals.SELL.value: 0,
            TradingSignals.NEUTRAL.value: 1,
            TradingSignals.BUY.value: 2
        }
        
        actual_class = signal_map.get(actual_signal, 1)  # Default to NEUTRAL
        
        # Record outcome
        self.ab_manager.record_outcome(
            model_name=self.last_model_used,
            predicted_class=self.last_prediction_class,
            actual_class=actual_class,
            pnl=pnl
        )
    
    def get_performance_report(self) -> Dict:
        """Get A/B testing performance comparison"""
        if self.ab_manager is None:
            return {}
        return self.ab_manager.get_performance_report()
    
    def print_performance_report(self):
        """Print A/B testing performance report"""
        if self.ab_manager is None:
            logger.warning("A/B testing not enabled")
            return
        self.ab_manager.print_report()
    
    def save_performance_report(self, filepath: str = "models/ab_test_report.json"):
        """Save A/B testing performance report to file"""
        if self.ab_manager is None:
            logger.warning("A/B testing not enabled")
            return
        self.ab_manager.save_report(filepath)


if __name__ == "__main__":
    """Test the A/B testing wrapper"""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    import numpy as np
    
    # Initialize
    brain = NexusBrainAB(
        enable_ab_testing=True,
        enhanced_allocation=0.20
    )
    
    # Create dummy candles
    dummy_candles = []
    base_price = 4000.0
    for i in range(102):
        candle = {
            'time': i,
            'open': base_price + np.random.randn() * 5,
            'high': base_price + abs(np.random.randn()) * 10,
            'low': base_price - abs(np.random.randn()) * 10,
            'close': base_price + np.random.randn() * 5,
            'volume': 1000 + np.random.randint(-200, 200),
            'tick_volume': 1000
        }
        dummy_candles.append(candle)
    
    # Test predictions
    print("\nTesting predictions...")
    for i in range(20):
        signal, confidence, volatility = brain.predict(dummy_candles)
        print(f"\nPrediction {i+1}: {signal} ({confidence:.2%} confidence, vol={volatility:.4f})")
        
        # Simulate outcome
        actual = np.random.choice([TradingSignals.SELL.value, TradingSignals.NEUTRAL.value, TradingSignals.BUY.value])
        pnl = np.random.randn() * 100
        brain.record_outcome(actual, pnl)
    
    # Print report
    print("\n" + "="*70)
    brain.print_performance_report()
    brain.save_performance_report()
    
    print("\n✓ NexusBrain A/B wrapper test complete")
