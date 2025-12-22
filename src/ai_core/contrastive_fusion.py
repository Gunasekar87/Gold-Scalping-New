import logging
import numpy as np
import time
from typing import Dict, List

logger = logging.getLogger("ContrastiveFusion")

class ContrastiveFusion:
    """
    Multi-Modal Signal Validator.
    Uses Contrastive Learning principles to align disparate data sources:
    1. Tick Velocity (Micro-structure)
    2. Candle Patterns (Price Action)
    3. Order Book Imbalance (Liquidity)
    
    If these modalities 'agree' (high cosine similarity), the signal is valid.
    If they diverge, the signal is rejected as noise.
    """
    def __init__(self):
        self.weights = {
            "tick_velocity": 0.4,
            "candle_pattern": 0.4,
            "order_book": 0.2
        }

        # Allow silencing of conflict logs via env (AETHER_FUSION_LOG=0)
        self._log_enabled = str(os.getenv("AETHER_FUSION_LOG", "1")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

        # Log-throttling for noisy conflict warnings (behavior unchanged; only reduces spam)
        self._conflict_log_interval_s = 5.0
        self._last_conflict_log_ts = 0.0
        self._conflict_suppressed = 0

    def compute_coherence(self, signals: Dict[str, float]) -> float:
        """
        Calculates a coherence score (0.0 to 1.0) for a set of normalized signals (-1.0 to 1.0).
        
        Args:
            signals: Dict with keys 'tick_velocity', 'candle_pattern', 'order_book'.
                     Values should be normalized between -1 (Strong Sell) and 1 (Strong Buy).
        
        Returns:
            float: Coherence score. High score means all signals point in the same direction.
        """
        # Extract vectors
        vec = np.array([
            signals.get("tick_velocity", 0.0),
            signals.get("candle_pattern", 0.0),
            signals.get("order_book", 0.0)
        ])
        
        # 1. Directional Agreement Check
        # If some are positive and some are negative, coherence drops drastically
        signs = np.sign(vec)
        # Filter out zeros for sign check
        non_zero_signs = signs[signs != 0]
        
        if len(non_zero_signs) > 0 and not np.all(non_zero_signs == non_zero_signs[0]):
            # Conflict detected (e.g. Price says Buy, OrderBook says Sell)
            if self._log_enabled:
                now = time.monotonic()
                if now - self._last_conflict_log_ts >= self._conflict_log_interval_s:
                    if self._conflict_suppressed:
                        logger.warning(
                            f"[FUSION] Signal Conflict Detected (throttled): {self._conflict_suppressed} repeats suppressed"
                        )
                        self._conflict_suppressed = 0
                    logger.warning(f"[FUSION] Signal Conflict Detected: {signals}")
                    self._last_conflict_log_ts = now
                else:
                    self._conflict_suppressed += 1
            return 0.1 # Penalty for divergence

        # 2. Magnitude Alignment (Weighted Average)
        # If direction is agreed, how strong is the consensus?
        weighted_sum = (
            abs(vec[0]) * self.weights["tick_velocity"] +
            abs(vec[1]) * self.weights["candle_pattern"] +
            abs(vec[2]) * self.weights["order_book"]
        )
        
        return min(1.0, weighted_sum)

    def validate_signal(self, raw_signal: float, coherence: float) -> float:
        """
        Adjusts the raw signal confidence based on coherence.
        """
        if coherence < 0.5:
            return 0.0 # Reject signal
        
        # Boost high coherence signals
        boost = 1.2 if coherence > 0.8 else 1.0
        return min(1.0, raw_signal * boost)
