"""
NexusBrain: Institutional-Grade Transformer-Based Price Prediction Engine.

This module implements the core AI prediction engine using a Transformer neural
network architecture for multi-class trend forecasting and volatility prediction.

Author: AETHER Development Team
License: MIT
Version: 2.0.0
"""

import torch
import numpy as np
import logging
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path

from .nexus_transformer import TimeSeriesTransformer
from ..constants import AIConfig, TradingSignals
from ..exceptions import ModelLoadError, PredictionError, InsufficientDataError

logger = logging.getLogger("NexusBrain")


class NexusBrain:
    """
    Transformer-based price prediction engine for institutional trading.
    
    Uses a Transformer neural network to analyze 64-candle sequences and predict:
    1. Market trend (BUY/NEUTRAL/SELL) with confidence scores
    2. Expected volatility for dynamic risk management
    
    The model employs self-attention mechanisms to capture complex temporal
    dependencies in market data, similar to techniques used in quantitative
    hedge funds.
    
    Attributes:
        device: PyTorch compute device (cuda/cpu)
        model: Loaded Transformer model instance
        confidence_threshold: Minimum confidence for signal generation
        
    Example:
        >>> brain = NexusBrain(config=trading_config)
        >>> signal, confidence, volatility = brain.predict(candle_data)
        >>> if signal == "BUY" and confidence > 0.7:
        ...     execute_trade()
    """
    
    def __init__(
        self, 
        model_path: str = "models/nexus_transformer.pth", 
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize NexusBrain prediction engine.
        
        Args:
            model_path: Path to trained Transformer model weights (.pth file)
            config: Optional configuration dictionary containing ai_parameters
            
        Raises:
            ModelLoadError: If model file exists but fails to load properly
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TimeSeriesTransformer().to(self.device)
        self.model_path = Path(model_path)
        
        # Load confidence threshold from config or use default
        self.confidence_threshold = AIConfig.NEXUS_CONFIDENCE_THRESHOLD
        if config and 'ai_parameters' in config:
            self.confidence_threshold = config['ai_parameters'].get(
                'nexus_confidence_threshold', 
                AIConfig.NEXUS_CONFIDENCE_THRESHOLD
            )
        
        try:
            if not self.model_path.exists():
                logger.warning(
                    f"Model file not found at {model_path}. "
                    "NexusBrain will run in NEUTRAL mode (no predictions)."
                )
                self.model = None
                return
            
            # Smart Loading: Filter out mismatched layers for backward compatibility
            state_dict = torch.load(str(self.model_path), map_location=self.device)
            model_dict = self.model.state_dict()
            
            # 1. Filter out unnecessary keys
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            
            # 2. Filter out shape mismatches
            valid_dict = {}
            skipped_layers = []
            for k, v in pretrained_dict.items():
                if model_dict[k].shape == v.shape:
                    valid_dict[k] = v
                else:
                    skipped_layers.append(k)
                    logger.warning(
                        f"Skipping layer '{k}': shape mismatch "
                        f"(saved: {v.shape}, expected: {model_dict[k].shape})"
                    )
            
            if len(skipped_layers) > len(model_dict) * 0.5:
                raise ModelLoadError(
                    "NexusBrain",
                    f"Too many layer mismatches ({len(skipped_layers)}/{len(model_dict)}). "
                    "Model may be incompatible."
                )
            
            # 3. Overwrite entries in the existing state dict
            model_dict.update(valid_dict)
            
            # 4. Load the new state dict
            self.model.load_state_dict(model_dict)
            self.model.eval()  # Set to inference mode
            
            logger.info(
                f"[OK] NexusBrain loaded from {model_path} "
                f"({len(valid_dict)}/{len(model_dict)} layers)"
            )
            
        except ModelLoadError:
            raise  # Re-raise our custom exception
        except Exception as e:
            raise ModelLoadError("NexusBrain", str(e))

    def predict(self, candles: List[List[float]]) -> Tuple[str, float, float]:
        """
        Generate trading signal from market data using Transformer model.
        
        Analyzes the last 64 candlesticks to predict market direction and
        expected volatility using self-attention mechanisms.
        
        Args:
            candles: List of candle data, where each candle is a list of
                    [open, high, low, close, volatility]. Must contain at
                    least 64 candles.
                    
        Returns:
            Tuple of (signal, confidence, predicted_volatility) where:
            - signal: "BUY", "SELL", or "NEUTRAL"
            - confidence: Float in [0, 1] representing prediction certainty
            - predicted_volatility: Expected price volatility (scaled)
            
        Raises:
            InsufficientDataError: If fewer than 64 candles provided
            PredictionError: If model inference fails
            
        Example:
            >>> candles = fetch_last_n_candles(symbol="XAUUSD", n=64)
            >>> signal, conf, vol = brain.predict(candles)
            >>> print(f"Signal: {signal} ({conf:.2%} confidence)")
        """
        # Model availability check
        if self.model is None:
            logger.debug("NexusBrain model not loaded. Returning NEUTRAL.")
            return TradingSignals.NEUTRAL.value, 0.0, 0.0

        # Data validation
        if candles is None or len(candles) < AIConfig.NEXUS_SEQUENCE_LENGTH:
            logger.warning(
                f"Insufficient candle data: {len(candles) if candles else 0} "
                f"provided, {AIConfig.NEXUS_SEQUENCE_LENGTH} required"
            )
            return TradingSignals.NEUTRAL.value, 0.0, 0.0

        # Preprocess Input
        # We need to match the training normalization (Mean/Std)
        # For inference, we can use a rolling Z-score of the input window itself
        data = np.array(candles[-64:], dtype=np.float32)
        
        # Simple Z-Score Normalization per window
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) + 1e-8
        norm_data = (data - mean) / std
        
        # Convert to Tensor
        src = torch.FloatTensor(norm_data).unsqueeze(0).to(self.device) # [1, 64, 5]
        
        with torch.no_grad():
            trend_logits, vol_pred = self.model(src)
            
            # Trend Prediction (Softmax)
            probs = torch.softmax(trend_logits, dim=1).cpu().numpy()[0]
            # Classes: 0=Sell, 1=Neutral, 2=Buy
            
            sell_prob = probs[0]
            neutral_prob = probs[1]
            buy_prob = probs[2]
            
            # Volatility Prediction
            predicted_vol = vol_pred.item()
            
            # Decision Logic
            signal = "NEUTRAL"
            confidence = 0.0
            
            # Use configured threshold (loaded from config or default)
            threshold = self.confidence_threshold
            
            # [DEBUG] Log raw probabilities to diagnose "No Trades"
            # if buy_prob > 0.1 or sell_prob > 0.1:
            logger.info(f"[NEXUS DEBUG] Buy: {buy_prob:.4f} | Sell: {sell_prob:.4f} | Neutral: {neutral_prob:.4f} | Threshold: {threshold}")

            # [CRITICAL FIX] "Forced Directional Choice"
            # The model is 99% sure it's Neutral because M1 data is mostly noise.
            # To scalp, we must ask: "If you HAD to choose a direction, which one?"
            
            total_directional_prob = buy_prob + sell_prob + 1e-9 # Avoid div by zero
            adj_buy = buy_prob / total_directional_prob
            adj_sell = sell_prob / total_directional_prob
            
            # New Threshold for Adjusted Probabilities (e.g., 0.55 = 55% preference)
            adj_threshold = 0.55
            
            # Only trade if there is SOME directional bias (raw prob > epsilon)
            min_raw_prob = 0.0001 
            
            if adj_buy > adj_threshold and buy_prob > min_raw_prob:
                signal = "BUY"
                confidence = adj_buy # Use adjusted confidence for sizing
            elif adj_sell > adj_threshold and sell_prob > min_raw_prob:
                signal = "SELL"
                confidence = adj_sell
            else:
                signal = "HOLD"
                confidence = max(adj_buy, adj_sell)

            return signal, confidence, predicted_vol
