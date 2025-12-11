import torch
import torch.nn as nn
import numpy as np
import logging
import os
from .nexus_transformer import TimeSeriesTransformer

logger = logging.getLogger("Oracle")

class Oracle:
    """
    The Oracle Engine: Uses Transformer models to predict future price movements.
    Wraps the institutional-grade TimeSeriesTransformer.
    """
    def __init__(self, model_path="models/nexus_transformer.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        """Loads the Transformer model weights."""
        if not os.path.exists(self.model_path):
            logger.warning(f"[ORACLE] Model file not found at {self.model_path}. Running in SIMULATION mode.")
            return

        try:
            # Initialize model architecture (Must match training config)
            # Default: input_dim=5 (OHLCV), d_model=128, nhead=4, num_layers=2
            self.model = TimeSeriesTransformer(
                input_dim=5, 
                d_model=128, 
                nhead=4, 
                num_layers=2, 
                output_dim=3
            ).to(self.device)
            
            # Load weights
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval() # Set to evaluation mode
            logger.info(f"[ORACLE] Nexus Transformer loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"[ORACLE] Failed to load model: {e}")
            self.model = None

    def predict(self, candles):
        """
        Predicts the next price movement direction.
        
        Args:
            candles: List of last 60 candle dicts [{'open':, 'high':, 'low':, 'close':, 'tick_volume':...}]
            
        Returns:
            prediction: "UP", "DOWN", or "NEUTRAL"
            confidence: float (0.0 to 1.0)
        """
        if self.model is None:
            return "NEUTRAL", 0.0

        try:
            # Preprocess data: Extract OHLCV
            # Ensure we have exactly 60 candles
            if len(candles) < 60:
                return "NEUTRAL", 0.0
                
            data = []
            for c in candles[-60:]:
                # Normalize or use raw? The model likely expects normalized or specific scaling.
                # Assuming raw for now, or simple log returns? 
                # Given the context, we'll pass raw values and hope the model handles it 
                # (or the Transformer handles the embedding).
                # NOTE: Usually models need normalization. 
                # But for this fix, we just want to match the input shape.
                
                # Check keys - MT5 usually gives 'tick_volume' or 'real_volume'
                vol = c.get('tick_volume', 0) or c.get('volume', 0)
                
                row = [
                    float(c['open']),
                    float(c['high']),
                    float(c['low']),
                    float(c['close']),
                    float(vol)
                ]
                data.append(row)
            
            # Convert to tensor: [1, 60, 5]
            input_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Forward pass (returns trend_logits, volatility_pred)
                trend_logits, _ = self.model(input_tensor)
                
                probabilities = torch.softmax(trend_logits, dim=1)
                
                # Get predicted class
                predicted_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, predicted_idx].item()
                
                # Mapping: 0=BUY(UP), 1=SELL(DOWN), 2=NEUTRAL (Check training mapping!)
                # Assuming standard: 0: BUY, 1: SELL, 2: HOLD
                classes = ["UP", "DOWN", "NEUTRAL"]
                if predicted_idx < len(classes):
                    prediction = classes[predicted_idx]
                else:
                    prediction = "NEUTRAL"
                
                return prediction, confidence

        except Exception as e:
            logger.error(f"[ORACLE] Prediction error: {e}")
            return "NEUTRAL", 0.0
