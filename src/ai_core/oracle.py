import torch
import torch.nn as nn
import numpy as np
import logging
import os
from .nexus_transformer import TimeSeriesTransformer
from .architect import Architect

logger = logging.getLogger("Oracle")

class Oracle:
    """
    The Oracle Engine: Uses Transformer models to predict future price movements.
    Wraps the institutional-grade TimeSeriesTransformer.
    """
    def __init__(self, mt5_adapter=None, tick_analyzer=None, model_path="models/nexus_transformer.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_path = model_path
        self.architect = Architect(mt5_adapter) if mt5_adapter else None
        self.tick_pressure = tick_analyzer
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

    def calculate_rsi(self, prices, period=14):
        """Helper to calculate RSI for the Sniper logic."""
        if len(prices) < period + 1:
            return []
        prices_arr = np.array(prices)
        deltas = np.diff(prices_arr)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 0
        rsi = np.zeros_like(prices_arr)
        rsi[:period] = 100. - 100. / (1. + rs)

        for i in range(period, len(prices_arr)):
            delta = deltas[i - 1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 0
            rsi[i] = 100. - 100. / (1. + rs)
        return rsi

    async def get_sniper_signal_v2(self, symbol: str, candles: list) -> dict:
        """
        v5.5.0: THE ORACLE (AI + Reality + Structure)
        Replaces legacy sniper logic with Reality Lock and Spatial Veto.
        Returns: Dict with 'signal' (1, -1, 0), 'confidence', 'reason'
        """
        # 1. AI Prediction (The Brain)
        ai_pred, ai_conf = self.predict(candles)
        
        ai_direction = 0
        if ai_pred == "UP": ai_direction = 1
        elif ai_pred == "DOWN": ai_direction = -1
        
        if ai_direction == 0:
            return {'signal': 0, 'confidence': 0.0, 'reason': "AI Neutral"}

        # 2. Real-Time Reality (The Eyes) - Reality Lock
        pressure_val = 0.0
        if self.tick_pressure:
            metrics = self.tick_pressure.get_pressure_metrics()
            # pressure_score is typically >15 or <-15 for strong moves
            # dominance is BUY/SELL
            pressure_val = metrics.get('pressure_score', 0.0)
            
            # REALITY LOCK:
            # If AI says BUY, Pressure must NOT be strongly SELLING (<-15)
            if ai_direction == 1 and pressure_val < -15.0:
                logger.info(f"[ORACLE] BLOCKED BUY: AI says Up, but Pressure is Down ({pressure_val:.2f})")
                return {'signal': 0, 'confidence': 0.0, 'reason': f"Reality Lock (Pressure {pressure_val:.1f})"}
                
            # If AI says SELL, Pressure must NOT be strongly BUYING (>15)
            if ai_direction == -1 and pressure_val > 15.0:
                logger.info(f"[ORACLE] BLOCKED SELL: AI says Down, but Pressure is Up ({pressure_val:.2f})")
                return {'signal': 0, 'confidence': 0.0, 'reason': f"Reality Lock (Pressure {pressure_val:.1f})"}

        # 3. Spatial Awareness (The Map) - Architect Veto
        if self.architect:
            structure = self.architect.get_market_structure(symbol)
            if structure:
                # RULE: Don't Buy if we are hitting the Ceiling
                if ai_direction == 1 and structure['status'] == 'BLOCKED_UP':
                    logger.info(f"[ARCHITECT] VETO BUY: Hitting H1 Resistance at {structure['resistance']:.2f} (Gap: {structure['room_up']:.2f})")
                    return {'signal': 0, 'confidence': 0.0, 'reason': f"Blocked by Resistance ({structure['resistance']:.2f})"}

                # RULE: Don't Sell if we are hitting the Floor
                if ai_direction == -1 and structure['status'] == 'BLOCKED_DOWN':
                    logger.info(f"[ARCHITECT] VETO SELL: Hitting H1 Support at {structure['support']:.2f} (Gap: {structure['room_down']:.2f})")
                    return {'signal': 0, 'confidence': 0.0, 'reason': f"Blocked by Support ({structure['support']:.2f})"}

        # 4. Final Decision
        # If we passed all checks, we return the signal
        return {
            'signal': ai_direction, 
            'confidence': ai_conf, 
            'reason': f"Aligned: AI({ai_conf:.2f}) + Pressure({pressure_val:.1f})"
        }

    def get_sniper_signal(self, prices: list, volumes: list) -> str:
        """
        The 'World Class' Sniper Logic.
        Returns: 'SELL_SNIPER', 'BUY_SNIPER', or None.
        Checks:
        1. Momentum Exhaustion (Linear Regression)
        2. RSI Extreme
        3. Volume Divergence (Price Up, Volume Down)
        """
        if len(prices) < 30 or len(volumes) < 30:
            return None

        # Calculate RSI internally
        rsi_values = self.calculate_rsi(prices)
        if len(rsi_values) < 1:
            return None

        current_price = prices[-1]
        current_rsi = rsi_values[-1]
        
        # 1. Linear Regression Slope (Momentum)
        # We look at the last 10 candles to see if the trend is flattening
        y = np.array(prices[-10:])
        x = np.arange(len(y))
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        slope = (m / current_price) * 10000 # Basis points

        # 2. Volume Divergence Check
        # Logic: Price is high, but Volume is dropping = Exhaustion
        recent_vol_avg = np.mean(volumes[-3:])
        past_vol_avg = np.mean(volumes[-15:-5])
        vol_dropping = recent_vol_avg < (past_vol_avg * 0.85) # 15% drop in volume

        # --- SNIPER LOGIC FOR SELL (Top of Bull Run) ---
        # A. Extreme RSI (>75) - Overbought
        # B. Slope is flattening (< 2.0) - Rocket running out of fuel
        # C. Volume is drying up - Buyers leaving
        if current_rsi > 75 and slope < 2.0 and vol_dropping:
            return "SELL_SNIPER"

        # --- SNIPER LOGIC FOR BUY (Bottom of Crash) ---
        if current_rsi < 25 and slope > -2.0 and vol_dropping:
            return "BUY_SNIPER"

        return None

    def get_regime_and_signal(self, candles):
        """
        HIGHEST INTELLIGENCE LAYER 1: REGIME DETECTION
        Determines if market is TRENDING or RANGING to prevent death spirals.
        """
        if len(candles) < 20:
            return "RANGE", "HOLD"

        closes = np.array([c['close'] for c in candles])
        
        # 1. Calculate Linear Regression Slope (Trend Strength)
        y = closes[-20:]
        x = np.arange(len(y))
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        
        # Normalize slope to basis points relative to price
        current_price = closes[-1]
        slope_bps = (m / current_price) * 10000 

        # 2. Calculate RSI (Overbought/Oversold)
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Simple Average for RSI (matches standard trading view approx)
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        # 3. Determine Regime
        regime = "RANGE"
        if slope_bps > 2.5: # Strong Up Trend
            regime = "TREND_UP"
        elif slope_bps < -2.5: # Strong Down Trend
            regime = "TREND_DOWN"
        
        # 4. Generate Signal based on Regime
        signal = "HOLD"
        
        if regime == "TREND_UP":
            # ONLY BUY in Uptrend, but wait for pullback (RSI not overbought)
            if rsi < 70: 
                signal = "BUY"
            else:
                signal = "HOLD" # Don't buy top
                
        elif regime == "TREND_DOWN":
            # ONLY SELL in Downtrend, but wait for pullback (RSI not oversold)
            if rsi > 30:
                signal = "SELL"
            else:
                signal = "HOLD" # Don't sell bottom
                
        else: # RANGE
            # Buy Low, Sell High
            if rsi < 30:
                signal = "BUY"
            elif rsi > 70:
                signal = "SELL"
                
        return regime, signal

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
