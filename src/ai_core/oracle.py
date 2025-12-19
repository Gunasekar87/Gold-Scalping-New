import torch
import torch.nn as nn
import numpy as np
import logging
import os
from typing import Optional
from .nexus_transformer import TimeSeriesTransformer
from .architect import Architect
from .graph_neural_net import GNNPredictor
from .bayesian_tuner import BayesianOptimizer
from .contrastive_fusion import ContrastiveFusion

try:
    import MetaTrader5 as mt5
except Exception:  # pragma: no cover
    mt5 = None

logger = logging.getLogger("Oracle")

class Oracle:
    """
    The Oracle Engine: Uses Transformer models to predict future price movements.
    Wraps the institutional-grade TimeSeriesTransformer.
    """
    def __init__(self, mt5_adapter=None, tick_analyzer=None, global_brain=None, model_path="models/nexus_transformer.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_path = model_path
        self.mt5_adapter = mt5_adapter
        self.architect = Architect(mt5_adapter) if mt5_adapter else None
        self.tick_pressure = tick_analyzer
        self.global_brain = global_brain
        self._market_book_symbols = set()
        
        # --- ADVANCED AI MODULES ---
        self.gnn = GNNPredictor()
        self.tuner = BayesianOptimizer()
        self.fusion = ContrastiveFusion()
        
        self.load_model()

    def _get_order_book_imbalance(self, symbol: str) -> Optional[float]:
        """Return normalized order-book imbalance in [-1, 1] when MT5 market book is available."""
        if not symbol:
            return None

        try:
            # Prefer the broker adapter if available (keeps behavior consistent with MarketDataManager).
            if self.mt5_adapter is not None and hasattr(self.mt5_adapter, 'get_order_book'):
                book = self.mt5_adapter.get_order_book(symbol)
                if book:
                    bids = book.get('bids', []) or []
                    asks = book.get('asks', []) or []
                    bid_vol = float(sum(item.get('volume', 0.0) for item in bids))
                    ask_vol = float(sum(item.get('volume', 0.0) for item in asks))
                    denom = bid_vol + ask_vol
                    if denom > 0:
                        imbalance = (bid_vol - ask_vol) / denom
                        return max(-1.0, min(1.0, float(imbalance)))

            # Fallback: direct MT5 Market Book
            if mt5 is None:
                return None

            # Ensure book subscription (MT5 requires market_book_add).
            if symbol not in self._market_book_symbols:
                try:
                    if mt5.market_book_add(symbol):
                        self._market_book_symbols.add(symbol)
                except Exception:
                    # If subscription fails, we can still try to read; MT5 may return None.
                    pass

            book = mt5.market_book_get(symbol)
            if not book:
                return None

            bid_vol = 0.0
            ask_vol = 0.0
            for row in book:
                # row.type: 0=BUY, 1=SELL (in most MT5 builds)
                rtype = getattr(row, 'type', None)
                rvol = float(getattr(row, 'volume', 0.0) or 0.0)
                if rvol <= 0:
                    continue
                if rtype == 0:
                    bid_vol += rvol
                elif rtype == 1:
                    ask_vol += rvol

            denom = bid_vol + ask_vol
            if denom <= 0:
                return None

            imbalance = (bid_vol - ask_vol) / denom
            # Hard clamp for safety
            return max(-1.0, min(1.0, float(imbalance)))
        except Exception:
            return None

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
        v6.0.0: THE ORACLE (AI + Macro + Fusion + Tuning)
        Replaces legacy logic with Advanced AI Stack.
        Returns: Dict with 'signal' (1, -1, 0), 'confidence', 'reason'
        """
        # 0. Bayesian Tuning: Get dynamic thresholds
        params = self.tuner.suggest_params()
        # Persist last params used so end-of-session feedback can tune against real outcomes.
        try:
            self.last_tuner_params = dict(params) if isinstance(params, dict) else None
        except Exception:
            self.last_tuner_params = None
        vel_threshold = params.get("velocity_threshold", 0.65)

        # 1. Macro Intelligence (GNN)
        # In a real scenario, we would fetch DXY/Oil prices here. 
        # For now, we simulate or use available data if possible.
        # We pass the current Gold price as a proxy for the node state.
        current_price = candles[-1]['close'] if candles else 0
        macro_signal = 0.0
        try:
            if self.global_brain is not None and hasattr(self.global_brain, 'get_bias'):
                macro_signal = float(self.global_brain.get_bias())
            else:
                # Fallback (legacy/demo): simulated macro
                macro_pred = self.gnn.predict(current_price)  # 0..1
                macro_signal = (macro_pred - 0.5) * 2
        except Exception:
            macro_signal = 0.0

        # 2. Micro Intelligence (Transformer)
        ai_pred, ai_conf = self.predict(candles)
        
        ai_score = 0.0
        if ai_pred == "UP": ai_score = ai_conf
        elif ai_pred == "DOWN": ai_score = -ai_conf
        
        # 3. Reality Lock (Tick Pressure)
        pressure_val = 0.0
        pressure_score = 0.0 # Directional micro-signal normalized to [-1, 1]
        if self.tick_pressure:
            metrics = self.tick_pressure.get_pressure_metrics()
            pressure_val = metrics.get('pressure_score', 0.0)

            # IMPORTANT:
            # The fusion key is named `tick_velocity`, but `pressure_score` is directional
            # (BUY/SELL) and can get large quickly. To avoid pinning at +/-1.0, we use:
            # - magnitude: normalized tick speed (ticks/sec)
            # - sign: direction from pressure (BUY vs SELL)
            velocity = float(metrics.get('velocity', 0.0) or 0.0)
            vel_norm = float(os.getenv("AETHER_VELOCITY_NORM", "15"))
            if vel_norm <= 0:
                vel_norm = 15.0

            direction = 0.0
            if pressure_val > 0:
                direction = 1.0
            elif pressure_val < 0:
                direction = -1.0

            pressure_score = float(direction * np.tanh(velocity / vel_norm))

        # 4. Contrastive Fusion (The Judge)
        # Order book modality: use real DOM when available; else fall back to macro proxy.
        order_book_source = "order_book"
        order_book_signal = self._get_order_book_imbalance(symbol)
        if order_book_signal is None:
            order_book_signal = macro_signal
            order_book_source = "macro_fallback"

        signals = {
            "tick_velocity": pressure_score,
            "candle_pattern": ai_score,
            "order_book": float(order_book_signal),
            # Extra fields (not used by coherence math) to make logs unambiguous.
            "order_book_source": order_book_source,
        }
        
        coherence = self.fusion.compute_coherence(signals)
        final_confidence = self.fusion.validate_signal(ai_score, coherence)

        # Log the AI Council deliberation
        if str(os.getenv("AETHER_ORACLE_FUSION_DEBUG", "0")).strip().lower() in ("1", "true", "yes", "on"):
            logger.info(
                f"[ORACLE] Council: Macro({macro_signal:.2f}) | Micro({ai_score:.2f}) | Pressure({pressure_score:.2f}, raw={pressure_val:.2f}) | OBI({float(order_book_signal):.2f}) -> Coherence: {coherence:.2f}"
            )
        else:
            logger.info(
                f"[ORACLE] Council: Macro({macro_signal:.2f}) | Micro({ai_score:.2f}) | Pressure({pressure_score:.2f}) -> Coherence: {coherence:.2f}"
            )

        # 5. Decision Logic
        signal = 0
        reason = "Neutral"
        
        if final_confidence > vel_threshold: # Dynamic Threshold from Bayesian Tuner
            if ai_score > 0:
                signal = 1
                reason = f"BUY (Conf: {final_confidence:.2f} | Coh: {coherence:.2f})"
            elif ai_score < 0:
                signal = -1
                reason = f"SELL (Conf: {final_confidence:.2f} | Coh: {coherence:.2f})"
        else:
            reason = f"Weak Signal (Conf: {final_confidence:.2f} < {vel_threshold:.2f})"

        # 6. Spatial Veto (Architect) - The final safety check
        if signal != 0 and self.architect:
            structure = self.architect.get_market_structure(symbol)
            if structure:
                if signal == 1 and structure['status'] == 'BLOCKED_UP':
                    logger.info(f"[ARCHITECT] VETO BUY: Resistance at {structure['resistance']:.2f}")
                    return {'signal': 0, 'confidence': 0.0, 'reason': "Blocked by Resistance"}
                if signal == -1 and structure['status'] == 'BLOCKED_DOWN':
                    logger.info(f"[ARCHITECT] VETO SELL: Support at {structure['support']:.2f}")
                    return {'signal': 0, 'confidence': 0.0, 'reason': "Blocked by Support"}

        return {
            'signal': signal, 
            'confidence': final_confidence, 
            'reason': reason
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
                
            use_raw = str(os.getenv("AETHER_ORACLE_USE_RAW_OHLCV", "0")).strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )

            data = []
            recent = candles[-60:]
            prev_close = float(recent[0].get('close', 0.0) or 0.0)
            for c in recent:
                # Check keys - MT5 usually gives 'tick_volume' or 'real_volume'
                vol = float(c.get('tick_volume', 0) or c.get('volume', 0) or 0.0)

                o = float(c.get('open', 0.0) or 0.0)
                h = float(c.get('high', 0.0) or 0.0)
                l = float(c.get('low', 0.0) or 0.0)
                cl = float(c.get('close', 0.0) or 0.0)

                if use_raw or prev_close <= 0 or cl <= 0:
                    row = [o, h, l, cl, vol]
                else:
                    # Normalize to returns relative to previous close to avoid saturation
                    row = [
                        (o / prev_close) - 1.0,
                        (h / prev_close) - 1.0,
                        (l / prev_close) - 1.0,
                        (cl / prev_close) - 1.0,
                        float(np.log1p(max(0.0, vol))),
                    ]

                data.append(row)
                prev_close = cl if cl > 0 else prev_close
            
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
