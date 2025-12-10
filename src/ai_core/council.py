import logging
import random

logger = logging.getLogger("COUNCIL")
# [CRITICAL] Get the specific UI logger that run_bot.py listens to
ui_logger = logging.getLogger("AETHER_UI")

class Council:
    """
    The 'Agentic' Orchestrator.
    Instead of a single script making decisions, the Council gathers votes 
    from specialized AI Agents to reach a consensus.
    """
    def __init__(self, nexus_brain, macro_sentinel, iron_shield, quantum_arbiter=None, sentiment_engine=None):
        self.nexus = nexus_brain
        self.sentinel = macro_sentinel
        self.shield = iron_shield
        self.quantum = quantum_arbiter
        self.sentiment = sentiment_engine
        
        # Metacognition (Self-Reflection)
        self.recent_outcomes = [] # Stores 1 (Win) or 0 (Loss)
        self.caution_factor = 1.0 # 1.0 = Neutral, >1.0 = Cautious, <1.0 = Aggressive

    def reflect(self, was_profitable):
        """
        The Council reviews the outcome of its decisions to adjust future confidence.
        """
        self.recent_outcomes.append(1 if was_profitable else 0)
        if len(self.recent_outcomes) > 20:
            self.recent_outcomes.pop(0)
            
        # Calculate Win Rate
        if len(self.recent_outcomes) > 0:
            win_rate = sum(self.recent_outcomes) / len(self.recent_outcomes)
            
            # Adaptive Logic:
            # If we are losing (WR < 50%), we become more cautious (Factor > 1.0)
            # If we are winning (WR > 50%), we stay neutral or slightly aggressive.
            # Formula: 1.5 - WR. (e.g., 0.4 WR -> 1.1 Factor. 0.8 WR -> 0.7 Factor)
            self.caution_factor = max(0.8, 1.5 - win_rate)
            
            logger.info(f"Metacognition: WinRate {win_rate:.2f} -> Caution Factor {self.caution_factor:.2f}")

    def detect_traps(self, signal: str, candles: list) -> tuple[bool, str]:
        """
        Analyzes the last few candles for 'Fakeout' patterns.
        Returns: (IsTrap, Reason)
        """
        # Fix: Check for None explicitly to avoid numpy ambiguity error
        if candles is None or len(candles) < 3:
            return False, ""
            
        last_candle = candles[-1]
        
        # 1. Wick Rejection (Pin Bar)
        # If BUY signal, but huge upper wick -> Selling Pressure (Trap)
        # If SELL signal, but huge lower wick -> Buying Pressure (Trap)
        
        # Ensure we are accessing dictionary keys or object attributes correctly
        # Assuming candles are dicts from MT5 or similar structure
        try:
            open_p = float(last_candle['open'])
            close_p = float(last_candle['close'])
            high_p = float(last_candle['high'])
            low_p = float(last_candle['low'])
        except (KeyError, TypeError):
            # Fallback if candles are not dicts (e.g. objects or tuples)
            return False, "Invalid Candle Data"

        body_size = abs(close_p - open_p)
        upper_wick = high_p - max(open_p, close_p)
        lower_wick = min(open_p, close_p) - low_p
        
        # Avoid division by zero
        if body_size == 0: body_size = 0.00001
            
        if signal == "BUY":
            # Trap: Price tried to go up (High) but was pushed all the way down (Close near Low)
            # Ratio: Upper Wick is 2x larger than Body
            if upper_wick > (body_size * 2.0):
                return True, f"Trap Detected: Strong Selling Wick (Pin Bar) rejecting highs."
                
        elif signal == "SELL":
            # Trap: Price tried to go down (Low) but was pushed all the way up (Close near High)
            # Ratio: Lower Wick is 2x larger than Body
            if lower_wick > (body_size * 2.0):
                return True, f"Trap Detected: Strong Buying Wick (Pin Bar) rejecting lows."
                
        return False, ""

    def deliberate(self, symbol, tick_history, nexus_candles, full_candles, current_equity, current_balance, macro_trend_slope=0.0, quantum_signal="NEUTRAL", sentiment_score=0.0):
        """
        Conducts a debate among agents.
        Returns: (Decision, Reason, Metadata)
        Decision: "BUY", "SELL", "HOLD"
        """
        
        # Input validation
        try:
            if not tick_history or len(tick_history) < 10:
                return "HOLD", "Insufficient historical data", {}
            if not nexus_candles or len(nexus_candles) < 64:
                return "HOLD", "Insufficient candle data for Nexus", {}
            if not isinstance(current_equity, (int, float)) or current_equity <= 0:
                return "HOLD", f"Invalid equity value: {current_equity}", {}
        except Exception as validation_error:
            logger.error(f"Input validation failed: {validation_error}")
            return "HOLD", "Input validation error", {}
        
        # 1. Macro Sentinel (The Guardian) speaks first
        # If the market is too dangerous, we don't even look at charts.
        try:
            market_status = self.sentinel.analyze_market_condition(tick_history)
            if market_status == "DANGEROUS":
                return "HOLD", "MacroSentinel: Market Turbulence Detected (News/Crash Risk)", {}
        except Exception as sentinel_error:
            logger.error(f"MacroSentinel failed: {sentinel_error}")
            market_status = "NORMAL"  # Fail-safe: continue with caution
            
        # 2. Iron Shield (The Risk Manager) speaks second
        # If we are over-leveraged, we cannot open new trades.
        try:
            if not self.shield.can_trade(current_equity, current_balance):
                return "HOLD", "IronShield: Risk Exposure Limit Reached", {}
        except Exception as shield_error:
            logger.error(f"IronShield check failed: {shield_error}")
            return "HOLD", "IronShield Error: Risk check unavailable", {}
            
        # 3. Nexus Brain (The Institutional Mind) speaks last
        # Uses Transformer Neural Network to predict Trend and Volatility
        try:
            signal, confidence, predicted_vol = self.nexus.predict(nexus_candles)
            
            # Convert numpy types to native Python types
            confidence = float(confidence)
            predicted_vol = float(predicted_vol)
            
            # Validate Nexus output
            if signal not in ["BUY", "SELL", "HOLD"]:
                logger.error(f"Invalid signal from NexusBrain: {signal}")
                return "HOLD", "NexusBrain: Invalid signal output", {}
            if confidence < 0 or confidence > 1:
                logger.error(f"Invalid confidence from NexusBrain: {confidence} (out of range [0,1])")
                confidence = 0.0
        except Exception as nexus_error:
            logger.error(f"NexusBrain prediction failed: {nexus_error}")
            return "HOLD", f"NexusBrain Error: {str(nexus_error)[:100]}", {}
        
        # Prepare Metadata for Dashboard
        try:
            metadata = {
                "nexus_signal": signal,
                "nexus_confidence": float(confidence),
                "nexus_volatility": float(predicted_vol),
                "market_status": market_status,
                "macro_slope": macro_trend_slope,
                "quantum_signal": quantum_signal,
                "sentiment_score": sentiment_score,
                "caution_factor": self.caution_factor
            }
        except Exception as metadata_error:
            logger.error(f"Failed to create metadata: {metadata_error}")
            metadata = {}
        
        # --- SENTIMENT FILTER ---
        # If News is extremely negative, do not BUY.
        # If News is extremely positive, do not SELL.
        if sentiment_score < -0.5 and signal == "BUY":
             return "HOLD", f"Council: Vetoed by Sentiment Engine (Bearish News: {sentiment_score:.2f})", metadata
        if sentiment_score > 0.5 and signal == "SELL":
             return "HOLD", f"Council: Vetoed by Sentiment Engine (Bullish News: {sentiment_score:.2f})", metadata

        # --- QUANTUM ARBITRATION ---
        # If Quantum Arbiter has a strong opinion (Statistical Anomaly), it can override or boost.
        if quantum_signal != "NEUTRAL":
            # Quantum Signal Format: "BUY_A_SELL_B" (Buy Gold) or "SELL_A_BUY_B" (Sell Gold)
            # Asset A is the primary symbol (XAUUSD).
            
            quantum_vote = "NEUTRAL"
            if "BUY_A" in quantum_signal:
                quantum_vote = "BUY"
            elif "SELL_A" in quantum_signal:
                quantum_vote = "SELL"
                
            # Conflict Resolution
            if quantum_vote == "BUY" and signal == "BUY":
                confidence += 0.2 # Boost
                metadata['quantum_boost'] = True
            elif quantum_vote == "SELL" and signal == "SELL":
                confidence += 0.2 # Boost
                metadata['quantum_boost'] = True
            elif quantum_vote == "BUY" and signal == "SELL":
                return "HOLD", "Council: Quantum Arbiter (Mean Reversion) conflicts with Nexus (Trend)", metadata
            elif quantum_vote == "SELL" and signal == "BUY":
                return "HOLD", "Council: Quantum Arbiter (Mean Reversion) conflicts with Nexus (Trend)", metadata

        # Filter by Confidence
        if confidence < 0.3: # Minimum confidence threshold (Lowered for more activity)
            return "HOLD", f"NexusBrain: Low Confidence ({confidence:.2f})", metadata

        # 4. Trap Detection (The Skeptic)
        # We check if the market is faking us out.
        is_trap, trap_reason = self.detect_traps(signal, full_candles)
        if is_trap:
            return "HOLD", f"Council: Vetoed by Trap Detector. {trap_reason}", metadata
        
        # 5. The Consensus (with Holographic Confluence)
        # We check if the Micro Trend (Nexus) aligns with the Macro Trend (H1 Slope)
        
        # Apply Metacognition: Adjust threshold based on recent performance
        # Nexus confidence is already 0-1.
        # We can use caution_factor to raise the bar.
        # UPDATE: Lowered base from 0.6 to 0.3 to match NexusBrain and allow more trades.
        required_confidence = 0.3 * self.caution_factor
        
        # Log every deliberation for debugging
        # if random.random() < 0.1: # Log 10% of deliberations to avoid spam
        logger.info(f"[COUNCIL] DELIBERATION: {signal} ({confidence:.2f}) | Trap: {is_trap} | Sentiment: {sentiment_score:.2f}")
        logger.info(f"   [NEXUS]: {signal} (conf: {confidence:.2f}, vol: {predicted_vol:.4f})")
        logger.info(f"   [RISK]: Can trade: {self.shield.can_trade(current_equity, current_balance)}")
        logger.info(f"   [MARKET]: Status {market_status}, Macro slope: {macro_trend_slope:.5f}")
        logger.info(f"   [DECISION]: {'HOLD' if confidence < required_confidence else signal.upper()} (required: {required_confidence:.2f})")
        
        confluence_msg = "Neutral"
        
        if macro_trend_slope != 0:
            # If Signal is BUY and Macro is UP (>0)
            if signal == "BUY" and macro_trend_slope > 0:
                confluence_msg = "Aligned with H1 Trend"
            # If Signal is SELL and Macro is DOWN (<0)
            elif signal == "SELL" and macro_trend_slope < 0:
                confluence_msg = "Aligned with H1 Trend"
            # If Opposed
            elif (signal == "BUY" and macro_trend_slope < -0.0005) or (signal == "SELL" and macro_trend_slope > 0.0005):
                # Strong opposition from Macro Trend
                return "HOLD", f"Council: Vetoed. Signal {signal} opposes Macro Trend (Slope: {macro_trend_slope:.5f})", metadata

        if confidence < required_confidence:
             return "HOLD", f"Council: Confidence {confidence:.2f} below Caution Threshold {required_confidence:.2f}", metadata

        if signal == "BUY":
            decision = "BUY"
            reason = f"NexusBrain: Bullish Signal (Conf: {confidence:.2f}, Caution: {self.caution_factor:.2f}, Macro: {confluence_msg})"
        elif signal == "SELL":
            decision = "SELL"
            reason = f"NexusBrain: Bearish Signal (Conf: {confidence:.2f}, Caution: {self.caution_factor:.2f}, Macro: {confluence_msg})"
        else:
            decision = "HOLD"
            reason = "Neutral Signal"

        # [ADD THIS] --- VISUAL AI COUNCIL SUMMARY ---
        if decision != "HOLD":
            # Define emoji version
            emoji_commentary = (
                f"\n🧠 ---------------- AI COUNCIL ---------------- 🧠\n"
                f"🗳️ Vote:     {signal} (Confidence: {confidence:.2f})\n"
                f"🗣️ Reason:   {reason}\n"
                f"📊 Macro:    {confluence_msg} (Slope: {macro_trend_slope:.5f})\n"
                f"🛡️ Trap:     {'Detected' if is_trap else 'Clear'}\n"
                f"-----------------------------------------------"
            )
            
            # Define clean version
            clean_commentary = (
                f"\n--- AI COUNCIL ---\n"
                f"Vote:     {signal} (Confidence: {confidence:.2f})\n"
                f"Reason:   {reason}\n"
                f"Macro:    {confluence_msg} (Slope: {macro_trend_slope:.5f})\n"
                f"Trap:     {'Detected' if is_trap else 'Clear'}\n"
                f"-----------------------------------------------"
            )

            # [FIX] Windows Console Compatibility - Force clean version on Windows to prevent crashes
            # DISABLED BY USER REQUEST (Rolling notes)
            pass
            # import sys
            # if sys.platform == 'win32':
            #     ui_logger.info(clean_commentary)
            # else:
            #     try:
            #         ui_logger.info(emoji_commentary)
            #     except Exception:
            #         ui_logger.info(clean_commentary)
            
        return decision, reason, metadata
            
        return "HOLD", "Uncertain", metadata
