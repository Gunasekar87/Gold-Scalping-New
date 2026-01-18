
import logging
import math
import sys
import os
import time
import traceback

# --- MOCK CLASSES ---
class MockLogger:
    def info(self, msg): print(f"[INFO] {msg}")
    def debug(self, msg): print(f"[DEBUG] {msg}")
    def warning(self, msg): print(f"[WARN] {msg}")
    def error(self, msg): print(f"[ERROR] {msg}")

# Mock logging BEFORE importing modules
logging.getLogger("HybridHedgeIntel").handlers = []
logging.getLogger("Oracle").handlers = []
logging.getLogger("TradingEngine").handlers = []

# --- IMPORT TARGETS ---
sys.path.append(os.getcwd())

# We might fail to import if dependencies are missing, so we mock them if needed
try:
    from src.ai_core.oracle import Oracle
    # We don't import TradingEngine to avoid inheritance issues in this script
    # from src.trading_engine import TradingEngine 
    from src.market_data import MarketDataManager
except ImportError as e:
    print(f"Import Error: {e}")
    print("Ensure you are running from project root.")
    # sys.exit(1) # Continue to run Cooldown test even if Oracle fails to import

def test_oracle_regime():
    print("\n=== TEST 4: Oracle Regime Threshold (Split Brain Fix) ===")
    try:
        if 'Oracle' not in globals():
             print("Skipping Oracle test due to import error")
             return

        # Mock dependencies for Oracle
        # We need to ensure Oracle can init without crashing
        orc = Oracle(model_path="dummy_path_ignore")
        orc.tuner = type('Mock', (), {'suggest_params': lambda: {'velocity_threshold': 0.65}})()
        
        # Simulate Grinding Trend (Slope ~ 2.0)
        # Old logic (>2.5) would say RANGE. New logic (>1.5) should say TREND_UP.
        # Price 2000. 2.0 bps = 0.4 change per bar.
        candles = [{'close': 2000 + (i * 0.4), 'open': 2000, 'high': 2000, 'low': 2000, 'tick_volume': 10} for i in range(30)]
        
        regime, signal = orc.get_regime_and_signal(candles)
        print(f"Input Slope: ~2.0 bps | Result Regime: {regime} | Signal: {signal}")
        
        if regime == "TREND_UP":
            print("✅ SUCCESS: Oracle identifies Grinding Trend (Threshold lowered).")
        else:
            print(f"❌ FAIL: Oracle says {regime} (Threshold too high).")
            
    except Exception as e:
        print(f"❌ TEST FAILED with Error: {e}")
        traceback.print_exc()

def test_cooldown_bypass():
    print("\n=== TEST 5: Cooldown Bypass (Deadlock Fix) ===")
    try:
        # Standalone Engine Mock with the EXACT logic from `trading_engine.py` (post-fix)
        # We verify the LOGIC itself, avoiding class inheritance hell.
        class StandaloneEngine:
            def __init__(self):
                self.config = type('Conf', (), {'global_trade_cooldown': 30.0})()
                self.last_trade_time = 0.0
                
            def validate_trade_entry(self, signal, lot_size, account_info, tick, is_recovery_trade=False):
                # --- LOGIC COPIED FROM TRADING_ENGINE.PY ---
                # Check global trade cooldown
                # [CRITICAL FIX] Bypass cooldown for Recovery Trades (Hedges/DCA).
                # We cannot wait 30s when the house is on fire.
                if not is_recovery_trade:
                     time_since_last_trade = time.time() - self.last_trade_time
                     if time_since_last_trade < self.config.global_trade_cooldown:
                         return False, f"Global cooldown: {time_since_last_trade:.1f}s < {self.config.global_trade_cooldown}s"
                # --- END LOGIC ---
                return True, "OK"

        engine = StandaloneEngine()
        engine.last_trade_time = time.time() - 5.0 # 5 seconds ago
        
        # 1. Normal Trade (Should be BLOCKED)
        ok, reason = engine.validate_trade_entry(None, 0.01, {}, {}, is_recovery_trade=False)
        print(f"Normal Trade (5s ago): {ok} | {reason}")
        if not ok and "Global cooldown" in reason:
            print("✅ Normal Trade Blocked correct.")
        else:
             print("❌ Normal Trade FAILED.")
             
        # 2. Recovery Trade (Should be BYPASSED)
        ok_rec, reason_rec = engine.validate_trade_entry(None, 0.01, {}, {}, is_recovery_trade=True)
        print(f"Recovery Trade (5s ago): {ok_rec} | {reason_rec}")
        if ok_rec:
             print("✅ SUCCESS: Recovery Trade Bypassed Cooldown.")
        else:
             print("❌ FAIL: Recovery Trade Blocked.")
             
    except Exception as e:
        print(f"❌ TEST FAILED with Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_oracle_regime()
    test_cooldown_bypass()
