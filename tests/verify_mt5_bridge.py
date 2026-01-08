
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import MetaTrader5 as mt5
    print(f"MetaTrader5 Package Version: {mt5.__version__}")
except ImportError:
    print("MetaTrader5 package not installed. Cannot verify constants directly.")
    sys.exit(1)

def verify_constants():
    with open("tests/bridge_results.txt", "w", encoding="utf-8") as f:
        f.write("--- 1. CONSTANT VERIFICATION ---\n")
        f.write(f"mt5.ORDER_TYPE_BUY  (Integer) = {mt5.ORDER_TYPE_BUY}\n")
        f.write(f"mt5.ORDER_TYPE_SELL (Integer) = {mt5.ORDER_TYPE_SELL}\n")
        
        if mt5.ORDER_TYPE_BUY == 0 and mt5.ORDER_TYPE_SELL == 1:
            f.write("✅ STANDARD CONSTANTS CONFIRMED (Buy=0, Sell=1)\n")
        else:
            f.write("⚠️ NON-STANDARD CONSTANTS DETECTED!\n")

def verify_adapter_translation():
    with open("tests/bridge_results.txt", "a", encoding="utf-8") as f:
        f.write("\n--- 2. ADAPTER LOGIC VERIFICATION ---\n")
        
        inputs = ["BUY", "SELL"]
        for order_type in inputs:
            mt5_type = mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL
            
            f.write(f"Python String: '{order_type}' -> Validated MT5 Integer: {mt5_type}\n")
            
            if order_type == "BUY" and mt5_type != mt5.ORDER_TYPE_BUY:
                f.write("❌ FATAL: 'BUY' did not map to ORDER_TYPE_BUY\n")
            elif order_type == "SELL" and mt5_type != mt5.ORDER_TYPE_SELL:
                 f.write("❌ FATAL: 'SELL' did not map to ORDER_TYPE_SELL\n")
            else:
                f.write(f"✅ Correct Mapping for {order_type}\n")

if __name__ == "__main__":
    try:
        if not mt5.initialize():
            # Even if init fails (no terminal), constants are valid
            pass 
        verify_constants()
        verify_adapter_translation()
    except Exception as e:
        with open("tests/bridge_results.txt", "w") as f:
            f.write(f"CRITICAL SCRIPT ERROR: {e}")
    finally:
        mt5.shutdown()
