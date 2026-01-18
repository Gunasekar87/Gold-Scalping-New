
import sys
import os
import time
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MT5Check")

try:
    from src.bridge.mt5_adapter import MT5Adapter
except ImportError:
    print("Could not import MT5Adapter. Make sure you are running from the project root.")
    sys.exit(1)

def test_connection():
    print(">>> Testing connection...")
    adapter = MT5Adapter()
    if adapter.connect():
        print(">>> Connection SUCCESS")
        return adapter
    else:
        print(">>> Connection FAILED")
        return None

def test_symbol_resolution(adapter, symbol):
    print(f"\n>>> Testing symbol resolution for '{symbol}'...")
    
    # Access the internal helper directly to test logic
    resolved = adapter._resolve_symbol(symbol)
    
    if resolved:
        print(f">>> Resolution SUCCESS: '{symbol}' -> '{resolved}'")
        
        # Test Tick
        tick = adapter.get_tick(symbol)
        if tick:
            print(f">>> Tick Data: Bid={tick['bid']} Ask={tick['ask']} Time={tick['time']}")
            
            current_time = time.time()
            age = current_time - tick['time']
            print(f">>> Tick Age: {age:.2f} seconds")
            
            if age > 60:
                print(">>> WARNING: Tick data is OLD (Stale Tick Issue Persists?)")
            else:
                print(">>> Tick data is FRESH")
        else:
            print(">>> FAILED to get tick (Returned None)")
    else:
        print(f">>> Resolution FAILED: Could not find '{symbol}' or any variant.")

if __name__ == "__main__":
    adapter = test_connection()
    if adapter:
        # Test XAUUSD (Gold)
        test_symbol_resolution(adapter, "XAUUSD")
        
        # Test common variants manually to allow user to see what works
        test_symbol_resolution(adapter, "EURUSD")
        test_symbol_resolution(adapter, "GBPUSD")
        
        print("\n>>> Done.")
