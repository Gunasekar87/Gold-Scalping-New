import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

try:
    from src.trading_engine import TradingEngine
    print("Successfully imported TradingEngine")
except ImportError as e:
    print(f"ImportError: {e}")
except SyntaxError as e:
    print(f"SyntaxError: {e}")
except Exception as e:
    print(f"Error: {e}")
