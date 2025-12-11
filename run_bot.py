import asyncio
import sys
import os
import io
import logging
import signal
import gc  # [CRITICAL] Added for memory tuning
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# [CRITICAL] Pre-load torch to prevent DLL loading hangs on Windows
print(">>> [SYSTEM] Starting Aether Bot v5.0...", flush=True)
print(">>> [SYSTEM] Pre-loading AI Libraries (Torch)...", flush=True)
try:
    import torch
    print(">>> [SYSTEM] AI Libraries Loaded.", flush=True)
except ImportError:
    print(">>> [WARN] Torch not found.", flush=True)
    pass

# Force UTF-8 encoding for stdout and stderr to handle emojis on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# [CRITICAL] Try to import psutil for high priority setting
try:
    import psutil
except ImportError:
    psutil = None
    print("[WARN] SYSTEM: 'psutil' not found. Install with 'pip install psutil' for maximum speed.")

# Load environment variables
load_dotenv("config/secrets.env")

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- PERFORMANCE OPTIMIZATION: HIGH PRIORITY ---
def set_high_priority():
    """
    Sets the process priority to HIGH to minimize OS scheduling latency.
    This ensures the bot reacts faster than other Windows background tasks.
    """
    if psutil is None:
        return
        
    try:
        p = psutil.Process(os.getpid())
        if sys.platform == 'win32':
            p.nice(psutil.HIGH_PRIORITY_CLASS)
        else:
            p.nice(-10) # High priority on Linux
        print("[INFO] SYSTEM: Process Priority set to HIGH for maximum speed.")
    except Exception as e:
        print(f"[WARN] SYSTEM: Could not set high priority: {e}")

# --- PERFORMANCE OPTIMIZATION: GC TUNING ---
def tune_garbage_collector():
    """
    Delays automatic garbage collection to prevent 'Stop-the-World' pauses
    during critical trade execution.
    """
    # Increase thresholds so GC runs less often
    # (700, 10, 10) is default. We increase to delay pauses.
    gc.set_threshold(50000, 50, 50) 
    print("[INFO] SYSTEM: Garbage Collector tuned for low-latency.")

# Force UTF-8 for stdout/stderr to handle emojis on Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Configure logging handlers
# Console handler - ONLY for specific UI messages
# Use a robust stream creation for Windows to ensure UTF-8
if sys.platform == 'win32':
    import codecs
    # Use codecs.getwriter to wrap stdout with error handling
    # This is more robust than TextIOWrapper for some console environments
    try:
        # Force UTF-8 encoding for the stream used by logging
        stream = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
    except Exception:
        stream = sys.stdout
else:
    stream = sys.stdout

# Define a filter to clean up UI noise
class UIFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        # Filter out volatility spam from console (keep in file)
        if "High Volatility Detected" in msg:
            return False
        # Filter out noisy position management logs
        if "[POS_MGMT]" in msg:
            return False
        if "[TP CHECK]" in msg:
            return False
        if "[TP_CHECK_SKIP]" in msg:
            return False
        if "[BUCKET] Checking exit" in msg:
            return False
        if "[BUCKET] REUSING EXISTING" in msg:
            return False
        return True

console_handler = logging.StreamHandler(stream)
console_handler.setLevel(logging.INFO)
console_handler.addFilter(UIFilter()) # Apply the filter
console_formatter = logging.Formatter('%(message)s') # Clean format for UI
console_handler.setFormatter(console_formatter)

# Ensure UTF-8 encoding for console output
if hasattr(console_handler.stream, 'reconfigure'):
    try:
        console_handler.stream.reconfigure(encoding='utf-8')
    except Exception:
        pass  # Fallback if reconfigure not available

# File handler - captures EVERYTHING for background tracking
file_handler = logging.FileHandler('bot_execution.log', mode='w')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Configure root logger - File ONLY (Background)
logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler] # Remove console_handler from root to silence terminal
)

# [FIX] Ensure warnings/errors from ANY module are printed to console
console_handler.setLevel(logging.WARNING)
logging.getLogger().addHandler(console_handler)

# Configure UI Logger - This one gets the console
ui_logger = logging.getLogger("AETHER_UI")
ui_logger.setLevel(logging.INFO)
ui_logger.addHandler(console_handler)
ui_logger.propagate = False # Don't send to root (avoid double logging to file if root has file handler)
# Also add file handler to UI logger so UI messages go to file too
ui_logger.addHandler(file_handler)

# Suppress TensorFlow warnings completely
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TF logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

# Set specific loggers to WARNING to reduce noise
logging.getLogger("tensorflow").setLevel(logging.ERROR)  # Suppress TF warnings
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("h5py").setLevel(logging.WARNING)
logging.getLogger("gymnasium").setLevel(logging.WARNING)
logging.getLogger("stable_baselines3").setLevel(logging.WARNING)

# Suppress noisy libraries (keep existing suppressions)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("hpack").setLevel(logging.WARNING)
logging.getLogger("aiosqlite").setLevel(logging.WARNING)

# Configure bot component logging
logging.getLogger("RiskManager").setLevel(logging.WARNING)   # Show position management

# TradingEngine: Set to INFO for file logs, but DO NOT attach console handler
# This keeps the terminal clean (only AETHER_UI summaries will show)
logging.getLogger("TradingEngine").setLevel(logging.INFO)

logging.getLogger("COUNCIL").setLevel(logging.WARNING)    # Only show important Council decisions (reduces spam)
logging.getLogger("AetherBot").setLevel(logging.INFO)     # Show system status

# PositionManager: Set to INFO for file logs, but DO NOT attach console handler
logging.getLogger("PositionManager").setLevel(logging.INFO)

logging.getLogger("AETHER_UI").setLevel(logging.INFO)     # Show clean trade plans

logger = logging.getLogger("BotRunner")

from src.main_bot import AetherBot, load_configuration

@asynccontextmanager
async def timeout_context(seconds: int):
    """Context manager for timeout that works on Windows and Unix"""
    if hasattr(signal, 'SIGALRM'):
        # Unix-like systems
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Operation timed out after {seconds} seconds")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
    else:
        # Windows - use asyncio timeout
        try:
            yield
        except asyncio.TimeoutError:
            raise TimeoutError(f"Operation timed out after {seconds} seconds")

async def run_bot_with_timeout(timeout_seconds: int = 90*60):
    """Run the bot with a timeout"""
    logger.info("Starting bot execution...")

    # Load configuration
    config = load_configuration()
    logger.info("Configuration loaded")

    bot = None
    try:
        # Create bot
        bot = AetherBot(config)
        logger.info("Bot created successfully")

        # Run with timeout
        if hasattr(signal, 'SIGALRM'):
            # Unix timeout
            async with timeout_context(timeout_seconds):
                await bot.run()
        else:
            # Windows timeout using asyncio
            await asyncio.wait_for(bot.run(), timeout=timeout_seconds)

        logger.info("Bot execution completed successfully")

    except TimeoutError:
        logger.info(f"Bot execution timed out after {timeout_seconds} seconds")
    except Exception as e:
        logger.error(f"Bot execution failed: {e}")
        raise

if __name__ == "__main__":
    # --- APPLY SYSTEM OPTIMIZATIONS ---
    set_high_priority()
    tune_garbage_collector()

    # Default timeout 90 minutes
    timeout = 90 * 60

    if len(sys.argv) > 1:
        try:
            timeout = int(sys.argv[1])
        except ValueError:
            logger.error("Invalid timeout value, using default 90 minutes")
            timeout = 90 * 60

    try:
        asyncio.run(run_bot_with_timeout(timeout))
    except KeyboardInterrupt:
        logger.info("Bot execution interrupted by user")
    except Exception as e:
        import traceback
        sys.stderr.write(f"CRITICAL ERROR: {e}\n")
        traceback.print_exc()
        sys.stderr.flush()
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)