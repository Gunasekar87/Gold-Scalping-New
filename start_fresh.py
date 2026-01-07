"""
AETHER Bot - Safe Startup Script
Ensures latest code is always loaded by clearing Python cache before starting.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def clear_python_cache():
    """Remove all __pycache__ directories and .pyc files."""
    print("🧹 Clearing Python cache to ensure latest code...")
    
    cache_cleared = 0
    project_root = Path(__file__).parent
    
    # Remove __pycache__ directories
    for pycache_dir in project_root.rglob("__pycache__"):
        try:
            shutil.rmtree(pycache_dir)
            cache_cleared += 1
            print(f"   ✓ Removed: {pycache_dir.relative_to(project_root)}")
        except Exception as e:
            print(f"   ⚠ Failed to remove {pycache_dir}: {e}")
    
    # Remove .pyc files
    for pyc_file in project_root.rglob("*.pyc"):
        try:
            pyc_file.unlink()
            cache_cleared += 1
        except Exception as e:
            print(f"   ⚠ Failed to remove {pyc_file}: {e}")
    
    # Remove .pyo files
    for pyo_file in project_root.rglob("*.pyo"):
        try:
            pyo_file.unlink()
            cache_cleared += 1
        except Exception as e:
            print(f"   ⚠ Failed to remove {pyo_file}: {e}")
    
    if cache_cleared > 0:
        print(f"✅ Cleared {cache_cleared} cache items")
    else:
        print("✅ No cache to clear (already clean)")
    print()

def verify_version():
    """Verify we're running the latest version."""
    print("🔍 Verifying bot version...")
    
    try:
        # Read version from constants.py
        constants_file = Path(__file__).parent / "src" / "constants.py"
        with open(constants_file, 'r', encoding='utf-8') as f:
            for line in f:
                if 'SYSTEM_VERSION' in line and '=' in line:
                    version = line.split('=')[1].strip().strip('"').strip("'")
                    print(f"✅ Bot Version: {version}")
                    break
    except Exception as e:
        print(f"⚠ Could not verify version: {e}")
    print()

def check_git_status():
    """Check if there are uncommitted changes."""
    print("🔍 Checking Git status...")
    
    try:
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        
        if result.returncode == 0:
            if result.stdout.strip():
                print("⚠ WARNING: Uncommitted changes detected!")
                print("   You may be running modified code.")
                print(result.stdout)
            else:
                print("✅ Git status: Clean (no uncommitted changes)")
        else:
            print("⚠ Could not check Git status")
    except FileNotFoundError:
        print("⚠ Git not found (skipping check)")
    except Exception as e:
        print(f"⚠ Git check failed: {e}")
    print()

def start_bot():
    """Start the bot with fresh Python environment."""
    print("🚀 Starting AETHER Bot...")
    print("=" * 60)
    print()
    
    # Use python -B to prevent bytecode generation
    # This ensures we always run from source
    bot_script = Path(__file__).parent / "run_bot.py"
    
    # Start bot with -B flag (no bytecode)
    # and -u flag (unbuffered output)
    subprocess.run([
        sys.executable,
        '-B',  # Don't write .pyc files
        '-u',  # Unbuffered output
        str(bot_script)
    ])

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  AETHER BOT - SAFE STARTUP")
    print("  Ensuring Latest Code is Loaded")
    print("=" * 60)
    print()
    
    # Step 1: Clear cache
    clear_python_cache()
    
    # Step 2: Verify version
    verify_version()
    
    # Step 3: Check Git status
    check_git_status()
    
    # Step 4: Start bot
    start_bot()
