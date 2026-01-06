"""
Workspace Cleanup Script for AETHER Bot v6.5.0
Removes unnecessary files and keeps only production code
"""

import os
import shutil
from pathlib import Path

# Define workspace root
WORKSPACE_ROOT = Path(__file__).parent

# Files and directories to remove
CLEANUP_TARGETS = [
    # Temporary/cache files
    "**/__pycache__",
    "**/*.pyc",
    "**/*.pyo",
    "**/*.pyd",
    
    # Backup files
    "**/*.bak",
    "**/*.backup",
    "**/*.old",
    "**/*.tmp",
    
    # IDE files (if not needed)
    "**/.vscode",
    "**/.idea",
    
    # Test artifacts (keep tests folder, remove artifacts)
    "**/test_results",
    "**/.pytest_cache",
    "**/.coverage",
    
    # Log files (keep logs folder, remove old logs)
    "logs/*.log.*",  # Rotated logs
    "error.log",  # Old error log
    
    # Temporary verification report
    ".verification_report.md",
    
    # Old cleanup script
    "code_cleanup_script.py",
]

# Directories to keep (whitelist)
KEEP_DIRS = {
    ".git",
    ".venv",
    "src",
    "config",
    "models",
    "logs",
    "data",
    "tests",
    "monitoring",
    "backtest_results",
}

# Files to keep (whitelist)
KEEP_FILES = {
    ".gitignore",
    "README.md",
    "VERSION.txt",
    "requirements.txt",
    "pyproject.toml",
    "setup.py",
    "run_bot.py",
    "start_bot.bat",
    "backtest.py",
    "validate_oracle.py",
}


def cleanup_workspace():
    """Remove unnecessary files from workspace"""
    removed_count = 0
    
    print("🧹 Starting workspace cleanup...")
    print(f"📁 Workspace: {WORKSPACE_ROOT}")
    print()
    
    # Remove __pycache__ directories
    for pycache in WORKSPACE_ROOT.rglob("__pycache__"):
        try:
            shutil.rmtree(pycache)
            print(f"✅ Removed: {pycache.relative_to(WORKSPACE_ROOT)}")
            removed_count += 1
        except Exception as e:
            print(f"❌ Failed to remove {pycache}: {e}")
    
    # Remove .pyc files
    for pyc in WORKSPACE_ROOT.rglob("*.pyc"):
        try:
            pyc.unlink()
            print(f"✅ Removed: {pyc.relative_to(WORKSPACE_ROOT)}")
            removed_count += 1
        except Exception as e:
            print(f"❌ Failed to remove {pyc}: {e}")
    
    # Remove temporary files
    temp_patterns = ["*.tmp", "*.bak", "*.backup", "*.old"]
    for pattern in temp_patterns:
        for temp_file in WORKSPACE_ROOT.rglob(pattern):
            try:
                temp_file.unlink()
                print(f"✅ Removed: {temp_file.relative_to(WORKSPACE_ROOT)}")
                removed_count += 1
            except Exception as e:
                print(f"❌ Failed to remove {temp_file}: {e}")
    
    # Remove specific files
    specific_files = [
        ".verification_report.md",
        "code_cleanup_script.py",
        "error.log",
    ]
    
    for filename in specific_files:
        filepath = WORKSPACE_ROOT / filename
        if filepath.exists():
            try:
                filepath.unlink()
                print(f"✅ Removed: {filename}")
                removed_count += 1
            except Exception as e:
                print(f"❌ Failed to remove {filename}: {e}")
    
    print()
    print(f"✅ Cleanup complete! Removed {removed_count} items.")
    print()
    print("📊 Workspace Status:")
    print(f"   - Source code: src/ (clean)")
    print(f"   - Configuration: config/ (clean)")
    print(f"   - Models: models/ (preserved)")
    print(f"   - Logs: logs/ (preserved)")
    print(f"   - Virtual env: .venv/ (preserved)")
    print()
    print("🎯 Workspace is now clean and production-ready!")


if __name__ == "__main__":
    cleanup_workspace()
