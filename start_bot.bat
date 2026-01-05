@echo off
REM ============================================================================
REM AETHER Trading System - VPS Production Launcher
REM Version: 5.6.5
REM Updated: January 5, 2026
REM
REM This script:
REM - Works from any directory (VPS-ready)
REM - Auto-detects script location
REM - Validates environment
REM - Loads all configurations
REM - Handles errors gracefully
REM ============================================================================

setlocal enabledelayedexpansion

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM Change to script directory
cd /d "%SCRIPT_DIR%"

echo ============================================================================
echo    AETHER Trading Bot v5.6.5
echo    AI-Powered Trading System for MT5
echo ============================================================================
echo.
echo [INFO] Script Location: %SCRIPT_DIR%
echo [INFO] Current Time: %DATE% %TIME%
echo.

REM ============================================================================
REM STEP 1: Environment Validation
REM ============================================================================

echo [STEP 1/6] Validating Environment...

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Please install Python 3.8 or higher.
    echo [ERROR] Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Display Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Python %PYTHON_VERSION% detected
echo.

REM ============================================================================
REM STEP 2: Virtual Environment Setup
REM ============================================================================

echo [STEP 2/6] Setting Up Virtual Environment...

if not exist ".venv" (
    echo [INFO] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment exists
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment!
    pause
    exit /b 1
)
echo [OK] Virtual environment activated
echo.

REM ============================================================================
REM STEP 3: Dependencies Installation
REM ============================================================================

echo [STEP 3/6] Installing/Updating Dependencies...

if not exist "requirements.txt" (
    echo [ERROR] requirements.txt not found!
    pause
    exit /b 1
)

echo [INFO] Installing packages (this may take a few minutes)...
pip install -r requirements.txt --quiet --disable-pip-version-check
if errorlevel 1 (
    echo [WARNING] Some packages failed to install. Continuing anyway...
) else (
    echo [OK] All dependencies installed
)
echo.

REM ============================================================================
REM STEP 4: Configuration Validation
REM ============================================================================

echo [STEP 4/6] Validating Configuration...

REM Check secrets.env
if not exist "config\secrets.env" (
    echo [ERROR] config\secrets.env not found!
    echo.
    echo Please create config\secrets.env with your MT5 credentials:
    echo.
    echo MT5_LOGIN=your_account_number
    echo MT5_PASSWORD=your_password
    echo MT5_SERVER=your_broker_server
    echo.
    pause
    exit /b 1
)
echo [OK] secrets.env found

REM Check settings.yaml
if not exist "config\settings.yaml" (
    echo [ERROR] config\settings.yaml not found!
    pause
    exit /b 1
)
echo [OK] settings.yaml found

REM Check run_bot.py
if not exist "run_bot.py" (
    echo [ERROR] run_bot.py not found!
    pause
    exit /b 1
)
echo [OK] run_bot.py found
echo.

REM ============================================================================
REM STEP 5: Environment Variables Setup
REM ============================================================================

echo [STEP 5/6] Setting Environment Variables...

REM Critical: MT5 timezone offset (adjust for your broker)
set AETHER_TIME_OFFSET_SECS=7196
echo [OK] Timezone offset: %AETHER_TIME_OFFSET_SECS%s

REM Performance optimizations
set AETHER_ENABLE_FRESHNESS_GATE=1
set AETHER_FRESH_TICK_MAX_AGE_S=10.0
echo [OK] Freshness gate enabled (10s threshold)

REM Dashboard settings
set AETHER_TRADER_DASHBOARD=1
set AETHER_SUPPRESS_TECHNICAL_LOGS=1
echo [OK] Trader dashboard enabled

REM Optional: Uncomment to enable debug mode
REM set AETHER_DEBUG=1
REM set AETHER_RUNTIME_TRACE=1

echo.

REM ============================================================================
REM STEP 6: Trading Mode Confirmation
REM ============================================================================

echo [STEP 6/6] Checking Trading Mode...

REM Check if PAPER or LIVE mode
findstr /C:"mode: \"PAPER\"" config\settings.yaml >nul 2>&1
if %errorlevel%==0 (
    echo [INFO] Running in PAPER TRADING mode
    echo [INFO] Safe for testing - no real money at risk
    set TRADING_MODE=PAPER
) else (
    echo.
    echo ============================================================================
    echo    WARNING: LIVE TRADING MODE DETECTED!
    echo ============================================================================
    echo.
    echo This will trade with REAL MONEY on your MT5 account.
    echo.
    echo Press Y to continue with LIVE trading
    echo Press N to cancel and switch to PAPER mode
    echo.
    choice /C YN /M "Continue with LIVE trading?"
    if errorlevel 2 (
        echo.
        echo [INFO] Trading cancelled. Please edit config\settings.yaml
        echo [INFO] Change 'mode: "LIVE"' to 'mode: "PAPER"' for safe testing
        pause
        exit /b 0
    )
    set TRADING_MODE=LIVE
)

echo.
echo ============================================================================
echo    STARTING AETHER BOT
echo ============================================================================
echo.
echo [INFO] Mode: %TRADING_MODE%
echo [INFO] Version: 5.6.5
echo [INFO] Directory: %SCRIPT_DIR%
echo [INFO] Press Ctrl+C to stop the bot
echo.
echo ============================================================================
echo.

REM ============================================================================
REM Launch the bot
REM ============================================================================

REM Run with error handling
python run_bot.py
set EXIT_CODE=%errorlevel%

echo.
echo ============================================================================
echo    BOT STOPPED
echo ============================================================================
echo.

if %EXIT_CODE% neq 0 (
    echo [ERROR] Bot exited with error code: %EXIT_CODE%
    echo [INFO] Check the logs above for details
) else (
    echo [INFO] Bot exited normally
)

echo.
echo Press any key to close this window...
pause >nul

exit /b %EXIT_CODE%
