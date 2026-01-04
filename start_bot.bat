@echo off
REM AETHER Trading System - Quick Start
REM Version 5.5.3
@echo off
echo ================================================
echo    AETHER Trading Bot v5.5.6
echo    Production-Ready AI Trading System
echo ================================================
echo.

REM [CRITICAL FIX] Set timezone offset for MT5 server (fixes freshness gate)
set AETHER_TIME_OFFSET_SECS=7196

REM Activate virtual environment exists
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Install/update dependencies
echo Installing dependencies...
pip install -r requirements.txt --quiet
echo.

REM Check configuration
if not exist config\secrets.env (
    echo WARNING: config\secrets.env not found!
    echo Please create it with your MT5 credentials.
    echo.
    echo Example:
    echo MT5_LOGIN=your_account_number
    echo MT5_PASSWORD=your_password
    echo MT5_SERVER=your_broker_server
    echo.
    pause
    exit /b 1
)

REM Check mode
findstr /C:"mode: \"PAPER\"" config\settings.yaml >nul
if %errorlevel%==0 (
    echo [INFO] Running in PAPER mode
) else (
    echo [WARNING] Running in LIVE mode!
    choice /C YN /M "Continue with LIVE trading?"
    if errorlevel 2 exit /b
)

echo.
echo Starting AETHER Bot...
echo Press Ctrl+C to stop
echo.

python run_bot.py 3600

pause

