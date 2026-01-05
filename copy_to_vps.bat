@echo off
REM ============================================================================
REM AETHER - Copy to VPS Local Drive
REM This script copies the bot from RDP shared drive to VPS local drive
REM ============================================================================

echo ============================================================================
echo    AETHER Bot - Copy to VPS Local Drive
echo ============================================================================
echo.

REM Get current directory (source)
set "SOURCE_DIR=%~dp0"
set "SOURCE_DIR=%SOURCE_DIR:~0,-1%"

REM Default destination
set "DEST_DIR=C:\Scalping_Gold"

echo Current location: %SOURCE_DIR%
echo.
echo This script will copy the bot to: %DEST_DIR%
echo.
echo Press Y to continue
echo Press N to specify a different location
echo.
choice /C YN /M "Use default location (C:\Scalping_Gold)?"

if errorlevel 2 (
    echo.
    set /p DEST_DIR="Enter destination path (e.g., C:\Trading\AETHER): "
)

echo.
echo ============================================================================
echo    Copying Files
echo ============================================================================
echo.
echo From: %SOURCE_DIR%
echo To:   %DEST_DIR%
echo.

REM Create destination directory if it doesn't exist
if not exist "%DEST_DIR%" (
    echo Creating directory: %DEST_DIR%
    mkdir "%DEST_DIR%"
)

REM Copy all files
echo Copying files (this may take a minute)...
xcopy "%SOURCE_DIR%\*" "%DEST_DIR%\" /E /I /Y /EXCLUDE:%SOURCE_DIR%\.gitignore

if errorlevel 1 (
    echo.
    echo [ERROR] Copy failed!
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo    Copy Complete!
echo ============================================================================
echo.
echo Bot copied to: %DEST_DIR%
echo.
echo NEXT STEPS:
echo 1. Navigate to: %DEST_DIR%
echo 2. Edit config\secrets.env with your MT5 credentials
echo 3. Run start_bot.bat
echo.
echo Opening destination folder...
explorer "%DEST_DIR%"

echo.
pause
