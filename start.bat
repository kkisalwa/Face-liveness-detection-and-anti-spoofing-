@echo off
title FaceShield — Anti-Spoofing Detection System
color 0B

echo.
echo  ============================================================
echo    FaceShield  -  Face Anti-Spoofing Detection System
echo  ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  [ERROR] Python is not installed or not in PATH.
    echo          Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

:: Install dependencies
echo  [STEP 1/2] Installing dependencies...
cd backend
pip install -r requirements.txt --quiet

if %errorlevel% neq 0 (
    echo  [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo  [STEP 2/2] Starting FaceShield server...
echo.
echo  ============================================================
echo    Dashboard  ^>  http://localhost:8000
echo    API Docs   ^>  http://localhost:8000/docs
echo  ============================================================
echo.
echo  [TIP] Press CTRL+C to stop the server.
echo.

:: Open browser after a short delay
start "" /B cmd /C "timeout /t 2 /nobreak >nul && start http://localhost:8000"

:: Start server
python main.py

pause
