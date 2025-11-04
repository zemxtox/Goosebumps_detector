@echo off
echo ========================================
echo  CHILLER HTTPS Setup
echo ========================================
echo.

REM Check if certificates exist
if not exist "localhost.crt" (
    echo 🔐 Generating SSL certificates...
    python generate_cert.py
    echo.
)

if exist "localhost.crt" (
    echo 🚀 Starting HTTPS server...
    echo.
    echo 📝 Instructions:
    echo 1. Browser will show security warning
    echo 2. Click "Advanced" or "Show details"
    echo 3. Click "Proceed to localhost (unsafe)"
    echo 4. This is normal for self-signed certificates
    echo.
    echo 🌐 Opening browser...
    timeout /t 3 /nobreak >nul
    start https://localhost:8000
    echo.
    python chiller_https.py
) else (
    echo ❌ Failed to generate certificates
    echo Falling back to HTTP version...
    python chiller.py
)

pause