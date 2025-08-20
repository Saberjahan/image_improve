@echo off
title Starting Image Repair AI Project

REM --- Configuration ---
SET PROJECT_ROOT=O:\Data\LLM\Image_Improve
SET FRONTEND_DIR=%PROJECT_ROOT%\client
SET BACKEND_DIR=%PROJECT_ROOT%\server
SET REQUIREMENTS_FILE=%BACKEND_DIR%\requirements.txt

echo.
echo === Starting Image Repair AI Project ===
echo.

REM --- Install Python dependencies ---
echo [0/2] Ensuring Python requirements are installed...
IF EXIST "%REQUIREMENTS_FILE%" (
    pushd %BACKEND_DIR%
    call .\venv\Scripts\activate
    pip install -r requirements.txt
    popd
) ELSE (
    echo Requirements file not found at %REQUIREMENTS_FILE%
)

REM --- Start Backend (Flask) ---
echo [1/2] Launching Flask Backend...
start "Flask Backend" cmd /k "cd /d %BACKEND_DIR% && .\venv\Scripts\activate && python -m flask run --port 3001 --debug"

REM --- Start Frontend (React) ---
echo [2/2] Launching React Frontend...
start "React Frontend" cmd /k "cd /d %FRONTEND_DIR% && npm install && npm start"

echo.
echo ----------------------------------------
echo Both servers are starting in new windows.
echo Flask Backend: http://localhost:3001
echo React Frontend: http://localhost:3000
echo ----------------------------------------
echo Press any key to close this window...
pause >nul
exit
