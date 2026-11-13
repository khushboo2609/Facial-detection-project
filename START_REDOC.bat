
@echo off
title ArcFace API - ReDoc Portal

cd /d "%~dp0"

echo ==========================================
echo   ArcFace Facial Recognition API
echo ==========================================
echo.
echo Starting the API server...
echo ReDoc:   http://127.0.0.1:8000/redoc
echo Swagger: http://127.0.0.1:8000/docs
echo.

start "ArcFace API Server" cmd /k "cd /d ""%~dp0"" && call arcface_env\Scripts\activate.bat && python -m uvicorn app.main:app --reload"

timeout /t 5 /nobreak >nul

start "" "http://127.0.0.1:8000/redoc"

exit
