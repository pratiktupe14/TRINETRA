@echo off
cd %~dp0
echo Installing Backend Dependencies...
pip install -r requirements.txt
echo.
echo Setup Complete!
