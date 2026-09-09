@echo off
REM Setup script for Windows local environment
echo ========================================================
echo Setting up Self-Improving Research Assistant Environment
echo ========================================================

IF NOT EXIST "venv" (
    echo Creating virtual environment 'venv'...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies via pip...
pip install -r requirements.txt

echo Environment setup complete!
echo To run tests: pytest -v tests/
echo To run self-improvement loop: python scripts/run_self_improvement_loop.py
echo To start web server: python scripts/run_server.py
