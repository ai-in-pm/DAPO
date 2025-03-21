@echo off

REM Activate the virtual environment
if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
) else (
    echo Virtual environment not found. Please run activate.bat first.
    exit /b 1
)

REM Check if database file exists
if not exist "data\dapo.db" (
    echo Database file not found. Initializing database...
    call "run_init_database.bat"
)

REM Run the visualization script
python scripts\visualize_metrics.py --db-path data\dapo.db

pause
