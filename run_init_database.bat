@echo off

REM Activate the virtual environment
if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
) else (
    echo Virtual environment not found. Please run activate.bat first.
    exit /b 1
)

REM Create data directory
if not exist "data" mkdir data

REM Run the database initialization script
python scripts\init_database.py --db-path data\dapo.db

echo.
echo Database initialization completed.
echo.

pause
