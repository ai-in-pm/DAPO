@echo off

REM Activate the virtual environment
if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
) else (
    echo Virtual environment not found. Please run activate.bat first.
    exit /b 1
)

REM Create data directories
if not exist "data" mkdir data

REM Run the dataset generation script
python scripts\generate_sample_data.py --num-samples 200 --output data\dataset.jsonl

echo.
echo Data generation completed. You can now train the model using:
echo python train.py --config config/default.yaml
echo.

pause
