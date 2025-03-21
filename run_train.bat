@echo off

REM Activate the virtual environment
if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
) else (
    echo Virtual environment not found. Please run activate.bat first.
    exit /b 1
)

REM Create necessary directories
if not exist "logs" mkdir logs
if not exist "models\checkpoints" mkdir models\checkpoints

REM Check if training data exists
if not exist "data\dataset_train.jsonl" (
    echo Training data not found. Running data generation first...
    python scripts\generate_sample_data.py --num-samples 200 --output data\dataset.jsonl
    echo Data generation completed.
)

REM Run the training script
python train.py --config config/default.yaml

echo.
echo Training completed. You can evaluate the model using:
echo python eval.py --model models/checkpoints/dapo_final --eval-file data/dataset_eval.jsonl
echo.

pause
