@echo off

REM Activate the virtual environment
if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
) else (
    echo Virtual environment not found. Please run activate.bat first.
    exit /b 1
)

REM Check if there's a final model, otherwise use the latest epoch model
if exist "models\checkpoints\dapo_final" (
    set MODEL_PATH=models\checkpoints\dapo_final
) else (
    echo Looking for the latest checkpoint...
    for /f "delims=" %%i in ('dir /b /ad /o-n "models\checkpoints\dapo_epoch_*" 2^>nul') do (
        set MODEL_PATH=models\checkpoints\%%i
        goto found_model
    )
    
    echo No trained model found. Please run training first.
    exit /b 1
    
    :found_model
)

echo Using model: %MODEL_PATH%

REM Run the interactive mode
python main.py interactive --model %MODEL_PATH% --config config/default.yaml

pause
