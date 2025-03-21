@echo off

REM Activate the virtual environment
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call "venv\Scripts\activate.bat"
    echo Virtual environment activated.
) else (
    echo Virtual environment not found. Creating new virtual environment...
    python -m venv venv
    call "venv\Scripts\activate.bat"
    echo Installing requirements...
    pip install -r requirements.txt
    echo Setup completed.
)

REM Display help information
echo.
echo DAPO AI Agent - Environment Activated
echo.
echo Available commands:
echo - python train.py --config config/default.yaml : Train the DAPO agent
echo - python eval.py --model [model_path] --eval-file [file_path] : Evaluate a trained model
echo - python main.py interactive --model [model_path] : Run interactive mode
echo - python scripts/generate_sample_data.py : Generate sample training data
echo.

REM Set the current directory
cd /d "%~dp0"
