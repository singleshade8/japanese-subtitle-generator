@echo off
cd /d "%~dp0"

echo ===============================
echo Setting up Subtitle Generator
echo ===============================

REM Check if venv exists, else create it
if not exist venv\Scripts\python.exe (
    echo Creating virtual environment...
    py -3.10 -m venv venv
    if errorlevel 1 (
        echo Failed to create venv. Make sure Python 3.10 is installed.
        pause
        exit /b 1
    )
) else (
    echo Virtual environment already exists.
)

REM Activate venv
call venv\Scripts\activate

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
if exist requirements.txt (
    echo Installing requirements...
    python -m pip install -r requirements.txt
) else (
    echo requirements.txt not found!
    pause
    exit /b 1
)

echo ===============================
echo Setup complete!
echo To run the app:
echo   venv\Scripts\activate
echo   streamlit run app.py
echo ===============================

pause
