@echo off
rem One-click start for Windows: creates .venv and installs deps on first run, then opens the web dashboard.
rem Usage: double-click start.bat, or run "start.bat --port 9000" in cmd.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo First run: creating .venv and installing dependencies ...
  where py >nul 2>nul && (py -3 -m venv .venv) || (python -m venv .venv)
  if not exist ".venv\Scripts\python.exe" (
    echo Failed to create venv. Install Python 3.9+ from https://www.python.org/downloads/ and tick "Add python.exe to PATH".
    pause
    exit /b 1
  )
  ".venv\Scripts\python.exe" -m pip install -q -r requirements.txt || (pause & exit /b 1)
)
".venv\Scripts\python.exe" -m emo.cli web --open %*
pause
