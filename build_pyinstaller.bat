@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  set "PY_EXE=.venv\Scripts\python.exe"
) else (
  set "PY_EXE=python"
)

%PY_EXE% -m pip show pyinstaller >nul 2>&1
if errorlevel 1 (
  echo PyInstaller is not installed in this Python environment.
  echo Install it with:
  echo   %PY_EXE% -m pip install pyinstaller
  echo.
  pause
  exit /b 1
)

%PY_EXE% -m PyInstaller --noconfirm --clean --onefile --windowed ^
  --name "PyProjects" ^
  --add-data "Pictures;Pictures" ^
  Launch_UI.py

if errorlevel 1 (
  echo.
  echo Build failed. See messages above.
  pause
  exit /b 1
)

echo.
echo Build complete. EXE is in dist\PyProjects.exe
pause
