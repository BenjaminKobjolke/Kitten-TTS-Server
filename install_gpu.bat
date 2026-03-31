@echo off
title Kitten-TTS GPU Installation

echo Creating virtual environment with Python 3.11...
uv venv --python 3.11 --seed venv
if errorlevel 1 (
    echo Failed to create virtual environment.
    pause
    exit /b 1
)

echo.
echo Step 1: Installing GPU-enabled ONNX Runtime...
venv\Scripts\python.exe -m pip install onnxruntime-gpu
if errorlevel 1 (
    echo Failed to install onnxruntime-gpu.
    pause
    exit /b 1
)

echo.
echo Step 2: Installing PyTorch with CUDA 12.1 support...
venv\Scripts\python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo Failed to install PyTorch.
    pause
    exit /b 1
)

echo.
echo Step 3: Installing remaining dependencies...
venv\Scripts\python.exe -m pip install -r requirements-nvidia.txt
if errorlevel 1 (
    echo Failed to install requirements.
    pause
    exit /b 1
)

echo.
echo Installation complete!
pause
