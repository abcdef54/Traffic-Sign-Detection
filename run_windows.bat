@echo off
setlocal enabledelayedexpansion

:: --- Global Configurations ---
set "INPUT=test_videos/fix_2mins.mp4"
set "MODEL_ENGINE=models/signs/best_dynamic.engine"
set "MODEL_PYTORCH=models/signs/best.pt"
set "OUTPUT=test_videos/output.mp4"
set "FLAGS=--save --verbose"

:: --- Automated Image Verification ---
echo [INFO] Checking for local Docker image 'traffic-sign-detection:latest'...
docker images --format "{{.Repository}}:{{.Tag}}" | findstr /X "traffic-sign-detection:latest" >nul 2>&1

if %errorlevel% neq 0 (
    echo [WARN] 'traffic-sign-detection:latest' not found. Initiating clean build...
    echo [INFO] Executing: docker build --no-cache -t traffic-sign-detection:latest .
    docker build -t traffic-sign-detection:latest .
    if !errorlevel! neq 0 (
        echo [ERROR] Docker build failed. Please check your Dockerfile configuration.
        pause
        exit /b !errorlevel!
    )
    echo [INFO] Environment built successfully. Proceeding...
    echo =========================================================
) else (
    echo [INFO] Found valid 'traffic-sign-detection:latest' image environment. Skipping build phase.
    echo =========================================================
)

echo.
echo =========================================================
echo  Traffic Sign Detection - Containerized Execution
echo =========================================================
echo.
echo Enter your input source:
echo 1. Local Video File (outputs_vids/fix_2mins.mp4)
echo 2. Live Webcam (Device 0)
set /p "choice=Enter choice [1-2]: "

if "%choice%"=="1" (
    echo.
    echo Select model format
    echo 1. .engine (NVIDIA TensorRT)
    echo 2. .pt (PyTorch)
    set /p "model_choice=Enter choice [1-2]: "

    if "!model_choice!"=="1" (
        set "MODEL_PATH=%MODEL_ENGINE%"
        set "MODEL_FLAG=--model models/signs/best_dynamic.engine"
    ) else (
        set "MODEL_PATH=%MODEL_PYTORCH%"
        set "MODEL_FLAG=--model models/signs/best.pt"
    )

    echo.
    echo Perform slice inference (y/n)
    set /p "slice_choice=Enter choice [y/n]: "

    if "!slice_choice!"=="n" (
        set "FLAGS=!FLAGS! --no-slice"
    )

    echo.
    echo Show video in real-time (y/n)
    set /p "show_choice=Enter choice [y/n]: "

    if "!show_choice!"=="y" (
        set "FLAGS=!FLAGS! --show"
    )

    echo [INFO] Using model: !MODEL_PATH!
    echo [INFO] Running inference...

    docker run --gpus all --rm ^
      -v "%cd%/models:/app/models" ^
      -v "%cd%/outputs_vids:/app/outputs_vids" ^
      traffic-sign-detection --input %INPUT% --output %OUTPUT% !FLAGS! !MODEL_FLAG!

) else if "%choice%"=="2" (
    echo.
    echo Select model format
    echo 1. .engine (NVIDIA TensorRT)
    echo 2. .pt (PyTorch)
    set /p "model_choice=Enter choice [1-2]: "

    if "!model_choice!"=="1" (
        set "MODEL_PATH=%MODEL_ENGINE%"
        set "MODEL_FLAG=--model models/signs/best_dynamic.engine"
    ) else (
        set "MODEL_PATH=%MODEL_PYTORCH%"
        set "MODEL_FLAG=--model models/signs/best.pt"
    )

    echo.
    echo Perform slice inference (y/n)
    set /p "slice_choice=Enter choice [y/n]: "

    if "!slice_choice!"=="n" (
        set "FLAGS=!FLAGS! --no-slice"
    )

    echo [INFO] Using model: !MODEL_PATH!
    echo [INFO] Running inference...

    :: Note: On Windows Docker Desktop, passing the host display engine requires a running 
    :: XServer configuration tool (like VcXsrv) on the host machine.
    docker run --gpus all --rm --net=host --ipc=host ^
      -e DISPLAY=host.docker.internal:0.0 ^
      -v "%cd%/models:/app/models" ^
      traffic-sign-detection --input 0 !FLAGS! !MODEL_FLAG! --show
)

pause
endlocal