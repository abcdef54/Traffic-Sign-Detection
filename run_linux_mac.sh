#!/bin/bash


INPUT="test_videos/fix_2mins.mp4"
MODEL_ENGINE="models/signs/best_dynamic.engine"
MODEL_PYTORCH="models/signs/best.pt"
OUTPUT="test_videos/output.mp4"

FLAGS="--save --verbose"

# --- Automated Image Verification ---
echo "[INFO] Checking for local Docker image 'traffic-sign-detection'..."

if [ -z "$(docker images -q traffic-sign-detection 2> /dev/null)" ]; then
    echo "[WARN] Docker image not found locally. Initiating environment build..."
    echo "[INFO] Executing: docker build -t traffic-sign-detection ."
    
    docker build -t traffic-sign-detection .
    
    if [ $? -ne 0 ]; then
        echo "[ERROR] Docker build failed. Please check your Dockerfile and internet connection."
        exit 1
    fi
    echo "[INFO] Environment built successfully. Proceeding..."
    echo "========================================================="
else
    echo "[INFO] Found existing local image environment. Skipping build phase."
    echo "========================================================="
fi

echo "========================================================="
echo "  Traffic Sign Detection - Containerized Execution"
echo "========================================================="
echo "Enter your input source:"
echo "1) Local Video File (outputs_vids/fix_2mins.mp4)"
echo "2) Live Webcam (Device 0)"
read -p "Enter choice [1-2]: " choice

if [ "$choice$"  == "1" ]; then
    echo "Select model format"
    echo "1) .engine (NVIDIA TensorRT)"
    echo "2) .pt (Pytorch)"
    read -p "Enter choice [1-2]: " model_choice

    if [ "$model_choice" == "1" ]; then
        MODEL_PATH="$MODEL_ENGINE"
        MODEL_FLAG="--engine"
    else
        MODEL_PATH="$MODEL_PYTORCH"
        MODEL_FLAG="--pt"
    fi

    echo "Perform slice inference (y/n)"
    read -p "Enter choice [y/n]: " slice_choice

    if [ "$slice_choice" == "n" ]; then
        FLAGS="$FLAGS --no-slice"
    fi

    echo "Show video in real-time (y/n)"
    read -p "Enter choice [y/n]: " show_choice

    if [ "$show_choice" == "y" ]; then
        FLAGS="$FLAGS --show"
    fi

    echo "[INFO] Using model: $MODEL_PATH"
    echo "[INFO] Running inference..."

    docker run --gpus all --rm \
      -v $(pwd)/models:/app/models \
      -v $(pwd)/outputs_vids:/app/outputs_vids \
      traffic-sign-detection --input $INPUT --output $OUTPUT $FLAGS $MODEL_FLAG

elif [ "$choice$" == "2" ]; then
    echo "Select model format"
    echo "1) .engine (NVIDIA TensorRT)"
    echo "2) .pt (Pytorch)"
    read -p "Enter choice [1-2]: " model_choice

    if [ "$model_choice" == "1" ]; then
        MODEL_PATH="$MODEL_ENGINE"
        MODEL_FLAG="--engine"
    else
        MODEL_PATH="$MODEL_PYTORCH"
        MODEL_FLAG="--pt"
    fi

    echo "Perform slice inference (y/n)"
    read -p "Enter choice [y/n]: " slice_choice

    if [ "$slice_choice" == "n" ]; then
        FLAGS="$FLAGS --no-slice"
    fi

    echo "[INFO] Using model: $MODEL_PATH"
    echo "[INFO] Running inference..."

    docker run --gpus all --rm --net=host --ipc=host --device=/dev/video0:/dev/video0 \
           -e DISPLAY=$DISPLAY \
           -v /tmp/.X11-unix:/tmp/.X11-unix \
           -v $(pwd)/models:/app/models \
           traffic-sign-detection --input 0 --output /dev/video0 $FLAGS $MODEL_FLAG
fi