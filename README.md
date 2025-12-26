# 🚦 Real-Time Traffic Sign & Obstacle Detection (Vietnam)

A Computer Vision system designed to detect **Vietnamese Traffic Signs** and **Road Obstacles** (Pedestrians, Vehicles) in real-time.

Built with **YOLOv11**, **TensorRT**, and **SAHI-style Slicing**.

## 🌟 Key Features

* **⚡ TensorRT Optimized:** Runs purely on `.engine` models for maximum inference speed on NVIDIA GPUs.
* **🔪 Dynamic Slicing (SAHI):** Automatically chops 1080p video into smaller tiles (e.g., 960x960) to detect tiny, far-away traffic signs that standard resizing would miss.
* **🧠 Dual-Core Inference:** Capable of running two distinct models simultaneously:
  * **Core A:** Custom Traffic Sign Model (YOLOv11s).
  * **Core B:** Obstacle/Pedestrian Model (YOLOv8n - COCO).
* **🔄 Hybrid Speed System:** Alternates between "Detailed Slicing" (every N frames) and "Fast Full-Frame" inference to balance accuracy and FPS.
* **⚖️ Label Stabilization:** Uses a custom Voting/Decay algorithm (`PredictionStabilizer`) to prevent label flickering when signs are far away or blurry.
* **🧵 Multithreaded I/O:** Decouples video reading from processing to prevent I/O bottlenecks.

## 🛠️ Installation

### Prerequisites

* **GPU:** NVIDIA RTX 30/40 Series recommended (Tested on RTX 4050 & 3060).
* **Drivers:** CUDA 11.8 or 12.x installed.
* **Python:** 3.8+.

### Setup

1. **Clone the repository:**

    ```bash
    git clone https://github.com/abcdef54/Traffic-Sign-Detection.git
    cd Traffic-Sign-Detection
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    *(Ensure you have `ultralytics`, `supervision`, `opencv-python`, and `numpy`)*.

3. **Prepare Models:**
    * Place your trained Sign model (`.engine` or `.pt` or `.onnx`) in `models/signs/`.
    * (Optional) Place a standard YOLOv8n model in `models/peds/` for pedestrian detection.

## 📂 Project Structure

```text
Traffic-Sign-Detection/
│
├── datasets/                   # Training Data
│   └── VietNamSigns/           # Vietnamese Traffic Sign Dataset
│       ├── data.yaml
│       ├── train/
│       └── val/
│
├── models/                     # Model Weights
│   ├── pedestrians/            # YOLOv8n (COCO) for obstacles
│   └── signs/                  # YOLOv11s (Custom) for signs
│       └── best.engine         # TensorRT Optimized Weight
│
├── src/                        # Core Logic Modules
│   ├── __init__.py
│   ├── model.py                # TensorRTSliceModel (Slicing & Dual-Core Logic)
│   ├── video_reader.py         # Multithreaded Video Capture
│   ├── voting.py               # PredictionStabilizer (Label Smoothing)
│   └── distance.py             # (Placeholder) Distance Estimation
│
├── runs/                       # Training/Inference Outputs
│
├── main.py                     # 🚀 MAIN EXECUTABLE
├── requirements.txt            # Python Dependencies
└── README.md                   # Project Documentation

```

## 🚀 Usage

### 1. Basic Webcam Demo (Signs Only)Run the sign detector on your default webcam (ID 0). Slicing is enabled by default.

```bash
python main.py --input 0 --model models/signs/best.engine --show

```

### 2. Video Processing (Signs + Pedestrians)Run both "Cores" on a video file and save the result.

```bash
python main.py \
  --input videos/dashcam_footage.mp4 \
  --output results/output.mp4 \
  --model models/signs/best.engine \
  --ped-model models/peds/yolov8n.engine \
  --save \
  --verbose

```

### 3. Fast Mode (No Slicing)Disable slicing for maximum FPS (good for testing logic, but may miss small signs)

```bash
python main.py --model models/signs/best.engine --no-slice --show --input "0"

```

## ⚙️ Configuration ArgumentsYou can tweak the system performance via command-line arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--input` | `"0"` | Path to video file or webcam ID (`0`, `1`). |
| `--model` | `Required` | Path to the **Sign Detection** model (`.pt` or `.engine`). |
| `--ped-model` | `""` | Path to **Pedestrian** model. If empty, Dual-Core is disabled. |
| `--no-slice` | `False` | Add this flag to **disable** slicing (run standard Resize only). |
| `--slice-interval` | `5` | Run Slicing every `N` frames. Lower = More accurate but slower. |
| `--conf-detect` | `0.2` | Confidence threshold to detect an object. |
| `--conf-track` | `0.55` | Confidence threshold to start tracking an object. |
| `--verbose` | `False` | Print detailed FPS and detection logs to console. |


## 📜 Credits* **Dataset:** [VNTS merge Computer Vision Model (Roboflow)](https://universe.roboflow.com/nl-gt2le/vnts-merge)

* **Frameworks:** [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), [Supervision](https://github.com/roboflow/supervision)
