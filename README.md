# Traffic-Sign-Detection

Traffic-Sign-Recognition/
│
├── datasets/                   # <--- YOUR DATA (For Training Signs Only)
│   ├── raw_downloads/          # Zips you downloaded (GTSRB, GTSDB)
│   └── sign_dataset_v1/        # The dataset you actually train on
│       ├── data.yaml           # The config file for the SIGN model
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       └── val/
│           ├── images/
│           └── labels/
│
├── models/                     # <--- YOUR BRAINS (Now Dual-Core)
│   ├── pedestrians/            # Brain 1: Obstacle Detection
│   │   └── yolov8n.pt          # Pre-trained COCO model (detects people/cars)
│   │
│   ├── signs/                  # Brain 2: Traffic Rules
│   │   ├── yolo11s.pt          # Base model (starting point)
│   │   ├── best_sign_model.pt  # YOUR trained model (backup)
│   │   └── best_sign_model.engine # <--- ACTIVE MODEL: The file "Night Eagle" uses
│   │
│   └── tracker_config.yaml     # ByteTrack settings
│
├── src/                        # <--- YOUR 4 LAYERS (The Code)
│   ├── __init__.py
│   ├── video_loader.py         # <--- NEW: The Threaded Reader (Crucial for 100+ FPS)
│   ├── model.py                # <--- UPDATED: Contains 'SlicedYoloEngine' (InferenceSlicer)
│   ├── tracker.py              # Layer 2: ByteTrack wrapper
│   ├── voting.py                # Layer 3: Voting & Smoothing (Sign stability)
│   └── distance.py             # Layer 4: Math formulas
│
├── runs/                       # <--- AUTOMATIC OUTPUTS
│   └── detect/                 # YOLO saves training charts here
│
├── main.py                     # <--- EXECUTE THIS FILE to run the project
├── train.py                    # Script to train ONLY the Sign model with Night Augmentation
├── calibrate.py                # Script to find your "Focal Length"
└── requirements.txt            # Libraries (now includes onnxruntime-gpu, sahi)