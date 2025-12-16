import cv2
import time
import os
import sys
import numpy as np
from collections import deque, defaultdict
from src import TensorRTSliceModel
import supervision as sv

# --- BENCHMARK CONFIGURATION ---
# Swap this path to test each model one by one
ENGINE_PATH  = {0: "models/signs/best_f16_static_960_batch1.engine", 
                1: "models/signs/best_f16_static_960_batch8.engine",
                2: "models/signs/best_f16_static_1280_batch1.engine",
                3: "models/signs/best_f16_static_1920_batch1.engine",
                4: "models/signs/best_f16_dynamic_1920_batch16.engine" }
VIDEO_PATH  = "SanityCheck/videos/Test_8mins_Part3.mp4"

IMG_SIZE = 1920        # MUST match engine export
WARMUP = 100
TEST_FRAMES = 1000

engine = TensorRTSliceModel(
    sign_model_path=ENGINE_PATH[3],
    ped_model_path=None,
    slice_inference=False,   # ❗ critical
    imgsz=IMG_SIZE,
    conf=0.25
)

cap = cv2.VideoCapture(VIDEO_PATH)

# ---- Warmup ----
for _ in range(WARMUP):
    ret, frame = cap.read()
    if not ret:
        break
    engine(frame)

# ---- Benchmark ----
latencies = []
frame_count = 0
start = time.time()

while frame_count < TEST_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break

    t0 = time.time()
    _ = engine(frame)
    t1 = time.time()

    latencies.append((t1 - t0) * 1000)
    frame_count += 1

end = time.time()
cap.release()

lat = np.array(latencies)

print("\n========== BENCHMARK ==========")
print(f"Engine: {ENGINE_PATH}")
print(f"Resolution: {IMG_SIZE}")
print(f"Avg latency: {lat.mean():.2f} ms")
print(f"P95 latency: {np.percentile(lat, 95):.2f} ms")
print(f"FPS: {TEST_FRAMES / (end - start):.2f}")
print("================================")