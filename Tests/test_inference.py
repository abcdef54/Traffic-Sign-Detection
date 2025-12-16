import cv2
import time
import numpy as np
from ultralytics import YOLO

# 1. Setup
PT_MODEL_PATH = 'models/signs/yolo11s.pt'      # Replace with your path
TRT_MODEL_PATH = 'models/signs/best_sign_model.engine' # Replace with your path
IMAGE_SIZE = 960

def benchmark(model_path, name):
    print(f"\n--- Testing {name} ---")
    model = YOLO(model_path, task='detection')
    
    # Create a dummy image (Random noise) in RAM
    dummy_frame = np.random.randint(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    
    # Warmup (GPUs need a few runs to 'wake up')
    print("Warming up...")
    for _ in range(10):
        model(dummy_frame, verbose=False)
        
    # Benchmark Loop
    print("Running 500 inference loops (No I/O)...")
    start_time = time.time()
    
    for _ in range(500):
        # We perform inference on the same memory object
        # This measures Pure Compute + Python Overhead
        model(dummy_frame, verbose=False)
        
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_ms = (total_time / 500) * 1000
    fps = 1 / (total_time / 500)
    
    print(f"Results for {name}:")
    print(f"  Avg Latency: {avg_ms:.2f} ms")
    print(f"  Throughput:  {fps:.2f} FPS")

# Run both
if __name__ == "__main__":
    # 1. Test PyTorch
    benchmark(PT_MODEL_PATH, "PyTorch (.pt)")
    
    # 2. Test TensorRT
    benchmark(TRT_MODEL_PATH, "TensorRT (.engine)")