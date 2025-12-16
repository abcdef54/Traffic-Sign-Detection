import time
import numpy as np
from ultralytics import YOLO

# --- CONFIGURATION ---
PT_MODEL_PATH = 'runs/detect/night_eagle_vietnam2/weights/best.pt'
TRT_MODEL_PATH = 'runs/detect/night_eagle_vietnam2/weights/best.engine'
IMAGE_SIZE = 960
NUM_FRAMES = 500  # How many frames to test

def run_benchmark(model_path, name):
    print(f"\n[{name}] Loading model...")
    try:
        # task='detect' is safer for exported models
        model = YOLO(model_path, task='detect') 
    except Exception as e:
        print(f"Error loading {name}: {e}")
        return None

    # Create fake image in RAM (H, W, C)
    dummy_image = np.random.randint(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

    print(f"[{name}] Warming up GPU (10 runs)...")
    for _ in range(10):
        model(dummy_image, verbose=False)

    print(f"[{name}] Running {NUM_FRAMES} inference loops...")
    start_time = time.time()

    for _ in range(NUM_FRAMES):
        model(dummy_image, verbose=False)

    end_time = time.time()
    
    # Calculate stats
    duration = end_time - start_time
    latency_ms = (duration / NUM_FRAMES) * 1000
    fps = NUM_FRAMES / duration
    
    return latency_ms, fps

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print(f"BENCHMARKING: PyTorch (.pt) vs TensorRT (.engine)")
    print(f"Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print("--------------------------------------------------")

    # 1. Test PyTorch
    pt_stats = run_benchmark(PT_MODEL_PATH, "PyTorch")

    # 2. Test TensorRT
    trt_stats = run_benchmark(TRT_MODEL_PATH, "TensorRT")

    # 3. Print Comparison
    print("\n\n==================================================")
    print(f"FINAL RESULTS (Lower Latency / Higher FPS is better)")
    print("==================================================")
    
    if pt_stats:
        print(f"PyTorch (.pt):     {pt_stats[0]:6.2f} ms  |  {pt_stats[1]:6.2f} FPS")
    
    if trt_stats:
        print(f"TensorRT (.engine):{trt_stats[0]:6.2f} ms  |  {trt_stats[1]:6.2f} FPS")

    if pt_stats and trt_stats:
        speedup = pt_stats[1] / trt_stats[1]
        print("--------------------------------------------------")
        print(f"SPEEDUP FACTOR:    {speedup:.2f}x FASTER")
        print("==================================================")