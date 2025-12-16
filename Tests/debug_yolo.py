from ultralytics import YOLO
import cv2
import os

# CONFIGURATION
MODEL_PATH = "models/signs/untouch/yolo11s.engine" # Try .pt if this fails
IMAGE_PATH = "cars.jpg"

def sanity_check():
    print(f"--- DEBUGGING: {MODEL_PATH} on {IMAGE_PATH} ---")
    
    # 1. Check Paths
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return
    if not os.path.exists(IMAGE_PATH):
        print(f"ERROR: Image not found at {IMAGE_PATH}")
        return

    # 2. Load Image manually (Simulate what SAHI does)
    # OpenCV loads as BGR by default
    img_bgr = cv2.imread(IMAGE_PATH)
    print(f"Image Shape: {img_bgr.shape}")

    # 3. Load Model
    try:
        model = YOLO(MODEL_PATH, task='detect')
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Model load failed: {e}")
        return

    # 4. Run Inference DIRECTLY
    print("Running raw inference...")
    results = model(img_bgr, verbose=True, conf=0.10) # Low conf to see EVERYTHING

    # 5. Check Results
    for r in results:
        print(f"\nRaw Detections found: {len(r.boxes)}")
        for box in r.boxes:
            print(f" - Class: {int(box.cls)} | Conf: {float(box.conf):.2f} | Coords: {box.xyxy.tolist()}")
        
        # Save a debug image directly from Ultralytics
        r.save(filename="debug_direct_output.jpg")
        print("\nSaved 'debug_direct_output.jpg'. Open this to see if boxes exist.")

if __name__ == "__main__":
    sanity_check()