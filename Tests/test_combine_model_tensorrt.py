import cv2
import time
import numpy as np
from ultralytics import YOLO

# SAHI imports
from sahi.prediction import ObjectPrediction
from sahi.postprocess.combine import NMSPostprocess

# --- CONFIGURATION ---
MODEL_PATH = 'yolo11s.engine'
CLASS_NAMES = {0: 'speed_limit', 1: 'stop', 2: 'yield', 3: 'no_entry'} # Update yours!

# SLICING CONFIG (2x2 Grid)
SLICE_W, SLICE_H = 320, 240 # Adjust based on your camera resolution (e.g. 640x480)
OVERLAP = 0.2

def load_model():
    try:
        model = YOLO(MODEL_PATH, task='detect')
        # model.model.names = CLASS_NAMES 
        # Warmup
        model(np.zeros((640,640,3), np.uint8), verbose=False)
        return model
    except Exception as e:
        print(f"Error: {e}")
        return None

def convert_to_sahi(results, offset=(0,0)):
    """ Converts YOLO results to SAHI format with coordinate offset """
    sahi_preds = []
    off_x, off_y = offset
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            
            # Add Offset (Map slice coords -> Full Image coords)
            sahi_preds.append(ObjectPrediction(
                bbox=[x1 + off_x, y1 + off_y, x2 + off_x, y2 + off_y],
                category_id=cls_id,
                category_name=CLASS_NAMES.get(cls_id, str(cls_id)),
                score=conf
            ))
    return sahi_preds

# --- MAIN ---
model = load_model()
cap = cv2.VideoCapture(0)

# Verify Camera Size to set slices correctly
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # e.g., 640
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # e.g., 480

# Define 4 fixed slices (Top-Left, Top-Right, Bot-Left, Bot-Right)
# This is hardcoded for speed, assuming 640x480 input. 
# If your cam is 1920x1080, adjust coords accordingly.
slices = [
    (0, 0, W//2 + 50, H//2 + 50),       # TL (with overlap)
    (W//2 - 50, 0, W, H//2 + 50),       # TR
    (0, H//2 - 50, W//2 + 50, H),       # BL
    (W//2 - 50, H//2 - 50, W, H)        # BR
]

nms = NMSPostprocess(match_threshold=0.5, match_metric="IOS")
prev_time = 0

print("Running Real-Time SAHI (TensorRT)... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    all_preds = []

    # 1. Standard Inference (Full Frame) - Detects big objects
    full_results = model(frame, verbose=False, conf=0.25)
    all_preds.extend(convert_to_sahi(full_results))

    # 2. Sliced Inference (4 Slices) - Detects small objects
    # We manually slice numpy arrays for max speed (faster than sahi library function)
    for (x1, y1, x2, y2) in slices:
        # Crop
        slice_img = frame[y1:y2, x1:x2]
        
        # Infer
        slice_results = model(slice_img, verbose=False, conf=0.15) # Lower conf for small/partial objects
        
        # Convert & Offset
        all_preds.extend(convert_to_sahi(slice_results, offset=(x1, y1)))

    # 3. SAHI NMS (Merge overlaps)
    final_results = nms(all_preds)

    # 4. Visualization
    for pred in final_results:
        box = pred.bbox
        x1, y1, x2, y2 = map(int, box.to_xyxy())
        label = f"{pred.category.name} {pred.score.value:.2f}"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 5. FPS Counter
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
    prev_time = curr_time
    
    cv2.putText(frame, f"SAHI FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    cv2.imshow("TensorRT SAHI", frame)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()