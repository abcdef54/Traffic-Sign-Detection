import cv2
import time
from ultralytics import YOLO
import math

# 1. Load BOTH models
# Brain A: Standard PyTorch model for People
print("Loading Brain A (Pedestrians)...")
coco_model = YOLO('models/pedestrians/yolov8n.pt') 

# Brain B: ONNX model for Signs
print("Loading Brain B (Signs - ONNX)...")
sign_model = YOLO('yolo11s.onnx', task='detect') 

cap = cv2.VideoCapture(0)

# --- VIDEO RECORDING SETUP ---
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_out = 30.0 
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter('output_fps_test.mp4', fourcc, fps_out, (frame_width, frame_height))
# -----------------------------

# --- FPS VARS ---
prev_frame_time = 0
new_frame_time = 0

print("System Active. Recording... Press 'q' to stop.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # 2. Run Inference
    # Brain 1 (People)
    coco_results = coco_model(frame, stream=True, classes=[0], verbose=False)
    
    # Brain 2 (Signs - ONNX)
    # Looking for Traffic Light (9) and Stop Sign (11) for testing
    for _ in range(4):
        sign_results = sign_model(frame, stream=True, verbose=False, classes=[9, 11])

    # 3. Draw Results
    # -- People --
    for r in coco_results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # -- Signs (ONNX) --
    for r in sign_results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            # Handle class names safely for ONNX
            if hasattr(sign_model, 'names'):
                label = sign_model.names[cls_id]
            else:
                label = str(cls_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # --- CALCULATE FPS ---
    new_frame_time = time.time()
    
    # FPS = 1 / (time taken to process this frame)
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    
    # Convert to integer
    fps = int(fps)
    
    # Draw FPS on screen (Top Left, Yellow)
    cv2.putText(frame, f"FPS: {fps}", (7, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    # ---------------------

    out.write(frame)
    cv2.imshow("Night Eagle Speed Test", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Test finished. Last FPS reading: {fps}")