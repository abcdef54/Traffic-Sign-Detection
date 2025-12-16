import cv2
import time  # <--- IMPORTED
from ultralytics import YOLO

# 1. Load BOTH models
# Note: Ensure these paths match your actual folder structure!
coco_model = YOLO('models/pedestrians/yolov8n.pt')      
sign_model = YOLO('models/signs/yolo11s.pt')

cap = cv2.VideoCapture(0)

# --- VIDEO RECORDING SETUP ---
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_out = 30.0 
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter('output_test_pytorch.mp4', fourcc, fps_out, (frame_width, frame_height))
# -----------------------------

# --- FPS VARIABLES ---
prev_frame_time = 0
new_frame_time = 0

print("Recording started... Press 'q' to stop and save.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # 2. Run Inference TWICE (Standard PyTorch)
    # Brain 1: Look for People (Class 0)
    coco_results = coco_model(frame, stream=True, classes=[0], verbose=False)
    
    # Brain 2: Look for Signs (Class 9 & 11 for test)
    start_infer = time.time()
    sign_results = list(sign_model(frame, stream=True)) # Force execution
    end_infer = time.time()

    ms = (end_infer - start_infer) * 1000
    print(f"Inference Time: {ms:.2f} ms")

    # 3. Combine Results
    # -- Draw People --
    for r in coco_results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # -- Draw Signs --
    for r in sign_results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = sign_model.names[cls_id] 
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # --- CALCULATE FPS ---
    new_frame_time = time.time()
    
    # Avoid division by zero on the very first frame
    if new_frame_time - prev_frame_time > 0:
        fps = 1 / (new_frame_time - prev_frame_time)
    else:
        fps = 0
        
    prev_frame_time = new_frame_time
    fps = int(fps)
    
    # Draw FPS in Red (to differentiate from ONNX Yellow)
    cv2.putText(frame, f"FPS (PyTorch): {fps}", (7, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    # ---------------------

    # --- WRITE FRAME TO FILE ---
    out.write(frame)
    
    cv2.imshow("Dual AI System (PyTorch Baseline)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# --- CLEANUP ---
cap.release()
out.release()
cv2.destroyAllWindows()
print("Video saved as 'output_test_pytorch.mp4'")