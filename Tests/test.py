import cv2
import time
import numpy as np
from ultralytics import YOLO
import torch

print(torch.cuda.is_available())


# Load Models
# trt_model = YOLO('yolo11s.engine', task='detect')
# py_model = YOLO('models/signs/yolo11s.pt')

# # Manually set names for TRT to avoid warning
# # trt_model.model.names = {0: 'sign'} 

# cap = cv2.VideoCapture(0)
# ret, frame = cap.read() # Read ONCE
# cap.release() # Disconnect camera immediately

# if not ret:
#     print("Camera failed, using black image.")
#     frame = np.zeros((640, 480, 3), dtype=np.uint8)

# def run_uncapped_test(model, name):
#     print(f"--- Testing {name} (Uncapped) ---")
    
#     # Warmup
#     for _ in range(10): model(frame, verbose=False)

#     prev_time = time.time()
#     frame_count = 0
    
#     # Run for 3 seconds
#     start_test = time.time()
#     while time.time() - start_test < 3:
#         # Run Inference
#         results = model(frame, verbose=False)
        
#         # Calculate Live FPS
#         curr_time = time.time()
#         fps = 1 / (curr_time - prev_time)
#         prev_time = curr_time
#         frame_count += 1
        
#         # Visualize (Simulate real app load)
#         annotated_frame = results[0].plot()
#         cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 50), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
#         cv2.imshow("Uncapped Speed Test", annotated_frame)
#         if cv2.waitKey(1) == ord('q'): break
    
#     avg_fps = frame_count / 3
#     print(f"AVERAGE FPS: {int(avg_fps)}")
#     cv2.destroyAllWindows()

# # Run Tests
# run_uncapped_test(py_model, "PyTorch")
# run_uncapped_test(trt_model, "TensorRT")