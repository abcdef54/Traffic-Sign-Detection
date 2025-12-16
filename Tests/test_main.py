# src/video_loader.py
import cv2
import threading
import queue
import time

class ThreadedVideoReader:
    def __init__(self, source, queue_size=4):
        self.stream = cv2.VideoCapture(source)
        if not self.stream.isOpened():
            raise Exception(f"Could not open video source: {source}")
            
        self.stopped = False
        self.Q = queue.Queue(maxsize=queue_size)
        
        # Start the thread to read frames
        self.t = threading.Thread(target=self.update, args=())
        self.t.daemon = True # Thread dies when main program dies
        self.t.start()

    def update(self):
        while True:
            if self.stopped:
                return

            if not self.Q.full():
                ret, frame = self.stream.read()
                if not ret:
                    self.stop()
                    return
                
                self.Q.put(frame)
            else:
                # If queue is full, wait a tiny bit to let GPU catch up
                time.sleep(0.001)

    def read(self):
        # Return next frame in the queue
        return self.Q.get()

    def more(self):
        # Returns True if there are still frames in the queue
        return self.Q.qsize() > 0

    def stop(self):
        self.stopped = True
        self.stream.release()


# main.py
import cv2
import time
from ultralytics import YOLO
from src.video_loader import ThreadedVideoReader

# 1. Setup
model_path = 'runs/detect/night_eagle_vietnam2/weights/best.engine'
video_path = 'test_video.mp4'

print("Loading Engine...")
model = YOLO(model_path, task='detect')

print("Starting Video Thread...")
cap = ThreadedVideoReader(video_path)
# Give it a moment to buffer a few frames
time.sleep(1.0) 

print("Starting Inference Loop...")
prev_time = 0

while True:
    if cap.more():
        # 1. Grab frame (Instant - because it was already read in background)
        frame = cap.read()
        
        # 2. Inference (Your ~8.5ms logic)
        results = model(frame, verbose=False)
        
        # 3. FPS Calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        # 4. Visualization
        annotated_frame = results[0].plot()
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Night Eagle View", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        # Video finished
        if cap.stopped: 
            break
        # Buffer empty? Wait a tiny bit
        time.sleep(0.001)

cap.stop()
cv2.destroyAllWindows()