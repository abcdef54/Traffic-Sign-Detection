import threading
import cv2
import queue
import time

class MultithreadVideoCapture:
    def __init__(self, source, queue_size=1) -> None:
        # self.stream = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        self.stream = cv2.VideoCapture(source)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not self.stream.isOpened():
            raise Exception(f"Could not open video source: {source}")
        
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.stopped = False
        self.q = queue.Queue(maxsize=queue_size)
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True # Thread dies when main program dies
        self.thread.start()
    
    def update(self):
        while not self.stopped:
            # 1. Always read the frame to clear the camera buffer
            ret, frame = self.stream.read()
            
            if not ret:
                self.stop()
                return

            # 2. If queue is full, remove the old frame to make room for the new one
            if not self.q.empty():
                try:
                    self.q.get_nowait() # Discard the old frame
                except queue.Empty:
                    pass

            # 3. Add the new frame
            self.q.put(frame)
    
    def read(self):
        return self.q.get()
    
    def more(self):
        # Returns True if there are still frames in the queue
        return self.q.qsize() > 0
    
    def stop(self):
        self.stopped = True
        self.stream.release()