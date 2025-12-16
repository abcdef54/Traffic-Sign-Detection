import threading
import cv2
import queue

class MultithreadVideoCapture:
    def __init__(self, source, queue_size=1) -> None:
        self.stream = cv2.VideoCapture(source)
        self.width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)

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
            ret, frame = self.stream.read()
            
            if not ret:
                self.stop()
                return

            # If queue is full, remove the old frame to make room for the new one
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass

            self.q.put(frame)
    
    def read(self):
        return self.q.get()
    
    def more(self):
        return self.q.qsize() > 0
    
    def stop(self):
        self.stopped = True
        self.stream.release()
    
    def print_config(self) -> None:
        print(f"Camera Config: Resolution {self.height}x{self.width} - FPS {self.fps} - Queue size: {self.q.maxsize}")