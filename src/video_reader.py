import threading
import cv2
import queue

class MultithreadVideoCapture:
    def __init__(self, source, queue_size=1, drop_old_frames=True) -> None:
        self.stream = cv2.VideoCapture(source)
        self.width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)

        if not self.stream.isOpened():
            raise Exception(f"Could not open video source: {source}")
        
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.stopped = False
        self.drop_old_frames = drop_old_frames
        self.q = queue.Queue(maxsize=queue_size)
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True 
        self.thread.start()
    
    def update(self):
        while not self.stopped:
            ret, frame = self.stream.read()
            
            if not ret:
                self.stopped = True
                break
            
            if self.drop_old_frames:
                if self.q.full():
                    try:
                        self.q.get_nowait()
                    except queue.Empty:
                        pass
                self.q.put(frame)
            else:
                self.q.put(frame, block=True)
        
        self.stream.release()
    
    def read(self):
        """Return the next frame or None if stream is over."""
        try:
            return self.q.get(timeout=1.0) 
        except queue.Empty:
            if self.stopped:
                return None
            return None
    
    def more(self):
        return self.q.qsize() > 0
    
    def release(self):
        self.stopped = True
        self.stream.release()
    
    def print_config(self) -> None:
        print(f"Camera Config: Resolution {self.height}x{self.width} - FPS {self.fps} - Queue size: {self.q.maxsize}")