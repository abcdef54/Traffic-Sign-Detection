import threading
import queue
import cv2

class ThreadedVideoWriter:
    def __init__(self, path, fourcc, fps, frame_size, queue_size=30):
        self.path = path
        self.writer = cv2.VideoWriter(path, fourcc, fps, frame_size)
        self.queue = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def write(self, frame):
        if self.stopped: return
        
        if not self.queue.full():
            self.queue.put(frame.copy()) 

    def _worker(self):
        while not self.stopped or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=1)
                self.writer.write(frame)
                self.queue.task_done()
            except queue.Empty:
                continue

    def release(self):
        self.stopped = True
        self.thread.join()
        self.writer.release()
        print("[INFO] Video Writer Released.")