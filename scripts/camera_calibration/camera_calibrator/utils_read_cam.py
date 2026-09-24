import time, threading
import cv2


def open_camera(camera_index=2, width=1280, height=800, fps=120):
    cap = cv2.VideoCapture(camera_index)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open webcam.")
    return cap



class ThreadedCamera:
    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.latest = None
        self.latest_ts = 0.0
        self.frame_id = 0
        self.running = False
        self.t = None

    def start(self, warmup_timeout=2.0):
        """Start the reader thread and block until the first frame is ready or timeout."""
        if self.running:
            return self

        self.running = True
        self.t = threading.Thread(target=self._reader, daemon=True)
        self.t.start()

        # Wait for the first frame
        start_time = time.perf_counter()
        while self.latest is None:
            if time.perf_counter() - start_time > warmup_timeout:
                raise RuntimeError("Camera warm-up timeout: no frame received.")
            time.sleep(0.005)

        return self

    def _reader(self):
        for _ in range(60): self.cap.read()
        
        while self.running:
            if not self.cap.grab():
                time.sleep(0.001)
                continue
            ok, frame = self.cap.retrieve()
            if not ok:
                time.sleep(0.001)
                continue
            ts = time.perf_counter()

            with self.lock:
                self.latest = frame
                self.latest_ts = ts
                self.frame_id += 1   # increment ONLY when a NEW frame is retrieved

    def read(self):
        with self.lock:
            if self.latest is None:
                #print("Warning: No frame available yet.")
                return False, None, 0.0, 0
            return True, self.latest, self.latest_ts, self.frame_id
        

    def stop(self):
        self.running = False
        if self.t is not None:
            self.t.join(timeout=1.0)
        self.cap.release()