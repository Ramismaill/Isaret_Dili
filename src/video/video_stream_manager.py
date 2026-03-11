import cv2
import threading
from collections import deque


class VideoStreamManager:
    """
    Video akışını kamera veya dosyadan alan,
    kareleri okuyan ve tampon belleği yöneten sınıf.
    """

    def __init__(self, source=0, buffer_size=30, width=640, height=480):
        self.source = source
        self.buffer_size = buffer_size
        self.width = width
        self.height = height
        self.buffer = deque(maxlen=buffer_size)
        self.cap = None
        self.is_running = False
        self._thread = None

    def start(self):
        self.cap = cv2.VideoCapture(self.source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self.cap.isOpened():
            raise RuntimeError(f"Kamera/video açılamadı: {self.source}")
        self.is_running = True
        self._thread = threading.Thread(target=self._read_frames, daemon=True)
        self._thread.start()
        print(f"[VideoStreamManager] Başlatıldı → kaynak: {self.source}")

    def _read_frames(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                self.buffer.append(frame)

    def get_frame(self):
        if self.buffer:
            return self.buffer[-1]
        return None

    def stop(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        print("[VideoStreamManager] Durduruldu.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()