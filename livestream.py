import cv2
from interruptable_locking_thread import InteruptableLockingThread


class LiveStream(InteruptableLockingThread):
    def __init__(self, stream_spec):
        self.cap = cv2.VideoCapture(stream_spec)
        self.frame = None
        super().__init__(name="stream", sleep_ms=10)

    def work(self):
        def crop_to_square(cap):
            im = cap[1]
            if im is None:
                return None
            shape = im.shape
            half_crop = (shape[1] - shape[0]) // 2
            return im[:, half_crop:half_crop+shape[0], :]

        self.frame = crop_to_square(self.cap.read())

    def get_frame(self):
        with self.lock:
            return self.frame
