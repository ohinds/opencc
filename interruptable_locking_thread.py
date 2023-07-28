import threading
import time


class InteruptableLockingThread(threading.Thread):
    def __init__(self, name, sleep_ms=100):
        super().__init__()
        self.name = name
        self.sleep_ms = sleep_ms
        self.lock = threading.Lock()
        self.running = True
        self.start()

    def run(self):
        while True:
            with self.lock:
                if not self.running:
                    break
                self.work()

            time.sleep(self.sleep_ms / 1000)

    def join(self):
        with self.lock:
            self.running = False
        super().join()
