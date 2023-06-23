"""opencc

use publicly-available image classification models to predict contents
of video streams.

"""

import cv2
import logging
import numpy as np
from PIL import ImageTk, Image
import sys
import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.applications import imagenet_utils
import threading
import time
import tkinter as tk

logging.getLogger().setLevel(logging.INFO)


class InteruptableLockingThread(threading.Thread):
    def __init__(self, name, sleep_secs=0.1):
        super().__init__()
        self.name = name
        self.sleep_secs = sleep_secs
        self.lock = threading.Lock()
        self.running = True
        self.start()

    def run(self):
        while self.running:
            with self.lock:
                self.work()
                time.sleep(self.sleep_secs)

    def join(self):
        self.running = False
        super().join()


class ModelProvider(InteruptableLockingThread):
    available_models = [x for x in dir(applications) if x[0].isupper()]
    default_model = 'ResNet152'

    def __init__(self, model_name=None):
        self.model = getattr(applications, model_name or self.default_model)()
        self.image_shape = self.model.input_shape[1:3]
        self.frames = []
        super().__init__(name="model provider", sleep_secs=0.01)

    def work(self):
        if len(self.frames) > 0:
            self.predict(self.frames.pop(0))

    def add_frame(self, im):
        with self.lock:
            self.frames.append(im)

    def predict(self, im):
        im = np.expand_dims(im[:, :, :3], axis=0)
        im = tf.image.resize(im, self.image_shape)
        im = imagenet_utils.preprocess_input(im)

        y = self.model.predict(im)
        decoded = imagenet_utils.decode_predictions(y, top=10)
        for (i, (imagenetID, label, prob)) in enumerate(decoded[0]):
            print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))


class Livestream(InteruptableLockingThread):
    def __init__(self, stream_spec):
        self.cap = cv2.VideoCapture(stream_spec)
        self.frame = None
        super().__init__(name="stream", sleep_secs=0)

    def work(self):
        _, self.frame = self.cap.read()


class Window(tk.Tk):
    def __init__(self, livestream_addrs):
        self.title = "Open CC"
        super().__init__()

        self.options = tk.StringVar(self)
        self.options.set(ModelProvider.default_model)
        self.options.trace('w', self.change_model)
        self.model_chooser = tk.OptionMenu(self, self.options,
                                           *ModelProvider.available_models)
        self.model_chooser.pack()
        self.model_provider = ModelProvider()
        self.bind('<KeyPress>', lambda e: self.key_down(e))
        self.bind('<Control-q>', lambda key: self.destroy())

        self.image_labels = [None] * len(livestream_addrs)
        self.livestreams = []
        for livestream_addr in livestream_addrs:
            self.livestreams.append(Livestream(livestream_addr))
        self.after(1, self.update_livestreams)

    def destroy(self):
        logging.info("shutting down")
        for livestream in self.livestreams:
            livestream.join()
        self.model_provider.join()
        super().destroy()

    def change_model(self, *args):
        self.model_provider.join()
        new_model = self.options.get()
        logging.info(f"changing model to {new_model}")
        self.model_provider = ModelProvider(new_model)

    def update_livestreams(self):
        for index, livestream in enumerate(self.livestreams):
            im = livestream.frame
            if im is not None:
                self.show_image(im, index)
        self.after(1, self.update_livestreams)

    def show_image(self, buf, index):
        im = Image.fromarray(buf)
        im.thumbnail((1024, 1024), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(im)

        if self.image_labels[index] is None:
            self.image_labels[index] = tk.Label(self, image=photo)
            self.image_labels[index].pack()

        self.image_labels[index].configure(image=photo)
        self.image_labels[index].img = photo

    def key_down(self, event):
        if event.keysym == 'Return':
            for livestream in self.livestreams:
                self.model_provider.add_frame(livestream.frame)


def main(args):
    if len(args) < 2:
        logging.error(
            f"usage {args[0]} <video stream spec> [<video stream spec> ...]")
        return 1

    win = Window(livestream_addrs=args[1:])
    win.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
