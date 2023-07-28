"""opencc static viewer

Use publicly-available image classification models to predict contents
of static image files.

"""
from functools import partial
import logging
from model_provider import ModelProvider
import numpy as np
from pathlib import Path
from PIL import ImageTk, Image
import sys
from tensorflow.keras.preprocessing import image
import threading
import tkinter as tk

logging.getLogger().setLevel(logging.INFO)


class Window(tk.Tk):
    def __init__(self, filespec):
        self.title = "Open CC"
        super().__init__()

        self.prediction_lock = threading.Lock()

        self.image_label = None
        self.category_label = tk.Label()
        self.category_prediction = None

        self.options = tk.StringVar(self)
        self.options.set(ModelProvider.default_model)
        self.options.trace('w', self.change_model)
        self.model_chooser = tk.OptionMenu(self, self.options,
                                           *ModelProvider.available_models)
        self.model_chooser.pack()
        self.model_provider = ModelProvider()
        self.bind('<Control-q>', lambda key: self.destroy())
        self.bind('<Right>', lambda key: self.right())
        self.bind('<Left>', lambda key: self.left())
        self.bind('<Up>', lambda key: self.up())
        self.bind('<Return>', lambda key: self.predict())

        self.images = []
        def find_images(path):
            if path.is_dir():
                for p in path.iterdir():
                    find_images(p)
            else:
                self.images.append(path)
        find_images(Path(filespec))
        self.cur_image_index = 0
        self.show_image()

        self.after(30, self.maybe_update_category_labels)


    def destroy(self):
        logging.info("shutting down")
        self.model_provider.join()
        super().destroy()

    def up(self):
        self.rotate_image()

    def right(self):
        self.cur_image_index = (self.cur_image_index + 1) % len(self.images)
        self.show_image()

    def left(self):
        self.cur_image_index = (self.cur_image_index - 1) % len(self.images)
        self.show_image()

    def predict(self):
        img = image.load_img(str(self.images[self.cur_image_index]))
        img = image.img_to_array(img)
        self.model_provider.predict(img)

    def change_model(self, *args):
        self.model_provider.join()
        new_model = self.options.get()
        logging.info(f"changing model to {new_model}")
        self.model_provider = ModelProvider(model_name=new_model)

    def show_image(self):
        img = Image.open(self.images[self.cur_image_index])
        img = img.resize((1024, 1024), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        if self.image_label is None:
            self.image_label = tk.Label(self, image=img)
            self.image_label.pack()
            self.category_label = tk.Label(self, text="None yet")
            self.category_label.pack()

        self.image_label.configure(image=img)
        self.image_label.img = img

    def set_category_label(self, index, prediction):
        with self.prediction_lock:
            self.category_prediction = prediction

    def maybe_update_category_labels(self):
        with self.prediction_lock:
            if self.category_prediction and self.category_label:
                text = '\n'.join([
                    f"{round(100 * pred[i][2]):2d}%: {pred[i][1]}"
                    for i in range(5)])
                self.category_label['text'] = text
                self.category_prediction = None
        self.after(30, self.maybe_update_category_labels)


def main(args):
    if len(args) != 2:
        logging.error(
            f"usage {args[0]} <img or dir>")
        return 1

    win = Window(filespec=args[1])
    win.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
