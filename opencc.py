"""opencc

Use publicly-available image classification models to predict contents
of video streams.

"""
from functools import partial
from livestream import LiveStream
import logging
from model_provider import ModelProvider
from PIL import ImageTk, Image
import sys
import threading
import tkinter as tk

logging.getLogger().setLevel(logging.INFO)


class Window(tk.Tk):
    def __init__(self, livestream_addrs):
        self.title = "Open CC"
        super().__init__()

        self.prediction_lock = threading.Lock()

        self.image_labels = [None] * len(livestream_addrs)
        self.category_labels = [None] * len(livestream_addrs)
        self.category_predictions = [None] * len(livestream_addrs)
        for col in range(len(livestream_addrs)):
            self.columnconfigure(col, weight=1)

        self.livestreams = []
        for livestream_addr in livestream_addrs:
            self.livestreams.append(LiveStream(livestream_addr))

        self.options = tk.StringVar(self)
        self.options.set(ModelProvider.default_model)
        self.options.trace('w', self.change_model)
        self.model_chooser = tk.OptionMenu(self, self.options,
                                           *ModelProvider.available_models)
        self.model_chooser.grid(row=0, column=0)
        self.model_provider = ModelProvider()
        self.bind('<Control-q>', lambda key: self.destroy())

        self.after(30, self.update_livestreams)
        self.after(300, self.maybe_dispatch_frames)
        self.after(30, self.maybe_update_category_labels)

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
        self.model_provider = ModelProvider(model_name=new_model)

    def update_livestreams(self):
        for index, livestream in enumerate(self.livestreams):
            im = livestream.get_frame()
            if im is not None:
                self.show_image(im, index)
        self.after(30, self.update_livestreams)

    def show_image(self, buf, index):
        im = Image.fromarray(buf)
        im.thumbnail((1024, 1024), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(im)

        if self.image_labels[index] is None:
            self.image_labels[index] = tk.Label(self, image=photo)
            self.image_labels[index].grid(row=1, column=index)
            self.category_labels[index] = tk.Label(self, text="None yet")
            self.category_labels[index].grid(row=2, column=index)

        self.image_labels[index].configure(image=photo)
        self.image_labels[index].img = photo

    def set_category_label(self, index, prediction):
        with self.prediction_lock:
            self.category_predictions[index] = prediction

    def maybe_update_category_labels(self):
        with self.prediction_lock:
            for index, pred in enumerate(self.category_predictions):
                if pred and self.category_labels[index]:
                    text = '\n'.join([
                        f"{round(100 * pred[i][2]):2d}%: {pred[i][1]}"
                        for i in range(5)]
                    )
                    self.category_labels[index]['text'] = text

                    self.category_predictions[index] = None
        self.after(30, self.maybe_update_category_labels)

    def maybe_dispatch_frames(self):
        if self.model_provider.empty():
            for index, livestream in enumerate(self.livestreams):
                self.model_provider.add_frame(
                    livestream.frame, partial(self.set_category_label, index))
        self.after(300, self.maybe_dispatch_frames)


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
