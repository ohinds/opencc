from interruptable_locking_thread import InteruptableLockingThread
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import applications


class ModelProvider(InteruptableLockingThread):
    available_models = ['cvd-finetnued', 'cvd-resnet152-finetuned'] + [
        x for x in dir(applications) if x[0].isupper()]
    default_model = 'ResNet152'

    def __init__(self, model_name=None):
        try:
            self.model = getattr(applications,
                                 model_name or self.default_model)()
        except:
            model_path = os.path.join(
                os.getenv('HOME'), '.keras', 'models', model_name + '.h5')
            self.model = tf.keras.models.load_model(model_path)
        self.image_shape = self.model.input_shape[1:3]
        self.frames = []
        super().__init__(name="model provider", sleep_ms=1000)

    def work(self):
        if len(self.frames) > 0:
            im, callback = self.frames.pop(0)
            self.predict(im, callback)

    def empty(self):
        no_frames = False
        with self.lock:
            no_frames = len(self.frames) == 0
        return no_frames

    def add_frame(self, im, callback):
        with self.lock:
            self.frames.append((im, callback))

    def predict(self, im, callback=None):
        im = np.expand_dims(im[:, :, :3], axis=0)
        im = tf.image.resize(im, self.image_shape)
        im = applications.resnet50.preprocess_input(im)

        y = self.model.predict(im, verbose=0)
        print(y)

        try:
            decoded = applications.imagenet_utils.decode_predictions(y, top=10)
            if callback:
                callback(decoded[0])
            else:
                for (i, (imagenetID, label, prob)) in enumerate(decoded[0]):
                    print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))
        except:
            pass
