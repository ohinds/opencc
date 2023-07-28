"""opencc headless static classifier

Use publicly-available image classification models to predict contents
of static image files.

"""
import logging
from model_provider import ModelProvider
from pathlib import Path
import sys
from tensorflow.keras.preprocessing import image

logging.getLogger().setLevel(logging.INFO)


def main(args):
    if len(args) != 2:
        logging.error(
            f"usage {args[0]} <img or dir>")
        return 1

    model_provider = ModelProvider()

    images = []
    def find_images(path):
        if path.is_dir():
            for p in path.iterdir():
                find_images(p)
        else:
            images.append(path)
    find_images(Path(args[1]))

    for image_path in images:
        img = image.load_img(str(image_path))
        img = image.img_to_array(img)
        print(image_path)
        y = model_provider.predict(img)

    model_provider.join()

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
