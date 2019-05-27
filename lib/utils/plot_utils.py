import pandas as pd
from IPython.display import display
from PIL import Image

pd.set_option('display.max_colwidth', 400)


def show_img(path, image_width=500):
    img = Image.open(path)
    wpercent = (image_width / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((image_width, hsize), Image.ANTIALIAS)
    display(img)


def remove_begin_and_end(text): return text.replace("$", "").replace("#", "").strip()


def show_sample(sample, predicted_descriptions=[], image_width=300):
    show_img(sample[0], image_width=image_width)
    display(
        pd.DataFrame(
            [[remove_begin_and_end(desc)] for desc in sample[1]],
            columns=["Description"]
        )
    )
    if len(predicted_descriptions) > 0:
        print('')
        display(pd.DataFrame(
            [[desc.strip(), f'{score:2.10f}'] for desc, score in predicted_descriptions][:3],
            columns=["Predicted description", "Score"]
        ))


class PredictionInspector:
    def __init__(self, search, image_features, image_width=500):
        self.__search = search
        self.__image_features = image_features
        self.__image_width = image_width

    def __predict(self, image_path): return self.__search.perform(self.__image_features[image_path])

    def inspect(self, sample):
        image_path, descriptions = sample
        show_sample(
            sample,
            predicted_descriptions=self.__predict(image_path),
            image_width=self.__image_width
        )
