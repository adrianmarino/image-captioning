from IPython.display import display
from PIL import Image
import pandas as pd

pd.set_option('display.max_colwidth', 400)

def show_img(path, base_width):
    img = Image.open(path)
    wpercent = (base_width / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((base_width, hsize), Image.ANTIALIAS)
    display(img)


class PredictionInspector:
    def __init__(self, search, image_features):
        self.__search = search
        self.__image_features = image_features

    def __predict(self, image_path):
        return self.__search.perform(self.__image_features[image_path])

    def inspect(self, sample):
        image_path, descriptions = sample

        predicted_descriptions = self.__predict(image_path)

        show_img(image_path, base_width=500)
        display(pd.DataFrame(
            [[desc.replace("$", "").replace("#","").strip()] for desc in descriptions],
            columns=["Original Description"]
        ))
        print('')
        display(pd.DataFrame(
            [[desc.strip(), f'{score:2.10f}'] for desc, score in predicted_descriptions],
            columns=["Predicted description", "Score"]
        ))
