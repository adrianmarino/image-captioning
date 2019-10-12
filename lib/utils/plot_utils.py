import pandas as pd
from IPython.display import display, HTML
from PIL import Image
import json
from IPython.display import JSON

from lib.utils.word_utils import remove_pre_post_fix

pd.set_option('display.max_colwidth', 400)


def display_table(table):
    display(HTML(table.to_html(index=False, justify='left')))


def show_img(path, image_width=500):
    img = Image.open(path)
    wpercent = (image_width / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((image_width, hsize), Image.ANTIALIAS)
    display(img)


def show_sample(sample, image_width=300):
    show_img(sample.path, image_width=image_width)
    display_table(
        pd.DataFrame([[remove_pre_post_fix(desc)] for desc in sample.descriptions], columns=["Description"])
    )


def as_json(obj):
    return JSON(json.dumps(obj.__dict__), expanded=True)
