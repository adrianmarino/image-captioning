from IPython.display import display
from PIL import Image


def show_img(path, base_width):
    img = Image.open(path)
    wpercent = (base_width / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((base_width, hsize), Image.ANTIALIAS)
    display(img)


def show_sample(image_path, descriptions):
    show_img(image_path, base_width=550)
    for desc, score in descriptions:
        print(f'Score: {score}, Desc: {desc}')
