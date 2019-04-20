from pickle import dump, load


def save_obj(path, value):
    with open(path, "wb") as encoded_pickle:
        dump(value, encoded_pickle)


def load_obj(path):
    return load(open(path, "rb"))
