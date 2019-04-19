import os


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def load_text_from(file_path):
    content = ''
    with open(file_path, 'r') as file:
        content = file.read()
    return content

