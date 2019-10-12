import mmap
import os


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def load_from(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content


def lines_from(file_path):
    return load_from(file_path).splitlines()


def blocks(files, size=65536):
    while True:
        b = files.read(size)
        if not b: break
        yield b


def get_num_lines(file_path):
    with open(file_path, "r+") as f:
        return sum(bl.count("\n") for bl in blocks(f))
