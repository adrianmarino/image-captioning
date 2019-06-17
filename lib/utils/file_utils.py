import mmap
import os


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def load_text_from(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines
