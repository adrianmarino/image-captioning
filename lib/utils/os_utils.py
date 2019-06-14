import glob
import os


def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def min_file_path_from(path, min_func):
    list_of_files = glob.glob(path)
    latest_file = min(list_of_files, key=min_func)
    return latest_file