import glob
import os


def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def min_file_path_from(path, min_func):
    list_of_files = glob.glob(path)
    return min(list_of_files, key=min_func) if len(list_of_files) > 0 else Exception(f'Not found files that match with: {path}.')
