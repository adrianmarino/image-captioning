import glob
import os


def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def min_file_path_from(path, min_func):
    list_of_files = glob.glob(path)
    if len(list_of_files) > 0:
        return min(list_of_files, key=min_func)
    else:
        raise Exception(f'Not found files that match with: {path}.')
