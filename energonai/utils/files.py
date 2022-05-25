import os


def ensure_directory_exists(path: str):
    # ensure the directory exists
    # dir = os.path.dirname(path)
    if not os.path.exists(path):
        os.makedirs(path)