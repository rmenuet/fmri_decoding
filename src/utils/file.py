import os
from glob import glob


def get_json_files_in_dir(path):
    """
    List all files in directory with json extension

    :param path: str
    :return: list of str paths
    """
    if os.path.isdir(path):
        configurations = glob(path + "*.json")
    elif path[-5:] == ".json":
        configurations = [path]
    else:
        raise ValueError("Please enter a configuration file "
                         "or a folder containing configurations")

    return configurations
