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


def mkdir(path):
    """Creates a folder if it does not already exist,
    returns True if the folder was created, else False."""
    try:
        created = False
        if not os.path.exists(path):
            os.makedirs(path)
            created = True
        return created
    except OSError:
        print('Error: Creating directory. ' + path)
