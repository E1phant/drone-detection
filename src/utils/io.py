import json
import os
import subprocess
from os import chdir, getcwd
from os.path import exists
from typing import Any, List, Mapping, Optional



def load_json(path: str) -> Mapping[str, Any]:
    """
    Read .json file and return dict
    """
    with open(path, "r") as read_file:
        loaded_dict = json.load(read_file)
    return loaded_dict


def write_json(
    path,
    data,
    encoding="utf-8",
    ensure_ascii=False,
    indent=4,
    json_encoder=None,
    **kwargs,
):
    with open(path, "w", encoding=encoding) as f:
        result = json.dump(
            data,
            f,
            ensure_ascii=ensure_ascii,
            indent=indent,
            cls=json_encoder,
            **kwargs,
        )
    return result


def dvc_pull(path, cwd=None):
    kwargs = {}
    if cwd is not None:
        kwargs = {"cwd": cwd}
    print(f"DVC pulling {path} ...")
    subprocess.call((f"dvc pull {path} --jobs 32"), shell=True, **kwargs)
    print(f"{path} DVC pulled!")


def dvc_check_and_pull(folder_path, directory=None):
    if folder_path.endswith(".dvc"):
        folder_path = os.path.splitext(folder_path)[0]
    if exists(folder_path):
        return
    previous_dir = getcwd()
    if directory is None:
        directory = previous_dir
    dvc_path = folder_path + ".dvc"
    chdir(directory)
    if not exists(folder_path) and exists(dvc_path):
        dvc_pull(dvc_path)
    chdir(previous_dir)


def list_to_files(input: List[str], filename: str):
    """
    Takes a list of filenames and makes a text file of filenames
    """
    with open(filename, "w") as the_file:
        for idx, f in enumerate(input):
            if idx == len(input) - 1:
                the_file.write(f)
            else:
                the_file.write(f + "\n")


def files_to_list(filename: str, root_path: Optional[str] = None):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding="utf-8") as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    if root_path is not None:
        files = [os.path.join(root_path, el) for el in files]
    return files