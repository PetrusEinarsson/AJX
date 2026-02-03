""" This file contains extra functions for saving files with unique names. """

import sys
import os
from pathlib import Path
import pickle


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError(f"invalid default answer: {default}")

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")


def uniquify(path):
    """Make a filepath unque by adding a number"""
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path


def create_path(path):
    """Create the directories required to make the path exist"""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_safe_unique(filepath, data, description="Data", force_unsafe=False):
    """Dump data to the given filepath but ensure the filepath is unique by
    quering yes/no questions."""
    folder_path, _ = os.path.split(filepath)
    if not os.path.isdir(folder_path):
        create_folder = True
        if not force_unsafe:
            create_folder = query_yes_no(
                f"Path {folder_path} does not exist. Create folders?"
            )
        if create_folder:
            create_path(folder_path)
        else:
            raise OSError("Could not create folder!")

    overwrite = True
    save = True
    if os.path.isfile(filepath):
        if force_unsafe:
            overwrite = True
        else:
            overwrite = query_yes_no(
                f"{filepath} already exists. Overwrite previous file?"
            )

    unique_fname = filepath
    if not overwrite:
        unique_fname = uniquify(filepath)
        save = not query_yes_no(f"Throw away data?")

    if save:
        with open(unique_fname, "wb") as file:
            pickle.dump(data, file)
        print(f"{description} was saved to {unique_fname}")
    else:
        print(f"{description} was not saved")
    return unique_fname
