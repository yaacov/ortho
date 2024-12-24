import os
import random


def get_image_files(input_dir: str, max_files=None):
    """
    Retrieve a list of image files from the specified directory.

    Args:
        input_dir (str): Path to the directory containing image files.
        max_files (int, optional): Maximum number of files to retrieve.
            If None, all image files are returned. Defaults to None.

    Returns:
        list: A list of filenames (str) of image files in the directory.
            If `max_files` is specified, the list contains a random selection
            of up to `max_files` image files.
    """
    # Get a list of all images in the input directory
    image_files = [
        f
        for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
        and os.path.isfile(os.path.join(input_dir, f))
    ]

    if max_files is not None:
        selected_files = random.sample(image_files, min(max_files, len(image_files)))
    else:
        selected_files = image_files

    return selected_files
