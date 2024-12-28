import os
import shutil
import random
import argparse


def get_files_in_directory(directory_path):
    """
    Get list of files in directory.

    Args:
        directory_path (str): Path to directory to scan.

    Returns:
        list: List of filenames in directory.
    """
    return [
        f for f in os.listdir(directory_path)
        if os.path.isfile(os.path.join(directory_path, f))
    ]

def copy_random_files(source_dir, target_dir, clean_dir, max_files, clear):
    """
    Copies a random selection of files from subdirectories in the source directory
    to identically named subdirectories in the target directory.

    Args:
        source_dir (str): The path to the source directory containing subdirectories with files.
        target_dir (str): The path to the target directory where subdirectories and files will be copied.
        max_files (int): The maximum number of files to copy from each subdirectory.
        clear (bool): Whether to clear the target directory before copying files.

    Raises:
        FileNotFoundError: If the source directory does not exist.
        ValueError: If max_files is not a positive integer.
    """
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Source directory '{source_dir}' does not exist.")

    if max_files <= 0:
        raise ValueError("max_files must be a positive integer.")

    if clear and os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Iterate through each subdirectory in the source directory
    for sub_dir in os.listdir(source_dir):
        source_sub_dir = os.path.join(source_dir, sub_dir)

        if not os.path.isdir(source_sub_dir):
            continue

        target_sub_dir = os.path.join(target_dir, sub_dir)
        os.makedirs(target_sub_dir, exist_ok=True)

        # Get a list of files in the source subdirectory
        # Get all files from source subdirectory
        source_files = get_files_in_directory(source_sub_dir)

        if clean_dir:
            # Get files already in clean directory
            clean_sub_dir = os.path.join(clean_dir, sub_dir)
            if os.path.exists(clean_sub_dir):
                clean_files = get_files_in_directory(clean_sub_dir)
                # Filter out files that already exist in clean directory
                source_files = [f for f in source_files if f not in clean_files]

        # Randomly select up to max_files files
        selected_files = random.sample(source_files, min(len(source_files), max_files))

        # Copy the selected files to the target subdirectory
        for file in selected_files:
            shutil.copy(
                os.path.join(source_sub_dir, file), os.path.join(target_sub_dir, file)
            )

        print(f"Copied {len(selected_files)} files to {target_sub_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Randomly copy files from subdirectories in the source directory "
        "to identically named subdirectories in the target directory."
    )
    parser.add_argument(
        "source_dir",
        nargs="?",
        default="data/processed/predicted",
        help="The path to the source directory containing subdirectories with files. Defaults to 'data/processed/predicted'.",
    )
    parser.add_argument(
        "--target_dir",
        default=None,
        help="The path to the target directory where files will be copied. Defaults to <source>_copy.",
    )
    parser.add_argument(
        "--clean_dir",
        default=None,
        help="The path to a clean directory to check existing files. Defatuls to None.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=75,
        help="The maximum number of files to randomly copy from each subdirectory. Defaults to 75.",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="If set, the target directory will be cleared before copying files. Defaults to False.",
    )

    args = parser.parse_args()

    target_dir = args.target_dir or f"{args.source_dir}_copy"

    copy_random_files(
        args.source_dir, target_dir, args.clean_dir, args.max_files, args.clear
    )
