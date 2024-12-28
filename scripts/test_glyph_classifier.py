import argparse
import sys
import torch
from skimage import io
import os
import shutil


# Allow inport of data and model modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data.glyph_classes import GLYPH_CLASSES
from data.transform import CustomTransform
from utils.get_image_files import get_image_files
from models.load_model import load_model


def predict(image_paths, root_dir):
    """
    Predict the class of images in a directory.

    Args:
        image_paths (list): List of paths to the images to be predicted.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model()
    transform = CustomTransform()

    # Create output directories
    predicted_dir = os.path.join(root_dir, "predicted")

    for class_name in GLYPH_CLASSES:
        os.makedirs(os.path.join(predicted_dir, class_name), exist_ok=True)

    for image_path in image_paths:
        # Transform image to model inputs
        image = io.imread(image_path)
        image = transform(image).unsqueeze(0).to(device)

        # Run model on image
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        class_label = GLYPH_CLASSES[predicted.item()]

        # Copy image to the corresponding subdirectory
        shutil.copy(image_path, os.path.join(predicted_dir, class_label))


def main():
    """
    Main function to handle command-line arguments and run the appropriate functionality.
    """
    parser = argparse.ArgumentParser(
        description="Test a CNN to classify images into fonts."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="data/processed",
        help="Root directory containing training images.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        help="Maximum number of files to predict.",
    )

    args = parser.parse_args()

    predict_dir = os.path.join(args.root_dir, "glyphs")
    if not os.path.isdir(predict_dir):
        print("The provided predict directory is not valid.")
        return

    image_files = get_image_files(predict_dir, args.max_files)
    image_paths = [os.path.join(predict_dir, f) for f in image_files]

    if not image_paths:
        print("No valid image files found in the directory.")
        return

    predict(image_paths, args.root_dir)


if __name__ == "__main__":
    main()
