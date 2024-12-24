import sys
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import argparse
import os


# Allow inport of data and model modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data.glyph_classes import GLYPH_CLASSES, categorize_glyph_class
from models.load_model import load_model
from utils.get_image_files import get_image_files
from utils.load_image import preprocess_image
from utils.process_and_classify_image import (
    load_process_and_classify_image,
    process_and_classify_image,
)


def overlay_glyphs_with_rects(image, results):
    height, width = image.shape[:2]

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.imshow(image, cmap="gray")

    # Define colors for each class
    colors = plt.colormaps["tab10"]

    for result in results:
        minr, minc, maxr, maxc = result["bbox"]
        class_label = result["label"]
        class_idx = categorize_glyph_class(class_label)

        # Add rectangle for bounding box
        rect = patches.Rectangle(
            (minc, minr),
            maxc - minc,
            maxr - minr,
            linewidth=2,
            edgecolor=colors(class_idx),
            facecolor="none",
        )
        ax.add_patch(rect)

        # Add text label
        if class_label != "71_points":
            ax.text(
                minc,
                minr - 5,
                class_label[3:],
                color=colors(class_idx),
                fontsize=6,
                bbox=dict(facecolor="white", alpha=0.5),
            )

    plt.axis("off")
    fig.set_size_inches(width / fig.dpi, height / fig.dpi)

    # Render the figure to a numpy array
    canvas = FigureCanvas(fig)
    canvas.draw()

    # Ensure the canvas size matches the figure size
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    float_image = (
        buf.reshape(fig.canvas.get_width_height()[::-1] + (4,)).astype(np.float32)
        / 255.0
    )

    # Convert RGBA to RGB by ignoring the alpha channel
    float_image = float_image[:, :, :3]

    plt.close(fig)  # Close the figure to release resources

    return float_image


def process_and_visualize_image(image_file, input_dir, output_dir, model):
    """
    Process and visualize an image file.

    Parameters:
    image_file (str): The name of the image file to process.
    input_dir (str): The directory containing the input image file.
    output_dir (str): The directory to save the output image file.
    model (object): The model used for processing and classification.

    Returns:
    None
    """
    input_path = os.path.join(input_dir, image_file)

    # Load, process, and classify the image
    image, _, bboxes, char_height, resize_factor, height, width = (
        load_process_and_classify_image(input_path, model)
    )

    print(
        f"Processing {image_file} [{height}, {width}] resize is {resize_factor} \nchar height is {char_height} found {len(bboxes)} fonts."
    )

    # Prepare a black and white threshold clean sub-image
    clean_image = preprocess_image(image)
    output_path = os.path.join(output_dir, f"g_{image_file}")

    # Load and preprocess the image column
    bboxes, _, char_height = process_and_classify_image(clean_image, model)

    # Overlay glyphs with rectangles or render text lines based on the label argument
    float_image = overlay_glyphs_with_rects(image, bboxes)

    # Save the resulting image
    plt.imsave(output_path, float_image)


# CLI for input and output
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify and visualize bounding boxes in images."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to the input directory containing images.",
    )
    parser.add_argument(
        "--input_file",
        required=False,
        help="Path to a single input image file.",
    )
    parser.add_argument(
        "--output_dir",
        required=False,
        default="data/processed/lines",
        help="Path to save the output images. Defaults to 'data/processed/lines'.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=80,
        help="Maximum number of files to process. Defaults to 80.",
    )

    args = parser.parse_args()
    input_dir = args.input_dir
    input_file = args.input_file
    output_dir = args.output_dir
    max_files = args.max_files

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = load_model()

    if input_file:
        process_and_visualize_image(input_file, input_dir, output_dir, model)
    else:
        # Get a list of images in the input directory
        selected_files = get_image_files(input_dir, max_files)
        for image_file in selected_files:
            process_and_visualize_image(image_file, input_dir, output_dir, model)
