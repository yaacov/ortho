import sys
from skimage import draw
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import argparse
import os

# Allow inport of data and model modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models.load_model import load_model
from utils.get_image_files import get_image_files
from utils.load_image import preprocess_image
from utils.process_and_classify_image import (
    load_process_and_classify_image,
    process_and_classify_image,
)
from utils.connect_vertical_gaps import connect_vertical_gaps
from utils.count_bboxes_in_components import count_bboxes_in_components


def draw_columns_rects(cimage, columns_sorted, char_height):
    """
    Rraw rectangles on the image for each column.

    Parameters:
    cimage (ndarray): The input image.
    columns_sorted (list): A list of sorted columns with bounding boxes and counts.
    char_height (int): The character height to increase the rectangle size.

    Returns:
    ndarray: The color float image with the column rectangles.
    """
    height, width = cimage.shape[:2]

    # Create a color map
    colors = plt.colormaps.get_cmap("tab10")

    # Create a figure and axis to overlay rectangles
    fig, ax = plt.subplots()
    ax.imshow(cimage, cmap="gray")

    for idx, column in enumerate(columns_sorted):
        minr, minc, maxr, maxc = column["bbox"]

        # Increase the rectangle size by char_height in all directions
        minr = max(minr - char_height, 0)
        minc = max(minc - char_height, 0)
        maxr = min(maxr + char_height, height)
        maxc = min(maxc + char_height, width)

        # Draw the rectangle
        rect = patches.Rectangle(
            (minc, minr),
            maxc - minc,
            maxr - minr,
            linewidth=2,
            edgecolor=colors(idx),
            facecolor="none",
        )
        ax.add_patch(rect)

        # Add text label
        ax.text(
            minc,
            minr - 5,
            str(idx),
            color=colors(idx),
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5),
        )

    plt.axis("off")
    fig.set_size_inches(cimage.shape[1] / fig.dpi, cimage.shape[0] / fig.dpi)

    # Render the figure to a numpy array
    canvas = FigureCanvas(fig)
    canvas.draw()

    # Convert the canvas to a numpy array
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    float_image = (
        buf.reshape(fig.canvas.get_width_height()[::-1] + (4,)).astype(np.float32)
        / 255.0
    )

    # Convert RGBA to RGB by ignoring the alpha channel
    float_image = float_image[:, :, :3]

    plt.close(fig)  # Close the figure to release resources

    return float_image


def get_columns_bboxs(cimage, glyph_bboxes, char_height):
    """
    Get the columns from the image based on bounding boxes and character height.

    Parameters:
    cimage (ndarray): The input image.
    glyph_bboxes (list): A list of bounding boxes with class labels.
    char_height (int): The character height.

    Returns:
    list: A list of columns with bounding boxes and counts.
    """
    height, width = cimage.shape[:2]

    # Create a blank image of the same size
    blank_image = np.zeros((height, width), dtype=np.uint8)

    # Draw rectangles for each bounding box on the blank image
    for bbox in glyph_bboxes:
        minr, minc, maxr, maxc = bbox["bbox"]
        class_label = bbox["label"]

        # Initialize rr and cc arrays
        rr, cc = np.array([], dtype=int), np.array([], dtype=int)

        # Draw rectangles for all objects except noise and points
        if class_label not in ["70_noise", "71_points"]:
            rr, cc = draw.rectangle(
                start=(minr, minc), extent=(maxr - minr, maxc - minc)
            )

        # Clip to image bounds
        rr = np.clip(rr, 0, height - 1)
        cc = np.clip(cc, 0, width - 1)
        blank_image[rr, cc] = 255

    # Connect vertical gaps in the blank image
    blank_image = connect_vertical_gaps(blank_image, char_height)

    # Count bounding boxes in connected components
    columns = count_bboxes_in_components(blank_image, glyph_bboxes)

    return columns


def overlay_columns(cimage, model):
    """
    Render the image with text paragraphs overlaid.

    Parameters:
    cimage (ndarray): The input image.
    model (object): The model used for processing and classification.

    Returns:
    ndarray: The image with text paragraphs overlaid.
    """
    # Preprocess the input image
    clean_column_image = preprocess_image(cimage)

    # Process and classify the preprocessed image to get bounding boxes and character height
    glyph_bboxes, _, char_height = process_and_classify_image(clean_column_image, model)

    # Get the columns based on bounding boxes and character height
    column_bboxes = get_columns_bboxs(cimage, glyph_bboxes, char_height)

    # Overlay the columns on the input image
    overlay_image = draw_columns_rects(cimage, column_bboxes, char_height)

    return overlay_image


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

    # Print diagnostic information
    print(f"\nProcessing: {image_file}")
    print(f"  Original size: {width}x{height}")
    print(f"  Resize factor: {resize_factor:.2f}")
    print(f"  Character height: {char_height} pixels")

    # Generate the output path
    output_path = os.path.join(output_dir, f"p_{image_file}")

    # Render the text paragraphs on the image
    float_image = overlay_columns(image, model)

    # Save the resulting image
    plt.imsave(output_path, float_image)


# CLI for input and output
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify and visualize bounding boxes in images."
    )
    parser.add_argument(
        "--input_dir",
        default="data/raw",
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
        default="data/processed/img-paragraphs",
        help="Path to save the output images. Defaults to 'data/processed/lines'.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of files to process.",
    )

    args = parser.parse_args()
    input_dir = args.input_dir
    input_file = args.input_file
    output_dir = args.output_dir
    max_files = args.max_files

    model = load_model()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if input_file:
        process_and_visualize_image(input_file, input_dir, output_dir, model)
    else:
        # Get a list of images in the input directory
        selected_files = get_image_files(input_dir, max_files)
        for image_file in selected_files:
            process_and_visualize_image(image_file, input_dir, output_dir, model)
