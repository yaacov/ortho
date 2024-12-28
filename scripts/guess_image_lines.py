import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, draw
from skimage.measure import label, regionprops

# Allow import of data and model modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data.glyph_classes import categorize_glyph_class
from models.load_model import load_model
from utils.get_image_files import get_image_files
from utils.process_and_classify_image import (
    load_process_and_classify_image,
    process_and_classify_image,
)
from utils.connect_horizontal_gaps import connect_horizontal_gaps
from guess_image_paragraphs import get_columns_bboxs
from utils.overlay_mask_on_image import overlay_mask_on_image
from utils.load_image import convert_to_grayscale, preprocess_image
from utils.draw_rectangle_on_image import draw_rectangle_on_image
import gc


def get_lines_mask(cimage, bboxes, char_height):
    image = convert_to_grayscale(cimage)
    height, width = image.shape

    # Create a blank image of the same size
    blank_image = np.zeros((height, width), dtype=np.uint8)

    line_height = char_height

    # Draw rectangles for each bounding box on the blank image
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox["bbox"]
        class_label = bbox["label"]
        rect_height = line_height // 3

        # Check heght of the bounding box
        if maxr - minr < rect_height // 2:
            continue

        # Initialize rr and cc arrays
        rr, cc = np.array([], dtype=int), np.array([], dtype=int)

        # For over-the-line fonts, start 1/3 font size below the top
        if class_label in ["12_lamed", "32_fused_lamed"]:
            top = minr + char_height // 3
            rr, cc = draw.rectangle(
                start=(top, minc), extent=(rect_height, maxc - minc)
            )

        # For yud, make a bigger square
        elif class_label == "10_yud":
            rr, cc = draw.rectangle(
                start=(minr, minc), extent=(rect_height, maxc - minc)
            )

        # Regular fonts
        elif (
            categorize_glyph_class(class_label) == 0
            or categorize_glyph_class(class_label) == 2
            or class_label == "31_fused"
        ):
            rr, cc = draw.rectangle(
                start=(minr, minc), extent=(rect_height, maxc - minc)
            )

        # Clip to image bounds
        rr = np.clip(rr, 0, height - 1)
        cc = np.clip(cc, 0, width - 1)
        blank_image[rr, cc] = 255

    # Apply a kernel to connect elements with horizontal gaps
    blank_image = connect_horizontal_gaps(blank_image, char_height)

    return blank_image


def process_columns(image, bboxes, char_height):
    """Get columns and prepare overlay image."""
    # Get columns
    column_bboxes = get_columns_bboxs(image, bboxes, char_height)

    # Create overlay image
    if image.ndim == 2:
        overlay_image = np.stack([image] * 3, axis=-1).astype(np.float32)
    else:
        overlay_image = image.astype(np.float32)

    if overlay_image.max() > 1.0:
        overlay_image /= 255

    return column_bboxes, overlay_image


def process_column_lines(cimage, column_bbox, model):
    """Process single column to find lines."""
    minr, minc, maxr, maxc = column_bbox["bbox"]

    # Get and process sub-image
    sub_image = cimage[minr:maxr, minc:maxc].copy()
    clean_column_image = preprocess_image(sub_image)

    # Find lines in column
    bboxes, _, char_height = process_and_classify_image(clean_column_image, model)
    mask = get_lines_mask(sub_image, bboxes, char_height)

    # Cleanup
    del bboxes, clean_column_image, sub_image
    gc.collect()

    return mask if np.any(mask) else None


def process_and_visualize_image(
    image_file, input_dir, output_dir, model, one_column=False
):
    input_path = os.path.join(input_dir, image_file)

    # Load and process image
    image, _, bboxes, char_height, resize_factor, height, width = (
        load_process_and_classify_image(input_path, model)
    )

    if one_column:
        # Create single column bbox for full image
        column_bboxes = [{"bbox": (0, 0, height, width)}]
        overlay_image = process_columns(image, bboxes, char_height)[1]
    else:
        # Get columns and prepare overlay
        column_bboxes, overlay_image = process_columns(image, bboxes, char_height)

    # Process each column
    for idx, column_bbox in enumerate(column_bboxes):
        mask = process_column_lines(overlay_image, column_bbox, model)

        if mask is not None:
            minr, minc, maxr, maxc = column_bbox["bbox"]

            # Apply mask and draw rectangle
            overlay_image = overlay_mask_on_image(
                overlay_image.copy(), mask, "yellow", 0.7, minr, minc
            )
            overlay_image = draw_rectangle_on_image(
                overlay_image.copy(), (minr, minc), (maxr, maxc), color="red"
            )

    # Save result
    with plt.ioff():
        plt.imsave(os.path.join(output_dir, f"l_{image_file}"), overlay_image)
        plt.close("all")


def main():
    parser = argparse.ArgumentParser(description="Process images and overlay lines.")
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
        default="data/processed/img-lines",
        help="Path to save the output images. Defaults to 'data/processed/lines'.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of files to process.",
    )
    parser.add_argument(
        "--one_column",
        action="store_true",
        help="Process image as single column.",
    )

    args = parser.parse_args()
    input_dir = args.input_dir
    input_file = args.input_file
    output_dir = args.output_dir
    max_files = args.max_files
    one_column = args.one_column

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = load_model()

    if input_file:
        process_and_visualize_image(
            input_file, input_dir, output_dir, model, one_column
        )
    else:
        # Get a list of images in the input directory
        selected_files = get_image_files(input_dir, max_files)
        for image_file in selected_files:
            process_and_visualize_image(
                image_file, input_dir, output_dir, model, one_column
            )


if __name__ == "__main__":
    main()
