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
from utils.load_image import preprocess_image
from utils.draw_rectangle_on_image import draw_rectangle_on_image
import gc


def get_lines_mask(cimage, bboxes, char_height):
    image = color.rgb2gray(cimage) if len(cimage.shape) == 3 else cimage
    height, width = image.shape

    # Create a blank image of the same size
    blank_image = np.zeros((height, width), dtype=np.uint8)

    line_height = char_height

    # Draw rectangles for each bounding box on the blank image
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox["bbox"]
        class_label = bbox["label"]
        rect_height = line_height // 3

        # Initialize rr and cc arrays
        rr, cc = np.array([], dtype=int), np.array([], dtype=int)

        # For over-the-line fonts, start 1/3 font size below the top
        if class_label == "12_lamed":
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


def process_and_visualize_image(image_file, input_dir, output_dir, model):
    input_path = os.path.join(input_dir, image_file)

    # Load and process image with cleanup
    image, _, bboxes, char_height, resize_factor, height, width = (
        load_process_and_classify_image(input_path, model)
    )

    # Get columns and free memory
    column_bboxes = get_columns_bboxs(image, bboxes, char_height)

    # Create overlay image and free original
    if image.ndim == 2:  # Grayscale image
        overlay_image = np.stack([image] * 3, axis=-1).astype(np.float32)
    else:  # Color image
        overlay_image = image.astype(np.float32)

    if overlay_image.max() > 1.0:  # Check if normalization is needed
        overlay_image /= 255

    # Initialize master mask
    master_mask = np.zeros_like(overlay_image[..., 0], dtype=bool)

    for idx, column_bbox in enumerate(column_bboxes):
        minr, minc, maxr, maxc = column_bbox["bbox"]

        # Get sub-image and process
        sub_image = overlay_image[minr:maxr, minc:maxc].copy()
        clean_column_image = preprocess_image(sub_image)

        bboxes, _, char_height = process_and_classify_image(clean_column_image, model)

        # Get mask and free intermediate results
        mask = get_lines_mask(sub_image, bboxes, char_height)
        del bboxes, clean_column_image, sub_image
        gc.collect()

        if np.any(mask):
            # Combine masks
            master_mask[
                minr : minr + mask.shape[0], minc : minc + mask.shape[1]
            ] |= mask

            # Draw rectangle on the overlay image
            overlay_image = draw_rectangle_on_image(
                overlay_image.copy(), (minr, minc), (maxr, maxc), color="red"
            )

    # Apply the master mask to the overlay image
    overlay_image = overlay_mask_on_image(
        overlay_image.copy(), master_mask, "yellow", 0.7
    )

    # Save result and cleanup
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
        "--output_dir",
        required=False,
        default="data/processed/img-lines",
        help="Path to save the output images. Defaults to 'data/processed/lines'.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of files to process. Defaults to None.",
    )

    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    max_files = args.max_files

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = load_model()

    # Get a list of images in the input directory
    selected_files = get_image_files(input_dir, max_files)
    for image_file in selected_files:
        process_and_visualize_image(image_file, input_dir, output_dir, model)


if __name__ == "__main__":
    main()
