import argparse
import os
import sys
import gc
import numpy as np
from skimage import draw
import matplotlib.pyplot as plt

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
from utils.load_image import ensure_rgb, preprocess_image


def get_lines_mask(cimage, bboxes, char_height):
    height, width = cimage.shape[:2]

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


def process_and_visualize_image(image_file, input_dir, output_dir, model):
    input_path = os.path.join(input_dir, image_file)
    
    # Save both the original image and the mask
    basename = os.path.splitext(image_file)[0]
    extension = os.path.splitext(image_file)[1]

    # Create train and val subdirectories
    train_dir = os.path.join(output_dir, "inputs")
    val_dir = os.path.join(output_dir, "targets")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Load and process image with cleanup
    image, _, bboxes, char_height, resize_factor, height, width = (
        load_process_and_classify_image(input_path, model)
    )

    # Save original image to train directory
    train_path = os.path.join(train_dir, f"{basename}{extension}")
    plt.imsave(train_path, ensure_rgb(image))

    # Get columns and free memory
    column_bboxes = get_columns_bboxs(image, bboxes, char_height)

    # Create full-size mask, handling both 2D and 3D images
    if len(image.shape) == 3:
        full_mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)
    else:
        full_mask = np.zeros_like(image, dtype=np.uint8)

    for idx, column_bbox in enumerate(column_bboxes):
        minr, minc, maxr, maxc = column_bbox["bbox"]

        # Get sub-image and process
        sub_image = image[minr:maxr, minc:maxc].copy()
        clean_column_image = preprocess_image(sub_image)

        bboxes, _, char_height = process_and_classify_image(clean_column_image, model)

        # Get mask and free intermediate results
        mask = get_lines_mask(sub_image, bboxes, char_height)

        if np.any(mask):
            # Place the sub mask in the correct position in the full mask
            full_mask[minr:maxr, minc:maxc] = np.maximum(
                full_mask[minr:maxr, minc:maxc], mask
            )

        del bboxes, clean_column_image, sub_image
        gc.collect()

    

    # Print diagnostic information
    print(f"\nProcessing: {image_file}")
    print(f"  Original size: {width}x{height}")
    print(f"  Resize factor: {resize_factor:.2f}")
    print(f"  Character height: {char_height} pixels")

    

    # Save mask to val directory
    val_path = os.path.join(val_dir, f"{basename}{extension}")
    plt.imsave(val_path, full_mask, cmap="gray")

    del image, full_mask
    gc.collect()


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
        default="data/processed/img-line-masks",
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


if __name__ == "__main__":
    main()
