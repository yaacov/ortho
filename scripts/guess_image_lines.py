import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, draw
from skimage.draw import rectangle_perimeter
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
from utils.draw_text_on_image import draw_text_on_image
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


def draw_line_boxes(overlay_image, mask, column_offset, color="red"):
    """Draw boxes and indices for each line in mask."""
    # Check if overlay_image is too small
    if overlay_image.shape[0] < 300 or overlay_image.shape[1] < 300:
        return result
    
    # Create new image 
    result = overlay_image.copy()
    del overlay_image
    gc.collect()

    # Get connected components
    labeled = label(mask)
    regions = regionprops(labeled)
    del labeled
    gc.collect()

    # Sort regions by vertical position
    sorted_regions = sorted(regions, key=lambda r: r.bbox[0])
    del regions
    gc.collect()

    # Draw box and index for each region
    minr_offset, minc_offset = column_offset
    for idx, region in enumerate(sorted_regions):
        minr, minc, maxr, maxc = region.bbox

        # Draw index with temp image
        text_x = maxc + minc_offset + 4
        text_y = (minr + maxr) // 2 + minr_offset + 24
        
        temp = draw_text_on_image(
            result, str(idx), text_x, text_y, color=color, fontsize=24
        )
        del result
        result = temp
        gc.collect()

    del sorted_regions
    gc.collect()

    return result


def process_and_visualize_image(image_file, input_dir, output_dir, model):
    input_path = os.path.join(input_dir, image_file)

    # Load and process image with cleanup
    image, _, bboxes, char_height, resize_factor, height, width = (
        load_process_and_classify_image(input_path, model)
    )

    # Get columns and free memory
    column_bboxes = get_columns_bboxs(image, bboxes, char_height)

    # Create overlay image and free original
    overlay_image = (
        np.stack([image] * 3, axis=-1) if image.ndim == 2 else image.copy()
    ).astype(np.float32) / 255

    # Process each column
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
            # Direct function calls instead of lambdas
            overlay_image = overlay_mask_on_image(
                overlay_image.copy(), mask, "yellow", 0.5, minr, minc
            )

            overlay_image = draw_rectangle_on_image(
                overlay_image.copy(), (minr, minc), (maxr, maxc), color="red"
            )

            """ overlay_image = draw_text_on_image(
                overlay_image.copy(),
                str(idx),
                maxc - 36,
                minr + 36,
                color="yellow",
                fontsize=24,
            ) """

            """ overlay_image = draw_line_boxes(
                overlay_image.copy(), mask, (minr, minc), color="blue"
            ) """

    # Save result and cleanup
    with plt.ioff():
        plt.imsave(os.path.join(output_dir, f"processed_{image_file}"), overlay_image)
        plt.close("all")


def main():
    parser = argparse.ArgumentParser(description="Process images and overlay lines.")
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to the input directory containing images.",
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
