import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, draw
from skimage.measure import label, regionprops
from scipy.ndimage import zoom

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


def extract_char_lines(image, mask, char_height, column_index, output_dir, image_file):
    from skimage.measure import label, regionprops
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Convert to float32 and normalize if needed
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    if image.max() > 1.0:
        image /= 255.0
    
    # Ensure image has 3 color channels
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)

    base_name, ext = os.path.splitext(image_file)
    height_constant = 2.3
    height = int(height_constant * char_height)
    offset = char_height // 4

    labeled_mask = label(mask)
    regions = regionprops(labeled_mask)
    sorted_regions = sorted(regions, key=lambda x: x.bbox[0])
    
    for idx, region in enumerate(sorted_regions):
        minr, minc, maxr, maxc = region.bbox
        width = maxc - minc
        region_label = region.label
        
        # Create line image
        line_img = np.zeros((height, width, 3), dtype=np.float32)
        
        # Process each column
        for col in range(width):
            # Get labeled mask column and find first pixel matching region label
            col_mask = labeled_mask[minr:maxr, minc + col]
            top_pixel = next((i for i in range(len(col_mask)) if col_mask[i] == region_label), 0)
            
            top = minr + top_pixel - offset
            src_col = image[top:top + height, minc + col]
            line_img[:len(src_col), col] = src_col
        
        # Build output filename anc construct full path
        out_filename = f"l_{base_name}-{ext[1:]}_{column_index:03d}_{idx:03d}.png"
        out_path = os.path.join(output_dir, out_filename)

        # Calculate resize ratio
        target_height = int(32 * height_constant)
        ratio = target_height / line_img.shape[0]
        
        # Resize image with cubic interpolation and anti-aliasing
        resized_img = zoom(line_img, (ratio, ratio, 1), order=3, prefilter=True)
        resized_img = np.clip(resized_img, 0, 1)

        # Save resized image
        plt.imsave(out_path, resized_img)
    
    return


def process_and_visualize_image(image_file, input_dir, output_dir, model):
    input_path = os.path.join(input_dir, image_file)

    # Load and process image with cleanup
    image, _, bboxes, char_height, resize_factor, height, width = (
        load_process_and_classify_image(input_path, model)
    )

    # Get columns and free memory
    column_bboxes = get_columns_bboxs(image, bboxes, char_height)

    for idx, column_bbox in enumerate(column_bboxes):
        minr, minc, maxr, maxc = column_bbox["bbox"]

        # Get sub-image and process
        sub_image = image[minr:maxr, minc:maxc].copy()
        clean_column_image = preprocess_image(sub_image)

        bboxes, _, char_height = process_and_classify_image(clean_column_image, model)

        # Get mask and free intermediate results
        mask = get_lines_mask(sub_image, bboxes, char_height)
        
        if np.any(mask):
            # Extract character lines
            extract_char_lines(sub_image, mask, char_height, idx, output_dir, image_file)

        del bboxes, clean_column_image, sub_image
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
        default="data/processed/img-text-lines",
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
