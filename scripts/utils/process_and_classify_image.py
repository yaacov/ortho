import sys
import torch
from skimage import measure
import numpy as np
import os


# Allow inport of data and model modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data.glyph_classes import GLYPH_CLASSES, categorize_glyph_class
from data.transform import CustomTransform
from utils.load_image import imread_auto_rotate, load_image, preprocess_image


def process_and_classify_image(cleaned_image, model, bucket_size=10, min_glyph_size=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = CustomTransform()

    # Label connected components
    labels = measure.label(
        cleaned_image > 0.5, connectivity=2
    )  # Binary threshold, adjustable
    regions = measure.regionprops(labels)

    results = []
    bbox_heights = []

    for region in regions:
        minr, minc, maxr, maxc = region.bbox  # Bounding box

        # Check size of bounding box
        if (maxr - minr) < min_glyph_size or (maxc - minc) < min_glyph_size:
            continue

        roi = cleaned_image[minr:maxr, minc:maxc]

        # Convert boolean ROI to float for resizing
        roi = roi.astype(float)

        # Resize ROI to fit model input size (assuming 224x224 for this example)
        roi_resized = transform(roi).unsqueeze(0).to(device)

        # Classify
        outputs = model(roi_resized)
        _, predicted = torch.max(outputs, 1)
        class_label = GLYPH_CLASSES[predicted.item()]

        if categorize_glyph_class(class_label) == 0:
            bbox_height = maxr - minr
            bbox_heights.append(bbox_height)

            results.append(
                {
                    "bbox": (minr, minc, maxr, maxc),
                    "label": class_label,
                }
            )

        if categorize_glyph_class(class_label) > 0:
            results.append(
                {
                    "bbox": (minr, minc, maxr, maxc),
                    "label": class_label,
                }
            )

    # Build histogram of bounding box heights
    if len(bbox_heights) > 0:
        hist, bin_edges = np.histogram(
            bbox_heights, bins=range(0, max(bbox_heights) + bucket_size, bucket_size)
        )

        # Find the most prevalent box height
        most_prevalent_height_bin = bin_edges[np.argmax(hist)] + bucket_size // 2
    else:
        # Handle cases where the image has no Exif data or other errors occur
        print(
            f"Warning: Could not extract glyphs from image, fallback to font size 30."
        )
        most_prevalent_height_bin = 0

    return results, cleaned_image, most_prevalent_height_bin


def load_process_and_classify_image(input_path, model):
    target_char_height = 32
    resize_factor = 1

    # Load and preprocess the image with initial resize factor
    cleaned_image = load_image(input_path, resize_factor)
    bboxes, _, char_height = process_and_classify_image(cleaned_image, model)

    if char_height > 0:  # Only adjust if we found characters
        # Calculate the resize factor needed to get characters close to target height
        resize_factor = target_char_height / char_height

        # Load the image with the calculated resize factor
        image = imread_auto_rotate(input_path, resize_factor)
        cleaned_image = preprocess_image(image)
        bboxes, _, char_height = process_and_classify_image(cleaned_image, model)
    else:
        # If no characters were detected, use original image
        image = imread_auto_rotate(input_path, resize_factor)

    height, width = cleaned_image.shape[:2]
    return image, cleaned_image, bboxes, char_height, resize_factor, height, width
