import sys
from skimage.measure import label, regionprops
import os


# Allow inport of data and model modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data.glyph_classes import GLYPH_CLASSES, categorize_glyph_class


def count_bboxes_in_components(blank_image, bboxes, min_count=1):
    """
    Count how many bounding boxes with a specific label are inside each connected component in the blank_image.

    Parameters:
    blank_image (ndarray): The input binary image with connected components.
    bboxes (list): A list of dictionaries with bounding boxes and class labels.
                   Each dictionary should have the format: {"bbox": (minr, minc, maxr, maxc), "label": str}
    min_count (int): The minimum number of bounding boxes required to include the region. Defaults to 1.

    Returns:
    list: A list of regions with an extra field 'count' for bounding boxes with the target label.
    """
    # Label connected components in the blank_image
    labeled_image = label(blank_image > 0.5, connectivity=2)
    regions = regionprops(labeled_image)

    # Initialize a list to store the regions with counts
    filtered_regions = []

    for region in regions:
        count = 0
        for bbox in bboxes:
            if categorize_glyph_class(bbox["label"]) in [0, 1, 2, 4]:
                bbox_minr, bbox_minc, bbox_maxr, bbox_maxc = bbox["bbox"]
                # Check if the center of the bbox is within the region
                center_r = (bbox_minr + bbox_maxr) // 2
                center_c = (bbox_minc + bbox_maxc) // 2
                if labeled_image[center_r, center_c] == region.label:
                    count += 1

        if count >= min_count:
            region_with_count = {"bbox": region.bbox, "count": count}
            filtered_regions.append(region_with_count)

    columns_sorted = sorted(filtered_regions, key=lambda x: x["bbox"][3], reverse=True)

    columns_filtered = remove_nested_bboxes(columns_sorted)

    return columns_filtered


def remove_nested_bboxes(columns_sorted):
    """
    Remove bounding boxes that are completely inside other bounding boxes.

    Parameters:
    columns_sorted (list): A list of sorted columns with bounding boxes and counts.

    Returns:
    list: A list of columns with nested bounding boxes removed.
    """
    filtered_columns = []

    for i, column in enumerate(columns_sorted):
        minr1, minc1, maxr1, maxc1 = column["bbox"]
        is_nested = False

        for j, other_column in enumerate(columns_sorted):
            if i != j:
                minr2, minc2, maxr2, maxc2 = other_column["bbox"]
                if (
                    minr1 >= minr2
                    and maxr1 <= maxr2
                    and minc1 >= minc2
                    and maxc1 <= maxc2
                ):
                    is_nested = True
                    break

        if not is_nested:
            filtered_columns.append(column)

    return filtered_columns
