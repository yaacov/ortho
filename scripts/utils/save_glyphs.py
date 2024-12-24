import os
import numpy as np
from skimage import io, measure
from skimage.transform import resize


def save_glyphs(binary_image, output_path, prefix, target_height=32):
    """
    Extract and save connected components from a binary image as individual image files.

    Parameters:
    - binary_image (numpy.ndarray): A 2D binary image with connected components.
    - output_path (str): Directory path to save the output component images.
    - prefix (str, optional): Prefix for naming saved component images.
    - target_height (int, optional): Target height for resizing components while maintaining aspect ratio. Defaults to 32.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Label connected components in the binary image
    labeled_image = measure.label(binary_image, connectivity=2)

    # Get properties of labeled regions
    regions = measure.regionprops(labeled_image)

    # Iterate over each connected component and save it
    num_saved_components = _process_and_save_components(
        regions, labeled_image, output_path, prefix, target_height
    )

    print(f"Saved {num_saved_components} connected components to {output_path}")


def _process_and_save_components(
    regions, labeled_image, output_path, prefix, target_height
):
    """
    Iterate over labeled regions, resize components, and save them as individual images.

    Parameters:
    - regions (list): List of labeled region properties from measure.regionprops.
    - labeled_image (numpy.ndarray): Labeled image with connected components.
    - output_path (str): Directory path to save output images.
    - prefix (str): Prefix for naming saved images.
    - target_height (int): Target height for resizing components while maintaining aspect ratio.

    Returns:
    - int: Number of connected components processed and saved.
    """
    label_count = 0
    for label, region in enumerate(regions, start=1):
        cropped_component = _crop_component(region, labeled_image)

        if _is_valid_component(cropped_component):
            resized_component = _resize_component(cropped_component, target_height)

            if resized_component is not None:
                component_filename = os.path.join(output_path, f"{prefix}{label}.png")
                io.imsave(component_filename, resized_component)

                label_count += 1
    return label_count


def _crop_component(region, labeled_image):
    """
    Crop a component using its bounding box.

    Parameters:
    - region (RegionProperties): Properties of the labeled region.
    - labeled_image (numpy.ndarray): Labeled image with connected components.

    Returns:
    - numpy.ndarray: Cropped binary component.
    """
    min_row, min_col, max_row, max_col = region.bbox
    return (labeled_image[min_row:max_row, min_col:max_col] != 0).astype(np.uint8) * 255


def _is_valid_component(component):
    """
    Determine if a component is valid based on its dimensions.

    Parameters:
    - component (numpy.ndarray): Cropped binary component.

    Returns:
    - bool: True if component dimensions are greater than 10x10, False otherwise.
    """
    height, width = component.shape
    return height > 10 and width > 10


def _resize_component(component, target_height):
    """
    Resize a component to the target height while maintaining aspect ratio.

    Parameters:
    - component (numpy.ndarray): Cropped binary component.
    - target_height (int): Target height for resizing.

    Returns:
    - numpy.ndarray or None: Resized component if the target width is valid, otherwise None.
    """
    current_height, current_width = component.shape
    aspect_ratio = current_width / current_height
    target_width = int(target_height * aspect_ratio)

    if target_width > 10:
        resized_component = (
            resize(component, (target_height, target_width), anti_aliasing=True) * 255
        )
        return resized_component.astype(np.uint8)

    return None
