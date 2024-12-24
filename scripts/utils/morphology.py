from skimage import morphology, transform
import numpy as np


def morphological_smoothing(image, char_height):
    # Use a circular kernel for smoothing (approximates rounded shapes)
    smoothing_kernel = morphology.disk(char_height)

    # Apply morphological opening to remove sharp protrusions
    opened_image = morphology.binary_opening(image, smoothing_kernel)

    # Apply morphological closing to fill in small indentations
    smoothed_image = morphology.binary_closing(opened_image, smoothing_kernel)

    return smoothed_image


def thinning_to_height(image, char_height):
    # Skeletonize the image to reduce it to a single-pixel-wide structure
    skeleton = morphology.skeletonize(image > 0)  # Ensure binary input

    # Expand the skeleton back to the target `char_height` using dilation
    reconstruction_kernel = morphology.rectangle(char_height, 1)
    reconstructed_image = morphology.binary_dilation(skeleton, reconstruction_kernel)

    return reconstructed_image


def process_image(blank_image, char_height, rect_height):
    height, width = blank_image.shape[:2]
    resize_factor = 4
    if char_height < 60:
        resize_factor = 2
    if char_height < 30:
        resize_factor = 1

    samll_blank_image = transform.resize(
        blank_image,
        (height // resize_factor, width // resize_factor),
        anti_aliasing=False,
    )

    samll_blank_image = apply_morphology(
        samll_blank_image, rect_height // resize_factor
    )
    samll_blank_image = apply_morphology(
        samll_blank_image, rect_height // resize_factor
    )

    samll_blank_image = thinning_to_height(
        samll_blank_image, char_height // resize_factor // 3
    )

    blank_image = transform.resize(
        samll_blank_image, (height, width), anti_aliasing=False
    )

    return blank_image
