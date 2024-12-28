from skimage import morphology


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
