import numpy as np
from skimage import io, color, morphology, exposure, transform
from skimage.filters import threshold_sauvola
import piexif


def load_image(image_path, resize_factor=1):
    """
    Load an image and preprocess it step by step: convert to grayscale, equalize,
    apply thresholding, and remove noise.

    Args:
        image_path (str): Path to the input image file.
        resize_factor (float, optional): Scaling factor to resize the image. Defaults to 1.

    Returns:
        ndarray: Cleaned binary image after preprocessing and noise removal.
    """
    # Load the image
    image = imread_auto_rotate(image_path, resize_factor)

    # Preprocess the image
    return preprocess_image(image)


def preprocess_image(image):
    """
    Preprocess an image by converting it to grayscale, equalizing, thresholding,
    and removing noise.

    Args:
        image (ndarray): Input image to preprocess.

    Returns:
        ndarray: Cleaned binary image after preprocessing and noise removal.
    """
    # Convert to grayscale
    gray_image = convert_to_grayscale(image)

    # Apply adaptive histogram equalization
    equalized_image = equalize_image(gray_image)

    # Apply adaptive thresholding
    binary_image = apply_thresholding(equalized_image)

    # Remove small objects (noise)
    cleaned_image = remove_noise(binary_image)

    return cleaned_image


def convert_to_grayscale(image):
    """
    Convert an image to grayscale if it is not already.

    Args:
        image (ndarray): Input image to convert.

    Returns:
        ndarray: Grayscale image.
    """
    if len(image.shape) == 2:  # Already grayscale
        return image
    elif image.shape[2] == 4:  # RGBA
        image = color.rgba2rgb(image)

    # RGB
    return color.rgb2gray(image)


def equalize_image(gray_image):
    """
    Apply adaptive histogram equalization to enhance contrast.

    Args:
        gray_image (ndarray): Input grayscale image.

    Returns:
        ndarray: Equalized image.
    """
    if gray_image.shape[0] < 32 or gray_image.shape[1] < 92:
        print(
            "Warning: Input image is too small for feature extraction", gray_image.shape
        )
    return exposure.equalize_adapthist(gray_image)


def load_and_preprocess_image(image_path, resize_factor=1):
    """
    Load an image, convert it to grayscale if necessary, and apply adaptive histogram equalization.

    Args:
        image_path (str): Path to the input image file.
        resize_factor (float, optional): Scaling factor to resize the image. Defaults to 1.

    Returns:
        ndarray: Grayscale and equalized image.
    """
    image = imread_auto_rotate(image_path, resize_factor)

    if len(image.shape) == 2:  # Grayscale image
        gray_image = image
    else:
        gray_image = color.rgb2gray(image)

    if gray_image.shape[0] < 32 or gray_image.shape[1] < 92:
        print(
            f"Warning: Input image is too small for feature extraction",
            gray_image.shape,
        )

    equalized_image = exposure.equalize_adapthist(gray_image)

    return equalized_image


def apply_thresholding(image, window_size=25):
    """
    Apply adaptive thresholding to a grayscale image using Sauvola's method.

    Args:
        image (ndarray): Input grayscale image.
        window_size (int, optional): Size of the window used for local thresholding. Defaults to 25.

    Returns:
        ndarray: Binary image obtained after applying the threshold.
    """
    threshold = threshold_sauvola(image, window_size=window_size)

    # Invert thresholding to work on black (0) pixels
    binary_image = image <= threshold
    return binary_image


def remove_noise(binary_image, min_size=20):
    """
    Remove small objects and holes from a binary image.

    Args:
        binary_image (ndarray): Input binary image.
        min_size (int, optional): Minimum size of objects to retain in the binary image. Defaults to 20.

    Returns:
        ndarray: Binary image after noise removal.
    """
    cleaned_image = morphology.remove_small_objects(binary_image, min_size=min_size)
    cleaned_image = morphology.remove_small_holes(
        binary_image, area_threshold=min_size // 5
    )

    return cleaned_image


def imread_auto_rotate(image_path, resize_factor=1):
    """
    Load an image, automatically correct its orientation based on Exif metadata,
    and resize it by a specified factor.

    Args:
        image_path (str): Path to the image file.
        resize_factor (float): Factor by which to resize the image. Default is 1 (no resizing).

    Returns:
        np.ndarray: The corrected and resized image.
    """
    # Load the image as raw data using skimage
    raw_image = io.imread(image_path)

    try:
        # Try reading Exif metadata using piexif
        exif_data = piexif.load(image_path)

        # Get the Orientation tag, defaulting to 1 (normal) if not present
        orientation = exif_data["0th"].get(piexif.ImageIFD.Orientation, 1)

        # Correct the orientation using NumPy and skimage
        if orientation == 3:  # Rotated 180°
            corrected_image = np.rot90(raw_image, 2)  # Rotate 180 degrees
        elif orientation == 6:  # Rotated 90° clockwise
            corrected_image = np.rot90(raw_image, -1)  # Rotate 90 degrees clockwise
        elif orientation == 8:  # Rotated 90° counterclockwise
            corrected_image = np.rot90(
                raw_image, 1
            )  # Rotate 90 degrees counterclockwise
        elif orientation == 2:  # Flipped horizontally
            corrected_image = np.fliplr(raw_image)  # Flip left to right
        elif orientation == 4:  # Flipped vertically
            corrected_image = np.flipud(raw_image)  # Flip top to bottom
        else:
            corrected_image = raw_image  # No transformation needed
    except Exception as e:
        print(f"Warning: Could not process Exif data. Error: {e}")
        corrected_image = raw_image

    # Resize the image if resize_factor > 1
    if resize_factor > 1:
        height, width = corrected_image.shape[:2]
        new_height, new_width = int(height * resize_factor), int(width * resize_factor)
        corrected_image = transform.resize(
            corrected_image, (new_height, new_width), anti_aliasing=True
        )

    return corrected_image
